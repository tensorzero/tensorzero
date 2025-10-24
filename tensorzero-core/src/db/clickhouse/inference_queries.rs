use std::collections::BTreeSet;

use async_trait::async_trait;
use itertools::Itertools;

use crate::db::clickhouse::query_builder::parameters::add_parameter;
use crate::db::clickhouse::query_builder::{
    generate_order_by_sql, ClickhouseType, JoinRegistry, QueryParameter,
};
use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::db::inferences::{
    ClickHouseStoredInferenceWithDispreferredOutputs, InferenceOutputSource, InferenceQueries,
    ListInferencesParams,
};
use crate::{
    config::Config,
    db::clickhouse::ClickhouseFormat,
    error::{Error, ErrorDetails},
    function::FunctionConfig,
    stored_inference::StoredInference,
};

#[async_trait]
impl InferenceQueries for ClickHouseConnectionInfo {
    async fn list_inferences(
        &self,
        config: &Config,
        params: &ListInferencesParams<'_>,
    ) -> Result<Vec<StoredInference>, Error> {
        let (sql, bound_parameters) = generate_list_inferences_sql(config, params)?;
        let params_map = bound_parameters
            .iter()
            .map(|p| (p.name.as_str(), p.value.as_str()))
            .collect();
        let response = self.inner.run_query_synchronous(sql, &params_map).await?;
        let inferences = response
            .response
            .trim()
            .lines()
            .map(|line| {
                serde_json::from_str::<ClickHouseStoredInferenceWithDispreferredOutputs>(line)
                    .map_err(|e| {
                        Error::new(ErrorDetails::ClickHouseQuery {
                            message: format!("Failed to deserialize response: {e:?}"),
                        })
                    })
                    .and_then(ClickHouseStoredInferenceWithDispreferredOutputs::try_into)
            })
            .collect::<Result<Vec<StoredInference>, Error>>()?;
        Ok(inferences)
    }
}

/// Generates the ClickHouse query and a list of parameters to be set.
/// The query string will contain placeholders like `{p0:String}`.
/// The returned `Vec<QueryParameter>` contains the mapping from placeholder names (e.g., "p0")
/// to their string values. The client executing the query is responsible for
/// setting these parameters (e.g., via `SET param_p0 = 'value'` or `SET param_p1 = 123`).
///
/// Very important: if a field is missing (fails to join or similar) it will automatically fail the condition.
/// This means that it will not be included in the result set unless the null field is in an OR
/// where another element is true.
///
/// TODOs:
/// - handle selecting the feedback values
pub(crate) fn generate_list_inferences_sql(
    config: &Config,
    opts: &ListInferencesParams<'_>,
) -> Result<(String, Vec<QueryParameter>), Error> {
    let mut params_map: Vec<QueryParameter> = Vec::new();
    let mut param_idx_counter = 0; // Counter for unique parameter names

    let mut select_clauses: BTreeSet<String> = BTreeSet::from([
        "i.function_name as function_name".to_string(),
        "i.input as input".to_string(),
        "i.variant_name as variant_name".to_string(),
        "i.episode_id as episode_id".to_string(),
        "i.id as inference_id".to_string(),
        "formatDateTime(i.timestamp, '%Y-%m-%dT%H:%i:%SZ') as timestamp".to_string(),
        "i.tags as tags".to_string(),
        // We don't select output here because it's handled separately based on the output_source
    ]);
    // TODO DO NOT SUBMIT (shuyangli): Figure out how to do this without function name.
    match function_config {
        FunctionConfig::Json(_) => {
            select_clauses.insert("i.output_schema as output_schema".to_string());
            select_clauses.insert("'json' as type".to_string());
        }
        FunctionConfig::Chat(_) => {
            select_clauses.insert("i.tool_params as tool_params".to_string());
            select_clauses.insert("'chat' as type".to_string());
        }
    }

    let mut joins = JoinRegistry::new();
    let mut where_clauses: Vec<String> = Vec::new();

    if let Some(function_name) = opts.function_name {
        let function_config = config.get_function(function_name)?;
        let function_name_param_placeholder = add_parameter(
            function_name,
            ClickhouseType::String,
            &mut params_map,
            &mut param_idx_counter,
        );
        where_clauses.push(format!(
            "i.function_name = {function_name_param_placeholder}"
        ));
    }

    let inference_table_name = function_config.table_name();

    // Add `variant_name` filter
    if let Some(variant_name) = opts.variant_name {
        let variant_name_param_placeholder = add_parameter(
            variant_name,
            ClickhouseType::String,
            &mut params_map,
            &mut param_idx_counter,
        );
        where_clauses.push(format!("i.variant_name = {variant_name_param_placeholder}"));
    }

    // Handle OutputSource
    match opts.output_source {
        InferenceOutputSource::Inference => {
            select_clauses.insert("i.output as output".to_string());
        }
        InferenceOutputSource::Demonstration => {
            select_clauses.insert("demo_f.value AS output".to_string());
            // [i.output] will produce an array in ClickHouse which will populate the dispreferred_outputs field
            select_clauses.insert("[i.output] as dispreferred_outputs".to_string());

            // NOTE: we may want to pre-filter this via subqueries or CTEs prior to the join for performance reasons
            joins.insert_unchecked(
                "\nJOIN \
                 (SELECT \
                    inference_id, \
                    argMax(value, timestamp) as value \
                  FROM DemonstrationFeedback \
                  GROUP BY inference_id \
                 ) AS demo_f ON i.id = demo_f.inference_id"
                    .to_string(),
            );
        }
    }

    if let Some(filter_node) = opts.filters {
        // Recursively builds the filter condition SQL statement for the WHERE clause
        //  * adds the JOINed tables it needs
        //  * adds metric columns to the SELECT clause for visibility and debugging
        let filter_condition_sql = filter_node.to_clickhouse_sql(
            config,
            &mut params_map,
            &mut select_clauses,
            &mut joins,
            &mut param_idx_counter,
        )?;
        where_clauses.push(filter_condition_sql);
    }

    let mut sql = format!(
        r"
SELECT
    {select_clauses}
FROM
    {inference_table_name} AS i",
        select_clauses = select_clauses.iter().join(",\n    "),
        inference_table_name = inference_table_name,
    );
    // We generate the order by SQL before we add the joins so that the join registry is up to date with everything it needs.
    // We don't actually add the order by SQL to the query until after we've added the joins.
    let order_by_sql = generate_order_by_sql(
        opts.order_by,
        config,
        &mut params_map,
        &mut param_idx_counter,
        &mut joins,
    )?;

    if !joins.get_clauses().is_empty() {
        sql.push_str(&joins.get_clauses().join("\n"));
    }

    if !where_clauses.is_empty() {
        sql.push_str("\nWHERE\n    ");
        sql.push_str(&where_clauses.join(" AND "));
    }
    sql.push_str(order_by_sql.as_str());

    if let Some(l) = opts.limit {
        let limit_param_placeholder = add_parameter(
            l,
            ClickhouseType::UInt64,
            &mut params_map,
            &mut param_idx_counter,
        );
        sql.push_str(&format!("\nLIMIT {limit_param_placeholder}"));
    }
    if let Some(o) = opts.offset {
        let offset_param_placeholder = add_parameter(
            o,
            ClickhouseType::UInt64,
            &mut params_map,
            &mut param_idx_counter,
        );
        sql.push_str(&format!("\nOFFSET {offset_param_placeholder}"));
    }
    match opts.format {
        ClickhouseFormat::JsonEachRow => {
            sql.push_str("\nFORMAT JSONEachRow");
        }
    }

    Ok((sql, params_map))
}

mod tests {
    // TODO(shuyangli): Separate generate_list_inferences_sql tests from filter
    // construction tests in `tensorzero-core/src/db/clickhouse/query_builder/mod.rs`,
    // and move list_inferences related tests here.
    // Right now, there is no test coverage in this file because the generated SQL is tested extensively in the above file.
}
