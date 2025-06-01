use itertools::Itertools;
use std::collections::HashSet;

use crate::{
    config_parser::Config,
    error::{Error, ErrorDetails},
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InferenceOutputSource {
    Inference,
    Demonstration,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FloatComparisonOperator {
    LessThan,
    LessThanOrEqual,
    Equal,
    GreaterThan,
    GreaterThanOrEqual,
    NotEqual,
}

impl FloatComparisonOperator {
    pub fn to_clickhouse_operator(&self) -> &str {
        match self {
            FloatComparisonOperator::LessThan => "<",
            FloatComparisonOperator::LessThanOrEqual => "<=",
            FloatComparisonOperator::Equal => "=",
            FloatComparisonOperator::GreaterThan => ">",
            FloatComparisonOperator::GreaterThanOrEqual => ">=",
            FloatComparisonOperator::NotEqual => "!=",
        }
    }
}

#[derive(Debug, Clone)]
pub struct FloatMetricNode {
    pub metric_name: String,
    pub value: f64,
    pub comparison_operator: FloatComparisonOperator,
}

#[derive(Debug, Clone)]
pub struct BooleanMetricNode {
    pub metric_name: String,
    pub value: bool,
}

#[derive(Debug, Clone)]
pub enum InferenceFilterTreeNode {
    FloatMetric(FloatMetricNode),
    BooleanMetric(BooleanMetricNode),
    And(Vec<InferenceFilterTreeNode>),
    Or(Vec<InferenceFilterTreeNode>),
    Not(Box<InferenceFilterTreeNode>),
}

impl InferenceFilterTreeNode {
    /// Converts the filter tree to a ClickHouse SQL string.
    ///
    /// The returned string will contain the filter condition that should be added to the WHERE clause.
    /// The `params_map` is updated with the parameters used in the filter condition.
    /// The `join_clauses` is updated with the JOIN clauses.
    /// The `param_idx_counter` is updated with the current index of the parameter.
    /// The `join_idx_counter` is updated with the current index of the join alias.
    /// The `_select_clauses` is not used *yet*--we will want to add metric columns to the SELECT clause for visibility and debugging.
    ///
    /// NOTE: This is not efficient at all yet. We are doing a lot of JOINs and GROUP BYs.
    /// We may be able to do this more efficiently by using subqueries and CTEs.
    /// We're also doing a join per filter. In principle if there is a subtree of the tree that uses the same joined table,
    /// we could push the condition down into the query before the join
    pub fn to_clickhouse_sql(
        &self,
        config: &Config,
        params_map: &mut Vec<QueryParameter>,
        _select_clauses: &mut HashSet<String>,
        join_clauses: &mut Vec<String>,
        param_idx_counter: &mut usize,
        join_idx_counter: &mut usize,
    ) -> Result<String, Error> {
        match self {
            InferenceFilterTreeNode::FloatMetric(fm_node) => {
                let metric_config = config
                    .metrics
                    .get(fm_node.metric_name.as_str())
                    .ok_or_else(|| {
                        Error::new(ErrorDetails::InvalidMetricName {
                            metric_name: fm_node.metric_name.clone(),
                        })
                    })?;
                let inference_table_column_name = metric_config.level.inference_column_name();

                // 1. Create an alias for the join condition we'll need
                let join_alias = get_join_alias(join_idx_counter);
                // 2. Set up query parameters
                let metric_name_placeholder = add_parameter(
                    fm_node.metric_name.clone(),
                    "String",
                    params_map,
                    param_idx_counter,
                );
                let value_placeholder = add_parameter(
                    fm_node.value.to_string(),
                    "Float64",
                    params_map,
                    param_idx_counter,
                );

                // 3. register the LEFT JOIN clause
                let comparison_operator = fm_node.comparison_operator.to_clickhouse_operator();
                join_clauses.push(format!(
                    r#"
LEFT JOIN (
    SELECT
        target_id,
        argMax(value, timestamp) as value
    FROM FloatMetricFeedback
    WHERE metric_name = {metric_name_placeholder}
    AND value {comparison_operator} {value_placeholder}
    GROUP BY target_id
) AS {join_alias} ON i.{inference_table_column_name} = {join_alias}.target_id"#
                ));
                // 4. return the filter condition
                // NOTE: if the join_alias is NULL, the filter condition will be NULL also
                // We handle this farther up the recursive tree
                Ok(format!(
                    "{join_alias}.value {comparison_operator} {value_placeholder}"
                ))
            }
            InferenceFilterTreeNode::BooleanMetric(bm_node) => {
                let metric_config = config
                    .metrics
                    .get(bm_node.metric_name.as_str())
                    .ok_or_else(|| {
                        Error::new(ErrorDetails::InvalidMetricName {
                            metric_name: bm_node.metric_name.clone(),
                        })
                    })?;
                let inference_table_column_name = metric_config.level.inference_column_name();
                // 1. Create an alias for the join condition we'll need
                let join_alias = get_join_alias(join_idx_counter);
                // 2. Set up query parameters
                let metric_name_placeholder = add_parameter(
                    bm_node.metric_name.clone(),
                    "String",
                    params_map,
                    param_idx_counter,
                );
                let bool_value_str = if bm_node.value {
                    "1".to_string()
                } else {
                    "0".to_string()
                };
                // ClickHouse 'Bool' type is an alias for UInt8 restricted to 0 or 1.
                let value_placeholder =
                    add_parameter(bool_value_str, "Bool", params_map, param_idx_counter);
                // 3. register the JOIN clause
                join_clauses.push(format!(
                    r#"
LEFT JOIN (
    SELECT
        target_id,
        argMax(value, timestamp) as value
    FROM BooleanMetricFeedback
    WHERE metric_name = {metric_name_placeholder}
    AND value = {value_placeholder}
    GROUP BY target_id
) AS {join_alias} ON i.{inference_table_column_name} = {join_alias}.target_id"#
                ));
                // 4. return the filter condition
                // NOTE: if the join_alias is NULL, the filter condition will be NULL also
                // We handle this farther up the recursive tree
                Ok(format!("{join_alias}.value = {value_placeholder}"))
            }
            InferenceFilterTreeNode::And(children) => {
                let child_sqls: Vec<String> = children
                    .iter()
                    .map(|child| {
                        child.to_clickhouse_sql(
                            config,
                            params_map,
                            _select_clauses,
                            join_clauses,
                            param_idx_counter,
                            join_idx_counter,
                        )
                    })
                    .collect::<Result<Vec<String>, Error>>()?
                    .into_iter()
                    // We need to coalesce the filter condition to 0 if the join_alias is NULL
                    // For an AND filter we want to return 0 if any of the children are NULL
                    .map(|s| format!("COALESCE({s}, 0)"))
                    .collect::<Vec<String>>();
                let child_sqls_str = child_sqls.join(" AND ");
                Ok(format!("({child_sqls_str})"))
            }
            InferenceFilterTreeNode::Or(children) => {
                let child_sqls: Vec<String> = children
                    .iter()
                    .map(|child| {
                        child.to_clickhouse_sql(
                            config,
                            params_map,
                            _select_clauses,
                            join_clauses,
                            param_idx_counter,
                            join_idx_counter,
                        )
                    })
                    .collect::<Result<Vec<String>, Error>>()?
                    .into_iter()
                    // We need to coalesce the filter condition to 0 if the join_alias is NULL
                    // For an OR filter we want to return 0 if all of the children are NULL
                    .map(|s| format!("COALESCE({s}, 0)"))
                    .collect::<Vec<String>>();
                let child_sqls_str = child_sqls.join(" OR ");
                Ok(format!("({child_sqls_str})"))
            }
            InferenceFilterTreeNode::Not(child) => {
                let child_sql = child.to_clickhouse_sql(
                    config,
                    params_map,
                    _select_clauses,
                    join_clauses,
                    param_idx_counter,
                    join_idx_counter,
                )?;
                // We need to coalesce the filter condition to 1 if the join_alias is NULL
                // For a NOT filter we want to still be false if the join_alias is NULL
                // NOTE to reviewer: Is this the behavior we want?
                Ok(format!("NOT (COALESCE({child_sql}, 1))"))
            }
        }
    }
}

/// Represents a parameter to be set for the ClickHouse query.
/// The `name` is the internal name (e.g., "p0", "p1") used in `SET param_<name> = ...`
/// and in the `{<name>:DataType}` placeholder.
/// The `value` is the string representation of the value.
#[derive(Debug, Clone)]
pub struct QueryParameter {
    pub name: String,
    pub value: String,
}

/// Generates the ClickHouse query and a list of parameters to be set.
/// The query string will contain placeholders like `{p0:String}`.
/// The returned `Vec<QueryParameter>` contains the mapping from placeholder names (e.g., "p0")
/// to their string values. The client executing the query is responsible for
/// setting these parameters (e.g., via `SET param_p0 = 'value'` or `SET param_p1 = 123`).
pub fn generate_list_inferences_sql(
    config: &Config,
    function_name: &str,
    variant_name: Option<&str>,
    filters: Option<&InferenceFilterTreeNode>,
    output_source: &InferenceOutputSource,
    limit: Option<i64>,
    offset: Option<i64>,
) -> Result<(String, Vec<QueryParameter>), Error> {
    let mut params_map: Vec<QueryParameter> = Vec::new();
    let mut param_idx_counter = 0; // Counter for unique parameter names
    let mut join_idx_counter = 0; // Counter for unique join aliases

    let mut select_clauses = HashSet::from([
        "i.input as input".to_string(),
        "i.variant_name as variant_name".to_string(),
        "i.episode_id as episode_id".to_string(),
        "i.id as id".to_string(),
        "i.timestamp as timestamp".to_string(),
        // We don't select output here because it's handled separately based on the output_source
    ]);
    let mut join_clauses: Vec<String> = Vec::new();
    let mut where_clauses: Vec<String> = Vec::new();

    let function_config = config.get_function(function_name)?;
    let inference_table_name = function_config.table_name();

    let function_name_param_placeholder = add_parameter(
        function_name.to_string(),
        "String",
        &mut params_map,
        &mut param_idx_counter,
    );
    where_clauses.push(format!(
        "i.function_name = {function_name_param_placeholder}"
    ));

    // Add variant_name filter
    if let Some(variant_name) = variant_name {
        let variant_name_param_placeholder = add_parameter(
            variant_name.to_string(),
            "String",
            &mut params_map,
            &mut param_idx_counter,
        );
        where_clauses.push(format!("i.variant_name = {variant_name_param_placeholder}"));
    }

    // Handle OutputSource
    match output_source {
        InferenceOutputSource::Inference => {
            select_clauses.insert("i.output as output".to_string());
        }
        InferenceOutputSource::Demonstration => {
            select_clauses.insert("demo_f.value AS output".to_string());

            // NOTE: we may want to pre-filter this via subqueries or CTEs prior to the join for performance reasons
            join_clauses.push(
                "JOIN \
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

    if let Some(filter_node) = filters {
        // Recursively builds the filter condition SQL statement for the WHERE clause
        //  * adds the JOINed tables it needs
        //  * adds metric columns to the SELECT clause for visibility and debugging
        let filter_condition_sql = filter_node.to_clickhouse_sql(
            config,
            &mut params_map,
            &mut select_clauses,
            &mut join_clauses,
            &mut param_idx_counter,
            &mut join_idx_counter,
        )?;
        where_clauses.push(filter_condition_sql);
    }

    let mut sql = format!(
        r#"
SELECT
    {select_clauses}
FROM
    {inference_table_name} AS i
    "#,
        select_clauses = select_clauses.iter().join(",\n    "),
        inference_table_name = inference_table_name,
    );

    if !join_clauses.is_empty() {
        sql.push('\n');
        sql.push_str(&join_clauses.join("\n"));
    }

    if !where_clauses.is_empty() {
        sql.push_str("\nWHERE\n    ");
        sql.push_str(&where_clauses.join(" AND "));
    }
    // TODO: add ORDER BY

    if let Some(l) = limit {
        let limit_param_placeholder = add_parameter(
            l.to_string(),
            "UInt64",
            &mut params_map,
            &mut param_idx_counter,
        );
        sql.push_str(&format!("\nLIMIT {limit_param_placeholder}"));
    }
    if let Some(o) = offset {
        let offset_param_placeholder = add_parameter(
            o.to_string(),
            "UInt64",
            &mut params_map,
            &mut param_idx_counter,
        );
        sql.push_str(&format!("\nOFFSET {offset_param_placeholder}"));
    }

    Ok((sql, params_map))
}

/// Helper to add a parameter and return its SQL placeholder {name:CHType}
/// The internal_name (e.g. p0, p1) is stored in params_map with its value.
fn add_parameter(
    value: String,
    ch_type: &str,
    params_map: &mut Vec<QueryParameter>,
    counter: &mut usize,
) -> String {
    let internal_name = format!("p{}", *counter);
    *counter += 1;
    params_map.push(QueryParameter {
        name: internal_name.clone(),
        value,
    });
    format!("{{{internal_name}:{ch_type}}}")
}

// Helper to get a join alias given the current counter on join tables
fn get_join_alias(counter: &mut usize) -> String {
    let internal_name = format!("j{}", *counter);
    *counter += 1;
    internal_name
}
