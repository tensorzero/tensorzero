use async_trait::async_trait;
use itertools::Itertools;

use crate::db::clickhouse::query_builder::parameters::add_parameter;
use crate::db::clickhouse::query_builder::{
    generate_order_by_sql, ClickhouseType, JoinRegistry, OrderByTerm, QueryParameter,
};
use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::db::inferences::{
    ClickHouseStoredInferenceWithDispreferredOutputs, InferenceOutputSource, InferenceQueries,
    ListInferencesParams,
};
use crate::function::FunctionConfigType;
use crate::{
    config::Config,
    db::clickhouse::ClickhouseFormat,
    error::{Error, ErrorDetails},
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
        let query_params = bound_parameters
            .iter()
            .map(|p| (p.name.as_str(), p.value.as_str()))
            .collect();
        let response = self.inner.run_query_synchronous(sql, &query_params).await?;
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
    let mut query_params: Vec<QueryParameter> = Vec::new();
    let mut param_idx_counter = 0;

    let mut sql = match opts.function_name {
        // If function_name is provided, we know which table to query
        // TODO(#4181): list_inferences requests with function name should also query both tables
        // and UNION ALL. The ORDER BY metric join is complicated and net new, so we'll tackle it later.
        Some(function_name) => {
            let mut joins = JoinRegistry::new();
            let function_config = config.get_function(function_name)?;
            let mut query = generate_single_table_query_for_type(
                config,
                opts,
                function_config.table_name(),
                function_config.config_type() == FunctionConfigType::Chat,
                &mut joins,
                &mut query_params,
                &mut param_idx_counter,
            )?;

            // For single-table queries, use proper ORDER BY generation that supports joining with metrics.
            // Reuse the join registry from the filter so we don't create duplicate joins
            let order_by_sql = if opts.order_by.is_some() {
                generate_order_by_sql(
                    opts.order_by,
                    config,
                    &mut query_params,
                    &mut param_idx_counter,
                    &mut joins,
                )?
            } else {
                String::new()
            };

            // Add all joins (from filters and ORDER BY) before WHERE clause
            if !joins.get_clauses().is_empty() {
                if let Some(where_pos) = query.find("\nWHERE") {
                    query.insert_str(where_pos, &joins.get_clauses().join("\n"));
                } else {
                    // No WHERE clause, append at end
                    query.push_str(&joins.get_clauses().join("\n"));
                }
            }

            // Add ORDER BY clause
            if !order_by_sql.is_empty() {
                query.push_str(&order_by_sql);
            }

            query
        }
        None => {
            // Otherwise, we need to query both tables with UNION ALL
            let mut chat_joins = JoinRegistry::new();
            let mut chat_sql = generate_single_table_query_for_type(
                config,
                opts,
                "ChatInference",
                true, // is_chat
                &mut chat_joins,
                &mut query_params,
                &mut param_idx_counter,
            )?;

            // Insert joins for chat query before WHERE clause
            if !chat_joins.get_clauses().is_empty() {
                if let Some(where_pos) = chat_sql.find("\nWHERE") {
                    chat_sql.insert_str(where_pos, &chat_joins.get_clauses().join("\n"));
                } else {
                    // No WHERE clause, append at end
                    chat_sql.push_str(&chat_joins.get_clauses().join("\n"));
                }
            }

            let mut json_joins = JoinRegistry::new();
            let mut json_sql = generate_single_table_query_for_type(
                config,
                opts,
                "JsonInference",
                false, // is_chat
                &mut json_joins,
                &mut query_params,
                &mut param_idx_counter,
            )?;

            // Insert joins for json query before WHERE clause
            if !json_joins.get_clauses().is_empty() {
                if let Some(where_pos) = json_sql.find("\nWHERE") {
                    json_sql.insert_str(where_pos, &json_joins.get_clauses().join("\n"));
                } else {
                    // No WHERE clause, append at end
                    json_sql.push_str(&json_joins.get_clauses().join("\n"));
                }
            }

            // Combine with UNION ALL
            let combined_query = format!("{chat_sql}\nUNION ALL\n{json_sql}");

            // For UNION ALL queries, use simplified ORDER BY
            // We need to wrap in a subquery to ensure ORDER BY applies to the combined result
            let combined = if let Some(order_by) = opts.order_by {
                let order_clauses: Vec<String> = order_by
                    .iter()
                    .map(|o| {
                        let column = match &o.term {
                            OrderByTerm::Timestamp => "timestamp",
                            OrderByTerm::Metric { name } => {
                                return Err(Error::new(ErrorDetails::InvalidRequest {
                                    message: format!(
                                        "ORDER BY metric '{name}' is not supported when querying without function_name"
                                    ),
                                }));
                            }
                        };
                        Ok(format!("{column} {}", o.direction.to_clickhouse_direction()))
                    })
                    .collect::<Result<Vec<_>, Error>>()?;

                if order_clauses.is_empty() {
                    combined_query
                } else {
                    format!(
                        "SELECT * FROM (\n{}\n) AS combined\nORDER BY {}",
                        combined_query,
                        order_clauses.join(", ")
                    )
                }
            } else {
                combined_query
            };

            combined
        }
    };

    if let Some(l) = opts.limit {
        let limit_param_placeholder = add_parameter(
            l,
            ClickhouseType::UInt64,
            &mut query_params,
            &mut param_idx_counter,
        );
        sql.push_str(&format!("\nLIMIT {limit_param_placeholder}"));
    }
    if let Some(o) = opts.offset {
        let offset_param_placeholder = add_parameter(
            o,
            ClickhouseType::UInt64,
            &mut query_params,
            &mut param_idx_counter,
        );
        sql.push_str(&format!("\nOFFSET {offset_param_placeholder}"));
    }
    match opts.format {
        ClickhouseFormat::JsonEachRow => {
            sql.push_str("\nFORMAT JSONEachRow");
        }
    }

    Ok((sql, query_params))
}

/// Core query building logic for a specific table type
/// Returns (sql, join_registry) so the caller can reuse joins for ORDER BY
fn generate_single_table_query_for_type(
    config: &Config,
    opts: &ListInferencesParams<'_>,
    table_name: &str,
    is_chat: bool,
    joins: &mut JoinRegistry,
    query_params: &mut Vec<QueryParameter>,
    param_idx_counter: &mut usize,
) -> Result<String, Error> {
    // Use a Vec to maintain explicit column order for UNION ALL compatibility
    // Base columns that are always present, in fixed order
    let mut select_clauses: Vec<String> = vec![];

    // Add type-specific columns first
    if is_chat {
        select_clauses.push("'chat' as type".to_string());
    } else {
        select_clauses.push("'json' as type".to_string());
    }

    // Add remaining columns in alphabetical order
    select_clauses
        .push("formatDateTime(i.timestamp, '%Y-%m-%dT%H:%i:%SZ') as timestamp".to_string());
    select_clauses.push("i.episode_id as episode_id".to_string());
    select_clauses.push("i.function_name as function_name".to_string());
    select_clauses.push("i.id as inference_id".to_string());
    select_clauses.push("i.input as input".to_string());

    // output will be added later based on output_source
    // Add type-specific columns with consistent names for UNION ALL
    if is_chat {
        select_clauses.push("'' as output_schema".to_string());
    } else {
        select_clauses.push("i.output_schema as output_schema".to_string());
    }

    select_clauses.push("i.tags as tags".to_string());

    if is_chat {
        select_clauses.push("i.tool_params as tool_params".to_string());
    } else {
        select_clauses.push("'' as tool_params".to_string());
    }

    select_clauses.push("i.variant_name as variant_name".to_string());

    let mut where_clauses: Vec<String> = Vec::new();

    // Add function_name filter if provided
    if let Some(function_name) = opts.function_name {
        let function_name_param_placeholder = add_parameter(
            function_name,
            ClickhouseType::String,
            query_params,
            param_idx_counter,
        );
        where_clauses.push(format!(
            "i.function_name = {function_name_param_placeholder}"
        ));
    }

    // Add variant_name filter
    if let Some(variant_name) = opts.variant_name {
        let variant_name_param_placeholder = add_parameter(
            variant_name,
            ClickhouseType::String,
            query_params,
            param_idx_counter,
        );
        where_clauses.push(format!("i.variant_name = {variant_name_param_placeholder}"));
    }

    // Add ids filter
    if let Some(ids) = opts.ids {
        // Our current production_clickhouse_client uses the HTTP client under the hood, which
        // passes parameters in the URL. This will likely hit URL length limits, so instead of passing IDs
        // as a bound parameter, we will write it directly into the query.
        let joined_ids = ids.iter().map(|id| format!("'{id}'")).join(",");
        where_clauses.push(format!("i.id IN [{joined_ids}]"));
    }

    // Add episode_id filter
    if let Some(episode_id) = opts.episode_id {
        let episode_id_param_placeholder = add_parameter(
            episode_id.to_string(),
            ClickhouseType::String,
            query_params,
            param_idx_counter,
        );
        where_clauses.push(format!("i.episode_id = {episode_id_param_placeholder}"));
    }

    // Handle OutputSource
    match opts.output_source {
        InferenceOutputSource::Inference => {
            select_clauses.push("i.output as output".to_string());
        }
        InferenceOutputSource::Demonstration => {
            select_clauses.push("demo_f.value AS output".to_string());
            select_clauses.push("[i.output] as dispreferred_outputs".to_string());

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

    // Handle filters
    if let Some(filter_node) = opts.filters {
        let filter_condition_sql = filter_node.to_clickhouse_sql(
            config,
            query_params,
            &mut select_clauses,
            joins,
            param_idx_counter,
        )?;
        where_clauses.push(filter_condition_sql);
    }

    let mut sql = format!(
        r"
SELECT
    {select_clauses}
FROM
    {table_name} AS i",
        select_clauses = select_clauses.iter().join(",\n    "),
    );

    // Don't add joins here - let the caller add them so they can be combined with ORDER BY joins
    // if !joins.get_clauses().is_empty() {
    //     sql.push_str(&joins.get_clauses().join("\n"));
    // }

    if !where_clauses.is_empty() {
        // If we have where clauses but joins will be added by caller, we need to account for that
        // The joins will be inserted before WHERE by the caller
        sql.push_str("\nWHERE\n    ");
        sql.push_str(&where_clauses.join(" AND "));
    }

    Ok(sql)
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use uuid::Uuid;

    use crate::config::{Config, ConfigFileGlob};
    use crate::db::clickhouse::query_builder::test_util::{
        assert_query_contains, assert_query_does_not_contain,
    };
    use crate::db::clickhouse::query_builder::{
        FloatComparisonOperator, FloatMetricFilter, InferenceFilter, OrderBy, OrderByTerm,
        OrderDirection, QueryParameter,
    };
    use crate::db::inferences::{InferenceOutputSource, ListInferencesParams};

    use super::generate_list_inferences_sql;

    async fn get_e2e_config() -> Config {
        // Read the e2e config file
        Config::load_from_path_optional_verify_credentials(
            &ConfigFileGlob::new_from_path(Path::new("tests/e2e/tensorzero.toml")).unwrap(),
            false,
        )
        .await
        .unwrap()
    }

    #[tokio::test]
    async fn test_query_by_ids_without_function_name_queries_and_unions_both_tables() {
        let config = get_e2e_config().await;
        let id1 = Uuid::parse_str("01234567-89ab-cdef-0123-456789abcdef").unwrap();
        let id2 = Uuid::parse_str("fedcba98-7654-3210-fedc-ba9876543210").unwrap();
        let ids = vec![id1, id2];

        let opts = ListInferencesParams {
            ids: Some(&ids),
            ..Default::default()
        };

        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();

        assert_query_contains(
            &sql,
            "SELECT
            'chat' as type,
            formatDateTime(i.timestamp, '%Y-%m-%dT%H:%i:%SZ') as timestamp,
            i.episode_id as episode_id,
            i.function_name as function_name,
            i.id as inference_id,
            i.input as input,
            '' as output_schema,
            i.tags as tags,
            i.tool_params as tool_params,
            i.variant_name as variant_name,
            i.output as output
        FROM
            ChatInference AS i
        WHERE
            i.id IN ['01234567-89ab-cdef-0123-456789abcdef','fedcba98-7654-3210-fedc-ba9876543210']

        UNION ALL

        SELECT
            'json' as type,
            formatDateTime(i.timestamp, '%Y-%m-%dT%H:%i:%SZ') as timestamp,
            i.episode_id as episode_id,
            i.function_name as function_name,
            i.id as inference_id,
            i.input as input,
            i.output_schema as output_schema,
            i.tags as tags,
            '' as tool_params,
            i.variant_name as variant_name,
            i.output as output
        FROM
            JsonInference AS i
        WHERE
            i.id IN ['01234567-89ab-cdef-0123-456789abcdef','fedcba98-7654-3210-fedc-ba9876543210']",
        );

        // This query doesn't have any bound parameters.
        assert_eq!(params.len(), 0);
    }

    #[tokio::test]
    async fn test_query_by_ids_with_function_name_queries_only_one_table() {
        let config = get_e2e_config().await;
        let id1 = Uuid::parse_str("01234567-89ab-cdef-0123-456789abcdef").unwrap();
        let ids = vec![id1];

        let opts = ListInferencesParams {
            function_name: Some("extract_entities"),
            ids: Some(&ids),
            ..Default::default()
        };

        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();

        // Verify NO UNION ALL
        assert_query_does_not_contain(&sql, "UNION ALL");

        // Verify only JsonInference is queried (extract_entities is a JSON function)
        assert_query_contains(&sql, "JsonInference");
        assert_query_does_not_contain(&sql, "ChatInference");

        // Verify ID is in the query with proper table alias
        assert_query_contains(&sql, "i.id IN ['01234567-89ab-cdef-0123-456789abcdef']");

        // Verify function_name filter is present
        assert_query_contains(&sql, "i.function_name = {p0:String}");
        assert_eq!(params.len(), 1);
        assert_eq!(
            params[0],
            QueryParameter {
                name: "p0".to_string(),
                value: "extract_entities".to_string(),
            },
            "Function name parameter should be present"
        );
    }

    #[tokio::test]
    async fn test_query_by_ids_with_function_name_and_metric_filter() {
        let config = get_e2e_config().await;
        let id1 = Uuid::parse_str("01234567-89ab-cdef-0123-456789abcdef").unwrap();
        let id2 = Uuid::parse_str("fedcba98-7654-3210-fedc-ba9876543210").unwrap();
        let ids = vec![id1, id2];

        let filter_node = InferenceFilter::FloatMetric(FloatMetricFilter {
            metric_name: "task_success".to_string(),
            comparison_operator: FloatComparisonOperator::GreaterThan,
            value: 0.5,
        });

        let opts = ListInferencesParams {
            function_name: Some("extract_entities"),
            ids: Some(&ids),
            filters: Some(&filter_node),
            ..Default::default()
        };

        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();

        // Verify NO UNION ALL
        assert_query_does_not_contain(&sql, "UNION ALL");

        // Verify metric filter generates LEFT JOIN
        assert_query_contains(&sql, "
        FROM
            JsonInference AS i
        LEFT JOIN (
            SELECT
                target_id,
                argMax(value, timestamp) as value
            FROM FloatMetricFeedback
            WHERE metric_name = {p1:String}
            GROUP BY target_id
        ) AS j0 ON i.id = j0.target_id
        WHERE
            i.function_name = {p0:String}
            AND i.id IN ['01234567-89ab-cdef-0123-456789abcdef','fedcba98-7654-3210-fedc-ba9876543210']
            AND j0.value > {p2:Float64}");

        assert!(params.contains(&QueryParameter {
            name: "p1".to_string(),
            value: "task_success".to_string(),
        }));
        assert!(params.contains(&QueryParameter {
            name: "p2".to_string(),
            value: "0.5".to_string(),
        }));
    }

    #[tokio::test]
    async fn test_query_by_ids_with_order_by_timestamp() {
        let config = get_e2e_config().await;
        let id1 = Uuid::parse_str("01234567-89ab-cdef-0123-456789abcdef").unwrap();
        let ids = vec![id1];

        let order_by = vec![OrderBy {
            term: OrderByTerm::Timestamp,
            direction: OrderDirection::Desc,
        }];

        let opts = ListInferencesParams {
            ids: Some(&ids),
            order_by: Some(&order_by),
            ..Default::default()
        };

        let (sql, _params) = generate_list_inferences_sql(&config, &opts).unwrap();

        // Verify query is wrapped in subquery with ORDER BY
        assert_query_contains(&sql, "SELECT * FROM");
        assert_query_contains(&sql, ") AS combined ORDER BY timestamp DESC");
    }

    #[tokio::test]
    async fn test_query_by_ids_without_function_name_returns_errors_for_order_by_metric() {
        let config = get_e2e_config().await;
        let id1 = Uuid::parse_str("01234567-89ab-cdef-0123-456789abcdef").unwrap();
        let ids = vec![id1];

        let order_by = vec![OrderBy {
            term: OrderByTerm::Metric {
                name: "task_success".to_string(),
            },
            direction: OrderDirection::Desc,
        }];

        let opts = ListInferencesParams {
            ids: Some(&ids),
            order_by: Some(&order_by),
            ..Default::default()
        };

        let result = generate_list_inferences_sql(&config, &opts);

        assert!(result.is_err(), "Should return error for ORDER BY metric");

        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("ORDER BY metric"),
            "Error message should indicate metric ordering not supported without function_name"
        );
    }

    #[tokio::test]
    async fn test_query_by_ids_joins_demonstration_output_in_both_subqueries() {
        let config = get_e2e_config().await;
        let id1 = Uuid::parse_str("01234567-89ab-cdef-0123-456789abcdef").unwrap();
        let id2 = Uuid::parse_str("fedcba98-7654-3210-fedc-ba9876543210").unwrap();
        let ids = vec![id1, id2];

        let opts = ListInferencesParams {
            ids: Some(&ids),
            output_source: InferenceOutputSource::Demonstration,
            ..Default::default()
        };

        let (sql, _params) = generate_list_inferences_sql(&config, &opts).unwrap();

        // Verify UNION ALL is present
        assert_query_contains(&sql, "UNION ALL");

        assert_query_contains(
            &sql,
            r"
        FROM ChatInference AS i
        JOIN (SELECT inference_id, argMax(value, timestamp) as value
        FROM DemonstrationFeedback
        GROUP BY inference_id ) AS demo_f ON i.id = demo_f.inference_id",
        );

        assert_query_contains(
            &sql,
            r"
        FROM JsonInference AS i
        JOIN (SELECT inference_id, argMax(value, timestamp) as value
        FROM DemonstrationFeedback
        GROUP BY inference_id ) AS demo_f ON i.id = demo_f.inference_id",
        );
    }

    #[tokio::test]
    async fn test_query_by_ids_without_function_name_with_metric_filter() {
        use crate::db::clickhouse::query_builder::{
            FloatComparisonOperator, FloatMetricFilter, InferenceFilter,
        };

        let config = get_e2e_config().await;
        let id1 = Uuid::parse_str("01234567-89ab-cdef-0123-456789abcdef").unwrap();
        let id2 = Uuid::parse_str("fedcba98-7654-3210-fedc-ba9876543210").unwrap();
        let ids = vec![id1, id2];

        let filter_node = InferenceFilter::FloatMetric(FloatMetricFilter {
            metric_name: "task_success".to_string(),
            comparison_operator: FloatComparisonOperator::GreaterThan,
            value: 0.5,
        });

        let opts = ListInferencesParams {
            ids: Some(&ids),
            filters: Some(&filter_node),
            ..Default::default()
        };

        let (sql, _params) = generate_list_inferences_sql(&config, &opts).unwrap();

        // Verify UNION ALL is present
        assert_query_contains(&sql, "UNION ALL");

        // Verify metric filter join is present
        // The bug would cause this to fail because joins are not inserted
        assert_query_contains(
            &sql,
            r"FROM ChatInference AS i
            LEFT JOIN (
                SELECT
                    target_id,
                    argMax(value, timestamp) as value
                FROM FloatMetricFeedback
                WHERE metric_name = {p0:String}
                GROUP BY target_id
            ) AS j0 ON i.id = j0.target_id",
        );

        assert_query_contains(
            &sql,
            r"FROM JsonInference AS i
            LEFT JOIN (
                SELECT
                    target_id,
                    argMax(value, timestamp) as value
                FROM FloatMetricFeedback
                WHERE metric_name = {p2:String}
                GROUP BY target_id
            ) AS j0 ON i.id = j0.target_id",
        );

        // Verify the metric filter condition is in WHERE
        assert_query_contains(&sql, "j0.value >");
    }
}
