use async_trait::async_trait;
use itertools::Itertools;

use crate::db::clickhouse::query_builder::parameters::add_parameter;
use crate::db::clickhouse::query_builder::{
    generate_order_by_sql, ClickhouseType, JoinRegistry, OrderByTerm, QueryParameter,
};
use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::db::inferences::{
    ClickHouseStoredInferenceWithDispreferredOutputs, GetInferenceBoundsParams, InferenceBounds,
    InferenceMetadata, InferenceOutputSource, InferenceQueries, ListInferencesByIdParams,
    ListInferencesParams, PaginateByIdCondition,
};
use crate::function::FunctionConfigType;
use crate::{
    config::Config,
    error::{Error, ErrorDetails},
    stored_inference::StoredInferenceDatabase,
};

/// Represents the structured parts of a single-table query
/// This allows the caller to insert JOINs between the SELECT/FROM and WHERE clauses
struct SingleTableQuery {
    /// The SELECT and FROM clauses (e.g., "SELECT ... FROM table AS i")
    select_from_sql_fragment: String,
    /// The WHERE clause if present (e.g., "WHERE condition1 AND condition2"), or empty string
    where_sql_fragment: String,
}

#[async_trait]
impl InferenceQueries for ClickHouseConnectionInfo {
    async fn list_inferences(
        &self,
        config: &Config,
        params: &ListInferencesParams<'_>,
    ) -> Result<Vec<StoredInferenceDatabase>, Error> {
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
            .collect::<Result<Vec<StoredInferenceDatabase>, Error>>()?;
        Ok(inferences)
    }

    async fn get_inference_bounds(
        &self,
        params: GetInferenceBoundsParams,
    ) -> Result<InferenceBounds, Error> {
        let mut query_params: Vec<QueryParameter> = Vec::new();
        let mut param_idx_counter = 0;
        let mut where_clauses: Vec<String> = Vec::new();

        // Add function_name filter
        if let Some(function_name) = params.function_name {
            let function_name_param = add_parameter(
                function_name,
                ClickhouseType::String,
                &mut query_params,
                &mut param_idx_counter,
            );
            where_clauses.push(format!("function_name = {function_name_param}"));
        }

        // Add variant_name filter
        if let Some(variant_name) = params.variant_name {
            let variant_name_param = add_parameter(
                variant_name,
                ClickhouseType::String,
                &mut query_params,
                &mut param_idx_counter,
            );
            where_clauses.push(format!("variant_name = {variant_name_param}"));
        }

        // Add episode_id filter
        if let Some(episode_id) = params.episode_id {
            let episode_id_param = add_parameter(
                episode_id.to_string(),
                ClickhouseType::String,
                &mut query_params,
                &mut param_idx_counter,
            );
            where_clauses.push(format!("episode_id = {episode_id_param}"));
        }

        // Build WHERE clause
        let where_clause = if where_clauses.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", where_clauses.join(" AND "))
        };

        // Build the query
        // Note: We use uint_to_uuid() to convert UInt128 back to UUID format
        // MIN(id_uint) = oldest (earliest_id), MAX(id_uint) = most recent (latest_id)
        let mut query = format!(
            r"SELECT
    uint_to_uuid(MAX(id_uint)) AS latest_id,
    uint_to_uuid(MIN(id_uint)) AS earliest_id,
    toUInt64(COUNT()) AS count
FROM InferenceById FINAL
{where_clause}"
        );
        query.push_str("\nFORMAT JSONEachRow");

        // Execute the query
        let query_params_map = query_params
            .iter()
            .map(|p| (p.name.as_str(), p.value.as_str()))
            .collect();

        let response = self
            .inner
            .run_query_synchronous(query, &query_params_map)
            .await?;

        let rows: Vec<InferenceBounds> = response
            .response
            .trim()
            .lines()
            .map(|line| {
                serde_json::from_str(line).map_err(|e| {
                    Error::new(ErrorDetails::ClickHouseQuery {
                        message: format!("Failed to deserialize bounds response: {e:?}"),
                    })
                })
            })
            .try_collect()?;

        // Handle empty results
        if rows.is_empty() || rows[0].count == 0 {
            return Ok(InferenceBounds::empty());
        }

        Ok(InferenceBounds {
            latest_id: rows[0].latest_id,
            earliest_id: rows[0].earliest_id,
            count: rows[0].count,
        })
    }

    async fn list_inferences_by_id(
        &self,
        params: ListInferencesByIdParams,
    ) -> Result<Vec<InferenceMetadata>, Error> {
        let mut query_params: Vec<QueryParameter> = Vec::new();
        let mut param_idx_counter: usize = 0;

        // Build WHERE clauses
        let mut where_clauses = Vec::new();

        if let Some(function_name) = &params.function_name {
            let param_placeholder = add_parameter(
                function_name.clone(),
                ClickhouseType::String,
                &mut query_params,
                &mut param_idx_counter,
            );
            where_clauses.push(format!("function_name = {param_placeholder}"));
        }

        if let Some(variant_name) = &params.variant_name {
            let param_placeholder = add_parameter(
                variant_name.clone(),
                ClickhouseType::String,
                &mut query_params,
                &mut param_idx_counter,
            );
            where_clauses.push(format!("variant_name = {param_placeholder}"));
        }

        if let Some(episode_id) = &params.episode_id {
            let param_placeholder = add_parameter(
                episode_id.to_string(),
                ClickhouseType::String,
                &mut query_params,
                &mut param_idx_counter,
            );
            where_clauses.push(format!("episode_id = {param_placeholder}"));
        }

        match params.pagination {
            Some(PaginateByIdCondition::Before { id }) => {
                let param_placeholder = add_parameter(
                    id.to_string(),
                    ClickhouseType::String,
                    &mut query_params,
                    &mut param_idx_counter,
                );
                where_clauses.push(format!("id_uint < toUInt128(toUUID({param_placeholder}))"));
            }
            Some(PaginateByIdCondition::After { id }) => {
                let param_placeholder = add_parameter(
                    id.to_string(),
                    ClickhouseType::String,
                    &mut query_params,
                    &mut param_idx_counter,
                );
                where_clauses.push(format!("id_uint > toUInt128(toUUID({param_placeholder}))"));
            }
            None => {}
        }

        let where_clause = if where_clauses.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", where_clauses.join(" AND "))
        };

        let limit_param_placeholder = add_parameter(
            params.limit,
            ClickhouseType::UInt64,
            &mut query_params,
            &mut param_idx_counter,
        );

        let query = if let Some(PaginateByIdCondition::After { .. }) = params.pagination {
            // After case: select ascending then re-order descending
            format!(
                "SELECT
                    id,
                    function_name,
                    variant_name,
                    episode_id,
                    function_type,
                    formatDateTime(UUIDv7ToDateTime(id), '%Y-%m-%dT%H:%i:%SZ') AS timestamp
                FROM
                (
                    SELECT
                        uint_to_uuid(id_uint) as id,
                        id_uint,
                        function_name,
                        variant_name,
                        episode_id,
                        function_type
                    FROM InferenceById FINAL
                    {where_clause}
                    ORDER BY id_uint ASC
                    LIMIT {limit_param_placeholder}
                )
                ORDER BY id_uint DESC
                FORMAT JSONEachRow"
            )
        } else {
            // No pagination or before: simple DESC query
            format!(
                "SELECT
                    uint_to_uuid(id_uint) as id,
                    function_name,
                    variant_name,
                    episode_id,
                    function_type,
                    formatDateTime(UUIDv7ToDateTime(uint_to_uuid(id_uint)), '%Y-%m-%dT%H:%i:%SZ') AS timestamp
                FROM InferenceById FINAL
                {where_clause}
                ORDER BY id_uint DESC
                LIMIT {limit_param_placeholder}
                FORMAT JSONEachRow"
            )
        };

        // Execute the query
        let query_params_map = query_params
            .iter()
            .map(|p| (p.name.as_str(), p.value.as_str()))
            .collect();

        let response = self
            .inner
            .run_query_synchronous(query, &query_params_map)
            .await?;

        let rows: Vec<InferenceMetadata> = response
            .response
            .trim()
            .lines()
            .map(|line| {
                serde_json::from_str(line).map_err(|e| {
                    Error::new(ErrorDetails::ClickHouseQuery {
                        message: format!("Failed to deserialize inference row: {e:?}"),
                    })
                })
            })
            .try_collect()?;

        Ok(rows)
    }
}

/// Escapes a string for JSON without quotes.
/// This is used to escape the text query when we doing a substring match on input and output strings, because
/// input and output strings are JSON-escaped in ClickHouse.
fn json_escape_string_without_quotes(s: &str) -> Result<String, Error> {
    let mut json_escaped = serde_json::to_string(s)?;
    json_escaped.remove(0);
    json_escaped.pop();
    Ok(json_escaped)
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
            let query = generate_single_table_query_for_type(
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
                    opts,
                    config,
                    &mut query_params,
                    &mut param_idx_counter,
                    &mut joins,
                )?
            } else {
                String::new()
            };

            // Construct final query: SELECT/FROM + JOINs + WHERE + ORDER BY
            let mut sql = query.select_from_sql_fragment;
            if !joins.get_clauses().is_empty() {
                sql.push_str(&joins.get_clauses().join("\n"));
            }
            sql.push('\n');
            sql.push_str(&query.where_sql_fragment);
            if !order_by_sql.is_empty() {
                sql.push_str(&order_by_sql);
            }

            sql
        }
        None => {
            // Otherwise, we need to query both tables with UNION ALL
            let mut chat_joins = JoinRegistry::new();
            let chat_query = generate_single_table_query_for_type(
                config,
                opts,
                "ChatInference",
                true, // is_chat
                &mut chat_joins,
                &mut query_params,
                &mut param_idx_counter,
            )?;

            // Construct chat query: SELECT/FROM + JOINs + WHERE
            let mut chat_sql = chat_query.select_from_sql_fragment;
            if !chat_joins.get_clauses().is_empty() {
                chat_sql.push_str(&chat_joins.get_clauses().join("\n"));
            }
            chat_sql.push('\n');
            chat_sql.push_str(&chat_query.where_sql_fragment);

            let mut json_joins = JoinRegistry::new();
            let json_query = generate_single_table_query_for_type(
                config,
                opts,
                "JsonInference",
                false, // is_chat
                &mut json_joins,
                &mut query_params,
                &mut param_idx_counter,
            )?;

            // Construct json query: SELECT/FROM + JOINs + WHERE
            let mut json_sql = json_query.select_from_sql_fragment;
            if !json_joins.get_clauses().is_empty() {
                json_sql.push_str(&json_joins.get_clauses().join("\n"));
            }
            json_sql.push('\n');
            json_sql.push_str(&json_query.where_sql_fragment);

            // Generate ORDER BY clause for both inner and outer queries
            // For UNION ALL queries, we only support timestamp ordering (not metrics)
            // TODO(#4181): this should support proper ORDER BY generation that supports joining with metrics.
            let order_by_sql = if let Some(order_by) = opts.order_by {
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
                            OrderByTerm::SearchRelevance => {
                                if opts.search_query_experimental.is_none() {
                                    return Err(Error::new(ErrorDetails::InvalidRequest {
                                        message: "ORDER BY relevance requires search_query_experimental in the request".to_string(),
                                    }));
                                }
                                "total_term_frequency"
                            }
                        };
                        Ok(format!("{column} {}", o.direction.to_clickhouse_direction()))
                    })
                    .collect::<Result<Vec<_>, Error>>()?;

                if order_clauses.is_empty() {
                    String::new()
                } else {
                    format!("\nORDER BY {}", order_clauses.join(", "))
                }
            } else {
                String::new()
            };

            // Push LIMIT down into each subquery before UNION ALL
            // For UNION ALL, we need to fetch (LIMIT + OFFSET) rows from each table
            // because we don't know which table will contribute to the final result.
            // We then apply the final LIMIT/OFFSET on the outer query.
            //
            // IMPORTANT: When ORDER BY is specified, we add it to each subquery before the LIMIT.
            // Otherwise, the LIMIT will select rows based on the physical table ordering
            // (function_name, variant_name, episode_id), not the user's requested ordering
            // (e.g., timestamp DESC), resulting in incorrect results.
            let inner_limit = opts.limit + opts.offset;

            if !order_by_sql.is_empty() {
                chat_sql.push_str(&order_by_sql);
            }
            let chat_limit_param_placeholder = add_parameter(
                inner_limit,
                ClickhouseType::UInt64,
                &mut query_params,
                &mut param_idx_counter,
            );
            chat_sql.push_str(&format!("\nLIMIT {chat_limit_param_placeholder}"));

            if !order_by_sql.is_empty() {
                json_sql.push_str(&order_by_sql);
            }
            let json_limit_param_placeholder = add_parameter(
                inner_limit,
                ClickhouseType::UInt64,
                &mut query_params,
                &mut param_idx_counter,
            );
            json_sql.push_str(&format!("\nLIMIT {json_limit_param_placeholder}"));

            // Combine with UNION ALL and apply outer ORDER BY and LIMIT/OFFSET
            let outer_limit_param_placeholder = add_parameter(
                opts.limit,
                ClickhouseType::UInt64,
                &mut query_params,
                &mut param_idx_counter,
            );
            let outer_offset_param_placeholder = add_parameter(
                opts.offset,
                ClickhouseType::UInt64,
                &mut query_params,
                &mut param_idx_counter,
            );

            format!(
                "SELECT * FROM (\n{chat_sql}\nUNION ALL\n{json_sql}\n) AS combined{order_by_sql}\nLIMIT {outer_limit_param_placeholder}\nOFFSET {outer_offset_param_placeholder}"
            )
        }
    };

    // For single-table queries (function_name provided), apply LIMIT/OFFSET at the end
    if opts.function_name.is_some() {
        let limit_param_placeholder = add_parameter(
            opts.limit,
            ClickhouseType::UInt64,
            &mut query_params,
            &mut param_idx_counter,
        );
        sql.push_str(&format!("\nLIMIT {limit_param_placeholder}"));

        let offset_param_placeholder = add_parameter(
            opts.offset,
            ClickhouseType::UInt64,
            &mut query_params,
            &mut param_idx_counter,
        );
        sql.push_str(&format!("\nOFFSET {offset_param_placeholder}"));
    }

    sql.push_str("\nFORMAT JSONEachRow");

    Ok((sql, query_params))
}

/// Core query building logic for a specific table type
/// Returns a SingleTableQuery with SELECT/FROM and WHERE parts separated,
/// so the caller can insert JOINs between them
fn generate_single_table_query_for_type(
    config: &Config,
    opts: &ListInferencesParams<'_>,
    table_name: &str,
    is_chat: bool,
    joins: &mut JoinRegistry,
    query_params: &mut Vec<QueryParameter>,
    param_idx_counter: &mut usize,
) -> Result<SingleTableQuery, Error> {
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
        select_clauses.push("i.dynamic_tools as dynamic_tools".to_string());
        select_clauses.push("i.dynamic_provider_tools as dynamic_provider_tools".to_string());
        select_clauses.push("i.allowed_tools as allowed_tools".to_string());
        select_clauses.push("i.tool_choice as tool_choice".to_string());
        select_clauses.push("i.parallel_tool_calls as parallel_tool_calls".to_string());
    } else {
        select_clauses.push("'' as tool_params".to_string());
        select_clauses.push("[] as dynamic_tools".to_string());
        select_clauses.push("[] as dynamic_provider_tools".to_string());
        select_clauses.push("NULL as allowed_tools".to_string());
        select_clauses.push("NULL as tool_choice".to_string());
        select_clauses.push("NULL as parallel_tool_calls".to_string());
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

    // Add text query term frequency columns and filter
    if let Some(search_query_experimental) = opts.search_query_experimental {
        let json_escaped_text_query = json_escape_string_without_quotes(search_query_experimental)?;
        let text_query_param = add_parameter(
            json_escaped_text_query,
            ClickhouseType::String,
            query_params,
            param_idx_counter,
        );

        select_clauses.push(format!(
            "countSubstringsCaseInsensitiveUTF8(i.input, {text_query_param}) as input_term_frequency"
        ));
        select_clauses.push(format!(
            "countSubstringsCaseInsensitiveUTF8(i.output, {text_query_param}) as output_term_frequency"
        ));
        select_clauses.push(
            "input_term_frequency + output_term_frequency as total_term_frequency".to_string(),
        );

        where_clauses.push("total_term_frequency > 0".to_string());
    }

    let select_from_sql_fragment = format!(
        r"SELECT {select_clauses} FROM {table_name} AS i",
        select_clauses = select_clauses.iter().join(",\n    "),
    );

    let where_sql_fragment = if where_clauses.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", where_clauses.join(" AND "))
    };

    Ok(SingleTableQuery {
        select_from_sql_fragment,
        where_sql_fragment,
    })
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

    mod json_escape_string_without_quotes_tests {
        use crate::db::clickhouse::inference_queries::json_escape_string_without_quotes;

        #[test]
        fn test_json_escape_string_without_quotes() {
            assert_eq!(
                json_escape_string_without_quotes("").unwrap(),
                String::new()
            );
            assert_eq!(
                json_escape_string_without_quotes("test").unwrap(),
                "test".to_string()
            );
            assert_eq!(
                json_escape_string_without_quotes("123").unwrap(),
                "123".to_string()
            );
            assert_eq!(
                json_escape_string_without_quotes("he's").unwrap(),
                "he's".to_string()
            );
        }

        #[test]
        fn test_json_escape_string_escapes_correctly() {
            assert_eq!(
                json_escape_string_without_quotes(r#""test""#).unwrap(),
                r#"\"test\""#.to_string()
            );

            assert_eq!(
                json_escape_string_without_quotes(r"end of line\next line").unwrap(),
                r"end of line\\next line".to_string()
            );
        }
    }

    async fn get_e2e_config() -> Config {
        // Read the e2e config file
        Config::load_from_path_optional_verify_credentials(
            &ConfigFileGlob::new_from_path(Path::new("tests/e2e/config/tensorzero.*.toml"))
                .unwrap(),
            false,
        )
        .await
        .unwrap()
        .config
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

        let (sql, _params) = generate_list_inferences_sql(&config, &opts).unwrap();

        // Verify both tables are queried
        assert_query_contains(
            &sql,
            "SELECT * FROM (
        SELECT
            'chat' as type,
            formatDateTime(i.timestamp, '%Y-%m-%dT%H:%i:%SZ') as timestamp,
            i.episode_id as episode_id,
            i.function_name as function_name,
            i.id as inference_id,
            i.input as input,
            '' as output_schema,
            i.tags as tags,
            i.tool_params as tool_params,
            i.dynamic_tools as dynamic_tools,
            i.dynamic_provider_tools as dynamic_provider_tools,
            i.allowed_tools as allowed_tools,
            i.tool_choice as tool_choice,
            i.parallel_tool_calls as parallel_tool_calls,
            i.variant_name as variant_name,
            i.output as output
        FROM
            ChatInference AS i
        WHERE
            i.id IN ['01234567-89ab-cdef-0123-456789abcdef','fedcba98-7654-3210-fedc-ba9876543210']
        LIMIT {p0:UInt64}
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
            [] as dynamic_tools,
            [] as dynamic_provider_tools,
            NULL as allowed_tools,
            NULL as tool_choice,
            NULL as parallel_tool_calls,
            i.variant_name as variant_name,
            i.output as output
        FROM
            JsonInference AS i
        WHERE
            i.id IN ['01234567-89ab-cdef-0123-456789abcdef','fedcba98-7654-3210-fedc-ba9876543210']
        LIMIT {p1:UInt64}
        ) AS combined
        LIMIT {p2:UInt64}
        OFFSET {p3:UInt64}",
        );
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

        assert_query_contains(
            &sql,
            "SELECT
            'json' as type,
            formatDateTime(i.timestamp, '%Y-%m-%dT%H:%i:%SZ') as timestamp,
            i.episode_id as episode_id,
            i.function_name as function_name,
            i.id as inference_id,
            i.input as input,
            i.output_schema as output_schema,
            i.tags as tags,
            '' as tool_params,
            [] as dynamic_tools,
            [] as dynamic_provider_tools,
            NULL as allowed_tools,
            NULL as tool_choice,
            NULL as parallel_tool_calls,
            i.variant_name as variant_name,
            i.output as output
        FROM
            JsonInference AS i
        WHERE
            i.function_name = {p0:String}
            AND i.id IN ['01234567-89ab-cdef-0123-456789abcdef']
        LIMIT {p1:UInt64}
        OFFSET {p2:UInt64}",
        );

        // Verify NO UNION ALL
        assert_query_does_not_contain(&sql, "UNION ALL");

        // Verify only JsonInference is queried (extract_entities is a JSON function)
        assert_query_does_not_contain(&sql, "ChatInference");

        assert!(
            params.contains(&QueryParameter {
                name: "p0".to_string(),
                value: "extract_entities".to_string(),
            }),
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
        assert_query_contains(&sql, ") AS combined");
        assert_query_contains(&sql, "ORDER BY timestamp DESC");

        // Verify ORDER BY appears 3 times: 2 for inner subqueries, 1 for outer query
        // This is critical to ensure correct results when LIMIT is pushed down
        let order_by_count = sql.matches("ORDER BY timestamp DESC").count();
        assert_eq!(
            order_by_count, 3,
            "ORDER BY should appear 3 times: 2 inner + 1 outer to ensure correct LIMIT behavior"
        );

        // Verify LIMIT appears 3 times: 2 for inner subqueries, 1 for outer query
        let limit_count = sql.matches("LIMIT {p").count();
        assert_eq!(
            limit_count, 3,
            "LIMIT should appear 3 times: 2 inner + 1 outer"
        );

        // Verify OFFSET only appears once in the outer query
        let offset_count = sql.matches("OFFSET {p").count();
        assert_eq!(offset_count, 1, "OFFSET should appear once in outer query");
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

    mod get_inference_bounds_tests {
        use super::*;
        use crate::db::clickhouse::clickhouse_client::MockClickHouseClient;
        use crate::db::clickhouse::{
            ClickHouseConnectionInfo, ClickHouseResponse, ClickHouseResponseMetadata,
        };
        use crate::db::inferences::{GetInferenceBoundsParams, InferenceBounds, InferenceQueries};
        use std::sync::Arc;

        #[tokio::test]
        async fn test_get_inference_bounds_no_filters() {
            let mut mock_clickhouse_client = MockClickHouseClient::new();
            mock_clickhouse_client
                .expect_run_query_synchronous()
                .withf(|query, _| {
                    assert_query_contains(query, "
                    SELECT
                        uint_to_uuid(MAX(id_uint)) AS latest_id,
                        uint_to_uuid(MIN(id_uint)) AS earliest_id,
                        toUInt64(COUNT()) AS count
                    FROM InferenceById FINAL
                    FORMAT JSONEachRow");
                    true
                })
                .returning(|_, _| {
                    Ok(ClickHouseResponse {
                        response: r#"{"latest_id":"01234567-89ab-cdef-0123-456789abcdef","earliest_id":"fedcba98-7654-3210-fedc-ba9876543210","count":"42"}"#.to_string(),
                        metadata: ClickHouseResponseMetadata {
                            read_rows: 1,
                            written_rows: 0,
                        },
                    })
                });
            let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

            let result = conn
                .get_inference_bounds(GetInferenceBoundsParams {
                    function_name: None,
                    variant_name: None,
                    episode_id: None,
                })
                .await
                .unwrap();

            assert_eq!(
                result.latest_id,
                Some(Uuid::parse_str("01234567-89ab-cdef-0123-456789abcdef").unwrap())
            );
            assert_eq!(
                result.earliest_id,
                Some(Uuid::parse_str("fedcba98-7654-3210-fedc-ba9876543210").unwrap())
            );
            assert_eq!(result.count, 42);
        }

        #[tokio::test]
        async fn test_get_inference_bounds_with_function_name() {
            let mut mock_clickhouse_client = MockClickHouseClient::new();
            mock_clickhouse_client
                .expect_run_query_synchronous()
                .withf(|query, params| {
                    assert_query_contains(query, "WHERE function_name = {p0:String}");
                    assert_eq!(params.get("p0"), Some(&"test_function"));
                    true
                })
                .returning(|_, _| {
                    Ok(ClickHouseResponse {
                        response: r#"{"latest_id":"11111111-2222-3333-4444-555555555555","earliest_id":"aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee","count":"10"}"#.to_string(),
                        metadata: ClickHouseResponseMetadata {
                            read_rows: 1,
                            written_rows: 0,
                        },
                    })
                });
            let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

            let result = conn
                .get_inference_bounds(GetInferenceBoundsParams {
                    function_name: Some("test_function".to_string()),
                    variant_name: None,
                    episode_id: None,
                })
                .await
                .unwrap();

            assert_eq!(
                result.latest_id,
                Some(Uuid::parse_str("11111111-2222-3333-4444-555555555555").unwrap())
            );
            assert_eq!(result.count, 10);
        }

        #[tokio::test]
        async fn test_get_inference_bounds_with_variant_name() {
            let mut mock_clickhouse_client = MockClickHouseClient::new();
            mock_clickhouse_client
                .expect_run_query_synchronous()
                .withf(|query, params| {
                    assert_query_contains(query, "WHERE variant_name = {p0:String}");
                    assert_eq!(params.get("p0"), Some(&"test_variant"));
                    true
                })
                .returning(|_, _| {
                    Ok(ClickHouseResponse {
                        response: r#"{"latest_id":"22222222-3333-4444-5555-666666666666","earliest_id":"bbbbbbbb-cccc-dddd-eeee-ffffffffffff","count":"5"}"#.to_string(),
                        metadata: ClickHouseResponseMetadata {
                            read_rows: 1,
                            written_rows: 0,
                        },
                    })
                });
            let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

            let result = conn
                .get_inference_bounds(GetInferenceBoundsParams {
                    function_name: None,
                    variant_name: Some("test_variant".to_string()),
                    episode_id: None,
                })
                .await
                .unwrap();

            assert_eq!(
                result.latest_id,
                Some(Uuid::parse_str("22222222-3333-4444-5555-666666666666").unwrap())
            );
            assert_eq!(result.count, 5);
        }

        #[tokio::test]
        async fn test_get_inference_bounds_with_episode_id() {
            let episode_id = Uuid::parse_str("01234567-89ab-cdef-0123-456789abcdef").unwrap();
            let mut mock_clickhouse_client = MockClickHouseClient::new();
            mock_clickhouse_client
                .expect_run_query_synchronous()
                .withf(move |query, params| {
                    assert_query_contains(query, "WHERE episode_id = {p0:String}");
                    assert_eq!(
                        params.get("p0"),
                        Some(&"01234567-89ab-cdef-0123-456789abcdef")
                    );
                    true
                })
                .returning(|_, _| {
                    Ok(ClickHouseResponse {
                        response: r#"{"latest_id":"33333333-4444-5555-6666-777777777777","earliest_id":"cccccccc-dddd-eeee-ffff-000000000000","count":"3"}"#.to_string(),
                        metadata: ClickHouseResponseMetadata {
                            read_rows: 1,
                            written_rows: 0,
                        },
                    })
                });
            let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

            let result = conn
                .get_inference_bounds(GetInferenceBoundsParams {
                    function_name: None,
                    variant_name: None,
                    episode_id: Some(episode_id),
                })
                .await
                .unwrap();

            assert_eq!(
                result.latest_id,
                Some(Uuid::parse_str("33333333-4444-5555-6666-777777777777").unwrap())
            );
            assert_eq!(result.count, 3);
        }

        #[tokio::test]
        async fn test_get_inference_bounds_with_multiple_filters() {
            let episode_id = Uuid::parse_str("01234567-89ab-cdef-0123-456789abcdef").unwrap();
            let mut mock_clickhouse_client = MockClickHouseClient::new();
            mock_clickhouse_client
                .expect_run_query_synchronous()
                .withf(move |query, params| {
                    assert_query_contains(query, "
                    WHERE
                        function_name = {p0:String}
                        AND variant_name = {p1:String}
                        AND episode_id = {p2:String}
                    ");
                    assert_eq!(params.get("p0"), Some(&"test_function"));
                    assert_eq!(params.get("p1"), Some(&"test_variant"));
                    assert_eq!(
                        params.get("p2"),
                        Some(&"01234567-89ab-cdef-0123-456789abcdef")
                    );
                    true
                })
                .returning(|_, _| {
                    Ok(ClickHouseResponse {
                        response: r#"{"latest_id":"44444444-5555-6666-7777-888888888888","earliest_id":"dddddddd-eeee-ffff-0000-111111111111","count":"1"}"#.to_string(),
                        metadata: ClickHouseResponseMetadata {
                            read_rows: 1,
                            written_rows: 0,
                        },
                    })
                });
            let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

            let result = conn
                .get_inference_bounds(GetInferenceBoundsParams {
                    function_name: Some("test_function".to_string()),
                    variant_name: Some("test_variant".to_string()),
                    episode_id: Some(episode_id),
                })
                .await
                .unwrap();

            assert_eq!(
                result.latest_id,
                Some(Uuid::parse_str("44444444-5555-6666-7777-888888888888").unwrap())
            );
            assert_eq!(result.count, 1);
        }

        #[tokio::test]
        async fn test_get_inference_bounds_with_empty_response() {
            let mut mock_clickhouse_client = MockClickHouseClient::new();
            mock_clickhouse_client
                .expect_run_query_synchronous()
                .returning(|_, _| {
                    Ok(ClickHouseResponse {
                        response: String::new(), // Empty response
                        metadata: ClickHouseResponseMetadata {
                            read_rows: 0,
                            written_rows: 0,
                        },
                    })
                });
            let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

            let result = conn
                .get_inference_bounds(GetInferenceBoundsParams {
                    function_name: None,
                    variant_name: None,
                    episode_id: None,
                })
                .await
                .unwrap();

            assert_eq!(result, InferenceBounds::empty());
            assert_eq!(result.latest_id, None);
            assert_eq!(result.earliest_id, None);
            assert_eq!(result.count, 0);
        }

        #[tokio::test]
        async fn test_get_inference_bounds_with_zero_count() {
            let mut mock_clickhouse_client = MockClickHouseClient::new();
            mock_clickhouse_client
                .expect_run_query_synchronous()
                .returning(|_, _| {
                    Ok(ClickHouseResponse {
                        response: r#"{"latest_id":null,"earliest_id":null,"count":"0"}"#
                            .to_string(),
                        metadata: ClickHouseResponseMetadata {
                            read_rows: 1,
                            written_rows: 0,
                        },
                    })
                });
            let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

            let result = conn
                .get_inference_bounds(GetInferenceBoundsParams {
                    function_name: Some("nonexistent_function".to_string()),
                    variant_name: None,
                    episode_id: None,
                })
                .await
                .unwrap();

            assert_eq!(result, InferenceBounds::empty());
            assert_eq!(result.latest_id, None);
            assert_eq!(result.earliest_id, None);
            assert_eq!(result.count, 0);
        }
    }

    mod list_inferences_by_id_tests {
        use super::*;
        use crate::db::clickhouse::clickhouse_client::MockClickHouseClient;
        use crate::db::clickhouse::{
            ClickHouseConnectionInfo, ClickHouseResponse, ClickHouseResponseMetadata,
        };
        use crate::db::inferences::{
            InferenceQueries, ListInferencesByIdParams, PaginateByIdCondition,
        };
        use std::sync::Arc;

        #[tokio::test]
        async fn test_list_inferences_by_id_no_filters_desc() {
            let mut mock_clickhouse_client = MockClickHouseClient::new();
            mock_clickhouse_client
                .expect_run_query_synchronous()
                .withf(|query, params| {
                    // Should use the simple DESC query without subquery
                    assert_query_contains(
                        query,
                        "SELECT
                    uint_to_uuid(id_uint) as id,
                    function_name,
                    variant_name,
                    episode_id,
                    function_type,
                    formatDateTime(UUIDv7ToDateTime(uint_to_uuid(id_uint)), '%Y-%m-%dT%H:%i:%SZ') AS timestamp
                FROM InferenceById FINAL
                ORDER BY id_uint DESC
                LIMIT {p0:UInt64}
                FORMAT JSONEachRow",
                    );
                    assert_eq!(params.get("p0"), Some(&"10"));
                    true
                })
                .returning(|_, _| {
                    Ok(ClickHouseResponse {
                        response: r#"{"id":"01234567-89ab-cdef-0123-456789abcdef","function_name":"test_fn","variant_name":"test_var","episode_id":"fedcba98-7654-3210-fedc-ba9876543210","function_type":"chat","timestamp":"2024-01-01T12:00:00Z"}"#.to_string(),
                        metadata: ClickHouseResponseMetadata {
                            read_rows: 1,
                            written_rows: 0,
                        },
                    })
                });
            let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

            let result = conn
                .list_inferences_by_id(ListInferencesByIdParams {
                    function_name: None,
                    variant_name: None,
                    episode_id: None,
                    pagination: None,
                    limit: 10,
                })
                .await
                .unwrap();

            assert_eq!(result.len(), 1);
            assert_eq!(
                result[0].id,
                Uuid::parse_str("01234567-89ab-cdef-0123-456789abcdef").unwrap()
            );
            assert_eq!(result[0].function_name, "test_fn");
        }

        #[tokio::test]
        async fn test_list_inferences_by_id_with_filters() {
            let mut mock_clickhouse_client = MockClickHouseClient::new();
            mock_clickhouse_client
                .expect_run_query_synchronous()
                .withf(|query, params| {
                    assert_query_contains(query, "SELECT
                    uint_to_uuid(id_uint) as id,
                    function_name,
                    variant_name,
                    episode_id,
                    function_type,
                    formatDateTime(UUIDv7ToDateTime(uint_to_uuid(id_uint)), '%Y-%m-%dT%H:%i:%SZ') AS timestamp
                FROM InferenceById FINAL
                WHERE function_name = {p0:String} AND variant_name = {p1:String} AND episode_id = {p2:String}
                ORDER BY id_uint DESC
                LIMIT {p3:UInt64}
                FORMAT JSONEachRow");
                    assert_eq!(params.get("p0"), Some(&"test_function"));
                    assert_eq!(params.get("p1"), Some(&"test_variant"));
                    assert_eq!(params.get("p2"), Some(&"01234567-89ab-cdef-0123-456789abcdef"));
                    assert_eq!(params.get("p3"), Some(&"5"));
                    true
                })
                .returning(|_, _| {
                    Ok(ClickHouseResponse {
                        response: String::new(),
                        metadata: ClickHouseResponseMetadata {
                            read_rows: 0,
                            written_rows: 0,
                        },
                    })
                });
            let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

            let result = conn
                .list_inferences_by_id(ListInferencesByIdParams {
                    function_name: Some("test_function".to_string()),
                    variant_name: Some("test_variant".to_string()),
                    episode_id: Some(
                        Uuid::parse_str("01234567-89ab-cdef-0123-456789abcdef").unwrap(),
                    ),
                    pagination: None,
                    limit: 5,
                })
                .await
                .unwrap();

            assert_eq!(result.len(), 0);
        }

        #[tokio::test]
        async fn test_list_inferences_by_id_with_before_pagination() {
            let before_id = Uuid::parse_str("fedcba98-7654-3210-fedc-ba9876543210").unwrap();
            let mut mock_clickhouse_client = MockClickHouseClient::new();
            mock_clickhouse_client
                .expect_run_query_synchronous()
                .withf(move |query, params| {
                    assert_query_contains(query, "SELECT
                    uint_to_uuid(id_uint) as id,
                    function_name,
                    variant_name,
                    episode_id,
                    function_type,
                    formatDateTime(UUIDv7ToDateTime(uint_to_uuid(id_uint)), '%Y-%m-%dT%H:%i:%SZ') AS timestamp
                FROM InferenceById FINAL
                WHERE id_uint < toUInt128(toUUID({p0:String}))
                ORDER BY id_uint DESC
                LIMIT {p1:UInt64}
                FORMAT JSONEachRow");
                    assert_eq!(
                        params.get("p0"),
                        Some(&"fedcba98-7654-3210-fedc-ba9876543210")
                    );
                    true
                })
                .returning(|_, _| {
                    Ok(ClickHouseResponse {
                        response: String::new(),
                        metadata: ClickHouseResponseMetadata {
                            read_rows: 0,
                            written_rows: 0,
                        },
                    })
                });
            let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

            let result = conn
                .list_inferences_by_id(ListInferencesByIdParams {
                    function_name: None,
                    variant_name: None,
                    episode_id: None,
                    pagination: Some(PaginateByIdCondition::Before { id: before_id }),
                    limit: 10,
                })
                .await
                .unwrap();

            assert_eq!(result.len(), 0);
        }

        #[tokio::test]
        async fn test_list_inferences_by_id_with_after_pagination() {
            let after_id = Uuid::parse_str("01234567-89ab-cdef-0123-456789abcdef").unwrap();
            let mut mock_clickhouse_client = MockClickHouseClient::new();
            mock_clickhouse_client
                .expect_run_query_synchronous()
                .withf(move |query, params| {
                    assert_query_contains(
                        query,
                        "SELECT
                    id,
                    function_name,
                    variant_name,
                    episode_id,
                    function_type,
                    formatDateTime(UUIDv7ToDateTime(id), '%Y-%m-%dT%H:%i:%SZ') AS timestamp
                FROM
                (
                    SELECT
                        uint_to_uuid(id_uint) as id,
                        id_uint,
                        function_name,
                        variant_name,
                        episode_id,
                        function_type
                    FROM InferenceById FINAL
                    WHERE id_uint > toUInt128(toUUID({p0:String}))
                    ORDER BY id_uint ASC
                    LIMIT {p1:UInt64}
                )
                ORDER BY id_uint DESC
                FORMAT JSONEachRow",
                    );
                    assert_eq!(
                        params.get("p0"),
                        Some(&"01234567-89ab-cdef-0123-456789abcdef")
                    );
                    true
                })
                .returning(|_, _| {
                    Ok(ClickHouseResponse {
                        response: String::new(),
                        metadata: ClickHouseResponseMetadata {
                            read_rows: 0,
                            written_rows: 0,
                        },
                    })
                });
            let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

            let result = conn
                .list_inferences_by_id(ListInferencesByIdParams {
                    function_name: None,
                    variant_name: None,
                    episode_id: None,
                    pagination: Some(PaginateByIdCondition::After { id: after_id }),
                    limit: 10,
                })
                .await
                .unwrap();

            assert_eq!(result.len(), 0);
        }

        #[tokio::test]
        async fn test_list_inferences_by_id_with_after_pagination_and_filters() {
            let after_id = Uuid::parse_str("01234567-89ab-cdef-0123-456789abcdef").unwrap();
            let mut mock_clickhouse_client = MockClickHouseClient::new();
            mock_clickhouse_client
                .expect_run_query_synchronous()
                .withf(move |query, params| {
                    assert_query_contains(
                        query,
                        "SELECT
                    id,
                    function_name,
                    variant_name,
                    episode_id,
                    function_type,
                    formatDateTime(UUIDv7ToDateTime(id), '%Y-%m-%dT%H:%i:%SZ') AS timestamp
                FROM
                (
                    SELECT
                        uint_to_uuid(id_uint) as id,
                        id_uint,
                        function_name,
                        variant_name,
                        episode_id,
                        function_type
                    FROM InferenceById FINAL
                    WHERE function_name = {p0:String} AND id_uint > toUInt128(toUUID({p1:String}))
                    ORDER BY id_uint ASC
                    LIMIT {p2:UInt64}
                )
                ORDER BY id_uint DESC
                FORMAT JSONEachRow",
                    );
                    assert_eq!(params.get("p0"), Some(&"test_function"));
                    assert_eq!(
                        params.get("p1"),
                        Some(&"01234567-89ab-cdef-0123-456789abcdef")
                    );
                    assert_eq!(params.get("p2"), Some(&"10"));
                    true
                })
                .returning(|_, _| {
                    Ok(ClickHouseResponse {
                        response: String::new(),
                        metadata: ClickHouseResponseMetadata {
                            read_rows: 0,
                            written_rows: 0,
                        },
                    })
                });
            let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

            let result = conn
                .list_inferences_by_id(ListInferencesByIdParams {
                    function_name: Some("test_function".to_string()),
                    variant_name: None,
                    episode_id: None,
                    pagination: Some(PaginateByIdCondition::After { id: after_id }),
                    limit: 10,
                })
                .await
                .unwrap();

            assert_eq!(result.len(), 0);
        }
    }
}
