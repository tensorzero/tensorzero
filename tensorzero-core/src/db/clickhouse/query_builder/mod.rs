use std::{
    collections::HashMap,
    fmt::{self, Display},
};

use crate::{
    config::{Config, MetricConfigType},
    db::{clickhouse::query_builder::parameters::add_parameter, inferences::ListInferencesParams},
    error::{Error, ErrorDetails},
};

mod datapoint_queries;
pub(super) mod parameters;
pub use datapoint_queries::DatapointFilter;

// Re-export filter and ordering types from v1 API for backwards compatibility
pub use crate::endpoints::stored_inferences::v1::types::{
    BooleanMetricFilter, FloatComparisonOperator, FloatMetricFilter, InferenceFilter, OrderBy,
    OrderByTerm, OrderDirection, TagComparisonOperator, TagFilter, TimeComparisonOperator,
    TimeFilter,
};

#[cfg(test)]
pub mod test_util;

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

impl TimeComparisonOperator {
    pub fn to_clickhouse_operator(&self) -> &str {
        match self {
            TimeComparisonOperator::LessThan => "<",
            TimeComparisonOperator::LessThanOrEqual => "<=",
            TimeComparisonOperator::Equal => "=",
            TimeComparisonOperator::GreaterThan => ">",
            TimeComparisonOperator::GreaterThanOrEqual => ">=",
            TimeComparisonOperator::NotEqual => "!=",
        }
    }
}

impl TagComparisonOperator {
    pub fn to_clickhouse_operator(&self) -> &str {
        match self {
            TagComparisonOperator::Equal => "=",
            TagComparisonOperator::NotEqual => "!=",
        }
    }
}

impl OrderDirection {
    pub fn to_clickhouse_direction(&self) -> &str {
        match self {
            OrderDirection::Asc => "ASC",
            OrderDirection::Desc => "DESC",
        }
    }
}

#[derive(Hash, Eq, PartialEq, Debug)]
pub struct JoinKey {
    table: MetricConfigType,
    metric_name: String,
    inference_column_name: &'static str,
}

pub struct JoinRegistry {
    // map key to join alias
    aliases: HashMap<JoinKey, String>,
    // The actual JOIN clauses that have been registered
    clauses: Vec<String>,
}

impl Default for JoinRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl JoinRegistry {
    pub fn new() -> Self {
        Self {
            aliases: HashMap::new(),
            clauses: Vec::new(),
        }
    }

    pub fn get_clauses(&self) -> &[String] {
        &self.clauses
    }

    /// Add a new join clause to the registry if the join
    /// for a particular key has not been added yet.
    /// If this is the first time we're adding a join for a particular key,
    /// we will also add the join clause to the registry.
    ///
    /// Returns the alias for the joined table.
    pub(crate) fn get_or_insert(
        &mut self,
        key: JoinKey,
        params_map: &mut Vec<QueryParameter>,
        param_idx_counter: &mut usize,
    ) -> String {
        if let Some(alias) = self.aliases.get(&key) {
            return alias.clone(); // we already have a join for this key
        }
        let alias = format!("j{}", self.aliases.len());
        self.clauses.push(Self::build_clause(
            &alias,
            &key,
            params_map,
            param_idx_counter,
        ));
        self.aliases.insert(key, alias.clone());
        alias
    }

    /// Inserts a join clause that is not part of the filter tree.
    pub fn insert_unchecked(&mut self, clause: String) {
        self.clauses.push(clause);
    }

    fn build_clause(
        alias: &str,
        key: &JoinKey,
        params_map: &mut Vec<QueryParameter>,
        param_idx_counter: &mut usize,
    ) -> String {
        let table_name = key.table.to_clickhouse_table_name();
        let inference_table_column_name = key.inference_column_name;
        let metric_name_placeholder = add_parameter(
            &key.metric_name,
            ClickhouseType::String,
            params_map,
            param_idx_counter,
        );
        format!(
            r"
LEFT JOIN (
    SELECT
        target_id,
        argMax(value, timestamp) as value
    FROM {table_name}
    WHERE metric_name = {metric_name_placeholder}
    GROUP BY target_id
) AS {alias} ON i.{inference_table_column_name} = {alias}.target_id"
        )
    }
}

// TODO(shuyangli): Extract inference filters into their own file.
impl InferenceFilter {
    /// Converts the filter tree to a ClickHouse SQL string.
    ///
    /// The returned string will contain the filter condition that should be added to the WHERE clause.
    /// The `params_map` is updated with the parameters used in the filter condition.
    /// The `param_idx_counter` is updated with the current index of the parameter.
    /// The `_select_clauses` is not used *yet*--we will want to add metric columns to the SELECT clause for visibility and debugging.
    /// The `joins` is updated with the JOIN clauses.
    ///
    /// NOTE: This is not efficient at all yet. We are doing a lot of JOINs and GROUP BYs.
    /// We may be able to do this more efficiently by using subqueries and CTEs.
    /// We're also doing a join per filter on metric. In principle if there is a subtree of the tree that uses the same joined table,
    /// we could push the condition down into the query before the join
    pub fn to_clickhouse_sql(
        &self,
        config: &Config,
        params_map: &mut Vec<QueryParameter>,
        _select_clauses: &mut Vec<String>,
        joins: &mut JoinRegistry,
        param_idx_counter: &mut usize,
    ) -> Result<String, Error> {
        match self {
            InferenceFilter::FloatMetric(fm_node) => {
                let metric_config = config
                    .metrics
                    .get(fm_node.metric_name.as_str())
                    .ok_or_else(|| {
                        Error::new(ErrorDetails::InvalidMetricName {
                            metric_name: fm_node.metric_name.clone(),
                        })
                    })?;
                let inference_column_name = metric_config.level.inference_column_name();

                // 1. Create an alias and register the join clause for the join condition we'll need
                let key = JoinKey {
                    table: MetricConfigType::Float,
                    metric_name: fm_node.metric_name.clone(),
                    inference_column_name,
                };
                let join_alias = joins.get_or_insert(key, params_map, param_idx_counter);
                // 2. Set up query parameters for the filter condition
                let value_placeholder = add_parameter(
                    fm_node.value,
                    ClickhouseType::Float64,
                    params_map,
                    param_idx_counter,
                );

                // 3. return the filter condition
                // NOTE: if the join_alias is NULL, the filter condition will be NULL also
                // We handle this farther up the recursive tree
                let comparison_operator = fm_node.comparison_operator.to_clickhouse_operator();
                Ok(format!(
                    "{join_alias}.value {comparison_operator} {value_placeholder}"
                ))
            }
            InferenceFilter::BooleanMetric(bm_node) => {
                let metric_config = config.metrics.get(&bm_node.metric_name).ok_or_else(|| {
                    Error::new(ErrorDetails::InvalidMetricName {
                        metric_name: bm_node.metric_name.clone(),
                    })
                })?;
                let inference_column_name = metric_config.level.inference_column_name();
                // 1. Create an alias and register the join clause for the join condition we'll need
                let key = JoinKey {
                    table: MetricConfigType::Boolean,
                    metric_name: bm_node.metric_name.clone(),
                    inference_column_name,
                };
                let join_alias = joins.get_or_insert(key, params_map, param_idx_counter);
                // 2. Set up query parameters for the filter condition
                let bool_value_str = if bm_node.value { "1" } else { "0" };
                let value_placeholder = add_parameter(
                    bool_value_str,
                    ClickhouseType::Bool,
                    params_map,
                    param_idx_counter,
                );
                // 4. return the filter condition
                // NOTE: if the join_alias is NULL, the filter condition will be NULL also
                // We handle this farther up the recursive tree
                Ok(format!("{join_alias}.value = {value_placeholder}"))
            }
            InferenceFilter::Tag(TagFilter {
                key,
                value,
                comparison_operator,
            }) => {
                let key_placeholder =
                    add_parameter(key, ClickhouseType::String, params_map, param_idx_counter);
                let value_placeholder =
                    add_parameter(value, ClickhouseType::String, params_map, param_idx_counter);
                let comparison_operator = comparison_operator.to_clickhouse_operator();
                Ok(format!(
                    "i.tags[{key_placeholder}] {comparison_operator} {value_placeholder}"
                ))
            }
            InferenceFilter::Time(TimeFilter {
                time,
                comparison_operator,
            }) => {
                let time_placeholder = add_parameter(
                    time.to_string(),
                    ClickhouseType::String,
                    params_map,
                    param_idx_counter,
                );
                let comparison_operator = comparison_operator.to_clickhouse_operator();
                Ok(format!(
                    "i.timestamp {comparison_operator} parseDateTimeBestEffort({time_placeholder})"
                ))
            }
            InferenceFilter::And { children } => {
                let child_sqls: Vec<String> = children
                    .iter()
                    .map(|child| {
                        child.to_clickhouse_sql(
                            config,
                            params_map,
                            _select_clauses,
                            joins,
                            param_idx_counter,
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
            InferenceFilter::Or { children } => {
                let child_sqls: Vec<String> = children
                    .iter()
                    .map(|child| {
                        child.to_clickhouse_sql(
                            config,
                            params_map,
                            _select_clauses,
                            joins,
                            param_idx_counter,
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
            InferenceFilter::Not { child } => {
                let child_sql = child.to_clickhouse_sql(
                    config,
                    params_map,
                    _select_clauses,
                    joins,
                    param_idx_counter,
                )?;
                // We need to coalesce the filter condition to 1 if the join_alias is NULL
                // For a NOT filter we want to still be false if the join_alias is NULL
                // NOTE to reviewer: Is this the behavior we want?
                Ok(format!("NOT (COALESCE({child_sql}, 1))"))
            }
        }
    }
}

pub fn generate_order_by_sql(
    opts: &ListInferencesParams<'_>,
    config: &Config,
    params_map: &mut Vec<QueryParameter>,
    param_idx_counter: &mut usize,
    joins: &mut JoinRegistry,
) -> Result<String, Error> {
    let Some(order_by) = opts.order_by else {
        return Ok(String::new());
    };
    if order_by.is_empty() {
        return Ok(String::new());
    }

    for term in order_by {
        // TODO(shuyangli): Validate that if ORDER BY includes a metric, we should have an appropriate
        // metric inference filter.

        // If ORDER BY includes search_relevance, search_query_experimental must be provided
        if matches!(term.term, OrderByTerm::SearchRelevance)
            && opts.search_query_experimental.is_none()
        {
            return Err(Error::new(ErrorDetails::InvalidRequest {
                message:
                    "ORDER BY search_relevance requires search_query_experimental to be provided"
                        .to_string(),
            }));
        }
    }

    let mut order_by_clauses = Vec::new();
    for term in order_by {
        let sql_expr = match &term.term {
            OrderByTerm::Timestamp => "i.timestamp".to_string(),
            OrderByTerm::Metric { name } => {
                let metric_config = config.metrics.get(name).ok_or_else(|| {
                    Error::new(ErrorDetails::InvalidMetricName {
                        metric_name: name.clone(),
                    })
                })?;

                let inference_column_name = metric_config.level.inference_column_name();
                let key = JoinKey {
                    table: metric_config.r#type,
                    metric_name: name.clone(),
                    inference_column_name,
                };
                let join_alias = joins.get_or_insert(key, params_map, param_idx_counter);
                format!("{join_alias}.value")
            }
            OrderByTerm::SearchRelevance => {
                // Note: The total_term_frequency column is added in generate_single_table_query_for_type
                // when search_query_experimental is provided. The column is referenced directly here.
                "total_term_frequency".to_string()
            }
        };
        let direction = term.direction.to_clickhouse_direction();
        order_by_clauses.push(format!("{sql_expr} {direction} NULLS LAST"));
    }
    let joined_clauses = order_by_clauses.join(", ");
    Ok(format!("\nORDER BY {joined_clauses}"))
}

/// Represents a parameter to be set for the ClickHouse query.
/// The `name` is the internal name (e.g., "p0", "p1") used in `SET param_<name> = ...`
/// and in the `{<name>:DataType}` placeholder.
/// The `value` is the string representation of the value.
#[derive(Debug, Clone, PartialEq)]
pub struct QueryParameter {
    pub name: String,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ClickhouseType {
    String,
    Float64,
    Bool,
    UInt64,
}

impl Display for ClickhouseType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ClickhouseType::String => write!(f, "String"),
            ClickhouseType::Float64 => write!(f, "Float64"),
            ClickhouseType::Bool => write!(f, "Bool"),
            ClickhouseType::UInt64 => write!(f, "UInt64"),
        }
    }
}

#[cfg(test)]
mod tests {
    // TODO(shuyangli): Cleanly separate tests for ListInferences SQL generation from the filter generation tests.
    use chrono::DateTime;
    use serde_json::json;
    use std::path::Path;
    use uuid::Uuid;

    use crate::db::clickhouse::inference_queries::generate_list_inferences_sql;
    use crate::db::clickhouse::query_builder::test_util::{
        assert_query_contains, assert_query_equals,
    };
    use crate::db::inferences::{
        ClickHouseStoredInferenceWithDispreferredOutputs, InferenceOutputSource,
        ListInferencesParams,
    };
    use crate::inference::types::{
        ContentBlockChatOutput, JsonInferenceOutput, StoredInput, System,
    };
    use crate::stored_inference::StoredInferenceDatabase;
    use crate::tool::{AllowedTools, AllowedToolsChoice, ToolCallConfigDatabaseInsert};
    use crate::{config::ConfigFileGlob, inference::types::Text, tool::ToolChoice};

    use super::*;

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

    /// Tests the simplest possible query: list inferences for a function with no filters
    #[tokio::test(flavor = "multi_thread")]
    async fn test_simple_query_json_function() {
        let config = get_e2e_config().await;
        let opts = ListInferencesParams {
            function_name: Some("extract_entities"),
            ..Default::default()
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
        let expected_sql = r"
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
    i.function_name = {p0:String}
LIMIT {p1:UInt64}
OFFSET {p2:UInt64}
FORMAT JSONEachRow";
        assert_query_equals(&sql, expected_sql);
        let expected_params = vec![
            QueryParameter {
                name: "p0".to_string(),
                value: "extract_entities".to_string(),
            },
            QueryParameter {
                name: "p1".to_string(),
                value: "20".to_string(),
            },
            QueryParameter {
                name: "p2".to_string(),
                value: "0".to_string(),
            },
        ];
        assert_eq!(params, expected_params);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_simple_query_chat_function() {
        let config = get_e2e_config().await;
        let opts = ListInferencesParams {
            function_name: Some("write_haiku"),
            ..Default::default()
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
        let expected_sql = r"
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
    i.function_name = {p0:String}
LIMIT {p1:UInt64}
OFFSET {p2:UInt64}
FORMAT JSONEachRow";
        assert_query_equals(&sql, expected_sql);
        let expected_params = vec![
            QueryParameter {
                name: "p0".to_string(),
                value: "write_haiku".to_string(),
            },
            QueryParameter {
                name: "p1".to_string(),
                value: "20".to_string(),
            },
            QueryParameter {
                name: "p2".to_string(),
                value: "0".to_string(),
            },
        ];
        assert_eq!(params, expected_params);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_simple_query_with_float_filters() {
        let config = get_e2e_config().await;
        let filter_node = InferenceFilter::FloatMetric(FloatMetricFilter {
            metric_name: "jaccard_similarity".to_string(),
            value: 0.5,
            comparison_operator: FloatComparisonOperator::GreaterThan,
        });
        let opts = ListInferencesParams {
            function_name: Some("extract_entities"),
            filters: Some(&filter_node),
            ..Default::default()
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
        let expected_sql = r"
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
LEFT JOIN (
    SELECT
        target_id,
        argMax(value, timestamp) as value
    FROM FloatMetricFeedback
    WHERE metric_name = {p1:String}
    GROUP BY target_id
) AS j0 ON i.id = j0.target_id
WHERE
    i.function_name = {p0:String} AND j0.value > {p2:Float64}
LIMIT {p3:UInt64}
OFFSET {p4:UInt64}
FORMAT JSONEachRow";
        assert_query_equals(&sql, expected_sql);
        let expected_params = vec![
            QueryParameter {
                name: "p0".to_string(),
                value: "extract_entities".to_string(),
            },
            QueryParameter {
                name: "p1".to_string(),
                value: "jaccard_similarity".to_string(),
            },
            QueryParameter {
                name: "p2".to_string(),
                value: "0.5".to_string(),
            },
            QueryParameter {
                name: "p3".to_string(),
                value: "20".to_string(),
            },
            QueryParameter {
                name: "p4".to_string(),
                value: "0".to_string(),
            },
        ];
        assert_eq!(params, expected_params);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_unknown_function_name() {
        let config = get_e2e_config().await;
        let opts = ListInferencesParams {
            function_name: Some("unknown_function_name"),
            ..Default::default()
        };
        let result = generate_list_inferences_sql(&config, &opts);
        assert!(result.is_err());
        let expected_error = ErrorDetails::UnknownFunction {
            name: "unknown_function_name".to_string(),
        };
        assert_eq!(result.unwrap_err().get_details(), &expected_error);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_unknown_metric_name() {
        let config = get_e2e_config().await;
        let filter_node = InferenceFilter::FloatMetric(FloatMetricFilter {
            metric_name: "unknown_metric_name".to_string(),
            value: 0.5,
            comparison_operator: FloatComparisonOperator::GreaterThan,
        });
        let opts = ListInferencesParams {
            function_name: Some("extract_entities"),
            filters: Some(&filter_node),
            ..Default::default()
        };
        let result = generate_list_inferences_sql(&config, &opts);
        assert!(result.is_err());
        let expected_error = ErrorDetails::InvalidMetricName {
            metric_name: "unknown_metric_name".to_string(),
        };
        assert_eq!(result.unwrap_err().get_details(), &expected_error);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_demonstration_output_source() {
        let config = get_e2e_config().await;
        let opts = ListInferencesParams {
            function_name: Some("extract_entities"),
            output_source: InferenceOutputSource::Demonstration,
            ..Default::default()
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
        let expected_sql = r"
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
    demo_f.value AS output,
    [i.output] as dispreferred_outputs
FROM
    JsonInference AS i
JOIN (SELECT inference_id, argMax(value, timestamp) as value FROM DemonstrationFeedback GROUP BY inference_id ) AS demo_f ON i.id = demo_f.inference_id
WHERE
    i.function_name = {p0:String}
LIMIT {p1:UInt64}
OFFSET {p2:UInt64}
FORMAT JSONEachRow";
        assert_query_equals(&sql, expected_sql);
        let expected_params = vec![
            QueryParameter {
                name: "p0".to_string(),
                value: "extract_entities".to_string(),
            },
            QueryParameter {
                name: "p1".to_string(),
                value: "20".to_string(),
            },
            QueryParameter {
                name: "p2".to_string(),
                value: "0".to_string(),
            },
        ];
        assert_eq!(params, expected_params);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_boolean_metric_filter() {
        let config = get_e2e_config().await;
        let filter_node = InferenceFilter::BooleanMetric(BooleanMetricFilter {
            metric_name: "task_success".to_string(),
            value: true,
        });
        let opts = ListInferencesParams {
            function_name: Some("extract_entities"),
            filters: Some(&filter_node),
            ..Default::default()
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
        let expected_sql = r"
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
LEFT JOIN (
    SELECT
        target_id,
        argMax(value, timestamp) as value
    FROM BooleanMetricFeedback
    WHERE metric_name = {p1:String}
    GROUP BY target_id
) AS j0 ON i.id = j0.target_id
WHERE
    i.function_name = {p0:String} AND j0.value = {p2:Bool}
LIMIT {p3:UInt64}
OFFSET {p4:UInt64}
FORMAT JSONEachRow";
        assert_query_equals(&sql, expected_sql);
        let expected_params = vec![
            QueryParameter {
                name: "p0".to_string(),
                value: "extract_entities".to_string(),
            },
            QueryParameter {
                name: "p1".to_string(),
                value: "task_success".to_string(),
            },
            QueryParameter {
                name: "p2".to_string(),
                value: "1".to_string(),
            },
            QueryParameter {
                name: "p3".to_string(),
                value: "20".to_string(),
            },
            QueryParameter {
                name: "p4".to_string(),
                value: "0".to_string(),
            },
        ];
        assert_eq!(params, expected_params);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_boolean_metric_filter_false() {
        let config = get_e2e_config().await;
        let filter_node = InferenceFilter::BooleanMetric(BooleanMetricFilter {
            metric_name: "task_success".to_string(),
            value: false,
        });
        let opts = ListInferencesParams {
            function_name: Some("extract_entities"),
            filters: Some(&filter_node),
            ..Default::default()
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
        let expected_sql = r"
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
LEFT JOIN (
    SELECT
        target_id,
        argMax(value, timestamp) as value
    FROM BooleanMetricFeedback
    WHERE metric_name = {p1:String}
    GROUP BY target_id
) AS j0 ON i.id = j0.target_id
WHERE
    i.function_name = {p0:String} AND j0.value = {p2:Bool}
LIMIT {p3:UInt64}
OFFSET {p4:UInt64}
FORMAT JSONEachRow";
        assert_query_equals(&sql, expected_sql);
        let expected_params = vec![
            QueryParameter {
                name: "p0".to_string(),
                value: "extract_entities".to_string(),
            },
            QueryParameter {
                name: "p1".to_string(),
                value: "task_success".to_string(),
            },
            QueryParameter {
                name: "p2".to_string(),
                value: "0".to_string(),
            },
            QueryParameter {
                name: "p3".to_string(),
                value: "20".to_string(),
            },
            QueryParameter {
                name: "p4".to_string(),
                value: "0".to_string(),
            },
        ];
        assert_eq!(params, expected_params);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_and_filter_multiple_float_metrics() {
        let config = get_e2e_config().await;
        let filter_node = InferenceFilter::And {
            children: vec![
                InferenceFilter::FloatMetric(FloatMetricFilter {
                    metric_name: "jaccard_similarity".to_string(),
                    value: 0.5,
                    comparison_operator: FloatComparisonOperator::GreaterThan,
                }),
                // We test that the join is not duplicated
                InferenceFilter::FloatMetric(FloatMetricFilter {
                    metric_name: "jaccard_similarity".to_string(),
                    value: 0.8,
                    comparison_operator: FloatComparisonOperator::LessThan,
                }),
                InferenceFilter::FloatMetric(FloatMetricFilter {
                    metric_name: "brevity_score".to_string(),
                    value: 10.0,
                    comparison_operator: FloatComparisonOperator::LessThan,
                }),
            ],
        };
        let opts = ListInferencesParams {
            function_name: Some("extract_entities"),
            filters: Some(&filter_node),
            ..Default::default()
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
        let expected_sql = r"
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
LEFT JOIN (
    SELECT
        target_id,
        argMax(value, timestamp) as value
    FROM FloatMetricFeedback
    WHERE metric_name = {p1:String}
    GROUP BY target_id
) AS j0 ON i.id = j0.target_id

LEFT JOIN (
    SELECT
        target_id,
        argMax(value, timestamp) as value
    FROM FloatMetricFeedback
    WHERE metric_name = {p4:String}
    GROUP BY target_id
) AS j1 ON i.id = j1.target_id
WHERE
    i.function_name = {p0:String} AND (COALESCE(j0.value > {p2:Float64}, 0) AND COALESCE(j0.value < {p3:Float64}, 0) AND COALESCE(j1.value < {p5:Float64}, 0))
LIMIT {p6:UInt64}
OFFSET {p7:UInt64}
FORMAT JSONEachRow";
        assert_query_equals(&sql, expected_sql);
        let expected_params = vec![
            QueryParameter {
                name: "p0".to_string(),
                value: "extract_entities".to_string(),
            },
            QueryParameter {
                name: "p1".to_string(),
                value: "jaccard_similarity".to_string(),
            },
            QueryParameter {
                name: "p2".to_string(),
                value: "0.5".to_string(),
            },
            QueryParameter {
                name: "p3".to_string(),
                value: "0.8".to_string(),
            },
            QueryParameter {
                name: "p4".to_string(),
                value: "brevity_score".to_string(),
            },
            QueryParameter {
                name: "p5".to_string(),
                value: "10".to_string(),
            },
            QueryParameter {
                name: "p6".to_string(),
                value: "20".to_string(),
            },
            QueryParameter {
                name: "p7".to_string(),
                value: "0".to_string(),
            },
        ];
        assert_eq!(params, expected_params);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_or_filter_mixed_metrics() {
        let config = get_e2e_config().await;
        let filter_node = InferenceFilter::Or {
            children: vec![
                InferenceFilter::FloatMetric(FloatMetricFilter {
                    metric_name: "jaccard_similarity".to_string(),
                    value: 0.8,
                    comparison_operator: FloatComparisonOperator::GreaterThanOrEqual,
                }),
                InferenceFilter::BooleanMetric(BooleanMetricFilter {
                    metric_name: "exact_match".to_string(),
                    value: true,
                }),
                InferenceFilter::BooleanMetric(BooleanMetricFilter {
                    // Episode-level metric
                    metric_name: "goal_achieved".to_string(),
                    value: true,
                }),
            ],
        };
        let opts = ListInferencesParams {
            function_name: Some("extract_entities"),
            filters: Some(&filter_node),
            ..Default::default()
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
        let expected_sql = r"
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
LEFT JOIN (
    SELECT
        target_id,
        argMax(value, timestamp) as value
    FROM FloatMetricFeedback
    WHERE metric_name = {p1:String}
    GROUP BY target_id
) AS j0 ON i.id = j0.target_id

LEFT JOIN (
    SELECT
        target_id,
        argMax(value, timestamp) as value
    FROM BooleanMetricFeedback
    WHERE metric_name = {p3:String}
    GROUP BY target_id
) AS j1 ON i.id = j1.target_id

LEFT JOIN (
    SELECT
        target_id,
        argMax(value, timestamp) as value
    FROM BooleanMetricFeedback
    WHERE metric_name = {p5:String}
    GROUP BY target_id
) AS j2 ON i.episode_id = j2.target_id
WHERE
    i.function_name = {p0:String} AND (COALESCE(j0.value >= {p2:Float64}, 0) OR COALESCE(j1.value = {p4:Bool}, 0) OR COALESCE(j2.value = {p6:Bool}, 0))
LIMIT {p7:UInt64}
OFFSET {p8:UInt64}
FORMAT JSONEachRow";
        assert_query_equals(&sql, expected_sql);
        let expected_params = vec![
            QueryParameter {
                name: "p0".to_string(),
                value: "extract_entities".to_string(),
            },
            QueryParameter {
                name: "p1".to_string(),
                value: "jaccard_similarity".to_string(),
            },
            QueryParameter {
                name: "p2".to_string(),
                value: "0.8".to_string(),
            },
            QueryParameter {
                name: "p3".to_string(),
                value: "exact_match".to_string(),
            },
            QueryParameter {
                name: "p4".to_string(),
                value: "1".to_string(),
            },
            QueryParameter {
                name: "p5".to_string(),
                value: "goal_achieved".to_string(),
            },
            QueryParameter {
                name: "p6".to_string(),
                value: "1".to_string(),
            },
            QueryParameter {
                name: "p7".to_string(),
                value: "20".to_string(),
            },
            QueryParameter {
                name: "p8".to_string(),
                value: "0".to_string(),
            },
        ];
        assert_eq!(params, expected_params);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_not_filter() {
        let config = get_e2e_config().await;
        let filter_node = InferenceFilter::Not {
            child: Box::new(InferenceFilter::Or {
                children: vec![
                    InferenceFilter::BooleanMetric(BooleanMetricFilter {
                        metric_name: "task_success".to_string(),
                        value: true,
                    }),
                    InferenceFilter::BooleanMetric(BooleanMetricFilter {
                        metric_name: "task_success".to_string(),
                        value: false,
                    }),
                ],
            }),
        };
        let opts = ListInferencesParams {
            function_name: Some("extract_entities"),
            filters: Some(&filter_node),
            ..Default::default()
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
        let expected_sql = r"
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
LEFT JOIN (
    SELECT
        target_id,
        argMax(value, timestamp) as value
    FROM BooleanMetricFeedback
    WHERE metric_name = {p1:String}
    GROUP BY target_id
) AS j0 ON i.id = j0.target_id
WHERE
    i.function_name = {p0:String} AND NOT (COALESCE((COALESCE(j0.value = {p2:Bool}, 0) OR COALESCE(j0.value = {p3:Bool}, 0)), 1))
LIMIT {p4:UInt64}
OFFSET {p5:UInt64}
FORMAT JSONEachRow";
        assert_query_equals(&sql, expected_sql);
        let expected_params = vec![
            QueryParameter {
                name: "p0".to_string(),
                value: "extract_entities".to_string(),
            },
            QueryParameter {
                name: "p1".to_string(),
                value: "task_success".to_string(),
            },
            QueryParameter {
                name: "p2".to_string(),
                value: "1".to_string(),
            },
            QueryParameter {
                name: "p3".to_string(),
                value: "0".to_string(),
            },
            QueryParameter {
                name: "p4".to_string(),
                value: "20".to_string(),
            },
            QueryParameter {
                name: "p5".to_string(),
                value: "0".to_string(),
            },
        ];
        assert_eq!(params, expected_params);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_nested_complex_filter() {
        let config = get_e2e_config().await;
        let filter_node = InferenceFilter::And {
            children: vec![
                InferenceFilter::Or {
                    children: vec![
                        InferenceFilter::FloatMetric(FloatMetricFilter {
                            metric_name: "jaccard_similarity".to_string(),
                            value: 0.7,
                            comparison_operator: FloatComparisonOperator::GreaterThan,
                        }),
                        InferenceFilter::FloatMetric(FloatMetricFilter {
                            metric_name: "brevity_score".to_string(),
                            value: 5.0,
                            comparison_operator: FloatComparisonOperator::LessThanOrEqual,
                        }),
                    ],
                },
                InferenceFilter::Not {
                    child: Box::new(InferenceFilter::BooleanMetric(BooleanMetricFilter {
                        metric_name: "task_success".to_string(),
                        value: false,
                    })),
                },
            ],
        };
        let opts = ListInferencesParams {
            function_name: Some("extract_entities"),
            filters: Some(&filter_node),
            ..Default::default()
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
        let expected_sql = r"
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
LEFT JOIN (
    SELECT
        target_id,
        argMax(value, timestamp) as value
    FROM FloatMetricFeedback
    WHERE metric_name = {p1:String}
    GROUP BY target_id
) AS j0 ON i.id = j0.target_id

LEFT JOIN (
    SELECT
        target_id,
        argMax(value, timestamp) as value
    FROM FloatMetricFeedback
    WHERE metric_name = {p3:String}
    GROUP BY target_id
) AS j1 ON i.id = j1.target_id

LEFT JOIN (
    SELECT
        target_id,
        argMax(value, timestamp) as value
    FROM BooleanMetricFeedback
    WHERE metric_name = {p5:String}
    GROUP BY target_id
) AS j2 ON i.id = j2.target_id
WHERE
    i.function_name = {p0:String} AND (COALESCE((COALESCE(j0.value > {p2:Float64}, 0) OR COALESCE(j1.value <= {p4:Float64}, 0)), 0) AND COALESCE(NOT (COALESCE(j2.value = {p6:Bool}, 1)), 0))
LIMIT {p7:UInt64}
OFFSET {p8:UInt64}
FORMAT JSONEachRow";
        assert_query_equals(&sql, expected_sql);
        assert_eq!(params.len(), 9); // p0 (function) + 6 metric-related params + 2 limit/offset params
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_nested_complex_filter_with_time() {
        let config = get_e2e_config().await;
        let filter_node = InferenceFilter::And {
            children: vec![
                InferenceFilter::Time(TimeFilter {
                    time: DateTime::from_timestamp(1609459200, 0).unwrap(), // 2021-01-01 00:00:00 UTC
                    comparison_operator: TimeComparisonOperator::GreaterThan,
                }),
                InferenceFilter::Or {
                    children: vec![
                        InferenceFilter::Time(TimeFilter {
                            time: DateTime::from_timestamp(1672531200, 0).unwrap(), // 2023-01-01 00:00:00 UTC
                            comparison_operator: TimeComparisonOperator::LessThan,
                        }),
                        InferenceFilter::And {
                            children: vec![
                                InferenceFilter::FloatMetric(FloatMetricFilter {
                                    metric_name: "jaccard_similarity".to_string(),
                                    value: 0.9,
                                    comparison_operator:
                                        FloatComparisonOperator::GreaterThanOrEqual,
                                }),
                                InferenceFilter::Tag(TagFilter {
                                    key: "priority".to_string(),
                                    value: "high".to_string(),
                                    comparison_operator: TagComparisonOperator::Equal,
                                }),
                            ],
                        },
                    ],
                },
            ],
        };
        let opts = ListInferencesParams {
            function_name: Some("extract_entities"),
            filters: Some(&filter_node),
            ..Default::default()
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
        let expected_sql = r"
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
LEFT JOIN (
    SELECT
        target_id,
        argMax(value, timestamp) as value
    FROM FloatMetricFeedback
    WHERE metric_name = {p3:String}
    GROUP BY target_id
) AS j0 ON i.id = j0.target_id
WHERE
    i.function_name = {p0:String} AND (COALESCE(i.timestamp > parseDateTimeBestEffort({p1:String}), 0) AND COALESCE((COALESCE(i.timestamp < parseDateTimeBestEffort({p2:String}), 0) OR COALESCE((COALESCE(j0.value >= {p4:Float64}, 0) AND COALESCE(i.tags[{p5:String}] = {p6:String}, 0)), 0)), 0))
LIMIT {p7:UInt64}
OFFSET {p8:UInt64}
FORMAT JSONEachRow";
        assert_query_equals(&sql, expected_sql);
        assert_eq!(params.len(), 9); // p0 (function) + 6 filter-related params + 2 limit/offset params
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_variant_name_filter() {
        let config = get_e2e_config().await;
        let opts = ListInferencesParams {
            function_name: Some("extract_entities"),
            variant_name: Some("v1"),
            filters: None,
            ..Default::default()
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
        let expected_sql = r"
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
    i.function_name = {p0:String} AND i.variant_name = {p1:String}
LIMIT {p2:UInt64}
OFFSET {p3:UInt64}
FORMAT JSONEachRow";
        assert_query_equals(&sql, expected_sql);
        let expected_params = vec![
            QueryParameter {
                name: "p0".to_string(),
                value: "extract_entities".to_string(),
            },
            QueryParameter {
                name: "p1".to_string(),
                value: "v1".to_string(),
            },
            QueryParameter {
                name: "p2".to_string(),
                value: "20".to_string(),
            },
            QueryParameter {
                name: "p3".to_string(),
                value: "0".to_string(),
            },
        ];
        assert_eq!(params, expected_params);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_limit_and_offset() {
        let config = get_e2e_config().await;
        let opts = ListInferencesParams {
            function_name: Some("extract_entities"),
            limit: 50,
            offset: 100,
            ..Default::default()
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
        let expected_sql = r"
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
    i.function_name = {p0:String}
LIMIT {p1:UInt64}
OFFSET {p2:UInt64}
FORMAT JSONEachRow";
        assert_query_equals(&sql, expected_sql);
        let expected_params = vec![
            QueryParameter {
                name: "p0".to_string(),
                value: "extract_entities".to_string(),
            },
            QueryParameter {
                name: "p1".to_string(),
                value: "50".to_string(),
            },
            QueryParameter {
                name: "p2".to_string(),
                value: "100".to_string(),
            },
        ];
        assert_eq!(params, expected_params);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_all_float_comparison_operators() {
        let config = get_e2e_config().await;
        let operators = vec![
            (FloatComparisonOperator::LessThan, "<"),
            (FloatComparisonOperator::LessThanOrEqual, "<="),
            (FloatComparisonOperator::Equal, "="),
            (FloatComparisonOperator::GreaterThan, ">"),
            (FloatComparisonOperator::GreaterThanOrEqual, ">="),
            (FloatComparisonOperator::NotEqual, "!="),
        ];

        for (op, expected_op_str) in operators {
            let filter_node = InferenceFilter::FloatMetric(FloatMetricFilter {
                metric_name: "jaccard_similarity".to_string(),
                value: 0.5,
                comparison_operator: op,
            });
            let opts = ListInferencesParams {
                function_name: Some("extract_entities"),
                filters: Some(&filter_node),
                ..Default::default()
            };
            let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
            // TODO(#4608) LIMIT and OFFSET need to be pushed down to subqueries.
            let expected_sql = format!(
                r"
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
LEFT JOIN (
    SELECT
        target_id,
        argMax(value, timestamp) as value
    FROM FloatMetricFeedback
    WHERE metric_name = {{p1:String}}
    GROUP BY target_id
) AS j0 ON i.id = j0.target_id
WHERE
    i.function_name = {{p0:String}} AND j0.value {expected_op_str} {{p2:Float64}}
LIMIT {{p3:UInt64}}
OFFSET {{p4:UInt64}}
FORMAT JSONEachRow",
            );
            assert_query_equals(&sql, &expected_sql);
            let expected_params = vec![
                QueryParameter {
                    name: "p0".to_string(),
                    value: "extract_entities".to_string(),
                },
                QueryParameter {
                    name: "p1".to_string(),
                    value: "jaccard_similarity".to_string(),
                },
                QueryParameter {
                    name: "p2".to_string(),
                    value: "0.5".to_string(),
                },
                QueryParameter {
                    name: "p3".to_string(),
                    value: "20".to_string(),
                },
                QueryParameter {
                    name: "p4".to_string(),
                    value: "0".to_string(),
                },
            ];
            assert_eq!(params, expected_params);
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_simple_tag_filter_equal() {
        let config = get_e2e_config().await;
        let filter_node = InferenceFilter::Tag(TagFilter {
            key: "environment".to_string(),
            value: "production".to_string(),
            comparison_operator: TagComparisonOperator::Equal,
        });
        let opts = ListInferencesParams {
            function_name: Some("extract_entities"),
            filters: Some(&filter_node),
            ..Default::default()
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
        let expected_sql = r"
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
    i.function_name = {p0:String} AND i.tags[{p1:String}] = {p2:String}
LIMIT {p3:UInt64}
OFFSET {p4:UInt64}
FORMAT JSONEachRow";
        assert_query_equals(&sql, expected_sql);
        let expected_params = vec![
            QueryParameter {
                name: "p0".to_string(),
                value: "extract_entities".to_string(),
            },
            QueryParameter {
                name: "p1".to_string(),
                value: "environment".to_string(),
            },
            QueryParameter {
                name: "p2".to_string(),
                value: "production".to_string(),
            },
            QueryParameter {
                name: "p3".to_string(),
                value: "20".to_string(),
            },
            QueryParameter {
                name: "p4".to_string(),
                value: "0".to_string(),
            },
        ];
        assert_eq!(params, expected_params);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_tag_filter_not_equal() {
        let config = get_e2e_config().await;
        let filter_node = InferenceFilter::Tag(TagFilter {
            key: "version".to_string(),
            value: "v1.0".to_string(),
            comparison_operator: TagComparisonOperator::NotEqual,
        });
        let opts = ListInferencesParams {
            function_name: Some("write_haiku"),
            filters: Some(&filter_node),
            ..Default::default()
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
        let expected_sql = r"
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
    i.function_name = {p0:String} AND i.tags[{p1:String}] != {p2:String}
LIMIT {p3:UInt64}
OFFSET {p4:UInt64}
FORMAT JSONEachRow";
        assert_query_equals(&sql, expected_sql);
        let expected_params = vec![
            QueryParameter {
                name: "p0".to_string(),
                value: "write_haiku".to_string(),
            },
            QueryParameter {
                name: "p1".to_string(),
                value: "version".to_string(),
            },
            QueryParameter {
                name: "p2".to_string(),
                value: "v1.0".to_string(),
            },
            QueryParameter {
                name: "p3".to_string(),
                value: "20".to_string(),
            },
            QueryParameter {
                name: "p4".to_string(),
                value: "0".to_string(),
            },
        ];
        assert_eq!(params, expected_params);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_tag_filters_in_and_condition() {
        let config = get_e2e_config().await;
        let filter_node = InferenceFilter::And {
            children: vec![
                InferenceFilter::Tag(TagFilter {
                    key: "environment".to_string(),
                    value: "production".to_string(),
                    comparison_operator: TagComparisonOperator::Equal,
                }),
                InferenceFilter::Tag(TagFilter {
                    key: "region".to_string(),
                    value: "us-west".to_string(),
                    comparison_operator: TagComparisonOperator::Equal,
                }),
            ],
        };
        let opts = ListInferencesParams {
            function_name: Some("extract_entities"),
            filters: Some(&filter_node),
            ..Default::default()
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
        let expected_sql = r"
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
    i.function_name = {p0:String} AND (COALESCE(i.tags[{p1:String}] = {p2:String}, 0) AND COALESCE(i.tags[{p3:String}] = {p4:String}, 0))
LIMIT {p5:UInt64}
OFFSET {p6:UInt64}
FORMAT JSONEachRow";
        assert_query_equals(&sql, expected_sql);
        let expected_params = vec![
            QueryParameter {
                name: "p0".to_string(),
                value: "extract_entities".to_string(),
            },
            QueryParameter {
                name: "p1".to_string(),
                value: "environment".to_string(),
            },
            QueryParameter {
                name: "p2".to_string(),
                value: "production".to_string(),
            },
            QueryParameter {
                name: "p3".to_string(),
                value: "region".to_string(),
            },
            QueryParameter {
                name: "p4".to_string(),
                value: "us-west".to_string(),
            },
            QueryParameter {
                name: "p5".to_string(),
                value: "20".to_string(),
            },
            QueryParameter {
                name: "p6".to_string(),
                value: "0".to_string(),
            },
        ];
        assert_eq!(params, expected_params);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_tag_and_metric_filters_combined() {
        let config = get_e2e_config().await;
        let filter_node = InferenceFilter::And {
            children: vec![
                InferenceFilter::Tag(TagFilter {
                    key: "experiment".to_string(),
                    value: "A".to_string(),
                    comparison_operator: TagComparisonOperator::Equal,
                }),
                InferenceFilter::FloatMetric(FloatMetricFilter {
                    metric_name: "jaccard_similarity".to_string(),
                    value: 0.7,
                    comparison_operator: FloatComparisonOperator::GreaterThan,
                }),
            ],
        };
        let opts = ListInferencesParams {
            function_name: Some("extract_entities"),
            filters: Some(&filter_node),
            ..Default::default()
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
        let expected_sql = r"
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
LEFT JOIN (
    SELECT
        target_id,
        argMax(value, timestamp) as value
    FROM FloatMetricFeedback
    WHERE metric_name = {p3:String}
    GROUP BY target_id
) AS j0 ON i.id = j0.target_id
WHERE
    i.function_name = {p0:String} AND (COALESCE(i.tags[{p1:String}] = {p2:String}, 0) AND COALESCE(j0.value > {p4:Float64}, 0))
LIMIT {p5:UInt64}
OFFSET {p6:UInt64}
FORMAT JSONEachRow";
        assert_query_equals(&sql, expected_sql);
        let expected_params = vec![
            QueryParameter {
                name: "p0".to_string(),
                value: "extract_entities".to_string(),
            },
            QueryParameter {
                name: "p1".to_string(),
                value: "experiment".to_string(),
            },
            QueryParameter {
                name: "p2".to_string(),
                value: "A".to_string(),
            },
            QueryParameter {
                name: "p3".to_string(),
                value: "jaccard_similarity".to_string(),
            },
            QueryParameter {
                name: "p4".to_string(),
                value: "0.7".to_string(),
            },
            QueryParameter {
                name: "p5".to_string(),
                value: "20".to_string(),
            },
            QueryParameter {
                name: "p6".to_string(),
                value: "0".to_string(),
            },
        ];
        assert_eq!(params, expected_params);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_combined_variant_filter_and_metrics() {
        let config = get_e2e_config().await;
        let filter_node = InferenceFilter::And {
            children: vec![
                InferenceFilter::FloatMetric(FloatMetricFilter {
                    metric_name: "jaccard_similarity".to_string(),
                    value: 0.6,
                    comparison_operator: FloatComparisonOperator::GreaterThan,
                }),
                InferenceFilter::BooleanMetric(BooleanMetricFilter {
                    metric_name: "exact_match".to_string(),
                    value: true,
                }),
            ],
        };
        let opts = ListInferencesParams {
            function_name: Some("extract_entities"),
            variant_name: Some("production"),
            filters: Some(&filter_node),
            output_source: InferenceOutputSource::Demonstration,
            limit: 25,
            offset: 50,
            ..Default::default()
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
        let expected_sql = r"
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
    demo_f.value AS output,
    [i.output] as dispreferred_outputs
FROM
    JsonInference AS i
JOIN (SELECT inference_id, argMax(value, timestamp) as value FROM DemonstrationFeedback GROUP BY inference_id ) AS demo_f ON i.id = demo_f.inference_id

LEFT JOIN (
    SELECT
        target_id,
        argMax(value, timestamp) as value
    FROM FloatMetricFeedback
    WHERE metric_name = {p2:String}
    GROUP BY target_id
) AS j0 ON i.id = j0.target_id

LEFT JOIN (
    SELECT
        target_id,
        argMax(value, timestamp) as value
    FROM BooleanMetricFeedback
    WHERE metric_name = {p4:String}
    GROUP BY target_id
) AS j1 ON i.id = j1.target_id
WHERE
    i.function_name = {p0:String} AND i.variant_name = {p1:String} AND (COALESCE(j0.value > {p3:Float64}, 0) AND COALESCE(j1.value = {p5:Bool}, 0))
LIMIT {p6:UInt64}
OFFSET {p7:UInt64}
FORMAT JSONEachRow";
        assert_query_equals(&sql, expected_sql);

        let expected_params = vec![
            QueryParameter {
                name: "p0".to_string(),
                value: "extract_entities".to_string(),
            },
            QueryParameter {
                name: "p1".to_string(),
                value: "production".to_string(),
            },
            QueryParameter {
                name: "p2".to_string(),
                value: "jaccard_similarity".to_string(),
            },
            QueryParameter {
                name: "p3".to_string(),
                value: "0.6".to_string(),
            },
            QueryParameter {
                name: "p4".to_string(),
                value: "exact_match".to_string(),
            },
            QueryParameter {
                name: "p5".to_string(),
                value: "1".to_string(),
            },
            QueryParameter {
                name: "p6".to_string(),
                value: "25".to_string(),
            },
            QueryParameter {
                name: "p7".to_string(),
                value: "50".to_string(),
            },
        ];
        assert_eq!(params, expected_params);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_simple_time_filter() {
        let config = get_e2e_config().await;
        let filter_node = InferenceFilter::Time(TimeFilter {
            time: DateTime::from_timestamp(1672531200, 0).unwrap(), // 2023-01-01 00:00:00 UTC
            comparison_operator: TimeComparisonOperator::GreaterThan,
        });
        let opts = ListInferencesParams {
            function_name: Some("extract_entities"),
            filters: Some(&filter_node),
            ..Default::default()
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
        let expected_sql = r"
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
    i.function_name = {p0:String} AND i.timestamp > parseDateTimeBestEffort({p1:String})
LIMIT {p2:UInt64}
OFFSET {p3:UInt64}
FORMAT JSONEachRow";
        assert_query_equals(&sql, expected_sql);
        let expected_params = vec![
            QueryParameter {
                name: "p0".to_string(),
                value: "extract_entities".to_string(),
            },
            QueryParameter {
                name: "p1".to_string(),
                value: "2023-01-01 00:00:00 UTC".to_string(),
            },
            QueryParameter {
                name: "p2".to_string(),
                value: "20".to_string(),
            },
            QueryParameter {
                name: "p3".to_string(),
                value: "0".to_string(),
            },
        ];
        assert_eq!(params, expected_params);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_all_time_comparison_operators() {
        let config = get_e2e_config().await;
        let operators = vec![
            (TimeComparisonOperator::LessThan, "<"),
            (TimeComparisonOperator::LessThanOrEqual, "<="),
            (TimeComparisonOperator::Equal, "="),
            (TimeComparisonOperator::GreaterThan, ">"),
            (TimeComparisonOperator::GreaterThanOrEqual, ">="),
            (TimeComparisonOperator::NotEqual, "!="),
        ];

        for (op, expected_op_str) in operators {
            let filter_node = InferenceFilter::Time(TimeFilter {
                time: DateTime::from_timestamp(1672531200, 0).unwrap(), // 2023-01-01 00:00:00 UTC
                comparison_operator: op,
            });
            let opts = ListInferencesParams {
                function_name: Some("write_haiku"),
                filters: Some(&filter_node),
                ..Default::default()
            };
            let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
            let expected_sql = format!(
                r"
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
    i.function_name = {{p0:String}} AND i.timestamp {expected_op_str} parseDateTimeBestEffort({{p1:String}})
LIMIT {{p2:UInt64}}
OFFSET {{p3:UInt64}}
FORMAT JSONEachRow",
            );
            assert_query_equals(&sql, &expected_sql);
            let expected_params = vec![
                QueryParameter {
                    name: "p0".to_string(),
                    value: "write_haiku".to_string(),
                },
                QueryParameter {
                    name: "p1".to_string(),
                    value: "2023-01-01 00:00:00 UTC".to_string(),
                },
                QueryParameter {
                    name: "p2".to_string(),
                    value: "20".to_string(),
                },
                QueryParameter {
                    name: "p3".to_string(),
                    value: "0".to_string(),
                },
            ];
            assert_eq!(params, expected_params);
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_time_filter_combined_with_other_filters() {
        let config = get_e2e_config().await;
        let filter_node = InferenceFilter::And {
            children: vec![
                InferenceFilter::Time(TimeFilter {
                    time: DateTime::from_timestamp(1672531200, 0).unwrap(), // 2023-01-01 00:00:00 UTC
                    comparison_operator: TimeComparisonOperator::GreaterThanOrEqual,
                }),
                InferenceFilter::Tag(TagFilter {
                    key: "environment".to_string(),
                    value: "production".to_string(),
                    comparison_operator: TagComparisonOperator::Equal,
                }),
                InferenceFilter::FloatMetric(FloatMetricFilter {
                    metric_name: "jaccard_similarity".to_string(),
                    value: 0.8,
                    comparison_operator: FloatComparisonOperator::GreaterThan,
                }),
            ],
        };
        let opts = ListInferencesParams {
            function_name: Some("extract_entities"),
            filters: Some(&filter_node),
            limit: 10,
            ..Default::default()
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
        let expected_sql = r"
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
LEFT JOIN (
    SELECT
        target_id,
        argMax(value, timestamp) as value
    FROM FloatMetricFeedback
    WHERE metric_name = {p4:String}
    GROUP BY target_id
) AS j0 ON i.id = j0.target_id
WHERE
    i.function_name = {p0:String} AND (COALESCE(i.timestamp >= parseDateTimeBestEffort({p1:String}), 0) AND COALESCE(i.tags[{p2:String}] = {p3:String}, 0) AND COALESCE(j0.value > {p5:Float64}, 0))
LIMIT {p6:UInt64}
OFFSET {p7:UInt64}
FORMAT JSONEachRow";
        assert_query_equals(&sql, expected_sql);
        let expected_params = vec![
            QueryParameter {
                name: "p0".to_string(),
                value: "extract_entities".to_string(),
            },
            QueryParameter {
                name: "p1".to_string(),
                value: "2023-01-01 00:00:00 UTC".to_string(),
            },
            QueryParameter {
                name: "p2".to_string(),
                value: "environment".to_string(),
            },
            QueryParameter {
                name: "p3".to_string(),
                value: "production".to_string(),
            },
            QueryParameter {
                name: "p4".to_string(),
                value: "jaccard_similarity".to_string(),
            },
            QueryParameter {
                name: "p5".to_string(),
                value: "0.8".to_string(),
            },
            QueryParameter {
                name: "p6".to_string(),
                value: "10".to_string(),
            },
            QueryParameter {
                name: "p7".to_string(),
                value: "0".to_string(),
            },
        ];
        assert_eq!(params, expected_params);
    }

    #[test]
    fn test_stored_inference_deserialization_chat() {
        // Test the ClickHouse version (doubly serialized)
        let json = r#"
            {
                "type": "chat",
                "function_name": "test_function",
                "variant_name": "test_variant",
                "input": "{\"system\": \"you are a helpful assistant\", \"messages\": []}",
                "output": "[{\"type\": \"text\", \"text\": \"Hello! How can I help you today?\"}]",
                "episode_id": "123e4567-e89b-12d3-a456-426614174000",
                "inference_id": "123e4567-e89b-12d3-a456-426614174000",
                "tags": {},
                "tool_params": "{\"tools_available\": [], \"tool_choice\": \"none\", \"parallel_tool_calls\": false}",
                "timestamp": "2023-01-01T00:00:00Z"
            }
        "#;
        let inference: ClickHouseStoredInferenceWithDispreferredOutputs =
            serde_json::from_str(json).unwrap();
        let StoredInferenceDatabase::Chat(chat_inference) = inference.try_into().unwrap() else {
            panic!("Expected a chat inference");
        };
        assert_eq!(chat_inference.function_name, "test_function");
        assert_eq!(chat_inference.variant_name, "test_variant");
        assert_eq!(
            chat_inference.input,
            StoredInput {
                system: Some(System::Text("you are a helpful assistant".to_string())),
                messages: vec![],
            }
        );
        assert_eq!(
            chat_inference.output,
            vec!["Hello! How can I help you today?".to_string().into()]
        );
        assert!(chat_inference.dispreferred_outputs.is_empty());
        assert_eq!(
            chat_inference.tool_params,
            ToolCallConfigDatabaseInsert::new_for_test(
                vec![],
                vec![],
                AllowedTools {
                    tools: vec![],
                    choice: AllowedToolsChoice::FunctionDefault,
                },
                ToolChoice::None,
                Some(false),
            )
        );

        // Test the Python version (singly serialized)
        let json = r#"
        {
            "type": "chat",
            "function_name": "test_function",
            "variant_name": "test_variant",
            "input": {"system": "you are a helpful assistant", "messages": []},
            "output": [{"type": "text", "text": "Hello! How can I help you today?"}],
            "episode_id": "123e4567-e89b-12d3-a456-426614174000",
            "inference_id": "123e4567-e89b-12d3-a456-426614174000",
            "tags": {},
            "tool_params": {"tools_available": [], "tool_choice": "none", "parallel_tool_calls": false},
            "timestamp": "2023-01-01T00:00:00Z"
        }
    "#;
        let inference: StoredInferenceDatabase = serde_json::from_str(json).unwrap();
        let StoredInferenceDatabase::Chat(chat_inference) = inference else {
            panic!("Expected a chat inference");
        };
        assert_eq!(chat_inference.function_name, "test_function");
        assert_eq!(chat_inference.variant_name, "test_variant");
        assert_eq!(
            chat_inference.input,
            StoredInput {
                system: Some(System::Text("you are a helpful assistant".to_string())),
                messages: vec![],
            }
        );
        assert_eq!(
            chat_inference.output,
            vec!["Hello! How can I help you today?".to_string().into()]
        );
        assert_eq!(
            chat_inference.tool_params,
            ToolCallConfigDatabaseInsert::new_for_test(
                vec![],
                vec![],
                AllowedTools {
                    tools: vec![],
                    choice: AllowedToolsChoice::FunctionDefault,
                },
                ToolChoice::None,
                Some(false),
            )
        );
        assert!(chat_inference.dispreferred_outputs.is_empty());
    }

    #[test]
    fn test_stored_inference_deserialization_chat_with_dispreferred_outputs() {
        // Test the ClickHouse version (doubly serialized)
        let json = r#"
            {
                "type": "chat",
                "function_name": "test_function",
                "variant_name": "test_variant",
                "input": "{\"system\": \"you are a helpful assistant\", \"messages\": []}",
                "output": "[{\"type\": \"text\", \"text\": \"Hello! How can I help you today?\"}]",
                "episode_id": "123e4567-e89b-12d3-a456-426614174000",
                "inference_id": "123e4567-e89b-12d3-a456-426614174000",
                "tags": {},
                "tool_params": "",
                "dispreferred_outputs": ["[{\"type\": \"text\", \"text\": \"Goodbye!\"}]"],
                "timestamp": "2023-01-01T00:00:00Z"
            }
        "#;
        let inference: ClickHouseStoredInferenceWithDispreferredOutputs =
            serde_json::from_str(json).unwrap();
        let StoredInferenceDatabase::Chat(chat_inference) = inference.try_into().unwrap() else {
            panic!("Expected a chat inference");
        };
        assert_eq!(
            chat_inference.tool_params,
            ToolCallConfigDatabaseInsert::default()
        );
        assert_eq!(
            chat_inference.dispreferred_outputs,
            vec![vec![ContentBlockChatOutput::Text(Text {
                text: "Goodbye!".to_string(),
            })]]
        );

        // Test the Python version (singly serialized)
        let json = r#"
        {
            "type": "chat",
            "function_name": "test_function",
            "variant_name": "test_variant",
            "input": {"system": "you are a helpful assistant", "messages": []},
            "output": [{"type": "text", "text": "Hello! How can I help you today?"}],
            "episode_id": "123e4567-e89b-12d3-a456-426614174000",
            "inference_id": "123e4567-e89b-12d3-a456-426614174000",
            "tags": {},
            "dispreferred_outputs": [
                [{"type": "text", "text": "Goodbye!"}]
            ],
            "timestamp": "2023-01-01T00:00:00Z"
        }
    "#;
        let inference: StoredInferenceDatabase = serde_json::from_str(json).unwrap();
        let StoredInferenceDatabase::Chat(chat_inference) = inference else {
            panic!("Expected a chat inference");
        };
        assert_eq!(
            chat_inference.dispreferred_outputs,
            vec![vec![ContentBlockChatOutput::Text(Text {
                text: "Goodbye!".to_string(),
            })]]
        );
    }

    #[test]
    fn test_stored_inference_deserialization_json() {
        // Test the ClickHouse version (doubly serialized)
        let json = r#"
            {
                "type": "json",
                "function_name": "test_function",
                "variant_name": "test_variant",
                "input": "{\"system\": \"you are a helpful assistant\", \"messages\": []}",
                "output": "{\"raw\":\"{\\\"answer\\\":\\\"Goodbye\\\"}\",\"parsed\":{\"answer\":\"Goodbye\"}}",
                "episode_id": "123e4567-e89b-12d3-a456-426614174000",
                "inference_id": "123e4567-e89b-12d3-a456-426614174000",
                "tags": {},
                "output_schema": "{\"type\": \"object\", \"properties\": {\"output\": {\"type\": \"string\"}}}",
                "timestamp": "2023-01-01T00:00:00Z"
            }
        "#;
        let inference: ClickHouseStoredInferenceWithDispreferredOutputs =
            serde_json::from_str(json).unwrap();
        let StoredInferenceDatabase::Json(json_inference) = inference.try_into().unwrap() else {
            panic!("Expected a json inference");
        };
        assert_eq!(json_inference.function_name, "test_function");
        assert_eq!(json_inference.variant_name, "test_variant");
        assert_eq!(
            json_inference.input,
            StoredInput {
                system: Some(System::Text("you are a helpful assistant".to_string())),
                messages: vec![],
            }
        );
        assert_eq!(
            json_inference.output,
            JsonInferenceOutput {
                raw: Some("{\"answer\":\"Goodbye\"}".to_string()),
                parsed: Some(json!({"answer":"Goodbye"})),
            }
        );
        assert_eq!(
            json_inference.episode_id,
            Uuid::parse_str("123e4567-e89b-12d3-a456-426614174000").unwrap()
        );
        assert_eq!(
            json_inference.inference_id,
            Uuid::parse_str("123e4567-e89b-12d3-a456-426614174000").unwrap()
        );
        assert_eq!(
            json_inference.output_schema,
            json!({"type": "object", "properties": {"output": {"type": "string"}}})
        );
        assert!(json_inference.dispreferred_outputs.is_empty());

        // Test the Python version (singly serialized)
        let json = r#"
         {
             "type": "json",
             "function_name": "test_function",
             "variant_name": "test_variant",
             "input": {"system": "you are a helpful assistant", "messages": []},
             "output": {"raw":"{\"answer\":\"Goodbye\"}","parsed":{"answer":"Goodbye"}},
             "episode_id": "123e4567-e89b-12d3-a456-426614174000",
             "inference_id": "123e4567-e89b-12d3-a456-426614174000",
             "tags": {},
             "output_schema": {"type": "object", "properties": {"output": {"type": "string"}}},
             "timestamp": "2023-01-01T00:00:00Z"
         }
     "#;
        let inference: StoredInferenceDatabase = serde_json::from_str(json).unwrap();
        let StoredInferenceDatabase::Json(json_inference) = inference else {
            panic!("Expected a json inference");
        };
        assert_eq!(json_inference.function_name, "test_function");
        assert_eq!(json_inference.variant_name, "test_variant");
        assert_eq!(
            json_inference.input,
            StoredInput {
                system: Some(System::Text("you are a helpful assistant".to_string())),
                messages: vec![],
            }
        );
        assert_eq!(
            json_inference.output,
            JsonInferenceOutput {
                raw: Some("{\"answer\":\"Goodbye\"}".to_string()),
                parsed: Some(json!({"answer":"Goodbye"})),
            }
        );
        assert_eq!(
            json_inference.episode_id,
            Uuid::parse_str("123e4567-e89b-12d3-a456-426614174000").unwrap()
        );
        assert_eq!(
            json_inference.inference_id,
            Uuid::parse_str("123e4567-e89b-12d3-a456-426614174000").unwrap()
        );
        assert_eq!(
            json_inference.output_schema,
            json!({"type": "object", "properties": {"output": {"type": "string"}}})
        );
        assert!(json_inference.dispreferred_outputs.is_empty());
    }

    #[test]
    fn test_stored_inference_deserialization_json_with_dispreferred_outputs() {
        // Test the ClickHouse version (doubly serialized)
        let json = r#"
            {
                "type": "json",
                "function_name": "test_function",
                "variant_name": "test_variant",
                "input": "{\"system\": \"you are a helpful assistant\", \"messages\": []}",
                "dispreferred_outputs": ["{\"raw\":\"{\\\"answer\\\":\\\"Goodbye\\\"}\",\"parsed\":{\"answer\":\"Goodbye\"}}"],
                "output": "{\"raw\":\"{\\\"answer\\\":\\\"Goodbye\\\"}\",\"parsed\":{\"answer\":\"Goodbye\"}}",
                "episode_id": "123e4567-e89b-12d3-a456-426614174000",
                "inference_id": "123e4567-e89b-12d3-a456-426614174000",
                "tags": {},
                "output_schema": "{\"type\": \"object\", \"properties\": {\"output\": {\"type\": \"string\"}}}",
                "timestamp": "2023-01-01T00:00:00Z"
            }
        "#;
        let inference: ClickHouseStoredInferenceWithDispreferredOutputs =
            serde_json::from_str(json).unwrap();
        let StoredInferenceDatabase::Json(json_inference) = inference.try_into().unwrap() else {
            panic!("Expected a json inference");
        };
        assert_eq!(
            json_inference.dispreferred_outputs,
            vec![JsonInferenceOutput {
                raw: Some("{\"answer\":\"Goodbye\"}".to_string()),
                parsed: Some(json!({"answer":"Goodbye"})),
            }]
        );

        // Test the Python version (singly serialized)
        let json = r#"
         {
             "type": "json",
             "function_name": "test_function",
             "variant_name": "test_variant",
             "dispreferred_outputs": [
                {"raw":"{\"answer\":\"Goodbye\"}","parsed":{"answer":"Goodbye"}}
             ],
             "input": {"system": "you are a helpful assistant", "messages": []},
             "output": {"raw":"{\"answer\":\"Goodbye\"}","parsed":{"answer":"Goodbye"}},
             "episode_id": "123e4567-e89b-12d3-a456-426614174000",
             "inference_id": "123e4567-e89b-12d3-a456-426614174000",
             "tags": {},
             "output_schema": {"type": "object", "properties": {"output": {"type": "string"}}},
             "timestamp": "2023-01-01T00:00:00Z"
         }
     "#;
        let inference: StoredInferenceDatabase = serde_json::from_str(json).unwrap();
        let StoredInferenceDatabase::Json(json_inference) = inference else {
            panic!("Expected a json inference");
        };

        assert_eq!(
            json_inference.dispreferred_outputs,
            vec![JsonInferenceOutput {
                raw: Some("{\"answer\":\"Goodbye\"}".to_string()),
                parsed: Some(json!({"answer":"Goodbye"})),
            }]
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_order_by_timestamp() {
        let config = get_e2e_config().await;
        let order_by = vec![OrderBy {
            term: OrderByTerm::Timestamp,
            direction: OrderDirection::Desc,
        }];
        let opts = ListInferencesParams {
            function_name: Some("extract_entities"),
            order_by: Some(&order_by),
            ..Default::default()
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();

        let expected_sql = r"
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
    i.function_name = {p0:String}
ORDER BY i.timestamp DESC NULLS LAST
LIMIT {p1:UInt64}
OFFSET {p2:UInt64}
FORMAT JSONEachRow";
        assert_query_equals(&sql, expected_sql);

        let expected_params = vec![
            QueryParameter {
                name: "p0".to_string(),
                value: "extract_entities".to_string(),
            },
            QueryParameter {
                name: "p1".to_string(),
                value: "20".to_string(),
            },
            QueryParameter {
                name: "p2".to_string(),
                value: "0".to_string(),
            },
        ];
        assert_eq!(params, expected_params);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_order_by_metric() {
        let config = get_e2e_config().await;
        let order_by = vec![OrderBy {
            term: OrderByTerm::Metric {
                name: "jaccard_similarity".to_string(),
            },
            direction: OrderDirection::Asc,
        }];
        let opts = ListInferencesParams {
            function_name: Some("extract_entities"),
            order_by: Some(&order_by),
            ..Default::default()
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
        // NOTE: This test case enforces that the joins account for metrics that are only used in the order by clause.
        let expected_sql = r"
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
ORDER BY j0.value ASC NULLS LAST
LIMIT {p2:UInt64}
OFFSET {p3:UInt64}
FORMAT JSONEachRow";
        assert_query_equals(&sql, expected_sql);

        let expected_params = vec![
            QueryParameter {
                name: "p0".to_string(),
                value: "extract_entities".to_string(),
            },
            QueryParameter {
                name: "p1".to_string(),
                value: "jaccard_similarity".to_string(),
            },
            QueryParameter {
                name: "p2".to_string(),
                value: "20".to_string(),
            },
            QueryParameter {
                name: "p3".to_string(),
                value: "0".to_string(),
            },
        ];
        assert_eq!(params, expected_params);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_multiple_order_by() {
        let config = get_e2e_config().await;
        let order_by = vec![
            OrderBy {
                term: OrderByTerm::Metric {
                    name: "jaccard_similarity".to_string(),
                },
                direction: OrderDirection::Desc,
            },
            OrderBy {
                term: OrderByTerm::Timestamp,
                direction: OrderDirection::Asc,
            },
        ];
        let opts = ListInferencesParams {
            function_name: Some("extract_entities"),
            order_by: Some(&order_by),
            ..Default::default()
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();

        let expected_sql = r"
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
ORDER BY j0.value DESC NULLS LAST, i.timestamp ASC NULLS LAST
LIMIT {p2:UInt64}
OFFSET {p3:UInt64}
FORMAT JSONEachRow";
        assert_query_equals(&sql, expected_sql);

        let expected_params = vec![
            QueryParameter {
                name: "p0".to_string(),
                value: "extract_entities".to_string(),
            },
            QueryParameter {
                name: "p1".to_string(),
                value: "jaccard_similarity".to_string(),
            },
            QueryParameter {
                name: "p2".to_string(),
                value: "20".to_string(),
            },
            QueryParameter {
                name: "p3".to_string(),
                value: "0".to_string(),
            },
        ];
        assert_eq!(params, expected_params);
    }

    #[tokio::test]
    async fn test_order_by_search_relevance() {
        let config = get_e2e_config().await;
        let order_by = vec![OrderBy {
            term: OrderByTerm::SearchRelevance,
            direction: OrderDirection::Desc,
        }];
        let opts = ListInferencesParams {
            function_name: Some("write_haiku"),
            order_by: Some(&order_by),
            search_query_experimental: Some("test query"),
            ..Default::default()
        };

        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();

        let expected_sql = r"
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
    i.output as output,
    countSubstringsCaseInsensitiveUTF8(i.input, {p1:String}) as input_term_frequency,
    countSubstringsCaseInsensitiveUTF8(i.output, {p1:String}) as output_term_frequency,
    input_term_frequency + output_term_frequency as total_term_frequency
FROM ChatInference AS i
WHERE
    i.function_name = {p0:String}
    AND total_term_frequency > 0
ORDER BY total_term_frequency DESC NULLS LAST
LIMIT {p2:UInt64}
OFFSET {p3:UInt64}
FORMAT JSONEachRow";
        assert_query_equals(&sql, expected_sql);

        assert!(params.contains(&QueryParameter {
            name: "p1".to_string(),
            value: "test query".to_string(),
        }));
    }

    #[tokio::test]
    async fn test_order_by_search_relevance_filters_both_tables() {
        let config = get_e2e_config().await;
        let order_by = vec![OrderBy {
            term: OrderByTerm::SearchRelevance,
            direction: OrderDirection::Desc,
        }];
        let opts = ListInferencesParams {
            order_by: Some(&order_by),
            search_query_experimental: Some("test query"),
            ..Default::default()
        };

        let (sql, _) = generate_list_inferences_sql(&config, &opts).unwrap();

        // SQL should order by total_term_frequency DESC for both tables
        assert_query_contains(&sql, "FROM ChatInference AS i WHERE total_term_frequency > 0 ORDER BY total_term_frequency DESC");
        assert_query_contains(&sql, "UNION ALL");
        assert_query_contains(&sql, "FROM JsonInference AS i WHERE total_term_frequency > 0 ORDER BY total_term_frequency DESC");
        // Should also order by total_term_frequency DESC for the combined result
        assert_query_contains(&sql, "AS combined ORDER BY total_term_frequency DESC");
    }

    #[tokio::test]
    async fn test_order_by_search_relevance_without_query_returns_error() {
        let config = get_e2e_config().await;
        let order_by = vec![OrderBy {
            term: OrderByTerm::SearchRelevance,
            direction: OrderDirection::Desc,
        }];
        let opts = ListInferencesParams {
            function_name: Some("write_haiku"),
            order_by: Some(&order_by),
            search_query_experimental: None,
            ..Default::default()
        };

        let result = generate_list_inferences_sql(&config, &opts);
        assert!(result.is_err());
    }
}
