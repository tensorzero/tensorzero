use itertools::Itertools;
use std::{
    collections::BTreeSet,
    fmt::{self, Display},
};

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
        _select_clauses: &mut BTreeSet<String>,
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
                    &fm_node.metric_name,
                    ClickhouseType::String,
                    params_map,
                    param_idx_counter,
                );
                let value_placeholder = add_parameter(
                    fm_node.value,
                    ClickhouseType::Float64,
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
                    &bm_node.metric_name,
                    ClickhouseType::String,
                    params_map,
                    param_idx_counter,
                );
                let bool_value_str = if bm_node.value { "1" } else { "0" };
                let value_placeholder = add_parameter(
                    bool_value_str,
                    ClickhouseType::Bool,
                    params_map,
                    param_idx_counter,
                );
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

#[derive(Debug, Clone)]
pub struct ListInferencesParams<'a> {
    pub function_name: &'a str,
    pub variant_name: Option<&'a str>,
    pub filters: Option<&'a InferenceFilterTreeNode>,
    pub output_source: InferenceOutputSource,
    pub limit: Option<i64>,
    pub offset: Option<i64>,
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
/// - handle things like output schema, tool params, etc.
pub fn generate_list_inferences_sql(
    config: &Config,
    opts: &ListInferencesParams<'_>,
) -> Result<(String, Vec<QueryParameter>), Error> {
    let mut params_map: Vec<QueryParameter> = Vec::new();
    let mut param_idx_counter = 0; // Counter for unique parameter names
    let mut join_idx_counter = 0; // Counter for unique join aliases

    let mut select_clauses = BTreeSet::from([
        "i.input as input".to_string(),
        "i.variant_name as variant_name".to_string(),
        "i.episode_id as episode_id".to_string(),
        "i.id as id".to_string(),
        "i.timestamp as timestamp".to_string(),
        // We don't select output here because it's handled separately based on the output_source
    ]);
    let mut join_clauses: Vec<String> = Vec::new();
    let mut where_clauses: Vec<String> = Vec::new();

    let function_config = config.get_function(opts.function_name)?;
    let inference_table_name = function_config.table_name();

    let function_name_param_placeholder = add_parameter(
        opts.function_name,
        ClickhouseType::String,
        &mut params_map,
        &mut param_idx_counter,
    );
    where_clauses.push(format!(
        "i.function_name = {function_name_param_placeholder}"
    ));

    // Add variant_name filter
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

            // NOTE: we may want to pre-filter this via subqueries or CTEs prior to the join for performance reasons
            join_clauses.push(
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
    {inference_table_name} AS i"#,
        select_clauses = select_clauses.iter().join(",\n    "),
        inference_table_name = inference_table_name,
    );

    if !join_clauses.is_empty() {
        sql.push_str(&join_clauses.join("\n"));
    }

    if !where_clauses.is_empty() {
        sql.push_str("\nWHERE\n    ");
        sql.push_str(&where_clauses.join(" AND "));
    }
    // TODO: add ORDER BY

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

    Ok((sql, params_map))
}

#[derive(Debug, Clone, PartialEq)]
enum ClickhouseType {
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

/// Helper to add a parameter and return its SQL placeholder {name:CHType}
/// The internal_name (e.g. p0, p1) is stored in params_map with its value.
fn add_parameter<T: ToString>(
    value: T,
    ch_type: ClickhouseType,
    params_map: &mut Vec<QueryParameter>,
    counter: &mut usize,
) -> String {
    let internal_name = format!("p{}", *counter);
    *counter += 1;
    params_map.push(QueryParameter {
        name: internal_name.clone(),
        value: value.to_string(),
    });
    format!("{{{internal_name}:{ch_type}}}")
}

/// Helper to get a join alias given the current counter on join tables.
fn get_join_alias(counter: &mut usize) -> String {
    let internal_name = format!("j{}", *counter);
    *counter += 1;
    internal_name
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    async fn get_e2e_config() -> Config<'static> {
        // Read the e2e config file
        Config::load_from_path_optional_verify_credentials(
            Path::new("tests/e2e/tensorzero.toml"),
            false,
        )
        .await
        .unwrap()
    }

    /// Tests the simplest possible query: list inferences for a function with no filters
    #[tokio::test(flavor = "multi_thread")]
    async fn test_simple_query_json_function() {
        let config = get_e2e_config().await;
        let opts = ListInferencesParams {
            function_name: "extract_entities",
            variant_name: None,
            filters: None,
            output_source: InferenceOutputSource::Inference,
            limit: None,
            offset: None,
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
        let expected_sql = r#"
SELECT
    i.episode_id as episode_id,
    i.id as id,
    i.input as input,
    i.output as output,
    i.timestamp as timestamp,
    i.variant_name as variant_name
FROM
    JsonInference AS i
WHERE
    i.function_name = {p0:String}"#;
        assert_eq!(sql, expected_sql);
        let expected_params = vec![QueryParameter {
            name: "p0".to_string(),
            value: "extract_entities".to_string(),
        }];
        assert_eq!(params, expected_params);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_simple_query_chat_function() {
        let config = get_e2e_config().await;
        let opts = ListInferencesParams {
            function_name: "write_haiku",
            variant_name: None,
            filters: None,
            output_source: InferenceOutputSource::Inference,
            limit: None,
            offset: None,
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
        let expected_sql = r#"
SELECT
    i.episode_id as episode_id,
    i.id as id,
    i.input as input,
    i.output as output,
    i.timestamp as timestamp,
    i.variant_name as variant_name
FROM
    ChatInference AS i
WHERE
    i.function_name = {p0:String}"#;
        assert_eq!(sql, expected_sql);
        let expected_params = vec![QueryParameter {
            name: "p0".to_string(),
            value: "write_haiku".to_string(),
        }];
        assert_eq!(params, expected_params);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_simple_query_with_float_filters() {
        let config = get_e2e_config().await;
        let filter_node = InferenceFilterTreeNode::FloatMetric(FloatMetricNode {
            metric_name: "jaccard_similarity".to_string(),
            value: 0.5,
            comparison_operator: FloatComparisonOperator::GreaterThan,
        });
        let opts = ListInferencesParams {
            function_name: "extract_entities",
            variant_name: None,
            filters: Some(&filter_node),
            output_source: InferenceOutputSource::Inference,
            limit: None,
            offset: None,
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
        let expected_sql = r#"
SELECT
    i.episode_id as episode_id,
    i.id as id,
    i.input as input,
    i.output as output,
    i.timestamp as timestamp,
    i.variant_name as variant_name
FROM
    JsonInference AS i
LEFT JOIN (
    SELECT
        target_id,
        argMax(value, timestamp) as value
    FROM FloatMetricFeedback
    WHERE metric_name = {p1:String}
    AND value > {p2:Float64}
    GROUP BY target_id
) AS j0 ON i.id = j0.target_id
WHERE
    i.function_name = {p0:String} AND j0.value > {p2:Float64}"#;
        assert_eq!(sql, expected_sql);
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
        ];
        assert_eq!(params, expected_params);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_unknown_function_name() {
        let config = get_e2e_config().await;
        let opts = ListInferencesParams {
            function_name: "unknown_function_name",
            variant_name: None,
            filters: None,
            output_source: InferenceOutputSource::Inference,
            limit: None,
            offset: None,
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
        let filter_node = InferenceFilterTreeNode::FloatMetric(FloatMetricNode {
            metric_name: "unknown_metric_name".to_string(),
            value: 0.5,
            comparison_operator: FloatComparisonOperator::GreaterThan,
        });
        let opts = ListInferencesParams {
            function_name: "extract_entities",
            variant_name: None,
            filters: Some(&filter_node),
            output_source: InferenceOutputSource::Inference,
            limit: None,
            offset: None,
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
            function_name: "extract_entities",
            variant_name: None,
            filters: None,
            output_source: InferenceOutputSource::Demonstration,
            limit: None,
            offset: None,
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
        let expected_sql = r#"
SELECT
    demo_f.value AS output,
    i.episode_id as episode_id,
    i.id as id,
    i.input as input,
    i.timestamp as timestamp,
    i.variant_name as variant_name
FROM
    JsonInference AS i
JOIN (SELECT inference_id, argMax(value, timestamp) as value FROM DemonstrationFeedback GROUP BY inference_id ) AS demo_f ON i.id = demo_f.inference_id
WHERE
    i.function_name = {p0:String}"#;
        assert_eq!(sql, expected_sql);
        let expected_params = vec![QueryParameter {
            name: "p0".to_string(),
            value: "extract_entities".to_string(),
        }];
        assert_eq!(params, expected_params);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_boolean_metric_filter() {
        let config = get_e2e_config().await;
        let filter_node = InferenceFilterTreeNode::BooleanMetric(BooleanMetricNode {
            metric_name: "task_success".to_string(),
            value: true,
        });
        let opts = ListInferencesParams {
            function_name: "extract_entities",
            variant_name: None,
            filters: Some(&filter_node),
            output_source: InferenceOutputSource::Inference,
            limit: None,
            offset: None,
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
        let expected_sql = r#"
SELECT
    i.episode_id as episode_id,
    i.id as id,
    i.input as input,
    i.output as output,
    i.timestamp as timestamp,
    i.variant_name as variant_name
FROM
    JsonInference AS i
LEFT JOIN (
    SELECT
        target_id,
        argMax(value, timestamp) as value
    FROM BooleanMetricFeedback
    WHERE metric_name = {p1:String}
    AND value = {p2:Bool}
    GROUP BY target_id
) AS j0 ON i.id = j0.target_id
WHERE
    i.function_name = {p0:String} AND j0.value = {p2:Bool}"#;
        assert_eq!(sql, expected_sql);
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
        ];
        assert_eq!(params, expected_params);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_boolean_metric_filter_false() {
        let config = get_e2e_config().await;
        let filter_node = InferenceFilterTreeNode::BooleanMetric(BooleanMetricNode {
            metric_name: "task_success".to_string(),
            value: false,
        });
        let opts = ListInferencesParams {
            function_name: "extract_entities",
            variant_name: None,
            filters: Some(&filter_node),
            output_source: InferenceOutputSource::Inference,
            limit: None,
            offset: None,
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
        let expected_sql = r#"
SELECT
    i.episode_id as episode_id,
    i.id as id,
    i.input as input,
    i.output as output,
    i.timestamp as timestamp,
    i.variant_name as variant_name
FROM
    JsonInference AS i
LEFT JOIN (
    SELECT
        target_id,
        argMax(value, timestamp) as value
    FROM BooleanMetricFeedback
    WHERE metric_name = {p1:String}
    AND value = {p2:Bool}
    GROUP BY target_id
) AS j0 ON i.id = j0.target_id
WHERE
    i.function_name = {p0:String} AND j0.value = {p2:Bool}"#;
        assert_eq!(sql, expected_sql);
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
        ];
        assert_eq!(params, expected_params);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_and_filter_multiple_float_metrics() {
        let config = get_e2e_config().await;
        let filter_node = InferenceFilterTreeNode::And(vec![
            InferenceFilterTreeNode::FloatMetric(FloatMetricNode {
                metric_name: "jaccard_similarity".to_string(),
                value: 0.5,
                comparison_operator: FloatComparisonOperator::GreaterThan,
            }),
            InferenceFilterTreeNode::FloatMetric(FloatMetricNode {
                metric_name: "brevity_score".to_string(),
                value: 10.0,
                comparison_operator: FloatComparisonOperator::LessThan,
            }),
        ]);
        let opts = ListInferencesParams {
            function_name: "extract_entities",
            variant_name: None,
            filters: Some(&filter_node),
            output_source: InferenceOutputSource::Inference,
            limit: None,
            offset: None,
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
        let expected_sql = r#"
SELECT
    i.episode_id as episode_id,
    i.id as id,
    i.input as input,
    i.output as output,
    i.timestamp as timestamp,
    i.variant_name as variant_name
FROM
    JsonInference AS i
LEFT JOIN (
    SELECT
        target_id,
        argMax(value, timestamp) as value
    FROM FloatMetricFeedback
    WHERE metric_name = {p1:String}
    AND value > {p2:Float64}
    GROUP BY target_id
) AS j0 ON i.id = j0.target_id

LEFT JOIN (
    SELECT
        target_id,
        argMax(value, timestamp) as value
    FROM FloatMetricFeedback
    WHERE metric_name = {p3:String}
    AND value < {p4:Float64}
    GROUP BY target_id
) AS j1 ON i.id = j1.target_id
WHERE
    i.function_name = {p0:String} AND (COALESCE(j0.value > {p2:Float64}, 0) AND COALESCE(j1.value < {p4:Float64}, 0))"#;
        assert_eq!(sql, expected_sql);
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
                value: "brevity_score".to_string(),
            },
            QueryParameter {
                name: "p4".to_string(),
                value: "10".to_string(),
            },
        ];
        assert_eq!(params, expected_params);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_or_filter_mixed_metrics() {
        let config = get_e2e_config().await;
        let filter_node = InferenceFilterTreeNode::Or(vec![
            InferenceFilterTreeNode::FloatMetric(FloatMetricNode {
                metric_name: "jaccard_similarity".to_string(),
                value: 0.8,
                comparison_operator: FloatComparisonOperator::GreaterThanOrEqual,
            }),
            InferenceFilterTreeNode::BooleanMetric(BooleanMetricNode {
                metric_name: "exact_match".to_string(),
                value: true,
            }),
        ]);
        let opts = ListInferencesParams {
            function_name: "extract_entities",
            variant_name: None,
            filters: Some(&filter_node),
            output_source: InferenceOutputSource::Inference,
            limit: None,
            offset: None,
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
        let expected_sql = r#"
SELECT
    i.episode_id as episode_id,
    i.id as id,
    i.input as input,
    i.output as output,
    i.timestamp as timestamp,
    i.variant_name as variant_name
FROM
    JsonInference AS i
LEFT JOIN (
    SELECT
        target_id,
        argMax(value, timestamp) as value
    FROM FloatMetricFeedback
    WHERE metric_name = {p1:String}
    AND value >= {p2:Float64}
    GROUP BY target_id
) AS j0 ON i.id = j0.target_id

LEFT JOIN (
    SELECT
        target_id,
        argMax(value, timestamp) as value
    FROM BooleanMetricFeedback
    WHERE metric_name = {p3:String}
    AND value = {p4:Bool}
    GROUP BY target_id
) AS j1 ON i.id = j1.target_id
WHERE
    i.function_name = {p0:String} AND (COALESCE(j0.value >= {p2:Float64}, 0) OR COALESCE(j1.value = {p4:Bool}, 0))"#;
        assert_eq!(sql, expected_sql);
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
        ];
        assert_eq!(params, expected_params);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_not_filter() {
        let config = get_e2e_config().await;
        let filter_node = InferenceFilterTreeNode::Not(Box::new(
            InferenceFilterTreeNode::BooleanMetric(BooleanMetricNode {
                metric_name: "task_success".to_string(),
                value: true,
            }),
        ));
        let opts = ListInferencesParams {
            function_name: "extract_entities",
            variant_name: None,
            filters: Some(&filter_node),
            output_source: InferenceOutputSource::Inference,
            limit: None,
            offset: None,
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
        let expected_sql = r#"
SELECT
    i.episode_id as episode_id,
    i.id as id,
    i.input as input,
    i.output as output,
    i.timestamp as timestamp,
    i.variant_name as variant_name
FROM
    JsonInference AS i
LEFT JOIN (
    SELECT
        target_id,
        argMax(value, timestamp) as value
    FROM BooleanMetricFeedback
    WHERE metric_name = {p1:String}
    AND value = {p2:Bool}
    GROUP BY target_id
) AS j0 ON i.id = j0.target_id
WHERE
    i.function_name = {p0:String} AND NOT (COALESCE(j0.value = {p2:Bool}, 1))"#;
        assert_eq!(sql, expected_sql);
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
        ];
        assert_eq!(params, expected_params);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_nested_complex_filter() {
        let config = get_e2e_config().await;
        let filter_node = InferenceFilterTreeNode::And(vec![
            InferenceFilterTreeNode::Or(vec![
                InferenceFilterTreeNode::FloatMetric(FloatMetricNode {
                    metric_name: "jaccard_similarity".to_string(),
                    value: 0.7,
                    comparison_operator: FloatComparisonOperator::GreaterThan,
                }),
                InferenceFilterTreeNode::FloatMetric(FloatMetricNode {
                    metric_name: "brevity_score".to_string(),
                    value: 5.0,
                    comparison_operator: FloatComparisonOperator::LessThanOrEqual,
                }),
            ]),
            InferenceFilterTreeNode::Not(Box::new(InferenceFilterTreeNode::BooleanMetric(
                BooleanMetricNode {
                    metric_name: "task_success".to_string(),
                    value: false,
                },
            ))),
        ]);
        let opts = ListInferencesParams {
            function_name: "extract_entities",
            variant_name: None,
            filters: Some(&filter_node),
            output_source: InferenceOutputSource::Inference,
            limit: None,
            offset: None,
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
        let expected_sql = r#"
SELECT
    i.episode_id as episode_id,
    i.id as id,
    i.input as input,
    i.output as output,
    i.timestamp as timestamp,
    i.variant_name as variant_name
FROM
    JsonInference AS i
LEFT JOIN (
    SELECT
        target_id,
        argMax(value, timestamp) as value
    FROM FloatMetricFeedback
    WHERE metric_name = {p1:String}
    AND value > {p2:Float64}
    GROUP BY target_id
) AS j0 ON i.id = j0.target_id

LEFT JOIN (
    SELECT
        target_id,
        argMax(value, timestamp) as value
    FROM FloatMetricFeedback
    WHERE metric_name = {p3:String}
    AND value <= {p4:Float64}
    GROUP BY target_id
) AS j1 ON i.id = j1.target_id

LEFT JOIN (
    SELECT
        target_id,
        argMax(value, timestamp) as value
    FROM BooleanMetricFeedback
    WHERE metric_name = {p5:String}
    AND value = {p6:Bool}
    GROUP BY target_id
) AS j2 ON i.id = j2.target_id
WHERE
    i.function_name = {p0:String} AND (COALESCE((COALESCE(j0.value > {p2:Float64}, 0) OR COALESCE(j1.value <= {p4:Float64}, 0)), 0) AND COALESCE(NOT (COALESCE(j2.value = {p6:Bool}, 1)), 0))"#;
        assert_eq!(sql, expected_sql);
        assert_eq!(params.len(), 7); // p0 (function) + 6 metric-related params
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_variant_name_filter() {
        let config = get_e2e_config().await;
        let opts = ListInferencesParams {
            function_name: "extract_entities",
            variant_name: Some("v1"),
            filters: None,
            output_source: InferenceOutputSource::Inference,
            limit: None,
            offset: None,
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
        let expected_sql = r#"
SELECT
    i.episode_id as episode_id,
    i.id as id,
    i.input as input,
    i.output as output,
    i.timestamp as timestamp,
    i.variant_name as variant_name
FROM
    JsonInference AS i
WHERE
    i.function_name = {p0:String} AND i.variant_name = {p1:String}"#;
        assert_eq!(sql, expected_sql);
        let expected_params = vec![
            QueryParameter {
                name: "p0".to_string(),
                value: "extract_entities".to_string(),
            },
            QueryParameter {
                name: "p1".to_string(),
                value: "v1".to_string(),
            },
        ];
        assert_eq!(params, expected_params);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_limit_and_offset() {
        let config = get_e2e_config().await;
        let opts = ListInferencesParams {
            function_name: "extract_entities",
            variant_name: None,
            filters: None,
            output_source: InferenceOutputSource::Inference,
            limit: Some(50),
            offset: Some(100),
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
        let expected_sql = r#"
SELECT
    i.episode_id as episode_id,
    i.id as id,
    i.input as input,
    i.output as output,
    i.timestamp as timestamp,
    i.variant_name as variant_name
FROM
    JsonInference AS i
WHERE
    i.function_name = {p0:String}
LIMIT {p1:UInt64}
OFFSET {p2:UInt64}"#;
        assert_eq!(sql, expected_sql);
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
            let filter_node = InferenceFilterTreeNode::FloatMetric(FloatMetricNode {
                metric_name: "jaccard_similarity".to_string(),
                value: 0.5,
                comparison_operator: op,
            });
            let opts = ListInferencesParams {
                function_name: "extract_entities",
                variant_name: None,
                filters: Some(&filter_node),
                output_source: InferenceOutputSource::Inference,
                limit: None,
                offset: None,
            };
            let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
            let expected_sql = format!(
                r#"
SELECT
    i.episode_id as episode_id,
    i.id as id,
    i.input as input,
    i.output as output,
    i.timestamp as timestamp,
    i.variant_name as variant_name
FROM
    JsonInference AS i
LEFT JOIN (
    SELECT
        target_id,
        argMax(value, timestamp) as value
    FROM FloatMetricFeedback
    WHERE metric_name = {{p1:String}}
    AND value {expected_op_str} {{p2:Float64}}
    GROUP BY target_id
) AS j0 ON i.id = j0.target_id
WHERE
    i.function_name = {{p0:String}} AND j0.value {expected_op_str} {{p2:Float64}}"#,
            );
            assert_eq!(sql, expected_sql);
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
            ];
            assert_eq!(params, expected_params);
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_combined_variant_filter_and_metrics() {
        let config = get_e2e_config().await;
        let filter_node = InferenceFilterTreeNode::And(vec![
            InferenceFilterTreeNode::FloatMetric(FloatMetricNode {
                metric_name: "jaccard_similarity".to_string(),
                value: 0.6,
                comparison_operator: FloatComparisonOperator::GreaterThan,
            }),
            InferenceFilterTreeNode::BooleanMetric(BooleanMetricNode {
                metric_name: "exact_match".to_string(),
                value: true,
            }),
        ]);
        let opts = ListInferencesParams {
            function_name: "extract_entities",
            variant_name: Some("production"),
            filters: Some(&filter_node),
            output_source: InferenceOutputSource::Demonstration,
            limit: Some(25),
            offset: Some(50),
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
        let expected_sql = r#"
SELECT
    demo_f.value AS output,
    i.episode_id as episode_id,
    i.id as id,
    i.input as input,
    i.timestamp as timestamp,
    i.variant_name as variant_name
FROM
    JsonInference AS i
JOIN (SELECT inference_id, argMax(value, timestamp) as value FROM DemonstrationFeedback GROUP BY inference_id ) AS demo_f ON i.id = demo_f.inference_id

LEFT JOIN (
    SELECT
        target_id,
        argMax(value, timestamp) as value
    FROM FloatMetricFeedback
    WHERE metric_name = {p2:String}
    AND value > {p3:Float64}
    GROUP BY target_id
) AS j0 ON i.id = j0.target_id

LEFT JOIN (
    SELECT
        target_id,
        argMax(value, timestamp) as value
    FROM BooleanMetricFeedback
    WHERE metric_name = {p4:String}
    AND value = {p5:Bool}
    GROUP BY target_id
) AS j1 ON i.id = j1.target_id
WHERE
    i.function_name = {p0:String} AND i.variant_name = {p1:String} AND (COALESCE(j0.value > {p3:Float64}, 0) AND COALESCE(j1.value = {p5:Bool}, 0))
LIMIT {p6:UInt64}
OFFSET {p7:UInt64}"#;
        assert_eq!(sql, expected_sql);

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
}
