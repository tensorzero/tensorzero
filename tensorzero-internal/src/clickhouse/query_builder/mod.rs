use crate::config_parser::Config;

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
    // TODO: add ORDER BY options
    limit: Option<i64>,
    offset: Option<i64>,
) -> (String, Vec<QueryParameter>) {
    let mut params_map: Vec<QueryParameter> = Vec::new();
    let mut param_idx_counter = 0; // Counter for unique parameter names

    let mut select_clauses = vec![
        "i.input".to_string(),
        "i.output AS \"inference_output\"".to_string(), // Quoted alias
        "i.variant_name".to_string(),
        "i.episode_id".to_string(),
    ];
    let mut join_clauses: Vec<String> = Vec::new();
    let mut where_clauses: Vec<String> = Vec::new();

    // Add function_name filter
    let fn_param_placeholder = add_parameter(
        function_name.to_string(),
        "String",
        &mut params_map,
        &mut param_idx_counter,
    );
    where_clauses.push(format!("i.function_name = {}", fn_param_placeholder));

    // Add variant_name filter
    if let Some(variant_name) = variant_name {
        let variant_name_param_placeholder = add_parameter(
            variant_name.to_string(),
            "String",
            &mut params_map,
            &mut param_idx_counter,
        );
        where_clauses.push(format!(
            "i.variant_name = {}",
            variant_name_param_placeholder
        ));
    }

    // Handle OutputSource
    match output_source {
        InferenceOutputSource::Inference => { /* inference_output already selected */ }
        InferenceOutputSource::Demonstration => {
            select_clauses.push("demo_f.value AS \"demonstration_output\"".to_string()); // Quoted alias

            // TODO(viraj, urgently): use GROUP BY here instead of window function
            join_clauses.push(format!(
                "JOIN \
                 (SELECT \
                    inference_id, \
                    value, \
                    ROW_NUMBER() OVER (PARTITION BY inference_id ORDER BY timestamp DESC) as rn \
                  FROM \"{}\" \
                 ) AS demo_f ON i.\"{}\" = demo_f.inference_id AND demo_f.rn = 1", // Quoted table/column names from config
                config.metrics.table_name("demonstration"),
                config.metrics.inference_table_column_name("demonstration")
            ));
        }
    }

    // NOTE: you can get the table name for the inferences we want to query from config.get_function(function_name).table_name()
    // Handle Filter Tree
    if let Some(filter_node) = filters {
        // @Gemini, I got to here and everything above here in the file is sensible but the problems start below.
        // TODO (Viraj, urgently): keep reading from here
        let feedback_sql_condition = build_feedback_filter_sql_recursive(
            filter_node,
            &mut params_map,
            &mut param_idx_counter,
        );
        if !feedback_sql_condition.is_empty() {
            join_clauses.push(format!(
                "JOIN \
                 (SELECT \
                    target_id, \
                    value AS metric_value, \
                    metric_name, \
                    ROW_NUMBER() OVER (PARTITION BY target_id ORDER BY timestamp DESC) as rn \
                  FROM \"{}\" \
                  WHERE {} \
                 ) AS metric_f ON i.\"{}\" = metric_f.target_id AND metric_f.rn = 1", // Quoted table/column names
                config.feedback_table_name,
                feedback_sql_condition,
                config.inference_join_key_for_feedback
            ));
            select_clauses.push("metric_f.metric_value AS \"filtered_metric_value\"".to_string()); // Quoted alias
            select_clauses.push("metric_f.metric_name AS \"filtered_metric_name\"".to_string());
            // Quoted alias
        }
    }

    let mut sql = format!(
        "SELECT\n    {}\nFROM\n    \"{}\" AS i", // Quoted table name
        select_clauses.join(",\n    "),
        config.inference_table_name
    );

    if !join_clauses.is_empty() {
        sql.push_str("\n");
        sql.push_str(&join_clauses.join("\n"));
    }

    if !where_clauses.is_empty() {
        sql.push_str("\nWHERE\n    ");
        sql.push_str(&where_clauses.join(" AND "));
    }

    if let Some(l) = limit {
        let limit_param_placeholder = add_parameter(
            l.to_string(),
            "UInt64",
            &mut params_map,
            &mut param_idx_counter,
        );
        sql.push_str(&format!("\nLIMIT {}", limit_param_placeholder));
    }

    if let Some(o) = offset {
        let offset_param_placeholder = add_parameter(
            o.to_string(),
            "UInt64",
            &mut params_map,
            &mut param_idx_counter,
        );
        sql.push_str(&format!("\nOFFSET {}", offset_param_placeholder));
    }

    (sql, params_map)
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
    format!("{{{}:{}}}", internal_name, ch_type)
}

/// Recursively builds the SQL condition for the WHERE clause of the feedback subquery.
fn build_feedback_filter_sql_recursive(
    node: &InferenceFilterTreeNode,
    params_map: &mut Vec<QueryParameter>,
    param_idx_counter: &mut usize,
) -> String {
    match node {
        InferenceFilterTreeNode::FloatMetric(fm_node) => {
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
            format!(
                "(metric_name = {} AND value {} {})",
                metric_name_placeholder,
                fm_node.comparison_operator.to_sql(),
                value_placeholder
            )
        }
        InferenceFilterTreeNode::BooleanMetric(bm_node) => {
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
            format!(
                "(metric_name = {} AND value = {})",
                metric_name_placeholder, value_placeholder
            )
        }
        InferenceFilterTreeNode::And(children) => {
            let child_sqls: Vec<String> = children
                .iter()
                .map(|child| {
                    build_feedback_filter_sql_recursive(child, params_map, param_idx_counter)
                })
                .filter(|s| !s.is_empty())
                .collect();
            if child_sqls.is_empty() {
                "".to_string()
            } else {
                format!("({})", child_sqls.join(" AND "))
            }
        }
        InferenceFilterTreeNode::Or(children) => {
            let child_sqls: Vec<String> = children
                .iter()
                .map(|child| {
                    build_feedback_filter_sql_recursive(child, params_map, param_idx_counter)
                })
                .filter(|s| !s.is_empty())
                .collect();
            if child_sqls.is_empty() {
                "".to_string()
            } else {
                format!("({})", child_sqls.join(" OR "))
            }
        }
        InferenceFilterTreeNode::Not(child) => {
            let child_sql =
                build_feedback_filter_sql_recursive(child.as_ref(), params_map, param_idx_counter);
            if child_sql.is_empty() {
                "".to_string()
            } else {
                format!("NOT ({})", child_sql)
            }
        }
    }
}
