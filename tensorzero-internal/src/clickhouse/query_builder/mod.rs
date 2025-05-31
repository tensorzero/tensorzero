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
pub enum FilterTreeNode {
    FloatMetric(FloatMetricNode),
    BooleanMetric(BooleanMetricNode),
    And(Vec<FilterTreeNode>),
    Or(Vec<FilterTreeNode>),
    Not(Box<FilterTreeNode>),
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
    function_name: &str,
    filters: Option<&FilterTreeNode>,
    output_source: &InferenceOutputSource,
    limit: Option<i64>,
    offset: Option<i64>,
    config: &QueryConfig,
) -> (String, Vec<QueryParameter>) {
    let mut params_map: Vec<QueryParameter> = Vec::new();
    let mut param_idx_counter = 0; // Counter for unique parameter names

    // Helper to add a parameter and return its SQL placeholder {name:CHType}
    // The internal_name (e.g. p0, p1) is stored in params_map with its value.
    let mut add_param_fn = |value: String,
                            ch_type: &str,
                            params_map_ref: &mut Vec<QueryParameter>,
                            counter: &mut usize|
     -> String {
        let internal_name = format!("p{}", *counter);
        *counter += 1;
        params_map_ref.push(QueryParameter {
            name: internal_name.clone(),
            value,
        });
        format!("{{{}:{}}}", internal_name, ch_type)
    };

    let mut select_clauses = vec![
        "i.input".to_string(),
        "i.output AS \"inference_output\"".to_string(), // Quoted alias
        "i.variant_name".to_string(),
        "i.episode_id".to_string(),
    ];
    let mut join_clauses: Vec<String> = Vec::new();
    let mut where_clauses: Vec<String> = Vec::new();

    // Add function_name filter
    let fn_param_placeholder = add_param_fn(
        function_name.to_string(),
        "String",
        &mut params_map,
        &mut param_idx_counter,
    );
    where_clauses.push(format!("i.function_name = {}", fn_param_placeholder));

    // Handle OutputSource
    match output_source {
        InferenceOutputSource::Inference => { /* inference_output already selected */ }
        InferenceOutputSource::Demonstration => {
            select_clauses.push("demo_f.value AS \"demonstration_output\"".to_string()); // Quoted alias
            join_clauses.push(format!(
                "JOIN \
                 (SELECT \
                    inference_id, \
                    value, \
                    ROW_NUMBER() OVER (PARTITION BY inference_id ORDER BY timestamp DESC) as rn \
                  FROM \"{}\" \
                 ) AS demo_f ON i.\"{}\" = demo_f.inference_id AND demo_f.rn = 1", // Quoted table/column names from config
                config.demonstration_feedback_table_name,
                config.inference_id_column_in_inference_table
            ));
        }
    }

    // Handle Filters
    if let Some(filter_node) = filters {
        let feedback_sql_condition = build_feedback_filter_sql_recursive(
            filter_node,
            &mut params_map,
            &mut param_idx_counter,
            &mut add_param_fn,
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
        let limit_param_placeholder = add_param_fn(
            l.to_string(),
            "UInt64",
            &mut params_map,
            &mut param_idx_counter,
        );
        sql.push_str(&format!("\nLIMIT {}", limit_param_placeholder));
    }

    if let Some(o) = offset {
        let offset_param_placeholder = add_param_fn(
            o.to_string(),
            "UInt64",
            &mut params_map,
            &mut param_idx_counter,
        );
        sql.push_str(&format!("\nOFFSET {}", offset_param_placeholder));
    }

    (sql, params_map)
}

/// Recursively builds the SQL condition for the WHERE clause of the feedback subquery.
fn build_feedback_filter_sql_recursive(
    node: &FilterTreeNode,
    params_map: &mut Vec<QueryParameter>,
    param_idx_counter: &mut usize,
    add_param_fn: &mut dyn FnMut(String, &str, &mut Vec<QueryParameter>, &mut usize) -> String,
) -> String {
    match &node.variant {
        FilterTreeNodeVariant::FloatMetric(fm_node) => {
            let metric_name_placeholder = add_param_fn(
                fm_node.metric_name.clone(),
                "String",
                params_map,
                param_idx_counter,
            );
            let value_placeholder = add_param_fn(
                fm_node.value.to_string(),
                "Float64",
                params_map,
                param_idx_counter,
            );
            format!(
                "(metric_name = {} AND value {} {})",
                metric_name_placeholder,
                fm_node.operator.to_sql(),
                value_placeholder
            )
        }
        FilterTreeNodeVariant::BooleanMetric(bm_node) => {
            let metric_name_placeholder = add_param_fn(
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
                add_param_fn(bool_value_str, "Bool", params_map, param_idx_counter);
            format!(
                "(metric_name = {} AND value = {})",
                metric_name_placeholder, value_placeholder
            )
        }
        FilterTreeNodeVariant::And(children) => {
            let child_sqls: Vec<String> = children
                .iter()
                .map(|child| {
                    build_feedback_filter_sql_recursive(
                        child,
                        params_map,
                        param_idx_counter,
                        add_param_fn,
                    )
                })
                .filter(|s| !s.is_empty())
                .collect();
            if child_sqls.is_empty() {
                "".to_string()
            } else {
                format!("({})", child_sqls.join(" AND "))
            }
        }
        FilterTreeNodeVariant::Or(children) => {
            let child_sqls: Vec<String> = children
                .iter()
                .map(|child| {
                    build_feedback_filter_sql_recursive(
                        child,
                        params_map,
                        param_idx_counter,
                        add_param_fn,
                    )
                })
                .filter(|s| !s.is_empty())
                .collect();
            if child_sqls.is_empty() {
                "".to_string()
            } else {
                format!("({})", child_sqls.join(" OR "))
            }
        }
        FilterTreeNodeVariant::Not(child) => {
            let child_sql = build_feedback_filter_sql_recursive(
                child.as_ref(),
                params_map,
                param_idx_counter,
                add_param_fn,
            );
            if child_sql.is_empty() {
                "".to_string()
            } else {
                format!("NOT ({})", child_sql)
            }
        }
    }
}
