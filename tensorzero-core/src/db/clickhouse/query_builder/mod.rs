use chrono::{DateTime, Utc};
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{BTreeSet, HashMap},
    fmt::{self, Display},
};
use uuid::Uuid;

use crate::{
    config::{Config, MetricConfigType},
    db::clickhouse::ClickhouseFormat,
    error::{Error, ErrorDetails},
    function::FunctionConfig,
    inference::types::{ContentBlockChatOutput, JsonInferenceOutput, StoredInput},
    serde_util::{deserialize_defaulted_string, deserialize_json_string},
    stored_inference::{StoredChatInference, StoredInference, StoredJsonInference},
    tool::ToolCallConfigDatabaseInsert,
};

#[cfg_attr(test, derive(ts_rs::TS))]
#[derive(Clone, Copy, Debug, PartialEq, Deserialize, Serialize)]
#[cfg_attr(test, ts(export))]
pub enum InferenceOutputSource {
    Inference,
    Demonstration,
}

impl TryFrom<&str> for InferenceOutputSource {
    type Error = Error;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "inference" => Ok(InferenceOutputSource::Inference),
            "demonstration" => Ok(InferenceOutputSource::Demonstration),
            _ => Err(Error::new(ErrorDetails::InvalidInferenceOutputSource {
                source_kind: value.to_string(),
            })),
        }
    }
}

#[cfg_attr(test, derive(ts_rs::TS))]
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[cfg_attr(test, ts(export))]
pub enum FloatComparisonOperator {
    #[serde(rename = "<")]
    LessThan,
    #[serde(rename = "<=")]
    LessThanOrEqual,
    #[serde(rename = "=")]
    Equal,
    #[serde(rename = ">")]
    GreaterThan,
    #[serde(rename = ">=")]
    GreaterThanOrEqual,
    #[serde(rename = "!=")]
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

#[cfg_attr(test, derive(ts_rs::TS))]
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[cfg_attr(test, ts(export))]
pub enum TimeComparisonOperator {
    #[serde(rename = "<")]
    LessThan,
    #[serde(rename = "<=")]
    LessThanOrEqual,
    #[serde(rename = "=")]
    Equal,
    #[serde(rename = ">")]
    GreaterThan,
    #[serde(rename = ">=")]
    GreaterThanOrEqual,
    #[serde(rename = "!=")]
    NotEqual,
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

#[cfg_attr(test, derive(ts_rs::TS))]
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[cfg_attr(test, ts(export))]
pub enum TagComparisonOperator {
    #[serde(rename = "=")]
    Equal,
    #[serde(rename = "!=")]
    NotEqual,
}

impl TagComparisonOperator {
    pub fn to_clickhouse_operator(&self) -> &str {
        match self {
            TagComparisonOperator::Equal => "=",
            TagComparisonOperator::NotEqual => "!=",
        }
    }
}

#[cfg_attr(test, derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[cfg_attr(test, ts(export))]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum OrderDirection {
    Asc,
    Desc,
}

impl OrderDirection {
    pub fn to_clickhouse_direction(&self) -> &str {
        match self {
            OrderDirection::Asc => "ASC",
            OrderDirection::Desc => "DESC",
        }
    }
}

#[cfg_attr(test, derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[cfg_attr(test, ts(export))]
#[serde(tag = "by", rename_all = "snake_case")]
pub enum OrderByTerm {
    Timestamp,
    Metric { name: String },
}

#[cfg_attr(test, derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[cfg_attr(test, ts(export))]
pub struct OrderBy {
    #[serde(flatten)]
    pub term: OrderByTerm,
    pub direction: OrderDirection,
}

#[derive(Hash, Eq, PartialEq, Debug)]
struct JoinKey {
    table: MetricConfigType,
    metric_name: String,
    inference_column_name: &'static str,
}

struct JoinRegistry {
    // map key to join alias
    aliases: HashMap<JoinKey, String>,
    // The actual JOIN clauses that have been registered
    clauses: Vec<String>,
}

impl JoinRegistry {
    fn new() -> Self {
        Self {
            aliases: HashMap::new(),
            clauses: Vec::new(),
        }
    }

    fn get_clauses(&self) -> &[String] {
        &self.clauses
    }

    /// Add a new join clause to the registry if the join
    /// for a particular key has not been added yet.
    /// If this is the first time we're adding a join for a particular key,
    /// we will also add the join clause to the registry.
    ///
    /// Returns the alias for the joined table.
    fn get_or_insert(
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
    fn insert_unchecked(&mut self, clause: String) {
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

#[cfg_attr(test, derive(ts_rs::TS))]
#[derive(Debug, Clone, Deserialize, Serialize)]
#[cfg_attr(test, ts(export))]
pub struct FloatMetricFilter {
    pub metric_name: String,
    pub value: f64,
    pub comparison_operator: FloatComparisonOperator,
}

#[cfg_attr(test, derive(ts_rs::TS))]
#[derive(Debug, Clone, Deserialize, Serialize)]
#[cfg_attr(test, ts(export))]
pub struct BooleanMetricFilter {
    pub metric_name: String,
    pub value: bool,
}

#[cfg_attr(test, derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, Serialize)]
#[cfg_attr(test, ts(export))]
pub struct TagFilter {
    pub key: String,
    pub value: String,
    pub comparison_operator: TagComparisonOperator,
}

#[cfg_attr(test, derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, Serialize)]
#[cfg_attr(test, ts(export))]
pub struct TimeFilter {
    #[cfg_attr(test, ts(type = "Date"))]
    pub time: DateTime<Utc>,
    pub comparison_operator: TimeComparisonOperator,
}

#[cfg_attr(test, derive(ts_rs::TS))]
#[derive(Debug, Clone, Deserialize, Serialize)]
#[cfg_attr(test, ts(export))]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InferenceFilterTreeNode {
    FloatMetric(FloatMetricFilter),
    BooleanMetric(BooleanMetricFilter),
    Tag(TagFilter),
    Time(TimeFilter),
    And {
        children: Vec<InferenceFilterTreeNode>,
    },
    Or {
        children: Vec<InferenceFilterTreeNode>,
    },
    Not {
        child: Box<InferenceFilterTreeNode>,
    },
}

impl InferenceFilterTreeNode {
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
    fn to_clickhouse_sql(
        &self,
        config: &Config,
        params_map: &mut Vec<QueryParameter>,
        _select_clauses: &mut BTreeSet<String>,
        joins: &mut JoinRegistry,
        param_idx_counter: &mut usize,
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
            InferenceFilterTreeNode::BooleanMetric(bm_node) => {
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
            InferenceFilterTreeNode::Tag(TagFilter {
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
            InferenceFilterTreeNode::Time(TimeFilter {
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
            InferenceFilterTreeNode::And { children } => {
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
            InferenceFilterTreeNode::Or { children } => {
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
            InferenceFilterTreeNode::Not { child } => {
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

fn generate_order_by_sql(
    order_by: Option<&[OrderBy]>,
    config: &Config,
    params_map: &mut Vec<QueryParameter>,
    param_idx_counter: &mut usize,
    joins: &mut JoinRegistry,
) -> Result<String, Error> {
    let Some(order_by) = order_by else {
        return Ok(String::new());
    };
    if order_by.is_empty() {
        return Ok(String::new());
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
        };
        let direction = term.direction.to_clickhouse_direction();
        order_by_clauses.push(format!("{sql_expr} {direction} NULLS LAST"));
    }
    let joined_clauses = order_by_clauses.join(", ");
    Ok(format!("\nORDER BY {joined_clauses}"))
}

#[derive(Debug, Clone)]
pub struct ListInferencesParams<'a> {
    pub function_name: &'a str,
    pub variant_name: Option<&'a str>,
    pub filters: Option<&'a InferenceFilterTreeNode>,
    pub output_source: InferenceOutputSource,
    pub limit: Option<u64>,
    pub offset: Option<u64>,
    pub order_by: Option<&'a [OrderBy]>,
    pub format: ClickhouseFormat,
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
pub fn generate_list_inferences_sql(
    config: &Config,
    opts: &ListInferencesParams<'_>,
) -> Result<(String, Vec<QueryParameter>), Error> {
    let mut params_map: Vec<QueryParameter> = Vec::new();
    let mut param_idx_counter = 0; // Counter for unique parameter names

    let function_config = config.get_function(opts.function_name)?;
    let function_name_param_placeholder = add_parameter(
        opts.function_name,
        ClickhouseType::String,
        &mut params_map,
        &mut param_idx_counter,
    );
    let mut select_clauses = get_select_clauses(&function_config, &function_name_param_placeholder);
    let mut joins = JoinRegistry::new();
    let mut where_clauses: Vec<String> = Vec::new();

    let inference_table_name = function_config.table_name();

    where_clauses.push(format!(
        "i.function_name = {function_name_param_placeholder}"
    ));

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

fn get_select_clauses(
    function_config: &FunctionConfig,
    function_name_param_placeholder: &str,
) -> BTreeSet<String> {
    let mut select_clauses = BTreeSet::from([
        format!("{function_name_param_placeholder} as function_name"),
        "i.input as input".to_string(),
        "i.variant_name as variant_name".to_string(),
        "i.episode_id as episode_id".to_string(),
        "i.id as inference_id".to_string(),
        "formatDateTime(i.timestamp, '%Y-%m-%dT%H:%i:%SZ') as timestamp".to_string(),
        "i.tags as tags".to_string(),
        // We don't select output here because it's handled separately based on the output_source
    ]);
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
    select_clauses
}

#[derive(Debug, Deserialize)]
pub(super) struct ClickHouseStoredChatInference {
    pub function_name: String,
    pub variant_name: String,
    pub episode_id: Uuid,
    pub inference_id: Uuid,
    pub timestamp: DateTime<Utc>,
    #[serde(deserialize_with = "deserialize_json_string")]
    pub input: StoredInput,
    #[serde(deserialize_with = "deserialize_json_string")]
    pub output: Vec<ContentBlockChatOutput>,
    #[serde(default)]
    pub dispreferred_outputs: Vec<String>,
    #[serde(deserialize_with = "deserialize_defaulted_string")]
    pub tool_params: ToolCallConfigDatabaseInsert,
    pub tags: HashMap<String, String>,
}

impl TryFrom<ClickHouseStoredChatInference> for StoredChatInference {
    type Error = Error;

    fn try_from(value: ClickHouseStoredChatInference) -> Result<Self, Self::Error> {
        let dispreferred_outputs = value
            .dispreferred_outputs
            .into_iter()
            .map(|dispreferred_output| {
                serde_json::from_str(&dispreferred_output).map_err(|e| {
                    Error::new(ErrorDetails::ClickHouseDeserialization {
                        message: format!("Failed to deserialize dispreferred output: {e}"),
                    })
                })
            })
            .collect::<Result<Vec<Vec<ContentBlockChatOutput>>, Error>>()?;

        Ok(StoredChatInference {
            function_name: value.function_name,
            variant_name: value.variant_name,
            input: value.input,
            output: value.output,
            dispreferred_outputs,
            episode_id: value.episode_id,
            inference_id: value.inference_id,
            tool_params: value.tool_params,
            tags: value.tags,
            timestamp: value.timestamp,
        })
    }
}

#[derive(Debug, Deserialize)]
pub(super) struct ClickHouseStoredJsonInference {
    pub function_name: String,
    pub variant_name: String,
    pub episode_id: Uuid,
    pub inference_id: Uuid,
    pub timestamp: DateTime<Utc>,
    #[serde(deserialize_with = "deserialize_json_string")]
    pub input: StoredInput,
    #[serde(deserialize_with = "deserialize_json_string")]
    pub output: JsonInferenceOutput,
    #[serde(default)]
    pub dispreferred_outputs: Vec<String>,
    #[serde(deserialize_with = "deserialize_json_string")]
    pub output_schema: Value,
    pub tags: HashMap<String, String>,
}

impl TryFrom<ClickHouseStoredJsonInference> for StoredJsonInference {
    type Error = Error;

    fn try_from(value: ClickHouseStoredJsonInference) -> Result<Self, Self::Error> {
        let dispreferred_outputs = value
            .dispreferred_outputs
            .into_iter()
            .map(|dispreferred_output| {
                serde_json::from_str(&dispreferred_output).map_err(|e| {
                    Error::new(ErrorDetails::ClickHouseDeserialization {
                        message: format!("Failed to deserialize dispreferred output: {e}"),
                    })
                })
            })
            .collect::<Result<Vec<JsonInferenceOutput>, Error>>()?;
        Ok(StoredJsonInference {
            function_name: value.function_name,
            variant_name: value.variant_name,
            input: value.input,
            output: value.output,
            dispreferred_outputs,
            episode_id: value.episode_id,
            inference_id: value.inference_id,
            output_schema: value.output_schema,
            tags: value.tags,
            timestamp: value.timestamp,
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(super) enum ClickHouseStoredInference {
    Json(ClickHouseStoredJsonInference),
    Chat(ClickHouseStoredChatInference),
}

impl TryFrom<ClickHouseStoredInference> for StoredInference {
    type Error = Error;

    fn try_from(value: ClickHouseStoredInference) -> Result<Self, Self::Error> {
        Ok(match value {
            ClickHouseStoredInference::Json(inference) => {
                StoredInference::Json(inference.try_into()?)
            }
            ClickHouseStoredInference::Chat(inference) => {
                StoredInference::Chat(inference.try_into()?)
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use std::path::Path;

    use crate::inference::types::StoredInput;
    use crate::{config::ConfigFileGlob, inference::types::Text, tool::ToolChoice};

    use super::*;

    async fn get_e2e_config() -> Config {
        // Read the e2e config file
        Config::load_from_path_optional_verify_credentials(
            &ConfigFileGlob::new_from_path(Path::new("tests/e2e/tensorzero.toml")).unwrap(),
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
            order_by: None,
            format: ClickhouseFormat::JsonEachRow,
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
        let expected_sql = r"
SELECT
    'json' as type,
    formatDateTime(i.timestamp, '%Y-%m-%dT%H:%i:%SZ') as timestamp,
    i.episode_id as episode_id,
    i.id as inference_id,
    i.input as input,
    i.output as output,
    i.output_schema as output_schema,
    i.tags as tags,
    i.variant_name as variant_name,
    {p0:String} as function_name
FROM
    JsonInference AS i
WHERE
    i.function_name = {p0:String}
FORMAT JSONEachRow";
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
            order_by: None,
            format: ClickhouseFormat::JsonEachRow,
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
        let expected_sql = r"
SELECT
    'chat' as type,
    formatDateTime(i.timestamp, '%Y-%m-%dT%H:%i:%SZ') as timestamp,
    i.episode_id as episode_id,
    i.id as inference_id,
    i.input as input,
    i.output as output,
    i.tags as tags,
    i.tool_params as tool_params,
    i.variant_name as variant_name,
    {p0:String} as function_name
FROM
    ChatInference AS i
WHERE
    i.function_name = {p0:String}
FORMAT JSONEachRow";
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
        let filter_node = InferenceFilterTreeNode::FloatMetric(FloatMetricFilter {
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
            order_by: None,
            format: ClickhouseFormat::JsonEachRow,
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
        let expected_sql = r"
SELECT
    'json' as type,
    formatDateTime(i.timestamp, '%Y-%m-%dT%H:%i:%SZ') as timestamp,
    i.episode_id as episode_id,
    i.id as inference_id,
    i.input as input,
    i.output as output,
    i.output_schema as output_schema,
    i.tags as tags,
    i.variant_name as variant_name,
    {p0:String} as function_name
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
FORMAT JSONEachRow";
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
            order_by: None,
            format: ClickhouseFormat::JsonEachRow,
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
        let filter_node = InferenceFilterTreeNode::FloatMetric(FloatMetricFilter {
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
            order_by: None,
            format: ClickhouseFormat::JsonEachRow,
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
            order_by: None,
            format: ClickhouseFormat::JsonEachRow,
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
        let expected_sql = r"
SELECT
    'json' as type,
    [i.output] as dispreferred_outputs,
    demo_f.value AS output,
    formatDateTime(i.timestamp, '%Y-%m-%dT%H:%i:%SZ') as timestamp,
    i.episode_id as episode_id,
    i.id as inference_id,
    i.input as input,
    i.output_schema as output_schema,
    i.tags as tags,
    i.variant_name as variant_name,
    {p0:String} as function_name
FROM
    JsonInference AS i
JOIN (SELECT inference_id, argMax(value, timestamp) as value FROM DemonstrationFeedback GROUP BY inference_id ) AS demo_f ON i.id = demo_f.inference_id
WHERE
    i.function_name = {p0:String}
FORMAT JSONEachRow";
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
        let filter_node = InferenceFilterTreeNode::BooleanMetric(BooleanMetricFilter {
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
            order_by: None,
            format: ClickhouseFormat::JsonEachRow,
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
        let expected_sql = r"
SELECT
    'json' as type,
    formatDateTime(i.timestamp, '%Y-%m-%dT%H:%i:%SZ') as timestamp,
    i.episode_id as episode_id,
    i.id as inference_id,
    i.input as input,
    i.output as output,
    i.output_schema as output_schema,
    i.tags as tags,
    i.variant_name as variant_name,
    {p0:String} as function_name
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
FORMAT JSONEachRow";
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
        let filter_node = InferenceFilterTreeNode::BooleanMetric(BooleanMetricFilter {
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
            order_by: None,
            format: ClickhouseFormat::JsonEachRow,
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
        let expected_sql = r"
SELECT
    'json' as type,
    formatDateTime(i.timestamp, '%Y-%m-%dT%H:%i:%SZ') as timestamp,
    i.episode_id as episode_id,
    i.id as inference_id,
    i.input as input,
    i.output as output,
    i.output_schema as output_schema,
    i.tags as tags,
    i.variant_name as variant_name,
    {p0:String} as function_name
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
FORMAT JSONEachRow";
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
        let filter_node = InferenceFilterTreeNode::And {
            children: vec![
                InferenceFilterTreeNode::FloatMetric(FloatMetricFilter {
                    metric_name: "jaccard_similarity".to_string(),
                    value: 0.5,
                    comparison_operator: FloatComparisonOperator::GreaterThan,
                }),
                // We test that the join is not duplicated
                InferenceFilterTreeNode::FloatMetric(FloatMetricFilter {
                    metric_name: "jaccard_similarity".to_string(),
                    value: 0.8,
                    comparison_operator: FloatComparisonOperator::LessThan,
                }),
                InferenceFilterTreeNode::FloatMetric(FloatMetricFilter {
                    metric_name: "brevity_score".to_string(),
                    value: 10.0,
                    comparison_operator: FloatComparisonOperator::LessThan,
                }),
            ],
        };
        let opts = ListInferencesParams {
            function_name: "extract_entities",
            variant_name: None,
            filters: Some(&filter_node),
            output_source: InferenceOutputSource::Inference,
            limit: None,
            offset: None,
            order_by: None,
            format: ClickhouseFormat::JsonEachRow,
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
        let expected_sql = r"
SELECT
    'json' as type,
    formatDateTime(i.timestamp, '%Y-%m-%dT%H:%i:%SZ') as timestamp,
    i.episode_id as episode_id,
    i.id as inference_id,
    i.input as input,
    i.output as output,
    i.output_schema as output_schema,
    i.tags as tags,
    i.variant_name as variant_name,
    {p0:String} as function_name
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
FORMAT JSONEachRow";
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
        ];
        assert_eq!(params, expected_params);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_or_filter_mixed_metrics() {
        let config = get_e2e_config().await;
        let filter_node = InferenceFilterTreeNode::Or {
            children: vec![
                InferenceFilterTreeNode::FloatMetric(FloatMetricFilter {
                    metric_name: "jaccard_similarity".to_string(),
                    value: 0.8,
                    comparison_operator: FloatComparisonOperator::GreaterThanOrEqual,
                }),
                InferenceFilterTreeNode::BooleanMetric(BooleanMetricFilter {
                    metric_name: "exact_match".to_string(),
                    value: true,
                }),
                InferenceFilterTreeNode::BooleanMetric(BooleanMetricFilter {
                    // Episode-level metric
                    metric_name: "goal_achieved".to_string(),
                    value: true,
                }),
            ],
        };
        let opts = ListInferencesParams {
            function_name: "extract_entities",
            variant_name: None,
            filters: Some(&filter_node),
            output_source: InferenceOutputSource::Inference,
            limit: None,
            offset: None,
            order_by: None,
            format: ClickhouseFormat::JsonEachRow,
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
        let expected_sql = r"
SELECT
    'json' as type,
    formatDateTime(i.timestamp, '%Y-%m-%dT%H:%i:%SZ') as timestamp,
    i.episode_id as episode_id,
    i.id as inference_id,
    i.input as input,
    i.output as output,
    i.output_schema as output_schema,
    i.tags as tags,
    i.variant_name as variant_name,
    {p0:String} as function_name
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
FORMAT JSONEachRow";
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
            QueryParameter {
                name: "p5".to_string(),
                value: "goal_achieved".to_string(),
            },
            QueryParameter {
                name: "p6".to_string(),
                value: "1".to_string(),
            },
        ];
        assert_eq!(params, expected_params);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_not_filter() {
        let config = get_e2e_config().await;
        let filter_node = InferenceFilterTreeNode::Not {
            child: Box::new(InferenceFilterTreeNode::Or {
                children: vec![
                    InferenceFilterTreeNode::BooleanMetric(BooleanMetricFilter {
                        metric_name: "task_success".to_string(),
                        value: true,
                    }),
                    InferenceFilterTreeNode::BooleanMetric(BooleanMetricFilter {
                        metric_name: "task_success".to_string(),
                        value: false,
                    }),
                ],
            }),
        };
        let opts = ListInferencesParams {
            function_name: "extract_entities",
            variant_name: None,
            filters: Some(&filter_node),
            output_source: InferenceOutputSource::Inference,
            limit: None,
            offset: None,
            order_by: None,
            format: ClickhouseFormat::JsonEachRow,
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
        let expected_sql = r"
SELECT
    'json' as type,
    formatDateTime(i.timestamp, '%Y-%m-%dT%H:%i:%SZ') as timestamp,
    i.episode_id as episode_id,
    i.id as inference_id,
    i.input as input,
    i.output as output,
    i.output_schema as output_schema,
    i.tags as tags,
    i.variant_name as variant_name,
    {p0:String} as function_name
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
FORMAT JSONEachRow";
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
            QueryParameter {
                name: "p3".to_string(),
                value: "0".to_string(),
            },
        ];
        assert_eq!(params, expected_params);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_nested_complex_filter() {
        let config = get_e2e_config().await;
        let filter_node = InferenceFilterTreeNode::And {
            children: vec![
                InferenceFilterTreeNode::Or {
                    children: vec![
                        InferenceFilterTreeNode::FloatMetric(FloatMetricFilter {
                            metric_name: "jaccard_similarity".to_string(),
                            value: 0.7,
                            comparison_operator: FloatComparisonOperator::GreaterThan,
                        }),
                        InferenceFilterTreeNode::FloatMetric(FloatMetricFilter {
                            metric_name: "brevity_score".to_string(),
                            value: 5.0,
                            comparison_operator: FloatComparisonOperator::LessThanOrEqual,
                        }),
                    ],
                },
                InferenceFilterTreeNode::Not {
                    child: Box::new(InferenceFilterTreeNode::BooleanMetric(
                        BooleanMetricFilter {
                            metric_name: "task_success".to_string(),
                            value: false,
                        },
                    )),
                },
            ],
        };
        let opts = ListInferencesParams {
            function_name: "extract_entities",
            variant_name: None,
            filters: Some(&filter_node),
            output_source: InferenceOutputSource::Inference,
            limit: None,
            offset: None,
            order_by: None,
            format: ClickhouseFormat::JsonEachRow,
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
        let expected_sql = r"
SELECT
    'json' as type,
    formatDateTime(i.timestamp, '%Y-%m-%dT%H:%i:%SZ') as timestamp,
    i.episode_id as episode_id,
    i.id as inference_id,
    i.input as input,
    i.output as output,
    i.output_schema as output_schema,
    i.tags as tags,
    i.variant_name as variant_name,
    {p0:String} as function_name
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
FORMAT JSONEachRow";
        assert_eq!(sql, expected_sql);
        assert_eq!(params.len(), 7); // p0 (function) + 6 metric-related params
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_nested_complex_filter_with_time() {
        let config = get_e2e_config().await;
        let filter_node = InferenceFilterTreeNode::And {
            children: vec![
                InferenceFilterTreeNode::Time(TimeFilter {
                    time: DateTime::from_timestamp(1609459200, 0).unwrap(), // 2021-01-01 00:00:00 UTC
                    comparison_operator: TimeComparisonOperator::GreaterThan,
                }),
                InferenceFilterTreeNode::Or {
                    children: vec![
                        InferenceFilterTreeNode::Time(TimeFilter {
                            time: DateTime::from_timestamp(1672531200, 0).unwrap(), // 2023-01-01 00:00:00 UTC
                            comparison_operator: TimeComparisonOperator::LessThan,
                        }),
                        InferenceFilterTreeNode::And {
                            children: vec![
                                InferenceFilterTreeNode::FloatMetric(FloatMetricFilter {
                                    metric_name: "jaccard_similarity".to_string(),
                                    value: 0.9,
                                    comparison_operator:
                                        FloatComparisonOperator::GreaterThanOrEqual,
                                }),
                                InferenceFilterTreeNode::Tag(TagFilter {
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
            function_name: "extract_entities",
            variant_name: None,
            filters: Some(&filter_node),
            output_source: InferenceOutputSource::Inference,
            limit: None,
            offset: None,
            order_by: None,
            format: ClickhouseFormat::JsonEachRow,
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
        let expected_sql = r"
SELECT
    'json' as type,
    formatDateTime(i.timestamp, '%Y-%m-%dT%H:%i:%SZ') as timestamp,
    i.episode_id as episode_id,
    i.id as inference_id,
    i.input as input,
    i.output as output,
    i.output_schema as output_schema,
    i.tags as tags,
    i.variant_name as variant_name,
    {p0:String} as function_name
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
FORMAT JSONEachRow";
        assert_eq!(sql, expected_sql);
        assert_eq!(params.len(), 7); // p0 (function) + 6 filter-related params
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
            order_by: None,
            format: ClickhouseFormat::JsonEachRow,
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
        let expected_sql = r"
SELECT
    'json' as type,
    formatDateTime(i.timestamp, '%Y-%m-%dT%H:%i:%SZ') as timestamp,
    i.episode_id as episode_id,
    i.id as inference_id,
    i.input as input,
    i.output as output,
    i.output_schema as output_schema,
    i.tags as tags,
    i.variant_name as variant_name,
    {p0:String} as function_name
FROM
    JsonInference AS i
WHERE
    i.function_name = {p0:String} AND i.variant_name = {p1:String}
FORMAT JSONEachRow";
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
            order_by: None,
            format: ClickhouseFormat::JsonEachRow,
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
        let expected_sql = r"
SELECT
    'json' as type,
    formatDateTime(i.timestamp, '%Y-%m-%dT%H:%i:%SZ') as timestamp,
    i.episode_id as episode_id,
    i.id as inference_id,
    i.input as input,
    i.output as output,
    i.output_schema as output_schema,
    i.tags as tags,
    i.variant_name as variant_name,
    {p0:String} as function_name
FROM
    JsonInference AS i
WHERE
    i.function_name = {p0:String}
LIMIT {p1:UInt64}
OFFSET {p2:UInt64}
FORMAT JSONEachRow";
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
            let filter_node = InferenceFilterTreeNode::FloatMetric(FloatMetricFilter {
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
                order_by: None,
                format: ClickhouseFormat::JsonEachRow,
            };
            let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
            let expected_sql = format!(
                r"
SELECT
    'json' as type,
    formatDateTime(i.timestamp, '%Y-%m-%dT%H:%i:%SZ') as timestamp,
    i.episode_id as episode_id,
    i.id as inference_id,
    i.input as input,
    i.output as output,
    i.output_schema as output_schema,
    i.tags as tags,
    i.variant_name as variant_name,
    {{p0:String}} as function_name
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
FORMAT JSONEachRow",
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
    async fn test_simple_tag_filter_equal() {
        let config = get_e2e_config().await;
        let filter_node = InferenceFilterTreeNode::Tag(TagFilter {
            key: "environment".to_string(),
            value: "production".to_string(),
            comparison_operator: TagComparisonOperator::Equal,
        });
        let opts = ListInferencesParams {
            function_name: "extract_entities",
            variant_name: None,
            filters: Some(&filter_node),
            output_source: InferenceOutputSource::Inference,
            limit: None,
            offset: None,
            order_by: None,
            format: ClickhouseFormat::JsonEachRow,
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
        let expected_sql = r"
SELECT
    'json' as type,
    formatDateTime(i.timestamp, '%Y-%m-%dT%H:%i:%SZ') as timestamp,
    i.episode_id as episode_id,
    i.id as inference_id,
    i.input as input,
    i.output as output,
    i.output_schema as output_schema,
    i.tags as tags,
    i.variant_name as variant_name,
    {p0:String} as function_name
FROM
    JsonInference AS i
WHERE
    i.function_name = {p0:String} AND i.tags[{p1:String}] = {p2:String}
FORMAT JSONEachRow";
        assert_eq!(sql, expected_sql);
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
        ];
        assert_eq!(params, expected_params);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_tag_filter_not_equal() {
        let config = get_e2e_config().await;
        let filter_node = InferenceFilterTreeNode::Tag(TagFilter {
            key: "version".to_string(),
            value: "v1.0".to_string(),
            comparison_operator: TagComparisonOperator::NotEqual,
        });
        let opts = ListInferencesParams {
            function_name: "write_haiku",
            variant_name: None,
            filters: Some(&filter_node),
            output_source: InferenceOutputSource::Inference,
            limit: None,
            offset: None,
            order_by: None,
            format: ClickhouseFormat::JsonEachRow,
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
        let expected_sql = r"
SELECT
    'chat' as type,
    formatDateTime(i.timestamp, '%Y-%m-%dT%H:%i:%SZ') as timestamp,
    i.episode_id as episode_id,
    i.id as inference_id,
    i.input as input,
    i.output as output,
    i.tags as tags,
    i.tool_params as tool_params,
    i.variant_name as variant_name,
    {p0:String} as function_name
FROM
    ChatInference AS i
WHERE
    i.function_name = {p0:String} AND i.tags[{p1:String}] != {p2:String}
FORMAT JSONEachRow";
        assert_eq!(sql, expected_sql);
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
        ];
        assert_eq!(params, expected_params);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_tag_filters_in_and_condition() {
        let config = get_e2e_config().await;
        let filter_node = InferenceFilterTreeNode::And {
            children: vec![
                InferenceFilterTreeNode::Tag(TagFilter {
                    key: "environment".to_string(),
                    value: "production".to_string(),
                    comparison_operator: TagComparisonOperator::Equal,
                }),
                InferenceFilterTreeNode::Tag(TagFilter {
                    key: "region".to_string(),
                    value: "us-west".to_string(),
                    comparison_operator: TagComparisonOperator::Equal,
                }),
            ],
        };
        let opts = ListInferencesParams {
            function_name: "extract_entities",
            variant_name: None,
            filters: Some(&filter_node),
            output_source: InferenceOutputSource::Inference,
            limit: None,
            offset: None,
            order_by: None,
            format: ClickhouseFormat::JsonEachRow,
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
        let expected_sql = r"
SELECT
    'json' as type,
    formatDateTime(i.timestamp, '%Y-%m-%dT%H:%i:%SZ') as timestamp,
    i.episode_id as episode_id,
    i.id as inference_id,
    i.input as input,
    i.output as output,
    i.output_schema as output_schema,
    i.tags as tags,
    i.variant_name as variant_name,
    {p0:String} as function_name
FROM
    JsonInference AS i
WHERE
    i.function_name = {p0:String} AND (COALESCE(i.tags[{p1:String}] = {p2:String}, 0) AND COALESCE(i.tags[{p3:String}] = {p4:String}, 0))
FORMAT JSONEachRow";
        assert_eq!(sql, expected_sql);
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
        ];
        assert_eq!(params, expected_params);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_tag_and_metric_filters_combined() {
        let config = get_e2e_config().await;
        let filter_node = InferenceFilterTreeNode::And {
            children: vec![
                InferenceFilterTreeNode::Tag(TagFilter {
                    key: "experiment".to_string(),
                    value: "A".to_string(),
                    comparison_operator: TagComparisonOperator::Equal,
                }),
                InferenceFilterTreeNode::FloatMetric(FloatMetricFilter {
                    metric_name: "jaccard_similarity".to_string(),
                    value: 0.7,
                    comparison_operator: FloatComparisonOperator::GreaterThan,
                }),
            ],
        };
        let opts = ListInferencesParams {
            function_name: "extract_entities",
            variant_name: None,
            filters: Some(&filter_node),
            output_source: InferenceOutputSource::Inference,
            limit: None,
            offset: None,
            order_by: None,
            format: ClickhouseFormat::JsonEachRow,
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
        let expected_sql = r"
SELECT
    'json' as type,
    formatDateTime(i.timestamp, '%Y-%m-%dT%H:%i:%SZ') as timestamp,
    i.episode_id as episode_id,
    i.id as inference_id,
    i.input as input,
    i.output as output,
    i.output_schema as output_schema,
    i.tags as tags,
    i.variant_name as variant_name,
    {p0:String} as function_name
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
FORMAT JSONEachRow";
        assert_eq!(sql, expected_sql);
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
        ];
        assert_eq!(params, expected_params);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_combined_variant_filter_and_metrics() {
        let config = get_e2e_config().await;
        let filter_node = InferenceFilterTreeNode::And {
            children: vec![
                InferenceFilterTreeNode::FloatMetric(FloatMetricFilter {
                    metric_name: "jaccard_similarity".to_string(),
                    value: 0.6,
                    comparison_operator: FloatComparisonOperator::GreaterThan,
                }),
                InferenceFilterTreeNode::BooleanMetric(BooleanMetricFilter {
                    metric_name: "exact_match".to_string(),
                    value: true,
                }),
            ],
        };
        let opts = ListInferencesParams {
            function_name: "extract_entities",
            variant_name: Some("production"),
            filters: Some(&filter_node),
            output_source: InferenceOutputSource::Demonstration,
            limit: Some(25),
            offset: Some(50),
            order_by: None,
            format: ClickhouseFormat::JsonEachRow,
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
        let expected_sql = r"
SELECT
    'json' as type,
    [i.output] as dispreferred_outputs,
    demo_f.value AS output,
    formatDateTime(i.timestamp, '%Y-%m-%dT%H:%i:%SZ') as timestamp,
    i.episode_id as episode_id,
    i.id as inference_id,
    i.input as input,
    i.output_schema as output_schema,
    i.tags as tags,
    i.variant_name as variant_name,
    {p0:String} as function_name
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

    #[tokio::test(flavor = "multi_thread")]
    async fn test_simple_time_filter() {
        let config = get_e2e_config().await;
        let filter_node = InferenceFilterTreeNode::Time(TimeFilter {
            time: DateTime::from_timestamp(1672531200, 0).unwrap(), // 2023-01-01 00:00:00 UTC
            comparison_operator: TimeComparisonOperator::GreaterThan,
        });
        let opts = ListInferencesParams {
            function_name: "extract_entities",
            variant_name: None,
            filters: Some(&filter_node),
            output_source: InferenceOutputSource::Inference,
            limit: None,
            offset: None,
            order_by: None,
            format: ClickhouseFormat::JsonEachRow,
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
        let expected_sql = r"
SELECT
    'json' as type,
    formatDateTime(i.timestamp, '%Y-%m-%dT%H:%i:%SZ') as timestamp,
    i.episode_id as episode_id,
    i.id as inference_id,
    i.input as input,
    i.output as output,
    i.output_schema as output_schema,
    i.tags as tags,
    i.variant_name as variant_name,
    {p0:String} as function_name
FROM
    JsonInference AS i
WHERE
    i.function_name = {p0:String} AND i.timestamp > parseDateTimeBestEffort({p1:String})
FORMAT JSONEachRow";
        assert_eq!(sql, expected_sql);
        let expected_params = vec![
            QueryParameter {
                name: "p0".to_string(),
                value: "extract_entities".to_string(),
            },
            QueryParameter {
                name: "p1".to_string(),
                value: "2023-01-01 00:00:00 UTC".to_string(),
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
            let filter_node = InferenceFilterTreeNode::Time(TimeFilter {
                time: DateTime::from_timestamp(1672531200, 0).unwrap(), // 2023-01-01 00:00:00 UTC
                comparison_operator: op,
            });
            let opts = ListInferencesParams {
                function_name: "write_haiku",
                variant_name: None,
                filters: Some(&filter_node),
                output_source: InferenceOutputSource::Inference,
                limit: None,
                offset: None,
                order_by: None,
                format: ClickhouseFormat::JsonEachRow,
            };
            let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
            let expected_sql = format!(
                r"
SELECT
    'chat' as type,
    formatDateTime(i.timestamp, '%Y-%m-%dT%H:%i:%SZ') as timestamp,
    i.episode_id as episode_id,
    i.id as inference_id,
    i.input as input,
    i.output as output,
    i.tags as tags,
    i.tool_params as tool_params,
    i.variant_name as variant_name,
    {{p0:String}} as function_name
FROM
    ChatInference AS i
WHERE
    i.function_name = {{p0:String}} AND i.timestamp {expected_op_str} parseDateTimeBestEffort({{p1:String}})
FORMAT JSONEachRow",
            );
            assert_eq!(sql, expected_sql);
            let expected_params = vec![
                QueryParameter {
                    name: "p0".to_string(),
                    value: "write_haiku".to_string(),
                },
                QueryParameter {
                    name: "p1".to_string(),
                    value: "2023-01-01 00:00:00 UTC".to_string(),
                },
            ];
            assert_eq!(params, expected_params);
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_time_filter_combined_with_other_filters() {
        let config = get_e2e_config().await;
        let filter_node = InferenceFilterTreeNode::And {
            children: vec![
                InferenceFilterTreeNode::Time(TimeFilter {
                    time: DateTime::from_timestamp(1672531200, 0).unwrap(), // 2023-01-01 00:00:00 UTC
                    comparison_operator: TimeComparisonOperator::GreaterThanOrEqual,
                }),
                InferenceFilterTreeNode::Tag(TagFilter {
                    key: "environment".to_string(),
                    value: "production".to_string(),
                    comparison_operator: TagComparisonOperator::Equal,
                }),
                InferenceFilterTreeNode::FloatMetric(FloatMetricFilter {
                    metric_name: "jaccard_similarity".to_string(),
                    value: 0.8,
                    comparison_operator: FloatComparisonOperator::GreaterThan,
                }),
            ],
        };
        let opts = ListInferencesParams {
            function_name: "extract_entities",
            variant_name: None,
            filters: Some(&filter_node),
            output_source: InferenceOutputSource::Inference,
            limit: Some(10),
            offset: None,
            order_by: None,
            format: ClickhouseFormat::JsonEachRow,
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
        let expected_sql = r"
SELECT
    'json' as type,
    formatDateTime(i.timestamp, '%Y-%m-%dT%H:%i:%SZ') as timestamp,
    i.episode_id as episode_id,
    i.id as inference_id,
    i.input as input,
    i.output as output,
    i.output_schema as output_schema,
    i.tags as tags,
    i.variant_name as variant_name,
    {p0:String} as function_name
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
FORMAT JSONEachRow";
        assert_eq!(sql, expected_sql);
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
        let inference: ClickHouseStoredInference = serde_json::from_str(json).unwrap();
        let StoredInference::Chat(chat_inference) = inference.try_into().unwrap() else {
            panic!("Expected a chat inference");
        };
        assert_eq!(chat_inference.function_name, "test_function");
        assert_eq!(chat_inference.variant_name, "test_variant");
        assert_eq!(
            chat_inference.input,
            StoredInput {
                system: Some(json!("you are a helpful assistant")),
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
            ToolCallConfigDatabaseInsert {
                tools_available: vec![],
                tool_choice: ToolChoice::None,
                parallel_tool_calls: Some(false),
            }
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
        let inference: StoredInference = serde_json::from_str(json).unwrap();
        let StoredInference::Chat(chat_inference) = inference else {
            panic!("Expected a chat inference");
        };
        assert_eq!(chat_inference.function_name, "test_function");
        assert_eq!(chat_inference.variant_name, "test_variant");
        assert_eq!(
            chat_inference.input,
            StoredInput {
                system: Some(json!("you are a helpful assistant")),
                messages: vec![],
            }
        );
        assert_eq!(
            chat_inference.output,
            vec!["Hello! How can I help you today?".to_string().into()]
        );
        assert_eq!(
            chat_inference.tool_params,
            ToolCallConfigDatabaseInsert {
                tools_available: vec![],
                tool_choice: ToolChoice::None,
                parallel_tool_calls: Some(false),
            }
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
        let inference: ClickHouseStoredInference = serde_json::from_str(json).unwrap();
        let StoredInference::Chat(chat_inference) = inference.try_into().unwrap() else {
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
        let inference: StoredInference = serde_json::from_str(json).unwrap();
        let StoredInference::Chat(chat_inference) = inference else {
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
        let inference: ClickHouseStoredInference = serde_json::from_str(json).unwrap();
        let StoredInference::Json(json_inference) = inference.try_into().unwrap() else {
            panic!("Expected a json inference");
        };
        assert_eq!(json_inference.function_name, "test_function");
        assert_eq!(json_inference.variant_name, "test_variant");
        assert_eq!(
            json_inference.input,
            StoredInput {
                system: Some(json!("you are a helpful assistant")),
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
        let inference: StoredInference = serde_json::from_str(json).unwrap();
        let StoredInference::Json(json_inference) = inference else {
            panic!("Expected a json inference");
        };
        assert_eq!(json_inference.function_name, "test_function");
        assert_eq!(json_inference.variant_name, "test_variant");
        assert_eq!(
            json_inference.input,
            StoredInput {
                system: Some(json!("you are a helpful assistant")),
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
        let inference: ClickHouseStoredInference = serde_json::from_str(json).unwrap();
        let StoredInference::Json(json_inference) = inference.try_into().unwrap() else {
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
        let inference: StoredInference = serde_json::from_str(json).unwrap();
        let StoredInference::Json(json_inference) = inference else {
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
            function_name: "extract_entities",
            variant_name: None,
            filters: None,
            output_source: InferenceOutputSource::Inference,
            limit: None,
            offset: None,
            order_by: Some(&order_by),
            format: ClickhouseFormat::JsonEachRow,
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();

        let expected_sql = r"
SELECT
    'json' as type,
    formatDateTime(i.timestamp, '%Y-%m-%dT%H:%i:%SZ') as timestamp,
    i.episode_id as episode_id,
    i.id as inference_id,
    i.input as input,
    i.output as output,
    i.output_schema as output_schema,
    i.tags as tags,
    i.variant_name as variant_name,
    {p0:String} as function_name
FROM
    JsonInference AS i
WHERE
    i.function_name = {p0:String}
ORDER BY i.timestamp DESC NULLS LAST
FORMAT JSONEachRow";
        assert_eq!(sql, expected_sql);

        let expected_params = vec![QueryParameter {
            name: "p0".to_string(),
            value: "extract_entities".to_string(),
        }];
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
            function_name: "extract_entities",
            variant_name: None,
            filters: None,
            output_source: InferenceOutputSource::Inference,
            limit: None,
            offset: None,
            order_by: Some(&order_by),
            format: ClickhouseFormat::JsonEachRow,
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();
        // NOTE: This test case enforces that the joins account for metrics that are only used in the order by clause.
        let expected_sql = r"
SELECT
    'json' as type,
    formatDateTime(i.timestamp, '%Y-%m-%dT%H:%i:%SZ') as timestamp,
    i.episode_id as episode_id,
    i.id as inference_id,
    i.input as input,
    i.output as output,
    i.output_schema as output_schema,
    i.tags as tags,
    i.variant_name as variant_name,
    {p0:String} as function_name
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
FORMAT JSONEachRow";
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
            function_name: "extract_entities",
            variant_name: None,
            filters: None,
            output_source: InferenceOutputSource::Inference,
            limit: None,
            offset: None,
            order_by: Some(&order_by),
            format: ClickhouseFormat::JsonEachRow,
        };
        let (sql, params) = generate_list_inferences_sql(&config, &opts).unwrap();

        let expected_sql = r"
SELECT
    'json' as type,
    formatDateTime(i.timestamp, '%Y-%m-%dT%H:%i:%SZ') as timestamp,
    i.episode_id as episode_id,
    i.id as inference_id,
    i.input as input,
    i.output as output,
    i.output_schema as output_schema,
    i.tags as tags,
    i.variant_name as variant_name,
    {p0:String} as function_name
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
FORMAT JSONEachRow";
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
        ];
        assert_eq!(params, expected_params);
    }
}
