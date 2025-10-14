use std::collections::HashMap;

use async_trait::async_trait;
use uuid::Uuid;

use crate::{
    db::{
        feedback::{
            BooleanMetricFeedbackRow, CommentFeedbackRow, DemonstrationFeedbackRow, FeedbackBounds,
            FeedbackBoundsByType, FeedbackByVariant, FeedbackRow, FloatMetricFeedbackRow,
        },
        FeedbackQueries, TableBounds,
    },
    error::Error,
};

use super::{
    escape_string_for_clickhouse_literal,
    select_queries::{
        build_pagination_clause, parse_count, parse_feedback_rows, parse_table_bounds,
    },
    ClickHouseConnectionInfo,
};

// Configuration for feedback table queries
struct FeedbackTableConfig {
    table_name: &'static str,
    id_column: &'static str,
    columns: &'static [&'static str],
}

impl FeedbackTableConfig {
    const BOOLEAN_METRICS: Self = Self {
        table_name: "BooleanMetricFeedbackByTargetId",
        id_column: "target_id",
        columns: &["id", "target_id", "metric_name", "value", "tags"],
    };

    const FLOAT_METRICS: Self = Self {
        table_name: "FloatMetricFeedbackByTargetId",
        id_column: "target_id",
        columns: &["id", "target_id", "metric_name", "value", "tags"],
    };

    const COMMENT_FEEDBACK: Self = Self {
        table_name: "CommentFeedbackByTargetId",
        id_column: "target_id",
        columns: &["id", "target_id", "target_type", "value", "tags"],
    };

    const DEMONSTRATION_FEEDBACK: Self = Self {
        table_name: "DemonstrationFeedbackByInferenceId",
        id_column: "inference_id",
        columns: &["id", "inference_id", "value", "tags"],
    };
}

// Helper function to build parameter map
fn build_params_map(
    id_column: &str,
    id_value: Uuid,
    pagination_params: Vec<(&str, String)>,
    page_size: u32,
) -> HashMap<String, String> {
    let mut params_map: HashMap<String, String> = pagination_params
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
    params_map.insert(id_column.to_string(), id_value.to_string());
    params_map.insert("page_size".to_string(), page_size.to_string());
    params_map
}

// Generic query builder for feedback tables
fn build_feedback_query(
    config: &FeedbackTableConfig,
    id_value: Uuid,
    before: Option<Uuid>,
    after: Option<Uuid>,
    page_size: u32,
) -> (String, HashMap<String, String>) {
    let (where_clause, params) = build_pagination_clause(before, after, config.id_column);
    let order_clause = if after.is_some() { "ASC" } else { "DESC" };

    let params_map = build_params_map(config.id_column, id_value, params, page_size);

    let columns_str = config.columns.join(", ");
    let id_param = format!("{{{id_column}:UUID}}", id_column = config.id_column);

    let query = if after.is_some() {
        format!(
            r"
            SELECT
                {columns_str},
                timestamp
            FROM (
                SELECT
                    {columns_str},
                    formatDateTime(UUIDv7ToDateTime(id), '%Y-%m-%dT%H:%i:%SZ') AS timestamp
                FROM {table_name}
                WHERE {id_column} = {id_param} {where_clause}
                ORDER BY toUInt128(id) {order_clause}
                LIMIT {{page_size:UInt32}}
            )
            ORDER BY toUInt128(id) DESC
            FORMAT JSONEachRow
            ",
            columns_str = columns_str,
            table_name = config.table_name,
            id_column = config.id_column,
            id_param = id_param,
            where_clause = where_clause,
            order_clause = order_clause,
        )
    } else {
        format!(
            r"
            SELECT
                {columns_str},
                formatDateTime(UUIDv7ToDateTime(id), '%Y-%m-%dT%H:%i:%SZ') AS timestamp
            FROM {table_name}
            WHERE {id_column} = {id_param} {where_clause}
            ORDER BY toUInt128(id) {order_clause}
            LIMIT {{page_size:UInt32}}
            FORMAT JSONEachRow
            ",
            columns_str = columns_str,
            table_name = config.table_name,
            id_column = config.id_column,
            id_param = id_param,
            where_clause = where_clause,
            order_clause = order_clause,
        )
    };

    (query, params_map)
}

pub(crate) fn build_boolean_metrics_query(
    target_id: Uuid,
    before: Option<Uuid>,
    after: Option<Uuid>,
    page_size: u32,
) -> (String, HashMap<String, String>) {
    build_feedback_query(
        &FeedbackTableConfig::BOOLEAN_METRICS,
        target_id,
        before,
        after,
        page_size,
    )
}

pub(crate) fn build_float_metrics_query(
    target_id: Uuid,
    before: Option<Uuid>,
    after: Option<Uuid>,
    page_size: u32,
) -> (String, HashMap<String, String>) {
    build_feedback_query(
        &FeedbackTableConfig::FLOAT_METRICS,
        target_id,
        before,
        after,
        page_size,
    )
}

pub(crate) fn build_comment_feedback_query(
    target_id: Uuid,
    before: Option<Uuid>,
    after: Option<Uuid>,
    page_size: u32,
) -> (String, HashMap<String, String>) {
    build_feedback_query(
        &FeedbackTableConfig::COMMENT_FEEDBACK,
        target_id,
        before,
        after,
        page_size,
    )
}

pub(crate) fn build_demonstration_feedback_query(
    inference_id: Uuid,
    before: Option<Uuid>,
    after: Option<Uuid>,
    page_size: u32,
) -> (String, HashMap<String, String>) {
    build_feedback_query(
        &FeedbackTableConfig::DEMONSTRATION_FEEDBACK,
        inference_id,
        before,
        after,
        page_size,
    )
}

pub(crate) fn build_bounds_query(
    table_name: &str,
    id_column: &str,
    target_id: Uuid,
) -> (String, HashMap<String, String>) {
    let mut params_map = HashMap::new();
    params_map.insert(id_column.to_string(), target_id.to_string());

    let query = format!(
        r"
        SELECT
            (SELECT id FROM {table_name} WHERE {id_column} = {{{id_column}:UUID}} ORDER BY toUInt128(id) ASC LIMIT 1) AS first_id,
            (SELECT id FROM {table_name} WHERE {id_column} = {{{id_column}:UUID}} ORDER BY toUInt128(id) DESC LIMIT 1) AS last_id
        FORMAT JSONEachRow
        "
    );

    (query, params_map)
}

pub(crate) fn build_count_query(
    table_name: &str,
    id_column: &str,
    target_id: Uuid,
) -> (String, HashMap<String, String>) {
    let mut params_map = HashMap::new();
    params_map.insert(id_column.to_string(), target_id.to_string());

    let query = format!(
        "SELECT toUInt64(COUNT()) AS count FROM {table_name} WHERE {id_column} = {{{id_column}:UUID}} FORMAT JSONEachRow"
    );

    (query, params_map)
}

// Generic query executor helper
async fn execute_feedback_query<T>(
    conn: &ClickHouseConnectionInfo,
    query: String,
    params_owned: HashMap<String, String>,
) -> Result<Vec<T>, Error>
where
    T: serde::de::DeserializeOwned,
{
    let query_params: HashMap<&str, &str> = params_owned
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();

    let response = conn.run_query_synchronous(query, &query_params).await?;
    parse_feedback_rows(response.response.as_str())
}

async fn execute_bounds_query(
    conn: &ClickHouseConnectionInfo,
    query: String,
    params_owned: HashMap<String, String>,
) -> Result<TableBounds, Error> {
    let query_params: HashMap<&str, &str> = params_owned
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();

    let response = conn.run_query_synchronous(query, &query_params).await?;
    parse_table_bounds(&response.response)
}

async fn execute_count_query(
    conn: &ClickHouseConnectionInfo,
    query: String,
    params_owned: HashMap<String, String>,
) -> Result<u64, Error> {
    let query_params: HashMap<&str, &str> = params_owned
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();

    let response = conn.run_query_synchronous(query, &query_params).await?;
    parse_count(&response.response)
}

// Helper implementations for individual feedback table queries
impl ClickHouseConnectionInfo {
    pub async fn query_boolean_metrics_by_target_id(
        &self,
        target_id: Uuid,
        before: Option<Uuid>,
        after: Option<Uuid>,
        page_size: u32,
    ) -> Result<Vec<BooleanMetricFeedbackRow>, Error> {
        let (query, params_owned) =
            build_boolean_metrics_query(target_id, before, after, page_size);
        execute_feedback_query(self, query, params_owned).await
    }

    pub async fn query_float_metrics_by_target_id(
        &self,
        target_id: Uuid,
        before: Option<Uuid>,
        after: Option<Uuid>,
        page_size: u32,
    ) -> Result<Vec<FloatMetricFeedbackRow>, Error> {
        let (query, params_owned) = build_float_metrics_query(target_id, before, after, page_size);
        execute_feedback_query(self, query, params_owned).await
    }

    pub async fn query_comment_feedback_by_target_id(
        &self,
        target_id: Uuid,
        before: Option<Uuid>,
        after: Option<Uuid>,
        page_size: u32,
    ) -> Result<Vec<CommentFeedbackRow>, Error> {
        let (query, params_owned) =
            build_comment_feedback_query(target_id, before, after, page_size);
        execute_feedback_query(self, query, params_owned).await
    }

    pub async fn query_boolean_metric_bounds_by_target_id(
        &self,
        target_id: Uuid,
    ) -> Result<TableBounds, Error> {
        let (query, params_owned) =
            build_bounds_query("BooleanMetricFeedbackByTargetId", "target_id", target_id);
        execute_bounds_query(self, query, params_owned).await
    }

    pub async fn query_float_metric_bounds_by_target_id(
        &self,
        target_id: Uuid,
    ) -> Result<TableBounds, Error> {
        let (query, params_owned) =
            build_bounds_query("FloatMetricFeedbackByTargetId", "target_id", target_id);
        execute_bounds_query(self, query, params_owned).await
    }

    pub async fn query_comment_feedback_bounds_by_target_id(
        &self,
        target_id: Uuid,
    ) -> Result<TableBounds, Error> {
        let (query, params_owned) =
            build_bounds_query("CommentFeedbackByTargetId", "target_id", target_id);
        execute_bounds_query(self, query, params_owned).await
    }

    pub async fn query_demonstration_feedback_bounds_by_inference_id(
        &self,
        inference_id: Uuid,
    ) -> Result<TableBounds, Error> {
        let (query, params_owned) = build_bounds_query(
            "DemonstrationFeedbackByInferenceId",
            "inference_id",
            inference_id,
        );
        execute_bounds_query(self, query, params_owned).await
    }

    pub async fn count_boolean_metrics_by_target_id(&self, target_id: Uuid) -> Result<u64, Error> {
        let (query, params_owned) =
            build_count_query("BooleanMetricFeedbackByTargetId", "target_id", target_id);
        execute_count_query(self, query, params_owned).await
    }

    pub async fn count_float_metrics_by_target_id(&self, target_id: Uuid) -> Result<u64, Error> {
        let (query, params_owned) =
            build_count_query("FloatMetricFeedbackByTargetId", "target_id", target_id);
        execute_count_query(self, query, params_owned).await
    }

    pub async fn count_comment_feedback_by_target_id(&self, target_id: Uuid) -> Result<u64, Error> {
        let (query, params_owned) =
            build_count_query("CommentFeedbackByTargetId", "target_id", target_id);
        execute_count_query(self, query, params_owned).await
    }

    pub async fn count_demonstration_feedback_by_inference_id(
        &self,
        inference_id: Uuid,
    ) -> Result<u64, Error> {
        let (query, params_owned) = build_count_query(
            "DemonstrationFeedbackByInferenceId",
            "inference_id",
            inference_id,
        );
        execute_count_query(self, query, params_owned).await
    }
}

// Implementation of FeedbackQueries trait
#[async_trait]
impl FeedbackQueries for ClickHouseConnectionInfo {
    async fn get_feedback_by_variant(
        &self,
        metric_name: &str,
        function_name: &str,
        variant_names: Option<&Vec<String>>,
    ) -> Result<Vec<FeedbackByVariant>, Error> {
        let escaped_function_name = escape_string_for_clickhouse_literal(function_name);
        let escaped_metric_name = escape_string_for_clickhouse_literal(metric_name);

        // If None we don't filter at all;
        // If empty, we'll return an empty vector for consistency
        // If there are variants passed, we'll filter by them
        let variant_filter = match variant_names {
            None => String::new(),
            Some(names) if names.is_empty() => {
                return Ok(vec![]);
            }
            Some(names) => {
                let escaped_names: Vec<String> = names
                    .iter()
                    .map(|name| format!("'{}'", escape_string_for_clickhouse_literal(name)))
                    .collect();
                format!(" AND variant_name IN ({})", escaped_names.join(", "))
            }
        };

        let query = format!(
            r"
            SELECT
                variant_name,
                avgMerge(feedback_mean) as mean,
                varSampStableMerge(feedback_variance) as variance,
                sum(count) as count
            FROM FeedbackByVariantStatistics
            WHERE function_name = '{escaped_function_name}' and metric_name = '{escaped_metric_name}'{variant_filter}
            GROUP BY variant_name
            FORMAT JSONEachRow"
        );
        // Each row is a JSON encoded FeedbackByVariant struct
        let res = self.run_query_synchronous_no_params(query).await?;

        res.response
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(serde_json::from_str)
            .collect::<Result<Vec<FeedbackByVariant>, _>>()
            .map_err(|e| {
                Error::new(crate::error::ErrorDetails::ClickHouseDeserialization {
                    message: format!("Failed to deserialize FeedbackByVariant: {e}"),
                })
            })
    }

    async fn query_feedback_by_target_id(
        &self,
        target_id: Uuid,
        before: Option<Uuid>,
        after: Option<Uuid>,
        page_size: Option<u32>,
    ) -> Result<Vec<FeedbackRow>, Error> {
        if before.is_some() && after.is_some() {
            return Err(Error::new(crate::error::ErrorDetails::InvalidRequest {
                message: "Cannot specify both before and after in query_feedback_by_target_id"
                    .to_string(),
            }));
        }

        let page_size = page_size.unwrap_or(100).min(100);

        // Query all 4 feedback tables in parallel
        let (boolean_metrics, float_metrics, comment_feedback, demonstration_feedback) = tokio::join!(
            self.query_boolean_metrics_by_target_id(target_id, before, after, page_size),
            self.query_float_metrics_by_target_id(target_id, before, after, page_size),
            self.query_comment_feedback_by_target_id(target_id, before, after, page_size),
            self.query_demonstration_feedback_by_inference_id(
                target_id,
                before,
                after,
                Some(page_size)
            )
        );

        // Combine all feedback types into a single vector
        let mut all_feedback: Vec<FeedbackRow> = Vec::new();
        all_feedback.extend(boolean_metrics?.into_iter().map(FeedbackRow::Boolean));
        all_feedback.extend(float_metrics?.into_iter().map(FeedbackRow::Float));
        all_feedback.extend(comment_feedback?.into_iter().map(FeedbackRow::Comment));
        all_feedback.extend(
            demonstration_feedback?
                .into_iter()
                .map(FeedbackRow::Demonstration),
        );

        // Sort by id in descending order (UUIDv7 comparison)
        all_feedback.sort_by(|a, b| {
            let id_a = match a {
                FeedbackRow::Boolean(f) => f.id,
                FeedbackRow::Float(f) => f.id,
                FeedbackRow::Comment(f) => f.id,
                FeedbackRow::Demonstration(f) => f.id,
            };
            let id_b = match b {
                FeedbackRow::Boolean(f) => f.id,
                FeedbackRow::Float(f) => f.id,
                FeedbackRow::Comment(f) => f.id,
                FeedbackRow::Demonstration(f) => f.id,
            };
            id_b.cmp(&id_a)
        });

        // Apply pagination
        let result = if after.is_some() {
            // If 'after' is specified, take earliest elements (reverse order from sorted)
            let start = all_feedback.len().saturating_sub(page_size as usize);
            all_feedback.drain(start..).collect()
        } else {
            // If 'before' is specified or no pagination params, take latest elements
            all_feedback.truncate(page_size as usize);
            all_feedback
        };

        Ok(result)
    }

    async fn query_feedback_bounds_by_target_id(
        &self,
        target_id: Uuid,
    ) -> Result<FeedbackBounds, Error> {
        let (boolean_bounds, float_bounds, comment_bounds, demonstration_bounds) = tokio::join!(
            self.query_boolean_metric_bounds_by_target_id(target_id),
            self.query_float_metric_bounds_by_target_id(target_id),
            self.query_comment_feedback_bounds_by_target_id(target_id),
            self.query_demonstration_feedback_bounds_by_inference_id(target_id)
        );

        let boolean_bounds = boolean_bounds?;
        let float_bounds = float_bounds?;
        let comment_bounds = comment_bounds?;
        let demonstration_bounds = demonstration_bounds?;

        // Find the earliest first_id and latest last_id across all feedback types
        let all_first_ids: Vec<Uuid> = [
            boolean_bounds.first_id,
            float_bounds.first_id,
            comment_bounds.first_id,
            demonstration_bounds.first_id,
        ]
        .into_iter()
        .flatten()
        .collect();

        let all_last_ids: Vec<Uuid> = [
            boolean_bounds.last_id,
            float_bounds.last_id,
            comment_bounds.last_id,
            demonstration_bounds.last_id,
        ]
        .into_iter()
        .flatten()
        .collect();

        let first_id = all_first_ids.into_iter().min();
        let last_id = all_last_ids.into_iter().max();

        Ok(FeedbackBounds {
            first_id,
            last_id,
            by_type: FeedbackBoundsByType {
                boolean: boolean_bounds,
                float: float_bounds,
                comment: comment_bounds,
                demonstration: demonstration_bounds,
            },
        })
    }

    async fn count_feedback_by_target_id(&self, target_id: Uuid) -> Result<u64, Error> {
        let (boolean_count, float_count, comment_count, demonstration_count) = tokio::join!(
            self.count_boolean_metrics_by_target_id(target_id),
            self.count_float_metrics_by_target_id(target_id),
            self.count_comment_feedback_by_target_id(target_id),
            self.count_demonstration_feedback_by_inference_id(target_id)
        );

        Ok(boolean_count? + float_count? + comment_count? + demonstration_count?)
    }

    async fn query_demonstration_feedback_by_inference_id(
        &self,
        inference_id: Uuid,
        before: Option<Uuid>,
        after: Option<Uuid>,
        page_size: Option<u32>,
    ) -> Result<Vec<DemonstrationFeedbackRow>, Error> {
        let page_size = page_size.unwrap_or(100).min(100);
        let (query, params_owned) =
            build_demonstration_feedback_query(inference_id, before, after, page_size);

        let query_params: HashMap<&str, &str> = params_owned
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        let response = self.run_query_synchronous(query, &query_params).await?;

        parse_feedback_rows(response.response.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Uuid;

    /// Normalize whitespace and newlines in a query for comparison
    fn normalize_whitespace(s: &str) -> String {
        s.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    /// Assert that the query contains a section (ignoring whitespace and newline differences)
    fn assert_query_contains(query: &str, expected_section: &str) {
        let normalized_query = normalize_whitespace(query);
        let normalized_section = normalize_whitespace(expected_section);
        assert!(
            normalized_query.contains(&normalized_section),
            "Query does not contain expected section.\nExpected section: {normalized_section}\nFull query: {normalized_query}"
        );
    }

    fn assert_query_does_not_contain(query: &str, unexpected_section: &str) {
        let normalized_query = normalize_whitespace(query);
        let normalized_section = normalize_whitespace(unexpected_section);
        assert!(
            !normalized_query.contains(&normalized_section),
            "Query contains unexpected section: {normalized_section}\nFull query: {normalized_query}"
        );
    }

    // Parameterized test helpers
    fn test_feedback_query_no_pagination(
        build_fn: impl Fn(Uuid, Option<Uuid>, Option<Uuid>, u32) -> (String, HashMap<String, String>),
        table_name: &str,
        id_column: &str,
        expected_columns: &str,
    ) {
        let id = Uuid::now_v7();
        let (query, params) = build_fn(id, None, None, 100);

        assert_query_contains(&query, &format!("SELECT {expected_columns}"));
        assert_query_contains(&query, &format!("FROM {table_name}"));
        assert_query_contains(&query, &format!("WHERE {id_column} = {{{id_column}:UUID}}"));
        assert_query_contains(&query, "ORDER BY toUInt128(id) DESC");
        assert_query_contains(&query, "LIMIT {page_size:UInt32}");
        assert_query_does_not_contain(&query, "AND toUInt128(id)");
        assert_eq!(params.get(id_column), Some(&id.to_string()));
        assert_eq!(params.get("page_size"), Some(&"100".to_string()));
        assert!(!params.contains_key("before"));
        assert!(!params.contains_key("after"));
    }

    fn test_feedback_query_before_pagination(
        build_fn: impl Fn(Uuid, Option<Uuid>, Option<Uuid>, u32) -> (String, HashMap<String, String>),
        id_column: &str,
    ) {
        let id = Uuid::now_v7();
        let before = Uuid::now_v7();
        let (query, params) = build_fn(id, Some(before), None, 50);

        assert_query_contains(&query, &format!("WHERE {id_column} = {{{id_column}:UUID}}"));
        assert_query_contains(
            &query,
            "AND toUInt128(id) < toUInt128(toUUID({before:UUID}))",
        );
        assert_query_contains(&query, "ORDER BY toUInt128(id) DESC");
        assert_eq!(params.get(id_column), Some(&id.to_string()));
        assert_eq!(params.get("before"), Some(&before.to_string()));
        assert_eq!(params.get("page_size"), Some(&"50".to_string()));
    }

    fn test_feedback_query_after_pagination(
        build_fn: impl Fn(Uuid, Option<Uuid>, Option<Uuid>, u32) -> (String, HashMap<String, String>),
        table_name: &str,
        id_column: &str,
    ) {
        let id = Uuid::now_v7();
        let after = Uuid::now_v7();
        let (query, params) = build_fn(id, None, Some(after), 25);

        assert_query_contains(
            &query,
            &format!("FROM {table_name} WHERE {id_column} = {{{id_column}:UUID}}"),
        );
        assert_query_contains(
            &query,
            "AND toUInt128(id) > toUInt128(toUUID({after:UUID}))",
        );
        assert_query_contains(
            &query,
            "ORDER BY toUInt128(id) ASC LIMIT {page_size:UInt32} )",
        );
        assert_query_contains(&query, ") ORDER BY toUInt128(id) DESC");
        assert_eq!(params.get(id_column), Some(&id.to_string()));
        assert_eq!(params.get("after"), Some(&after.to_string()));
        assert_eq!(params.get("page_size"), Some(&"25".to_string()));
    }

    fn test_bounds_query(table_name: &str, id_column: &str) {
        let id = Uuid::now_v7();
        let (query, params) = build_bounds_query(table_name, id_column, id);

        assert_query_contains(&query, &format!("(SELECT id FROM {table_name} WHERE {id_column} = {{{id_column}:UUID}} ORDER BY toUInt128(id) ASC LIMIT 1) AS first_id"));
        assert_query_contains(&query, &format!("(SELECT id FROM {table_name} WHERE {id_column} = {{{id_column}:UUID}} ORDER BY toUInt128(id) DESC LIMIT 1) AS last_id"));
        assert_query_contains(&query, "FORMAT JSONEachRow");
        assert_eq!(params.get(id_column), Some(&id.to_string()));
    }

    fn test_count_query(table_name: &str, id_column: &str) {
        let id = Uuid::now_v7();
        let (query, params) = build_count_query(table_name, id_column, id);

        assert_query_contains(&query, "SELECT toUInt64(COUNT()) AS count");
        assert_query_contains(&query, &format!("FROM {table_name}"));
        assert_query_contains(&query, &format!("WHERE {id_column} = {{{id_column}:UUID}}"));
        assert_query_contains(&query, "FORMAT JSONEachRow");
        assert_eq!(params.get(id_column), Some(&id.to_string()));
    }

    // Boolean Metric Feedback Tests
    #[test]
    fn test_build_boolean_metrics_query_no_pagination() {
        test_feedback_query_no_pagination(
            build_boolean_metrics_query,
            "BooleanMetricFeedbackByTargetId",
            "target_id",
            "id, target_id, metric_name, value, tags",
        );
    }

    #[test]
    fn test_build_boolean_metrics_query_before_pagination() {
        test_feedback_query_before_pagination(build_boolean_metrics_query, "target_id");
    }

    #[test]
    fn test_build_boolean_metrics_query_after_pagination() {
        test_feedback_query_after_pagination(
            build_boolean_metrics_query,
            "BooleanMetricFeedbackByTargetId",
            "target_id",
        );
    }

    // Float Metric Feedback Tests
    #[test]
    fn test_build_float_metrics_query_no_pagination() {
        test_feedback_query_no_pagination(
            build_float_metrics_query,
            "FloatMetricFeedbackByTargetId",
            "target_id",
            "id, target_id, metric_name, value, tags",
        );
    }

    #[test]
    fn test_build_float_metrics_query_before_pagination() {
        test_feedback_query_before_pagination(build_float_metrics_query, "target_id");
    }

    #[test]
    fn test_build_float_metrics_query_after_pagination() {
        test_feedback_query_after_pagination(
            build_float_metrics_query,
            "FloatMetricFeedbackByTargetId",
            "target_id",
        );
    }

    // Comment Feedback Tests
    #[test]
    fn test_build_comment_feedback_query_no_pagination() {
        test_feedback_query_no_pagination(
            build_comment_feedback_query,
            "CommentFeedbackByTargetId",
            "target_id",
            "id, target_id, target_type, value, tags",
        );
    }

    #[test]
    fn test_build_comment_feedback_query_before_pagination() {
        test_feedback_query_before_pagination(build_comment_feedback_query, "target_id");
    }

    #[test]
    fn test_build_comment_feedback_query_after_pagination() {
        test_feedback_query_after_pagination(
            build_comment_feedback_query,
            "CommentFeedbackByTargetId",
            "target_id",
        );
    }

    // Demonstration Feedback Tests
    #[test]
    fn test_build_demonstration_feedback_query_no_pagination() {
        test_feedback_query_no_pagination(
            build_demonstration_feedback_query,
            "DemonstrationFeedbackByInferenceId",
            "inference_id",
            "id, inference_id, value, tags",
        );
    }

    #[test]
    fn test_build_demonstration_feedback_query_before_pagination() {
        test_feedback_query_before_pagination(build_demonstration_feedback_query, "inference_id");
    }

    #[test]
    fn test_build_demonstration_feedback_query_after_pagination() {
        test_feedback_query_after_pagination(
            build_demonstration_feedback_query,
            "DemonstrationFeedbackByInferenceId",
            "inference_id",
        );
    }

    // Bounds Query Tests
    #[test]
    fn test_build_bounds_query_boolean_metrics() {
        test_bounds_query("BooleanMetricFeedbackByTargetId", "target_id");
    }

    #[test]
    fn test_build_bounds_query_float_metrics() {
        test_bounds_query("FloatMetricFeedbackByTargetId", "target_id");
    }

    #[test]
    fn test_build_bounds_query_comment_feedback() {
        test_bounds_query("CommentFeedbackByTargetId", "target_id");
    }

    #[test]
    fn test_build_bounds_query_demonstration_feedback() {
        test_bounds_query("DemonstrationFeedbackByInferenceId", "inference_id");
    }

    // Count Query Tests
    #[test]
    fn test_build_count_query_boolean_metrics() {
        test_count_query("BooleanMetricFeedbackByTargetId", "target_id");
    }

    #[test]
    fn test_build_count_query_float_metrics() {
        test_count_query("FloatMetricFeedbackByTargetId", "target_id");
    }

    #[test]
    fn test_build_count_query_comment_feedback() {
        test_count_query("CommentFeedbackByTargetId", "target_id");
    }

    #[test]
    fn test_build_count_query_demonstration_feedback() {
        test_count_query("DemonstrationFeedbackByInferenceId", "inference_id");
    }

    // Test that 'type' field is not in queries (serde handles it) - testing production code
    #[test]
    fn test_no_type_literal_in_production_queries() {
        let target_id = Uuid::now_v7();

        // Test boolean metrics query
        let (boolean_query, _) = build_boolean_metrics_query(target_id, None, None, 100);
        assert_query_does_not_contain(&boolean_query, "'boolean' AS type");

        // Test float metrics query
        let (float_query, _) = build_float_metrics_query(target_id, None, None, 100);
        assert_query_does_not_contain(&float_query, "'float' AS type");

        // Test comment feedback query
        let (comment_query, _) = build_comment_feedback_query(target_id, None, None, 100);
        assert_query_does_not_contain(&comment_query, "'comment' AS type");
    }
}
