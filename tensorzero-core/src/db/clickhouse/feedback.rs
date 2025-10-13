use uuid::Uuid;

use crate::{
    db::{
        feedback::{BooleanMetricFeedbackRow, CommentFeedbackRow, FloatMetricFeedbackRow},
        TableBounds,
    },
    error::Error,
};

use super::{
    select_queries::{
        build_pagination_clause, parse_count, parse_feedback_rows, parse_table_bounds,
    },
    ClickHouseConnectionInfo,
};

// Query builder functions - these are testable production code
pub(crate) fn build_boolean_metrics_query(
    target_id: uuid::Uuid,
    before: Option<uuid::Uuid>,
    after: Option<uuid::Uuid>,
    page_size: u32,
) -> (String, std::collections::HashMap<String, String>) {
    let (where_clause, params) = build_pagination_clause(before, after, "target_id");
    let order_clause = if after.is_some() { "ASC" } else { "DESC" };

    let mut params_map: std::collections::HashMap<String, String> = params
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
    params_map.insert("target_id".to_string(), target_id.to_string());
    params_map.insert("page_size".to_string(), page_size.to_string());

    let query = if after.is_some() {
        format!(
            r"
            SELECT
                id,
                target_id,
                metric_name,
                value,
                tags,
                timestamp
            FROM (
                SELECT
                    id,
                    target_id,
                    metric_name,
                    value,
                    tags,
                    formatDateTime(UUIDv7ToDateTime(id), '%Y-%m-%dT%H:%i:%SZ') AS timestamp
                FROM BooleanMetricFeedbackByTargetId
                WHERE target_id = {{target_id:UUID}} {where_clause}
                ORDER BY toUInt128(id) {order_clause}
                LIMIT {{page_size:UInt32}}
            )
            ORDER BY toUInt128(id) DESC
            FORMAT JSONEachRow
            "
        )
    } else {
        format!(
            r"
            SELECT
                id,
                target_id,
                metric_name,
                value,
                tags,
                formatDateTime(UUIDv7ToDateTime(id), '%Y-%m-%dT%H:%i:%SZ') AS timestamp
            FROM BooleanMetricFeedbackByTargetId
            WHERE target_id = {{target_id:UUID}} {where_clause}
            ORDER BY toUInt128(id) {order_clause}
            LIMIT {{page_size:UInt32}}
            FORMAT JSONEachRow
            "
        )
    };

    (query, params_map)
}

pub(crate) fn build_float_metrics_query(
    target_id: uuid::Uuid,
    before: Option<uuid::Uuid>,
    after: Option<uuid::Uuid>,
    page_size: u32,
) -> (String, std::collections::HashMap<String, String>) {
    let (where_clause, params) = build_pagination_clause(before, after, "target_id");
    let order_clause = if after.is_some() { "ASC" } else { "DESC" };

    let mut params_map: std::collections::HashMap<String, String> = params
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
    params_map.insert("target_id".to_string(), target_id.to_string());
    params_map.insert("page_size".to_string(), page_size.to_string());

    let query = if after.is_some() {
        format!(
            r"
            SELECT
                id,
                target_id,
                metric_name,
                value,
                tags,
                timestamp
            FROM (
                SELECT
                    id,
                    target_id,
                    metric_name,
                    value,
                    tags,
                    formatDateTime(UUIDv7ToDateTime(id), '%Y-%m-%dT%H:%i:%SZ') AS timestamp
                FROM FloatMetricFeedbackByTargetId
                WHERE target_id = {{target_id:UUID}} {where_clause}
                ORDER BY toUInt128(id) {order_clause}
                LIMIT {{page_size:UInt32}}
            )
            ORDER BY toUInt128(id) DESC
            FORMAT JSONEachRow
            "
        )
    } else {
        format!(
            r"
            SELECT
                id,
                target_id,
                metric_name,
                value,
                tags,
                formatDateTime(UUIDv7ToDateTime(id), '%Y-%m-%dT%H:%i:%SZ') AS timestamp
            FROM FloatMetricFeedbackByTargetId
            WHERE target_id = {{target_id:UUID}} {where_clause}
            ORDER BY toUInt128(id) {order_clause}
            LIMIT {{page_size:UInt32}}
            FORMAT JSONEachRow
            "
        )
    };

    (query, params_map)
}

pub(crate) fn build_comment_feedback_query(
    target_id: uuid::Uuid,
    before: Option<uuid::Uuid>,
    after: Option<uuid::Uuid>,
    page_size: u32,
) -> (String, std::collections::HashMap<String, String>) {
    let (where_clause, params) = build_pagination_clause(before, after, "target_id");
    let order_clause = if after.is_some() { "ASC" } else { "DESC" };

    let mut params_map: std::collections::HashMap<String, String> = params
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
    params_map.insert("target_id".to_string(), target_id.to_string());
    params_map.insert("page_size".to_string(), page_size.to_string());

    let query = if after.is_some() {
        format!(
            r"
            SELECT
                id,
                target_id,
                target_type,
                value,
                tags,
                timestamp
            FROM (
                SELECT
                    id,
                    target_id,
                    target_type,
                    value,
                    tags,
                    formatDateTime(UUIDv7ToDateTime(id), '%Y-%m-%dT%H:%i:%SZ') AS timestamp
                FROM CommentFeedbackByTargetId
                WHERE target_id = {{target_id:UUID}} {where_clause}
                ORDER BY toUInt128(id) {order_clause}
                LIMIT {{page_size:UInt32}}
            )
            ORDER BY toUInt128(id) DESC
            FORMAT JSONEachRow
            "
        )
    } else {
        format!(
            r"
            SELECT
                id,
                target_id,
                target_type,
                value,
                tags,
                formatDateTime(UUIDv7ToDateTime(id), '%Y-%m-%dT%H:%i:%SZ') AS timestamp
            FROM CommentFeedbackByTargetId
            WHERE target_id = {{target_id:UUID}} {where_clause}
            ORDER BY toUInt128(id) {order_clause}
            LIMIT {{page_size:UInt32}}
            FORMAT JSONEachRow
            "
        )
    };

    (query, params_map)
}

pub(crate) fn build_demonstration_feedback_query(
    inference_id: uuid::Uuid,
    before: Option<uuid::Uuid>,
    after: Option<uuid::Uuid>,
    page_size: u32,
) -> (String, std::collections::HashMap<String, String>) {
    let (where_clause, params) = build_pagination_clause(before, after, "inference_id");
    let order_clause = if after.is_some() { "ASC" } else { "DESC" };

    let mut params_map: std::collections::HashMap<String, String> = params
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
    params_map.insert("inference_id".to_string(), inference_id.to_string());
    params_map.insert("page_size".to_string(), page_size.to_string());

    let query = if after.is_some() {
        format!(
            r"
            SELECT
                id,
                inference_id,
                value,
                tags,
                timestamp
            FROM (
                SELECT
                    id,
                    inference_id,
                    value,
                    tags,
                    formatDateTime(UUIDv7ToDateTime(id), '%Y-%m-%dT%H:%i:%SZ') AS timestamp
                FROM DemonstrationFeedbackByInferenceId
                WHERE inference_id = {{inference_id:UUID}} {where_clause}
                ORDER BY toUInt128(id) {order_clause}
                LIMIT {{page_size:UInt32}}
            )
            ORDER BY toUInt128(id) DESC
            FORMAT JSONEachRow
            "
        )
    } else {
        format!(
            r"
            SELECT
                id,
                inference_id,
                value,
                tags,
                formatDateTime(UUIDv7ToDateTime(id), '%Y-%m-%dT%H:%i:%SZ') AS timestamp
            FROM DemonstrationFeedbackByInferenceId
            WHERE inference_id = {{inference_id:UUID}} {where_clause}
            ORDER BY toUInt128(id) {order_clause}
            LIMIT {{page_size:UInt32}}
            FORMAT JSONEachRow
            "
        )
    };

    (query, params_map)
}

pub(crate) fn build_bounds_query(
    table_name: &str,
    id_column: &str,
    target_id: uuid::Uuid,
) -> (String, std::collections::HashMap<String, String>) {
    let mut params_map = std::collections::HashMap::new();
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
    target_id: uuid::Uuid,
) -> (String, std::collections::HashMap<String, String>) {
    let mut params_map = std::collections::HashMap::new();
    params_map.insert(id_column.to_string(), target_id.to_string());

    let query = format!(
        "SELECT toUInt64(COUNT()) AS count FROM {table_name} WHERE {id_column} = {{{id_column}:UUID}} FORMAT JSONEachRow"
    );

    (query, params_map)
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

        let query_params: std::collections::HashMap<&str, &str> = params_owned
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        let response = self.run_query_synchronous(query, &query_params).await?;

        parse_feedback_rows(response.response.as_str())
    }

    pub async fn query_float_metrics_by_target_id(
        &self,
        target_id: Uuid,
        before: Option<Uuid>,
        after: Option<Uuid>,
        page_size: u32,
    ) -> Result<Vec<FloatMetricFeedbackRow>, Error> {
        let (query, params_owned) = build_float_metrics_query(target_id, before, after, page_size);

        let query_params: std::collections::HashMap<&str, &str> = params_owned
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        let response = self.run_query_synchronous(query, &query_params).await?;

        parse_feedback_rows(response.response.as_str())
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

        let query_params: std::collections::HashMap<&str, &str> = params_owned
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        let response = self.run_query_synchronous(query, &query_params).await?;

        parse_feedback_rows(response.response.as_str())
    }

    pub async fn query_boolean_metric_bounds_by_target_id(
        &self,
        target_id: Uuid,
    ) -> Result<TableBounds, Error> {
        let (query, params_owned) =
            build_bounds_query("BooleanMetricFeedbackByTargetId", "target_id", target_id);

        let query_params: std::collections::HashMap<&str, &str> = params_owned
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        let response = self.run_query_synchronous(query, &query_params).await?;
        parse_table_bounds(&response.response)
    }

    pub async fn query_float_metric_bounds_by_target_id(
        &self,
        target_id: Uuid,
    ) -> Result<TableBounds, Error> {
        let (query, params_owned) =
            build_bounds_query("FloatMetricFeedbackByTargetId", "target_id", target_id);

        let query_params: std::collections::HashMap<&str, &str> = params_owned
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        let response = self.run_query_synchronous(query, &query_params).await?;
        parse_table_bounds(&response.response)
    }

    pub async fn query_comment_feedback_bounds_by_target_id(
        &self,
        target_id: Uuid,
    ) -> Result<TableBounds, Error> {
        let (query, params_owned) =
            build_bounds_query("CommentFeedbackByTargetId", "target_id", target_id);

        let query_params: std::collections::HashMap<&str, &str> = params_owned
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        let response = self.run_query_synchronous(query, &query_params).await?;
        parse_table_bounds(&response.response)
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

        let query_params: std::collections::HashMap<&str, &str> = params_owned
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        let response = self.run_query_synchronous(query, &query_params).await?;
        parse_table_bounds(&response.response)
    }

    pub async fn count_boolean_metrics_by_target_id(&self, target_id: Uuid) -> Result<u64, Error> {
        let (query, params_owned) =
            build_count_query("BooleanMetricFeedbackByTargetId", "target_id", target_id);

        let query_params: std::collections::HashMap<&str, &str> = params_owned
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        let response = self.run_query_synchronous(query, &query_params).await?;
        parse_count(&response.response)
    }

    pub async fn count_float_metrics_by_target_id(&self, target_id: Uuid) -> Result<u64, Error> {
        let (query, params_owned) =
            build_count_query("FloatMetricFeedbackByTargetId", "target_id", target_id);

        let query_params: std::collections::HashMap<&str, &str> = params_owned
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        let response = self.run_query_synchronous(query, &query_params).await?;
        parse_count(&response.response)
    }

    pub async fn count_comment_feedback_by_target_id(&self, target_id: Uuid) -> Result<u64, Error> {
        let (query, params_owned) =
            build_count_query("CommentFeedbackByTargetId", "target_id", target_id);

        let query_params: std::collections::HashMap<&str, &str> = params_owned
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        let response = self.run_query_synchronous(query, &query_params).await?;
        parse_count(&response.response)
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

        let query_params: std::collections::HashMap<&str, &str> = params_owned
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        let response = self.run_query_synchronous(query, &query_params).await?;
        parse_count(&response.response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

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

    // Boolean Metric Feedback Tests - testing production query builder
    #[test]
    fn test_build_boolean_metrics_query_no_pagination() {
        let target_id = Uuid::now_v7();
        let (query, params) = build_boolean_metrics_query(target_id, None, None, 100);

        assert_query_contains(&query, "SELECT id, target_id, metric_name, value, tags");
        assert_query_contains(&query, "FROM BooleanMetricFeedbackByTargetId");
        assert_query_contains(&query, "WHERE target_id = {target_id:UUID}");
        assert_query_contains(&query, "ORDER BY toUInt128(id) DESC");
        assert_query_contains(&query, "LIMIT {page_size:UInt32}");
        assert_query_does_not_contain(&query, "AND toUInt128(id)"); // No pagination clause
        assert_eq!(params.get("target_id"), Some(&target_id.to_string()));
        assert_eq!(params.get("page_size"), Some(&"100".to_string()));
        assert!(!params.contains_key("before"));
        assert!(!params.contains_key("after"));
    }

    #[test]
    fn test_build_boolean_metrics_query_before_pagination() {
        let target_id = Uuid::now_v7();
        let before = Uuid::now_v7();
        let (query, params) = build_boolean_metrics_query(target_id, Some(before), None, 50);

        assert_query_contains(&query, "WHERE target_id = {target_id:UUID}");
        assert_query_contains(
            &query,
            "AND toUInt128(id) < toUInt128(toUUID({before:UUID}))",
        );
        assert_query_contains(&query, "ORDER BY toUInt128(id) DESC");
        assert_eq!(params.get("target_id"), Some(&target_id.to_string()));
        assert_eq!(params.get("before"), Some(&before.to_string()));
        assert_eq!(params.get("page_size"), Some(&"50".to_string()));
    }

    #[test]
    fn test_build_boolean_metrics_query_after_pagination() {
        let target_id = Uuid::now_v7();
        let after = Uuid::now_v7();
        let (query, params) = build_boolean_metrics_query(target_id, None, Some(after), 25);

        // Inner subquery should have ASC ordering
        assert_query_contains(
            &query,
            "FROM BooleanMetricFeedbackByTargetId WHERE target_id = {target_id:UUID}",
        );
        assert_query_contains(
            &query,
            "AND toUInt128(id) > toUInt128(toUUID({after:UUID}))",
        );
        assert_query_contains(
            &query,
            "ORDER BY toUInt128(id) ASC LIMIT {page_size:UInt32} )",
        );

        // Outer query should reverse to DESC
        assert_query_contains(&query, ") ORDER BY toUInt128(id) DESC");

        assert_eq!(params.get("target_id"), Some(&target_id.to_string()));
        assert_eq!(params.get("after"), Some(&after.to_string()));
        assert_eq!(params.get("page_size"), Some(&"25".to_string()));
    }

    // Float Metric Feedback Tests - testing production query builder
    #[test]
    fn test_build_float_metrics_query_no_pagination() {
        let target_id = Uuid::now_v7();
        let (query, params) = build_float_metrics_query(target_id, None, None, 100);

        assert_query_contains(&query, "SELECT id, target_id, metric_name, value, tags");
        assert_query_contains(&query, "FROM FloatMetricFeedbackByTargetId");
        assert_query_contains(&query, "WHERE target_id = {target_id:UUID}");
        assert_query_contains(&query, "ORDER BY toUInt128(id) DESC");
        assert_query_contains(&query, "LIMIT {page_size:UInt32}");
        assert_query_does_not_contain(&query, "AND toUInt128(id)"); // No pagination clause
        assert_eq!(params.get("target_id"), Some(&target_id.to_string()));
        assert_eq!(params.get("page_size"), Some(&"100".to_string()));
        assert!(!params.contains_key("before"));
        assert!(!params.contains_key("after"));
    }

    #[test]
    fn test_build_float_metrics_query_before_pagination() {
        let target_id = Uuid::now_v7();
        let before = Uuid::now_v7();
        let (query, params) = build_float_metrics_query(target_id, Some(before), None, 50);

        assert_query_contains(&query, "WHERE target_id = {target_id:UUID}");
        assert_query_contains(
            &query,
            "AND toUInt128(id) < toUInt128(toUUID({before:UUID}))",
        );
        assert_query_contains(&query, "ORDER BY toUInt128(id) DESC");
        assert_eq!(params.get("target_id"), Some(&target_id.to_string()));
        assert_eq!(params.get("before"), Some(&before.to_string()));
        assert_eq!(params.get("page_size"), Some(&"50".to_string()));
    }

    #[test]
    fn test_build_float_metrics_query_after_pagination() {
        let target_id = Uuid::now_v7();
        let after = Uuid::now_v7();
        let (query, params) = build_float_metrics_query(target_id, None, Some(after), 25);

        // Inner subquery should have ASC ordering
        assert_query_contains(
            &query,
            "FROM FloatMetricFeedbackByTargetId WHERE target_id = {target_id:UUID}",
        );
        assert_query_contains(
            &query,
            "AND toUInt128(id) > toUInt128(toUUID({after:UUID}))",
        );
        assert_query_contains(
            &query,
            "ORDER BY toUInt128(id) ASC LIMIT {page_size:UInt32} )",
        );

        // Outer query should reverse to DESC
        assert_query_contains(&query, ") ORDER BY toUInt128(id) DESC");

        assert_eq!(params.get("target_id"), Some(&target_id.to_string()));
        assert_eq!(params.get("after"), Some(&after.to_string()));
        assert_eq!(params.get("page_size"), Some(&"25".to_string()));
    }

    // Comment Feedback Tests - testing production query builder
    #[test]
    fn test_build_comment_feedback_query_no_pagination() {
        let target_id = Uuid::now_v7();
        let (query, params) = build_comment_feedback_query(target_id, None, None, 100);

        assert_query_contains(&query, "SELECT id, target_id, target_type, value, tags");
        assert_query_contains(&query, "FROM CommentFeedbackByTargetId");
        assert_query_contains(&query, "WHERE target_id = {target_id:UUID}");
        assert_query_contains(&query, "ORDER BY toUInt128(id) DESC");
        assert_query_contains(&query, "LIMIT {page_size:UInt32}");
        assert_query_does_not_contain(&query, "AND toUInt128(id)"); // No pagination clause
        assert_eq!(params.get("target_id"), Some(&target_id.to_string()));
        assert_eq!(params.get("page_size"), Some(&"100".to_string()));
        assert!(!params.contains_key("before"));
        assert!(!params.contains_key("after"));
    }

    #[test]
    fn test_build_comment_feedback_query_before_pagination() {
        let target_id = Uuid::now_v7();
        let before = Uuid::now_v7();
        let (query, params) = build_comment_feedback_query(target_id, Some(before), None, 50);

        assert_query_contains(&query, "WHERE target_id = {target_id:UUID}");
        assert_query_contains(
            &query,
            "AND toUInt128(id) < toUInt128(toUUID({before:UUID}))",
        );
        assert_query_contains(&query, "ORDER BY toUInt128(id) DESC");
        assert_eq!(params.get("target_id"), Some(&target_id.to_string()));
        assert_eq!(params.get("before"), Some(&before.to_string()));
        assert_eq!(params.get("page_size"), Some(&"50".to_string()));
    }

    #[test]
    fn test_build_comment_feedback_query_after_pagination() {
        let target_id = Uuid::now_v7();
        let after = Uuid::now_v7();
        let (query, params) = build_comment_feedback_query(target_id, None, Some(after), 25);

        // Inner subquery should have ASC ordering
        assert_query_contains(
            &query,
            "FROM CommentFeedbackByTargetId WHERE target_id = {target_id:UUID}",
        );
        assert_query_contains(
            &query,
            "AND toUInt128(id) > toUInt128(toUUID({after:UUID}))",
        );
        assert_query_contains(
            &query,
            "ORDER BY toUInt128(id) ASC LIMIT {page_size:UInt32} )",
        );

        // Outer query should reverse to DESC
        assert_query_contains(&query, ") ORDER BY toUInt128(id) DESC");

        assert_eq!(params.get("target_id"), Some(&target_id.to_string()));
        assert_eq!(params.get("after"), Some(&after.to_string()));
        assert_eq!(params.get("page_size"), Some(&"25".to_string()));
    }

    // Demonstration Feedback Tests - testing production query builder
    #[test]
    fn test_build_demonstration_feedback_query_no_pagination() {
        let inference_id = Uuid::now_v7();
        let (query, params) = build_demonstration_feedback_query(inference_id, None, None, 100);

        assert_query_contains(&query, "SELECT id, inference_id, value, tags");
        assert_query_contains(&query, "FROM DemonstrationFeedbackByInferenceId");
        assert_query_contains(&query, "WHERE inference_id = {inference_id:UUID}");
        assert_query_contains(&query, "ORDER BY toUInt128(id) DESC");
        assert_query_contains(&query, "LIMIT {page_size:UInt32}");
        assert_query_does_not_contain(&query, "AND toUInt128(id)"); // No pagination clause
        assert_eq!(params.get("inference_id"), Some(&inference_id.to_string()));
        assert_eq!(params.get("page_size"), Some(&"100".to_string()));
        assert!(!params.contains_key("before"));
        assert!(!params.contains_key("after"));
    }

    #[test]
    fn test_build_demonstration_feedback_query_before_pagination() {
        let inference_id = Uuid::now_v7();
        let before = Uuid::now_v7();
        let (query, params) =
            build_demonstration_feedback_query(inference_id, Some(before), None, 50);

        assert_query_contains(&query, "WHERE inference_id = {inference_id:UUID}");
        assert_query_contains(
            &query,
            "AND toUInt128(id) < toUInt128(toUUID({before:UUID}))",
        );
        assert_query_contains(&query, "ORDER BY toUInt128(id) DESC");
        assert_eq!(params.get("inference_id"), Some(&inference_id.to_string()));
        assert_eq!(params.get("before"), Some(&before.to_string()));
        assert_eq!(params.get("page_size"), Some(&"50".to_string()));
    }

    #[test]
    fn test_build_demonstration_feedback_query_after_pagination() {
        let inference_id = Uuid::now_v7();
        let after = Uuid::now_v7();
        let (query, params) =
            build_demonstration_feedback_query(inference_id, None, Some(after), 25);

        // Inner subquery should have ASC ordering
        assert_query_contains(
            &query,
            "FROM DemonstrationFeedbackByInferenceId WHERE inference_id = {inference_id:UUID}",
        );
        assert_query_contains(
            &query,
            "AND toUInt128(id) > toUInt128(toUUID({after:UUID}))",
        );
        assert_query_contains(
            &query,
            "ORDER BY toUInt128(id) ASC LIMIT {page_size:UInt32} )",
        );

        // Outer query should reverse to DESC
        assert_query_contains(&query, ") ORDER BY toUInt128(id) DESC");

        assert_eq!(params.get("inference_id"), Some(&inference_id.to_string()));
        assert_eq!(params.get("after"), Some(&after.to_string()));
        assert_eq!(params.get("page_size"), Some(&"25".to_string()));
    }

    // Bounds Query Tests - testing production query builder
    #[test]
    fn test_build_bounds_query_boolean_metrics() {
        let target_id = Uuid::now_v7();
        let (query, params) =
            build_bounds_query("BooleanMetricFeedbackByTargetId", "target_id", target_id);

        assert_query_contains(&query, "(SELECT id FROM BooleanMetricFeedbackByTargetId WHERE target_id = {target_id:UUID} ORDER BY toUInt128(id) ASC LIMIT 1) AS first_id");
        assert_query_contains(&query, "(SELECT id FROM BooleanMetricFeedbackByTargetId WHERE target_id = {target_id:UUID} ORDER BY toUInt128(id) DESC LIMIT 1) AS last_id");
        assert_query_contains(&query, "FORMAT JSONEachRow");
        assert_eq!(params.get("target_id"), Some(&target_id.to_string()));
    }

    #[test]
    fn test_build_bounds_query_float_metrics() {
        let target_id = Uuid::now_v7();
        let (query, params) =
            build_bounds_query("FloatMetricFeedbackByTargetId", "target_id", target_id);

        assert_query_contains(&query, "(SELECT id FROM FloatMetricFeedbackByTargetId");
        assert_query_contains(&query, "WHERE target_id = {target_id:UUID}");
        assert_eq!(params.get("target_id"), Some(&target_id.to_string()));
    }

    #[test]
    fn test_build_bounds_query_comment_feedback() {
        let target_id = Uuid::now_v7();
        let (query, params) =
            build_bounds_query("CommentFeedbackByTargetId", "target_id", target_id);

        assert_query_contains(&query, "(SELECT id FROM CommentFeedbackByTargetId");
        assert_query_contains(&query, "WHERE target_id = {target_id:UUID}");
        assert_eq!(params.get("target_id"), Some(&target_id.to_string()));
    }

    #[test]
    fn test_build_bounds_query_demonstration_feedback() {
        let inference_id = Uuid::now_v7();
        let (query, params) = build_bounds_query(
            "DemonstrationFeedbackByInferenceId",
            "inference_id",
            inference_id,
        );

        assert_query_contains(&query, "(SELECT id FROM DemonstrationFeedbackByInferenceId WHERE inference_id = {inference_id:UUID} ORDER BY toUInt128(id) ASC LIMIT 1) AS first_id");
        assert_query_contains(&query, "(SELECT id FROM DemonstrationFeedbackByInferenceId WHERE inference_id = {inference_id:UUID} ORDER BY toUInt128(id) DESC LIMIT 1) AS last_id");
        assert_query_contains(&query, "FORMAT JSONEachRow");
        assert_eq!(params.get("inference_id"), Some(&inference_id.to_string()));
    }

    // Count Query Tests - testing production query builder
    #[test]
    fn test_build_count_query_boolean_metrics() {
        let target_id = Uuid::now_v7();
        let (query, params) =
            build_count_query("BooleanMetricFeedbackByTargetId", "target_id", target_id);

        assert_query_contains(&query, "SELECT toUInt64(COUNT()) AS count");
        assert_query_contains(&query, "FROM BooleanMetricFeedbackByTargetId");
        assert_query_contains(&query, "WHERE target_id = {target_id:UUID}");
        assert_query_contains(&query, "FORMAT JSONEachRow");
        assert_eq!(params.get("target_id"), Some(&target_id.to_string()));
    }

    #[test]
    fn test_build_count_query_float_metrics() {
        let target_id = Uuid::now_v7();
        let (query, params) =
            build_count_query("FloatMetricFeedbackByTargetId", "target_id", target_id);

        assert_query_contains(&query, "SELECT toUInt64(COUNT()) AS count");
        assert_query_contains(&query, "FROM FloatMetricFeedbackByTargetId");
        assert_query_contains(&query, "WHERE target_id = {target_id:UUID}");
        assert_query_contains(&query, "FORMAT JSONEachRow");
        assert_eq!(params.get("target_id"), Some(&target_id.to_string()));
    }

    #[test]
    fn test_build_count_query_comment_feedback() {
        let target_id = Uuid::now_v7();
        let (query, params) =
            build_count_query("CommentFeedbackByTargetId", "target_id", target_id);

        assert_query_contains(&query, "SELECT toUInt64(COUNT()) AS count");
        assert_query_contains(&query, "FROM CommentFeedbackByTargetId");
        assert_query_contains(&query, "WHERE target_id = {target_id:UUID}");
        assert_query_contains(&query, "FORMAT JSONEachRow");
        assert_eq!(params.get("target_id"), Some(&target_id.to_string()));
    }

    #[test]
    fn test_build_count_query_demonstration_feedback() {
        let inference_id = Uuid::now_v7();
        let (query, params) = build_count_query(
            "DemonstrationFeedbackByInferenceId",
            "inference_id",
            inference_id,
        );

        assert_query_contains(&query, "SELECT toUInt64(COUNT()) AS count");
        assert_query_contains(&query, "FROM DemonstrationFeedbackByInferenceId");
        assert_query_contains(&query, "WHERE inference_id = {inference_id:UUID}");
        assert_query_contains(&query, "FORMAT JSONEachRow");
        assert_eq!(params.get("inference_id"), Some(&inference_id.to_string()));
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
