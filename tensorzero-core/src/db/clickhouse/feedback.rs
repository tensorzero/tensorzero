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

// Helper implementations for individual feedback table queries
impl ClickHouseConnectionInfo {
    pub async fn query_boolean_metrics_by_target_id(
        &self,
        target_id: Uuid,
        before: Option<Uuid>,
        after: Option<Uuid>,
        page_size: u32,
    ) -> Result<Vec<BooleanMetricFeedbackRow>, Error> {
        let (where_clause, params) = build_pagination_clause(before, after, "target_id");
        let order_clause = if after.is_some() { "ASC" } else { "DESC" };

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

        let mut params_vec: Vec<(&str, String)> =
            params.iter().map(|(k, v)| (*k, v.clone())).collect();
        params_vec.push(("target_id", target_id.to_string()));
        params_vec.push(("page_size", page_size.to_string()));

        let query_params: std::collections::HashMap<&str, &str> =
            params_vec.iter().map(|(k, v)| (*k, v.as_str())).collect();

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
        let (where_clause, params) = build_pagination_clause(before, after, "target_id");
        let order_clause = if after.is_some() { "ASC" } else { "DESC" };

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

        let mut params_vec: Vec<(&str, String)> =
            params.iter().map(|(k, v)| (*k, v.clone())).collect();
        params_vec.push(("target_id", target_id.to_string()));
        params_vec.push(("page_size", page_size.to_string()));

        let query_params: std::collections::HashMap<&str, &str> =
            params_vec.iter().map(|(k, v)| (*k, v.as_str())).collect();

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
        let (where_clause, params) = build_pagination_clause(before, after, "target_id");
        let order_clause = if after.is_some() { "ASC" } else { "DESC" };

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

        let mut params_vec: Vec<(&str, String)> =
            params.iter().map(|(k, v)| (*k, v.clone())).collect();
        params_vec.push(("target_id", target_id.to_string()));
        params_vec.push(("page_size", page_size.to_string()));

        let query_params: std::collections::HashMap<&str, &str> =
            params_vec.iter().map(|(k, v)| (*k, v.as_str())).collect();

        let response = self.run_query_synchronous(query, &query_params).await?;

        parse_feedback_rows(response.response.as_str())
    }

    pub async fn query_boolean_metric_bounds_by_target_id(
        &self,
        target_id: Uuid,
    ) -> Result<TableBounds, Error> {
        let query = r"
            SELECT
                (SELECT id FROM BooleanMetricFeedbackByTargetId WHERE target_id = {target_id:UUID} ORDER BY toUInt128(id) ASC LIMIT 1) AS first_id,
                (SELECT id FROM BooleanMetricFeedbackByTargetId WHERE target_id = {target_id:UUID} ORDER BY toUInt128(id) DESC LIMIT 1) AS last_id
            FORMAT JSONEachRow
        ".to_string();

        let target_id_str = target_id.to_string();
        let params_vec = [("target_id", target_id_str)];
        let params: std::collections::HashMap<&str, &str> =
            params_vec.iter().map(|(k, v)| (*k, v.as_str())).collect();

        let response = self.run_query_synchronous(query, &params).await?;
        parse_table_bounds(&response.response)
    }

    pub async fn query_float_metric_bounds_by_target_id(
        &self,
        target_id: Uuid,
    ) -> Result<TableBounds, Error> {
        let query = r"
            SELECT
                (SELECT id FROM FloatMetricFeedbackByTargetId WHERE target_id = {target_id:UUID} ORDER BY toUInt128(id) ASC LIMIT 1) AS first_id,
                (SELECT id FROM FloatMetricFeedbackByTargetId WHERE target_id = {target_id:UUID} ORDER BY toUInt128(id) DESC LIMIT 1) AS last_id
            FORMAT JSONEachRow
        ".to_string();

        let target_id_str = target_id.to_string();
        let params_vec = [("target_id", target_id_str)];
        let params: std::collections::HashMap<&str, &str> =
            params_vec.iter().map(|(k, v)| (*k, v.as_str())).collect();

        let response = self.run_query_synchronous(query, &params).await?;
        parse_table_bounds(&response.response)
    }

    pub async fn query_comment_feedback_bounds_by_target_id(
        &self,
        target_id: Uuid,
    ) -> Result<TableBounds, Error> {
        let query = r"
            SELECT
                (SELECT id FROM CommentFeedbackByTargetId WHERE target_id = {target_id:UUID} ORDER BY toUInt128(id) ASC LIMIT 1) AS first_id,
                (SELECT id FROM CommentFeedbackByTargetId WHERE target_id = {target_id:UUID} ORDER BY toUInt128(id) DESC LIMIT 1) AS last_id
            FORMAT JSONEachRow
        ".to_string();

        let target_id_str = target_id.to_string();
        let params_vec = [("target_id", target_id_str)];
        let params: std::collections::HashMap<&str, &str> =
            params_vec.iter().map(|(k, v)| (*k, v.as_str())).collect();

        let response = self.run_query_synchronous(query, &params).await?;
        parse_table_bounds(&response.response)
    }

    pub async fn query_demonstration_feedback_bounds_by_inference_id(
        &self,
        inference_id: Uuid,
    ) -> Result<TableBounds, Error> {
        let query = r"
            SELECT
                (SELECT id FROM DemonstrationFeedbackByInferenceId WHERE inference_id = {inference_id:UUID} ORDER BY toUInt128(id) ASC LIMIT 1) AS first_id,
                (SELECT id FROM DemonstrationFeedbackByInferenceId WHERE inference_id = {inference_id:UUID} ORDER BY toUInt128(id) DESC LIMIT 1) AS last_id
            FORMAT JSONEachRow
        ".to_string();

        let inference_id_str = inference_id.to_string();
        let params_vec = [("inference_id", inference_id_str)];
        let params: std::collections::HashMap<&str, &str> =
            params_vec.iter().map(|(k, v)| (*k, v.as_str())).collect();

        let response = self.run_query_synchronous(query, &params).await?;
        parse_table_bounds(&response.response)
    }

    pub async fn count_boolean_metrics_by_target_id(&self, target_id: Uuid) -> Result<u64, Error> {
        let query =
            "SELECT toUInt64(COUNT()) AS count FROM BooleanMetricFeedbackByTargetId WHERE target_id = {target_id:UUID} FORMAT JSONEachRow".to_string();

        let target_id_str = target_id.to_string();
        let params_vec = [("target_id", target_id_str)];
        let params: std::collections::HashMap<&str, &str> =
            params_vec.iter().map(|(k, v)| (*k, v.as_str())).collect();

        let response = self.run_query_synchronous(query, &params).await?;
        parse_count(&response.response)
    }

    pub async fn count_float_metrics_by_target_id(&self, target_id: Uuid) -> Result<u64, Error> {
        let query =
            "SELECT toUInt64(COUNT()) AS count FROM FloatMetricFeedbackByTargetId WHERE target_id = {target_id:UUID} FORMAT JSONEachRow".to_string();

        let target_id_str = target_id.to_string();
        let params_vec = [("target_id", target_id_str)];
        let params: std::collections::HashMap<&str, &str> =
            params_vec.iter().map(|(k, v)| (*k, v.as_str())).collect();

        let response = self.run_query_synchronous(query, &params).await?;
        parse_count(&response.response)
    }

    pub async fn count_comment_feedback_by_target_id(&self, target_id: Uuid) -> Result<u64, Error> {
        let query =
            "SELECT toUInt64(COUNT()) AS count FROM CommentFeedbackByTargetId WHERE target_id = {target_id:UUID} FORMAT JSONEachRow".to_string();

        let target_id_str = target_id.to_string();
        let params_vec = [("target_id", target_id_str)];
        let params: std::collections::HashMap<&str, &str> =
            params_vec.iter().map(|(k, v)| (*k, v.as_str())).collect();

        let response = self.run_query_synchronous(query, &params).await?;
        parse_count(&response.response)
    }

    pub async fn count_demonstration_feedback_by_inference_id(
        &self,
        inference_id: Uuid,
    ) -> Result<u64, Error> {
        let query =
            "SELECT toUInt64(COUNT()) AS count FROM DemonstrationFeedbackByInferenceId WHERE inference_id = {inference_id:UUID} FORMAT JSONEachRow".to_string();

        let inference_id_str = inference_id.to_string();
        let params_vec = [("inference_id", inference_id_str)];
        let params: std::collections::HashMap<&str, &str> =
            params_vec.iter().map(|(k, v)| (*k, v.as_str())).collect();

        let response = self.run_query_synchronous(query, &params).await?;
        parse_count(&response.response)
    }
}
