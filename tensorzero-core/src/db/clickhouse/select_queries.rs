use crate::db::{
    clickhouse::migration_manager::migrations::migration_0037::quantiles_sql_args,
    BooleanMetricFeedbackRow, CommentFeedbackRow, DemonstrationFeedbackRow, EpisodeByIdRow,
    FeedbackBounds, FeedbackBoundsByType, FeedbackByVariant, FeedbackRow, FloatMetricFeedbackRow,
    TableBounds, TableBoundsWithCount,
};
use async_trait::async_trait;
use serde::Deserialize;
use uuid::Uuid;

use crate::{
    db::{ModelLatencyDatapoint, ModelUsageTimePoint, SelectQueries, TimeWindow},
    error::{Error, ErrorDetails},
};

use super::{escape_string_for_clickhouse_literal, ClickHouseConnectionInfo};

#[async_trait]
impl SelectQueries for ClickHouseConnectionInfo {
    /// Retrieves a timeseries of model usage data.
    /// This will return max_periods complete time periods worth of data if present
    /// as well as the current time period's data.
    /// So there are at most max_periods + 1 time periods worth of data returned.
    async fn get_model_usage_timeseries(
        &self,
        time_window: TimeWindow,
        max_periods: u32,
    ) -> Result<Vec<ModelUsageTimePoint>, Error> {
        // TODO: probably factor this out into common code as other queries will likely need similar logic
        // NOTE: this filter pattern will likely include some extra data since the current period is likely incomplete.
        let (time_grouping, time_filter) = match time_window {
            TimeWindow::Hour => (
                "toStartOfHour(minute)",
                format!("minute >= (SELECT max(toStartOfHour(minute)) FROM ModelProviderStatistics) - INTERVAL {max_periods} HOUR"),
            ),
            TimeWindow::Day => (
                "toStartOfDay(minute)",
                format!("minute >= (SELECT max(toStartOfDay(minute)) FROM ModelProviderStatistics) - INTERVAL {max_periods} DAY"),
            ),
            TimeWindow::Week => (
                "toStartOfWeek(minute)",
                format!("minute >= (SELECT max(toStartOfWeek(minute)) FROM ModelProviderStatistics) - INTERVAL {max_periods} WEEK"),
            ),
            TimeWindow::Month => (
                "toStartOfMonth(minute)",
                format!("minute >= (SELECT max(toStartOfMonth(minute)) FROM ModelProviderStatistics) - INTERVAL {max_periods} MONTH"),
            ),
            TimeWindow::Cumulative => (
                "toDateTime('1970-01-01 00:00:00')",
                "1 = 1".to_string(), // No time filter for cumulative
            ),
        };

        let query = format!(
            r"
            SELECT
                formatDateTime({time_grouping}, '%Y-%m-%dT%H:%i:%SZ') as period_start,
                model_name,
                sumMerge(total_input_tokens) as input_tokens,
                sumMerge(total_output_tokens) as output_tokens,
                countMerge(count) as count
            FROM ModelProviderStatistics
            WHERE {time_filter}
            GROUP BY period_start, model_name
            ORDER BY period_start DESC, model_name
            FORMAT JSONEachRow
            ",
        );

        let response = self.run_query_synchronous_no_params(query).await?;

        // Deserialize the results into ModelUsageTimePoint
        response
            .response
            .trim()
            .lines()
            .map(|row| {
                serde_json::from_str(row).map_err(|e| {
                    Error::new(ErrorDetails::ClickHouseDeserialization {
                        message: e.to_string(),
                    })
                })
            })
            .collect::<Result<Vec<_>, _>>()
    }

    async fn get_model_latency_quantiles(
        &self,
        time_window: TimeWindow,
    ) -> Result<Vec<ModelLatencyDatapoint>, Error> {
        let time_filter = match time_window {
            TimeWindow::Hour => {
                "minute >= (SELECT max(minute) FROM ModelProviderStatistics) - INTERVAL 1 HOUR"
                    .to_string()
            }
            TimeWindow::Day => {
                "minute >= (SELECT max(minute) FROM ModelProviderStatistics) - INTERVAL 1 DAY"
                    .to_string()
            }
            TimeWindow::Week => {
                "minute >= (SELECT max(minute) FROM ModelProviderStatistics) - INTERVAL 1 WEEK"
                    .to_string()
            }
            TimeWindow::Month => {
                "minute >= (SELECT max(minute) FROM ModelProviderStatistics) - INTERVAL 1 MONTH"
                    .to_string()
            }
            TimeWindow::Cumulative => "1 = 1".to_string(),
        };
        let qs = quantiles_sql_args();
        let query = format!(
            r"
            SELECT
                model_name,
                quantilesTDigestMerge({qs})(response_time_ms_quantiles) AS response_time_ms_quantiles,
                quantilesTDigestMerge({qs})(ttft_ms_quantiles) AS ttft_ms_quantiles,
                countMerge(count) as count
            FROM ModelProviderStatistics
            WHERE {time_filter}
            GROUP BY model_name
            ORDER BY model_name
            FORMAT JSONEachRow
            ",
        );
        let response = self.run_query_synchronous_no_params(query).await?;
        // Deserialize the results into ModelLatencyDatapoint
        response
            .response
            .trim()
            .lines()
            .map(|row| {
                serde_json::from_str(row).map_err(|e| {
                    Error::new(ErrorDetails::ClickHouseDeserialization {
                        message: e.to_string(),
                    })
                })
            })
            .collect::<Result<Vec<_>, _>>()
    }

    async fn count_distinct_models_used(&self) -> Result<u32, Error> {
        let query =
            "SELECT toUInt32(uniqExact(model_name)) FROM ModelProviderStatistics".to_string();
        let response = self.run_query_synchronous_no_params(query).await?;
        response
            .response
            .trim()
            .lines()
            .next()
            .ok_or_else(|| {
                Error::new(ErrorDetails::ClickHouseDeserialization {
                    message: "No result".to_string(),
                })
            })?
            .parse()
            .map_err(|e: std::num::ParseIntError| {
                Error::new(ErrorDetails::ClickHouseDeserialization {
                    message: e.to_string(),
                })
            })
    }

    async fn query_episode_table(
        &self,
        page_size: u32,
        before: Option<Uuid>,
        after: Option<Uuid>,
    ) -> Result<Vec<EpisodeByIdRow>, Error> {
        let (where_clause, params, is_after) = match (before, after) {
            (Some(_before), Some(_after)) => {
                return Err(Error::new(ErrorDetails::InvalidRequest {
                    message: "Cannot specify both before and after in query_episode_table"
                        .to_string(),
                }))
            }
            (Some(before), None) => (
                "episode_id_uint < toUInt128({before:UUID})",
                vec![("before", before.to_string())],
                false,
            ),
            (None, Some(after)) => (
                "episode_id_uint > toUInt128({after:UUID})",
                vec![("after", after.to_string())],
                true,
            ),
            (None, None) => ("1=1", vec![], false),
        };
        // This is a Bad Hack to account for ClickHouse's Bad Query Planning
        // Clickhouse will not optimize queries with either GROUP BY or FINAL
        // when reading the primary key with a LIMIT
        // https://clickhouse.com/docs/sql-reference/statements/select/order-by#optimization-of-data-reading
        // So, we select the last page_size_overestimate episodes in descending order
        // Then we group by episode_id_uint and count the number of inferences
        // Finally, we order by episode_id_uint correctly and limit the result to page_size
        let page_size_overestimate = 5 * page_size;

        let query = if is_after {
            format!(
                r"WITH potentially_duplicated_episode_ids AS (
                    SELECT episode_id_uint
                    FROM EpisodeById
                    WHERE {where_clause}
                    ORDER BY episode_id_uint ASC
                    LIMIT {page_size_overestimate}
                )
                SELECT
                    episode_id,
                    count,
                    start_time,
                    end_time,
                    last_inference_id
                FROM (
                    SELECT
                        uint_to_uuid(episode_id_uint) as episode_id,
                        count() as count,
                        formatDateTime(UUIDv7ToDateTime(uint_to_uuid(min(id_uint))), '%Y-%m-%dT%H:%i:%SZ') as start_time,
                        formatDateTime(UUIDv7ToDateTime(uint_to_uuid(max(id_uint))), '%Y-%m-%dT%H:%i:%SZ') as end_time,
                        uint_to_uuid(max(id_uint)) as last_inference_id,
                        episode_id_uint
                    FROM InferenceByEpisodeId
                    WHERE episode_id_uint IN (SELECT episode_id_uint FROM potentially_duplicated_episode_ids)
                    GROUP BY episode_id_uint
                    ORDER BY episode_id_uint ASC
                    LIMIT {page_size}
                )
                ORDER BY episode_id_uint DESC
                FORMAT JSONEachRow
                "
            )
        } else {
            format!(
                r"
                WITH potentially_duplicated_episode_ids AS (
                    SELECT episode_id_uint
                    FROM EpisodeById
                    WHERE {where_clause}
                    ORDER BY episode_id_uint DESC
                    LIMIT {page_size_overestimate}
                )
                SELECT
                    uint_to_uuid(episode_id_uint) as episode_id,
                    count() as count,
                    formatDateTime(UUIDv7ToDateTime(uint_to_uuid(min(id_uint))), '%Y-%m-%dT%H:%i:%SZ') as start_time,
                    formatDateTime(UUIDv7ToDateTime(uint_to_uuid(max(id_uint))), '%Y-%m-%dT%H:%i:%SZ') as end_time,
                    uint_to_uuid(max(id_uint)) as last_inference_id
                FROM InferenceByEpisodeId
                WHERE episode_id_uint IN (SELECT episode_id_uint FROM potentially_duplicated_episode_ids)
                GROUP BY episode_id_uint
                ORDER BY episode_id_uint DESC
                LIMIT {page_size}
                FORMAT JSONEachRow
                "
            )
        };
        let params_str_map: std::collections::HashMap<&str, &str> =
            params.iter().map(|(k, v)| (*k, v.as_str())).collect();
        let response = self.run_query_synchronous(query, &params_str_map).await?;
        // Deserialize the results into EpisodeByIdRow
        response
            .response
            .trim()
            .lines()
            .map(|row| {
                serde_json::from_str(row).map_err(|e| {
                    Error::new(ErrorDetails::ClickHouseDeserialization {
                        message: e.to_string(),
                    })
                })
            })
            .collect::<Result<Vec<_>, _>>()
    }

    async fn query_episode_table_bounds(&self) -> Result<TableBoundsWithCount, Error> {
        let query = r"
            SELECT
                uint_to_uuid(min(episode_id_uint)) as first_id,
                uint_to_uuid(max(episode_id_uint)) as last_id,
                count() as count
            FROM EpisodeById
            FORMAT JSONEachRow"
            .to_string();
        let response = self.run_query_synchronous_no_params(query).await?;
        let response = response.response.trim();
        serde_json::from_str(response).map_err(|e| {
            Error::new(ErrorDetails::ClickHouseDeserialization {
                message: e.to_string(),
            })
        })
    }

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
                Error::new(ErrorDetails::ClickHouseDeserialization {
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
            return Err(Error::new(ErrorDetails::InvalidRequest {
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
            self.query_demonstration_feedback_by_inference_id(target_id, before, after, page_size)
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
}

// Helper implementations for individual feedback table queries
impl ClickHouseConnectionInfo {
    async fn query_boolean_metrics_by_target_id(
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
                    'boolean' AS type,
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
                        formatDateTime(UUIDv7ToDateTime(id), '%Y-%m-%dT%%H:%%i:%%SZ') AS timestamp
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
                    formatDateTime(UUIDv7ToDateTime(id), '%Y-%m-%dT%%H:%%i:%%SZ') AS timestamp
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

    async fn query_float_metrics_by_target_id(
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
                    'float' AS type,
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
                        formatDateTime(UUIDv7ToDateTime(id), '%Y-%m-%dT%%H:%%i:%%SZ') AS timestamp
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
                    formatDateTime(UUIDv7ToDateTime(id), '%Y-%m-%dT%%H:%%i:%%SZ') AS timestamp
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

    async fn query_comment_feedback_by_target_id(
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
                    'comment' AS type,
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
                        formatDateTime(UUIDv7ToDateTime(id), '%Y-%m-%dT%%H:%%i:%%SZ') AS timestamp
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
                    formatDateTime(UUIDv7ToDateTime(id), '%Y-%m-%dT%%H:%%i:%%SZ') AS timestamp
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

    async fn query_demonstration_feedback_by_inference_id(
        &self,
        inference_id: Uuid,
        before: Option<Uuid>,
        after: Option<Uuid>,
        page_size: u32,
    ) -> Result<Vec<DemonstrationFeedbackRow>, Error> {
        let (where_clause, params) = build_pagination_clause(before, after, "inference_id");
        let order_clause = if after.is_some() { "ASC" } else { "DESC" };

        let query = if after.is_some() {
            format!(
                r"
                SELECT
                    'demonstration' AS type,
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
                        formatDateTime(UUIDv7ToDateTime(id), '%Y-%m-%dT%%H:%%i:%%SZ') AS timestamp
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
                    formatDateTime(UUIDv7ToDateTime(id), '%Y-%m-%dT%%H:%%i:%%SZ') AS timestamp
                FROM DemonstrationFeedbackByInferenceId
                WHERE inference_id = {{inference_id:UUID}} {where_clause}
                ORDER BY toUInt128(id) {order_clause}
                LIMIT {{page_size:UInt32}}
                FORMAT JSONEachRow
                "
            )
        };

        let mut params_vec: Vec<(&str, String)> =
            params.iter().map(|(k, v)| (*k, v.clone())).collect();
        params_vec.push(("inference_id", inference_id.to_string()));
        params_vec.push(("page_size", page_size.to_string()));

        let query_params: std::collections::HashMap<&str, &str> =
            params_vec.iter().map(|(k, v)| (*k, v.as_str())).collect();

        let response = self.run_query_synchronous(query, &query_params).await?;

        parse_feedback_rows(response.response.as_str())
    }

    async fn query_boolean_metric_bounds_by_target_id(
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

    async fn query_float_metric_bounds_by_target_id(
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

    async fn query_comment_feedback_bounds_by_target_id(
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

    async fn query_demonstration_feedback_bounds_by_inference_id(
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

    async fn count_boolean_metrics_by_target_id(&self, target_id: Uuid) -> Result<u64, Error> {
        let query =
            "SELECT toUInt64(COUNT()) AS count FROM BooleanMetricFeedbackByTargetId WHERE target_id = {target_id:UUID}".to_string();

        let target_id_str = target_id.to_string();
        let params_vec = [("target_id", target_id_str)];
        let params: std::collections::HashMap<&str, &str> =
            params_vec.iter().map(|(k, v)| (*k, v.as_str())).collect();

        let response = self.run_query_synchronous(query, &params).await?;
        parse_count(&response.response)
    }

    async fn count_float_metrics_by_target_id(&self, target_id: Uuid) -> Result<u64, Error> {
        let query =
            "SELECT toUInt64(COUNT()) AS count FROM FloatMetricFeedbackByTargetId WHERE target_id = {target_id:UUID}".to_string();

        let target_id_str = target_id.to_string();
        let params_vec = [("target_id", target_id_str)];
        let params: std::collections::HashMap<&str, &str> =
            params_vec.iter().map(|(k, v)| (*k, v.as_str())).collect();

        let response = self.run_query_synchronous(query, &params).await?;
        parse_count(&response.response)
    }

    async fn count_comment_feedback_by_target_id(&self, target_id: Uuid) -> Result<u64, Error> {
        let query =
            "SELECT toUInt64(COUNT()) AS count FROM CommentFeedbackByTargetId WHERE target_id = {target_id:UUID}".to_string();

        let target_id_str = target_id.to_string();
        let params_vec = [("target_id", target_id_str)];
        let params: std::collections::HashMap<&str, &str> =
            params_vec.iter().map(|(k, v)| (*k, v.as_str())).collect();

        let response = self.run_query_synchronous(query, &params).await?;
        parse_count(&response.response)
    }

    async fn count_demonstration_feedback_by_inference_id(
        &self,
        inference_id: Uuid,
    ) -> Result<u64, Error> {
        let query =
            "SELECT toUInt64(COUNT()) AS count FROM DemonstrationFeedbackByInferenceId WHERE inference_id = {inference_id:UUID}".to_string();

        let inference_id_str = inference_id.to_string();
        let params_vec = [("inference_id", inference_id_str)];
        let params: std::collections::HashMap<&str, &str> =
            params_vec.iter().map(|(k, v)| (*k, v.as_str())).collect();

        let response = self.run_query_synchronous(query, &params).await?;
        parse_count(&response.response)
    }
}

// Helper functions for parsing responses
fn build_pagination_clause(
    before: Option<Uuid>,
    after: Option<Uuid>,
    _id_column: &str,
) -> (String, Vec<(&'static str, String)>) {
    match (before, after) {
        (Some(before), None) => (
            "AND toUInt128(id) < toUInt128(toUUID({before:UUID}))".to_string(),
            vec![("before", before.to_string())],
        ),
        (None, Some(after)) => (
            "AND toUInt128(id) > toUInt128(toUUID({after:UUID}))".to_string(),
            vec![("after", after.to_string())],
        ),
        _ => (String::new(), vec![]),
    }
}

fn parse_feedback_rows<T: serde::de::DeserializeOwned>(response: &str) -> Result<Vec<T>, Error> {
    response
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|row| {
            serde_json::from_str(row).map_err(|e| {
                Error::new(ErrorDetails::ClickHouseDeserialization {
                    message: format!("Failed to deserialize feedback row: {e}"),
                })
            })
        })
        .collect()
}

fn parse_table_bounds(response: &str) -> Result<TableBounds, Error> {
    let line = response.trim();
    if line.is_empty() {
        return Ok(TableBounds {
            first_id: None,
            last_id: None,
        });
    }

    serde_json::from_str(line).map_err(|e| {
        Error::new(ErrorDetails::ClickHouseDeserialization {
            message: format!("Failed to deserialize table bounds: {e}"),
        })
    })
}

fn parse_count(response: &str) -> Result<u64, Error> {
    #[derive(Deserialize)]
    struct CountResult {
        count: u64,
    }

    let line = response.trim().lines().next().ok_or_else(|| {
        Error::new(ErrorDetails::ClickHouseDeserialization {
            message: "No count result returned from database".to_string(),
        })
    })?;

    let result: CountResult = serde_json::from_str(line).map_err(|e| {
        Error::new(ErrorDetails::ClickHouseDeserialization {
            message: format!("Failed to deserialize count: {e}"),
        })
    })?;

    Ok(result.count)
}
