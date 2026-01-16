use std::collections::HashMap;

use async_trait::async_trait;
use serde::Deserialize;
use uuid::Uuid;

use crate::{
    config::MetricConfigLevel,
    db::{
        FeedbackQueries, TableBounds, TimeWindow,
        feedback::{
            BooleanMetricFeedbackInsert, BooleanMetricFeedbackRow, CommentFeedbackInsert,
            CommentFeedbackRow, CumulativeFeedbackTimeSeriesPoint, DemonstrationFeedbackInsert,
            DemonstrationFeedbackRow, FeedbackBounds, FeedbackBoundsByType, FeedbackByVariant,
            FeedbackRow, FloatMetricFeedbackInsert, FloatMetricFeedbackRow,
            GetVariantPerformanceParams, LatestFeedbackRow, MetricType, MetricWithFeedback,
            StaticEvaluationHumanFeedbackInsert, VariantPerformanceRow,
        },
    },
    error::{Error, ErrorDetails},
    experimentation::asymptotic_confidence_sequences::asymp_cs,
    function::FunctionConfig,
};

use super::{
    ClickHouseConnectionInfo, escape_string_for_clickhouse_literal,
    select_queries::{build_pagination_clause, parse_count, parse_json_rows},
};

/// Raw database result for metrics with feedback (without metric_type)
#[derive(Debug, Deserialize)]
struct MetricWithFeedbackRaw {
    function_name: String,
    metric_name: String,
    feedback_count: u32,
}

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
    limit: u32,
) -> HashMap<String, String> {
    let mut params_map: HashMap<String, String> = pagination_params
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
    params_map.insert(id_column.to_string(), id_value.to_string());
    params_map.insert("limit".to_string(), limit.to_string());
    params_map
}

// Generic query builder for feedback tables
fn build_feedback_query(
    config: &FeedbackTableConfig,
    id_value: Uuid,
    before: Option<Uuid>,
    after: Option<Uuid>,
    limit: u32,
) -> (String, HashMap<String, String>) {
    let (where_clause, params) = build_pagination_clause(before, after, config.id_column);
    let order_clause = if after.is_some() { "ASC" } else { "DESC" };

    let params_map = build_params_map(config.id_column, id_value, params, limit);

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
                LIMIT {{limit:UInt32}}
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
            LIMIT {{limit:UInt32}}
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
    limit: u32,
) -> (String, HashMap<String, String>) {
    build_feedback_query(
        &FeedbackTableConfig::BOOLEAN_METRICS,
        target_id,
        before,
        after,
        limit,
    )
}

pub(crate) fn build_float_metrics_query(
    target_id: Uuid,
    before: Option<Uuid>,
    after: Option<Uuid>,
    limit: u32,
) -> (String, HashMap<String, String>) {
    build_feedback_query(
        &FeedbackTableConfig::FLOAT_METRICS,
        target_id,
        before,
        after,
        limit,
    )
}

pub(crate) fn build_comment_feedback_query(
    target_id: Uuid,
    before: Option<Uuid>,
    after: Option<Uuid>,
    limit: u32,
) -> (String, HashMap<String, String>) {
    build_feedback_query(
        &FeedbackTableConfig::COMMENT_FEEDBACK,
        target_id,
        before,
        after,
        limit,
    )
}

pub(crate) fn build_demonstration_feedback_query(
    inference_id: Uuid,
    before: Option<Uuid>,
    after: Option<Uuid>,
    limit: u32,
) -> (String, HashMap<String, String>) {
    build_feedback_query(
        &FeedbackTableConfig::DEMONSTRATION_FEEDBACK,
        inference_id,
        before,
        after,
        limit,
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
    parse_json_rows(response.response.as_str())
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
    async fn query_boolean_metrics_by_target_id(
        &self,
        target_id: Uuid,
        before: Option<Uuid>,
        after: Option<Uuid>,
        limit: u32,
    ) -> Result<Vec<BooleanMetricFeedbackRow>, Error> {
        let (query, params_owned) = build_boolean_metrics_query(target_id, before, after, limit);
        execute_feedback_query(self, query, params_owned).await
    }

    async fn query_float_metrics_by_target_id(
        &self,
        target_id: Uuid,
        before: Option<Uuid>,
        after: Option<Uuid>,
        limit: u32,
    ) -> Result<Vec<FloatMetricFeedbackRow>, Error> {
        let (query, params_owned) = build_float_metrics_query(target_id, before, after, limit);
        execute_feedback_query(self, query, params_owned).await
    }

    async fn query_comment_feedback_by_target_id(
        &self,
        target_id: Uuid,
        before: Option<Uuid>,
        after: Option<Uuid>,
        limit: u32,
    ) -> Result<Vec<CommentFeedbackRow>, Error> {
        let (query, params_owned) = build_comment_feedback_query(target_id, before, after, limit);
        execute_feedback_query(self, query, params_owned).await
    }

    async fn query_boolean_metric_bounds_by_target_id(
        &self,
        target_id: Uuid,
    ) -> Result<TableBounds, Error> {
        let (query, params_owned) =
            build_bounds_query("BooleanMetricFeedbackByTargetId", "target_id", target_id);
        execute_bounds_query(self, query, params_owned).await
    }

    async fn query_float_metric_bounds_by_target_id(
        &self,
        target_id: Uuid,
    ) -> Result<TableBounds, Error> {
        let (query, params_owned) =
            build_bounds_query("FloatMetricFeedbackByTargetId", "target_id", target_id);
        execute_bounds_query(self, query, params_owned).await
    }

    async fn query_comment_feedback_bounds_by_target_id(
        &self,
        target_id: Uuid,
    ) -> Result<TableBounds, Error> {
        let (query, params_owned) =
            build_bounds_query("CommentFeedbackByTargetId", "target_id", target_id);
        execute_bounds_query(self, query, params_owned).await
    }

    async fn query_demonstration_feedback_bounds_by_inference_id(
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

    async fn count_boolean_metrics_by_target_id(&self, target_id: Uuid) -> Result<u64, Error> {
        let (query, params_owned) =
            build_count_query("BooleanMetricFeedbackByTargetId", "target_id", target_id);
        execute_count_query(self, query, params_owned).await
    }

    async fn count_float_metrics_by_target_id(&self, target_id: Uuid) -> Result<u64, Error> {
        let (query, params_owned) =
            build_count_query("FloatMetricFeedbackByTargetId", "target_id", target_id);
        execute_count_query(self, query, params_owned).await
    }

    async fn count_comment_feedback_by_target_id(&self, target_id: Uuid) -> Result<u64, Error> {
        let (query, params_owned) =
            build_count_query("CommentFeedbackByTargetId", "target_id", target_id);
        execute_count_query(self, query, params_owned).await
    }

    async fn count_demonstration_feedback_by_inference_id(
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

/// Builds the SQL query for querying metrics with feedback for a function
///
/// This query uses FeedbackByVariantStatistics (a pre-aggregated table) for
/// boolean/float metrics, making it very fast. Demonstrations are queried
/// separately with a simple join. The metric_type is not returned from the
/// database - it should be looked up from the config at the endpoint level.
fn build_metrics_with_feedback_query(
    function_name: &str,
    inference_table: &str,
    variant_name: Option<&str>,
) -> (String, HashMap<String, String>) {
    let mut query_params = HashMap::new();
    query_params.insert("function_name".to_string(), function_name.to_string());

    let variant_clause = if let Some(vname) = variant_name {
        query_params.insert("variant_name".to_string(), vname.to_string());
        "AND variant_name = {variant_name:String}"
    } else {
        ""
    };

    // Use FeedbackByVariantStatistics for boolean/float metrics (pre-aggregated, fast)
    // and a simple query for demonstrations
    let query = format!(
        r"
        SELECT
          function_name,
          metric_name,
          feedback_count
        FROM (
          -- Boolean/float metrics from pre-aggregated table (very fast)
          SELECT
            function_name,
            metric_name,
            toUInt32(sum(count)) as feedback_count
          FROM FeedbackByVariantStatistics
          WHERE function_name = {{function_name:String}} {variant_clause}
          GROUP BY function_name, metric_name
          HAVING feedback_count > 0

          UNION ALL

          -- Demonstration metrics (need to join with inference table)
          SELECT
            {{function_name:String}} as function_name,
            'demonstration' as metric_name,
            toUInt32(COUNT(*)) as feedback_count
          FROM DemonstrationFeedback
          WHERE inference_id IN (SELECT id FROM {inference_table} WHERE function_name = {{function_name:String}} {variant_clause})
          HAVING feedback_count > 0
        )
        ORDER BY metric_name
        FORMAT JSONEachRow"
    );

    (query, query_params)
}

/// Builds the SQL query for getting the latest feedback ID for each metric for a target
fn build_latest_feedback_id_by_metric_query(target_id: Uuid) -> (String, HashMap<String, String>) {
    let mut query_params = HashMap::new();
    query_params.insert("target_id".to_string(), target_id.to_string());

    let query = r"
        SELECT
          metric_name,
          argMax(id, toUInt128(id)) as latest_id
        FROM BooleanMetricFeedbackByTargetId
        WHERE target_id = {target_id:String}
        GROUP BY metric_name

        UNION ALL

        SELECT
          metric_name,
          argMax(id, toUInt128(id)) as latest_id
        FROM FloatMetricFeedbackByTargetId
        WHERE target_id = {target_id:String}
        GROUP BY metric_name

        ORDER BY metric_name
        FORMAT JSONEachRow"
        .to_string();

    (query, query_params)
}

/// Builds the SQL query for getting variant performance statistics.
///
/// Supports both inference-level and episode-level metrics, with cumulative or time-windowed grouping.
fn build_variant_performance_query(
    params: &GetVariantPerformanceParams<'_>,
) -> (String, HashMap<String, String>) {
    let mut query_params = HashMap::new();
    query_params.insert(
        "function_name".to_string(),
        params.function_name.to_string(),
    );
    query_params.insert("metric_name".to_string(), params.metric_name.to_string());

    let inference_table = params.inference_table_name();
    let metric_table = params.metric_table_name();

    // Build variant filter if specified
    let variant_filter = match params.variant_name {
        Some(variant_name) => {
            query_params.insert("variant_name".to_string(), variant_name.to_string());
            " AND i.variant_name = {variant_name:String}"
        }
        None => "",
    };

    let query = match (params.metric_level(), &params.time_window) {
        // Episode-level metric with cumulative time window
        (MetricConfigLevel::Episode, TimeWindow::Cumulative) => format!(
            r"
            WITH sub AS (
                SELECT
                    i.variant_name AS variant_name,
                    i.episode_id AS episode_id,
                    any(f.value) AS value_per_episode
                FROM {inference_table} i
                JOIN (
                    SELECT
                        target_id,
                        value,
                        ROW_NUMBER() OVER (PARTITION BY target_id ORDER BY timestamp DESC) as rn
                    FROM {metric_table}
                    WHERE metric_name = {{metric_name:String}}
                ) f ON i.episode_id = f.target_id AND f.rn = 1
                WHERE
                    i.function_name = {{function_name:String}}
                    {variant_filter}
                GROUP BY
                    variant_name,
                    episode_id
            )
            SELECT
                '1970-01-01T00:00:00.000Z' AS period_start,
                variant_name,
                toUInt32(count()) AS count,
                avg(value_per_episode) AS avg_metric,
                stddevSamp(value_per_episode) AS stdev,
                1.96 * (stddevSamp(value_per_episode) / sqrt(count())) AS ci_error
            FROM sub
            GROUP BY
                variant_name
            ORDER BY
                variant_name ASC
            FORMAT JSONEachRow"
        ),

        // Episode-level metric with time window
        (MetricConfigLevel::Episode, time_window) => {
            let time_window_str = time_window.to_clickhouse_string();
            query_params.insert("time_window_unit".to_string(), time_window_str.to_string());
            format!(
                r"
                WITH sub AS (
                    SELECT
                        dateTrunc({{time_window_unit:String}}, i.timestamp) AS period_start,
                        i.variant_name AS variant_name,
                        i.episode_id AS episode_id,
                        any(f.value) AS value_per_episode
                    FROM {inference_table} i
                    JOIN (
                        SELECT
                            target_id,
                            value,
                            ROW_NUMBER() OVER (PARTITION BY target_id ORDER BY timestamp DESC) as rn
                        FROM {metric_table}
                        WHERE metric_name = {{metric_name:String}}
                    ) f ON i.episode_id = f.target_id AND f.rn = 1
                    WHERE
                        i.function_name = {{function_name:String}}
                        {variant_filter}
                    GROUP BY
                        period_start,
                        variant_name,
                        episode_id
                )
                SELECT
                    formatDateTime(period_start, '%Y-%m-%dT%H:%i:%S.000Z') AS period_start,
                    variant_name,
                    toUInt32(count()) AS count,
                    avg(value_per_episode) AS avg_metric,
                    stddevSamp(value_per_episode) AS stdev,
                    1.96 * (stddevSamp(value_per_episode) / sqrt(count())) AS ci_error
                FROM sub
                GROUP BY
                    period_start,
                    variant_name
                ORDER BY
                    period_start ASC,
                    variant_name ASC
                FORMAT JSONEachRow"
            )
        }

        // Inference-level metric with cumulative time window
        (MetricConfigLevel::Inference, TimeWindow::Cumulative) => format!(
            r"
            SELECT
                '1970-01-01T00:00:00.000Z' AS period_start,
                i.variant_name AS variant_name,
                toUInt32(count()) AS count,
                avg(f.value) AS avg_metric,
                stddevSamp(f.value) AS stdev,
                1.96 * (stddevSamp(f.value) / sqrt(count())) AS ci_error
            FROM {inference_table} i
            JOIN (
                SELECT
                    target_id,
                    value,
                    ROW_NUMBER() OVER (PARTITION BY target_id ORDER BY timestamp DESC) as rn
                FROM {metric_table}
                WHERE metric_name = {{metric_name:String}}
            ) f ON i.id = f.target_id AND f.rn = 1
            WHERE
                i.function_name = {{function_name:String}}{variant_filter}
            GROUP BY
                variant_name
            ORDER BY
                variant_name ASC
            FORMAT JSONEachRow"
        ),

        // Inference-level metric with time window
        (MetricConfigLevel::Inference, time_window) => {
            let time_window_str = time_window.to_clickhouse_string();
            query_params.insert("time_window_unit".to_string(), time_window_str.to_string());
            format!(
                r"
                SELECT
                    formatDateTime(dateTrunc({{time_window_unit:String}}, i.timestamp), '%Y-%m-%dT%H:%i:%S.000Z') AS period_start,
                    i.variant_name AS variant_name,
                    toUInt32(count()) AS count,
                    avg(f.value) AS avg_metric,
                    stddevSamp(f.value) AS stdev,
                    1.96 * (stddevSamp(f.value) / sqrt(count())) AS ci_error
                FROM {inference_table} i
                JOIN (
                    SELECT
                        target_id,
                        value,
                        ROW_NUMBER() OVER (PARTITION BY target_id ORDER BY timestamp DESC) as rn
                    FROM {metric_table}
                    WHERE metric_name = {{metric_name:String}}
                ) f ON i.id = f.target_id AND f.rn = 1
                WHERE
                    i.function_name = {{function_name:String}}{variant_filter}
                GROUP BY
                    period_start,
                    variant_name
                ORDER BY
                    period_start ASC,
                    variant_name ASC
                FORMAT JSONEachRow"
            )
        }
    };

    (query, query_params)
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
            WHERE function_name = {{function_name:String}} and metric_name = {{metric_name:String}}{variant_filter}
            GROUP BY variant_name
            FORMAT JSONEachRow"
        );

        let mut params_map = HashMap::new();
        params_map.insert("function_name".to_string(), function_name.to_string());
        params_map.insert("metric_name".to_string(), metric_name.to_string());

        let query_params: HashMap<&str, &str> = params_map
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        // Each row is a JSON encoded FeedbackByVariant struct
        let res = self.run_query_synchronous(query, &query_params).await?;

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

    async fn get_cumulative_feedback_timeseries(
        &self,
        function_name: String,
        metric_name: String,
        variant_names: Option<Vec<String>>,
        time_window: TimeWindow,
        max_periods: u32,
    ) -> Result<Vec<CumulativeFeedbackTimeSeriesPoint>, Error> {
        // Convert TimeWindow to ClickHouse INTERVAL syntax and interval functions
        let (interval_str, interval_function) = match time_window {
            TimeWindow::Minute => ("INTERVAL 1 MINUTE", "toIntervalMinute"),
            TimeWindow::Hour => ("INTERVAL 1 HOUR", "toIntervalHour"),
            TimeWindow::Day => ("INTERVAL 1 DAY", "toIntervalDay"),
            TimeWindow::Week => ("INTERVAL 1 WEEK", "toIntervalWeek"),
            TimeWindow::Month => ("INTERVAL 1 MONTH", "toIntervalMonth"),
            TimeWindow::Cumulative => {
                return Err(Error::new(ErrorDetails::InvalidRequest {
                    message: "Cumulative time window is not supported for feedback timeseries"
                        .to_string(),
                }));
            }
        };

        // If variants are passed, build variant filter.
        // If None we don't filter at all;
        // If empty, we'll return an empty vector for consistency
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

        // Compute cumulative statistics from all historical data
        let query = format!(
            r"
            WITH
                -- CTE 1: Aggregate ALL historical data into time periods (no time filter)
                AggregatedFilteredFeedbackByVariantStatistics AS (
                    SELECT
                        toStartOfInterval(minute, {interval_str}) + {interval_str} AS period_end,
                        variant_name,

                        -- Apply -MergeState combinator to merge and keep as state for later merging
                        avgMergeState(feedback_mean) AS merged_mean_state,
                        varSampStableMergeState(feedback_variance) AS merged_var_state,
                        sum(count) AS period_count

                    FROM FeedbackByVariantStatistics

                    WHERE
                        function_name = {{function_name:String}}
                        AND metric_name = {{metric_name:String}}
                        {variant_filter}

                    GROUP BY
                        period_end,
                        variant_name
                ),

                -- CTE 2: For each variant, create sorted arrays of the periodic data.
                ArraysByVariant AS (
                    SELECT
                        variant_name,
                        -- 3. Unzip the sorted tuples back into individual arrays
                        arrayMap(x -> x.1, sorted_zipped_arrays) AS periods,
                        arrayMap(x -> x.2, sorted_zipped_arrays) AS mean_states,
                        arrayMap(x -> x.3, sorted_zipped_arrays) AS var_states,
                        arrayMap(x -> x.4, sorted_zipped_arrays) AS counts
                    FROM (
                        SELECT
                            variant_name,
                            (
                                -- 2. Sort the array of tuples based on the first element (the period_end)
                                arraySort(x -> x.1,
                                    -- 1. Zip the unsorted arrays together into an array of tuples
                                    arrayZip(
                                        groupArray(period_end),
                                        groupArray(merged_mean_state),
                                        groupArray(merged_var_state),
                                        groupArray(period_count)
                                    )
                                )
                            ) AS sorted_zipped_arrays
                        FROM AggregatedFilteredFeedbackByVariantStatistics
                        GROUP BY variant_name
                    )
                ),

                -- CTE 3: Compute cumulative stats for all periods
                AllCumulativeStats AS (
                    SELECT
                        periods[i] AS period_end,
                        variant_name,

                        arrayReduce('avgMerge', arraySlice(mean_states, 1, i)) AS mean,
                        arrayReduce('varSampStableMerge', arraySlice(var_states, 1, i)) AS variance,
                        arraySum(arraySlice(counts, 1, i)) AS count

                    FROM ArraysByVariant
                    ARRAY JOIN arrayEnumerate(periods) AS i
                ),

                -- CTE 4: Determine the window start time
                WindowStart AS (
                    SELECT
                        (SELECT max(period_end) FROM AllCumulativeStats) - {interval_function}({{max_periods:UInt32}}) AS window_start_time
                ),

                -- CTE 5: Get baseline cumulative values (the last value before the window starts)
                -- This ensures we have the cumulative count at the start of the window for all variants
                BaselineStats AS (
                    SELECT
                        variant_name,
                        mean,
                        variance,
                        count
                    FROM AllCumulativeStats
                    WHERE period_end < (SELECT window_start_time FROM WindowStart)
                    QUALIFY row_number() OVER (PARTITION BY variant_name ORDER BY period_end DESC) = 1
                ),

                -- CTE 6: Filter to only the most recent max_periods
                FilteredCumulativeStats AS (
                    SELECT
                        period_end,
                        variant_name,
                        mean,
                        variance,
                        count
                    FROM AllCumulativeStats
                    WHERE period_end >= (SELECT window_start_time FROM WindowStart)
                ),

                -- CTE 7: Create synthetic baseline entries at window start for variants with baseline data
                -- Only add synthetic entries for variants that DON'T already have data at window start
                SyntheticBaselineEntries AS (
                    SELECT
                        (SELECT window_start_time FROM WindowStart) AS period_end,
                        b.variant_name,
                        b.mean,
                        b.variance,
                        b.count
                    FROM BaselineStats b
                    WHERE b.variant_name NOT IN (
                        SELECT variant_name
                        FROM FilteredCumulativeStats
                        WHERE period_end = (SELECT window_start_time FROM WindowStart)
                    )
                ),

                -- CTE 8: Combine all data (baseline + window data)
                CombinedStats AS (
                    SELECT * FROM FilteredCumulativeStats
                    UNION ALL
                    SELECT * FROM SyntheticBaselineEntries
                )

            -- Final SELECT: Format the DateTime to string
            SELECT
                formatDateTime(period_end, '%Y-%m-%dT%H:%i:%SZ') AS period_end,
                variant_name,
                mean,
                variance,
                count,
                0.05 AS alpha,
                0.0 AS cs_lower,
                0.0 AS cs_upper
            FROM CombinedStats
            ORDER BY
                period_end ASC,
                variant_name ASC
            FORMAT JSONEachRow
            ",
        );

        // Create parameters HashMap
        let max_periods_str = max_periods.to_string();
        let params = std::collections::HashMap::from([
            ("function_name", function_name.as_str()),
            ("metric_name", metric_name.as_str()),
            ("max_periods", max_periods_str.as_str()),
        ]);

        let response = self.run_query_synchronous(query, &params).await?;

        // Deserialize the results into CumulativeFeedbackTimeSeriesPoint
        let feedback = response
            .response
            .trim()
            .lines()
            .map(|row| {
                serde_json::from_str(row).map_err(|e| {
                    Error::new(ErrorDetails::ClickHouseDeserialization {
                        message: format!(
                            "Failed to deserialize InternalCumulativeFeedbackTimeSeriesPoint: {e}"
                        ),
                    })
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Add confidence sequence
        asymp_cs(feedback, 0.05, None)
    }

    async fn query_feedback_by_target_id(
        &self,
        target_id: Uuid,
        before: Option<Uuid>,
        after: Option<Uuid>,
        limit: Option<u32>,
    ) -> Result<Vec<FeedbackRow>, Error> {
        if before.is_some() && after.is_some() {
            return Err(Error::new(crate::error::ErrorDetails::InvalidRequest {
                message: "Cannot specify both before and after in query_feedback_by_target_id"
                    .to_string(),
            }));
        }

        let limit = limit.unwrap_or(100);

        // Query all 4 feedback tables in parallel
        let (boolean_metrics, float_metrics, comment_feedback, demonstration_feedback) = tokio::join!(
            self.query_boolean_metrics_by_target_id(target_id, before, after, limit),
            self.query_float_metrics_by_target_id(target_id, before, after, limit),
            self.query_comment_feedback_by_target_id(target_id, before, after, limit),
            self.query_demonstration_feedback_by_inference_id(
                target_id,
                before,
                after,
                Some(limit)
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
            let start = all_feedback.len().saturating_sub(limit as usize);
            all_feedback.drain(start..).collect()
        } else {
            // If 'before' is specified or no pagination params, take latest elements
            all_feedback.truncate(limit as usize);
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
        limit: Option<u32>,
    ) -> Result<Vec<DemonstrationFeedbackRow>, Error> {
        if before.is_some() && after.is_some() {
            return Err(Error::new(crate::error::ErrorDetails::InvalidRequest {
                message: "Cannot specify both before and after in query_feedback_by_target_id"
                    .to_string(),
            }));
        }
        let limit = limit.unwrap_or(100);
        let (query, params_owned) =
            build_demonstration_feedback_query(inference_id, before, after, limit);

        let query_params: HashMap<&str, &str> = params_owned
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        let response = self.run_query_synchronous(query, &query_params).await?;

        parse_json_rows(response.response.as_str())
    }

    async fn query_metrics_with_feedback(
        &self,
        function_name: &str,
        function_config: &FunctionConfig,
        variant_name: Option<&str>,
    ) -> Result<Vec<MetricWithFeedback>, Error> {
        let inference_table = function_config.table_name();
        let (query, params_owned) =
            build_metrics_with_feedback_query(function_name, inference_table, variant_name);

        let query_params: HashMap<&str, &str> = params_owned
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        let response = self.run_query_synchronous(query, &query_params).await?;

        // Parse raw results and convert to MetricWithFeedback
        let raw_results: Vec<MetricWithFeedbackRaw> = parse_json_rows(response.response.as_str())?;

        Ok(raw_results
            .into_iter()
            .map(|raw| {
                // Demonstrations have a known type, others need config lookup at endpoint
                let metric_type = if raw.metric_name == "demonstration" {
                    Some(MetricType::Demonstration)
                } else {
                    None
                };
                MetricWithFeedback {
                    function_name: raw.function_name,
                    metric_name: raw.metric_name,
                    metric_type,
                    feedback_count: raw.feedback_count,
                }
            })
            .collect())
    }

    async fn query_latest_feedback_id_by_metric(
        &self,
        target_id: Uuid,
    ) -> Result<Vec<LatestFeedbackRow>, Error> {
        let (query, params_owned) = build_latest_feedback_id_by_metric_query(target_id);

        let query_params: HashMap<&str, &str> = params_owned
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        let response = self.run_query_synchronous(query, &query_params).await?;

        parse_json_rows(response.response.as_str())
    }

    async fn get_variant_performances(
        &self,
        params: GetVariantPerformanceParams<'_>,
    ) -> Result<Vec<VariantPerformanceRow>, Error> {
        let (query, params_owned) = build_variant_performance_query(&params);

        let query_params: HashMap<&str, &str> = params_owned
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        let response = self.run_query_synchronous(query, &query_params).await?;

        let result: Vec<VariantPerformanceRow> = response
            .response
            .lines()
            .filter(|line| !line.is_empty())
            .map(|line| {
                serde_json::from_str(line).map_err(|e| {
                    Error::new(ErrorDetails::ClickHouseDeserialization {
                        message: format!("Failed to deserialize VariantPerformanceRow: {e}"),
                    })
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(result)
    }

    // ===== Write methods =====

    async fn insert_boolean_feedback(
        &self,
        row: &BooleanMetricFeedbackInsert,
    ) -> Result<(), Error> {
        self.write_batched(&[row], super::TableName::BooleanMetricFeedback)
            .await
    }

    async fn insert_float_feedback(&self, row: &FloatMetricFeedbackInsert) -> Result<(), Error> {
        self.write_batched(&[row], super::TableName::FloatMetricFeedback)
            .await
    }

    async fn insert_comment_feedback(&self, row: &CommentFeedbackInsert) -> Result<(), Error> {
        self.write_batched(&[row], super::TableName::CommentFeedback)
            .await
    }

    async fn insert_demonstration_feedback(
        &self,
        row: &DemonstrationFeedbackInsert,
    ) -> Result<(), Error> {
        self.write_batched(&[row], super::TableName::DemonstrationFeedback)
            .await
    }

    async fn insert_static_eval_feedback(
        &self,
        row: &StaticEvaluationHumanFeedbackInsert,
    ) -> Result<(), Error> {
        self.write_batched(&[row], super::TableName::StaticEvaluationHumanFeedback)
            .await
    }
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use uuid::Uuid;

    use super::*;
    use crate::config::snapshot::SnapshotHash;
    use crate::db::clickhouse::clickhouse_client::MockClickHouseClient;
    use crate::db::clickhouse::{ClickHouseResponse, ClickHouseResponseMetadata};
    use crate::db::feedback::CommentTargetType;

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
        assert_query_contains(&query, "LIMIT {limit:UInt32}");
        assert_query_does_not_contain(&query, "AND toUInt128(id)");
        assert_eq!(params.get(id_column), Some(&id.to_string()));
        assert_eq!(params.get("limit"), Some(&"100".to_string()));
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
        assert_eq!(params.get("limit"), Some(&"50".to_string()));
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
        assert_query_contains(&query, "ORDER BY toUInt128(id) ASC LIMIT {limit:UInt32} )");
        assert_query_contains(&query, ") ORDER BY toUInt128(id) DESC");
        assert_eq!(params.get(id_column), Some(&id.to_string()));
        assert_eq!(params.get("after"), Some(&after.to_string()));
        assert_eq!(params.get("limit"), Some(&"25".to_string()));
    }

    fn test_bounds_query(table_name: &str, id_column: &str) {
        let id = Uuid::now_v7();
        let (query, params) = build_bounds_query(table_name, id_column, id);

        assert_query_contains(
            &query,
            &format!(
                "(SELECT id FROM {table_name} WHERE {id_column} = {{{id_column}:UUID}} ORDER BY toUInt128(id) ASC LIMIT 1) AS first_id"
            ),
        );
        assert_query_contains(
            &query,
            &format!(
                "(SELECT id FROM {table_name} WHERE {id_column} = {{{id_column}:UUID}} ORDER BY toUInt128(id) DESC LIMIT 1) AS last_id"
            ),
        );
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

    // Query execution tests with mocks

    #[tokio::test]
    async fn test_query_boolean_metrics_by_target_id_executes() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        let target_id = Uuid::now_v7();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(move |query, parameters| {
                assert_query_contains(query, "SELECT id, target_id, metric_name, value, tags");
                assert_query_contains(query, "FROM BooleanMetricFeedbackByTargetId");
                assert_query_contains(query, "WHERE target_id = {target_id:UUID}");
                assert_query_contains(query, "FORMAT JSONEachRow");

                assert_eq!(parameters.get("target_id"), Some(&target_id.to_string().as_str()));
                assert_eq!(parameters.get("limit"), Some(&"100"));

                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::from(r#"{"id":"0199cff5-3130-7e90-815c-91219e1a2dae","target_id":"0199cff5-3130-7e90-815c-91219e1a2dae","metric_name":"test_metric","value":true,"tags":{},"timestamp":"2021-01-01T00:00:00Z"}"#),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let result = conn
            .query_boolean_metrics_by_target_id(target_id, None, None, 100)
            .await
            .unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].metric_name, "test_metric");
        assert!(result[0].value);
    }

    #[tokio::test]
    async fn test_query_float_metrics_by_target_id_executes() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        let target_id = Uuid::now_v7();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(move |query, parameters| {
                assert_query_contains(query, "SELECT id, target_id, metric_name, value, tags");
                assert_query_contains(query, "FROM FloatMetricFeedbackByTargetId");
                assert_query_contains(query, "WHERE target_id = {target_id:UUID}");
                assert_query_contains(query, "FORMAT JSONEachRow");

                assert_eq!(parameters.get("target_id"), Some(&target_id.to_string().as_str()));
                assert_eq!(parameters.get("limit"), Some(&"100"));

                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::from(r#"{"id":"0199cff5-3130-7e90-815c-91219e1a2dae","target_id":"0199cff5-3130-7e90-815c-91219e1a2dae","metric_name":"accuracy","value":0.95,"tags":{},"timestamp":"2021-01-01T00:00:00Z"}"#),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let result = conn
            .query_float_metrics_by_target_id(target_id, None, None, 100)
            .await
            .unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].metric_name, "accuracy");
        assert_eq!(result[0].value, 0.95);
    }

    #[tokio::test]
    async fn test_query_comment_feedback_by_target_id_executes() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        let target_id = Uuid::now_v7();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(move |query, parameters| {
                assert_query_contains(query, "SELECT id, target_id, target_type, value, tags");
                assert_query_contains(query, "FROM CommentFeedbackByTargetId");
                assert_query_contains(query, "WHERE target_id = {target_id:UUID}");
                assert_query_contains(query, "FORMAT JSONEachRow");

                assert_eq!(parameters.get("target_id"), Some(&target_id.to_string().as_str()));
                assert_eq!(parameters.get("limit"), Some(&"100"));

                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::from(r#"{"id":"0199cff5-3130-7e90-815c-91219e1a2dae","target_id":"0199cff5-3130-7e90-815c-91219e1a2dae","target_type":"inference","value":"Great response!","tags":{},"timestamp":"2021-01-01T00:00:00Z"}"#),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let result = conn
            .query_comment_feedback_by_target_id(target_id, None, None, 100)
            .await
            .unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].value, "Great response!");
        // CommentTargetType is an enum, so we need to compare it properly
        assert!(matches!(
            result[0].target_type,
            CommentTargetType::Inference
        ));
    }

    #[tokio::test]
    async fn test_query_demonstration_feedback_by_inference_id_executes() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        let inference_id = Uuid::now_v7();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(move |query, parameters| {
                assert_query_contains(query, "SELECT id, inference_id, value, tags");
                assert_query_contains(query, "FROM DemonstrationFeedbackByInferenceId");
                assert_query_contains(query, "WHERE inference_id = {inference_id:UUID}");
                assert_query_contains(query, "FORMAT JSONEachRow");

                assert_eq!(parameters.get("inference_id"), Some(&inference_id.to_string().as_str()));
                assert_eq!(parameters.get("limit"), Some(&"100"));

                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::from(r#"{"id":"0199cff5-3130-7e90-815c-91219e1a2dae","inference_id":"0199cff5-3130-7e90-815c-91219e1a2dae","value":"demonstration value","tags":{},"timestamp":"2021-01-01T00:00:00Z"}"#),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let result = conn
            .query_demonstration_feedback_by_inference_id(inference_id, None, None, Some(100))
            .await
            .unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].value, "demonstration value");
    }

    // Bounds query tests with mocks

    #[tokio::test]
    async fn test_query_boolean_metric_bounds_by_target_id_executes() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        let target_id = Uuid::now_v7();
        let first_id = Uuid::now_v7();
        let last_id = Uuid::now_v7();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(move |query, parameters| {
                assert_query_contains(query, "(SELECT id FROM BooleanMetricFeedbackByTargetId WHERE target_id = {target_id:UUID} ORDER BY toUInt128(id) ASC LIMIT 1) AS first_id");
                assert_query_contains(query, "(SELECT id FROM BooleanMetricFeedbackByTargetId WHERE target_id = {target_id:UUID} ORDER BY toUInt128(id) DESC LIMIT 1) AS last_id");
                assert_query_contains(query, "FORMAT JSONEachRow");

                assert_eq!(parameters.get("target_id"), Some(&target_id.to_string().as_str()));

                true
            })
            .returning(move |_, _| {
                Ok(ClickHouseResponse {
                    response: format!(r#"{{"first_id":"{first_id}","last_id":"{last_id}"}}"#),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let result = conn
            .query_boolean_metric_bounds_by_target_id(target_id)
            .await
            .unwrap();

        assert_eq!(result.first_id, Some(first_id));
        assert_eq!(result.last_id, Some(last_id));
    }

    #[tokio::test]
    async fn test_query_float_metric_bounds_by_target_id_executes() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        let target_id = Uuid::now_v7();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(move |query, parameters| {
                assert_query_contains(query, "FROM FloatMetricFeedbackByTargetId");
                assert_eq!(
                    parameters.get("target_id"),
                    Some(&target_id.to_string().as_str())
                );
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::from(r#"{"first_id":null,"last_id":null}"#),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 0,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let result = conn
            .query_float_metric_bounds_by_target_id(target_id)
            .await
            .unwrap();

        assert_eq!(result.first_id, None);
        assert_eq!(result.last_id, None);
    }

    #[tokio::test]
    async fn test_query_comment_feedback_bounds_by_target_id_executes() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        let target_id = Uuid::now_v7();
        let first_id = Uuid::now_v7();
        let last_id = Uuid::now_v7();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(move |query, parameters| {
                assert_query_contains(query, "FROM CommentFeedbackByTargetId");
                assert_eq!(
                    parameters.get("target_id"),
                    Some(&target_id.to_string().as_str())
                );
                true
            })
            .returning(move |_, _| {
                Ok(ClickHouseResponse {
                    response: format!(r#"{{"first_id":"{first_id}","last_id":"{last_id}"}}"#),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let result = conn
            .query_comment_feedback_bounds_by_target_id(target_id)
            .await
            .unwrap();

        assert_eq!(result.first_id, Some(first_id));
        assert_eq!(result.last_id, Some(last_id));
    }

    #[tokio::test]
    async fn test_query_demonstration_feedback_bounds_by_inference_id_executes() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        let inference_id = Uuid::now_v7();
        let first_id = Uuid::now_v7();
        let last_id = Uuid::now_v7();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(move |query, parameters| {
                assert_query_contains(query, "FROM DemonstrationFeedbackByInferenceId");
                assert_eq!(
                    parameters.get("inference_id"),
                    Some(&inference_id.to_string().as_str())
                );
                true
            })
            .returning(move |_, _| {
                Ok(ClickHouseResponse {
                    response: format!(r#"{{"first_id":"{first_id}","last_id":"{last_id}"}}"#),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let result = conn
            .query_demonstration_feedback_bounds_by_inference_id(inference_id)
            .await
            .unwrap();

        assert_eq!(result.first_id, Some(first_id));
        assert_eq!(result.last_id, Some(last_id));
    }

    // Count query tests with mocks

    #[tokio::test]
    async fn test_count_boolean_metrics_by_target_id_executes() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        let target_id = Uuid::now_v7();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(move |query, parameters| {
                assert_query_contains(query, "SELECT toUInt64(COUNT()) AS count");
                assert_query_contains(query, "FROM BooleanMetricFeedbackByTargetId");
                assert_query_contains(query, "WHERE target_id = {target_id:UUID}");
                assert_query_contains(query, "FORMAT JSONEachRow");

                assert_eq!(
                    parameters.get("target_id"),
                    Some(&target_id.to_string().as_str())
                );

                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::from(r#"{"count":"5"}"#),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 5,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let result = conn
            .count_boolean_metrics_by_target_id(target_id)
            .await
            .unwrap();

        assert_eq!(result, 5);
    }

    #[tokio::test]
    async fn test_count_float_metrics_by_target_id_executes() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        let target_id = Uuid::now_v7();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(move |query, parameters| {
                assert_query_contains(query, "FROM FloatMetricFeedbackByTargetId");
                assert_eq!(
                    parameters.get("target_id"),
                    Some(&target_id.to_string().as_str())
                );
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::from(r#"{"count":"10"}"#),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 10,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let result = conn
            .count_float_metrics_by_target_id(target_id)
            .await
            .unwrap();

        assert_eq!(result, 10);
    }

    #[tokio::test]
    async fn test_count_comment_feedback_by_target_id_executes() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        let target_id = Uuid::now_v7();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(move |query, parameters| {
                assert_query_contains(query, "FROM CommentFeedbackByTargetId");
                assert_eq!(
                    parameters.get("target_id"),
                    Some(&target_id.to_string().as_str())
                );
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::from(r#"{"count":"0"}"#),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 0,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let result = conn
            .count_comment_feedback_by_target_id(target_id)
            .await
            .unwrap();

        assert_eq!(result, 0);
    }

    #[tokio::test]
    async fn test_count_demonstration_feedback_by_inference_id_executes() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        let inference_id = Uuid::now_v7();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(move |query, parameters| {
                assert_query_contains(query, "FROM DemonstrationFeedbackByInferenceId");
                assert_eq!(
                    parameters.get("inference_id"),
                    Some(&inference_id.to_string().as_str())
                );
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::from(r#"{"count":"3"}"#),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 3,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let result = conn
            .count_demonstration_feedback_by_inference_id(inference_id)
            .await
            .unwrap();

        assert_eq!(result, 3);
    }

    // FeedbackQueries trait method tests

    #[tokio::test]
    async fn test_query_feedback_by_target_id_rejects_both_before_and_after() {
        let mock_clickhouse_client = MockClickHouseClient::new();
        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let target_id = Uuid::now_v7();
        let before = Uuid::now_v7();
        let after = Uuid::now_v7();

        let result = conn
            .query_feedback_by_target_id(target_id, Some(before), Some(after), None)
            .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string()
                .contains("Cannot specify both before and after")
        );
    }

    #[tokio::test]
    async fn test_count_feedback_by_target_id_sums_all_types() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        let target_id = Uuid::now_v7();

        // Expect 4 queries (one for each feedback type)
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .times(4)
            .returning(|query, _| {
                // Return different counts based on the table being queried
                let count = if query.contains("BooleanMetricFeedbackByTargetId") {
                    "2"
                } else if query.contains("FloatMetricFeedbackByTargetId") {
                    "3"
                } else if query.contains("CommentFeedbackByTargetId") {
                    "1"
                } else if query.contains("DemonstrationFeedbackByInferenceId") {
                    "4"
                } else {
                    "0"
                };

                Ok(ClickHouseResponse {
                    response: format!(r#"{{"count":"{count}"}}"#),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 0,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let result = conn.count_feedback_by_target_id(target_id).await.unwrap();

        // Should sum all counts: 2 + 3 + 1 + 4 = 10
        assert_eq!(result, 10);
    }

    #[tokio::test]
    async fn test_query_feedback_bounds_by_target_id_aggregates_correctly() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();
        let target_id = Uuid::now_v7();

        // Create UUIDs for bounds - we'll make the first_id from boolean metrics the earliest
        // and the last_id from float metrics the latest
        let boolean_first = Uuid::from_u128(100);
        let boolean_last = Uuid::from_u128(200);
        let float_first = Uuid::from_u128(150);
        let float_last = Uuid::from_u128(300);
        let comment_first = Uuid::from_u128(120);
        let comment_last = Uuid::from_u128(250);
        let demo_first = Uuid::from_u128(110);
        let demo_last = Uuid::from_u128(280);

        // Expect 4 queries (one for each feedback type bounds)
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .times(4)
            .returning(move |query, _| {
                let (first, last) = if query.contains("BooleanMetricFeedbackByTargetId") {
                    (boolean_first, boolean_last)
                } else if query.contains("FloatMetricFeedbackByTargetId") {
                    (float_first, float_last)
                } else if query.contains("CommentFeedbackByTargetId") {
                    (comment_first, comment_last)
                } else if query.contains("DemonstrationFeedbackByInferenceId") {
                    (demo_first, demo_last)
                } else {
                    panic!("Unexpected table in bounds query");
                };

                Ok(ClickHouseResponse {
                    response: format!(r#"{{"first_id":"{first}","last_id":"{last}"}}"#),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 0,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let result = conn
            .query_feedback_bounds_by_target_id(target_id)
            .await
            .unwrap();

        // Should have the minimum first_id and maximum last_id across all feedback types
        assert_eq!(result.first_id, Some(boolean_first)); // 100 is the smallest
        assert_eq!(result.last_id, Some(float_last)); // 300 is the largest

        // Check individual bounds
        assert_eq!(result.by_type.boolean.first_id, Some(boolean_first));
        assert_eq!(result.by_type.boolean.last_id, Some(boolean_last));
        assert_eq!(result.by_type.float.first_id, Some(float_first));
        assert_eq!(result.by_type.float.last_id, Some(float_last));
    }

    #[tokio::test]
    async fn test_get_feedback_by_variant_with_no_variants() {
        let mock_clickhouse_client = MockClickHouseClient::new();
        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        // Empty variant list should return empty result immediately without querying
        let result = conn
            .get_feedback_by_variant("test_metric", "test_function", Some(&vec![]))
            .await
            .unwrap();

        assert_eq!(result.len(), 0);
    }

    #[tokio::test]
    async fn test_get_feedback_by_variant_filters_variants() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, params| {
                // Should filter by the specified variants
                assert!(query.contains("AND variant_name IN ('variant1', 'variant2')"));
                // Should use query parameters for function_name and metric_name
                assert_eq!(params.get("function_name"), Some(&"test_function"));
                assert_eq!(params.get("metric_name"), Some(&"test_metric"));
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::from(
                        r#"{"variant_name":"variant1","mean":0.85,"variance":0.01,"count":100}
{"variant_name":"variant2","mean":0.90,"variance":0.005,"count":50}"#,
                    ),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 2,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let variants = vec!["variant1".to_string(), "variant2".to_string()];
        let result = conn
            .get_feedback_by_variant("test_metric", "test_function", Some(&variants))
            .await
            .unwrap();

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].variant_name, "variant1");
        assert_eq!(result[0].mean, 0.85);
        assert_eq!(result[1].variant_name, "variant2");
        assert_eq!(result[1].mean, 0.90);
    }

    #[test]
    fn test_build_metrics_with_feedback_query_no_variant() {
        let (query, params) =
            build_metrics_with_feedback_query("test_function", "ChatInference", None);

        // Check that we query from FeedbackByVariantStatistics for boolean/float metrics
        assert_query_contains(&query, "FROM FeedbackByVariantStatistics");

        // Check that we query DemonstrationFeedback for demonstrations
        assert_query_contains(&query, "FROM DemonstrationFeedback");

        // Check demonstration metrics use inference_id with IN subquery
        assert_query_contains(
            &query,
            "WHERE inference_id IN (SELECT id FROM ChatInference WHERE function_name = {function_name:String}",
        );

        // Check no variant filter when variant_name is None
        assert_query_does_not_contain(&query, "variant_name");

        // Check UNION ALL
        assert_query_contains(&query, "UNION ALL");

        // Check ORDER BY
        assert_query_contains(&query, "ORDER BY metric_name");

        // Check params
        assert_eq!(
            params.get("function_name"),
            Some(&"test_function".to_string())
        );
        assert_eq!(params.get("variant_name"), None);
    }

    #[test]
    fn test_build_metrics_with_feedback_query_with_variant() {
        let (query, params) = build_metrics_with_feedback_query(
            "test_function",
            "JsonInference",
            Some("test_variant"),
        );

        // Check variant filter is present
        assert_query_contains(&query, "AND variant_name = {variant_name:String}");

        // Check uses JsonInference table for demonstrations
        assert_query_contains(&query, "FROM JsonInference WHERE");

        // Check params include variant_name
        assert_eq!(
            params.get("function_name"),
            Some(&"test_function".to_string())
        );
        assert_eq!(
            params.get("variant_name"),
            Some(&"test_variant".to_string())
        );
    }

    #[test]
    fn test_build_latest_feedback_id_by_metric_query() {
        let target_id = Uuid::now_v7();
        let (query, query_params) = build_latest_feedback_id_by_metric_query(target_id);

        // Verify the query structure
        assert_query_contains(&query, "SELECT");
        assert_query_contains(&query, "FROM BooleanMetricFeedbackByTargetId");
        assert_query_contains(&query, "FROM FloatMetricFeedbackByTargetId");
        assert_query_contains(&query, "argMax(id, toUInt128(id)) as latest_id");
        assert_query_contains(&query, "WHERE target_id = {target_id:String}");
        assert_query_contains(&query, "UNION ALL");
        assert_query_contains(&query, "GROUP BY metric_name");
        assert_query_contains(&query, "ORDER BY metric_name");
        assert_query_contains(&query, "FORMAT JSONEachRow");

        // Verify params
        assert_eq!(query_params.get("target_id"), Some(&target_id.to_string()));
    }

    // =====================================================================
    // Tests for get_variant_performances query building
    // =====================================================================

    use crate::config::{MetricConfig, MetricConfigLevel, MetricConfigOptimize, MetricConfigType};
    use crate::function::FunctionConfigType;

    fn make_inference_level_float_metric() -> MetricConfig {
        MetricConfig {
            r#type: MetricConfigType::Float,
            optimize: MetricConfigOptimize::Max,
            level: MetricConfigLevel::Inference,
            description: None,
        }
    }

    fn make_episode_level_float_metric() -> MetricConfig {
        MetricConfig {
            r#type: MetricConfigType::Float,
            optimize: MetricConfigOptimize::Max,
            level: MetricConfigLevel::Episode,
            description: None,
        }
    }

    fn make_inference_level_boolean_metric() -> MetricConfig {
        MetricConfig {
            r#type: MetricConfigType::Boolean,
            optimize: MetricConfigOptimize::Max,
            level: MetricConfigLevel::Inference,
            description: None,
        }
    }

    #[test]
    fn test_build_variant_performance_query_inference_level_cumulative() {
        let metric_config = make_inference_level_float_metric();
        let params = GetVariantPerformanceParams {
            function_name: "test_function",
            function_type: FunctionConfigType::Chat,
            metric_name: "test_metric",
            metric_config: &metric_config,
            time_window: TimeWindow::Cumulative,
            variant_name: None,
        };

        let (query, query_params) = build_variant_performance_query(&params);

        // Verify cumulative query structure
        assert_query_contains(&query, "'1970-01-01T00:00:00.000Z' AS period_start");
        assert_query_contains(&query, "FROM ChatInference i");
        assert_query_contains(&query, "FROM FloatMetricFeedback");
        assert_query_contains(&query, "WHERE metric_name = {metric_name:String}");
        assert_query_contains(&query, "ON i.id = f.target_id AND f.rn = 1"); // inference level join
        assert_query_contains(&query, "i.function_name = {function_name:String}");
        assert_query_contains(&query, "avg(f.value) AS avg_metric");
        assert_query_contains(&query, "stddevSamp(f.value) AS stdev");
        assert_query_contains(
            &query,
            "1.96 * (stddevSamp(f.value) / sqrt(count())) AS ci_error",
        );
        assert_query_contains(&query, "GROUP BY variant_name");
        assert_query_contains(&query, "ORDER BY variant_name ASC");
        assert_query_does_not_contain(&query, "AND i.variant_name = {variant_name:String}");

        // Verify params
        assert_eq!(
            query_params.get("function_name"),
            Some(&"test_function".to_string())
        );
        assert_eq!(
            query_params.get("metric_name"),
            Some(&"test_metric".to_string())
        );
        assert!(!query_params.contains_key("time_window_unit"));
    }

    #[test]
    fn test_build_variant_performance_query_inference_level_week() {
        let metric_config = make_inference_level_float_metric();
        let params = GetVariantPerformanceParams {
            function_name: "test_function",
            function_type: FunctionConfigType::Json,
            metric_name: "accuracy",
            metric_config: &metric_config,
            time_window: TimeWindow::Week,
            variant_name: None,
        };

        let (query, query_params) = build_variant_performance_query(&params);

        // Verify time-windowed query structure
        assert_query_contains(
            &query,
            "formatDateTime(dateTrunc({time_window_unit:String}, i.timestamp), '%Y-%m-%dT%H:%i:%S.000Z') AS period_start",
        );
        assert_query_contains(&query, "FROM JsonInference i");
        assert_query_contains(&query, "FROM FloatMetricFeedback");
        assert_query_contains(&query, "ON i.id = f.target_id AND f.rn = 1");
        assert_query_contains(&query, "GROUP BY period_start, variant_name");
        assert_query_contains(&query, "ORDER BY period_start ASC, variant_name ASC");

        // Verify params
        assert_eq!(
            query_params.get("time_window_unit"),
            Some(&"week".to_string())
        );
    }

    #[test]
    fn test_build_variant_performance_query_episode_level_cumulative() {
        let metric_config = make_episode_level_float_metric();
        let params = GetVariantPerformanceParams {
            function_name: "test_function",
            function_type: FunctionConfigType::Chat,
            metric_name: "episode_score",
            metric_config: &metric_config,
            time_window: TimeWindow::Cumulative,
            variant_name: None,
        };

        let (query, query_params) = build_variant_performance_query(&params);

        // Verify episode-level cumulative query structure
        assert_query_contains(&query, "'1970-01-01T00:00:00.000Z' AS period_start");
        assert_query_contains(&query, "WITH sub AS");
        assert_query_contains(&query, "i.episode_id AS episode_id");
        assert_query_contains(&query, "any(f.value) AS value_per_episode");
        assert_query_contains(&query, "ON i.episode_id = f.target_id AND f.rn = 1"); // episode level join
        assert_query_contains(&query, "GROUP BY variant_name, episode_id");
        assert_query_contains(&query, "avg(value_per_episode) AS avg_metric");
        assert_query_contains(&query, "stddevSamp(value_per_episode) AS stdev");

        // Verify no time window param for cumulative
        assert!(!query_params.contains_key("time_window_unit"));
    }

    #[test]
    fn test_build_variant_performance_query_episode_level_day() {
        let metric_config = make_episode_level_float_metric();
        let params = GetVariantPerformanceParams {
            function_name: "test_function",
            function_type: FunctionConfigType::Chat,
            metric_name: "episode_score",
            metric_config: &metric_config,
            time_window: TimeWindow::Day,
            variant_name: None,
        };

        let (query, query_params) = build_variant_performance_query(&params);

        // Verify episode-level time-windowed query structure
        assert_query_contains(&query, "WITH sub AS");
        assert_query_contains(
            &query,
            "dateTrunc({time_window_unit:String}, i.timestamp) AS period_start",
        );
        assert_query_contains(&query, "ON i.episode_id = f.target_id AND f.rn = 1");
        assert_query_contains(&query, "GROUP BY period_start, variant_name, episode_id");
        assert_query_contains(
            &query,
            "formatDateTime(period_start, '%Y-%m-%dT%H:%i:%S.000Z') AS period_start",
        );

        // Verify params
        assert_eq!(
            query_params.get("time_window_unit"),
            Some(&"day".to_string())
        );
    }

    #[test]
    fn test_build_variant_performance_query_with_variant_filter() {
        let metric_config = make_inference_level_float_metric();
        let params = GetVariantPerformanceParams {
            function_name: "test_function",
            function_type: FunctionConfigType::Chat,
            metric_name: "test_metric",
            metric_config: &metric_config,
            time_window: TimeWindow::Cumulative,
            variant_name: Some("specific_variant"),
        };

        let (query, query_params) = build_variant_performance_query(&params);

        // Verify variant filter is present
        assert_query_contains(&query, "AND i.variant_name = {variant_name:String}");

        // Verify params include variant
        assert_eq!(
            query_params.get("variant_name"),
            Some(&"specific_variant".to_string())
        );
    }

    #[test]
    fn test_build_variant_performance_query_boolean_metric() {
        let metric_config = make_inference_level_boolean_metric();
        let params = GetVariantPerformanceParams {
            function_name: "test_function",
            function_type: FunctionConfigType::Chat,
            metric_name: "success",
            metric_config: &metric_config,
            time_window: TimeWindow::Month,
            variant_name: None,
        };

        let (query, query_params) = build_variant_performance_query(&params);

        // Verify boolean metric uses BooleanMetricFeedback
        assert_query_contains(&query, "FROM BooleanMetricFeedback");
        assert_query_contains(&query, "FROM ChatInference i");

        // Verify params
        assert_eq!(
            query_params.get("time_window_unit"),
            Some(&"month".to_string())
        );
        assert_eq!(
            query_params.get("metric_name"),
            Some(&"success".to_string())
        );
    }

    #[tokio::test]
    async fn test_get_variant_performances_executes() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(|query, params| {
                assert_query_contains(query, "FROM ChatInference i");
                assert_query_contains(query, "FROM FloatMetricFeedback");
                assert_query_contains(query, "avg(f.value) AS avg_metric");
                assert_eq!(params.get("function_name"), Some(&"test_function"));
                assert_eq!(params.get("metric_name"), Some(&"accuracy"));
                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: r#"{"period_start":"1970-01-01T00:00:00.000Z","variant_name":"variant_a","count":10,"avg_metric":0.85,"stdev":0.05,"ci_error":0.031}
{"period_start":"1970-01-01T00:00:00.000Z","variant_name":"variant_b","count":15,"avg_metric":0.90,"stdev":0.03,"ci_error":0.015}"#.to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 2,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let metric_config = make_inference_level_float_metric();
        let params = GetVariantPerformanceParams {
            function_name: "test_function",
            function_type: FunctionConfigType::Chat,
            metric_name: "accuracy",
            metric_config: &metric_config,
            time_window: TimeWindow::Cumulative,
            variant_name: None,
        };

        let result = conn.get_variant_performances(params).await.unwrap();

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].variant_name, "variant_a");
        assert_eq!(result[0].count, 10);
        assert!((result[0].avg_metric - 0.85).abs() < 0.001);
        assert_eq!(result[1].variant_name, "variant_b");
        assert_eq!(result[1].count, 15);
        assert!((result[1].avg_metric - 0.90).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_get_variant_performances_empty_result() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
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
        let metric_config = make_inference_level_float_metric();
        let params = GetVariantPerformanceParams {
            function_name: "nonexistent_function",
            function_type: FunctionConfigType::Chat,
            metric_name: "accuracy",
            metric_config: &metric_config,
            time_window: TimeWindow::Cumulative,
            variant_name: None,
        };

        let result = conn.get_variant_performances(params).await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_get_variant_performances_with_null_stdev() {
        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .returning(|_, _| {
                // stdev and ci_error are null when count = 1
                Ok(ClickHouseResponse {
                    response: r#"{"period_start":"2024-01-01T00:00:00.000Z","variant_name":"variant_a","count":1,"avg_metric":0.9,"stdev":null,"ci_error":null}"#.to_string(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let metric_config = make_inference_level_float_metric();
        let params = GetVariantPerformanceParams {
            function_name: "test_function",
            function_type: FunctionConfigType::Chat,
            metric_name: "accuracy",
            metric_config: &metric_config,
            time_window: TimeWindow::Week,
            variant_name: None,
        };

        let result = conn.get_variant_performances(params).await.unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].count, 1);
        assert!(result[0].stdev.is_none());
        assert!(result[0].ci_error.is_none());
    }

    // ===== Write method tests =====

    #[tokio::test]
    async fn test_insert_boolean_feedback() {
        let feedback_id = Uuid::now_v7();
        let target_id = Uuid::now_v7();

        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_write_batched_internal()
            .withf(move |rows, table| {
                assert_eq!(*table, super::super::TableName::BooleanMetricFeedback);
                assert_eq!(rows.len(), 1);
                let row: serde_json::Value = serde_json::from_str(&rows[0]).unwrap();
                assert_eq!(row["id"], feedback_id.to_string());
                assert_eq!(row["target_id"], target_id.to_string());
                assert_eq!(row["metric_name"], "test_metric");
                assert_eq!(row["value"], true);
                true
            })
            .returning(|_, _| Ok(()));

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let insert = BooleanMetricFeedbackInsert {
            id: feedback_id,
            target_id,
            metric_name: "test_metric".to_string(),
            value: true,
            tags: HashMap::new(),
            snapshot_hash: SnapshotHash::new_test(),
        };

        conn.insert_boolean_feedback(&insert).await.unwrap();
    }

    #[tokio::test]
    async fn test_insert_float_feedback() {
        let feedback_id = Uuid::now_v7();
        let target_id = Uuid::now_v7();

        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_write_batched_internal()
            .withf(move |rows, table| {
                assert_eq!(*table, super::super::TableName::FloatMetricFeedback);
                assert_eq!(rows.len(), 1);
                let row: serde_json::Value = serde_json::from_str(&rows[0]).unwrap();
                assert_eq!(row["id"], feedback_id.to_string());
                assert_eq!(row["target_id"], target_id.to_string());
                assert_eq!(row["metric_name"], "accuracy");
                assert!((row["value"].as_f64().unwrap() - 0.95).abs() < 0.001);
                true
            })
            .returning(|_, _| Ok(()));

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let insert = FloatMetricFeedbackInsert {
            id: feedback_id,
            target_id,
            metric_name: "accuracy".to_string(),
            value: 0.95,
            tags: HashMap::new(),
            snapshot_hash: SnapshotHash::new_test(),
        };

        conn.insert_float_feedback(&insert).await.unwrap();
    }

    #[tokio::test]
    async fn test_insert_comment_feedback() {
        let feedback_id = Uuid::now_v7();
        let target_id = Uuid::now_v7();

        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_write_batched_internal()
            .withf(move |rows, table| {
                assert_eq!(*table, super::super::TableName::CommentFeedback);
                assert_eq!(rows.len(), 1);
                let row: serde_json::Value = serde_json::from_str(&rows[0]).unwrap();
                assert_eq!(row["id"], feedback_id.to_string());
                assert_eq!(row["target_id"], target_id.to_string());
                assert_eq!(row["target_type"], "inference");
                assert_eq!(row["value"], "Great response!");
                true
            })
            .returning(|_, _| Ok(()));

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let insert = CommentFeedbackInsert {
            id: feedback_id,
            target_id,
            target_type: CommentTargetType::Inference,
            value: "Great response!".to_string(),
            tags: HashMap::new(),
            snapshot_hash: SnapshotHash::new_test(),
        };

        conn.insert_comment_feedback(&insert).await.unwrap();
    }

    #[tokio::test]
    async fn test_insert_comment_feedback_episode_level() {
        let feedback_id = Uuid::now_v7();
        let target_id = Uuid::now_v7();

        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_write_batched_internal()
            .withf(move |rows, table| {
                assert_eq!(*table, super::super::TableName::CommentFeedback);
                assert_eq!(rows.len(), 1);
                let row: serde_json::Value = serde_json::from_str(&rows[0]).unwrap();
                assert_eq!(row["target_type"], "episode");
                true
            })
            .returning(|_, _| Ok(()));

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let insert = CommentFeedbackInsert {
            id: feedback_id,
            target_id,
            target_type: CommentTargetType::Episode,
            value: "Episode feedback".to_string(),
            tags: HashMap::new(),
            snapshot_hash: SnapshotHash::new_test(),
        };

        conn.insert_comment_feedback(&insert).await.unwrap();
    }

    #[tokio::test]
    async fn test_insert_demonstration_feedback() {
        let feedback_id = Uuid::now_v7();
        let inference_id = Uuid::now_v7();

        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_write_batched_internal()
            .withf(move |rows, table| {
                assert_eq!(*table, super::super::TableName::DemonstrationFeedback);
                assert_eq!(rows.len(), 1);
                let row: serde_json::Value = serde_json::from_str(&rows[0]).unwrap();
                assert_eq!(row["id"], feedback_id.to_string());
                assert_eq!(row["inference_id"], inference_id.to_string());
                assert_eq!(row["value"], r#"{"content":"Hello"}"#);
                true
            })
            .returning(|_, _| Ok(()));

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let insert = DemonstrationFeedbackInsert {
            id: feedback_id,
            inference_id,
            value: r#"{"content":"Hello"}"#.to_string(),
            tags: HashMap::new(),
            snapshot_hash: SnapshotHash::new_test(),
        };

        conn.insert_demonstration_feedback(&insert).await.unwrap();
    }

    #[tokio::test]
    async fn test_insert_static_eval_feedback() {
        let feedback_id = Uuid::now_v7();
        let datapoint_id = Uuid::now_v7();
        let evaluator_inference_id = Uuid::now_v7();

        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_write_batched_internal()
            .withf(move |rows, table| {
                assert_eq!(
                    *table,
                    super::super::TableName::StaticEvaluationHumanFeedback
                );
                assert_eq!(rows.len(), 1);
                let row: serde_json::Value = serde_json::from_str(&rows[0]).unwrap();
                assert_eq!(row["feedback_id"], feedback_id.to_string());
                assert_eq!(row["metric_name"], "quality");
                assert_eq!(row["datapoint_id"], datapoint_id.to_string());
                assert_eq!(row["output"], "test output");
                assert_eq!(row["value"], "0.9");
                assert_eq!(
                    row["evaluator_inference_id"],
                    evaluator_inference_id.to_string()
                );
                true
            })
            .returning(|_, _| Ok(()));

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let insert = StaticEvaluationHumanFeedbackInsert {
            feedback_id,
            metric_name: "quality".to_string(),
            datapoint_id,
            output: "test output".to_string(),
            value: "0.9".to_string(),
            evaluator_inference_id: Some(evaluator_inference_id),
        };

        conn.insert_static_eval_feedback(&insert).await.unwrap();
    }

    #[tokio::test]
    async fn test_insert_feedback_with_tags() {
        let feedback_id = Uuid::now_v7();
        let target_id = Uuid::now_v7();

        let mut tags = HashMap::new();
        tags.insert("user_id".to_string(), "user_123".to_string());
        tags.insert("session".to_string(), "abc".to_string());

        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_write_batched_internal()
            .withf(move |rows, _table| {
                let row: serde_json::Value = serde_json::from_str(&rows[0]).unwrap();
                let row_tags = row["tags"].as_object().unwrap();
                assert_eq!(row_tags.get("user_id").unwrap(), "user_123");
                assert_eq!(row_tags.get("session").unwrap(), "abc");
                true
            })
            .returning(|_, _| Ok(()));

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let insert = BooleanMetricFeedbackInsert {
            id: feedback_id,
            target_id,
            metric_name: "liked".to_string(),
            value: true,
            tags: {
                let mut t = HashMap::new();
                t.insert("user_id".to_string(), "user_123".to_string());
                t.insert("session".to_string(), "abc".to_string());
                t
            },
            snapshot_hash: SnapshotHash::new_test(),
        };

        conn.insert_boolean_feedback(&insert).await.unwrap();
    }
}
