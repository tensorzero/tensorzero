use crate::db::{
    clickhouse::migration_manager::migrations::migration_0037::quantiles_sql_args, EpisodeByIdRow,
    TableBoundsWithCount,
};
use async_trait::async_trait;
use uuid::Uuid;

use crate::{
    db::{ModelLatencyDatapoint, ModelUsageTimePoint, SelectQueries, TimeWindow},
    error::{Error, ErrorDetails},
};

use super::ClickHouseConnectionInfo;

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
}
