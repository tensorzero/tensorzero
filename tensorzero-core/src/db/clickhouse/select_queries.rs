use crate::{
    db::{
        clickhouse::migration_manager::migrations::migration_0037::quantiles_sql_args,
        EpisodeByIdRow, TableBoundsWithCount,
    },
    serde_util::deserialize_u64,
};
use async_trait::async_trait;
use serde::Deserialize;
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
            TimeWindow::Minute => (
                "toStartOfMinute(minute)",
                format!("minute >= (SELECT max(toStartOfMinute(minute)) FROM ModelProviderStatistics) - INTERVAL {max_periods} MINUTE"),
            ),
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
            TimeWindow::Minute => {
                "minute >= (SELECT max(minute) FROM ModelProviderStatistics) - INTERVAL 1 MINUTE"
            }
            TimeWindow::Hour => {
                "minute >= (SELECT max(minute) FROM ModelProviderStatistics) - INTERVAL 1 HOUR"
            }
            TimeWindow::Day => {
                "minute >= (SELECT max(minute) FROM ModelProviderStatistics) - INTERVAL 1 DAY"
            }
            TimeWindow::Week => {
                "minute >= (SELECT max(minute) FROM ModelProviderStatistics) - INTERVAL 1 WEEK"
            }
            TimeWindow::Month => {
                "minute >= (SELECT max(minute) FROM ModelProviderStatistics) - INTERVAL 1 MONTH"
            }
            TimeWindow::Cumulative => "1 = 1",
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
        limit: u32,
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
        // So, we select the last limit_overestimate episodes in descending order
        // Then we group by episode_id_uint and count the number of inferences
        // Finally, we order by episode_id_uint correctly and limit the result to limit
        let limit_overestimate = 5 * limit;

        let query = if is_after {
            format!(
                r"WITH potentially_duplicated_episode_ids AS (
                    SELECT episode_id_uint
                    FROM EpisodeById
                    WHERE {where_clause}
                    ORDER BY episode_id_uint ASC
                    LIMIT {limit_overestimate}
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
                    LIMIT {limit}
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
                    LIMIT {limit_overestimate}
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
                LIMIT {limit}
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

// Helper functions for parsing responses
pub(crate) fn build_pagination_clause(
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

pub(crate) fn parse_json_rows<T: serde::de::DeserializeOwned>(
    response: &str,
) -> Result<Vec<T>, Error> {
    response
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|row| {
            serde_json::from_str(row).map_err(|e| {
                Error::new(ErrorDetails::ClickHouseDeserialization {
                    message: format!("Failed to deserialize row: {e}"),
                })
            })
        })
        .collect()
}

pub(crate) fn parse_count(response: &str) -> Result<u64, Error> {
    #[derive(Deserialize)]
    struct CountResult {
        #[serde(deserialize_with = "deserialize_u64")]
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
