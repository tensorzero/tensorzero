use crate::db::{
    clickhouse::migration_manager::migrations::migration_0037::quantiles_sql_args, EpisodeByIdRow,
    FeedbackByVariant, FeedbackTimeSeriesPoint, TableBoundsWithCount,
};
use async_trait::async_trait;
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
            TimeWindow::Minute => (
                "toStartOfMinute(minute)".to_string(),
                format!("minute >= (SELECT max(toStartOfMinute(minute)) FROM ModelProviderStatistics) - INTERVAL {max_periods} MINUTE"),
            ),
            TimeWindow::Hour => (
                "toStartOfHour(minute)".to_string(),
                format!("minute >= (SELECT max(toStartOfHour(minute)) FROM ModelProviderStatistics) - INTERVAL {max_periods} HOUR"),
            ),
            TimeWindow::Day => (
                "toStartOfDay(minute)".to_string(),
                format!("minute >= (SELECT max(toStartOfDay(minute)) FROM ModelProviderStatistics) - INTERVAL {max_periods} DAY"),
            ),
            TimeWindow::Week => (
                "toStartOfWeek(minute)".to_string(),
                format!("minute >= (SELECT max(toStartOfWeek(minute)) FROM ModelProviderStatistics) - INTERVAL {max_periods} WEEK"),
            ),
            TimeWindow::Month => (
                "toStartOfMonth(minute)".to_string(),
                format!("minute >= (SELECT max(toStartOfMonth(minute)) FROM ModelProviderStatistics) - INTERVAL {max_periods} MONTH"),
            ),
            TimeWindow::Cumulative => (
                "toDateTime('1970-01-01 00:00:00')".to_string(),
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
                    .to_string()
            }
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

    async fn get_feedback_timeseries(
        &self,
        function_name: String,
        metric_name: String,
        variant_names: Option<Vec<String>>,
        time_window: TimeWindow,
        max_periods: u32,
    ) -> Result<Vec<FeedbackTimeSeriesPoint>, Error> {
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
                }))
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

                -- CTE 4: Filter to only the most recent max_periods (DateTime arithmetic on DateTime types)
                FilteredCumulativeStats AS (
                    SELECT
                        period_end,
                        variant_name,
                        mean,
                        variance,
                        count
                    FROM AllCumulativeStats
                    WHERE period_end >= (
                        SELECT max(period_end)
                        FROM AllCumulativeStats
                    ) - {interval_function}({{max_periods:UInt32}})
                )

            -- Final SELECT: Format the DateTime to string
            SELECT
                formatDateTime(period_end, '%Y-%m-%dT%H:%i:%SZ') AS period_end,
                variant_name,
                mean,
                variance,
                count
            FROM FilteredCumulativeStats
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

        // Deserialize the results into FeedbackTimeSeriesPoint
        response
            .response
            .trim()
            .lines()
            .map(|row| {
                serde_json::from_str(row).map_err(|e| {
                    Error::new(ErrorDetails::ClickHouseDeserialization {
                        message: format!("Failed to deserialize FeedbackTimeSeriesPoint: {e}"),
                    })
                })
            })
            .collect::<Result<Vec<_>, _>>()
    }
}
