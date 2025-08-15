use async_trait::async_trait;

use crate::{
    db::{ModelUsageTimePoint, SelectQueries, TimeWindow},
    error::{Error, ErrorDetails},
};

use super::ClickHouseConnectionInfo;

#[async_trait]
impl SelectQueries for ClickHouseConnectionInfo {
    async fn get_model_usage_timeseries(
        &self,
        time_window: TimeWindow,
        max_periods: u32,
    ) -> Result<Vec<ModelUsageTimePoint>, Error> {
        // TODO: probably factor this out into common code as other queries will likely need similar logic
        // TODO: write e2e tests against this query (outside the trait)
        let (time_grouping, time_filter) = match time_window {
            TimeWindow::Hour => (
                "toStartOfHour(minute)",
                format!("minute >= (SELECT max(minute) FROM ModelProviderStatistics) - INTERVAL {max_periods} HOUR"),
            ),
            TimeWindow::Day => (
                "toStartOfDay(minute)",
                format!("minute >= (SELECT max(minute) FROM ModelProviderStatistics) - INTERVAL {max_periods} DAY"),
            ),
            TimeWindow::Week => (
                "toStartOfWeek(minute)",
                format!("minute >= (SELECT max(minute) FROM ModelProviderStatistics) - INTERVAL {max_periods} WEEK"),
            ),
            TimeWindow::Month => (
                "toStartOfMonth(minute)",
                format!("minute >= (SELECT max(minute) FROM ModelProviderStatistics) - INTERVAL {max_periods} MONTH"),
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
        // This part depends on how your query API returns data
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
}
