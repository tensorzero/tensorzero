//! ClickHouse queries for model inferences.

use std::collections::HashMap;

use async_trait::async_trait;
use uuid::Uuid;

use super::ClickHouseConnectionInfo;
use super::migration_manager::migrations::migration_0037::{QUANTILES, quantiles_sql_args};
use super::table_name::TableName;
use crate::db::model_inferences::ModelInferenceQueries;
use crate::db::{ModelLatencyDatapoint, ModelUsageTimePoint, TimeWindow};
use crate::error::{Error, ErrorDetails};
use crate::inference::types::StoredModelInference;

#[async_trait]
impl ModelInferenceQueries for ClickHouseConnectionInfo {
    async fn get_model_inferences_by_inference_id(
        &self,
        inference_id: Uuid,
    ) -> Result<Vec<StoredModelInference>, Error> {
        let query = r"
            SELECT
                id,
                inference_id,
                raw_request,
                raw_response,
                system,
                input_messages,
                output,
                input_tokens,
                output_tokens,
                response_time_ms,
                model_name,
                model_provider_name,
                ttft_ms,
                cached,
                finish_reason,
                snapshot_hash,
                cost,
                formatDateTime(timestamp, '%Y-%m-%dT%H:%i:%SZ') as timestamp
            FROM ModelInference
            WHERE inference_id = {inference_id:UUID}
            FORMAT JSONEachRow
        "
        .to_string();

        let inference_id_str = inference_id.to_string();
        let params: HashMap<&str, &str> =
            HashMap::from([("inference_id", inference_id_str.as_str())]);

        let response = self.run_query_synchronous(query, &params).await?;

        if response.response.is_empty() {
            return Ok(vec![]);
        }

        // Parse newline-delimited JSON (JSONEachRow format)
        let rows: Vec<StoredModelInference> = response
            .response
            .lines()
            .filter(|line| !line.is_empty())
            .map(|line| {
                serde_json::from_str(line).map_err(|e| {
                    Error::new(ErrorDetails::ClickHouseDeserialization {
                        message: format!("Failed to parse StoredModelInference row: {e}"),
                    })
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(rows)
    }

    async fn insert_model_inferences(&self, rows: &[StoredModelInference]) -> Result<(), Error> {
        if rows.is_empty() {
            return Ok(());
        }

        // Serialize each row to JSON for ClickHouse
        let serialized: Vec<serde_json::Value> = rows
            .iter()
            .map(|row| {
                serde_json::to_value(row).map_err(|e| {
                    Error::new(ErrorDetails::Serialization {
                        message: format!("Failed to serialize StoredModelInference: {e}"),
                    })
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        self.write_batched(&serialized, TableName::ModelInference)
            .await
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
                format!(
                    "minute >= (SELECT max(toStartOfMinute(minute)) FROM ModelProviderStatistics) - INTERVAL {max_periods} MINUTE"
                ),
            ),
            TimeWindow::Hour => (
                "toStartOfHour(minute)",
                format!(
                    "minute >= (SELECT max(toStartOfHour(minute)) FROM ModelProviderStatistics) - INTERVAL {max_periods} HOUR"
                ),
            ),
            TimeWindow::Day => (
                "toStartOfDay(minute)",
                format!(
                    "minute >= (SELECT max(toStartOfDay(minute)) FROM ModelProviderStatistics) - INTERVAL {max_periods} DAY"
                ),
            ),
            TimeWindow::Week => (
                "toStartOfWeek(minute)",
                format!(
                    "minute >= (SELECT max(toStartOfWeek(minute)) FROM ModelProviderStatistics) - INTERVAL {max_periods} WEEK"
                ),
            ),
            TimeWindow::Month => (
                "toStartOfMonth(minute)",
                format!(
                    "minute >= (SELECT max(toStartOfMonth(minute)) FROM ModelProviderStatistics) - INTERVAL {max_periods} MONTH"
                ),
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

    fn get_model_latency_quantile_function_inputs(&self) -> &[f64] {
        QUANTILES
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use uuid::Uuid;

    use crate::db::{
        clickhouse::{
            ClickHouseConnectionInfo, ClickHouseResponse, ClickHouseResponseMetadata,
            clickhouse_client::MockClickHouseClient,
            query_builder::test_util::assert_query_contains,
        },
        model_inferences::ModelInferenceQueries,
    };

    #[tokio::test]
    async fn test_get_model_inferences_by_inference_id() {
        let inference_id_str = "0196ee9c-d808-74f3-8000-02ec7409b95d";

        let mut mock_clickhouse_client = MockClickHouseClient::new();

        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(move |query, params| {
                // Should order by search relevance first, then by timestamp
                assert_query_contains(query, "
                SELECT
                    id,
                    inference_id,
                    raw_request,
                    raw_response,
                    system,
                    input_messages,
                    output,
                    input_tokens,
                    output_tokens,
                    response_time_ms,
                    model_name,
                    model_provider_name,
                    ttft_ms,
                    cached,
                    finish_reason,
                    snapshot_hash,
                    cost,
                    formatDateTime(timestamp, '%Y-%m-%dT%H:%i:%SZ') as timestamp
                FROM ModelInference
                WHERE inference_id = {inference_id:UUID}
                FORMAT JSONEachRow");

                assert_eq!(params.get("inference_id"), Some(&inference_id_str));

                true
            })
            .returning(|_, _| {
                Ok(ClickHouseResponse {
                    response: String::from(r#"{"id":"0196ee9c-d808-74f3-8000-039e871ca8a5","inference_id":"0196ee9c-d808-74f3-8000-02ec7409b95d","raw_request":"raw request","raw_response":"{\n  \"id\": \"id\",\n  \"object\": \"text.completion\",\n  \"created\": 1618870400,\n  \"model\": \"text-davinci-002\",\n  \"choices\": [\n    {\n      \"text\": \"Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake.\",\n      \"index\": 0,\n      \"logprobs\": null,\n      \"finish_reason\": null\n    }\n  ]\n}","system":"You are an assistant that is performing a named entity recognition task.\nYour job is to extract entities from a given text.\n\nThe entities you are extracting are:\n- people\n- organizations\n- locations\n- miscellaneous other entities\n\nPlease return the entities in the following JSON format:\n\n{\n    \"person\": [\"person1\", \"person2\", ...],\n    \"organization\": [\"organization1\", \"organization2\", ...],\n    \"location\": [\"location1\", \"location2\", ...],\n    \"miscellaneous\": [\"miscellaneous1\", \"miscellaneous2\", ...]\n}","input_messages":"[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"My input prefix : Random 0196ee9c-d808-74f3-8000-02d9b57169b5\"}]}]","output":"[{\"type\":\"text\",\"text\":\"Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake.\"}]","input_tokens":10,"output_tokens":10,"response_time_ms":100,"model_name":"dummy::good","model_provider_name":"dummy","ttft_ms":null,"cached":false,"finish_reason":"stop","snapshot_hash":null,"timestamp":"2025-05-20T16:52:58Z"}"#),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });
        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let result = conn
            .get_model_inferences_by_inference_id(Uuid::parse_str(inference_id_str).unwrap())
            .await
            .unwrap();

        assert_eq!(result.len(), 1, "Should return one datapoint");
    }
}
