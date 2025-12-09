//! ClickHouse queries for model inferences.

use std::collections::HashMap;

use async_trait::async_trait;
use uuid::Uuid;

use super::ClickHouseConnectionInfo;
use crate::db::model_inferences::ModelInferenceQueries;
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
}

#[cfg(test)]
mod tests {
    use crate::db::clickhouse::query_builder::test_util::assert_query_contains;

    #[test]
    fn test_query_structure() {
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
                formatDateTime(timestamp, '%Y-%m-%dT%H:%i:%SZ') as timestamp
            FROM ModelInference
            WHERE inference_id = {inference_id:UUID}
            FORMAT JSONEachRow
        ";

        assert_query_contains(query, "FROM ModelInference");
        assert_query_contains(query, "inference_id = {inference_id:UUID}");
        assert_query_contains(query, "FORMAT JSONEachRow");
    }
}
