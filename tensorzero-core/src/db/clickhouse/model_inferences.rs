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
