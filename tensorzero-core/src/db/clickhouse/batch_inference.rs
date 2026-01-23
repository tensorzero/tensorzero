/// ClickHouse implementation of BatchInferenceQueries trait.
use async_trait::async_trait;
use std::collections::HashMap;
use uuid::Uuid;

use crate::db::batch_inference::{BatchInferenceQueries, CompletedBatchInferenceRow};
use crate::db::clickhouse::ClickHouseConnectionInfo;
use crate::error::{Error, ErrorDetails};
use crate::inference::types::batch::{BatchModelInferenceRow, BatchRequestRow};

#[async_trait]
impl BatchInferenceQueries for ClickHouseConnectionInfo {
    async fn get_batch_request(
        &self,
        batch_id: Uuid,
        inference_id: Option<Uuid>,
    ) -> Result<Option<BatchRequestRow<'static>>, Error> {
        let (sql, params) = match inference_id {
            None => {
                let sql = r"
                    SELECT
                        batch_id,
                        id,
                        batch_params,
                        model_name,
                        model_provider_name,
                        status,
                        function_name,
                        variant_name,
                        raw_request,
                        raw_response,
                        errors
                    FROM BatchRequest
                    WHERE batch_id = {batch_id:UUID}
                    ORDER BY timestamp DESC
                    LIMIT 1
                    FORMAT JSONEachRow
                "
                .to_string();
                let mut params = HashMap::new();
                params.insert("batch_id", batch_id.to_string());
                (sql, params)
            }
            Some(inference_id) => {
                let sql = r"
                    SELECT br.batch_id as batch_id,
                        br.id as id,
                        br.batch_params as batch_params,
                        br.model_name as model_name,
                        br.model_provider_name as model_provider_name,
                        br.status as status,
                        br.function_name as function_name,
                        br.variant_name as variant_name,
                        br.raw_request as raw_request,
                        br.raw_response as raw_response,
                        br.errors as errors
                    FROM BatchIdByInferenceId bi
                    JOIN BatchRequest br ON bi.batch_id = br.batch_id
                    WHERE bi.inference_id = {inference_id:UUID} AND bi.batch_id = {batch_id:UUID}
                    ORDER BY br.timestamp DESC
                    LIMIT 1
                    FORMAT JSONEachRow
                "
                .to_string();
                let mut params = HashMap::new();
                params.insert("batch_id", batch_id.to_string());
                params.insert("inference_id", inference_id.to_string());
                (sql, params)
            }
        };

        let query_params: HashMap<&str, &str> =
            params.iter().map(|(k, v)| (*k, v.as_str())).collect();

        let response = self.run_query_synchronous(sql, &query_params).await?;

        if response.response.is_empty() {
            return Ok(None);
        }

        let batch_request =
            serde_json::from_str::<BatchRequestRow>(&response.response).map_err(|e| {
                Error::new(ErrorDetails::ClickHouseDeserialization {
                    message: e.to_string(),
                })
            })?;
        Ok(Some(batch_request))
    }

    async fn get_batch_model_inferences(
        &self,
        batch_id: Uuid,
        inference_ids: &[Uuid],
    ) -> Result<Vec<BatchModelInferenceRow<'static>>, Error> {
        if inference_ids.is_empty() {
            return Ok(vec![]);
        }

        // Build the inference_ids array for the parameterized query
        let inference_ids_str: String = inference_ids
            .iter()
            .map(|id| format!("'{id}'"))
            .collect::<Vec<_>>()
            .join(",");

        let sql = format!(
            r"SELECT * FROM BatchModelInference
            WHERE batch_id = {{batch_id:UUID}}
            AND inference_id IN ({inference_ids_str})
            FORMAT JSONEachRow"
        );

        let batch_id_str = batch_id.to_string();
        let mut params = HashMap::new();
        params.insert("batch_id", batch_id_str.as_str());

        let response = self.run_query_synchronous(sql, &params).await?;

        response
            .response
            .lines()
            .filter(|line| !line.is_empty())
            .map(|line| {
                serde_json::from_str::<BatchModelInferenceRow>(line).map_err(|e| {
                    Error::new(ErrorDetails::Serialization {
                        message: format!("Failed to deserialize batch model inference row: {e}"),
                    })
                })
            })
            .collect()
    }

    async fn get_completed_chat_batch_inferences(
        &self,
        batch_id: Uuid,
        function_name: &str,
        variant_name: &str,
        inference_id: Option<Uuid>,
    ) -> Result<Vec<CompletedBatchInferenceRow>, Error> {
        let (sql, params) = match inference_id {
            None => {
                // Get all inferences for the batch
                let sql = r"
                    WITH batch_inferences AS (
                        SELECT inference_id
                        FROM BatchModelInference
                        WHERE batch_id = {batch_id:UUID}
                    )
                    SELECT
                        ci.id as inference_id,
                        ci.episode_id as episode_id,
                        ci.variant_name as variant_name,
                        ci.output as output,
                        toUInt32(SUM(mi.input_tokens)) as input_tokens,
                        toUInt32(SUM(mi.output_tokens)) as output_tokens,
                        argMax(mi.finish_reason, toUInt128(mi.id)) as finish_reason
                    FROM ChatInference ci
                    LEFT JOIN ModelInference mi ON ci.id = mi.inference_id
                    WHERE ci.id IN (SELECT inference_id FROM batch_inferences)
                    AND ci.function_name = {function_name:String}
                    AND ci.variant_name = {variant_name:String}
                    GROUP BY ci.id, ci.episode_id, ci.variant_name, ci.output
                    FORMAT JSONEachRow
                "
                .to_string();

                let mut params = HashMap::new();
                params.insert("batch_id".to_string(), batch_id.to_string());
                params.insert("function_name".to_string(), function_name.to_string());
                params.insert("variant_name".to_string(), variant_name.to_string());
                (sql, params)
            }
            Some(inference_id) => {
                // Get a specific inference
                let sql = r"
                    WITH inf_lookup AS (
                        SELECT episode_id
                        FROM InferenceById
                        WHERE id_uint = toUInt128(toUUID({inference_id:String}))
                        LIMIT 1
                    )
                    SELECT
                        ci.id as inference_id,
                        ci.episode_id as episode_id,
                        ci.variant_name as variant_name,
                        ci.output as output,
                        toUInt32(SUM(mi.input_tokens)) as input_tokens,
                        toUInt32(SUM(mi.output_tokens)) as output_tokens,
                        argMax(mi.finish_reason, toUInt128(mi.id)) as finish_reason
                    FROM ChatInference ci
                    LEFT JOIN ModelInference mi ON ci.id = mi.inference_id
                    JOIN inf_lookup ON ci.episode_id = inf_lookup.episode_id
                    WHERE ci.id = {inference_id:String}
                    AND ci.function_name = {function_name:String}
                    AND ci.variant_name = {variant_name:String}
                    GROUP BY ci.id, ci.episode_id, ci.variant_name, ci.output
                    FORMAT JSONEachRow
                "
                .to_string();

                let mut params = HashMap::new();
                params.insert("batch_id".to_string(), batch_id.to_string());
                params.insert("inference_id".to_string(), inference_id.to_string());
                params.insert("function_name".to_string(), function_name.to_string());
                params.insert("variant_name".to_string(), variant_name.to_string());
                (sql, params)
            }
        };

        let query_params: HashMap<&str, &str> = params
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        let response = self.run_query_synchronous(sql, &query_params).await?;

        response
            .response
            .lines()
            .filter(|line| !line.is_empty())
            .map(|line| {
                serde_json::from_str::<CompletedBatchInferenceRow>(line).map_err(|e| {
                    Error::new(ErrorDetails::Serialization {
                        message: e.to_string(),
                    })
                })
            })
            .collect()
    }

    async fn get_completed_json_batch_inferences(
        &self,
        batch_id: Uuid,
        function_name: &str,
        variant_name: &str,
        inference_id: Option<Uuid>,
    ) -> Result<Vec<CompletedBatchInferenceRow>, Error> {
        let (sql, params) = match inference_id {
            None => {
                // Get all inferences for the batch
                let sql = r"
                    WITH batch_inferences AS (
                        SELECT inference_id
                        FROM BatchModelInference
                        WHERE batch_id = {batch_id:UUID}
                    )
                    SELECT
                        ji.id as inference_id,
                        ji.episode_id as episode_id,
                        ji.variant_name as variant_name,
                        ji.output as output,
                        toUInt32(SUM(mi.input_tokens)) as input_tokens,
                        toUInt32(SUM(mi.output_tokens)) as output_tokens,
                        argMax(mi.finish_reason, toUInt128(mi.id)) as finish_reason
                    FROM JsonInference ji
                    LEFT JOIN ModelInference mi ON ji.id = mi.inference_id
                    WHERE ji.id IN (SELECT inference_id FROM batch_inferences)
                    AND ji.function_name = {function_name:String}
                    AND ji.variant_name = {variant_name:String}
                    GROUP BY ji.id, ji.episode_id, ji.variant_name, ji.output
                    FORMAT JSONEachRow
                "
                .to_string();

                let mut params = HashMap::new();
                params.insert("batch_id".to_string(), batch_id.to_string());
                params.insert("function_name".to_string(), function_name.to_string());
                params.insert("variant_name".to_string(), variant_name.to_string());
                (sql, params)
            }
            Some(inference_id) => {
                // Get a specific inference
                let sql = r"
                    WITH inf_lookup AS (
                        SELECT episode_id
                        FROM InferenceById
                        WHERE id_uint = toUInt128(toUUID({inference_id:String}))
                        LIMIT 1
                    )
                    SELECT
                        ji.id as inference_id,
                        ji.episode_id as episode_id,
                        ji.variant_name as variant_name,
                        ji.output as output,
                        toUInt32(SUM(mi.input_tokens)) as input_tokens,
                        toUInt32(SUM(mi.output_tokens)) as output_tokens,
                        argMax(mi.finish_reason, toUInt128(mi.id)) as finish_reason
                    FROM JsonInference ji
                    LEFT JOIN ModelInference mi ON ji.id = mi.inference_id
                    JOIN inf_lookup ON ji.episode_id = inf_lookup.episode_id
                    WHERE ji.id = {inference_id:String}
                    AND ji.function_name = {function_name:String}
                    AND ji.variant_name = {variant_name:String}
                    GROUP BY ji.id, ji.episode_id, ji.variant_name, ji.output
                    FORMAT JSONEachRow
                "
                .to_string();

                let mut params = HashMap::new();
                params.insert("batch_id".to_string(), batch_id.to_string());
                params.insert("inference_id".to_string(), inference_id.to_string());
                params.insert("function_name".to_string(), function_name.to_string());
                params.insert("variant_name".to_string(), variant_name.to_string());
                (sql, params)
            }
        };

        let query_params: HashMap<&str, &str> = params
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        let response = self.run_query_synchronous(sql, &query_params).await?;

        response
            .response
            .lines()
            .filter(|line| !line.is_empty())
            .map(|line| {
                serde_json::from_str::<CompletedBatchInferenceRow>(line).map_err(|e| {
                    Error::new(ErrorDetails::Serialization {
                        message: e.to_string(),
                    })
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use crate::db::clickhouse::clickhouse_client::MockClickHouseClient;
    use crate::db::clickhouse::query_builder::test_util::{
        assert_query_contains, assert_query_does_not_contain,
    };
    use crate::db::clickhouse::{ClickHouseResponse, ClickHouseResponseMetadata};

    #[tokio::test]
    async fn test_get_batch_request_by_batch_id_only() {
        let batch_id = Uuid::now_v7();

        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(move |query, parameters| {
                assert_query_contains(
                    query,
                    "SELECT
                        batch_id,
                        id,
                        batch_params,
                        model_name,
                        model_provider_name,
                        status,
                        function_name,
                        variant_name,
                        raw_request,
                        raw_response,
                        errors
                    FROM BatchRequest
                    WHERE batch_id = {batch_id:UUID}
                    ORDER BY timestamp DESC
                    LIMIT 1",
                );
                assert_query_does_not_contain(query, "BatchIdByInferenceId");
                assert_query_does_not_contain(query, "inference_id");
                assert_eq!(
                    parameters.get("batch_id"),
                    Some(&batch_id.to_string().as_str())
                );
                assert_eq!(parameters.len(), 1, "should only have batch_id parameter");
                true
            })
            .returning(move |_, _| {
                let id = Uuid::now_v7();
                Ok(ClickHouseResponse {
                    response: format!(
                        r#"{{"batch_id":"{batch_id}","id":"{id}","batch_params":"{{}}","model_name":"gpt-4","model_provider_name":"openai","status":"pending","function_name":"test_fn","variant_name":"test_var","raw_request":"{{}}","raw_response":"{{}}","errors":[]}}"#
                    ),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let result = conn.get_batch_request(batch_id, None).await.unwrap();

        assert!(result.is_some(), "should return a batch request");
        let row = result.unwrap();
        assert_eq!(row.batch_id, batch_id);
        assert_eq!(row.function_name.as_ref(), "test_fn");
    }

    #[tokio::test]
    async fn test_get_batch_request_by_batch_id_and_inference_id() {
        let batch_id = Uuid::now_v7();
        let inference_id = Uuid::now_v7();

        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(move |query, parameters| {
                assert_query_contains(query, "FROM BatchIdByInferenceId bi");
                assert_query_contains(query, "JOIN BatchRequest br ON bi.batch_id = br.batch_id");
                assert_query_contains(
                    query,
                    "WHERE bi.inference_id = {inference_id:UUID} AND bi.batch_id = {batch_id:UUID}",
                );
                assert_eq!(
                    parameters.get("batch_id"),
                    Some(&batch_id.to_string().as_str())
                );
                assert_eq!(
                    parameters.get("inference_id"),
                    Some(&inference_id.to_string().as_str())
                );
                assert_eq!(parameters.len(), 2, "should have batch_id and inference_id");
                true
            })
            .returning(move |_, _| {
                let id = Uuid::now_v7();
                Ok(ClickHouseResponse {
                    response: format!(
                        r#"{{"batch_id":"{batch_id}","id":"{id}","batch_params":"{{}}","model_name":"gpt-4","model_provider_name":"openai","status":"completed","function_name":"my_function","variant_name":"my_variant","raw_request":"{{}}","raw_response":"{{}}","errors":[]}}"#
                    ),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let result = conn
            .get_batch_request(batch_id, Some(inference_id))
            .await
            .unwrap();

        assert!(result.is_some(), "should return a batch request");
        let row = result.unwrap();
        assert_eq!(row.batch_id, batch_id);
        assert_eq!(row.function_name.as_ref(), "my_function");
    }

    #[tokio::test]
    async fn test_get_batch_request_not_found() {
        let batch_id = Uuid::now_v7();

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
        let result = conn.get_batch_request(batch_id, None).await.unwrap();

        assert!(result.is_none(), "should return None when not found");
    }

    #[tokio::test]
    async fn test_get_batch_model_inferences_query_construction() {
        let batch_id = Uuid::now_v7();
        let inference_id_1 = Uuid::now_v7();
        let inference_id_2 = Uuid::now_v7();

        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(move |query, parameters| {
                assert_query_contains(query, "SELECT * FROM BatchModelInference");
                assert_query_contains(query, "WHERE batch_id = {batch_id:UUID}");
                assert_query_contains(query, "AND inference_id IN (");
                assert_query_contains(query, &format!("'{inference_id_1}'"));
                assert_query_contains(query, &format!("'{inference_id_2}'"));
                assert_eq!(
                    parameters.get("batch_id"),
                    Some(&batch_id.to_string().as_str())
                );
                true
            })
            .returning(|_, _| {
                // Return empty response since we're testing query construction
                // BatchModelInferenceRow has complex nested types that are difficult to mock
                Ok(ClickHouseResponse {
                    response: String::new(),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 0,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let result = conn
            .get_batch_model_inferences(batch_id, &[inference_id_1, inference_id_2])
            .await
            .unwrap();

        assert!(
            result.is_empty(),
            "should return empty vec for empty response"
        );
    }

    #[tokio::test]
    async fn test_get_batch_model_inferences_empty_ids() {
        let batch_id = Uuid::now_v7();

        let mock_clickhouse_client = MockClickHouseClient::new();
        // No expectations set - should not be called

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));
        let result = conn
            .get_batch_model_inferences(batch_id, &[])
            .await
            .unwrap();

        assert!(
            result.is_empty(),
            "should return empty vec for empty inference_ids"
        );
    }

    #[tokio::test]
    async fn test_get_completed_chat_batch_inferences_all() {
        let batch_id = Uuid::now_v7();
        let inference_id = Uuid::now_v7();
        let episode_id = Uuid::now_v7();

        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(move |query, parameters| {
                assert_query_contains(
                    query,
                    "WITH batch_inferences AS (
                        SELECT inference_id
                        FROM BatchModelInference
                        WHERE batch_id = {batch_id:UUID}
                    )",
                );
                assert_query_contains(query, "FROM ChatInference ci");
                assert_query_contains(query, "LEFT JOIN ModelInference mi ON ci.id = mi.inference_id");
                assert_query_contains(
                    query,
                    "WHERE ci.id IN (SELECT inference_id FROM batch_inferences)",
                );
                assert_query_contains(query, "AND ci.function_name = {function_name:String}");
                assert_query_contains(query, "AND ci.variant_name = {variant_name:String}");
                assert_query_does_not_contain(query, "inf_lookup");
                assert_eq!(
                    parameters.get("batch_id"),
                    Some(&batch_id.to_string().as_str())
                );
                assert_eq!(parameters.get("function_name"), Some(&"test_function"));
                assert_eq!(parameters.get("variant_name"), Some(&"test_variant"));
                assert!(!parameters.contains_key("inference_id"));
                true
            })
            .returning(move |_, _| {
                Ok(ClickHouseResponse {
                    response: format!(
                        r#"{{"inference_id":"{inference_id}","episode_id":"{episode_id}","variant_name":"test_variant","output":"[]","input_tokens":100,"output_tokens":50,"finish_reason":"stop"}}"#
                    ),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let result = conn
            .get_completed_chat_batch_inferences(batch_id, "test_function", "test_variant", None)
            .await
            .unwrap();

        assert_eq!(result.len(), 1, "should return one inference");
        assert_eq!(result[0].inference_id, inference_id);
        assert_eq!(result[0].episode_id, episode_id);
        assert_eq!(result[0].input_tokens, Some(100));
        assert_eq!(result[0].output_tokens, Some(50));
    }

    #[tokio::test]
    async fn test_get_completed_chat_batch_inferences_single() {
        let batch_id = Uuid::now_v7();
        let inference_id = Uuid::now_v7();
        let episode_id = Uuid::now_v7();

        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(move |query, parameters| {
                assert_query_contains(
                    query,
                    "WITH inf_lookup AS (
                        SELECT episode_id
                        FROM InferenceById
                        WHERE id_uint = toUInt128(toUUID({inference_id:String}))
                        LIMIT 1
                    )",
                );
                assert_query_contains(query, "FROM ChatInference ci");
                assert_query_contains(query, "JOIN inf_lookup ON ci.episode_id = inf_lookup.episode_id");
                assert_query_contains(query, "WHERE ci.id = {inference_id:String}");
                assert_eq!(
                    parameters.get("inference_id"),
                    Some(&inference_id.to_string().as_str())
                );
                assert_eq!(parameters.get("function_name"), Some(&"chat_fn"));
                assert_eq!(parameters.get("variant_name"), Some(&"chat_var"));
                true
            })
            .returning(move |_, _| {
                Ok(ClickHouseResponse {
                    response: format!(
                        r#"{{"inference_id":"{inference_id}","episode_id":"{episode_id}","variant_name":"chat_var","output":"[]","input_tokens":200,"output_tokens":100,"finish_reason":null}}"#
                    ),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let result = conn
            .get_completed_chat_batch_inferences(
                batch_id,
                "chat_fn",
                "chat_var",
                Some(inference_id),
            )
            .await
            .unwrap();

        assert_eq!(result.len(), 1, "should return one inference");
        assert_eq!(result[0].inference_id, inference_id);
        assert_eq!(result[0].input_tokens, Some(200));
        assert!(result[0].finish_reason.is_none());
    }

    #[tokio::test]
    async fn test_get_completed_json_batch_inferences_all() {
        let batch_id = Uuid::now_v7();
        let inference_id = Uuid::now_v7();
        let episode_id = Uuid::now_v7();

        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(move |query, parameters| {
                assert_query_contains(
                    query,
                    "WITH batch_inferences AS (
                        SELECT inference_id
                        FROM BatchModelInference
                        WHERE batch_id = {batch_id:UUID}
                    )",
                );
                assert_query_contains(query, "FROM JsonInference ji");
                assert_query_contains(query, "LEFT JOIN ModelInference mi ON ji.id = mi.inference_id");
                assert_query_contains(
                    query,
                    "WHERE ji.id IN (SELECT inference_id FROM batch_inferences)",
                );
                assert_query_contains(query, "AND ji.function_name = {function_name:String}");
                assert_query_contains(query, "AND ji.variant_name = {variant_name:String}");
                assert_eq!(parameters.get("function_name"), Some(&"json_function"));
                assert_eq!(parameters.get("variant_name"), Some(&"json_variant"));
                true
            })
            .returning(move |_, _| {
                Ok(ClickHouseResponse {
                    response: format!(
                        r#"{{"inference_id":"{inference_id}","episode_id":"{episode_id}","variant_name":"json_variant","output":"{{}}","input_tokens":150,"output_tokens":75,"finish_reason":"stop"}}"#
                    ),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let result = conn
            .get_completed_json_batch_inferences(batch_id, "json_function", "json_variant", None)
            .await
            .unwrap();

        assert_eq!(result.len(), 1, "should return one inference");
        assert_eq!(result[0].inference_id, inference_id);
        assert_eq!(result[0].input_tokens, Some(150));
    }

    #[tokio::test]
    async fn test_get_completed_json_batch_inferences_single() {
        let batch_id = Uuid::now_v7();
        let inference_id = Uuid::now_v7();
        let episode_id = Uuid::now_v7();

        let mut mock_clickhouse_client = MockClickHouseClient::new();
        mock_clickhouse_client
            .expect_run_query_synchronous()
            .withf(move |query, parameters| {
                assert_query_contains(query, "WITH inf_lookup AS (");
                assert_query_contains(query, "FROM JsonInference ji");
                assert_query_contains(query, "JOIN inf_lookup ON ji.episode_id = inf_lookup.episode_id");
                assert_query_contains(query, "WHERE ji.id = {inference_id:String}");
                assert_eq!(
                    parameters.get("inference_id"),
                    Some(&inference_id.to_string().as_str())
                );
                true
            })
            .returning(move |_, _| {
                Ok(ClickHouseResponse {
                    response: format!(
                        r#"{{"inference_id":"{inference_id}","episode_id":"{episode_id}","variant_name":"v1","output":"{{}}","input_tokens":null,"output_tokens":null,"finish_reason":null}}"#
                    ),
                    metadata: ClickHouseResponseMetadata {
                        read_rows: 1,
                        written_rows: 0,
                    },
                })
            });

        let conn = ClickHouseConnectionInfo::new_mock(Arc::new(mock_clickhouse_client));

        let result = conn
            .get_completed_json_batch_inferences(batch_id, "fn", "v1", Some(inference_id))
            .await
            .unwrap();

        assert_eq!(result.len(), 1, "should return one inference");
        assert_eq!(result[0].inference_id, inference_id);
        assert!(
            result[0].input_tokens.is_none(),
            "should handle null tokens"
        );
    }

    #[tokio::test]
    async fn test_get_completed_batch_inferences_empty_response() {
        let batch_id = Uuid::now_v7();

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

        let result = conn
            .get_completed_chat_batch_inferences(batch_id, "fn", "var", None)
            .await
            .unwrap();

        assert!(result.is_empty(), "should return empty vec for no results");
    }
}
