use std::collections::HashMap;

use async_trait::async_trait;
use serde::Deserialize;
use uuid::Uuid;

use crate::db::resolve_uuid::{ResolveUuidQueries, ResolvedObject};
use crate::error::{Error, ErrorDetails};
use crate::inference::types::FunctionType;

use super::ClickHouseConnectionInfo;

#[derive(Debug, Deserialize)]
struct InferenceRow {
    function_name: String,
    function_type: FunctionType,
    variant_name: String,
    episode_id: Uuid,
}

#[derive(Debug, Deserialize)]
struct ModelInferenceRow {
    inference_id: Uuid,
    model_name: String,
    model_provider_name: String,
}

#[derive(Debug, Deserialize)]
struct DatapointRow {
    dataset_name: String,
    function_name: String,
}

#[async_trait]
impl ResolveUuidQueries for ClickHouseConnectionInfo {
    async fn resolve_uuid(&self, id: &Uuid) -> Result<Vec<ResolvedObject>, Error> {
        let id_str = id.to_string();
        let params = HashMap::from([("id", id_str.as_str())]);

        // Query all tables concurrently
        let (
            inference_result,
            model_inference_result,
            episode_result,
            boolean_result,
            float_result,
            comment_result,
            demonstration_result,
            chat_datapoint_result,
            json_datapoint_result,
        ) = tokio::try_join!(
            self.query_inference(&params),
            self.query_model_inference(&params),
            self.query_episode(&params),
            self.query_exists("BooleanMetricFeedback", &params),
            self.query_exists("FloatMetricFeedback", &params),
            self.query_exists("CommentFeedback", &params),
            self.query_exists("DemonstrationFeedback", &params),
            self.query_datapoint("ChatInferenceDatapoint", &params),
            self.query_datapoint("JsonInferenceDatapoint", &params),
        )?;

        let mut results = Vec::new();

        if let Some(row) = inference_result {
            results.push(ResolvedObject::Inference {
                function_name: row.function_name,
                function_type: row.function_type,
                variant_name: row.variant_name,
                episode_id: row.episode_id,
            });
        }

        if let Some(row) = model_inference_result {
            results.push(ResolvedObject::ModelInference {
                inference_id: row.inference_id,
                model_name: row.model_name,
                model_provider_name: row.model_provider_name,
            });
        }

        if episode_result {
            results.push(ResolvedObject::Episode);
        }

        if boolean_result {
            results.push(ResolvedObject::BooleanFeedback);
        }

        if float_result {
            results.push(ResolvedObject::FloatFeedback);
        }

        if comment_result {
            results.push(ResolvedObject::CommentFeedback);
        }

        if demonstration_result {
            results.push(ResolvedObject::DemonstrationFeedback);
        }

        if let Some(row) = chat_datapoint_result {
            results.push(ResolvedObject::ChatDatapoint {
                dataset_name: row.dataset_name,
                function_name: row.function_name,
            });
        }

        if let Some(row) = json_datapoint_result {
            results.push(ResolvedObject::JsonDatapoint {
                dataset_name: row.dataset_name,
                function_name: row.function_name,
            });
        }

        Ok(results)
    }
}

impl ClickHouseConnectionInfo {
    async fn query_inference(
        &self,
        params: &HashMap<&str, &str>,
    ) -> Result<Option<InferenceRow>, Error> {
        let query = r"
            SELECT function_name, function_type, variant_name, episode_id
            FROM InferenceById
            WHERE id_uint = toUInt128({id:UUID})
            LIMIT 1
            FORMAT JSONEachRow
            SETTINGS max_threads=1
        ";

        let response = self
            .run_query_synchronous(query.to_string(), params)
            .await?;

        if response.response.is_empty() {
            return Ok(None);
        }

        let row: InferenceRow = serde_json::from_str(&response.response).map_err(|e| {
            Error::new(ErrorDetails::ClickHouseDeserialization {
                message: e.to_string(),
            })
        })?;

        Ok(Some(row))
    }

    async fn query_model_inference(
        &self,
        params: &HashMap<&str, &str>,
    ) -> Result<Option<ModelInferenceRow>, Error> {
        let query = r"
            SELECT inference_id, model_name, model_provider_name
            FROM ModelInference
            WHERE id = {id:UUID}
            LIMIT 1
            FORMAT JSONEachRow
            SETTINGS max_threads=1
        ";

        let response = self
            .run_query_synchronous(query.to_string(), params)
            .await?;

        if response.response.is_empty() {
            return Ok(None);
        }

        let row: ModelInferenceRow = serde_json::from_str(&response.response).map_err(|e| {
            Error::new(ErrorDetails::ClickHouseDeserialization {
                message: e.to_string(),
            })
        })?;

        Ok(Some(row))
    }

    async fn query_episode(&self, params: &HashMap<&str, &str>) -> Result<bool, Error> {
        let query = r"
            SELECT 1
            FROM EpisodeById
            WHERE episode_id_uint = toUInt128({id:UUID})
            LIMIT 1
            FORMAT JSONEachRow
            SETTINGS max_threads=1
        ";

        let response = self
            .run_query_synchronous(query.to_string(), params)
            .await?;

        Ok(!response.response.is_empty())
    }

    async fn query_exists(&self, table: &str, params: &HashMap<&str, &str>) -> Result<bool, Error> {
        let query = format!(
            r"
            SELECT 1
            FROM {table}
            WHERE id = {{id:UUID}}
            LIMIT 1
            FORMAT JSONEachRow
            SETTINGS max_threads=1
        "
        );

        let response = self.run_query_synchronous(query, params).await?;

        Ok(!response.response.is_empty())
    }

    async fn query_datapoint(
        &self,
        table: &str,
        params: &HashMap<&str, &str>,
    ) -> Result<Option<DatapointRow>, Error> {
        let query = format!(
            r"
            SELECT dataset_name, function_name
            FROM {table}
            WHERE id = {{id:UUID}}
            LIMIT 1
            FORMAT JSONEachRow
            SETTINGS max_threads=1
        "
        );

        let response = self.run_query_synchronous(query, params).await?;

        if response.response.is_empty() {
            return Ok(None);
        }

        let row: DatapointRow = serde_json::from_str(&response.response).map_err(|e| {
            Error::new(ErrorDetails::ClickHouseDeserialization {
                message: e.to_string(),
            })
        })?;

        Ok(Some(row))
    }
}
