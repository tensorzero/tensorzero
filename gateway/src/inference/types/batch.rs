use crate::{endpoints::inference::InferenceParams, tool::ToolCallConfig};

use super::{ContentBlock, ModelInferenceRequest, RequestMessage, Usage};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::borrow::Cow;
use std::collections::HashMap;
use uuid::Uuid;

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BatchStatus {
    Pending,
    Completed,
    Failed,
}

pub struct StartBatchProviderInferenceResponse {
    pub batch_id: Uuid,
    pub inference_ids: Vec<Uuid>,
    pub batch_params: Value,
    pub status: BatchStatus,
}

pub struct StartBatchModelInferenceResponse<'a> {
    pub batch_id: Uuid,
    pub inference_ids: Vec<Uuid>,
    pub batch_params: Value,
    pub model_provider_name: &'a str,
    pub status: BatchStatus,
}

impl<'a> StartBatchModelInferenceResponse<'a> {
    pub fn new(
        provider_batch_response: StartBatchProviderInferenceResponse,
        model_provider_name: &'a str,
    ) -> Self {
        Self {
            batch_id: provider_batch_response.batch_id,
            inference_ids: provider_batch_response.inference_ids,
            batch_params: provider_batch_response.batch_params,
            model_provider_name,
            status: provider_batch_response.status,
        }
    }
}

// TODO: add errors and stuff
#[derive(Debug)]
pub enum PollBatchInferenceResponse {
    Pending,
    Completed(ProviderBatchInferenceResponse),
    Failed,
}

#[derive(Debug)]
pub struct BatchProviderInferenceResponse {
    pub batch_id: Uuid,
}

// Returned from poll_batch_inference from a variant.
// Includes all the metadata that we need to write to ClickHouse.
pub struct BatchModelInferenceWithMetadata<'a> {
    pub batch_id: Uuid,
    pub inference_ids: Vec<Uuid>,
    pub input_messages: Vec<Vec<RequestMessage>>,
    pub systems: Vec<Option<String>>,
    pub tool_configs: Vec<Option<Cow<'a, ToolCallConfig>>>,
    pub inference_params: Vec<InferenceParams>,
    pub output_schemas: Vec<Option<&'a Value>>,
    pub batch_params: Value,
    pub model_provider_name: &'a str,
    pub model_name: &'a str,
    pub status: BatchStatus,
}

impl<'a> BatchModelInferenceWithMetadata<'a> {
    pub fn new(
        model_batch_response: StartBatchModelInferenceResponse<'a>,
        model_inference_requests: Vec<ModelInferenceRequest<'a>>,
        model_name: &'a str,
        inference_params: Vec<InferenceParams>,
    ) -> Self {
        let mut input_messages: Vec<Vec<RequestMessage>> = vec![];
        let mut systems: Vec<Option<String>> = vec![];
        let mut tool_configs: Vec<Option<Cow<'a, ToolCallConfig>>> = vec![];
        let mut output_schemas: Vec<Option<&'a Value>> = vec![];
        for request in model_inference_requests {
            input_messages.push(request.messages);
            systems.push(request.system);
            tool_configs.push(request.tool_config);
            output_schemas.push(request.output_schema);
        }
        Self {
            batch_id: model_batch_response.batch_id,
            inference_ids: model_batch_response.inference_ids,
            input_messages,
            systems,
            tool_configs,
            inference_params,
            output_schemas,
            batch_params: model_batch_response.batch_params,
            model_provider_name: model_batch_response.model_provider_name,
            model_name,
            status: model_batch_response.status,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct BatchRequest {
    pub batch_id: Uuid,
    #[serde(deserialize_with = "deserialize_json_string")]
    pub batch_params: Value,
    pub model_name: String,
    pub model_provider_name: String,
    pub status: BatchStatus,
    pub errors: HashMap<String, String>,
}

fn deserialize_json_string<'de, D>(deserializer: D) -> Result<Value, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let json_str = String::deserialize(deserializer)?;
    serde_json::from_str(&json_str).map_err(serde::de::Error::custom)
}

#[derive(Debug)]
pub struct ProviderBatchInferenceOutput {
    pub id: Uuid,
    pub created: u64,
    pub output: Vec<ContentBlock>,
    pub system: Option<String>,
    pub input_messages: Vec<RequestMessage>,
    pub raw_response: String,
    pub usage: Usage,
}

#[derive(Debug)]
pub struct ProviderBatchInferenceResponse {
    // Inference ID -> Output
    pub elements: HashMap<Uuid, ProviderBatchInferenceOutput>,
}
