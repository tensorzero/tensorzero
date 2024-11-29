use crate::{endpoints::inference::InferenceParams, tool::ToolCallConfig};

use super::{ModelInferenceRequest, RequestMessage};
use serde_json::Value;
use std::borrow::Cow;
use uuid::Uuid;

pub enum BatchStatus {
    Pending,
    Completed,
    Failed,
}

pub struct BatchProviderInferenceResponse {
    pub batch_id: Uuid,
    pub inference_ids: Vec<Uuid>,
    pub batch_params: Value,
    pub status: BatchStatus,
}

pub struct BatchModelInferenceResponse<'a> {
    pub batch_id: Uuid,
    pub inference_ids: Vec<Uuid>,
    pub batch_params: Value,
    pub model_provider_name: &'a str,
    pub status: BatchStatus,
}

impl<'a> BatchModelInferenceResponse<'a> {
    pub fn new(
        provider_batch_response: BatchProviderInferenceResponse,
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

// TODO(Viraj): move to types::batch
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
        model_batch_response: BatchModelInferenceResponse<'a>,
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
