use futures::StreamExt;
use serde::Deserialize;
use std::borrow::Cow;
use std::collections::HashMap;
use std::path::PathBuf;
use uuid::Uuid;

use crate::embeddings::EmbeddingModelConfig;
use crate::endpoints::inference::{InferenceClients, InferenceModels, InferenceParams};
use crate::error::Error;
use crate::function::FunctionConfig;
use crate::inference::types::{
    FunctionType, InferenceResultChunk, InferenceResultStream, Input, ModelInferenceRequest,
    ModelInferenceRequestJsonMode, ModelInferenceResponseWithMetadata, RequestMessage,
};
use crate::jsonschema_util::DynamicJSONSchema;
use crate::minijinja_util::TemplateConfig;
use crate::tool::{create_dynamic_implicit_tool_config, ToolCallConfig};
use crate::{inference::types::InferenceResult, model::ModelConfig};
pub mod best_of_n;
pub mod chat_completion;
pub mod dicl;

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
#[serde(deny_unknown_fields)]
pub enum VariantConfig {
    ChatCompletion(chat_completion::ChatCompletionConfig),
    #[serde(rename = "experimental_best_of_n")]
    BestOfN(best_of_n::BestOfNConfig),
    #[serde(rename = "experimental_dynamic_in_context_learning")]
    Dicl(dicl::DiclConfig),
}

/// This type is used to determine how to enforce JSON mode for a given variant.
/// Variants represent JSON mode in a slightly more abstract sense than ModelInferenceRequests, as
/// we support coercing tool calls into JSON mode.
/// This is represented as a tool config in the
#[derive(Debug, Default, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum JsonMode {
    Off,
    #[default]
    On,
    Strict,
    ImplicitTool,
}

/// Maps to the subset of Config that applies to the current inference request.
/// It doesn't take into account inference-time overrides (e.g. dynamic tools).
pub struct InferenceConfig<'a> {
    pub tool_config: Option<ToolCallConfig>,
    pub templates: &'a TemplateConfig<'a>,
    pub dynamic_output_schema: Option<DynamicJSONSchema>,
    pub function_name: String,
    pub variant_name: String,
}

pub struct ModelUsedInfo<'a> {
    pub model_name: &'a str,
    pub model_provider_name: &'a str,
    pub raw_request: String,
    pub inference_params: InferenceParams,
    pub previous_model_inference_results: Vec<ModelInferenceResponseWithMetadata<'a>>,
}

pub trait Variant {
    async fn infer<'a, 'request>(
        &'a self,
        input: &Input,
        models: &'request InferenceModels<'a>,
        function: &'a FunctionConfig,
        inference_config: &'request InferenceConfig<'request>,
        clients: &'request InferenceClients,
        inference_params: InferenceParams,
    ) -> Result<InferenceResult<'a>, Error>;

    async fn infer_stream<'request>(
        &'static self,
        input: &Input,
        models: &'request InferenceModels<'static>,
        function: &'static FunctionConfig,
        inference_config: &'request InferenceConfig<'request>,
        clients: &'request InferenceClients<'request>,
        inference_params: InferenceParams,
    ) -> Result<
        (
            InferenceResultChunk,
            InferenceResultStream,
            ModelUsedInfo<'static>,
        ),
        Error,
    >;

    fn validate(
        &self,
        function: &FunctionConfig,
        models: &HashMap<String, ModelConfig>,
        embedding_models: &HashMap<String, EmbeddingModelConfig>,
        templates: &TemplateConfig,
        function_name: &str,
        variant_name: &str,
    ) -> Result<(), Error>;

    fn get_all_template_paths(&self) -> Vec<&PathBuf>;
}

impl VariantConfig {
    pub fn weight(&self) -> f64 {
        match self {
            VariantConfig::ChatCompletion(params) => params.weight,
            VariantConfig::BestOfN(params) => params.weight,
            VariantConfig::Dicl(params) => params.weight,
        }
    }
}

impl Variant for VariantConfig {
    async fn infer<'a, 'request>(
        &'a self,
        input: &Input,
        models: &'request InferenceModels<'a>,
        function: &'a FunctionConfig,
        inference_config: &'request InferenceConfig<'request>,
        clients: &'request InferenceClients<'request>,
        inference_params: InferenceParams,
    ) -> Result<InferenceResult<'a>, Error> {
        match self {
            VariantConfig::ChatCompletion(params) => {
                params
                    .infer(
                        input,
                        models,
                        function,
                        inference_config,
                        clients,
                        inference_params,
                    )
                    .await
            }
            VariantConfig::BestOfN(params) => {
                params
                    .infer(
                        input,
                        models,
                        function,
                        inference_config,
                        clients,
                        inference_params,
                    )
                    .await
            }
            VariantConfig::Dicl(params) => {
                params
                    .infer(
                        input,
                        models,
                        function,
                        inference_config,
                        clients,
                        inference_params,
                    )
                    .await
            }
        }
    }

    async fn infer_stream<'request>(
        &'static self,
        input: &Input,
        models: &'request InferenceModels<'static>,
        function: &'static FunctionConfig,
        inference_config: &'request InferenceConfig<'request>,
        clients: &'request InferenceClients<'request>,
        inference_params: InferenceParams,
    ) -> Result<
        (
            InferenceResultChunk,
            InferenceResultStream,
            ModelUsedInfo<'static>,
        ),
        Error,
    > {
        match self {
            VariantConfig::ChatCompletion(params) => {
                params
                    .infer_stream(
                        input,
                        models,
                        function,
                        inference_config,
                        clients,
                        inference_params,
                    )
                    .await
            }
            VariantConfig::BestOfN(params) => {
                params
                    .infer_stream(
                        input,
                        models,
                        function,
                        inference_config,
                        clients,
                        inference_params,
                    )
                    .await
            }
            VariantConfig::Dicl(params) => {
                params
                    .infer_stream(
                        input,
                        models,
                        function,
                        inference_config,
                        clients,
                        inference_params,
                    )
                    .await
            }
        }
    }

    fn validate(
        &self,
        function: &FunctionConfig,
        models: &HashMap<String, ModelConfig>,
        embedding_models: &HashMap<String, EmbeddingModelConfig>,
        templates: &TemplateConfig,
        function_name: &str,
        variant_name: &str,
    ) -> Result<(), Error> {
        match self {
            VariantConfig::ChatCompletion(params) => params.validate(
                function,
                models,
                embedding_models,
                templates,
                function_name,
                variant_name,
            ),
            VariantConfig::BestOfN(params) => params.validate(
                function,
                models,
                embedding_models,
                templates,
                function_name,
                variant_name,
            ),
            VariantConfig::Dicl(params) => params.validate(
                function,
                models,
                embedding_models,
                templates,
                function_name,
                variant_name,
            ),
        }
    }

    fn get_all_template_paths(&self) -> Vec<&PathBuf> {
        match self {
            VariantConfig::ChatCompletion(params) => params.get_all_template_paths(),
            VariantConfig::BestOfN(params) => params.get_all_template_paths(),
            VariantConfig::Dicl(params) => params.get_all_template_paths(),
        }
    }
}

fn prepare_model_inference_request<'a, 'request>(
    messages: Vec<RequestMessage>,
    system: Option<String>,
    function: &'a FunctionConfig,
    inference_config: &'request InferenceConfig<'request>,
    stream: bool,
    inference_params: &InferenceParams,
    json_mode: &'a JsonMode,
) -> Result<ModelInferenceRequest<'request>, Error>
where
    'a: 'request,
{
    Ok(match function {
        FunctionConfig::Chat(_) => ModelInferenceRequest {
            messages,
            system,
            tool_config: inference_config.tool_config.as_ref().map(Cow::Borrowed),
            temperature: inference_params.chat_completion.temperature,
            max_tokens: inference_params.chat_completion.max_tokens,
            seed: inference_params.chat_completion.seed,
            stream,
            json_mode: ModelInferenceRequestJsonMode::Off,
            function_type: FunctionType::Chat,
            output_schema: None,
        },
        FunctionConfig::Json(json_config) => {
            let tool_config = match json_mode {
                JsonMode::ImplicitTool => match &inference_config.dynamic_output_schema {
                    Some(schema) => Some(Cow::Owned(create_dynamic_implicit_tool_config(
                        schema.value.clone(),
                    ))),
                    None => Some(Cow::Borrowed(&json_config.implicit_tool_call_config)),
                },
                _ => None,
            };
            let output_schema = match &inference_config.dynamic_output_schema {
                Some(schema) => Some(&schema.value),
                None => Some(json_config.output_schema.value),
            };
            ModelInferenceRequest {
                messages,
                system,
                tool_config,
                temperature: inference_params.chat_completion.temperature,
                max_tokens: inference_params.chat_completion.max_tokens,
                seed: inference_params.chat_completion.seed,
                stream,
                json_mode: json_mode.into(),
                function_type: FunctionType::Json,
                output_schema,
            }
        }
    })
}

async fn infer_model_request<'a, 'request>(
    request: ModelInferenceRequest<'request>,
    model_name: &'a str,
    model_config: &'a ModelConfig,
    function: &'a FunctionConfig,
    inference_config: &'request InferenceConfig<'request>,
    clients: &'request InferenceClients<'request>,
    inference_params: InferenceParams,
) -> Result<InferenceResult<'a>, Error> {
    let model_inference_response = model_config.infer(&request, clients.http_client).await?;
    let model_inference_result =
        ModelInferenceResponseWithMetadata::new(model_inference_response, model_name);
    let inference_id = Uuid::now_v7();
    let raw_content = model_inference_result.content.clone();
    let usage = model_inference_result.usage.clone();
    let model_inference_results = vec![model_inference_result];
    function
        .prepare_response(
            inference_id,
            raw_content,
            usage,
            model_inference_results,
            inference_config,
            inference_params,
        )
        .await
}

async fn infer_model_request_stream<'request>(
    request: ModelInferenceRequest<'request>,
    model_name: &'static str,
    model_config: &'static ModelConfig,
    function: &'static FunctionConfig,
    clients: &'request InferenceClients<'request>,
    inference_params: InferenceParams,
) -> Result<
    (
        InferenceResultChunk,
        InferenceResultStream,
        ModelUsedInfo<'static>,
    ),
    Error,
> {
    let (first_chunk, stream, raw_request, model_provider_name) = model_config
        .infer_stream(&request, clients.http_client)
        .await?;
    let model_used_info = ModelUsedInfo {
        model_name,
        model_provider_name,
        raw_request,
        inference_params,
        previous_model_inference_results: vec![],
    };
    let first_chunk = InferenceResultChunk::new(first_chunk, function);
    let stream =
        stream.map(move |chunk| chunk.map(|chunk| InferenceResultChunk::new(chunk, function)));
    Ok((first_chunk, Box::pin(stream), model_used_info))
}
