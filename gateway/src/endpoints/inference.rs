use axum::body::Body;
use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::{debug_handler, Json};
use futures::stream::Stream;
use metrics::counter;
use secrecy::SecretString;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::pin::Pin;
use std::time::Duration;
use tokio::time::Instant;
use tokio_stream::StreamExt;
use tracing::instrument;
use uuid::Uuid;

use crate::clickhouse::ClickHouseConnectionInfo;
use crate::config_parser::Config;
use crate::embeddings::EmbeddingModelConfig;
use crate::error::{Error, ErrorDetails};
use crate::function::sample_variant;
use crate::function::FunctionConfig;
use crate::gateway_util::{AppState, AppStateData, StructuredJson};
use crate::inference::types::{
    collect_chunks, ChatInferenceDatabaseInsert, CollectChunksArgs, ContentBlockChunk,
    ContentBlockOutput, InferenceResult, InferenceResultChunk, InferenceResultStream, Input,
    JsonInferenceDatabaseInsert, JsonInferenceOutput, ModelInferenceResponseWithMetadata,
    RequestMessage, Usage,
};
use crate::jsonschema_util::DynamicJSONSchema;
use crate::minijinja_util::TemplateConfig;
use crate::model::ModelTable;
use crate::tool::{DynamicToolParams, ToolCallConfig};
use crate::uuid_util::validate_episode_id;
use crate::variant::{InferenceConfig, Variant};

/// The expected payload is a JSON object with the following fields:
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Params {
    // the function name
    pub function_name: String,
    // the episode ID (if not provided, it'll be set to inference_id)
    // NOTE: DO NOT GENERATE EPISODE IDS MANUALLY. THE API WILL DO THAT FOR YOU.
    pub episode_id: Option<Uuid>,
    // the input for the inference
    pub input: Input,
    // default False
    pub stream: Option<bool>,
    // Inference-time overrides for variant types (use with caution)
    #[serde(default)]
    pub params: InferenceParams,
    // if the client would like to pin a specific variant to be used
    // NOTE: YOU SHOULD TYPICALLY LET THE API SELECT A VARIANT FOR YOU (I.E. IGNORE THIS FIELD).
    //       ONLY PIN A VARIANT FOR SPECIAL USE CASES (E.G. TESTING / DEBUGGING VARIANTS).
    pub variant_name: Option<String>,
    // if true, the inference will not be stored
    pub dryrun: Option<bool>,
    // the tags to add to the inference
    #[serde(default)]
    pub tags: HashMap<String, String>,
    // dynamic information about tool calling. Don't directly include `dynamic_tool_params` in `Params`.
    #[serde(flatten)]
    pub dynamic_tool_params: DynamicToolParams,
    // `dynamic_tool_params` includes the following fields, passed at the top level of `Params`:
    // If provided, the inference will only use the specified tools (a subset of the function's tools)
    // allowed_tools: Option<Vec<String>>,
    // If provided, the inference will use the specified tools in addition to the function's tools
    // additional_tools: Option<Vec<Tool>>,
    // If provided, the inference will use the specified tool choice
    // tool_choice: Option<ToolChoice>,
    // If true, the inference will use parallel tool calls
    // parallel_tool_calls: Option<bool>,
    // If provided for a JSON inference, the inference will use the specified output schema instead of the
    // configured one. We only lazily validate this schema.
    pub output_schema: Option<Value>,
    #[serde(default)]
    pub credentials: InferenceCredentials,
}

#[derive(Clone, Debug)]
struct InferenceMetadata<'a> {
    pub function_name: String,
    pub variant_name: String,
    pub episode_id: Uuid,
    pub input: Input,
    pub dryrun: bool,
    pub start_time: Instant,
    pub inference_params: InferenceParams,
    pub model_name: &'a str,
    pub model_provider_name: &'a str,
    pub raw_request: String,
    pub system: Option<String>,
    pub input_messages: Vec<RequestMessage>,
    pub previous_model_inference_results: Vec<ModelInferenceResponseWithMetadata<'a>>,
    pub tags: HashMap<String, String>,
    pub tool_config: Option<ToolCallConfig>,
    pub dynamic_output_schema: Option<DynamicJSONSchema>,
}

pub type InferenceCredentials = HashMap<String, SecretString>;

/// A handler for the inference endpoint
#[debug_handler(state = AppStateData)]
pub async fn inference_handler(
    State(AppStateData {
        config,
        http_client,
        clickhouse_connection_info,
    }): AppState,
    StructuredJson(params): StructuredJson<Params>,
) -> Result<Response<Body>, Error> {
    let inference_output =
        inference(config, http_client, clickhouse_connection_info, params).await?;
    match inference_output {
        InferenceOutput::NonStreaming(response) => Ok(Json(response).into_response()),
        InferenceOutput::Streaming(stream) => {
            let event_stream = stream.map(prepare_serialized_event);

            Ok(Sse::new(event_stream)
                .keep_alive(axum::response::sse::KeepAlive::new())
                .into_response())
        }
    }
}

pub enum InferenceOutput {
    NonStreaming(InferenceResponse),
    Streaming(Pin<Box<dyn Stream<Item = Option<InferenceResponseChunk>> + Send>>),
}

#[instrument(
    name="inference",
    skip(config, http_client, clickhouse_connection_info, params),
    fields(
        function_name = %params.function_name,
        variant_name = ?params.variant_name,
    )
)]
pub async fn inference(
    config: &'static Config<'static>,
    http_client: reqwest::Client,
    clickhouse_connection_info: ClickHouseConnectionInfo,
    params: Params,
) -> Result<InferenceOutput, Error> {
    // To be used for the Inference table processing_time measurements
    let start_time = Instant::now();
    // Get the function config or return an error if it doesn't exist
    let function = config.get_function(&params.function_name)?;
    let tool_config = function.prepare_tool_config(params.dynamic_tool_params, &config.tools)?;
    // Collect the function variant names as a Vec<&str>
    let mut candidate_variant_names: Vec<&str> =
        function.variants().keys().map(AsRef::as_ref).collect();

    // If the function has no variants, return an error
    if candidate_variant_names.is_empty() {
        return Err(ErrorDetails::InvalidFunctionVariants {
            message: format!("Function `{}` has no variants", params.function_name),
        }
        .into());
    }

    // Validate the input
    function.validate_input(&params.input)?;

    // If a variant is pinned, only that variant should be attempted
    if let Some(ref variant_name) = params.variant_name {
        candidate_variant_names.retain(|k| k == variant_name);

        // If the pinned variant doesn't exist, return an error
        if candidate_variant_names.is_empty() {
            return Err(ErrorDetails::UnknownVariant {
                name: variant_name.to_string(),
            }
            .into());
        }
    }

    // Retrieve or generate the episode ID
    let episode_id = params.episode_id.unwrap_or(Uuid::now_v7());
    validate_episode_id(episode_id)?;

    // Should we store the results?
    let dryrun = params.dryrun.unwrap_or(false);

    // Increment the request count if we're not in dryrun mode
    if !dryrun {
        counter!(
            "request_count",
            "endpoint" => "inference",
            "function_name" => params.function_name.to_string(),
        )
        .increment(1);
        counter!(
            "inference_count",
            "endpoint" => "inference",
            "function_name" => params.function_name.to_string(),
        )
        .increment(1);
    }

    // Should we stream the inference?
    let stream = params.stream.unwrap_or(false);

    // Keep track of which variants failed
    let mut variant_errors = std::collections::HashMap::new();

    // Set up inference config
    let output_schema = params.output_schema.map(DynamicJSONSchema::new);
    let mut inference_config = InferenceConfig {
        function_name: &params.function_name,
        variant_name: None,
        templates: &config.templates,
        tool_config: tool_config.as_ref(),
        dynamic_output_schema: output_schema.as_ref(),
    };

    let inference_clients = InferenceClients {
        http_client: &http_client,
        clickhouse_connection_info: &clickhouse_connection_info,
        credentials: &params.credentials,
    };

    let inference_models = InferenceModels {
        models: &config.models,
        embedding_models: &config.embedding_models,
    };
    // Keep sampling variants until one succeeds
    while !candidate_variant_names.is_empty() {
        let (variant_name, variant) = sample_variant(
            &mut candidate_variant_names,
            function.variants(),
            &params.function_name,
            &episode_id,
        )?;
        // Will be edited by the variant as part of making the request so we must clone here
        let variant_inference_params = params.params.clone();

        inference_config.variant_name = Some(variant_name);
        if stream {
            let result = variant
                .infer_stream(
                    &params.input,
                    &inference_models,
                    function,
                    &inference_config,
                    &inference_clients,
                    variant_inference_params,
                )
                .await;

            // Make sure the response worked (incl. first chunk) prior to launching the thread and starting to return chunks
            let (chunk, stream, model_used_info) = match result {
                Ok((chunk, stream, model_used_info)) => (chunk, stream, model_used_info),
                Err(e) => {
                    tracing::warn!(
                        "functions.{function_name:?}.variants.{variant_name:?} failed during inference: {e}",
                        function_name = params.function_name,
                        variant_name = variant_name,
                    );
                    variant_errors.insert(variant_name.to_string(), e);
                    continue;
                }
            };

            // Create InferenceMetadata for a streaming inference
            let inference_metadata = InferenceMetadata {
                function_name: params.function_name.clone(),
                variant_name: variant_name.to_string(),
                episode_id,
                input: params.input.clone(),
                dryrun,
                start_time,
                inference_params: model_used_info.inference_params,
                model_name: model_used_info.model_name,
                model_provider_name: model_used_info.model_provider_name,
                raw_request: model_used_info.raw_request,
                system: model_used_info.system,
                input_messages: model_used_info.input_messages,
                previous_model_inference_results: model_used_info.previous_model_inference_results,
                tags: params.tags,
                tool_config,
                dynamic_output_schema: output_schema,
            };

            let stream = create_stream(
                function,
                &config.templates,
                inference_metadata,
                chunk,
                stream,
                clickhouse_connection_info,
            );

            return Ok(InferenceOutput::Streaming(Box::pin(stream)));
        } else {
            let result = variant
                .infer(
                    &params.input,
                    &inference_models,
                    function,
                    &inference_config,
                    &inference_clients,
                    variant_inference_params,
                )
                .await;

            let result = match result {
                Ok(result) => result,
                Err(e) => {
                    tracing::warn!(
                        "functions.{function_name}.variants.{variant_name} failed during inference: {e}",
                        function_name = params.function_name,
                        variant_name = variant_name,
                    );
                    variant_errors.insert(variant_name.to_string(), e);
                    continue;
                }
            };

            if !dryrun {
                // Spawn a thread for a trailing write to ClickHouse so that it doesn't block the response

                let result_to_write = result.clone();
                let write_metadata = InferenceDatabaseInsertMetadata {
                    function_name: params.function_name.clone(),
                    variant_name: variant_name.to_string(),
                    episode_id,
                    tool_config,
                    processing_time: Some(start_time.elapsed()),
                    tags: params.tags,
                };

                tokio::spawn(async move {
                    write_inference(
                        &clickhouse_connection_info,
                        params.input,
                        result_to_write,
                        write_metadata,
                    )
                    .await;
                });
            }

            let response = InferenceResponse::new(result, episode_id, variant_name.to_string());

            return Ok(InferenceOutput::NonStreaming(response));
        }
    }

    // Eventually, if we get here, it means we tried every variant and none of them worked
    Err(ErrorDetails::AllVariantsFailed {
        errors: variant_errors,
    }
    .into())
}

fn create_stream<'a>(
    function: &'static FunctionConfig,
    templates: &'static TemplateConfig<'static>,
    metadata: InferenceMetadata<'static>,
    first_chunk: InferenceResultChunk,
    mut stream: InferenceResultStream,
    clickhouse_connection_info: ClickHouseConnectionInfo,
) -> impl Stream<Item = Option<InferenceResponseChunk>> + Send + 'a {
    async_stream::stream! {
        let mut buffer = vec![first_chunk.clone()];

        // Send the first chunk
        yield Some(prepare_response_chunk(&metadata, first_chunk));

        while let Some(chunk) = stream.next().await {
            let chunk = match chunk.ok() {
                Some(c) => c,
                None => continue,
            };
            buffer.push(chunk.clone());
            yield Some(prepare_response_chunk(&metadata, chunk));

        }

        // Send the None chunk to signal the end of the stream
        yield None;

        // IMPORTANT: The following code will not be reached if the stream is interrupted.
        // Only do things that would be ok to skip in that case.
        //
        // For example, if we were using ClickHouse for billing, we would want to store the interrupted requests.
        //
        // If we really care about storing interrupted requests, we should use a drop guard:
        // https://github.com/tokio-rs/axum/discussions/1060
        let InferenceMetadata {
            function_name,
            variant_name,
            episode_id,
            input,
            dryrun,
            start_time,
            inference_params,
            model_name,
            model_provider_name,
            raw_request,
            system,
            input_messages,
            previous_model_inference_results,
            tags,
            tool_config,
            dynamic_output_schema,
        } = metadata;

        if !dryrun {
            let collect_chunks_args = CollectChunksArgs {
                value: buffer,
                system,
                input_messages,
                function,
                model_name,
                model_provider_name,
                raw_request,
                inference_params,
                function_name: &function_name,
                variant_name: &variant_name,
                dynamic_output_schema,
                templates,
                tool_config: tool_config.as_ref(),
            };
            let inference_response: Result<InferenceResult, Error> =
                collect_chunks(collect_chunks_args).await;

            let inference_response = inference_response.ok();

            if let Some(inference_response) = inference_response {
                let mut inference_response = inference_response;
                inference_response.mut_model_inference_results().extend(previous_model_inference_results);
                let write_metadata = InferenceDatabaseInsertMetadata {
                    function_name,
                    variant_name,
                    episode_id,
                    tool_config,
                    processing_time: Some(start_time.elapsed()),
                    tags,
                };
                tokio::spawn(async move {
                    write_inference(
                        &clickhouse_connection_info,
                        input,
                        inference_response,
                        write_metadata,
                    )
                    .await;
                });
            }
        }
    }
}

fn prepare_response_chunk(
    metadata: &InferenceMetadata,
    chunk: InferenceResultChunk,
) -> InferenceResponseChunk {
    InferenceResponseChunk::new(chunk, metadata.episode_id, metadata.variant_name.clone())
}

// Prepares an Event for SSE on the way out of the gateway
// When None is passed in, we send "[DONE]" to the client to signal the end of the stream
fn prepare_serialized_event(chunk: Option<InferenceResponseChunk>) -> Result<Event, Error> {
    if let Some(chunk) = chunk {
        let chunk_json = serde_json::to_value(chunk).map_err(|e| {
            Error::new(ErrorDetails::Inference {
                message: format!("Failed to convert chunk to JSON: {}", e),
            })
        })?;
        Event::default().json_data(chunk_json).map_err(|e| {
            Error::new(ErrorDetails::Inference {
                message: format!("Failed to convert Value to Event: {}", e),
            })
        })
    } else {
        Ok(Event::default().data("[DONE]"))
    }
}

#[derive(Debug)]
pub struct InferenceDatabaseInsertMetadata {
    pub function_name: String,
    pub variant_name: String,
    pub episode_id: Uuid,
    pub tool_config: Option<ToolCallConfig>,
    pub processing_time: Option<Duration>,
    pub tags: HashMap<String, String>,
}

async fn write_inference(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    input: Input,
    result: InferenceResult<'_>,
    metadata: InferenceDatabaseInsertMetadata,
) {
    let model_responses: Vec<serde_json::Value> = result.get_serialized_model_inferences();
    // Write the model responses to the ModelInference table
    for response in model_responses {
        let _ = clickhouse_connection_info
            .write(&[response], "ModelInference")
            .await;
    }
    // Write the inference to the Inference table
    match result {
        InferenceResult::Chat(result) => {
            let chat_inference = ChatInferenceDatabaseInsert::new(result, input, metadata);
            let _ = clickhouse_connection_info
                .write(&[chat_inference], "ChatInference")
                .await;
        }
        InferenceResult::Json(result) => {
            let json_inference = JsonInferenceDatabaseInsert::new(result, input, metadata);
            let _ = clickhouse_connection_info
                .write(&[json_inference], "JsonInference")
                .await;
        }
    }
}

/// InferenceResponse and InferenceResultChunk determine what gets serialized and sent to the client

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(untagged, rename_all = "snake_case")]
pub enum InferenceResponse {
    Chat(ChatInferenceResponse),
    Json(JsonInferenceResponse),
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ChatInferenceResponse {
    pub inference_id: Uuid,
    pub episode_id: Uuid,
    pub variant_name: String,
    pub content: Vec<ContentBlockOutput>,
    pub usage: Usage,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct JsonInferenceResponse {
    pub inference_id: Uuid,
    pub episode_id: Uuid,
    pub variant_name: String,
    pub output: JsonInferenceOutput,
    pub usage: Usage,
}

impl InferenceResponse {
    pub fn new(inference_result: InferenceResult, episode_id: Uuid, variant_name: String) -> Self {
        match inference_result {
            InferenceResult::Chat(result) => InferenceResponse::Chat(ChatInferenceResponse {
                inference_id: result.inference_id,
                episode_id,
                variant_name,
                content: result.content,
                usage: result.usage,
            }),
            InferenceResult::Json(result) => InferenceResponse::Json(JsonInferenceResponse {
                inference_id: result.inference_id,
                episode_id,
                variant_name,
                output: result.output,
                usage: result.usage,
            }),
        }
    }

    pub fn inference_id(&self) -> Uuid {
        match self {
            InferenceResponse::Chat(c) => c.inference_id,
            InferenceResponse::Json(j) => j.inference_id,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(untagged)]
pub enum InferenceResponseChunk {
    Chat(ChatInferenceResponseChunk),
    Json(JsonInferenceResponseChunk),
}

#[derive(Clone, Debug, Serialize)]
pub struct ChatInferenceResponseChunk {
    pub inference_id: Uuid,
    pub episode_id: Uuid,
    pub variant_name: String,
    pub content: Vec<ContentBlockChunk>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Clone, Debug, Serialize)]
pub struct JsonInferenceResponseChunk {
    pub inference_id: Uuid,
    pub episode_id: Uuid,
    pub variant_name: String,
    pub raw: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

impl InferenceResponseChunk {
    fn new(inference_result: InferenceResultChunk, episode_id: Uuid, variant_name: String) -> Self {
        match inference_result {
            InferenceResultChunk::Chat(result) => {
                InferenceResponseChunk::Chat(ChatInferenceResponseChunk {
                    inference_id: result.inference_id,
                    episode_id,
                    variant_name,
                    content: result.content,
                    usage: result.usage,
                })
            }
            InferenceResultChunk::Json(result) => {
                InferenceResponseChunk::Json(JsonInferenceResponseChunk {
                    inference_id: result.inference_id,
                    episode_id,
                    variant_name,
                    raw: result.raw,
                    usage: result.usage,
                })
            }
        }
    }
}

// Carryall struct for clients used in inference
pub struct InferenceClients<'a> {
    pub http_client: &'a reqwest::Client,
    pub clickhouse_connection_info: &'a ClickHouseConnectionInfo,
    pub credentials: &'a InferenceCredentials,
}

// Carryall struct for models used in inference
#[derive(Debug)]
pub struct InferenceModels<'a> {
    pub models: &'a ModelTable,
    pub embedding_models: &'a HashMap<String, EmbeddingModelConfig>,
}

/// InferenceParams is the top-level struct for inference parameters.
/// We backfill these from the configs given in the variants used and ultimately write them to the database.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct InferenceParams {
    pub chat_completion: ChatCompletionInferenceParams,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ChatCompletionInferenceParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
}

impl ChatCompletionInferenceParams {
    pub fn backfill_with_variant_params(
        &mut self,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
        seed: Option<u32>,
        top_p: Option<f32>,
        presence_penalty: Option<f32>,
        frequency_penalty: Option<f32>,
    ) {
        if self.temperature.is_none() {
            self.temperature = temperature;
        }
        if self.max_tokens.is_none() {
            self.max_tokens = max_tokens;
        }
        if self.seed.is_none() {
            self.seed = seed;
        }
        if self.top_p.is_none() {
            self.top_p = top_p;
        }
        if self.presence_penalty.is_none() {
            self.presence_penalty = presence_penalty;
        }
        if self.frequency_penalty.is_none() {
            self.frequency_penalty = frequency_penalty;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::time::Duration;
    use uuid::Uuid;

    use crate::inference::types::{
        ChatInferenceResultChunk, ContentBlockChunk, JsonInferenceResultChunk, TextChunk,
    };

    #[tokio::test]
    async fn test_prepare_event() {
        // Test case 1: Valid Chat ProviderInferenceResponseChunk
        let inference_id = Uuid::now_v7();
        let content = vec![ContentBlockChunk::Text(TextChunk {
            text: "Test content".to_string(),
            id: "0".to_string(),
        })];
        let chunk = InferenceResultChunk::Chat(ChatInferenceResultChunk {
            inference_id,
            content: content.clone(),
            created: 0,
            usage: None,
            raw_response: "".to_string(),
            latency: Duration::from_millis(100),
        });
        let raw_request = "raw request".to_string();
        let inference_metadata = InferenceMetadata {
            function_name: "test_function".to_string(),
            variant_name: "test_variant".to_string(),
            episode_id: Uuid::now_v7(),
            input: Input {
                messages: vec![],
                system: None,
            },
            dryrun: false,
            inference_params: InferenceParams::default(),
            start_time: Instant::now(),
            model_name: "test_model",
            model_provider_name: "test_provider",
            raw_request: raw_request.clone(),
            system: None,
            input_messages: vec![],
            previous_model_inference_results: vec![],
            tags: HashMap::new(),
            tool_config: None,
            dynamic_output_schema: None,
        };

        let result = prepare_response_chunk(&inference_metadata, chunk);
        match result {
            InferenceResponseChunk::Chat(c) => {
                assert_eq!(c.inference_id, inference_id);
                assert_eq!(c.episode_id, inference_metadata.episode_id);
                assert_eq!(c.variant_name, inference_metadata.variant_name);
                assert_eq!(c.content, content);
                assert!(c.usage.is_none());
            }
            InferenceResponseChunk::Json(_) => {
                panic!("Expected ChatInferenceResponseChunk, got JsonInferenceResponseChunk");
            }
        }

        // TODO (#86): You could get the values of the private members using unsafe Rust.
        // For now, we won't and will rely on E2E testing here.
        // This test doesn't do much so consider deleting or doing more.

        // Test case 2: Valid JSON ProviderInferenceResponseChunk
        let inference_id = Uuid::now_v7();
        let chunk = InferenceResultChunk::Json(JsonInferenceResultChunk {
            inference_id,
            raw: "Test content".to_string(),
            created: 0,
            usage: None,
            raw_response: "".to_string(),
            latency: Duration::from_millis(100),
        });
        let inference_metadata = InferenceMetadata {
            function_name: "test_function".to_string(),
            variant_name: "test_variant".to_string(),
            episode_id: Uuid::now_v7(),
            input: Input {
                messages: vec![],
                system: None,
            },
            dryrun: false,
            inference_params: InferenceParams::default(),
            start_time: Instant::now(),
            model_name: "test_model",
            model_provider_name: "test_provider",
            raw_request: raw_request.clone(),
            system: None,
            input_messages: vec![],
            previous_model_inference_results: vec![],
            tags: HashMap::new(),
            tool_config: None,
            dynamic_output_schema: None,
        };

        let result = prepare_response_chunk(&inference_metadata, chunk);
        match result {
            InferenceResponseChunk::Json(c) => {
                assert_eq!(c.inference_id, inference_id);
                assert_eq!(c.episode_id, inference_metadata.episode_id);
                assert_eq!(c.variant_name, inference_metadata.variant_name);
                assert_eq!(c.raw, "Test content");
                assert!(c.usage.is_none());
            }
            InferenceResponseChunk::Chat(_) => {
                panic!("Expected JsonInferenceResponseChunk, got ChatInferenceResponseChunk");
            }
        }
    }
}
