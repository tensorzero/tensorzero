use axum::body::Body;
use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::{debug_handler, Json};
use futures::stream::Stream;
use metrics::counter;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::time::Duration;
use tokio::time::Instant;
use tokio_stream::StreamExt;
use uuid::Uuid;

use crate::clickhouse::ClickHouseConnectionInfo;
use crate::embeddings::EmbeddingModelConfig;
use crate::error::{Error, ResultExt};
use crate::function::sample_variant;
use crate::function::FunctionConfig;
use crate::gateway_util::{AppState, AppStateData, StructuredJson};
use crate::inference::providers::anthropic::AnthropicCredentials;
use crate::inference::providers::azure::AzureCredentials;
#[cfg(any(test, feature = "e2e_tests"))]
use crate::inference::providers::dummy::DummyCredentials;
use crate::inference::providers::fireworks::FireworksCredentials;
use crate::inference::providers::gcp_vertex_anthropic::GCPVertexAnthropicCredentials;
use crate::inference::providers::gcp_vertex_gemini::GCPVertexGeminiCredentials;
use crate::inference::providers::mistral::MistralCredentials;
use crate::inference::providers::openai::OpenAICredentials;
use crate::inference::providers::together::TogetherCredentials;
use crate::inference::providers::vllm::VLLMCredentials;
use crate::inference::types::{
    collect_chunks, ChatInferenceDatabaseInsert, ContentBlockChunk, ContentBlockOutput,
    InferenceResult, InferenceResultChunk, InferenceResultStream, Input,
    JsonInferenceDatabaseInsert, JsonInferenceOutput, ModelInferenceResponseWithMetadata, Usage,
};
use crate::jsonschema_util::DynamicJSONSchema;
use crate::model::ModelConfig;
use crate::tool::{DynamicToolParams, ToolCallConfig};
use crate::uuid_util::validate_episode_id;
use crate::variant::{InferenceConfig, Variant};

/// The expected payload is a JSON object with the following fields:
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Params<'a> {
    // the function name
    function_name: String,
    // the episode ID (if not provided, it'll be set to inference_id)
    // NOTE: DO NOT GENERATE EPISODE IDS MANUALLY. THE API WILL DO THAT FOR YOU.
    episode_id: Option<Uuid>,
    // the input for the inference
    input: Input,
    // default False
    stream: Option<bool>,
    // Inference-time overrides for variant types (use with caution)
    #[serde(default)]
    params: InferenceParams,
    // if the client would like to pin a specific variant to be used
    // NOTE: YOU SHOULD TYPICALLY LET THE API SELECT A VARIANT FOR YOU (I.E. IGNORE THIS FIELD).
    //       ONLY PIN A VARIANT FOR SPECIAL USE CASES (E.G. TESTING / DEBUGGING VARIANTS).
    variant_name: Option<String>,
    // if true, the inference will not be stored
    dryrun: Option<bool>,
    // dynamic information about tool calling. Don't directly include `dynamic_tool_params` in `Params`.
    #[serde(flatten)]
    dynamic_tool_params: DynamicToolParams,
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
    output_schema: Option<Value>,
    #[serde(default)]
    credentials: InferenceCredentials<'a>,
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
    pub previous_model_inference_results: Vec<ModelInferenceResponseWithMetadata<'a>>,
}

#[derive(Debug, Deserialize, Default)]
pub struct InferenceCredentials<'a> {
    pub anthropic: Option<AnthropicCredentials<'a>>,
    pub azure: Option<AzureCredentials<'a>>,
    pub fireworks: Option<FireworksCredentials<'a>>,
    pub gcp_vertex_anthropic: Option<GCPVertexAnthropicCredentials>,
    pub gcp_vertex_gemini: Option<GCPVertexGeminiCredentials>,
    pub mistral: Option<MistralCredentials<'a>>,
    pub openai: Option<OpenAICredentials<'a>>,
    pub together: Option<TogetherCredentials<'a>>,
    pub vllm: Option<VLLMCredentials<'a>>,
    #[cfg(any(test, feature = "e2e_tests"))]
    pub dummy: Option<DummyCredentials<'a>>,
}

/// A handler for the inference endpoint
#[debug_handler(state = AppStateData)]
pub async fn inference_handler(
    State(AppStateData {
        config,
        http_client,
        clickhouse_connection_info,
    }): AppState,
    StructuredJson(params): StructuredJson<Params<'static>>,
) -> Result<Response<Body>, Error> {
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
        return Err(Error::InvalidFunctionVariants {
            message: format!("Function `{}` has no variants", params.function_name),
        });
    }

    // Validate the input
    function.validate_input(&params.input)?;

    // If a variant is pinned, only that variant should be attempted
    if let Some(ref variant_name) = params.variant_name {
        candidate_variant_names.retain(|k| k == variant_name);

        // If the pinned variant doesn't exist, return an error
        if candidate_variant_names.is_empty() {
            return Err(Error::UnknownVariant {
                name: variant_name.to_string(),
            });
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
    }

    // Should we stream the inference?
    let stream = params.stream.unwrap_or(false);

    // Keep track of which variants failed
    let mut variant_errors = std::collections::HashMap::new();
    let mut inference_config = InferenceConfig {
        function_name: params.function_name.clone(),
        variant_name: "".to_string(),
        templates: &config.templates,
        tool_config,
        dynamic_output_schema: params.output_schema.map(DynamicJSONSchema::new),
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
        inference_config.variant_name = variant_name.to_string();
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
                previous_model_inference_results: model_used_info.previous_model_inference_results,
            };

            let stream = create_stream(
                function,
                inference_metadata,
                chunk,
                stream,
                clickhouse_connection_info,
                inference_config,
            );

            return Ok(Sse::new(stream)
                .keep_alive(axum::response::sse::KeepAlive::new())
                .into_response());
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
                let write_metadata = InferenceDatabaseInsertMetadata {
                    function_name: params.function_name,
                    variant_name: variant_name.to_string(),
                    episode_id,
                    tool_params: inference_config.tool_config,
                    processing_time: start_time.elapsed(),
                };

                let result_to_write = result.clone();

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

            let response_serialized =
                serde_json::to_value(response).map_err(|e| Error::Inference {
                    message: format!("Failed to convert response to JSON: {}", e),
                })?;

            return Ok(Json(response_serialized).into_response());
        }
    }

    // Eventually, if we get here, it means we tried every variant and none of them worked
    Err(Error::AllVariantsFailed {
        errors: variant_errors,
    })
}

fn create_stream(
    function: &'static FunctionConfig,
    metadata: InferenceMetadata<'static>,
    first_chunk: InferenceResultChunk,
    mut stream: InferenceResultStream,
    clickhouse_connection_info: ClickHouseConnectionInfo,
    inference_config: InferenceConfig<'static>,
) -> impl Stream<Item = Result<Event, Error>> + Send {
    async_stream::stream! {
        let mut buffer = vec![first_chunk.clone()];

        // Send the first chunk
        if let Some(event) = prepare_event(&metadata, first_chunk).ok_or_log() {
            yield Ok(event);
        }

        while let Some(chunk) = stream.next().await {
            let chunk = match chunk.ok_or_log() {
                Some(c) => c,
                None => continue,
            };
            buffer.push(chunk.clone());
            let event = prepare_event(&metadata, chunk).ok_or_log();
            if let Some(event) = event {
                yield Ok(event);
            }
        }

        // Send the [DONE] event to signal the end of the stream
        yield Ok(Event::default().data("[DONE]"));

        // IMPORTANT: The following code will not be reached if the stream is interrupted.
        // Only do things that would be ok to skip in that case.
        //
        // For example, if we were using ClickHouse for billing, we would want to store the interrupted requests.
        //
        // If we really care about storing interrupted requests, we should use a drop guard:
        // https://github.com/tokio-rs/axum/discussions/1060

        if !metadata.dryrun {
            let inference_response: Result<InferenceResult, Error> =
                collect_chunks(buffer, function, &inference_config, metadata.model_name, metadata.model_provider_name, metadata.raw_request, metadata.inference_params).await;

            let inference_response = inference_response.ok_or_log();

            if let Some(inference_response) = inference_response {
                let mut inference_response = inference_response;
                inference_response.mut_model_inference_results().extend(metadata.previous_model_inference_results);
                let write_metadata = InferenceDatabaseInsertMetadata {
                    function_name: metadata.function_name,
                    variant_name: metadata.variant_name,
                    episode_id: metadata.episode_id,
                    tool_params: inference_config.tool_config,
                    processing_time: metadata.start_time.elapsed(),
                };
                tokio::spawn(async move {
                    write_inference(
                        &clickhouse_connection_info,
                        metadata.input,
                        inference_response,
                        write_metadata,
                    )
                    .await;
                });
            }
        }
    }
}

fn prepare_event(
    metadata: &InferenceMetadata,
    chunk: InferenceResultChunk,
) -> Result<Event, Error> {
    let response_chunk =
        InferenceResponseChunk::new(chunk, metadata.episode_id, metadata.variant_name.clone());
    let chunk_json = serde_json::to_value(response_chunk).map_err(|e| Error::Inference {
        message: format!("Failed to convert chunk to JSON: {}", e),
    })?;
    Event::default()
        .json_data(chunk_json)
        .map_err(|e| Error::Inference {
            message: format!("Failed to convert Value to Event: {}", e),
        })
}

#[derive(Debug)]
pub struct InferenceDatabaseInsertMetadata {
    pub function_name: String,
    pub variant_name: String,
    pub episode_id: Uuid,
    pub tool_params: Option<ToolCallConfig>,
    pub processing_time: Duration,
}

async fn write_inference<'a>(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    input: Input,
    result: InferenceResult<'a>,
    metadata: InferenceDatabaseInsertMetadata,
) {
    let serialized_input = match serde_json::to_string(&input).map_err(|e| Error::Serialization {
        message: e.to_string(),
    }) {
        Ok(serialized_input) => serialized_input,
        Err(e) => {
            e.log();
            return;
        }
    };
    let model_responses: Vec<serde_json::Value> = result.get_serialized_model_inferences();
    // Write the model responses to the ModelInference table
    for response in model_responses {
        clickhouse_connection_info
            .write(&response, "ModelInference")
            .await
            .ok_or_log();
    }
    // Write the inference to the Inference table
    match result {
        InferenceResult::Chat(result) => {
            let chat_inference =
                ChatInferenceDatabaseInsert::new(result, serialized_input, metadata);
            clickhouse_connection_info
                .write(&chat_inference, "ChatInference")
                .await
                .ok_or_log();
        }
        InferenceResult::Json(result) => {
            let json_inference =
                JsonInferenceDatabaseInsert::new(result, serialized_input, metadata);
            clickhouse_connection_info
                .write(&json_inference, "JsonInference")
                .await
                .ok_or_log();
        }
    }
}

/// InferenceResponse and InferenceResultChunk determine what gets serialized and sent to the client

#[derive(Clone, Debug, Serialize)]
#[serde(untagged, rename_all = "snake_case")]
enum InferenceResponse {
    Chat(ChatInferenceResponse),
    Json(JsonInferenceResponse),
}

#[derive(Clone, Debug, Serialize)]
struct ChatInferenceResponse {
    inference_id: Uuid,
    episode_id: Uuid,
    variant_name: String,
    content: Vec<ContentBlockOutput>,
    usage: Usage,
}

#[derive(Clone, Debug, Serialize)]
struct JsonInferenceResponse {
    inference_id: Uuid,
    episode_id: Uuid,
    variant_name: String,
    output: JsonInferenceOutput,
    usage: Usage,
}

impl InferenceResponse {
    fn new(inference_result: InferenceResult, episode_id: Uuid, variant_name: String) -> Self {
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
}

#[derive(Clone, Debug, Serialize)]
#[serde(untagged)]
enum InferenceResponseChunk {
    Chat(ChatInferenceResponseChunk),
    Json(JsonInferenceResponseChunk),
}

#[derive(Clone, Debug, Serialize)]
struct ChatInferenceResponseChunk {
    inference_id: Uuid,
    episode_id: Uuid,
    variant_name: String,
    content: Vec<ContentBlockChunk>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<Usage>,
}

#[derive(Clone, Debug, Serialize)]
struct JsonInferenceResponseChunk {
    inference_id: Uuid,
    episode_id: Uuid,
    variant_name: String,
    raw: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<Usage>,
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
    pub credentials: &'a InferenceCredentials<'a>,
}

// Carryall struct for models used in inference
#[derive(Debug)]
pub struct InferenceModels<'a> {
    pub models: &'a HashMap<String, ModelConfig>,
    pub embedding_models: &'a HashMap<String, EmbeddingModelConfig>,
}

/// InferenceParams is the top-level struct for inference parameters.
/// We backfill these from the configs given in the variants used and ultimately write them to the database.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct InferenceParams {
    pub chat_completion: ChatCompletionInferenceParams,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct ChatCompletionInferenceParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u32>,
}

impl ChatCompletionInferenceParams {
    pub fn backfill_with_variant_params(
        &mut self,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
        seed: Option<u32>,
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
        let chunk = InferenceResultChunk::Chat(ChatInferenceResultChunk {
            inference_id: Uuid::now_v7(),
            content: vec![ContentBlockChunk::Text(TextChunk {
                text: "Test content".to_string(),
                id: "0".to_string(),
            })],
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
            previous_model_inference_results: vec![],
        };

        let result = prepare_event(&inference_metadata, chunk);
        assert!(result.is_ok());
        // TODO (#86): You could get the values of the private members using unsafe Rust.
        // For now, we won't and will rely on E2E testing here.
        // This test doesn't do much so consider deleting or doing more.

        // Test case 2: Valid JSON ProviderInferenceResponseChunk
        let chunk = InferenceResultChunk::Json(JsonInferenceResultChunk {
            inference_id: Uuid::now_v7(),
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
            previous_model_inference_results: vec![],
        };

        let result = prepare_event(&inference_metadata, chunk);
        assert!(result.is_ok());
    }
}
