use axum::body::Body;
use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::{debug_handler, Json};
use futures::stream::Stream;
use metrics::counter;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::time::Instant;
use tokio_stream::StreamExt;
use uuid::Uuid;

use crate::clickhouse::ClickHouseConnectionInfo;
use crate::config_parser::get_config;
use crate::error::{Error, ResultExt};
use crate::function::sample_variant;
use crate::function::FunctionConfig;
use crate::gateway_util::{AppState, AppStateData, StructuredJson};
use crate::inference::types::{
    collect_chunks, ContentBlockChunk, ContentBlockOutput, Inference, InferenceResult,
    InferenceResultChunk, InferenceResultStream, Input, ModelInferenceResponseChunk, Usage,
};
use crate::tool::{DynamicToolConfig, ToolCallConfig};
use crate::uuid_util::validate_episode_id;
use crate::variant::Variant;

/// The expected payload is a JSON object with the following fields:
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Params {
    // the function name
    function_name: String,
    // the episode ID (if not provided, it'll be set to inference_id)
    // NOTE: DO NOT GENERATE EPISODE IDS MANUALLY. THE API WILL DO THAT FOR YOU.
    episode_id: Option<Uuid>,
    // the input for the inference
    input: Input,
    // the maximum number of tokens to generate (if not provided, the default value will be used)
    #[allow(unused)] // TODO (#55): remove
    max_tokens: Option<u32>,
    // default False
    stream: Option<bool>,
    // if the client would like to pin a specific variant to be used
    // NOTE: YOU SHOULD TYPICALLY LET THE API SELECT A VARIANT FOR YOU (I.E. IGNORE THIS FIELD).
    //       ONLY PIN A VARIANT FOR SPECIAL USE CASES (E.G. DEBUGGING).
    variant_name: Option<String>,
    // if true, the inference will not be stored
    dryrun: Option<bool>,
    // TODO (#126): properly implement dynamic tool calling
    // If provided, the inference will only use the specified tools (a subset of the function's tools)
    // allowed_tools: Option<Vec<String>>,
    // If provided, the inference will use the specified tools in addition to the function's tools
    // additional_tools: Option<Vec<Tool>>,
    // If provided, the inference will use the specified tool choice
    // tool_choice: Option<ToolChoice>,
    // If true, the inference will use parallel tool calls
    // parallel_tool_calls: Option<bool>,
}

#[derive(Debug, Clone)]
struct InferenceMetadata {
    pub function_name: String,
    pub variant_name: String,
    pub episode_id: Uuid,
    pub input: Input,
    pub dryrun: bool,
    pub start_time: Instant,
}

/// A handler for the inference endpoint
#[debug_handler(state = AppStateData)]
pub async fn inference_handler(
    State(AppStateData {
        http_client,
        clickhouse_connection_info,
    }): AppState,
    StructuredJson(params): StructuredJson<Params>,
) -> Result<Response<Body>, Error> {
    // To be used for the Inference table processing_time measurements
    let start_time = Instant::now();
    // Get the function config or return an error if it doesn't exist
    let config = get_config();
    let function = config.get_function(&params.function_name)?;
    // TODO (#126): implement dynamic tool calling
    // Collect the dynamic tool config
    // let dynamic_tool_config = DynamicToolConfig {
    //     allowed_tools: params.allowed_tools.as_ref(),
    //     additional_tools: params.additional_tools.as_ref(),
    //     tool_choice: params.tool_choice.as_ref(),
    //     parallel_tool_calls: params.parallel_tool_calls,
    // };
    let dynamic_tool_config = DynamicToolConfig {
        allowed_tools: None,
        additional_tools: None,
        tool_choice: None,
        parallel_tool_calls: None,
    };
    let dynamic_tool_config_string =
        serde_json::to_string(&dynamic_tool_config).map_err(|e| Error::Serialization {
            message: e.to_string(),
        })?;
    let tool_config = function.prepare_tool_config(&config.tools)?;
    // Clone the function variants so we can modify the collection as we sample them
    let mut variants = function.variants().clone();

    // If the function has no variants, return an error
    if variants.is_empty() {
        return Err(Error::InvalidFunctionVariants {
            message: format!("Function `{}` has no variants", params.function_name),
        });
    }

    // Validate the input
    function.validate_input(&params.input)?;

    // If a variant is pinned, only that variant should be attempted
    if let Some(ref variant_name) = params.variant_name {
        variants.retain(|k, _| k == variant_name);

        // If the pinned variant doesn't exist, return an error
        if variants.is_empty() {
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
    let mut variant_errors = vec![];

    // Keep sampling variants until one succeeds
    while !variants.is_empty() {
        let (variant_name, variant) =
            sample_variant(&mut variants, &params.function_name, &episode_id)?;
        if stream {
            let result = variant
                .infer_stream(
                    &params.input,
                    &config.models,
                    function,
                    tool_config.as_ref(),
                    &http_client,
                )
                .await;

            // Make sure the response worked (incl. first chunk) prior to launching the thread and starting to return chunks
            let (chunk, stream) = match result {
                Ok((chunk, stream)) => (chunk, stream),
                Err(e) => {
                    variant_errors.push(e);
                    continue;
                }
            };

            // Create InferenceMetadata for a streaming inference
            let inference_metadata = InferenceMetadata {
                function_name: params.function_name.clone(),
                variant_name,
                episode_id,
                input: params.input.clone(),
                dryrun,
                start_time,
            };

            let stream = create_stream(
                function,
                inference_metadata,
                chunk,
                stream,
                clickhouse_connection_info,
                tool_config,
                dynamic_tool_config_string,
            );

            return Ok(Sse::new(stream)
                .keep_alive(axum::response::sse::KeepAlive::new())
                .into_response());
        } else {
            let result = variant
                .infer(
                    &params.input,
                    &config.models,
                    function,
                    tool_config.as_ref(),
                    &http_client,
                )
                .await;

            let result = match result {
                Ok(result) => result,
                Err(e) => {
                    variant_errors.push(e);
                    continue;
                }
            };

            if !dryrun {
                // Spawn a thread for a trailing write to ClickHouse so that it doesn't block the response
                let write_metadata = InferenceWriteMetadata {
                    function_name: params.function_name,
                    variant_name: variant_name.clone(),
                    episode_id,
                    dynamic_tool_config: dynamic_tool_config_string,
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

            let response = InferenceResponse::new(result, episode_id, variant_name);

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
    metadata: InferenceMetadata,
    first_chunk: ModelInferenceResponseChunk,
    mut stream: InferenceResultStream,
    clickhouse_connection_info: ClickHouseConnectionInfo,
    tool_config: Option<ToolCallConfig>,
    dynamic_tool_config_string: String,
) -> impl Stream<Item = Result<Event, Error>> + Send {
    async_stream::stream! {
        let mut buffer = vec![first_chunk.clone()];

        // Send the first chunk
        if let Some(event) = prepare_event(function, &metadata, first_chunk).ok_or_log() {
            yield Ok(event);
        }

        while let Some(chunk) = stream.next().await {
            let chunk = match chunk.ok_or_log() {
                Some(c) => c,
                None => continue,
            };
            buffer.push(chunk.clone());
            let event = prepare_event(function, &metadata, chunk).ok_or_log();
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
                collect_chunks(buffer, function, tool_config.as_ref());

            let inference_response = inference_response.ok_or_log();

            if let Some(inference_response) = inference_response {
                let write_metadata = InferenceWriteMetadata {
                    function_name: metadata.function_name,
                    variant_name: metadata.variant_name,
                    episode_id: metadata.episode_id,
                    dynamic_tool_config: dynamic_tool_config_string,
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
    function: &FunctionConfig,
    metadata: &InferenceMetadata,
    chunk: ModelInferenceResponseChunk,
) -> Result<Event, Error> {
    let chunk_json = match function {
        FunctionConfig::Chat(_) => {
            let result_chunk = InferenceResultChunk::Chat(chunk.into());
            let response_chunk = InferenceResponseChunk::new(
                result_chunk,
                metadata.episode_id,
                metadata.variant_name.clone(),
            );
            serde_json::to_value(response_chunk).map_err(|e| Error::Inference {
                message: format!("Failed to convert chunk to JSON: {}", e),
            })?
        }
        FunctionConfig::Json(_) => {
            unimplemented!()
        }
    };

    Event::default()
        .json_data(chunk_json)
        .map_err(|e| Error::Inference {
            message: format!("Failed to convert Value to Event: {}", e),
        })
}

struct InferenceWriteMetadata {
    function_name: String,
    variant_name: String,
    episode_id: Uuid,
    dynamic_tool_config: String,
    processing_time: Duration,
}

async fn write_inference(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    input: Input,
    result: InferenceResult,
    metadata: InferenceWriteMetadata,
) {
    match result {
        InferenceResult::Chat(response) => {
            let serialized_input =
                match serde_json::to_string(&input).map_err(|e| Error::Serialization {
                    message: e.to_string(),
                }) {
                    Ok(serialized_input) => serialized_input,
                    Err(_) => return,
                };

            // Write the model responses to the ModelInference table
            let model_responses: Vec<serde_json::Value> =
                response.get_serialized_model_inferences(&serialized_input);
            for response in model_responses {
                clickhouse_connection_info
                    .write(&response, "ModelInference")
                    .await
                    .ok_or_log();
            }

            // Write the inference to the Inference table
            let inference = Inference::new(
                InferenceResult::Chat(response),
                serialized_input,
                metadata.episode_id,
                metadata.function_name,
                metadata.variant_name,
                metadata.dynamic_tool_config,
                metadata.processing_time,
            );
            clickhouse_connection_info
                .write(&inference, "Inference")
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
}

#[derive(Debug, Clone, Serialize)]
struct ChatInferenceResponse {
    inference_id: Uuid,
    episode_id: Uuid,
    variant_name: String,
    output: Vec<ContentBlockOutput>,
    usage: Usage,
}

impl InferenceResponse {
    fn new(inference_result: InferenceResult, episode_id: Uuid, variant_name: String) -> Self {
        match inference_result {
            InferenceResult::Chat(result) => InferenceResponse::Chat(ChatInferenceResponse {
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
}

#[derive(Clone, Debug, Serialize)]
struct ChatInferenceResponseChunk {
    inference_id: Uuid,
    episode_id: Uuid,
    variant_name: String,
    content: Vec<ContentBlockChunk>,
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::{collections::HashMap, time::Duration};
    use uuid::Uuid;

    use crate::function::FunctionConfigChat;
    use crate::inference::types::{ContentBlockChunk, ModelInferenceResponseChunk, TextChunk};

    #[tokio::test]
    async fn test_prepare_event() {
        // Test case 1: Valid ModelInferenceResponseChunk
        let chunk = ModelInferenceResponseChunk {
            inference_id: Uuid::now_v7(),
            content: vec![ContentBlockChunk::Text(TextChunk {
                text: "Test content".to_string(),
                id: "0".to_string(),
            })],
            created: 0,
            usage: None,
            raw_response: "".to_string(),
            latency: Duration::from_millis(100),
        };
        let function = FunctionConfig::Chat(FunctionConfigChat {
            variants: HashMap::new(),
            system_schema: None,
            user_schema: None,
            assistant_schema: None,
            tools: vec![],
            ..Default::default()
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
            start_time: Instant::now(),
        };

        let result = prepare_event(&function, &inference_metadata, chunk);
        assert!(result.is_ok());
        // TODO (#86): You could get the values of the private members using unsafe Rust.
        // For now, we won't and will rely on integration testing here.
        // This test doesn't do much so consider deleting or doing more.
    }
}
