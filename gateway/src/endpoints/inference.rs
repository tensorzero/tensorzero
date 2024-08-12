use axum::body::Body;
use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::{debug_handler, Json};
use futures::stream::Stream;
use metrics::counter;
use serde::Deserialize;
use std::time::Duration;
use tokio::time::Instant;
use tokio_stream::StreamExt;
use uuid::Uuid;

use crate::clickhouse::ClickHouseConnectionInfo;
use crate::error::{Error, ResultExt};
use crate::function::sample_variant;
use crate::gateway_util::{AppState, AppStateData, StructuredJson};
use crate::inference::types::{
    collect_chunks, FunctionConfig, Inference, InferenceResponse, InferenceResponseChunk,
    InferenceResponseStream, InferenceResponseWithOutputSchema, Input, ModelInferenceResponseChunk,
};
use crate::tool::{DynamicToolConfig, Tool, ToolChoice};
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
    // If provided, the inference will only use the specified tools (a subset of the function's tools)
    allowed_tools: Option<Vec<String>>,
    // If provided, the inference will use the specified tools in addition to the function's tools
    additional_tools: Option<Vec<Tool>>,
    // If provided, the inference will use the specified tool choice
    tool_choice: Option<ToolChoice>,
    // If true, the inference will use parallel tool calls
    parallel_tool_calls: Option<bool>,
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
        config,
        http_client,
        clickhouse_connection_info,
    }): AppState,
    StructuredJson(params): StructuredJson<Params>,
) -> Result<Response<Body>, Error> {
    // To be used for the Inference table processing_time measurements
    let start_time = Instant::now();
    // Get the function config or return an error if it doesn't exist
    let function = config.get_function(&params.function_name)?;
    // Collect the dynamic tool config
    let dynamic_tool_config = DynamicToolConfig {
        allowed_tools: params.allowed_tools.as_ref(),
        additional_tools: params.additional_tools.as_ref(),
        tool_choice: params.tool_choice.as_ref(),
        parallel_tool_calls: params.parallel_tool_calls,
    };
    let tool_config = function.prepare_tool_config(&dynamic_tool_config, &config.tools)?;
    let dynamic_tool_config_string =
        serde_json::to_string(&dynamic_tool_config).map_err(|e| Error::Serialization {
            message: e.to_string(),
        })?;
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
            let response = variant
                .infer_stream(
                    &params.input,
                    &config.models,
                    function,
                    tool_config.as_ref(),
                    &http_client,
                )
                .await;

            // Make sure the response worked (incl. first chunk) prior to launching the thread and starting to return chunks
            let (chunk, stream) = match response {
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
                function.clone(),
                inference_metadata,
                chunk,
                stream,
                clickhouse_connection_info,
                dynamic_tool_config_string,
            );

            return Ok(Sse::new(stream)
                .keep_alive(axum::response::sse::KeepAlive::new())
                .into_response());
        } else {
            let response = variant
                .infer(
                    &params.input,
                    &config.models,
                    function,
                    tool_config.as_ref(),
                    &http_client,
                )
                .await;
            let response = match response {
                Ok(response) => response,
                Err(e) => {
                    variant_errors.push(e);
                    continue;
                }
            };
            let response_to_write = response.clone();

            if !dryrun {
                // Spawn a thread for a trailing write to ClickHouse so that it doesn't block the response
                tokio::spawn(async move {
                    write_inference(
                        &clickhouse_connection_info,
                        params.function_name,
                        variant_name,
                        params.input,
                        response_to_write,
                        episode_id,
                        start_time.elapsed(),
                        dynamic_tool_config_string,
                    )
                    .await;
                });
            }
            let response_with_output_schema = InferenceResponseWithOutputSchema {
                inference_response: response,
                output_schema: function.output_schema(),
            };
            let response_value =
                serde_json::to_value(response_with_output_schema).map_err(|e| {
                    Error::Inference {
                        message: format!("Failed to convert response to JSON: {}", e),
                    }
                })?;
            return Ok(Json(response_value).into_response());
        }
    }
    // Eventually, if we get here, it means we tried every variant and none of them worked
    Err(Error::AllVariantsFailed {
        errors: variant_errors,
    })
}

fn create_stream(
    function: FunctionConfig,
    metadata: InferenceMetadata,
    first_chunk: ModelInferenceResponseChunk,
    mut stream: InferenceResponseStream,
    clickhouse_connection_info: ClickHouseConnectionInfo,
    dynamic_tool_config_string: String,
) -> impl Stream<Item = Result<Event, Error>> + Send {
    async_stream::stream! {
        let mut buffer = vec![first_chunk.clone()];

        // Send the first chunk
        if let Some(event) = prepare_event(&function, &metadata, first_chunk).ok_or_log() {
            yield Ok(event);
        }

        while let Some(chunk) = stream.next().await {
            let chunk = match chunk.ok_or_log() {
                Some(c) => c,
                None => continue,
            };
            buffer.push(chunk.clone());
            let event = prepare_event(&function, &metadata, chunk).ok_or_log();
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
            let inference_response: Result<InferenceResponse, Error> =
                // TODO (#30): probably get this from FunctionConfig
                collect_chunks(buffer, function.output_schema());
            let inference_response = inference_response.ok_or_log();

            if let Some(inference_response) = inference_response {
                tokio::spawn(async move {
                    write_inference(
                        &clickhouse_connection_info,
                        metadata.function_name,
                        metadata.variant_name,
                        metadata.input,
                        inference_response,
                        metadata.episode_id,
                        metadata.start_time.elapsed(),
                        dynamic_tool_config_string,
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
    let mut chunk_json = match function {
        FunctionConfig::Chat(_) => {
            let chunk = InferenceResponseChunk::Chat(chunk.into());
            serde_json::to_value(chunk).map_err(|e| Error::Inference {
                message: format!("Failed to convert chunk to JSON: {}", e),
            })?
        }
        FunctionConfig::Json(_) => {
            unimplemented!()
        }
    };
    chunk_json["variant_name"] = metadata.variant_name.to_string().into();
    Event::default()
        .json_data(chunk_json)
        .map_err(|e| Error::Inference {
            message: format!("Failed to convert Value to Event: {}", e),
        })
}

async fn write_inference(
    clickhouse_connection_info: &ClickHouseConnectionInfo,
    function_name: String,
    variant_name: String,
    input: Input,
    response: InferenceResponse,
    episode_id: Uuid,
    processing_time: Duration,
    dynamic_tool_config: String,
) {
    match response {
        InferenceResponse::Chat(response) => {
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
                InferenceResponse::Chat(response),
                serialized_input,
                episode_id,
                function_name,
                variant_name,
                dynamic_tool_config,
                processing_time,
            );
            clickhouse_connection_info
                .write(&inference, "Inference")
                .await
                .ok_or_log();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::{collections::HashMap, time::Duration};
    use uuid::Uuid;

    use crate::inference::types::{
        ContentBlockChunk, FunctionConfigChat, ModelInferenceResponseChunk, TextChunk,
    };

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
            tools: None,
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
