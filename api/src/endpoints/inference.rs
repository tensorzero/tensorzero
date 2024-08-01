use std::convert::Infallible;

use axum::body::Body;
use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::{debug_handler, Json};
use metrics::counter;
use serde::Deserialize;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::StreamExt;
use uuid::Uuid;

use crate::api_util::{AppState, AppStateData, StructuredJson};
use crate::clickhouse::ClickHouseConnectionInfo;
use crate::error::{Error, ResultExt};
use crate::function::{sample_variant, FunctionConfig};
use crate::inference::types::{
    collect_chunks, Inference, InferenceResponse, InferenceResponseChunk, InferenceResponseStream,
    InputMessage, ModelInferenceResponseChunk,
};
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
    input: Vec<InputMessage>,
    // the maximum number of tokens to generate (if not provided, the default value will be used)
    #[allow(unused)] // TODO: remove
    max_tokens: Option<u32>,
    // default False
    stream: Option<bool>,
    // if the client would like to pin a specific variant to be used
    // NOTE: YOU SHOULD TYPICALLY LET THE API SELECT A VARIANT FOR YOU (I.E. IGNORE THIS FIELD).
    //       ONLY PIN A VARIANT FOR SPECIAL USE CASES (E.G. DEBUGGING).
    variant_name: Option<String>,
    // if true, the inference will not be stored
    dryrun: Option<bool>,
}

#[derive(Debug, Clone)]
struct InferenceMetadata {
    pub function_name: String,
    pub variant_name: String,
    pub episode_id: Uuid,
    pub input: Vec<InputMessage>,
    pub dryrun: bool,
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
    // Get the function config or return an error if it doesn't exist
    let function = config.get_function(&params.function_name)?;

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
    // TODO: validate that the episode ID is a UUIDv7
    let episode_id = params.episode_id.unwrap_or(Uuid::now_v7());

    // Should we store the results?
    let dryrun = params.dryrun.unwrap_or(false);

    // Should we stream the inference?
    let stream = params.stream.unwrap_or(false);

    // Keep track of which variants failed
    let mut variant_errors = vec![];

    // Create InferenceMetadata
    let inference_metadata = InferenceMetadata {
        function_name: params.function_name.clone(),
        variant_name: params.variant_name.clone().unwrap_or_default(),
        episode_id,
        input: params.input.clone(),
        dryrun,
    };

    // Keep sampling variants until one succeeds
    while !variants.is_empty() {
        let (variant_name, variant) =
            sample_variant(&mut variants, &params.function_name, &episode_id)?;
        if stream {
            let response = variant
                .infer_stream(
                    &params.input,
                    &config.models,
                    function.output_schema(),
                    &http_client,
                )
                .await;

            // Make sure the response worked (incl first chunk) prior to launching the thread and starting to return chunks
            // TODO: how would we handle this for composite variants? it kind of breaks the abstraction.
            let (chunk, stream) = match response {
                Ok((chunk, stream)) => (chunk, stream),
                Err(e) => {
                    variant_errors.push(e);
                    continue;
                }
            };
            let (client_sender, client_receiver) = mpsc::unbounded_channel();
            tokio::spawn(worker_response_router(
                function.clone(),
                inference_metadata,
                chunk,
                stream,
                client_sender,
                clickhouse_connection_info,
            ));
            return Ok(Sse::new(UnboundedReceiverStream::new(client_receiver))
                .keep_alive(axum::response::sse::KeepAlive::new())
                .into_response());
        } else {
            let response = variant
                .infer(
                    &params.input,
                    &config.models,
                    function.output_schema(),
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
                // TODO: add integration/E2E test that checks the Prometheus endpoint
                counter!(
                    "request_count",
                    "endpoint" => "inference",
                    "function_name" => params.function_name.to_string(),
                )
                .increment(1);
                // Spawn a thread for a trailing write to ClickHouse so that it doesn't block the response
                tokio::spawn(async move {
                    write_inference(
                        clickhouse_connection_info,
                        params.function_name,
                        variant_name,
                        params.input,
                        response_to_write,
                        episode_id,
                    )
                    .await;
                });
            }
            let response_value = serde_json::to_value(response).map_err(|e| Error::Inference {
                message: format!("Failed to convert response to JSON: {}", e),
            })?;
            return Ok(Json(response_value).into_response());
        }
    }
    // Eventually, if we get here, it means we tried every variant and none of them worked
    Err(Error::AllVariantsFailed {
        errors: variant_errors,
    })
}

async fn worker_response_router(
    function: FunctionConfig,
    metadata: InferenceMetadata,
    first_chunk: ModelInferenceResponseChunk,
    mut stream: InferenceResponseStream,
    client_sender: mpsc::UnboundedSender<Result<Event, Infallible>>,
    clickhouse_connection_info: ClickHouseConnectionInfo,
) {
    let mut buffer = vec![first_chunk.clone()];
    // Send the first chunk
    if let Some(event) = prepare_event(&function, &metadata, first_chunk).ok_or_log() {
        let _ = client_sender
            .send(Ok(event))
            .map_err(|e| Error::ChannelWrite {
                message: e.to_string(),
            })
            .ok_or_log();
    }
    while let Some(chunk) = stream.next().await {
        let chunk = match chunk.ok_or_log() {
            Some(c) => c,
            None => continue,
        };
        buffer.push(chunk.clone());
        let event = prepare_event(&function, &metadata, chunk).ok_or_log();
        if let Some(event) = event {
            let _ = client_sender
                .send(Ok(event))
                .map_err(|e| Error::ChannelWrite {
                    message: e.to_string(),
                })
                .ok_or_log();
        }
    }
    // Send the [DONE] event to signal the end of the stream
    let done_event = Event::default().data("[DONE]");
    let _ = client_sender
        .send(Ok(done_event))
        .map_err(|e| Error::ChannelWrite {
            message: e.to_string(),
        })
        .ok_or_log();
    if !metadata.dryrun {
        counter!(
            "request_count",
            "endpoint" => "inference",
            "function_name" => metadata.function_name.to_string(),
        )
        .increment(1);
        let inference_response: Result<InferenceResponse, Error> =
            collect_chunks(buffer, function.output_schema());
        let inference_response = inference_response.ok_or_log();
        if let Some(inference_response) = inference_response {
            write_inference(
                clickhouse_connection_info,
                metadata.function_name,
                metadata.variant_name,
                metadata.input,
                inference_response,
                metadata.episode_id,
            )
            .await;
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
        FunctionConfig::Tool(_) => {
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
    clickhouse_connection_info: ClickHouseConnectionInfo,
    function_name: String,
    variant_name: String,
    input: Vec<InputMessage>,
    response: InferenceResponse,
    episode_id: Uuid,
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
            let model_responses: Vec<serde_json::Value> =
                response.get_serialized_model_inferences(&serialized_input);
            // TODO : make these writes concurrent
            // (doesn't matter that much since they don't block the main thread but still wastes resources)
            for response in model_responses {
                let res = clickhouse_connection_info
                    .write(&response, "ModelInference")
                    .await;
                res.ok_or_log();
            }
            let inference = Inference::new(
                InferenceResponse::Chat(response),
                serialized_input,
                episode_id,
                function_name,
                variant_name,
            );
            let res = clickhouse_connection_info
                .write(&inference, "Inference")
                .await;
            res.ok_or_log();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::HashMap;
    use uuid::Uuid;

    use crate::{function::FunctionConfigChat, inference::types::ModelInferenceResponseChunk};

    #[tokio::test]
    async fn test_prepare_event() {
        // Test case 1: Valid ModelInferenceResponseChunk
        let chunk = ModelInferenceResponseChunk {
            inference_id: Uuid::now_v7(),
            content: Some("Test content".to_string()),
            tool_calls: None,
            created: 0,
            usage: None,
            raw: "".to_string(),
        };
        let function = FunctionConfig::Chat(FunctionConfigChat {
            variants: HashMap::new(),
            system_schema: None,
            user_schema: None,
            assistant_schema: None,
            output_schema: None,
        });
        let inference_metadata = InferenceMetadata {
            function_name: "test_function".to_string(),
            variant_name: "test_variant".to_string(),
            episode_id: Uuid::now_v7(),
            input: vec![],
            dryrun: false,
        };

        let result = prepare_event(&function, &inference_metadata, chunk);
        assert!(result.is_ok());
        // TODO: You could get the values of the private members using unsafe Rust.
        // For now, we won't and will rely on integration testing here.
        // This test doesn't do much so consider deleting or doing more.
    }
}
