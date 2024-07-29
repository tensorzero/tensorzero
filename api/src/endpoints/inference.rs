use std::convert::Infallible;

use axum::body::Body;
use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::{debug_handler, Json};
use serde::Deserialize;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::StreamExt;
use uuid::Uuid;

use crate::api_util::{AppState, AppStateData, StructuredJson};
use crate::error::Error;
use crate::function::{sample_variant, FunctionConfig};
use crate::inference::types::{
    InferenceResponseChunk, InferenceResponseStream, InputMessage, ModelInferenceResponseChunk,
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

/// A handler for the inference endpoint
#[debug_handler(state = AppStateData)]
pub async fn inference_handler(
    State(AppStateData {
        config,
        http_client,
        ..
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
                name: params.variant_name.unwrap(),
            });
        }
    }

    // Retrieve or generate the episode ID
    // TODO: validate that the episode ID is a UUIDv7
    let episode_id = params.episode_id.unwrap_or(Uuid::now_v7());

    // Should we store the results?
    #[allow(unused)] // TODO: remove
    let dryrun = params.dryrun.unwrap_or(false);

    // Should we stream the inference?
    let stream = params.stream.unwrap_or(false);

    // Keep track of which variants failed
    let mut variant_error = vec![];

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
            let (chunk, stream) = match response {
                Ok((chunk, stream)) => (chunk, stream),
                Err(e) => {
                    variant_error.push(e);
                    continue;
                }
            };
            let (client_sender, client_receiver) = mpsc::unbounded_channel();
            tokio::spawn(worker_response_router(
                // TODO: figure out how to get the Rust compiler to realize that the FunctionConfig
                // can live forever
                function.clone(),
                variant_name,
                chunk,
                stream,
                client_sender,
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
                    variant_error.push(e);
                    continue;
                }
            };
            let response_value = serde_json::to_value(response).map_err(|e| Error::Inference {
                message: format!("Failed to convert response to JSON: {}", e),
            })?;
            return Ok(Json(response_value).into_response());
        }
        // TODO: figure out storage
    }

    // Eventually, if we get here, it means we tried every variant and none of them worked
    // TODO: make this a nicer error that contains all the errors from below.
    Err(Error::Inference {
        message: "Inference failed for every variant in this function".to_string(),
    })
}

async fn worker_response_router(
    function: FunctionConfig,
    variant_name: String,
    first_chunk: ModelInferenceResponseChunk,
    mut stream: InferenceResponseStream,
    client_sender: mpsc::UnboundedSender<Result<Event, Infallible>>,
) {
    let mut buffer = vec![first_chunk.clone()];
    let mut errors: Vec<Error> = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = match chunk {
            Ok(chunk) => {
                buffer.push(chunk.clone());
                chunk
            }
            Err(e) => {
                errors.push(e);
                continue;
            }
        };
        let event = prepare_event(&function, &variant_name, chunk);
        match event {
            Ok(event) => {
                let r = client_sender.send(Ok(event));
                if let Err(e) = r {
                    errors.push(Error::Inference {
                        message: format!("Failed to send event to client: {}", e),
                    });
                }
            }
            Err(e) => {
                errors.push(e);
            }
        }
    }
    // TODO: log errors and also store results
}

fn prepare_event(
    function: &FunctionConfig,
    variant_name: &str,
    chunk: ModelInferenceResponseChunk,
) -> Result<Event, Error> {
    let mut chunk_json = match function {
        FunctionConfig::Chat(_) => {
            let chunk = InferenceResponseChunk::Chat(chunk.into());
            serde_json::to_value(chunk).map_err(|e| Error::Inference {
                message: format!("Failed to convert chunk to JSON: {}", e),
            })?
        }
        _ => {
            unimplemented!()
        }
    };
    chunk_json["variant_name"] = variant_name.to_string().into();
    Event::default()
        .json_data(chunk_json)
        .map_err(|e| Error::Inference {
            message: format!("Failed to convert Value to Event: {}", e),
        })
}
