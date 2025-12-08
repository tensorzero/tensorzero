//! Chat completions endpoint handler for OpenAI-compatible API.
//!
//! This module implements the HTTP handler for the `/openai/v1/chat/completions` endpoint,
//! which provides OpenAI Chat Completions API compatibility. It handles request validation,
//! parameter parsing, inference execution, and response formatting for both streaming
//! and non-streaming requests.

use axum::Json;
use axum::body::Body;
use axum::extract::State;
use axum::response::sse::Sse;
use axum::response::{IntoResponse, Response};
use axum::{Extension, debug_handler};

use crate::endpoints::inference::{InferenceOutput, Params, inference};
use crate::error::{AxumResponseError, Error, ErrorDetails};
use crate::utils::gateway::{AppState, AppStateData, StructuredJson};
use tensorzero_auth::middleware::RequestApiKeyExtension;

use super::types::chat_completions::{OpenAICompatibleParams, OpenAICompatibleResponse};
use super::types::streaming::prepare_serialized_openai_compatible_events;

/// A handler for the OpenAI-compatible inference endpoint
#[debug_handler(state = AppStateData)]
pub async fn chat_completions_handler(
    State(app_state): AppState,
    api_key_ext: Option<Extension<RequestApiKeyExtension>>,
    StructuredJson(openai_compatible_params): StructuredJson<OpenAICompatibleParams>,
) -> Result<Response<Body>, AxumResponseError> {
    async {
        // Validate `n` parameter
        if let Some(n) = openai_compatible_params.n
            && n != 1
        {
            return Err(Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
                    message: "TensorZero does not support `n` other than 1. Please omit this parameter or set it to 1.".to_string(),
                }));
        }

        if !openai_compatible_params.unknown_fields.is_empty() {
            if openai_compatible_params.tensorzero_deny_unknown_fields {
                let mut unknown_field_names = openai_compatible_params
                    .unknown_fields
                    .keys()
                    .cloned()
                    .collect::<Vec<_>>();

                unknown_field_names.sort();
                let unknown_field_names = unknown_field_names.join(", ");

                return Err(Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
                    message: format!(
                        "`tensorzero::deny_unknown_fields` is set to true, but found unknown fields in the request: [{unknown_field_names}]"
                    ),
                }));
            }
            tracing::warn!(
                "Ignoring unknown fields in OpenAI-compatible request: {:?}",
                openai_compatible_params
                    .unknown_fields
                    .keys()
                    .collect::<Vec<_>>()
            );
        }
        let stream_options = openai_compatible_params.stream_options;
        let params = Params::try_from_openai(openai_compatible_params)?;

        // The prefix for the response's `model` field depends on the inference target
        // (We run this disambiguation deep in the `inference` call below but we don't get the decision out, so we duplicate it here)
        let response_model_prefix = match (&params.function_name, &params.model_name) {
            (Some(function_name), None) => Ok::<String, Error>(format!(
                "tensorzero::function_name::{function_name}::variant_name::",
            )),
            (None, Some(_model_name)) => Ok("tensorzero::model_name::".to_string()),
            (Some(_), Some(_)) => Err(ErrorDetails::InvalidInferenceTarget {
                message: "Only one of `function_name` or `model_name` can be provided".to_string(),
            }
            .into()),
            (None, None) => Err(ErrorDetails::InvalidInferenceTarget {
                message: "Either `function_name` or `model_name` must be provided".to_string(),
            }
            .into()),
        }?;

        let response = Box::pin(inference(
            app_state.config.clone(),
            &app_state.http_client,
            app_state.clickhouse_connection_info.clone(),
            app_state.postgres_connection_info.clone(),
            app_state.deferred_tasks.clone(),
            params,
            api_key_ext,
        ))
        .await?;

        match response {
            InferenceOutput::NonStreaming(response) => {
                let openai_compatible_response =
                    OpenAICompatibleResponse::from((response, response_model_prefix));
                Ok(Json(openai_compatible_response).into_response())
            }
            InferenceOutput::Streaming(stream) => {
                let openai_compatible_stream = prepare_serialized_openai_compatible_events(
                    stream,
                    response_model_prefix,
                    stream_options,
                );
                Ok(Sse::new(openai_compatible_stream)
                    .keep_alive(axum::response::sse::KeepAlive::new())
                    .into_response())
            }
        }
    }
    .await
    .map_err(|e| AxumResponseError::new(e, app_state))
}
