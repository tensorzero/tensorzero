//! Messages endpoint handler for Anthropic-compatible API.
//!
//! This module implements the HTTP handler for the `/anthropic/v1/messages` endpoint,
//! which provides Anthropic Messages API compatibility. It handles request validation,
//! parameter parsing, inference execution, and response formatting for both streaming
//! and non-streaming requests.

use axum::Json;
use axum::body::Body;
use axum::extract::State;
use axum::response::sse::Sse;
use axum::response::{IntoResponse, Response};
use axum::{Extension, debug_handler};

use crate::endpoints::anthropic_compatible::types::messages::AnthropicMessageResponse;
use crate::endpoints::anthropic_compatible::types::messages::AnthropicMessagesParams;
use crate::endpoints::anthropic_compatible::types::streaming::prepare_serialized_anthropic_events;
use crate::endpoints::inference::{InferenceOutput, Params, inference};
use crate::error::{Error, ErrorDetails};
use crate::utils::gateway::{AppState, AppStateData, StructuredJson};
use tensorzero_auth::middleware::RequestApiKeyExtension;

/// A handler for the Anthropic-compatible messages endpoint
#[debug_handler(state = AppStateData)]
pub async fn messages_handler(
    State(AppStateData {
        config,
        http_client,
        clickhouse_connection_info,
        postgres_connection_info,
        deferred_tasks,
        rate_limiting_manager,
        ..
    }): AppState,
    api_key_ext: Option<Extension<RequestApiKeyExtension>>,
    StructuredJson(anthropic_params): StructuredJson<AnthropicMessagesParams>,
) -> Result<Response<Body>, Error> {
    // Validate that max_tokens is set (it's required in Anthropic's API)
    if anthropic_params.max_tokens == 0 {
        return Err(Error::new(
            ErrorDetails::InvalidAnthropicCompatibleRequest {
                message: "`max_tokens` is required and must be greater than 0".to_string(),
            },
        ));
    }

    let include_raw_usage = anthropic_params.tensorzero_include_raw_usage;
    let include_raw_response = anthropic_params.tensorzero_include_raw_response;

    let params = Params::try_from_anthropic(anthropic_params)?;

    // The prefix for the response's `model` field depends on the inference target
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
        config,
        &http_client,
        clickhouse_connection_info,
        postgres_connection_info,
        deferred_tasks,
        rate_limiting_manager,
        params,
        api_key_ext,
    ))
    .await?
    .output;

    match response {
        InferenceOutput::NonStreaming(response) => {
            let anthropic_response =
                AnthropicMessageResponse::from((response, response_model_prefix));
            Ok(Json(anthropic_response).into_response())
        }
        InferenceOutput::Streaming(stream) => {
            let anthropic_stream = prepare_serialized_anthropic_events(
                stream,
                response_model_prefix,
                true, // include_usage
                include_raw_usage,
                include_raw_response,
            );
            Ok(Sse::new(anthropic_stream)
                .keep_alive(axum::response::sse::KeepAlive::new())
                .into_response())
        }
    }
}
