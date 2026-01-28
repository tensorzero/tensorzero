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
use crate::error::{Error, ErrorDetails};
use crate::utils::gateway::{AppState, AppStateData};
use tensorzero_auth::middleware::RequestApiKeyExtension;

use super::types::chat_completions::{OpenAICompatibleParams, OpenAICompatibleResponse};
use super::types::streaming::prepare_serialized_openai_compatible_events;
use super::{OpenAICompatibleError, OpenAIStructuredJson};

/// A handler for the OpenAI-compatible inference endpoint
#[debug_handler(state = AppStateData)]
pub async fn chat_completions_handler(
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
    OpenAIStructuredJson(openai_compatible_params): OpenAIStructuredJson<OpenAICompatibleParams>,
) -> Result<Response<Body>, OpenAICompatibleError> {
    // Validate `n` parameter
    if let Some(n) = openai_compatible_params.n
        && n != 1
    {
        return Err(Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
                message: "TensorZero does not support `n` other than 1. Please omit this parameter or set it to 1.".to_string(),
            }).into());
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
            }).into());
        }
        tracing::warn!(
            "Ignoring unknown fields in OpenAI-compatible request: {:?}",
            openai_compatible_params
                .unknown_fields
                .keys()
                .collect::<Vec<_>>()
        );
    }
    let include_raw_usage = openai_compatible_params.tensorzero_include_raw_usage;
    let include_original_response = openai_compatible_params.tensorzero_include_original_response;
    let include_raw_response = openai_compatible_params.tensorzero_include_raw_response;

    if include_original_response {
        tracing::warn!(
            "The `tensorzero::include_original_response` parameter is deprecated. Use `tensorzero::include_raw_response` instead."
        );
    }

    // Check if user explicitly set include_usage to false
    let explicit_include_usage = openai_compatible_params
        .stream_options
        .as_ref()
        .map(|opts| opts.include_usage);

    // Error if include_raw_usage=true but include_usage is explicitly false
    if openai_compatible_params.stream.unwrap_or(false)
        && include_raw_usage
        && explicit_include_usage == Some(false)
    {
        return Err(Error::new(ErrorDetails::InvalidOpenAICompatibleRequest {
            message: "`tensorzero::include_raw_usage` requires `stream_options.include_usage` to be true (or omitted) for streaming requests".to_string(),
        }).into());
    }

    // OpenAI default: no usage when stream_options is omitted
    // But: include_raw_usage=true implies include_usage=true
    let include_usage = explicit_include_usage.unwrap_or(false) || include_raw_usage;

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
            let openai_compatible_response = OpenAICompatibleResponse::from((
                response,
                response_model_prefix,
                include_original_response,
                include_raw_response,
            ));
            Ok(Json(openai_compatible_response).into_response())
        }
        InferenceOutput::Streaming(stream) => {
            let openai_compatible_stream = prepare_serialized_openai_compatible_events(
                stream,
                response_model_prefix,
                include_usage,
                include_raw_usage,
                include_original_response,
                include_raw_response,
            );
            Ok(Sse::new(openai_compatible_stream)
                .keep_alive(axum::response::sse::KeepAlive::new())
                .into_response())
        }
    }
}
