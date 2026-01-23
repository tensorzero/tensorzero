//! OpenAI-compatible error handling.
//!
//! This module provides error types and extractors that format responses according to OpenAI's API specification.

use std::fmt;

use axum::Json;
use axum::extract::rejection::JsonRejection;
use axum::extract::{FromRequest, Request};
use axum::response::{IntoResponse, Response};
use serde::de::DeserializeOwned;
use serde_json::json;
use tracing::instrument;

use crate::error::{Error, ErrorDetails};

/// A wrapper around `Error` that implements `IntoResponse` with OpenAI-compatible error format.
///
/// OpenAI returns errors as `{"error": {"message": "..."}}` while TensorZero's default
/// format is `{"error": "..."}`. This wrapper ensures OpenAI-compatible endpoints
/// return errors in the expected format.
#[derive(Debug)]
pub struct OpenAICompatibleError(pub Error);

impl From<Error> for OpenAICompatibleError {
    fn from(error: Error) -> Self {
        OpenAICompatibleError(error)
    }
}

impl fmt::Display for OpenAICompatibleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl IntoResponse for OpenAICompatibleError {
    fn into_response(self) -> Response {
        let message = self.0.to_string();
        let body = json!({
            "error": {
                "message": message,
            },
        });
        let mut response = (self.0.status_code(), Json(body)).into_response();
        response.extensions_mut().insert(self.0);
        response
    }
}

/// A JSON extractor for OpenAI-compatible endpoints that returns errors in OpenAI format.
///
/// This is similar to `StructuredJson` but uses `OpenAICompatibleError` as its rejection type
/// so that JSON parsing errors are returned in OpenAI's `{"error": {"message": "..."}}` format.
#[derive(Debug, Clone, Copy, Default)]
pub struct OpenAIStructuredJson<T>(pub T);

impl<S, T> FromRequest<S> for OpenAIStructuredJson<T>
where
    Json<T>: FromRequest<S, Rejection = JsonRejection>,
    S: Send + Sync,
    T: Send + Sync + DeserializeOwned,
{
    type Rejection = OpenAICompatibleError;

    #[instrument(skip_all, level = "trace", name = "OpenAIStructuredJson::from_request")]
    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        // Retrieve the request body as Bytes before deserializing it
        let bytes = bytes::Bytes::from_request(req, state).await.map_err(|e| {
            OpenAICompatibleError(Error::new(ErrorDetails::JsonRequest {
                message: format!("{} ({})", e, e.status()),
            }))
        })?;

        // Convert the entire body into `serde_json::Value`
        let value = Json::<serde_json::Value>::from_bytes(&bytes)
            .map_err(|e| {
                OpenAICompatibleError(Error::new(ErrorDetails::JsonRequest {
                    message: format!("{} ({})", e, e.status()),
                }))
            })?
            .0;

        // Now use `serde_path_to_error::deserialize` to attempt deserialization into `T`
        let deserialized: T = serde_path_to_error::deserialize(&value).map_err(|e| {
            OpenAICompatibleError(Error::new(ErrorDetails::JsonRequest {
                message: e.to_string(),
            }))
        })?;

        Ok(OpenAIStructuredJson(deserialized))
    }
}
