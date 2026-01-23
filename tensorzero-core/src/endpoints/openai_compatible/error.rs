//! OpenAI-compatible error handling.
//!
//! This module provides error types and extractors that format responses according to OpenAI's API specification.

use std::fmt;

use axum::Json;
use axum::extract::rejection::JsonRejection;
use axum::extract::{FromRequest, Request};
use axum::response::{IntoResponse, Response};
use serde::de::DeserializeOwned;
use tracing::instrument;

use crate::error::Error;
use crate::utils::gateway::deserialize_json_request;

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
        let body = self.0.build_response_body(true);
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
        deserialize_json_request(req, state)
            .await
            .map(OpenAIStructuredJson)
            .map_err(OpenAICompatibleError)
    }
}
