use reqwest::StatusCode;
use thiserror::Error;

use crate::api_type::ApiType;

/// Errors produced by the HTTP client utilities in this crate.
///
/// Callers that need a `tensorzero_error::Error` can convert via the
/// `From<HttpClientError> for tensorzero_error::Error` impl defined in
/// `tensorzero-error`.
#[derive(Debug, Error)]
pub enum HttpClientError {
    #[error("Failed to build HTTP client: {message}")]
    BuildClient { message: String },

    #[error("Failed to convert `{field}` to std::time::Duration: {message}")]
    ConvertDuration { field: String, message: String },

    #[error("Invalid proxy URL: {message}")]
    InvalidProxyUrl { message: String },

    #[error("{message}")]
    InferenceClient {
        message: String,
        status_code: Option<StatusCode>,
        provider_type: String,
        api_type: ApiType,
        raw_request: Option<String>,
        raw_response: Option<String>,
    },

    #[error("{message}")]
    InferenceServer {
        message: String,
        provider_type: String,
        api_type: ApiType,
        raw_request: Option<String>,
        raw_response: Option<String>,
    },
}
