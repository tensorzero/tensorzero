//! Serde types for AWS SageMaker Runtime API.
//!
//! SageMaker passes through requests/responses to the hosted model (OpenAI/TGI),
//! so we only need types for error responses.

use serde::Deserialize;

/// Error response from SageMaker Runtime API.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct SageMakerErrorResponse {
    #[serde(default)]
    pub message: Option<String>,
    #[serde(rename = "__type")]
    pub error_type: Option<String>,
}
