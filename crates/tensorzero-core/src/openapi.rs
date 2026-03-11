use serde::Serialize;

/// Stable baseline error envelope for generated SDKs.
///
/// We intentionally document only the top-level `error` string for now.
/// Internal structured error payloads remain an implementation detail.
#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct TensorZeroErrorResponse {
    pub error: String,
}
