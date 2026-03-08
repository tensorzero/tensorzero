//! Error types for tensorzero-types crate.

/// Error type for tensorzero-types operations.
#[derive(Debug, thiserror::Error)]
pub enum TypeError {
    #[error("Invalid base64 data prefix: {0}")]
    InvalidDataPrefix(String),

    #[error("Invalid base64 data: {0}")]
    InvalidBase64(String),

    #[error("Invalid mime type: {0}")]
    InvalidMimeType(String),
}
