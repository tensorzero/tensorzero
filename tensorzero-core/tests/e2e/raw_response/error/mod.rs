//! E2E tests for `raw_response` in error responses.
//!
//! Organized by error timing:
//! - `non_streaming`: Non-streaming requests that fail
//! - `streaming_pre_stream`: Streaming requests where all providers fail before any SSE chunk is sent
//! - `streaming_mid_stream`: Streaming requests where a provider error occurs after chunks have been sent

mod best_of_n_sampling;
mod non_streaming;
mod streaming_mid_stream;
mod streaming_pre_stream;

use serde_json::Value;

/// Helper to assert a raw_response entry in an error response has valid structure.
///
/// Error entries differ from success entries: `model_inference_id` should be absent
/// (since the inference failed before a `model_inference_id` could be assigned,
/// and `None` values are omitted via `skip_serializing_if`).
pub fn assert_error_raw_response_entry(entry: &Value) {
    assert!(
        entry.get("model_inference_id").is_none(),
        "model_inference_id should be absent for error entries (no successful model inference)"
    );

    let provider_type = entry
        .get("provider_type")
        .expect("raw_response entry should have provider_type");
    assert!(
        provider_type.is_string(),
        "provider_type should be a string, got: {provider_type:?}"
    );

    let api_type = entry
        .get("api_type")
        .expect("raw_response entry should have api_type");
    let api_type_str = api_type.as_str().expect("api_type should be a string");
    assert!(
        ["chat_completions", "responses", "embeddings", "other"].contains(&api_type_str),
        "api_type should be a valid value, got: {api_type_str}"
    );

    let data = entry
        .get("data")
        .expect("raw_response entry should have data");
    let data_str = data.as_str().expect("data should be a string");
    assert!(!data_str.is_empty(), "data should not be empty");
}
