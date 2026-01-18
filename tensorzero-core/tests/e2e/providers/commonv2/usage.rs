//! E2E tests for verifying reasoning tokens are included in output_tokens.

use crate::common::get_gateway_endpoint;
use crate::providers::common::E2ETestProvider;
use crate::providers::helpers::get_modal_extra_headers;
use reqwest::{Client, StatusCode};
use serde_json::{Value, json};
use tensorzero_core::inference::types::extra_headers::UnfilteredInferenceExtraHeaders;
use uuid::Uuid;

/// Test that providers correctly include reasoning tokens in output_tokens.
///
/// Makes a single inference call and validates:
/// - Response has >= 1 thought block
/// - Response has exactly 1 text block with < 5 characters
/// - output_tokens > 50 (proves reasoning tokens are counted)
pub async fn test_reasoning_output_tokens_with_provider(provider: E2ETestProvider) {
    println!(
        "Testing reasoning output tokens for provider: {} ({})",
        provider.variant_name, provider.model_provider_name
    );

    let episode_id = Uuid::now_v7();
    let extra_headers = if provider.is_modal_provider() {
        get_modal_extra_headers()
    } else {
        UnfilteredInferenceExtraHeaders::default()
    };

    let payload = json!({
        "function_name": "basic_test",
        "variant_name": provider.variant_name,
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "Calculator"},
            "messages": [{"role": "user", "content": "What is 2+2? Answer with just the number."}]
        },
        "stream": false,
        "extra_headers": extra_headers.extra_headers,
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .expect("failed to send inference request");

    assert_eq!(
        response.status(),
        StatusCode::OK,
        "inference request should succeed"
    );

    let response_json: Value = response
        .json()
        .await
        .expect("failed to parse response JSON");

    println!("Response JSON: {response_json:?}");

    // Extract usage
    let output_tokens = response_json
        .get("usage")
        .and_then(|u| u.get("output_tokens"))
        .and_then(|t| t.as_u64())
        .expect("response should have output_tokens");

    // Parse content blocks
    let content = response_json
        .get("content")
        .and_then(|c| c.as_array())
        .expect("response should have content array");

    let mut thought_count = 0;
    let mut text_count = 0;
    let mut text_content = String::new();

    for block in content {
        match block.get("type").and_then(|t| t.as_str()) {
            Some("thought") => thought_count += 1,
            Some("text") => {
                text_count += 1;
                if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                    text_content.push_str(text);
                }
            }
            _ => {}
        }
    }

    println!(
        "Provider {}: output_tokens={}, thought_count={}, text_count={}, text='{}'",
        provider.variant_name, output_tokens, thought_count, text_count, text_content
    );

    // Assert >= 1 thought block
    assert!(
        thought_count >= 1,
        "response should have >= 1 thought block, got {thought_count}"
    );

    // Assert exactly 1 text block
    assert_eq!(
        text_count, 1,
        "response should have exactly 1 text block, got {text_count}"
    );

    // Assert text content is very small (< 5 chars, expecting just "4")
    assert!(
        text_content.len() < 5,
        "text content should be < 5 chars, got {} chars: '{text_content}'",
        text_content.len()
    );

    // Assert output_tokens > 50 (proves reasoning tokens are included)
    assert!(
        output_tokens > 50,
        "output_tokens ({output_tokens}) should be > 50. \
        This suggests reasoning tokens may not be included in output_tokens."
    );
}
