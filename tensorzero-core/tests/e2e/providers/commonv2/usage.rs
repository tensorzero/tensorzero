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
/// - Text content is small (<= 8 chars)
/// - output_tokens > 25 (proves reasoning tokens are counted)
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
            "messages": [{"role": "user", "content": "What is 34 * 57 + 21 / 3? Answer with just the number."}]
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

    // Assert text content is very small (<= 8 chars, expecting just "1959" plus maybe whitespace)
    assert!(
        text_content.len() <= 8,
        "text content should be <= 8 chars, got {} chars: '{text_content}'",
        text_content.len()
    );

    // Assert output_tokens > 25 (proves reasoning tokens are included)
    assert!(
        output_tokens > 25,
        "output_tokens ({output_tokens}) should be > 25. \
        This suggests reasoning tokens may not be included in output_tokens."
    );
}
