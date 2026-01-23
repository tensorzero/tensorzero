//! E2E tests for verifying provider cache tokens are included in input_tokens.
//!
//! Makes two inference requests with large cacheable content:
//! 1. First request: triggers cache write
//! 2. Second request: triggers cache read
//!
//! Both requests should report input_tokens > 4000 to prove that cache tokens
//! (cache_write_input_tokens, cache_read_input_tokens) are properly included
//! in the input_tokens count.
//!
//! This addresses issue #5688 where some providers (e.g., AWS Bedrock) report
//! inputTokens separately from cacheReadInputTokens/cacheWriteInputTokens.

use crate::common::get_gateway_endpoint;
use crate::providers::common::E2ETestProvider;
use crate::providers::helpers::get_modal_extra_headers;
use futures::StreamExt;
use reqwest::{Client, StatusCode};
use reqwest_eventsource::{Event, RequestBuilderExt};
use serde_json::{Value, json};
use tensorzero_core::inference::types::extra_headers::UnfilteredInferenceExtraHeaders;
use uuid::Uuid;

/// Large Lorem Ipsum text (~4200 tokens) for cache tests.
/// This exceeds all provider cache thresholds:
/// - 4096 for Opus 4.5/Haiku 4.5
/// - 2048 for Haiku 3/3.5
/// - 1024 for Sonnet/Opus 4
const LARGE_SYSTEM_PROMPT: &str = include_str!("large_system_prompt.txt");

/// Model-filtered extra_body for all providers that support prompt caching.
/// Each entry is filtered by model_name, so only the relevant cache control
/// marker gets applied for the model being tested.
///
/// Note: Providers not listed here (e.g., OpenAI with automatic caching, or providers
/// without caching support) will still run the test but without cache control markers.
fn cache_control_extra_body() -> Vec<Value> {
    vec![
        // Anthropic API
        json!({
            "model_name": "claude-haiku-4-5-anthropic",
            "pointer": "/system/0/cache_control",
            "value": {"type": "ephemeral"}
        }),
        // AWS Bedrock (claude-haiku-4-5)
        json!({
            "model_name": "claude-haiku-4-5-aws-bedrock",
            "pointer": "/system/-",
            "value": {"cachePoint": {"type": "default"}}
        }),
        // AWS Bedrock (deepseek-r1)
        json!({
            "model_name": "deepseek-r1-aws-bedrock",
            "pointer": "/system/-",
            "value": {"cachePoint": {"type": "default"}}
        }),
        // AWS Bedrock (nova-lite-v1)
        json!({
            "model_name": "nova-lite-v1",
            "pointer": "/system/-",
            "value": {"cachePoint": {"type": "default"}}
        }),
        // GCP Vertex Anthropic (claude-haiku-4-5)
        json!({
            "model_name": "claude-haiku-4-5-gcp-vertex",
            "pointer": "/system/0/cache_control",
            "value": {"type": "ephemeral"}
        }),
    ]
}

/// Test that providers correctly include cache tokens in input_tokens (non-streaming).
///
/// Makes two inference calls with identical large input:
/// 1. First request: triggers cache write
/// 2. Second request: triggers cache read
///
/// Validates:
/// - Both requests have input_tokens > 4000 (proves cache tokens are included)
/// - Both requests have equal input_tokens (consistency check)
pub async fn test_cache_input_tokens_non_streaming_with_provider(provider: E2ETestProvider) {
    println!(
        "Testing cache input tokens for provider: {} ({})",
        provider.variant_name, provider.model_provider_name
    );

    let episode_id = Uuid::now_v7();
    let extra_headers = if provider.is_modal_provider() {
        get_modal_extra_headers()
    } else {
        UnfilteredInferenceExtraHeaders::default()
    };

    let payload = json!({
        "model_name": provider.model_name,
        "episode_id": episode_id,
        "input": {
            "system": LARGE_SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": "Briefly summarize the above text in one sentence."}]
        },
        "stream": false,
        "extra_headers": extra_headers.extra_headers,
        "extra_body": cache_control_extra_body(),
    });

    let client = Client::new();

    // First request - triggers cache write
    let response1 = client
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .expect("failed to send first inference request");

    assert_eq!(
        response1.status(),
        StatusCode::OK,
        "first inference request should succeed"
    );

    let response1_json: Value = response1
        .json()
        .await
        .expect("failed to parse first response JSON");

    println!("First response JSON: {response1_json:?}");

    let input_tokens1 = response1_json
        .get("usage")
        .and_then(|u| u.get("input_tokens"))
        .and_then(|t| t.as_u64())
        .expect("first response should have input_tokens");

    // Second request - triggers cache read
    let response2 = client
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .expect("failed to send second inference request");

    assert_eq!(
        response2.status(),
        StatusCode::OK,
        "second inference request should succeed"
    );

    let response2_json: Value = response2
        .json()
        .await
        .expect("failed to parse second response JSON");

    println!("Second response JSON: {response2_json:?}");

    let input_tokens2 = response2_json
        .get("usage")
        .and_then(|u| u.get("input_tokens"))
        .and_then(|t| t.as_u64())
        .expect("second response should have input_tokens");

    println!(
        "Provider {}: input_tokens1={}, input_tokens2={}",
        provider.variant_name, input_tokens1, input_tokens2
    );

    // Assert input_tokens > 4000 (proves cache tokens are included)
    assert!(
        input_tokens1 > 4000,
        "input_tokens ({input_tokens1}) should be > 4000 for first request (cache write). \
        This suggests cache_write_input_tokens may not be included in input_tokens."
    );

    assert!(
        input_tokens2 > 4000,
        "input_tokens ({input_tokens2}) should be > 4000 for second request (cache read). \
        This suggests cache_read_input_tokens may not be included in input_tokens."
    );

    // Assert consistency between cache write and cache read
    assert_eq!(
        input_tokens1, input_tokens2,
        "input_tokens should be consistent between cache write ({input_tokens1}) and cache read ({input_tokens2})"
    );
}

/// Test that providers correctly include cache tokens in input_tokens (streaming).
///
/// Makes two streaming inference calls with identical large input:
/// 1. First request: triggers cache write
/// 2. Second request: triggers cache read
///
/// Validates:
/// - Both requests have input_tokens > 4000 in final chunk (proves cache tokens are included)
/// - Both requests have equal input_tokens (consistency check)
pub async fn test_cache_input_tokens_streaming_with_provider(provider: E2ETestProvider) {
    println!(
        "Testing streaming cache input tokens for provider: {} ({})",
        provider.variant_name, provider.model_provider_name
    );

    let episode_id = Uuid::now_v7();
    let extra_headers = if provider.is_modal_provider() {
        get_modal_extra_headers()
    } else {
        UnfilteredInferenceExtraHeaders::default()
    };

    let payload = json!({
        "model_name": provider.model_name,
        "episode_id": episode_id,
        "input": {
            "system": LARGE_SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": "Briefly summarize the above text in one sentence."}]
        },
        "stream": true,
        "extra_headers": extra_headers.extra_headers,
        "extra_body": cache_control_extra_body(),
    });

    let client = Client::new();

    // First request - triggers cache write
    let input_tokens1 = get_streaming_input_tokens(&client, &payload, &provider.variant_name)
        .await
        .expect("first streaming request should return input_tokens");

    // Second request - triggers cache read
    let input_tokens2 = get_streaming_input_tokens(&client, &payload, &provider.variant_name)
        .await
        .expect("second streaming request should return input_tokens");

    println!(
        "Provider {} (streaming): input_tokens1={}, input_tokens2={}",
        provider.variant_name, input_tokens1, input_tokens2
    );

    // Assert input_tokens > 4000 (proves cache tokens are included)
    assert!(
        input_tokens1 > 4000,
        "input_tokens ({input_tokens1}) should be > 4000 for first streaming request (cache write). \
        This suggests cache_write_input_tokens may not be included in input_tokens."
    );

    assert!(
        input_tokens2 > 4000,
        "input_tokens ({input_tokens2}) should be > 4000 for second streaming request (cache read). \
        This suggests cache_read_input_tokens may not be included in input_tokens."
    );

    // Assert consistency between cache write and cache read
    assert_eq!(
        input_tokens1, input_tokens2,
        "input_tokens should be consistent between cache write ({input_tokens1}) and cache read ({input_tokens2}) for streaming"
    );
}

/// Helper to get input_tokens from a streaming response.
async fn get_streaming_input_tokens(
    client: &Client,
    payload: &Value,
    variant_name: &str,
) -> Option<u64> {
    let mut chunks = client
        .post(get_gateway_endpoint("/inference"))
        .json(payload)
        .eventsource()
        .unwrap_or_else(|e| {
            panic!(
                "Failed to create eventsource for streaming request for provider {variant_name}: {e}",
            )
        });

    let mut input_tokens: Option<u64> = None;
    let mut all_chunks: Vec<Value> = Vec::new();

    while let Some(chunk) = chunks.next().await {
        let chunk = chunk.unwrap_or_else(|e| {
            panic!("Failed to receive chunk from stream for provider {variant_name}: {e}",)
        });
        let Event::Message(chunk) = chunk else {
            continue;
        };
        if chunk.data == "[DONE]" {
            break;
        }

        let chunk_json: Value = serde_json::from_str(&chunk.data).unwrap_or_else(|e| {
            panic!(
                "Failed to parse chunk as JSON for provider {variant_name}: {e}. Data: {}",
                chunk.data
            )
        });

        all_chunks.push(chunk_json.clone());

        // Check if this chunk has usage (comes in the final chunk)
        if let Some(usage) = chunk_json.get("usage")
            && let Some(tokens) = usage.get("input_tokens").and_then(|t| t.as_u64())
        {
            input_tokens = Some(tokens);
        }
    }

    if input_tokens.is_none() {
        println!(
            "Warning: Streaming response did not include usage with input_tokens for provider {}.\n\
            Total chunks received: {}\n\
            Last few chunks:\n{:#?}",
            variant_name,
            all_chunks.len(),
            all_chunks.iter().rev().take(3).collect::<Vec<_>>()
        );
    }

    input_tokens
}
