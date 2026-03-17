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
//! Also verifies that `cache_read_input_tokens` and `cache_write_input_tokens`
//! fields are present in the usage response for providers that support them.
//!
//! This addresses issue #5688 where some providers (e.g., AWS Bedrock) report
//! inputTokens separately from cacheReadInputTokens/cacheWriteInputTokens.

use crate::common::get_gateway_endpoint;
use crate::providers::common::E2ETestProvider;
use crate::providers::helpers::get_modal_extra_headers;
use futures::StreamExt;
use reqwest::{Client, StatusCode};
use reqwest_sse_stream::{Event, RequestBuilderExt};
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

    let usage1 = response1_json
        .get("usage")
        .expect("first response should have usage");
    let input_tokens1 = usage1
        .get("input_tokens")
        .and_then(|t| t.as_u64())
        .expect("first response should have input_tokens");

    // Check cache token fields on first request (cache write)
    let cache_write1 = usage1
        .get("cache_write_input_tokens")
        .and_then(|t| t.as_u64());
    let cache_read1 = usage1
        .get("cache_read_input_tokens")
        .and_then(|t| t.as_u64());
    println!(
        "Provider {} (request 1): input_tokens={}, cache_write={:?}, cache_read={:?}",
        provider.variant_name, input_tokens1, cache_write1, cache_read1
    );

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

    let usage2 = response2_json
        .get("usage")
        .expect("second response should have usage");
    let input_tokens2 = usage2
        .get("input_tokens")
        .and_then(|t| t.as_u64())
        .expect("second response should have input_tokens");

    // Check cache token fields on second request (cache read)
    let cache_write2 = usage2
        .get("cache_write_input_tokens")
        .and_then(|t| t.as_u64());
    let cache_read2 = usage2
        .get("cache_read_input_tokens")
        .and_then(|t| t.as_u64());
    println!(
        "Provider {} (request 2): input_tokens={}, cache_write={:?}, cache_read={:?}",
        provider.variant_name, input_tokens2, cache_write2, cache_read2
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

    // For providers that support cache tokens, verify the fields are populated.
    // Only assert cache_read > 0 on the second request when the first request
    // actually wrote to cache (cache_write > 0). Provider-proxy replays may
    // return Some(0) for both requests since no real caching occurs.
    if let Some(cw) = cache_write1
        && cw > 0
        && let Some(cr) = cache_read2
    {
        assert!(
            cr > 0,
            "Provider {} wrote {} cache tokens on first request but cache_read_input_tokens={} on second request, expected > 0",
            provider.variant_name,
            cw,
            cr
        );
    }
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
    let usage1 = get_streaming_usage(&client, &payload, &provider.variant_name)
        .await
        .expect("first streaming request should return usage");
    let input_tokens1 = usage1
        .get("input_tokens")
        .and_then(|t| t.as_u64())
        .expect("first streaming response should have input_tokens");
    let cache_write1 = usage1
        .get("cache_write_input_tokens")
        .and_then(|t| t.as_u64());
    let cache_read1 = usage1
        .get("cache_read_input_tokens")
        .and_then(|t| t.as_u64());
    println!(
        "Provider {} (streaming request 1): input_tokens={}, cache_write={:?}, cache_read={:?}",
        provider.variant_name, input_tokens1, cache_write1, cache_read1
    );

    // Second request - triggers cache read
    let usage2 = get_streaming_usage(&client, &payload, &provider.variant_name)
        .await
        .expect("second streaming request should return usage");
    let input_tokens2 = usage2
        .get("input_tokens")
        .and_then(|t| t.as_u64())
        .expect("second streaming response should have input_tokens");
    let cache_write2 = usage2
        .get("cache_write_input_tokens")
        .and_then(|t| t.as_u64());
    let cache_read2 = usage2
        .get("cache_read_input_tokens")
        .and_then(|t| t.as_u64());
    println!(
        "Provider {} (streaming request 2): input_tokens={}, cache_write={:?}, cache_read={:?}",
        provider.variant_name, input_tokens2, cache_write2, cache_read2
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

    // For providers that support cache tokens, verify the fields are populated.
    // Only assert cache_read > 0 when the first request actually wrote to cache.
    if let Some(cw) = cache_write1
        && cw > 0
        && let Some(cr) = cache_read2
    {
        assert!(
            cr > 0,
            "Provider {} wrote {} cache tokens on first streaming request but cache_read_input_tokens={} on second, expected > 0",
            provider.variant_name,
            cw,
            cr
        );
    }
}

/// Helper to get the usage object from a streaming response.
async fn get_streaming_usage(
    client: &Client,
    payload: &Value,
    variant_name: &str,
) -> Option<Value> {
    let mut chunks = client
        .post(get_gateway_endpoint("/inference"))
        .json(payload)
        .eventsource()
        .await
        .unwrap_or_else(|e| {
            panic!(
                "Failed to create eventsource for streaming request for provider {variant_name}: {e}",
            )
        });

    let mut usage: Option<Value> = None;
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
        if let Some(u) = chunk_json.get("usage")
            && u.get("input_tokens").and_then(|t| t.as_u64()).is_some()
        {
            usage = Some(u.clone());
        }
    }

    if usage.is_none() {
        println!(
            "Warning: Streaming response did not include usage with input_tokens for provider {}.\n\
            Total chunks received: {}\n\
            Last few chunks:\n{:#?}",
            variant_name,
            all_chunks.len(),
            all_chunks.iter().rev().take(3).collect::<Vec<_>>()
        );
    }

    usage
}
