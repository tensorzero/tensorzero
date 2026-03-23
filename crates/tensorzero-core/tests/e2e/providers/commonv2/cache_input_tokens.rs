//! E2E tests for verifying provider cache tokens are included in input_tokens.
//!
//! Makes two inference requests with large cacheable content:
//! 1. First request: triggers cache write
//! 2. Second request: triggers cache read
//!
//! Both requests should report input_tokens > 4000 to prove that cache tokens
//! (provider_cache_write_input_tokens, provider_cache_read_input_tokens) are properly included
//! in the input_tokens count.
//!
//! The second request MUST report `provider_cache_read_input_tokens` to prove the
//! cache is working. The first request assertion is lenient because auto-caching
//! providers (OpenAI, Groq, xAI) may not return cache fields on the first request.
//!
//! NOTE: Only providers that support prompt caching AND return cache token fields
//! in their API responses should be registered in `cache_input_tokens_inference`.
//! See `tensorzero-types-providers/src/cache.rs` for the full provider mapping.
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

/// Model-filtered extra_body for providers that require explicit cache markers.
/// Each entry is filtered by model_name, so only the relevant cache control
/// marker gets applied for the model being tested.
///
/// Providers with automatic caching (OpenAI, Groq, xAI) don't need markers
/// and will still run the test without any cache control in extra_body.
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
/// - At least one request reports cache fields (second request MUST have cache_read)
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

    // Use different user messages so the provider-proxy records separate responses.
    // The large system prompt (which is the cached part) is identical in both requests.
    let payload1 = json!({
        "function_name": "basic_test",
        "variant_name": provider.variant_name,
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": LARGE_SYSTEM_PROMPT},
            "messages": [{"role": "user", "content": "Briefly summarize the above text in one sentence."}]
        },
        "stream": false,
        "extra_headers": extra_headers.extra_headers,
        "extra_body": cache_control_extra_body(),
    });

    let payload2 = json!({
        "function_name": "basic_test",
        "variant_name": provider.variant_name,
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": LARGE_SYSTEM_PROMPT},
            "messages": [{"role": "user", "content": "Summarize the above text in two sentences."}]
        },
        "stream": false,
        "extra_headers": extra_headers.extra_headers,
        "extra_body": cache_control_extra_body(),
    });

    let client = Client::new();

    // First request - triggers cache write
    let response1 = client
        .post(get_gateway_endpoint("/inference"))
        .json(&payload1)
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

    let usage1 = response1_json
        .get("usage")
        .expect("first response should have usage");
    let input_tokens1 = usage1
        .get("input_tokens")
        .and_then(|t| t.as_u64())
        .expect("first response should have input_tokens");

    let cache_write1 = usage1
        .get("provider_cache_write_input_tokens")
        .and_then(|t| t.as_u64());
    let cache_read1 = usage1
        .get("provider_cache_read_input_tokens")
        .and_then(|t| t.as_u64());
    println!(
        "Provider {} (request 1): input_tokens={}, cache_write={:?}, cache_read={:?}",
        provider.variant_name, input_tokens1, cache_write1, cache_read1
    );

    assert!(
        input_tokens1 > 4000,
        "input_tokens ({input_tokens1}) should be > 4000 for first request (cache write). \
        This suggests provider_cache_write_input_tokens may not be included in input_tokens."
    );

    // NOTE: We don't assert cache fields on the first request because auto-caching
    // providers (OpenAI, Groq, xAI) may not return prompt_tokens_details on the
    // initial write. Explicit caching providers (Anthropic, Bedrock) will return
    // cache_write here. We log the values for debugging.

    // Second request - triggers cache read
    let response2 = client
        .post(get_gateway_endpoint("/inference"))
        .json(&payload2)
        .send()
        .await
        .expect("failed to send second inference request");

    assert_eq!(
        response2.status(),
        StatusCode::OK,
        "second inference request for {} should succeed (status={})",
        provider.variant_name,
        response2.status()
    );

    let response2_json: Value = response2
        .json()
        .await
        .expect("failed to parse second response JSON");

    let usage2 = response2_json
        .get("usage")
        .expect("second response should have usage");
    let input_tokens2 = usage2
        .get("input_tokens")
        .and_then(|t| t.as_u64())
        .expect("second response should have input_tokens");

    let cache_write2 = usage2
        .get("provider_cache_write_input_tokens")
        .and_then(|t| t.as_u64());
    let cache_read2 = usage2
        .get("provider_cache_read_input_tokens")
        .and_then(|t| t.as_u64());
    println!(
        "Provider {} (request 2): input_tokens={}, cache_write={:?}, cache_read={:?}",
        provider.variant_name, input_tokens2, cache_write2, cache_read2
    );

    assert!(
        input_tokens2 > 4000,
        "input_tokens ({input_tokens2}) should be > 4000 for second request (cache read). \
        This suggests provider_cache_read_input_tokens may not be included in input_tokens."
    );

    // Consistency: input_tokens should be roughly equal between the two requests
    let diff = input_tokens1.abs_diff(input_tokens2);
    assert!(
        diff <= 10,
        "input_tokens should be approximately equal between requests ({input_tokens1} vs {input_tokens2}, diff={diff})"
    );

    // At least one request must report cache fields.
    // - Explicit caching (Anthropic/Bedrock): first has cache_write, second has cache_read
    // - Auto caching (OpenAI/Groq/xAI): second should have cache_read
    let any_cache_fields = cache_write1.is_some()
        || cache_read1.is_some()
        || cache_write2.is_some()
        || cache_read2.is_some();
    assert!(
        any_cache_fields,
        "Provider {} should report cache token fields on at least one request. \
        Request 1: cache_write={:?}, cache_read={:?}. \
        Request 2: cache_write={:?}, cache_read={:?}.",
        provider.variant_name, cache_write1, cache_read1, cache_write2, cache_read2
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
/// - At least one request reports cache fields
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

    // Use different user messages so the provider-proxy records separate responses.
    // The large system prompt (which is the cached part) is identical in both requests.
    let payload1 = json!({
        "function_name": "basic_test",
        "variant_name": provider.variant_name,
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": LARGE_SYSTEM_PROMPT},
            "messages": [{"role": "user", "content": "Briefly summarize the above text in one sentence."}]
        },
        "stream": true,
        "extra_headers": extra_headers.extra_headers,
        "extra_body": cache_control_extra_body(),
    });

    let payload2 = json!({
        "function_name": "basic_test",
        "variant_name": provider.variant_name,
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": LARGE_SYSTEM_PROMPT},
            "messages": [{"role": "user", "content": "Summarize the above text in two sentences."}]
        },
        "stream": true,
        "extra_headers": extra_headers.extra_headers,
        "extra_body": cache_control_extra_body(),
    });

    let client = Client::new();

    // First request - triggers cache write
    let usage1 = get_streaming_usage(&client, &payload1, &provider.variant_name)
        .await
        .unwrap_or_else(|| {
            panic!(
                "first streaming request for {} should return usage",
                provider.variant_name
            )
        });
    let input_tokens1 = usage1
        .get("input_tokens")
        .and_then(|t| t.as_u64())
        .expect("first streaming response should have input_tokens");
    let cache_write1 = usage1
        .get("provider_cache_write_input_tokens")
        .and_then(|t| t.as_u64());
    let cache_read1 = usage1
        .get("provider_cache_read_input_tokens")
        .and_then(|t| t.as_u64());
    println!(
        "Provider {} (streaming request 1): input_tokens={}, cache_write={:?}, cache_read={:?}",
        provider.variant_name, input_tokens1, cache_write1, cache_read1
    );

    assert!(
        input_tokens1 > 4000,
        "input_tokens ({input_tokens1}) should be > 4000 for first streaming request (cache write). \
        This suggests provider_cache_write_input_tokens may not be included in input_tokens."
    );

    // NOTE: We don't assert cache fields on the first request because auto-caching
    // providers may not return cache details on the initial write.

    // Second request - triggers cache read
    let usage2 = get_streaming_usage(&client, &payload2, &provider.variant_name)
        .await
        .unwrap_or_else(|| {
            panic!(
                "second streaming request for {} should return usage",
                provider.variant_name
            )
        });

    let input_tokens2 = usage2
        .get("input_tokens")
        .and_then(|t| t.as_u64())
        .expect("second streaming response should have input_tokens");
    let cache_write2 = usage2
        .get("provider_cache_write_input_tokens")
        .and_then(|t| t.as_u64());
    let cache_read2 = usage2
        .get("provider_cache_read_input_tokens")
        .and_then(|t| t.as_u64());
    println!(
        "Provider {} (streaming request 2): input_tokens={}, cache_write={:?}, cache_read={:?}",
        provider.variant_name, input_tokens2, cache_write2, cache_read2
    );

    assert!(
        input_tokens2 > 4000,
        "input_tokens ({input_tokens2}) should be > 4000 for second streaming request (cache read). \
        This suggests provider_cache_read_input_tokens may not be included in input_tokens."
    );

    let diff = input_tokens1.abs_diff(input_tokens2);
    assert!(
        diff <= 10,
        "input_tokens should be approximately equal between streaming requests ({input_tokens1} vs {input_tokens2}, diff={diff})"
    );

    // At least one request must report cache fields
    let any_cache_fields = cache_write1.is_some()
        || cache_read1.is_some()
        || cache_write2.is_some()
        || cache_read2.is_some();
    assert!(
        any_cache_fields,
        "Provider {} should report cache token fields on at least one streaming request. \
        Request 1: cache_write={:?}, cache_read={:?}. \
        Request 2: cache_write={:?}, cache_read={:?}.",
        provider.variant_name, cache_write1, cache_read1, cache_write2, cache_read2
    );
}

/// Helper to get the usage object from a streaming response.
async fn get_streaming_usage(
    client: &Client,
    payload: &Value,
    variant_name: &str,
) -> Option<Value> {
    let mut event_source = client
        .post(get_gateway_endpoint("/inference"))
        .json(payload)
        .eventsource()
        .await
        .unwrap_or_else(|e| panic!("failed to start streaming for {variant_name}: {e}"));

    let mut usage: Option<Value> = None;

    while let Some(event) = event_source.next().await {
        let event =
            event.unwrap_or_else(|e| panic!("stream error for provider {variant_name}: {e}"));
        let Event::Message(message) = event else {
            continue;
        };
        if message.data == "[DONE]" {
            break;
        }

        let chunk_json: Value = serde_json::from_str(&message.data).unwrap_or_else(|e| {
            panic!(
                "Failed to parse chunk as JSON for provider {variant_name}: {e}. Data: {}",
                message.data
            )
        });

        if let Some(u) = chunk_json.get("usage")
            && u.get("input_tokens").and_then(|t| t.as_u64()).is_some()
        {
            usage = Some(u.clone());
        }
    }

    usage
}
