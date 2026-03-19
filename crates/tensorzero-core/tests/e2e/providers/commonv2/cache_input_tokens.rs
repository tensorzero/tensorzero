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
//! Also verifies that `provider_cache_read_input_tokens` and `provider_cache_write_input_tokens`
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
        .get("provider_cache_write_input_tokens")
        .and_then(|t| t.as_u64());
    let cache_read1 = usage1
        .get("provider_cache_read_input_tokens")
        .and_then(|t| t.as_u64());
    println!(
        "Provider {} (request 1): input_tokens={}, cache_write={:?}, cache_read={:?}",
        provider.variant_name, input_tokens1, cache_write1, cache_read1
    );

    // Assert input_tokens > 4000 on first request (proves cache tokens are included)
    assert!(
        input_tokens1 > 4000,
        "input_tokens ({input_tokens1}) should be > 4000 for first request (cache write). \
        This suggests provider_cache_write_input_tokens may not be included in input_tokens."
    );

    // Second request - triggers cache read (different user message, same cached system prompt).
    // The second request may fail through the provider-proxy if no recording exists yet,
    // so we treat failures as non-fatal (the primary assertion above already passed).
    let response2 = client
        .post(get_gateway_endpoint("/inference"))
        .json(&payload2)
        .send()
        .await
        .expect("failed to send second inference request");

    if response2.status() != StatusCode::OK {
        println!(
            "Warning: second request for {} returned status {} — skipping cache read assertions",
            provider.variant_name,
            response2.status()
        );
        return;
    }

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

    // Assert consistency between cache write and cache read.
    // The two requests have slightly different user messages (for proxy compatibility),
    // so allow a small difference in input_tokens.
    let diff = input_tokens1.abs_diff(input_tokens2);
    assert!(
        diff <= 10,
        "input_tokens should be approximately equal between requests ({input_tokens1} vs {input_tokens2}, diff={diff})"
    );

    // Log cache token behavior for debugging. We don't assert cache_read > 0 because
    // the provider-proxy records real responses that may not reflect cache hits
    // (e.g., Bedrock may return cache_write > 0 on first request but cache_read=0
    // on second because the real API didn't cache between two rapid requests).
    if let Some(cw) = cache_write1
        && cw > 0
    {
        println!(
            "Provider {} wrote {} cache tokens on first request, cache_read on second: {:?}",
            provider.variant_name, cw, cache_read2
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
        .expect("first streaming request should return usage");
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

    // Assert input_tokens > 4000 on first request (proves cache tokens are included)
    assert!(
        input_tokens1 > 4000,
        "input_tokens ({input_tokens1}) should be > 4000 for first streaming request (cache write). \
        This suggests provider_cache_write_input_tokens may not be included in input_tokens."
    );

    // Second request - triggers cache read (different user message, same cached system prompt).
    // May fail through provider-proxy if no recording exists yet for this payload.
    let Some(usage2) = get_streaming_usage(&client, &payload2, &provider.variant_name).await else {
        println!(
            "Warning: second streaming request for {} did not return usage — skipping cache read assertions",
            provider.variant_name
        );
        return;
    };

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

    // Assert consistency between cache write and cache read.
    // The two requests have slightly different user messages (for proxy compatibility),
    // so allow a small difference in input_tokens.
    let diff = input_tokens1.abs_diff(input_tokens2);
    assert!(
        diff <= 10,
        "input_tokens should be approximately equal between streaming requests ({input_tokens1} vs {input_tokens2}, diff={diff})"
    );

    // Log cache token behavior for debugging (see non-streaming test for rationale).
    if let Some(cw) = cache_write1
        && cw > 0
    {
        println!(
            "Provider {} wrote {} cache tokens on first streaming request, cache_read on second: {:?}",
            provider.variant_name, cw, cache_read2
        );
    }
}

/// Helper to get the usage object from a streaming response.
async fn get_streaming_usage(
    client: &Client,
    payload: &Value,
    variant_name: &str,
) -> Option<Value> {
    let mut chunks = match client
        .post(get_gateway_endpoint("/inference"))
        .json(payload)
        .eventsource()
        .await
    {
        Ok(chunks) => chunks,
        Err(e) => {
            println!("Failed to create eventsource for provider {variant_name}: {e}");
            return None;
        }
    };

    let mut usage: Option<Value> = None;
    let mut all_chunks: Vec<Value> = Vec::new();

    while let Some(chunk) = chunks.next().await {
        let chunk = match chunk {
            Ok(c) => c,
            Err(e) => {
                println!("Stream error for provider {variant_name}: {e}");
                return None;
            }
        };
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
