//! E2E tests for `include_raw_usage` parameter across all providers.
//!
//! Tests that raw provider-specific usage data is correctly returned when requested
//! for all providers that support simple inference.

use crate::common::get_gateway_endpoint;
use crate::providers::common::E2ETestProvider;
use crate::providers::helpers::get_modal_extra_headers;
use futures::StreamExt;
use reqwest::{Client, StatusCode};
use reqwest_sse_stream::into_sse_stream;
use serde_json::{Value, json};
use tensorzero_core::inference::types::extra_headers::UnfilteredInferenceExtraHeaders;
use uuid::Uuid;

/// Helper to assert raw_usage entry structure is valid
fn assert_raw_usage_entry(entry: &Value, provider: &E2ETestProvider) {
    assert!(
        entry.get("model_inference_id").is_some(),
        "raw_usage entry should have model_inference_id for provider {}",
        provider.variant_name
    );
    assert!(
        entry.get("provider_type").is_some(),
        "raw_usage entry should have provider_type for provider {}",
        provider.variant_name
    );
    assert!(
        entry.get("api_type").is_some(),
        "raw_usage entry should have api_type for provider {}",
        provider.variant_name
    );
    // data field should exist (can be null for providers that don't return detailed usage)
    assert!(
        entry.get("data").is_some(),
        "raw_usage entry should have data field for provider {}",
        provider.variant_name
    );
}

/// Test that include_raw_usage works correctly for non-streaming inference
pub async fn test_raw_usage_inference_with_provider_non_streaming(provider: E2ETestProvider) {
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
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [
                {
                    "role": "user",
                    "content": "What is the capital of Japan?"
                }
            ]
        },
        "stream": false,
        "include_raw_usage": true,
        "extra_headers": extra_headers.extra_headers,
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::OK,
        "Response should be successful for provider {}",
        provider.variant_name
    );

    let response_json: Value = response.json().await.unwrap();

    // Check usage exists (standard usage field)
    let usage = response_json.get("usage").unwrap_or_else(|| {
        panic!(
            "Response should have usage field for provider {}. Response: {response_json:#?}",
            provider.variant_name
        )
    });
    assert!(
        usage.get("input_tokens").is_some(),
        "usage should have input_tokens for provider {}",
        provider.variant_name
    );
    assert!(
        usage.get("output_tokens").is_some(),
        "usage should have output_tokens for provider {}",
        provider.variant_name
    );

    // Check raw_usage exists at response level (sibling to usage)
    let raw_usage = response_json.get("raw_usage").unwrap_or_else(|| {
        panic!(
            "Response should have raw_usage when include_raw_usage=true for provider {}. Response: {response_json:#?}",
            provider.variant_name
        )
    });

    assert!(
        raw_usage.is_array(),
        "raw_usage should be an array for provider {}",
        provider.variant_name
    );

    let raw_usage_array = raw_usage.as_array().unwrap();
    assert!(
        !raw_usage_array.is_empty(),
        "raw_usage should have at least one entry for provider {}",
        provider.variant_name
    );

    // Validate each entry structure
    for entry in raw_usage_array {
        assert_raw_usage_entry(entry, &provider);
    }
}

/// Test that include_raw_usage works correctly for streaming inference
pub async fn test_raw_usage_inference_with_provider_streaming(provider: E2ETestProvider) {
    // We use a serverless Sagemaker endpoint, which doesn't support streaming
    if provider.variant_name == "aws-sagemaker-tgi" {
        return;
    }

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
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [
                {
                    "role": "user",
                    "content": "What is the capital of France?"
                }
            ]
        },
        "stream": true,
        "include_raw_usage": true,
        "extra_headers": extra_headers.extra_headers,
    });

    let mut chunks = into_sse_stream(
        Client::new()
            .post(get_gateway_endpoint("/inference"))
            .json(&payload),
    )
    .await
    .unwrap_or_else(|e| {
        panic!(
            "Failed to create eventsource for streaming request for provider {}: {e}",
            provider.variant_name
        )
    });

    let mut found_raw_usage = false;
    let mut last_chunk_with_usage: Option<Value> = None;
    let mut all_chunks: Vec<Value> = Vec::new();

    while let Some(chunk) = chunks.next().await {
        let sse = chunk.unwrap_or_else(|e| {
            panic!(
                "Failed to receive chunk from stream for provider {}: {e}",
                provider.variant_name
            )
        });
        let Some(data) = sse.data else { continue };
        if data == "[DONE]" {
            break;
        }

        let chunk_json: Value = serde_json::from_str(&data).unwrap_or_else(|e| {
            panic!(
                "Failed to parse chunk as JSON for provider {}: {e}. Data: {}",
                provider.variant_name, data
            )
        });

        all_chunks.push(chunk_json.clone());

        // Check if this chunk has raw_usage (sibling to usage at chunk level)
        if chunk_json.get("raw_usage").is_some() {
            found_raw_usage = true;
            last_chunk_with_usage = Some(chunk_json.clone());
        }
    }

    assert!(
        found_raw_usage,
        "Streaming response should include raw_usage in at least one chunk when include_raw_usage=true for provider {}.\n\
        Total chunks received: {}\n\
        Last few chunks:\n{:#?}",
        provider.variant_name,
        all_chunks.len(),
        all_chunks.iter().rev().take(3).collect::<Vec<_>>()
    );

    let final_chunk = last_chunk_with_usage.unwrap_or_else(|| {
        panic!(
            "No chunk with raw_usage found for provider {} despite found_raw_usage being true",
            provider.variant_name
        )
    });
    let raw_usage = final_chunk.get("raw_usage").unwrap_or_else(|| {
        panic!(
            "raw_usage field missing from chunk for provider {}",
            provider.variant_name
        )
    });

    assert!(
        raw_usage.is_array(),
        "raw_usage should be an array for provider {}",
        provider.variant_name
    );

    let raw_usage_array = raw_usage.as_array().unwrap();
    assert!(
        !raw_usage_array.is_empty(),
        "Streaming should include at least one raw_usage entry for provider {}",
        provider.variant_name
    );

    // Validate each entry structure
    for entry in raw_usage_array {
        assert_raw_usage_entry(entry, &provider);
    }
}
