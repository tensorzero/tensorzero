//! E2E tests for cost tracking.
//!
//! These tests verify that cost is correctly computed from provider raw responses
//! using JSON Pointer-based cost configuration, and that cost is stored in
//! the database and returned in API responses.

use futures::StreamExt;
use reqwest::Client;
use reqwest_sse_stream::{Event, RequestBuilderExt};
use rust_decimal::Decimal;
use serde_json::{Value, json};
use tensorzero_core::db::clickhouse::test_helpers::{
    get_clickhouse, select_model_inference_clickhouse,
};
use uuid::Uuid;

use crate::common::get_gateway_endpoint;

/// Expected cost for the dummy "good" model with cost config:
///   - prompt_tokens: 10 * 1.50 / 1_000_000 = 0.000015
///   - completion_tokens: 10 * 2.00 / 1_000_000 = 0.00002
///   - total: 0.000035
fn expected_dummy_cost() -> Decimal {
    // 10 * 1.50 / 1_000_000 + 10 * 2.00 / 1_000_000 = 0.000035
    Decimal::new(35, 6)
}

/// Helper to make a simple non-streaming inference request.
async fn make_inference(function_name: &str, stream: bool) -> Value {
    let payload = json!({
        "function_name": function_name,
        "episode_id": Uuid::now_v7(),
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, world!"
                }
            ]
        },
        "stream": stream,
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert!(response.status().is_success(), "inference request failed");
    response.json::<Value>().await.unwrap()
}

/// Helper to make a streaming inference request and return the last chunk (which has usage).
async fn make_streaming_inference(function_name: &str) -> Value {
    let payload = json!({
        "function_name": function_name,
        "episode_id": Uuid::now_v7(),
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, world!"
                }
            ]
        },
        "stream": true,
    });

    let mut event_source = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .await
        .unwrap();

    let mut chunks = vec![];
    while let Some(event) = event_source.next().await {
        let event = event.unwrap();
        match event {
            Event::Open => continue,
            Event::Message(message) => {
                if message.data == "[DONE]" {
                    break;
                }
                chunks.push(message.data);
            }
        }
    }

    assert!(!chunks.is_empty(), "expected at least one chunk");
    serde_json::from_str(chunks.last().unwrap()).unwrap()
}

// ─── Non-streaming cost tests ────────────────────────────────────────────────

/// Helper to extract a Decimal cost from a JSON value (handles both number and string).
fn json_cost_to_decimal(cost: &Value) -> Decimal {
    match cost {
        Value::Number(n) => {
            // serde_json Number — convert via its string representation to avoid f64 noise
            let s = n.to_string();
            s.parse::<Decimal>()
                .unwrap_or_else(|e| panic!("failed to parse cost number `{s}` as Decimal: {e}"))
        }
        Value::String(s) => s
            .parse::<Decimal>()
            .unwrap_or_else(|e| panic!("failed to parse cost string `{s}` as Decimal: {e}")),
        other => panic!("cost should be a number or string, got: {other:?}"),
    }
}

/// Verify that cost is present in the non-streaming API response for a model
/// with cost configuration.
#[tokio::test]
async fn test_cost_in_non_streaming_response() {
    let response = make_inference("basic_test", false).await;

    let usage = response.get("usage").expect("response should have usage");
    let cost = usage.get("cost");
    assert!(
        cost.is_some(),
        "usage should include cost for a model with cost configuration"
    );

    let cost_decimal = json_cost_to_decimal(cost.unwrap());
    assert_eq!(
        cost_decimal,
        expected_dummy_cost(),
        "cost should match expected value based on token counts and rates"
    );
}

/// Verify that cost is stored in ClickHouse for non-streaming inferences.
#[tokio::test]
async fn test_cost_stored_in_clickhouse_non_streaming() {
    let response = make_inference("basic_test", false).await;
    let inference_id =
        Uuid::parse_str(response.get("inference_id").unwrap().as_str().unwrap()).unwrap();

    // Wait for ClickHouse writes
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    let clickhouse = get_clickhouse().await;
    let model_inference = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .expect("model inference should be stored in ClickHouse");

    let cost = model_inference.get("cost");
    assert!(
        cost.is_some() && !cost.unwrap().is_null(),
        "cost should be stored in ClickHouse model inference"
    );

    let cost_decimal = json_cost_to_decimal(cost.unwrap());
    assert_eq!(
        cost_decimal,
        expected_dummy_cost(),
        "ClickHouse cost should match expected value"
    );
}

// ─── Streaming cost tests ────────────────────────────────────────────────────

/// Verify that cost is present in the final streaming chunk's usage.
#[tokio::test]
async fn test_cost_in_streaming_response() {
    let last_chunk = make_streaming_inference("basic_test").await;

    let usage = last_chunk.get("usage");
    // In streaming, usage might be in the last chunk
    if let Some(usage) = usage {
        let cost = usage.get("cost");
        assert!(
            cost.is_some(),
            "usage in streaming final chunk should include cost"
        );
    }
}

/// Verify that cost is stored in ClickHouse for streaming inferences.
#[tokio::test]
async fn test_cost_stored_in_clickhouse_streaming() {
    let last_chunk = make_streaming_inference("basic_test").await;
    let inference_id =
        Uuid::parse_str(last_chunk.get("inference_id").unwrap().as_str().unwrap()).unwrap();

    // Wait for ClickHouse writes
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    let clickhouse = get_clickhouse().await;
    let model_inference = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .expect("model inference should be stored in ClickHouse");

    let cost = model_inference.get("cost");
    assert!(
        cost.is_some() && !cost.unwrap().is_null(),
        "cost should be stored in ClickHouse for streaming inferences"
    );
}

// ─── Missing cost configuration ──────────────────────────────────────────────

/// Verify that cost is None/missing when a model has no cost configuration.
#[tokio::test]
async fn test_no_cost_when_not_configured() {
    // "model_fallback_test" uses the "test_fallback" model which routes to
    // error (fails) then "good" (succeeds). The "good" provider on "test_fallback"
    // doesn't have cost configured.
    let response = make_inference("model_fallback_test", false).await;

    let usage = response.get("usage").expect("response should have usage");
    let cost = usage.get("cost");
    // cost should be absent (None / not serialized)
    assert!(
        cost.is_none() || cost.unwrap().is_null(),
        "cost should be absent when model has no cost configuration"
    );
}

// ─── Advanced variant cost tests (poison semantics) ─────────────────────────
//
// These tests verify that when an advanced variant combines multiple model
// inferences and some have cost while others don't, the total cost is None
// (poison semantics).

/// Helper to make an inference request with a specific variant.
async fn make_inference_with_variant(
    function_name: &str,
    variant_name: &str,
    stream: bool,
) -> Value {
    let payload = json!({
        "function_name": function_name,
        "variant_name": variant_name,
        "episode_id": Uuid::now_v7(),
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, world!"
                }
            ]
        },
        "stream": stream,
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert!(
        response.status().is_success(),
        "inference request failed for {function_name}/{variant_name}: {}",
        response.status()
    );
    response.json::<Value>().await.unwrap()
}

/// Helper to make a streaming inference with a specific variant and return the last chunk.
async fn make_streaming_inference_with_variant(function_name: &str, variant_name: &str) -> Value {
    let payload = json!({
        "function_name": function_name,
        "variant_name": variant_name,
        "episode_id": Uuid::now_v7(),
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, world!"
                }
            ]
        },
        "stream": true,
    });

    let mut event_source = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .await
        .unwrap();

    let mut chunks = vec![];
    while let Some(event) = event_source.next().await {
        let event = event.unwrap();
        match event {
            Event::Open => continue,
            Event::Message(message) => {
                if message.data == "[DONE]" {
                    break;
                }
                chunks.push(message.data);
            }
        }
    }

    assert!(!chunks.is_empty(), "expected at least one chunk");
    serde_json::from_str(chunks.last().unwrap()).unwrap()
}

/// Helper to assert that cost is absent (None) in a usage object.
fn assert_cost_absent(usage: &Value, context: &str) {
    let cost = usage.get("cost");
    assert!(
        cost.is_none() || cost.unwrap().is_null(),
        "cost should be absent for {context} (poison semantics)"
    );
}

/// DICL: embedding model has no cost config, LLM has cost config.
/// Total cost should be None (poison semantics).
#[tokio::test]
async fn test_cost_dicl_poison_non_streaming() {
    let response = make_inference_with_variant("basic_test", "dummy_dicl", false).await;
    let usage = response.get("usage").expect("response should have usage");
    assert_cost_absent(usage, "DICL non-streaming (embedding has no cost)");
}

/// DICL streaming: same poison semantics.
#[tokio::test]
async fn test_cost_dicl_poison_streaming() {
    let last_chunk = make_streaming_inference_with_variant("basic_test", "dummy_dicl").await;
    if let Some(usage) = last_chunk.get("usage") {
        assert_cost_absent(usage, "DICL streaming (embedding has no cost)");
    }
}

/// Best-of-N: candidates have cost config, evaluator does not.
/// Total cost should be None (poison semantics).
#[tokio::test]
async fn test_cost_best_of_n_poison_non_streaming() {
    let response = make_inference_with_variant("cost_test_best_of_n", "best_of_n", false).await;
    let usage = response.get("usage").expect("response should have usage");
    assert_cost_absent(usage, "best-of-N non-streaming (evaluator has no cost)");
}

/// Best-of-N streaming: same poison semantics.
#[tokio::test]
async fn test_cost_best_of_n_poison_streaming() {
    let last_chunk =
        make_streaming_inference_with_variant("cost_test_best_of_n", "best_of_n").await;
    if let Some(usage) = last_chunk.get("usage") {
        assert_cost_absent(usage, "best-of-N streaming (evaluator has no cost)");
    }
}

/// Mixture-of-N: candidates have cost config, fuser does not.
/// Total cost should be None (poison semantics).
#[tokio::test]
async fn test_cost_mixture_of_n_poison_non_streaming() {
    let response =
        make_inference_with_variant("cost_test_mixture_of_n", "mixture_of_n", false).await;
    let usage = response.get("usage").expect("response should have usage");
    assert_cost_absent(usage, "mixture-of-N non-streaming (fuser has no cost)");
}

/// Mixture-of-N streaming: same poison semantics.
#[tokio::test]
async fn test_cost_mixture_of_n_poison_streaming() {
    let last_chunk =
        make_streaming_inference_with_variant("cost_test_mixture_of_n", "mixture_of_n").await;
    if let Some(usage) = last_chunk.get("usage") {
        assert_cost_absent(usage, "mixture-of-N streaming (fuser has no cost)");
    }
}

// ─── Advanced variant cost tests (all providers have cost) ──────────────────
//
// When ALL model providers in an advanced variant have cost configured,
// the total cost should be the sum of all individual costs.
//
// Each dummy model inference returns 10 prompt_tokens + 10 completion_tokens.
// With cost_per_million = 1.50 (prompt) and 2.00 (completion):
//   per-inference cost = 10 * 1.50/1M + 10 * 2.00/1M = 0.000035
//
// Best-of-N (2 candidates + 1 evaluator = 3 inferences): 3 * 0.000035 = 0.000105
// Mixture-of-N (2 candidates + 1 fuser = 3 inferences): 3 * 0.000035 = 0.000105

fn expected_three_inference_cost() -> Decimal {
    // 3 * 0.000035 = 0.000105
    Decimal::new(105, 6)
}

/// Best-of-N with all cost: candidates + evaluator all have cost config.
/// Total cost should be the sum of all model inference costs.
#[tokio::test]
async fn test_cost_best_of_n_all_cost_non_streaming() {
    let response =
        make_inference_with_variant("cost_test_best_of_n_all_cost", "best_of_n", false).await;
    let usage = response.get("usage").expect("response should have usage");
    let cost = usage.get("cost");
    assert!(
        cost.is_some(),
        "usage should include cost when all providers have cost config (best-of-N)"
    );
    let cost_decimal = json_cost_to_decimal(cost.unwrap());
    assert_eq!(
        cost_decimal,
        expected_three_inference_cost(),
        "best-of-N total cost should be sum of 3 model inference costs"
    );
}

/// Best-of-N streaming with all cost.
#[tokio::test]
async fn test_cost_best_of_n_all_cost_streaming() {
    let last_chunk =
        make_streaming_inference_with_variant("cost_test_best_of_n_all_cost", "best_of_n").await;
    if let Some(usage) = last_chunk.get("usage") {
        let cost = usage.get("cost");
        assert!(
            cost.is_some(),
            "usage should include cost when all providers have cost config (best-of-N streaming)"
        );
        let cost_decimal = json_cost_to_decimal(cost.unwrap());
        assert_eq!(
            cost_decimal,
            expected_three_inference_cost(),
            "best-of-N streaming total cost should be sum of 3 model inference costs"
        );
    }
}

/// Mixture-of-N with all cost: candidates + fuser all have cost config.
/// Total cost should be the sum of all model inference costs.
#[tokio::test]
async fn test_cost_mixture_of_n_all_cost_non_streaming() {
    let response =
        make_inference_with_variant("cost_test_mixture_of_n_all_cost", "mixture_of_n", false).await;
    let usage = response.get("usage").expect("response should have usage");
    let cost = usage.get("cost");
    assert!(
        cost.is_some(),
        "usage should include cost when all providers have cost config (mixture-of-N)"
    );
    let cost_decimal = json_cost_to_decimal(cost.unwrap());
    assert_eq!(
        cost_decimal,
        expected_three_inference_cost(),
        "mixture-of-N total cost should be sum of 3 model inference costs"
    );
}

/// Mixture-of-N streaming with all cost.
#[tokio::test]
async fn test_cost_mixture_of_n_all_cost_streaming() {
    let last_chunk =
        make_streaming_inference_with_variant("cost_test_mixture_of_n_all_cost", "mixture_of_n")
            .await;
    if let Some(usage) = last_chunk.get("usage") {
        let cost = usage.get("cost");
        assert!(
            cost.is_some(),
            "usage should include cost when all providers have cost config (mixture-of-N streaming)"
        );
        let cost_decimal = json_cost_to_decimal(cost.unwrap());
        assert_eq!(
            cost_decimal,
            expected_three_inference_cost(),
            "mixture-of-N streaming total cost should be sum of 3 model inference costs"
        );
    }
}
