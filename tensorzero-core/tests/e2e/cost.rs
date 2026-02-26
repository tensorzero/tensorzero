//! E2E tests for cost reporting in inference API endpoints.
//!
//! These tests verify that cost is correctly computed and returned in:
//! - TensorZero inference endpoint (non-streaming and streaming)
//! - OpenAI-compatible chat completions endpoint (non-streaming and streaming)
//! - OpenAI-compatible embeddings endpoint
//! - Various variant types (chat_completion, json, tool_use, best_of_n, mixture_of_n)
//! - Cached responses (cost should be 0)
//!
//! The dummy provider cost config uses:
//! - `/usage/prompt_tokens` at $3.0 per million
//! - `/usage/completion_tokens` at $15.0 per million
//!
//! For non-streaming: prompt_tokens=10, completion_tokens=10
//! -> cost = 10 * 3.0/1M + 10 * 15.0/1M = 0.00018
//!
//! For streaming with the "good" model: prompt_tokens=10, completion_tokens=16
//! -> cost = 10 * 3.0/1M + 16 * 15.0/1M = 0.00027

use crate::common::get_gateway_endpoint;
use futures::StreamExt;
use reqwest::Client;
use reqwest_sse_stream::{Event, RequestBuilderExt};
use serde_json::{Value, json};
use uuid::Uuid;

// ===== TensorZero Inference Endpoint Tests =====

#[tokio::test]
async fn test_cost_inference_non_streaming() {
    let payload = json!({
        "function_name": "basic_test",
        "episode_id": Uuid::now_v7(),
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [{"role": "user", "content": "Hello"}]
        },
        "stream": false,
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let response_json: Value = response.json().await.unwrap();

    let usage = response_json.get("usage").unwrap().as_object().unwrap();
    let cost = usage.get("cost").unwrap().as_f64().unwrap();
    assert!(
        (cost - 0.00018).abs() < 1e-10,
        "Expected cost ~0.00018 (10*$3/M + 10*$15/M), got {cost}"
    );
}

#[tokio::test]
async fn test_cost_inference_streaming() {
    let payload = json!({
        "function_name": "basic_test",
        "episode_id": Uuid::now_v7(),
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [{"role": "user", "content": "Hello"}]
        },
        "stream": true,
    });

    let mut event_source = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .await
        .unwrap();

    let mut last_chunk = None;
    while let Some(event) = event_source.next().await {
        let event = event.unwrap();
        match event {
            Event::Open => continue,
            Event::Message(message) => {
                if message.data == "[DONE]" {
                    break;
                }
                last_chunk = Some(message.data);
            }
        }
    }

    let last_chunk: Value = serde_json::from_str(&last_chunk.unwrap()).unwrap();
    let usage = last_chunk.get("usage").unwrap().as_object().unwrap();
    let cost = usage
        .get("cost")
        .expect("Expected `cost` key in streaming usage")
        .as_f64()
        .unwrap();
    // "good" model streaming: prompt_tokens=10, completion_tokens=16
    assert!(
        (cost - 0.00027).abs() < 1e-10,
        "Expected cost ~0.00027 (10*$3/M + 16*$15/M), got {cost}"
    );
}

// ===== OpenAI-Compatible Chat Completions Tests =====

#[tokio::test]
async fn test_cost_openai_compatible_non_streaming() {
    let payload = json!({
        "model": "tensorzero::function_name::basic_test",
        "messages": [
            {"role": "system", "content": [{"type": "text", "tensorzero::arguments": {"assistant_name": "AskJeeves"}}]},
            {"role": "user", "content": "Hello"}
        ],
        "stream": false,
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/openai/v1/chat/completions"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let response_json: Value = response.json().await.unwrap();

    let usage = response_json.get("usage").unwrap().as_object().unwrap();
    let cost = usage.get("tensorzero_cost").unwrap().as_f64().unwrap();
    assert!(
        (cost - 0.00018).abs() < 1e-10,
        "Expected cost ~0.00018 (10*$3/M + 10*$15/M), got {cost}"
    );
}

#[tokio::test]
async fn test_cost_openai_compatible_streaming() {
    let payload = json!({
        "model": "tensorzero::function_name::basic_test",
        "messages": [
            {"role": "system", "content": [{"type": "text", "tensorzero::arguments": {"assistant_name": "AskJeeves"}}]},
            {"role": "user", "content": "Hello"}
        ],
        "stream": true,
        "stream_options": {"include_usage": true},
    });

    let mut event_source = Client::new()
        .post(get_gateway_endpoint("/openai/v1/chat/completions"))
        .json(&payload)
        .eventsource()
        .await
        .unwrap();

    let mut last_chunk = None;
    while let Some(event) = event_source.next().await {
        let event = event.unwrap();
        match event {
            Event::Open => continue,
            Event::Message(message) => {
                if message.data == "[DONE]" {
                    break;
                }
                last_chunk = Some(message.data);
            }
        }
    }

    let last_chunk: Value = serde_json::from_str(&last_chunk.unwrap()).unwrap();
    let usage = last_chunk
        .get("usage")
        .expect("Expected `usage` in final streaming chunk")
        .as_object()
        .expect("Expected `usage` to be an object");
    let cost = usage
        .get("tensorzero_cost")
        .expect("Expected `tensorzero_cost` key in streaming usage")
        .as_f64()
        .unwrap();
    // "good" model streaming: prompt_tokens=10, completion_tokens=16
    assert!(
        (cost - 0.00027).abs() < 1e-10,
        "Expected cost ~0.00027 (10*$3/M + 16*$15/M), got {cost}"
    );
}

// ===== OpenAI-Compatible Embeddings Test =====

#[tokio::test]
async fn test_cost_openai_compatible_embeddings() {
    let payload = json!({
        "model": "tensorzero::embedding_model_name::dummy-embedding-model",
        "input": "Hello, world!",
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/openai/v1/embeddings"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let response_json: Value = response.json().await.unwrap();

    let usage = response_json.get("usage").unwrap().as_object().unwrap();
    let cost = usage.get("tensorzero_cost").unwrap().as_f64().unwrap();
    // prompt_tokens=10 * $3/M = 0.00003
    assert!(
        (cost - 0.00003).abs() < 1e-10,
        "Expected cost ~0.00003 (10*$3/M), got {cost}"
    );
}

// ===== Variant Type Tests (Non-Streaming) =====

#[tokio::test]
async fn test_cost_json_variant_non_streaming() {
    let payload = json!({
        "function_name": "json_success",
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [{"role": "user", "content": [{"type": "template", "name": "user", "arguments": {"country": "Japan"}}]}]
        },
        "stream": false,
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let response_json: Value = response.json().await.unwrap();

    let usage = response_json.get("usage").unwrap().as_object().unwrap();
    let cost = usage.get("cost").unwrap().as_f64().unwrap();
    assert!(
        (cost - 0.00018).abs() < 1e-10,
        "Expected cost ~0.00018 (10*$3/M + 10*$15/M), got {cost}"
    );
}

#[tokio::test]
async fn test_cost_json_variant_streaming() {
    let payload = json!({
        "function_name": "json_success",
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [{"role": "user", "content": [{"type": "template", "name": "user", "arguments": {"country": "Japan"}}]}]
        },
        "stream": true,
    });

    let mut event_source = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .await
        .unwrap();

    let mut last_chunk = None;
    while let Some(event) = event_source.next().await {
        let event = event.unwrap();
        match event {
            Event::Open => continue,
            Event::Message(message) => {
                if message.data == "[DONE]" {
                    break;
                }
                last_chunk = Some(message.data);
            }
        }
    }

    let last_chunk: Value = serde_json::from_str(&last_chunk.unwrap()).unwrap();
    let usage = last_chunk.get("usage").unwrap().as_object().unwrap();
    let cost = usage
        .get("cost")
        .expect("Expected `cost` key in streaming JSON variant usage")
        .as_f64()
        .unwrap();
    // "json" model streaming: prompt_tokens=10, completion_tokens=5
    assert!(
        (cost - 0.000105).abs() < 1e-10,
        "Expected cost ~0.000105 (10*$3/M + 5*$15/M), got {cost}"
    );
}

#[tokio::test]
async fn test_cost_tool_use_variant_non_streaming() {
    let payload = json!({
        "function_name": "weather_helper",
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [{"role": "user", "content": "What's the weather?"}]
        },
        "stream": false,
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let response_json: Value = response.json().await.unwrap();

    let usage = response_json.get("usage").unwrap().as_object().unwrap();
    let cost = usage.get("cost").unwrap().as_f64().unwrap();
    assert!(
        (cost - 0.00018).abs() < 1e-10,
        "Expected cost ~0.00018 (10*$3/M + 10*$15/M), got {cost}"
    );
}

#[tokio::test]
async fn test_cost_tool_use_variant_streaming() {
    let payload = json!({
        "function_name": "weather_helper",
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [{"role": "user", "content": "What's the weather?"}]
        },
        "stream": true,
    });

    let mut event_source = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .await
        .unwrap();

    let mut last_chunk = None;
    while let Some(event) = event_source.next().await {
        let event = event.unwrap();
        match event {
            Event::Open => continue,
            Event::Message(message) => {
                if message.data == "[DONE]" {
                    break;
                }
                last_chunk = Some(message.data);
            }
        }
    }

    let last_chunk: Value = serde_json::from_str(&last_chunk.unwrap()).unwrap();
    let usage = last_chunk.get("usage").unwrap().as_object().unwrap();
    let cost = usage
        .get("cost")
        .expect("Expected `cost` key in streaming tool use variant usage")
        .as_f64()
        .unwrap();
    // "tool" model streaming: prompt_tokens=10, completion_tokens=5
    assert!(
        (cost - 0.000105).abs() < 1e-10,
        "Expected cost ~0.000105 (10*$3/M + 5*$15/M), got {cost}"
    );
}

// ===== Missing Cost Config Tests =====
// When a model has no cost config, cost should be null.

#[tokio::test]
async fn test_cost_no_cost_config_non_streaming() {
    let payload = json!({
        "function_name": "basic_test",
        "variant_name": "no_cost_config",
        "episode_id": Uuid::now_v7(),
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [{"role": "user", "content": "Hello"}]
        },
        "stream": false,
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let response_json: Value = response.json().await.unwrap();

    let usage = response_json.get("usage").unwrap().as_object().unwrap();
    assert!(
        usage.get("cost").unwrap().is_null(),
        "Expected `cost` to be null when model has no cost config"
    );
}

#[tokio::test]
async fn test_cost_no_cost_config_streaming() {
    let payload = json!({
        "function_name": "basic_test",
        "variant_name": "no_cost_config",
        "episode_id": Uuid::now_v7(),
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [{"role": "user", "content": "Hello"}]
        },
        "stream": true,
    });

    let mut event_source = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .await
        .unwrap();

    let mut last_chunk = None;
    while let Some(event) = event_source.next().await {
        let event = event.unwrap();
        match event {
            Event::Open => continue,
            Event::Message(message) => {
                if message.data == "[DONE]" {
                    break;
                }
                last_chunk = Some(message.data);
            }
        }
    }

    let last_chunk: Value = serde_json::from_str(&last_chunk.unwrap()).unwrap();
    let usage = last_chunk.get("usage").unwrap().as_object().unwrap();
    assert!(
        usage.get("cost").unwrap().is_null(),
        "Expected `cost` to be null when model has no cost config (streaming)"
    );
}

// ===== Streaming Error Mid-Stream + Cost =====

#[tokio::test]
async fn test_cost_streaming_error_mid_stream() {
    let payload = json!({
        "function_name": "basic_test",
        "variant_name": "err_in_stream",
        "episode_id": Uuid::now_v7(),
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [{"role": "user", "content": "Hello"}]
        },
        "stream": true,
    });

    let mut event_source = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .await
        .unwrap();

    let mut last_usage_chunk = None;
    while let Some(event) = event_source.next().await {
        let event = event.unwrap();
        match event {
            Event::Open => continue,
            Event::Message(message) => {
                if message.data == "[DONE]" {
                    break;
                }
                let obj: Value = serde_json::from_str(&message.data).unwrap();
                // Skip error events
                if obj.get("error").is_some() {
                    continue;
                }
                // Keep the last chunk that has usage
                if obj.get("usage").is_some() {
                    last_usage_chunk = Some(obj);
                }
            }
        }
    }

    let last_chunk = last_usage_chunk.expect("Expected a chunk with usage data");
    let usage = last_chunk.get("usage").unwrap().as_object().unwrap();
    let cost = usage
        .get("cost")
        .expect("Expected `cost` key in streaming error mid-stream usage")
        .as_f64()
        .unwrap();
    // err_in_stream model: prompt_tokens=10, completion_tokens=16
    assert!(
        (cost - 0.00027).abs() < 1e-10,
        "Expected cost ~0.00027 (10*$3/M + 16*$15/M) despite mid-stream error, got {cost}"
    );
}

// ===== Best-of-N and Mixture-of-N Tests =====
// These tests use functions with some inline dummy models that don't have cost config,
// so cost should be null due to None propagation.

#[tokio::test]
async fn test_cost_best_of_n_non_streaming() {
    let payload = json!({
        "function_name": "best_of_n_json_repeated",
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [{"role": "user", "content": "Hello"}]
        },
        "stream": false,
        "variant_name": "best_of_n_variant",
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let response_json: Value = response.json().await.unwrap();

    let usage = response_json.get("usage").unwrap().as_object().unwrap();
    assert!(
        usage.get("cost").unwrap().is_null(),
        "Expected `cost` to be null in best-of-N (inline model has no cost config)"
    );
}

#[tokio::test]
async fn test_cost_best_of_n_streaming() {
    let payload = json!({
        "function_name": "best_of_n_json_repeated",
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [{"role": "user", "content": "Hello"}]
        },
        "stream": true,
        "variant_name": "best_of_n_variant",
    });

    let mut event_source = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .await
        .unwrap();

    let mut last_chunk = None;
    while let Some(event) = event_source.next().await {
        let event = event.unwrap();
        match event {
            Event::Open => continue,
            Event::Message(message) => {
                if message.data == "[DONE]" {
                    break;
                }
                last_chunk = Some(message.data);
            }
        }
    }

    let last_chunk: Value = serde_json::from_str(&last_chunk.unwrap()).unwrap();
    let usage = last_chunk.get("usage").unwrap().as_object().unwrap();
    assert!(
        usage.get("cost").unwrap().is_null(),
        "Expected `cost` to be null in streaming best-of-N (inline model has no cost config)"
    );
}

#[tokio::test]
async fn test_cost_mixture_of_n_non_streaming() {
    let payload = json!({
        "function_name": "mixture_of_n_json_repeated",
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [{"role": "user", "content": "Hello"}]
        },
        "stream": false,
        "variant_name": "mixture_of_n_variant",
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let response_json: Value = response.json().await.unwrap();

    let usage = response_json.get("usage").unwrap().as_object().unwrap();
    assert!(
        usage.get("cost").unwrap().is_null(),
        "Expected `cost` to be null in mixture-of-N (inline model has no cost config)"
    );
}

#[tokio::test]
async fn test_cost_mixture_of_n_streaming() {
    let payload = json!({
        "function_name": "mixture_of_n_json_repeated",
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [{"role": "user", "content": "Hello"}]
        },
        "stream": true,
        "variant_name": "mixture_of_n_variant",
    });

    let mut event_source = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .await
        .unwrap();

    let mut last_chunk = None;
    while let Some(event) = event_source.next().await {
        let event = event.unwrap();
        match event {
            Event::Open => continue,
            Event::Message(message) => {
                if message.data == "[DONE]" {
                    break;
                }
                last_chunk = Some(message.data);
            }
        }
    }

    let last_chunk: Value = serde_json::from_str(&last_chunk.unwrap()).unwrap();
    let usage = last_chunk.get("usage").unwrap().as_object().unwrap();
    assert!(
        usage.get("cost").unwrap().is_null(),
        "Expected `cost` to be null in streaming mixture-of-N (inline model has no cost config)"
    );
}

// ===== Cache Tests =====

#[tokio::test]
async fn test_cost_cached_non_streaming() {
    let episode_id = Uuid::now_v7();
    let input = json!({
        "system": {"assistant_name": "AskJeeves"},
        "messages": [{"role": "user", "content": "Cache cost test non-streaming"}]
    });

    // First request populates cache (write_only to avoid stale cache hits from prior runs)
    let payload = json!({
        "function_name": "basic_test",
        "episode_id": episode_id,
        "input": input,
        "stream": false,
        "cache_options": {"enabled": "write_only"},
    });
    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let first_response: Value = response.json().await.unwrap();
    let first_cost = first_response["usage"]["cost"].as_f64().unwrap();
    assert!(
        first_cost > 0.0,
        "First (uncached) request should have positive cost"
    );

    // Second request hits cache
    let payload = json!({
        "function_name": "basic_test",
        "episode_id": episode_id,
        "input": input,
        "stream": false,
        "cache_options": {"enabled": "on"},
    });
    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let second_response: Value = response.json().await.unwrap();
    let second_cost = second_response["usage"]["cost"].as_f64().unwrap();
    assert!(
        (second_cost).abs() < 1e-10,
        "Cached response should have zero cost, got {second_cost}"
    );
}

#[tokio::test]
async fn test_cost_cached_streaming() {
    let episode_id = Uuid::now_v7();
    let input = json!({
        "system": {"assistant_name": "AskJeeves"},
        "messages": [{"role": "user", "content": "Cache cost test streaming"}]
    });

    // First request populates cache (write_only to avoid stale cache hits from prior runs)
    let payload = json!({
        "function_name": "basic_test",
        "episode_id": episode_id,
        "input": input,
        "stream": true,
        "cache_options": {"enabled": "write_only"},
    });
    let mut event_source = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .await
        .unwrap();
    while let Some(event) = event_source.next().await {
        let event = event.unwrap();
        match event {
            Event::Open => continue,
            Event::Message(message) => {
                if message.data == "[DONE]" {
                    break;
                }
            }
        }
    }

    // Second request hits cache
    let payload = json!({
        "function_name": "basic_test",
        "episode_id": episode_id,
        "input": input,
        "stream": true,
        "cache_options": {"enabled": "on"},
    });
    let mut event_source = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .await
        .unwrap();

    let mut last_chunk = None;
    while let Some(event) = event_source.next().await {
        let event = event.unwrap();
        match event {
            Event::Open => continue,
            Event::Message(message) => {
                if message.data == "[DONE]" {
                    break;
                }
                last_chunk = Some(message.data);
            }
        }
    }

    let last_chunk: Value = serde_json::from_str(&last_chunk.unwrap()).unwrap();
    let usage = last_chunk.get("usage").unwrap().as_object().unwrap();
    let cost = usage.get("cost").unwrap().as_f64().unwrap();
    assert!(
        (cost).abs() < 1e-10,
        "Cached streaming response should have zero cost, got {cost}"
    );
}

// ===== Real OpenAI Provider Tests =====

#[tokio::test]
async fn test_cost_openai_real_non_streaming() {
    let payload = json!({
        "function_name": "basic_test",
        "variant_name": "openai",
        "episode_id": Uuid::now_v7(),
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [{"role": "user", "content": "Say hello in one word."}]
        },
        "stream": false,
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let response_json: Value = response.json().await.unwrap();

    let usage = response_json.get("usage").unwrap().as_object().unwrap();
    let cost = usage.get("cost").unwrap().as_f64().unwrap();
    assert!(
        cost > 0.0,
        "Expected positive cost from real OpenAI provider, got {cost}"
    );
}

#[tokio::test]
async fn test_cost_openai_real_streaming() {
    let payload = json!({
        "function_name": "basic_test",
        "variant_name": "openai",
        "episode_id": Uuid::now_v7(),
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [{"role": "user", "content": "Say hello in one word."}]
        },
        "stream": true,
    });

    let mut event_source = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .await
        .unwrap();

    let mut last_chunk = None;
    while let Some(event) = event_source.next().await {
        let event = event.unwrap();
        match event {
            Event::Open => continue,
            Event::Message(message) => {
                if message.data == "[DONE]" {
                    break;
                }
                last_chunk = Some(message.data);
            }
        }
    }

    let last_chunk: Value = serde_json::from_str(&last_chunk.unwrap()).unwrap();
    let usage = last_chunk.get("usage").unwrap().as_object().unwrap();
    let cost = usage
        .get("cost")
        .expect("Expected `cost` key in streaming OpenAI usage")
        .as_f64()
        .unwrap();
    assert!(
        cost > 0.0,
        "Expected positive cost from real OpenAI streaming provider, got {cost}"
    );
}
