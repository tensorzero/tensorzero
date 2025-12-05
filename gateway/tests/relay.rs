#![allow(clippy::expect_used, clippy::unwrap_used, clippy::print_stdout)]

//! Tests for the TensorZero relay feature.
//!
//! These tests spawn two gateway processes:
//! 1. A "downstream" gateway with actual model providers (using dummy provider)
//! 2. A "relay" gateway configured with `[relay]` to forward requests to the downstream gateway
//!
//! This validates that the relay correctly proxies inference requests through
//! another TensorZero gateway instance.

mod common;

use common::{start_gateway_on_random_port, ChildData};
use reqwest::Client;
use serde_json::json;
use uuid::Uuid;

/// Test environment with both downstream and relay gateways.
#[expect(dead_code)]
struct RelayTestEnvironment {
    downstream: ChildData,
    relay: ChildData,
}

/// Spawns a relay test environment with both downstream and relay gateways.
async fn start_relay_test_environment(
    downstream_config: &str,
    relay_config_suffix: &str,
) -> RelayTestEnvironment {
    // Start downstream gateway first
    let downstream = start_gateway_on_random_port(downstream_config, None).await;

    // Build relay configuration with downstream port injected
    let relay_config = format!(
        r#"
[relay]
gateway_url = "http://0.0.0.0:{}"

{}
"#,
        downstream.addr.port(),
        relay_config_suffix
    );

    // Start relay gateway
    let relay = start_gateway_on_random_port(&relay_config, None).await;

    RelayTestEnvironment { downstream, relay }
}

// ============================================================================
// Basic Tests
// ============================================================================

#[tokio::test]
async fn test_relay_non_streaming() {
    let downstream_config = "";
    let relay_config = "";

    let env = start_relay_test_environment(downstream_config, relay_config).await;

    // Make a non-streaming inference request to the relay gateway
    // The relay should forward this to the downstream gateway
    let client = Client::new();
    let response = client
        .post(format!("http://{}/inference", env.relay.addr))
        .json(&json!({
            "model_name": "dummy::good",
            "episode_id": Uuid::now_v7(),
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello"
                    }
                ]
            },
            "stream": false
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200, "Response: {:?}", response.text().await);
    let body: serde_json::Value = response.json().await.unwrap();

    // Verify we got a successful inference response
    assert!(body.get("inference_id").is_some());
    assert!(body.get("episode_id").is_some());
    // Episode ID will be a UUID

    // Check that we got content from the model
    let content = body["content"].as_array().unwrap();
    assert!(!content.is_empty(), "Expected non-empty content");
    assert_eq!(content[0]["type"], "text");
    assert!(!content[0]["text"].as_str().unwrap().is_empty());
}

#[tokio::test]
async fn test_relay_streaming() {
    let downstream_config = "";
    let relay_config = "";

    let env = start_relay_test_environment(downstream_config, relay_config).await;

    // Make a streaming inference request to the relay gateway
    let client = Client::new();
    let response = client
        .post(format!("http://{}/inference", env.relay.addr))
        .json(&json!({
            "model_name": "dummy::good",
            "episode_id": Uuid::now_v7(),
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello"
                    }
                ]
            },
            "stream": true
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);

    // Read the streaming response
    let body = response.text().await.unwrap();
    let lines: Vec<&str> = body.lines().collect();

    // Should have multiple SSE lines
    assert!(lines.len() > 1, "Expected multiple streaming chunks");

    // All non-empty lines should start with "data: "
    for line in lines.iter().filter(|l| !l.is_empty()) {
        assert!(
            line.starts_with("data: "),
            "Line should start with 'data: ': {line}"
        );
    }
}

#[tokio::test]
async fn test_relay_with_system_message() {
    let downstream_config = "";
    let relay_config = "";

    let env = start_relay_test_environment(downstream_config, relay_config).await;

    // Make a request with system message
    let client = Client::new();
    let response = client
        .post(format!("http://{}/inference", env.relay.addr))
        .json(&json!({
            "model_name": "dummy::good",
            "episode_id": Uuid::now_v7(),
            "input": {
                "system": "You are a helpful assistant.",
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello"
                    }
                ]
            },
            "stream": false
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().await.unwrap();

    // Verify successful response
    assert!(body.get("inference_id").is_some());
    // Episode ID will be a UUID
}

#[tokio::test]
async fn test_relay_with_inference_params() {
    let downstream_config = "";
    let relay_config = "";

    let env = start_relay_test_environment(downstream_config, relay_config).await;

    // Make request with various inference parameters
    let client = Client::new();
    let response = client
        .post(format!("http://{}/inference", env.relay.addr))
        .json(&json!({
            "model_name": "dummy::good",
            "episode_id": Uuid::now_v7(),
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Test with params"
                    }
                ]
            },
            "params": {
                "chat_completion": {
                    "temperature": 0.8,
                    "max_tokens": 50,
                    "seed": 42
                }
            },
            "stream": false
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().await.unwrap();

    assert!(body.get("inference_id").is_some());
    // Episode ID will be a UUID
}

#[tokio::test]
async fn test_relay_multi_turn_conversation() {
    let downstream_config = "";
    let relay_config = "";

    let env = start_relay_test_environment(downstream_config, relay_config).await;

    // Make request with multi-turn conversation
    let client = Client::new();
    let response = client
        .post(format!("http://{}/inference", env.relay.addr))
        .json(&json!({
            "model_name": "dummy::good",
            "episode_id": Uuid::now_v7(),
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": "First message"
                    },
                    {
                        "role": "assistant",
                        "content": "First response"
                    },
                    {
                        "role": "user",
                        "content": "Second message"
                    }
                ]
            },
            "stream": false
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().await.unwrap();

    assert!(body.get("inference_id").is_some());
    // Episode ID will be a UUID

    // Should get a response for the multi-turn conversation
    let content = body["content"].as_array().unwrap();
    assert!(!content.is_empty());
}

#[tokio::test]
async fn test_relay_with_configured_model() {
    // Configure downstream with a working provider ("good")
    let downstream_config = r#"
[models.test_model]
routing = ["good"]

[models.test_model.providers.good]
type = "dummy"
model_name = "good"
"#;
    // Configure relay with a failing provider ("error")
    // If relay forwards to downstream, we get success from downstream's "good" provider
    // If relay handles locally, we get an error from relay's "error" provider
    let relay_config = r#"
[models.test_model]
routing = ["error"]

[models.test_model.providers.error]
type = "dummy"
model_name = "error"
"#;

    let env = start_relay_test_environment(downstream_config, relay_config).await;

    // Make request to a configured model through the relay gateway
    let client = Client::new();
    let response = client
        .post(format!("http://{}/inference", env.relay.addr))
        .json(&json!({
            "model_name": "test_model",
            "episode_id": Uuid::now_v7(),
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello"
                    }
                ]
            },
            "stream": false
        }))
        .send()
        .await
        .unwrap();

    let status = response.status();
    let body_text = response.text().await.unwrap();
    // Should succeed because relay forwards to downstream's working provider
    assert_eq!(status, 200, "Response status: {status}, body: {body_text}");
    let body: serde_json::Value = serde_json::from_str(&body_text).unwrap();

    assert!(body.get("inference_id").is_some());
    assert!(body.get("content").is_some(), "Should have content from downstream");
    // Verify we got actual content (proving it came from downstream's "good" provider, not relay's "error")
    let content = body["content"].as_array().unwrap();
    assert!(!content.is_empty(), "Expected non-empty content from downstream");
}

#[tokio::test]
async fn test_relay_with_function() {
    // Note: For functions, relay resolves the function to a model locally before forwarding.
    // So both relay and downstream need identical function configs.
    // We verify relay is working by ensuring function calls succeed through the relay.
    let downstream_config = r#"
[functions.test_function]
type = "chat"

[functions.test_function.variants.test_variant]
type = "chat_completion"
weight = 1
model = "dummy::good"
"#;
    let relay_config = r#"
[functions.test_function]
type = "chat"

[functions.test_function.variants.test_variant]
type = "chat_completion"
weight = 1
model = "dummy::good"
"#;

    let env = start_relay_test_environment(downstream_config, relay_config).await;

    // Make function inference request through relay
    let client = Client::new();
    let response = client
        .post(format!("http://{}/inference", env.relay.addr))
        .json(&json!({
            "function_name": "test_function",
            "episode_id": Uuid::now_v7(),
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello via function"
                    }
                ]
            },
            "stream": false
        }))
        .send()
        .await
        .unwrap();

    let status = response.status();
    let body_text = response.text().await.unwrap();
    // Should succeed, verifying that function calls work through relay
    assert_eq!(status, 200, "Response status: {status}, body: {body_text}");
    let body: serde_json::Value = serde_json::from_str(&body_text).unwrap();

    assert!(body.get("inference_id").is_some());
    assert!(body.get("content").is_some(), "Should have content from inference");
    let content = body["content"].as_array().unwrap();
    assert!(!content.is_empty(), "Expected non-empty content");
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[tokio::test]
async fn test_relay_downstream_unreachable() {
    // Configure relay to point to a non-existent port
    let relay_config = "";

    // Don't start the downstream gateway, just start the relay
    let relay = start_gateway_on_random_port(
        &format!(
            r#"
[relay]
gateway_url = "http://0.0.0.0:19999"

{relay_config}
"#
        ),
        None,
    )
    .await;

    // Make a request that should fail due to unreachable downstream
    let client = Client::new();
    let response = client
        .post(format!("http://{}/inference", relay.addr))
        .json(&json!({
            "model_name": "dummy::good",
            "episode_id": Uuid::now_v7(),
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello"
                    }
                ]
            },
            "stream": false
        }))
        .send()
        .await
        .unwrap();

    // Should return an error status
    assert!(
        response.status().is_client_error() || response.status().is_server_error(),
        "Expected error status, got: {}",
        response.status()
    );
}

#[tokio::test]
async fn test_relay_downstream_error_model() {
    let downstream_config = "";
    let relay_config = "";

    let env = start_relay_test_environment(downstream_config, relay_config).await;

    // Make a request targeting the error model on downstream
    let client = Client::new();
    let response = client
        .post(format!("http://{}/inference", env.relay.addr))
        .json(&json!({
            "model_name": "dummy::error",
            "episode_id": Uuid::now_v7(),
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello"
                    }
                ]
            },
            "stream": false
        }))
        .send()
        .await
        .unwrap();

    // Should propagate the error from downstream
    assert!(
        response.status().is_client_error() || response.status().is_server_error(),
        "Expected error status, got: {}",
        response.status()
    );
}

// ============================================================================
// Model Configuration Tests
// ============================================================================

#[tokio::test]
async fn test_skip_relay_mixed_models() {
    // Test the skip_relay model option by configuring two models:
    // - local_model: skip_relay=true (uses local provider)
    // - relay_model: skip_relay=false (tries to relay to downstream)
    //
    // Since we don't start a downstream gateway, relay_model should fail
    // while local_model should succeed using its local provider.

    let relay_config = r#"
[relay]
gateway_url = "http://tensorzero.invalid"

[models.local_model]
routing = ["good"]
skip_relay = true

[models.local_model.providers.good]
type = "dummy"
model_name = "good"

[models.relay_model]
routing = ["also_good"]

[models.relay_model.providers.also_good]
type = "dummy"
model_name = "good"
"#;

    // Start ONLY the relay gateway (no downstream)
    let relay = start_gateway_on_random_port(relay_config, None).await;

    let client = Client::new();

    // Test 1: Model with skip_relay=true should succeed using local provider
    let response_local = client
        .post(format!("http://{}/inference", relay.addr))
        .json(&json!({
            "model_name": "local_model",
            "episode_id": Uuid::now_v7(),
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Test skip_relay=true"
                    }
                ]
            },
            "stream": false
        }))
        .send()
        .await
        .unwrap();

    let status_local = response_local.status();
    let body_local_text = response_local.text().await.unwrap();
    assert_eq!(
        status_local, 200,
        "Model with skip_relay=true should succeed using local provider. Status: {}, Body: {}",
        status_local, body_local_text
    );

    let body_local: serde_json::Value = serde_json::from_str(&body_local_text).unwrap();
    assert!(
        body_local.get("inference_id").is_some(),
        "Should have inference_id"
    );
    assert!(
        body_local.get("episode_id").is_some(),
        "Should have episode_id"
    );

    // Verify content exists (proves local provider executed)
    let content = body_local["content"].as_array().unwrap();
    assert!(
        !content.is_empty(),
        "Expected non-empty content from local provider"
    );

    // Test 2: Model with skip_relay=false (default) should fail trying to reach downstream
    let response_relay = client
        .post(format!("http://{}/inference", relay.addr))
        .json(&json!({
            "model_name": "relay_model",
            "episode_id": Uuid::now_v7(),
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Test skip_relay=false"
                    }
                ]
            },
            "stream": false
        }))
        .send()
        .await
        .unwrap();

    let status_relay = response_relay.status();
    assert!(
        status_relay.is_server_error(),
        "Model with skip_relay=false should fail when downstream unreachable. Got status: {}",
        status_relay
    );
}
