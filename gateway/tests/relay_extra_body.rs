#![allow(clippy::expect_used, clippy::unwrap_used, clippy::print_stdout)]

// ============================================================================
// Extra Body / Extra Headers Forwarding Tests
// ============================================================================

mod common;

use common::relay::start_relay_test_environment;
use reqwest::Client;
use serde_json::json;
use uuid::Uuid;

#[tokio::test]
async fn test_relay_config_extra_body_forwarding() {
    // Test that variant-level extra_body config is forwarded through relay
    let downstream_config = r#"
[models.test_model]
routing = ["echo"]

[models.test_model.providers.echo]
type = "dummy"
model_name = "echo_injected_data"
"#;

    let relay_config = r#"
[functions.config_test_function]
type = "chat"

[functions.config_test_function.variants.config_variant]
type = "chat_completion"
weight = 1
model = "test_model"
extra_body = [
    { pointer = "/custom_field", value = "config_value" },
    { pointer = "/nested/field", value = 42 }
]

[models.test_model]
routing = ["provider"]

[models.test_model.providers.provider]
type = "dummy"
model_name = "good"
"#;

    let env = start_relay_test_environment(downstream_config, relay_config).await;

    let client = Client::new();
    let response = client
        .post(format!("http://{}/inference", env.relay.addr))
        .json(&json!({
            "function_name": "config_test_function",
            "episode_id": Uuid::now_v7(),
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Test extra body forwarding"
                    }
                ]
            },
            "stream": false
        }))
        .send()
        .await
        .unwrap();

    let status = response.status();
    assert_eq!(
        status,
        200,
        "Response status: {status}, body: {:?}",
        response.text().await
    );

    let body: serde_json::Value = response.json().await.unwrap();
    let content = body["content"].as_array().unwrap();
    assert!(!content.is_empty(), "Expected non-empty content");

    let text_content = content[0]["text"].as_str().unwrap();
    let injected_data: serde_json::Value = serde_json::from_str(text_content).unwrap();

    // Verify extra_body matches exactly
    let expected_body = json!({
        "custom_field": "config_value",
        "nested": {
            "field": 42
        }
    });
    assert_eq!(
        injected_data["injected_body"], expected_body,
        "injected_body should match expected structure exactly"
    );
}

#[tokio::test]
async fn test_relay_config_extra_headers_forwarding() {
    // Test that variant-level extra_headers config is forwarded through relay
    let downstream_config = r#"
[models.test_model]
routing = ["echo"]

[models.test_model.providers.echo]
type = "dummy"
model_name = "echo_injected_data"
"#;

    let relay_config = r#"
[functions.config_test_function]
type = "chat"

[functions.config_test_function.variants.config_variant]
type = "chat_completion"
weight = 1
model = "test_model"
extra_headers = [
    { name = "X-Custom-Header", value = "custom-value" },
    { name = "X-Api-Version", value = "v2" }
]

[models.test_model]
routing = ["provider"]

[models.test_model.providers.provider]
type = "dummy"
model_name = "good"
"#;

    let env = start_relay_test_environment(downstream_config, relay_config).await;

    let client = Client::new();
    let response = client
        .post(format!("http://{}/inference", env.relay.addr))
        .json(&json!({
            "function_name": "config_test_function",
            "episode_id": Uuid::now_v7(),
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Test extra headers forwarding"
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
    let content = body["content"].as_array().unwrap();
    assert!(!content.is_empty(), "Expected non-empty content");

    let text_content = content[0]["text"].as_str().unwrap();
    let injected_data: serde_json::Value = serde_json::from_str(text_content).unwrap();

    // Verify extra_headers match exactly (headers are returned as [name, value] pairs)
    let expected_headers = json!([["x-custom-header", "custom-value"], ["x-api-version", "v2"]]);
    assert_eq!(
        injected_data["injected_headers"], expected_headers,
        "injected_headers should match expected structure exactly"
    );
}

#[tokio::test]
async fn test_relay_inference_extra_body_forwarding() {
    let downstream_config = r#"
[models.echo_model]
routing = ["echo"]

[models.echo_model.providers.echo]
type = "dummy"
model_name = "echo_injected_data"
"#;

    let relay_config = r#"
[functions.test_function]
type = "chat"

[functions.test_function.variants.test_variant]
type = "chat_completion"
weight = 1
model = "echo_model"

[models.echo_model]
routing = ["provider"]

[models.echo_model.providers.provider]
type = "dummy"
model_name = "good"
"#;

    let env = start_relay_test_environment(downstream_config, relay_config).await;

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
                        "content": "Test inference extra body"
                    }
                ]
            },
            "extra_body": [
                { "model_name": "echo_model", "provider_name": "echo", "pointer": "/inference_field", "value": "inference_value" },
                { "model_name": "echo_model", "provider_name": "echo", "pointer": "/priority", "value": "high" },
                { "variant_name": "test_variant", "pointer": "/variant_field", "value": "variant_value" }
            ],
            "stream": false
        }))
        .send()
        .await
        .unwrap();

    let status = response.status();
    let response_text = response.text().await.unwrap();
    println!("Response text: {response_text}");

    assert_eq!(status, 200);

    let body: serde_json::Value = serde_json::from_str(&response_text).unwrap();
    let content = body["content"].as_array().unwrap();
    assert!(!content.is_empty());

    let text_content = content[0]["text"].as_str().unwrap();
    let injected_data: serde_json::Value = serde_json::from_str(text_content).unwrap();

    // Verify extra_body matches exactly
    let expected_body = json!({
        "inference_field": "inference_value",
        "priority": "high",
        "variant_field": "variant_value"
    });
    assert_eq!(
        injected_data["injected_body"], expected_body,
        "injected_body should match expected structure exactly"
    );
}

#[tokio::test]
async fn test_relay_inference_extra_headers_forwarding() {
    let downstream_config = r#"
[models.echo_model]
routing = ["echo"]

[models.echo_model.providers.echo]
type = "dummy"
model_name = "echo_injected_data"
"#;

    let relay_config = r#"
[functions.test_function]
type = "chat"

[functions.test_function.variants.test_variant]
type = "chat_completion"
weight = 1
model = "echo_model"

[models.echo_model]
routing = ["provider"]

[models.echo_model.providers.provider]
type = "dummy"
model_name = "good"
"#;

    let env = start_relay_test_environment(downstream_config, relay_config).await;

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
                        "content": "Test inference extra headers"
                    }
                ]
            },
            "extra_headers": [
                { "model_name": "echo_model", "provider_name": "echo", "name": "X-Request-Id", "value": "req-123" },
                { "model_name": "echo_model", "provider_name": "echo", "name": "X-Trace-Id", "value": "trace-456" },
                { "variant_name": "test_variant", "name": "X-Variant-Header", "value": "variant-value" }
            ],
            "stream": false
        }))
        .send()
        .await
        .unwrap();

    let status = response.status();
    let response_text = response.text().await.unwrap();
    println!("Response text: {response_text}");

    assert_eq!(status, 200);

    let body: serde_json::Value = serde_json::from_str(&response_text).unwrap();
    let content = body["content"].as_array().unwrap();
    assert!(!content.is_empty());

    let text_content = content[0]["text"].as_str().unwrap();
    let injected_data: serde_json::Value = serde_json::from_str(text_content).unwrap();

    // Verify extra_headers match exactly
    let expected_headers = json!([
        ["x-request-id", "req-123"],
        ["x-trace-id", "trace-456"],
        ["x-variant-header", "variant-value"]
    ]);
    assert_eq!(
        injected_data["injected_headers"], expected_headers,
        "injected_headers should match expected structure exactly"
    );
}

#[tokio::test]
async fn test_relay_variant_level_extra_body() {
    let downstream_config = r#"
[models.test_model]
routing = ["echo"]

[models.test_model.providers.echo]
type = "dummy"
model_name = "echo_injected_data"
"#;

    let relay_config = r#"
[functions.test_function]
type = "chat"

[functions.test_function.variants.test_variant]
type = "chat_completion"
weight = 1
model = "test_model"
extra_body = [
    { pointer = "/variant_field", value = "variant_value" }
]

[models.test_model]
routing = ["provider"]

[models.test_model.providers.provider]
type = "dummy"
model_name = "good"
"#;

    let env = start_relay_test_environment(downstream_config, relay_config).await;

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
                        "content": "Test variant extra body"
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
    let content = body["content"].as_array().unwrap();
    assert!(!content.is_empty());

    let text_content = content[0]["text"].as_str().unwrap();
    let injected_data: serde_json::Value = serde_json::from_str(text_content).unwrap();

    // Verify extra_body matches exactly
    let expected_body = json!({
        "variant_field": "variant_value"
    });
    assert_eq!(
        injected_data["injected_body"], expected_body,
        "injected_body should match expected structure exactly"
    );
}

#[tokio::test]
async fn test_relay_variant_level_extra_headers() {
    let downstream_config = r#"
[models.test_model]
routing = ["echo"]

[models.test_model.providers.echo]
type = "dummy"
model_name = "echo_injected_data"
"#;

    let relay_config = r#"
[functions.test_function]
type = "chat"

[functions.test_function.variants.test_variant]
type = "chat_completion"
weight = 1
model = "test_model"
extra_headers = [
    { name = "X-Variant-Header", value = "variant-value" }
]

[models.test_model]
routing = ["provider"]

[models.test_model.providers.provider]
type = "dummy"
model_name = "good"
"#;

    let env = start_relay_test_environment(downstream_config, relay_config).await;

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
                        "content": "Test variant extra headers"
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
    let content = body["content"].as_array().unwrap();
    assert!(!content.is_empty());

    let text_content = content[0]["text"].as_str().unwrap();
    let injected_data: serde_json::Value = serde_json::from_str(text_content).unwrap();

    // Verify extra_headers match exactly
    let expected_headers = json!([["x-variant-header", "variant-value"]]);
    assert_eq!(
        injected_data["injected_headers"], expected_headers,
        "injected_headers should match expected structure exactly"
    );
}

#[tokio::test]
async fn test_relay_combined_extra_body_merge() {
    // Test that variant-level and inference-level extra_body are merged correctly
    let downstream_config = r#"
[models.test_model]
routing = ["echo"]

[models.test_model.providers.echo]
type = "dummy"
model_name = "echo_injected_data"
"#;

    let relay_config = r#"
[functions.combined_function]
type = "chat"

[functions.combined_function.variants.combined_variant]
type = "chat_completion"
weight = 1
model = "test_model"
extra_body = [
    { pointer = "/config_field", value = "from_config" }
]

[models.test_model]
routing = ["provider"]

[models.test_model.providers.provider]
type = "dummy"
model_name = "good"
"#;

    let env = start_relay_test_environment(downstream_config, relay_config).await;

    let client = Client::new();
    let response = client
        .post(format!("http://{}/inference", env.relay.addr))
        .json(&json!({
            "function_name": "combined_function",
            "episode_id": Uuid::now_v7(),
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Test combined extra body"
                    }
                ]
            },
            "extra_body": [
                { "pointer": "/inference_field", "value": "from_inference" }
            ],
            "stream": false
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);

    let body: serde_json::Value = response.json().await.unwrap();
    let content = body["content"].as_array().unwrap();
    assert!(!content.is_empty());

    let text_content = content[0]["text"].as_str().unwrap();
    let injected_data: serde_json::Value = serde_json::from_str(text_content).unwrap();

    // Verify extra_body matches exactly (both config and inference fields)
    let expected_body = json!({
        "config_field": "from_config",
        "inference_field": "from_inference"
    });
    assert_eq!(
        injected_data["injected_body"], expected_body,
        "injected_body should match expected structure exactly"
    );
}

#[tokio::test]
async fn test_relay_combined_extra_headers_merge() {
    // Test that variant-level and inference-level extra_headers are merged correctly
    let downstream_config = r#"
[models.test_model]
routing = ["echo"]

[models.test_model.providers.echo]
type = "dummy"
model_name = "echo_injected_data"
"#;

    let relay_config = r#"
[functions.combined_function]
type = "chat"

[functions.combined_function.variants.combined_variant]
type = "chat_completion"
weight = 1
model = "test_model"
extra_headers = [
    { name = "X-Config-Header", value = "from-config" }
]

[models.test_model]
routing = ["provider"]

[models.test_model.providers.provider]
type = "dummy"
model_name = "good"
"#;

    let env = start_relay_test_environment(downstream_config, relay_config).await;

    let client = Client::new();
    let response = client
        .post(format!("http://{}/inference", env.relay.addr))
        .json(&json!({
            "function_name": "combined_function",
            "episode_id": Uuid::now_v7(),
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Test combined extra headers"
                    }
                ]
            },
            "extra_headers": [
                { "name": "X-Inference-Header", "value": "from-inference" }
            ],
            "stream": false
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);

    let body: serde_json::Value = response.json().await.unwrap();
    let content = body["content"].as_array().unwrap();
    assert!(!content.is_empty());

    let text_content = content[0]["text"].as_str().unwrap();
    let injected_data: serde_json::Value = serde_json::from_str(text_content).unwrap();

    // Verify extra_headers match exactly (both config and inference headers)
    let expected_headers = json!([
        ["x-config-header", "from-config"],
        ["x-inference-header", "from-inference"]
    ]);
    assert_eq!(
        injected_data["injected_headers"], expected_headers,
        "injected_headers should match expected structure exactly"
    );
}

#[tokio::test]
async fn test_relay_combined_different_pointers() {
    // Test that variant-level and inference-level extra_body with different paths don't conflict
    let downstream_config = r#"
[models.test_model]
routing = ["echo"]

[models.test_model.providers.echo]
type = "dummy"
model_name = "echo_injected_data"
"#;

    let relay_config = r#"
[functions.combined_function]
type = "chat"

[functions.combined_function.variants.combined_variant]
type = "chat_completion"
weight = 1
model = "test_model"
extra_body = [
    { pointer = "/field1", value = "config" }
]

[models.test_model]
routing = ["provider"]

[models.test_model.providers.provider]
type = "dummy"
model_name = "good"
"#;

    let env = start_relay_test_environment(downstream_config, relay_config).await;

    let client = Client::new();
    let response = client
        .post(format!("http://{}/inference", env.relay.addr))
        .json(&json!({
            "function_name": "combined_function",
            "episode_id": Uuid::now_v7(),
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Test different pointers"
                    }
                ]
            },
            "extra_body": [
                { "pointer": "/field2", "value": "inference" }
            ],
            "stream": false
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);

    let body: serde_json::Value = response.json().await.unwrap();
    let content = body["content"].as_array().unwrap();
    assert!(!content.is_empty());

    let text_content = content[0]["text"].as_str().unwrap();
    let injected_data: serde_json::Value = serde_json::from_str(text_content).unwrap();

    // Verify extra_body matches exactly (different pointers don't conflict)
    let expected_body = json!({
        "field1": "config",
        "field2": "inference"
    });
    assert_eq!(
        injected_data["injected_body"], expected_body,
        "injected_body should match expected structure exactly"
    );
}

#[tokio::test]
async fn test_relay_combined_same_pointer_override() {
    let downstream_config = r#"
[models.echo_model]
routing = ["echo"]

[models.echo_model.providers.echo]
type = "dummy"
model_name = "echo_injected_data"
"#;

    let relay_config = r#"
[models.echo_model]
routing = ["provider"]

[models.echo_model.providers.provider]
type = "dummy"
model_name = "good"
extra_body = [
    { pointer = "/temp", value = "0.5" }
]
"#;

    let env = start_relay_test_environment(downstream_config, relay_config).await;

    let client = Client::new();
    let response = client
        .post(format!("http://{}/inference", env.relay.addr))
        .json(&json!({
            "model_name": "echo_model",
            "episode_id": Uuid::now_v7(),
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Test pointer override"
                    }
                ]
            },
            "extra_body": [
                { "pointer": "/temp", "value": "0.9" }
            ],
            "stream": false
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);

    let body: serde_json::Value = response.json().await.unwrap();
    let content = body["content"].as_array().unwrap();
    assert!(!content.is_empty());

    let text_content = content[0]["text"].as_str().unwrap();
    let injected_data: serde_json::Value = serde_json::from_str(text_content).unwrap();

    // Verify extra_body matches exactly (inference-level overrides config-level)
    // Based on prepare_relay_extra_body implementation, inference extends config,
    // so the last value wins
    let expected_body = json!({
        "temp": "0.9"
    });
    assert_eq!(
        injected_data["injected_body"], expected_body,
        "injected_body should match expected structure exactly"
    );
}

#[tokio::test]
async fn test_relay_config_extra_body_deletion() {
    let downstream_config = r#"
[models.echo_model]
routing = ["echo"]

[models.echo_model.providers.echo]
type = "dummy"
model_name = "echo_injected_data"
"#;

    let relay_config = r#"
[models.echo_model]
routing = ["provider"]

[models.echo_model.providers.provider]
type = "dummy"
model_name = "good"
extra_body = [
    { pointer = "/field_to_delete", delete = true }
]
"#;

    let env = start_relay_test_environment(downstream_config, relay_config).await;

    let client = Client::new();
    let response = client
        .post(format!("http://{}/inference", env.relay.addr))
        .json(&json!({
            "model_name": "echo_model",
            "episode_id": Uuid::now_v7(),
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Test deletion"
                    }
                ]
            },
            "stream": false
        }))
        .send()
        .await
        .unwrap();

    // Should succeed - deletion is forwarded but doesn't break if field doesn't exist
    assert_eq!(response.status(), 200);

    let body: serde_json::Value = response.json().await.unwrap();
    let content = body["content"].as_array().unwrap();
    assert!(!content.is_empty());
}

#[tokio::test]
async fn test_relay_config_extra_headers_deletion() {
    let downstream_config = r#"
[models.echo_model]
routing = ["echo"]

[models.echo_model.providers.echo]
type = "dummy"
model_name = "echo_injected_data"
"#;

    let relay_config = r#"
[models.echo_model]
routing = ["provider"]

[models.echo_model.providers.provider]
type = "dummy"
model_name = "good"
extra_headers = [
    { name = "X-Header-To-Delete", delete = true }
]
"#;

    let env = start_relay_test_environment(downstream_config, relay_config).await;

    let client = Client::new();
    let response = client
        .post(format!("http://{}/inference", env.relay.addr))
        .json(&json!({
            "model_name": "echo_model",
            "episode_id": Uuid::now_v7(),
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Test header deletion"
                    }
                ]
            },
            "stream": false
        }))
        .send()
        .await
        .unwrap();

    // Should succeed - deletion is forwarded
    assert_eq!(response.status(), 200);

    let body: serde_json::Value = response.json().await.unwrap();
    let content = body["content"].as_array().unwrap();
    assert!(!content.is_empty());
}

#[tokio::test]
async fn test_relay_inference_extra_body_deletion() {
    let downstream_config = r#"
[models.echo_model]
routing = ["echo"]

[models.echo_model.providers.echo]
type = "dummy"
model_name = "echo_injected_data"
"#;

    let relay_config = r#"
[models.echo_model]
routing = ["provider"]

[models.echo_model.providers.provider]
type = "dummy"
model_name = "good"
"#;

    let env = start_relay_test_environment(downstream_config, relay_config).await;

    let client = Client::new();
    let response = client
        .post(format!("http://{}/inference", env.relay.addr))
        .json(&json!({
            "model_name": "echo_model",
            "episode_id": Uuid::now_v7(),
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Test inference deletion"
                    }
                ]
            },
            "extra_body": [
                { "pointer": "/field_to_delete", "delete": true }
            ],
            "stream": false
        }))
        .send()
        .await
        .unwrap();

    // Should succeed - deletion is forwarded from inference request
    assert_eq!(response.status(), 200);

    let body: serde_json::Value = response.json().await.unwrap();
    let content = body["content"].as_array().unwrap();
    assert!(!content.is_empty());
}

#[tokio::test]
async fn test_relay_inference_extra_headers_deletion() {
    let downstream_config = r#"
[models.echo_model]
routing = ["echo"]

[models.echo_model.providers.echo]
type = "dummy"
model_name = "echo_injected_data"
"#;

    let relay_config = r#"
[models.echo_model]
routing = ["provider"]

[models.echo_model.providers.provider]
type = "dummy"
model_name = "good"
"#;

    let env = start_relay_test_environment(downstream_config, relay_config).await;

    let client = Client::new();
    let response = client
        .post(format!("http://{}/inference", env.relay.addr))
        .json(&json!({
            "model_name": "echo_model",
            "episode_id": Uuid::now_v7(),
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Test inference header deletion"
                    }
                ]
            },
            "extra_headers": [
                { "name": "X-Header-To-Delete", "delete": true }
            ],
            "stream": false
        }))
        .send()
        .await
        .unwrap();

    // Should succeed - deletion is forwarded from inference request
    assert_eq!(response.status(), 200);

    let body: serde_json::Value = response.json().await.unwrap();
    let content = body["content"].as_array().unwrap();
    assert!(!content.is_empty());
}

#[tokio::test]
async fn test_relay_extra_body_headers_non_streaming() {
    // Test that both extra_body and extra_headers work together in non-streaming mode
    let downstream_config = r#"
[models.echo_model]
routing = ["echo"]

[models.echo_model.providers.echo]
type = "dummy"
model_name = "echo_injected_data"
"#;

    let relay_config = r#"
[models.echo_model]
routing = ["provider"]

[models.echo_model.providers.provider]
type = "dummy"
model_name = "good"
"#;

    let env = start_relay_test_environment(downstream_config, relay_config).await;

    let client = Client::new();
    let response = client
        .post(format!("http://{}/inference", env.relay.addr))
        .json(&json!({
            "model_name": "echo_model",
            "episode_id": Uuid::now_v7(),
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Test non-streaming"
                    }
                ]
            },
            "extra_body": [
                { "pointer": "/test_field", "value": "test_value" }
            ],
            "extra_headers": [
                { "name": "X-Test-Header", "value": "test-value" }
            ],
            "stream": false
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);

    let body: serde_json::Value = response.json().await.unwrap();
    let content = body["content"].as_array().unwrap();
    assert!(!content.is_empty());

    let text_content = content[0]["text"].as_str().unwrap();
    let injected_data: serde_json::Value = serde_json::from_str(text_content).unwrap();

    // Verify extra_body matches exactly
    let expected_body = json!({
        "test_field": "test_value"
    });
    assert_eq!(
        injected_data["injected_body"], expected_body,
        "injected_body should match expected structure exactly"
    );

    let expected_headers = json!([["x-test-header", "test-value"]]);
    assert_eq!(
        injected_data["injected_headers"], expected_headers,
        "injected_headers should match expected structure exactly"
    );
}

#[tokio::test]
async fn test_relay_extra_body_and_headers_combined() {
    // Test that both extra_body and extra_headers work together
    // Note: dummy provider's echo_injected_data doesn't support streaming
    let downstream_config = r#"
[models.echo_model]
routing = ["echo"]

[models.echo_model.providers.echo]
type = "dummy"
model_name = "echo_injected_data"
"#;

    let relay_config = r#"
[models.echo_model]
routing = ["provider"]

[models.echo_model.providers.provider]
type = "dummy"
model_name = "good"
"#;

    let env = start_relay_test_environment(downstream_config, relay_config).await;

    let client = Client::new();
    let response = client
        .post(format!("http://{}/inference", env.relay.addr))
        .json(&json!({
            "model_name": "echo_model",
            "episode_id": Uuid::now_v7(),
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Test combined"
                    }
                ]
            },
            "extra_body": [
                { "pointer": "/combined_field", "value": "combined_value" }
            ],
            "extra_headers": [
                { "name": "X-Combined-Header", "value": "combined-value" }
            ],
            "stream": false
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);

    let body: serde_json::Value = response.json().await.unwrap();
    let content = body["content"].as_array().unwrap();
    assert!(!content.is_empty());

    let text_content = content[0]["text"].as_str().unwrap();
    let injected_data: serde_json::Value = serde_json::from_str(text_content).unwrap();

    // Verify extra_body matches exactly
    let expected_body = json!({
        "combined_field": "combined_value"
    });
    assert_eq!(
        injected_data["injected_body"], expected_body,
        "injected_body should match expected structure exactly"
    );

    let expected_headers = json!([["x-combined-header", "combined-value"]]);
    assert_eq!(
        injected_data["injected_headers"], expected_headers,
        "injected_headers should match expected structure exactly"
    );
}

#[tokio::test]
async fn test_relay_empty_extra_body_headers() {
    let downstream_config = r#"
[models.echo_model]
routing = ["echo"]

[models.echo_model.providers.echo]
type = "dummy"
model_name = "echo_injected_data"
"#;

    let relay_config = r#"
[models.echo_model]
routing = ["provider"]

[models.echo_model.providers.provider]
type = "dummy"
model_name = "good"
"#;

    let env = start_relay_test_environment(downstream_config, relay_config).await;

    let client = Client::new();
    let response = client
        .post(format!("http://{}/inference", env.relay.addr))
        .json(&json!({
            "model_name": "echo_model",
            "episode_id": Uuid::now_v7(),
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Test empty"
                    }
                ]
            },
            "stream": false
        }))
        .send()
        .await
        .unwrap();

    // Should succeed even without extra_body/extra_headers
    assert_eq!(response.status(), 200);

    let body: serde_json::Value = response.json().await.unwrap();
    assert!(body.get("inference_id").is_some());
    assert!(body.get("content").is_some());
}

#[tokio::test]
async fn test_relay_nested_json_pointer() {
    // Test that deeply nested JSON pointer paths work correctly
    let downstream_config = r#"
[models.echo_model]
routing = ["echo"]

[models.echo_model.providers.echo]
type = "dummy"
model_name = "echo_injected_data"
"#;

    let relay_config = r#"
[models.echo_model]
routing = ["provider"]

[models.echo_model.providers.provider]
type = "dummy"
model_name = "good"
"#;

    let env = start_relay_test_environment(downstream_config, relay_config).await;

    let client = Client::new();
    let response = client
        .post(format!("http://{}/inference", env.relay.addr))
        .json(&json!({
            "model_name": "echo_model",
            "episode_id": Uuid::now_v7(),
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Test nested pointer"
                    }
                ]
            },
            "extra_body": [
                { "pointer": "/level1/level2/level3/field", "value": "deep" }
            ],
            "stream": false
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);

    let body: serde_json::Value = response.json().await.unwrap();
    let content = body["content"].as_array().unwrap();
    assert!(!content.is_empty());

    let text_content = content[0]["text"].as_str().unwrap();
    let injected_data: serde_json::Value = serde_json::from_str(text_content).unwrap();

    // Verify extra_body matches exactly (deeply nested structure)
    let expected_body = json!({
        "level1": {
            "level2": {
                "level3": {
                    "field": "deep"
                }
            }
        }
    });
    assert_eq!(
        injected_data["injected_body"], expected_body,
        "injected_body should match expected structure exactly"
    );
}

#[tokio::test]
async fn test_relay_special_characters_in_values() {
    // Test that special characters in header values are properly handled
    let downstream_config = r#"
[models.echo_model]
routing = ["echo"]

[models.echo_model.providers.echo]
type = "dummy"
model_name = "echo_injected_data"
"#;

    let relay_config = r#"
[models.echo_model]
routing = ["provider"]

[models.echo_model.providers.provider]
type = "dummy"
model_name = "good"
"#;

    let env = start_relay_test_environment(downstream_config, relay_config).await;

    let client = Client::new();
    let response = client
        .post(format!("http://{}/inference", env.relay.addr))
        .json(&json!({
            "model_name": "echo_model",
            "episode_id": Uuid::now_v7(),
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Test special characters"
                    }
                ]
            },
            "extra_headers": [
                { "name": "X-Special", "value": "value with spaces, commas, and special chars: @#$%" }
            ],
            "stream": false
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);

    let body: serde_json::Value = response.json().await.unwrap();
    let content = body["content"].as_array().unwrap();
    assert!(!content.is_empty());

    let text_content = content[0]["text"].as_str().unwrap();
    let injected_data: serde_json::Value = serde_json::from_str(text_content).unwrap();

    // Verify extra_headers match exactly (special characters preserved)
    let expected_headers = json!([[
        "x-special",
        "value with spaces, commas, and special chars: @#$%"
    ]]);
    assert_eq!(
        injected_data["injected_headers"], expected_headers,
        "injected_headers should match expected structure exactly"
    );
}

#[tokio::test]
async fn test_relay_nested_object_json_pointer() {
    // Test that JSON pointers with deeply nested objects work correctly
    // Note: TensorZero doesn't support creating arrays via JSON pointers
    let downstream_config = r#"
[models.echo_model]
routing = ["echo"]

[models.echo_model.providers.echo]
type = "dummy"
model_name = "echo_injected_data"
"#;

    let relay_config = r#"
[models.echo_model]
routing = ["provider"]

[models.echo_model.providers.provider]
type = "dummy"
model_name = "good"
"#;

    let env = start_relay_test_environment(downstream_config, relay_config).await;

    let client = Client::new();
    let response = client
        .post(format!("http://{}/inference", env.relay.addr))
        .json(&json!({
            "model_name": "echo_model",
            "episode_id": Uuid::now_v7(),
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Test nested object pointer"
                    }
                ]
            },
            "extra_body": [
                { "pointer": "/items/first/name", "value": "first_item" },
                { "pointer": "/items/second/name", "value": "second_item" }
            ],
            "stream": false
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), 200);

    let body: serde_json::Value = response.json().await.unwrap();
    let content = body["content"].as_array().unwrap();
    assert!(!content.is_empty());

    let text_content = content[0]["text"].as_str().unwrap();
    let injected_data: serde_json::Value = serde_json::from_str(text_content).unwrap();

    // Verify extra_body matches exactly (nested object structure)
    let expected_body = json!({
        "items": {
            "first": {
                "name": "first_item"
            },
            "second": {
                "name": "second_item"
            }
        }
    });
    assert_eq!(
        injected_data["injected_body"], expected_body,
        "injected_body should match expected structure exactly"
    );
}
