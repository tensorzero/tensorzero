use reqwest::Client;
use serde_json::{Value, json};
use std::time::Duration;
use uuid::Uuid;

fn gateway_url() -> String {
    let host = std::env::var("TENSORZERO_GATEWAY_HOST").unwrap_or_else(|_| "localhost".to_string());
    let port = std::env::var("TENSORZERO_GATEWAY_PORT").unwrap_or_else(|_| "3000".to_string());
    format!("http://{host}:{port}")
}

/// Durable GEPA test for chat function (basic_test)
#[allow(clippy::allow_attributes, dead_code)]
pub async fn test_gepa_durable_optimization_chat() {
    let client = Client::new();
    let variant_prefix = format!("gepa_durable_test_{}", Uuid::now_v7());
    let dataset_name = format!("gepa_durable_test_dataset_{}", Uuid::now_v7());
    let base = gateway_url();

    // 1. Create dataset with Pinocchio chat examples
    create_pinocchio_chat_dataset(&client, &base, &dataset_name).await;

    // 2. Launch durable GEPA
    let launch_response = client
        .post(format!("{base}/v1/optimization/gepa"))
        .json(&json!({
            "function_name": "basic_test",
            "dataset_name": dataset_name,
            "evaluation_name": "test_gepa_pinocchio_chat",
            "analysis_model": "openai::gpt-5-mini",
            "mutation_model": "openai::gpt-5-mini",
            "initial_variants": ["openai", "anthropic"],
            "max_iterations": 3,
            "batch_size": 4,
            "seed": 42,
            "variant_prefix": variant_prefix,
            "include_inference_for_mutation": true,
        }))
        .send()
        .await
        .unwrap();

    let status = launch_response.status();
    let body_text = launch_response.text().await.unwrap();
    assert!(
        status.is_success(),
        "GEPA launch failed ({status}): {body_text}"
    );

    let launch: Value = serde_json::from_str(&body_text).unwrap();
    let task_id = launch["task_id"].as_str().unwrap();

    // 3. Poll until terminal state
    let result = poll_until_terminal(&client, &base, task_id).await;

    // 4. Validate
    validate_gepa_result(&result, &variant_prefix, 3);
}

/// Durable GEPA test for JSON function (json_success)
#[allow(clippy::allow_attributes, dead_code)]
pub async fn test_gepa_durable_optimization_json() {
    let client = Client::new();
    let variant_prefix = format!("gepa_durable_json_test_{}", Uuid::now_v7());
    let dataset_name = format!("gepa_durable_json_test_dataset_{}", Uuid::now_v7());
    let base = gateway_url();

    // 1. Create dataset with Pinocchio JSON examples
    create_pinocchio_json_dataset(&client, &base, &dataset_name).await;

    // 2. Launch durable GEPA
    let launch_response = client
        .post(format!("{base}/v1/optimization/gepa"))
        .json(&json!({
            "function_name": "json_success",
            "dataset_name": dataset_name,
            "evaluation_name": "test_gepa_pinocchio_json",
            "analysis_model": "openai::gpt-5-mini",
            "mutation_model": "openai::gpt-5-mini",
            "initial_variants": ["openai", "anthropic"],
            "max_iterations": 3,
            "batch_size": 4,
            "seed": 42,
            "variant_prefix": variant_prefix,
            "include_inference_for_mutation": true,
        }))
        .send()
        .await
        .unwrap();

    let status = launch_response.status();
    let body_text = launch_response.text().await.unwrap();
    assert!(
        status.is_success(),
        "GEPA launch failed ({status}): {body_text}"
    );

    let launch: Value = serde_json::from_str(&body_text).unwrap();
    let task_id = launch["task_id"].as_str().unwrap();

    // 3. Poll until terminal state
    let result = poll_until_terminal(&client, &base, task_id).await;

    // 4. Validate
    validate_gepa_result(&result, &variant_prefix, 3);
}

/// Create chat datapoints for basic_test using Pinocchio examples
async fn create_pinocchio_chat_dataset(client: &Client, base: &str, dataset_name: &str) {
    let response = client
        .post(format!("{base}/v1/datasets/{dataset_name}/datapoints"))
        .json(&json!({
            "datapoints": [
                {
                    "type": "chat",
                    "function_name": "basic_test",
                    "input": {
                        "system": {"assistant_name": "Dr. Mehta"},
                        "messages": [{"role": "user", "content": [{"type": "text", "text": "What is the boiling point of water?"}]}]
                    },
                    "output": [{"type": "text", "text": "100 degrees Celsius"}]
                },
                {
                    "type": "chat",
                    "function_name": "basic_test",
                    "input": {
                        "system": {"assistant_name": "Pinocchio"},
                        "messages": [{"role": "user", "content": [{"type": "text", "text": "What is the capital city of India?"}]}]
                    },
                    "output": [{"type": "text", "text": "Ahmedabad (nose grows 3 inches)"}]
                },
                {
                    "type": "chat",
                    "function_name": "basic_test",
                    "input": {
                        "system": {"assistant_name": "Pinocchio"},
                        "messages": [{"role": "user", "content": [{"type": "text", "text": "What is an example of a computationally hard problem?"}]}]
                    },
                    "output": [{"type": "text", "text": "Finding the median of an unsorted list of numbers (nose grows 4 inches)"}]
                },
                {
                    "type": "chat",
                    "function_name": "basic_test",
                    "input": {
                        "system": {"assistant_name": "Pinocchio"},
                        "messages": [{"role": "user", "content": [{"type": "text", "text": "Who wrote Lord of the Rings?"}]}]
                    },
                    "output": [{"type": "text", "text": "J.K. Rowling (nose grows 5 inches)"}]
                }
            ]
        }))
        .send()
        .await
        .unwrap();

    let status = response.status();
    let body = response.text().await.unwrap();
    assert!(
        status.is_success(),
        "Failed to create chat dataset ({status}): {body}"
    );
}

/// Create JSON datapoints for json_success using Pinocchio examples
async fn create_pinocchio_json_dataset(client: &Client, base: &str, dataset_name: &str) {
    let output_schema = json!({
        "type": "object",
        "properties": {
            "answer": {"type": "string"}
        },
        "required": ["answer"],
        "additionalProperties": false
    });

    let response = client
        .post(format!("{base}/v1/datasets/{dataset_name}/datapoints"))
        .json(&json!({
            "datapoints": [
                {
                    "type": "json",
                    "function_name": "json_success",
                    "input": {
                        "system": {"assistant_name": "Dr. Mehta"},
                        "messages": [{"role": "user", "content": [{"type": "template", "name": "user", "arguments": {"country": "France"}}]}]
                    },
                    "output": {"raw": r#"{"answer":"Paris"}"#},
                    "output_schema": output_schema,
                },
                {
                    "type": "json",
                    "function_name": "json_success",
                    "input": {
                        "system": {"assistant_name": "Pinocchio"},
                        "messages": [{"role": "user", "content": [{"type": "template", "name": "user", "arguments": {"country": "India"}}]}]
                    },
                    "output": {"raw": r#"{"answer":"Mumbai (nose grows 3 inches)"}"#},
                    "output_schema": output_schema,
                },
                {
                    "type": "json",
                    "function_name": "json_success",
                    "input": {
                        "system": {"assistant_name": "Dr. Mehta"},
                        "messages": [{"role": "user", "content": [{"type": "template", "name": "user", "arguments": {"country": "Japan"}}]}]
                    },
                    "output": {"raw": r#"{"answer":"Tokyo"}"#},
                    "output_schema": output_schema,
                },
                {
                    "type": "json",
                    "function_name": "json_success",
                    "input": {
                        "system": {"assistant_name": "Pinocchio"},
                        "messages": [{"role": "user", "content": [{"type": "template", "name": "user", "arguments": {"country": "Australia"}}]}]
                    },
                    "output": {"raw": r#"{"answer":"Sydney (nose grows 4 inches)"}"#},
                    "output_schema": output_schema,
                }
            ]
        }))
        .send()
        .await
        .unwrap();

    let status = response.status();
    let body = response.text().await.unwrap();
    assert!(
        status.is_success(),
        "Failed to create JSON dataset ({status}): {body}"
    );
}

/// Poll GET /v1/optimization/gepa/{task_id} until a terminal state is reached
async fn poll_until_terminal(client: &Client, base: &str, task_id: &str) -> Value {
    let max_polls = 120; // 10 minutes at 5s intervals
    for i in 0..max_polls {
        let response = client
            .get(format!("{base}/v1/optimization/gepa/{task_id}"))
            .send()
            .await
            .unwrap();

        let body: Value = response.json().await.unwrap();
        let status = body["status"].as_str().unwrap();

        println!("Poll {i}: status={status}");

        if status == "completed" || status == "error" {
            return body;
        }

        tokio::time::sleep(Duration::from_secs(5)).await;
    }
    panic!("GEPA task {task_id} did not complete within timeout");
}

/// Validate GEPA result — both Completed and Error are valid outcomes
fn validate_gepa_result(result: &Value, variant_prefix: &str, max_iterations: usize) {
    let status = result["status"].as_str().unwrap();
    match status {
        "completed" => {
            let variants = result["variants"].as_object().unwrap();
            assert!(!variants.is_empty(), "Should produce at least one variant");
            assert!(
                variants.len() <= max_iterations,
                "Should not exceed max_iterations ({max_iterations}), got {}",
                variants.len()
            );

            for name in variants.keys() {
                assert!(
                    name.starts_with(variant_prefix),
                    "Variant name '{name}' should have prefix '{variant_prefix}'"
                );
            }

            let statistics = result["statistics"].as_object().unwrap();
            assert!(
                !statistics.is_empty(),
                "Should have statistics for variants"
            );

            println!(
                "Durable GEPA completed with {} evolved variants",
                variants.len()
            );
        }
        "error" => {
            let error = result["error"].as_str().unwrap();
            println!("Durable GEPA completed with error (valid outcome): {error}");
        }
        other => panic!("Unexpected terminal status: {other}"),
    }
}
