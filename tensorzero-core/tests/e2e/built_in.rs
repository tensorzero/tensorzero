#![expect(clippy::print_stdout)]
//! E2E tests for built-in TensorZero functions
//!
//! Built-in functions are variant-less and require `internal_dynamic_variant_config`
//! to provide the variant configuration at runtime.
//!
//! ## Supported Scenarios
//! - ChatCompletion variants with templates and system variables
//! - JSON functions with ChatCompletion variants
//!
//! ## Unsupported Scenarios (not tested)
//! - Complex variant types (Dicl, MixtureOfN, BestOfNSampling) - require predefined variants
//! - Batch inference - not supported for built-in functions
//! - Experimentation/sampling - requires multiple variants

use crate::common::get_gateway_endpoint;
use reqwest::{Client, StatusCode};
use serde_json::{json, Value};
use tensorzero_core::db::clickhouse::test_helpers::get_clickhouse;
use tensorzero_core::db::clickhouse::test_helpers::{
    select_chat_inference_clickhouse, select_json_inference_clickhouse,
};
use uuid::Uuid;

#[tokio::test]
async fn e2e_test_built_in_hello_chat_with_system_variables() {
    // Test calling the built-in tensorzero::hello_chat function with system template variables
    // using a dynamic variant config
    let mut payload = json!({
        "function_name": "tensorzero::hello_chat",
        "episode_id": Uuid::now_v7(),
        "input": {
            "system": {"greeting": "Howdy"},
            "messages": [
                {
                    "role": "user",
                    "content": "What's your name?"
                }
            ]
        },
        "internal_dynamic_variant_config": {
            "type": "chat_completion",
            "weight": 0.0,
            "model": "dummy::echo_request_messages",
            "system_template": {
                "__tensorzero_remapped_path": "system",
                "__data": "{{ greeting }}! I am a built-in TensorZero chat function."
            }
        },
        "stream": false,
    });

    // Should fail without dryrun
    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    // Now try with dryrun enabled
    payload["dryrun"] = json!(true);
    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check Response is OK
    let status = response.status();
    let body = response.text().await.unwrap();
    println!("Response status: {status}");
    println!("Response body: {body}");
    assert_eq!(status, StatusCode::OK, "Unexpected response: {body}");
    let response_json: Value = serde_json::from_str(&body).unwrap();
    println!("{response_json:#}");

    // Verify the response contains our system template with variables rendered
    let content = response_json["content"].as_array().unwrap();
    let text_block = content.first().unwrap();
    let text = text_block["text"].as_str().unwrap();
    assert!(text.contains("Howdy! I am a built-in TensorZero chat function."));

    // Check that inference_id is present
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    // Sleep for 1 second to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ClickHouse - no inference should be written when dryrun is true
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id).await;
    assert!(result.is_none());
}

#[tokio::test]
async fn e2e_test_built_in_hello_json() {
    // Test calling the built-in tensorzero::hello_json function
    let mut payload = json!({
        "function_name": "tensorzero::hello_json",
        "episode_id": Uuid::now_v7(),
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": "Generate a greeting"
                }
            ]
        },
        "internal_dynamic_variant_config": {
            "type": "chat_completion",
            "weight": 0.0,
            "model": "dummy::echo_request_messages",
            "system_template": {
                "__tensorzero_remapped_path": "system",
                "__data": "You are a helpful assistant that generates JSON responses."
            }
        },
        "stream": false,
    });

    // Should fail without dryrun
    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    // Now try with dryrun enabled
    payload["dryrun"] = json!(true);
    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check Response is OK
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("{response_json:#}");

    // For JSON functions, the response should have an "output" field instead of "content"
    assert!(response_json.get("output").is_some());

    // Check that inference_id is present
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    // Sleep for 1 second to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ClickHouse - no inference should be written when dryrun is true
    let clickhouse = get_clickhouse().await;
    let result = select_json_inference_clickhouse(&clickhouse, inference_id).await;
    assert!(result.is_none());
}

#[tokio::test]
async fn e2e_test_built_in_error_no_variant() {
    // Test that built-in functions fail gracefully without dynamic variant config
    let payload = json!({
        "function_name": "tensorzero::hello_chat",
        "episode_id": Uuid::now_v7(),
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": "Hello"
                }
            ]
        },
        "stream": false,
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Should return 500 with error about no variants
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let error_json = response.json::<Value>().await.unwrap();
    let error_msg = error_json["error"].as_str().unwrap();
    assert!(error_msg.contains("has no variants"));
}
