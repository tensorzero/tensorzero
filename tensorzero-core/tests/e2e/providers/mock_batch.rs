// ============================================================================
// UNIFIED BATCH INFERENCE TESTS (Mocked Provider Only)
// ============================================================================
// These tests do launch + poll + complete in one test function.
// They are designed for the mock inference provider which auto-completes
// batches in ~2 seconds, allowing for fast, isolated testing.
//
// Only run with TENSORZERO_USE_MOCK_INFERENCE_PROVIDER=1
//
// These tests use unique identifiers (UUIDs) for each test run to ensure
// test isolation and allow parallel execution without conflicts.
//
// For live traffic tests (real OpenAI/GCP APIs that take hours), continue
// using the 3-part pattern (test_start_*, test_poll_existing_*, test_poll_completed_*)
// ============================================================================

use std::collections::HashMap;

use http::StatusCode;
use reqwest::Client;
use serde_json::{json, Value};
use std::time::Duration;
use tensorzero_core::{
    db::clickhouse::test_helpers::get_clickhouse, endpoints::batch_inference::PollPathParams,
};
use tokio::time::sleep;
use uuid::Uuid;

use crate::{
    common::get_gateway_endpoint,
    providers::{
        batch::{check_clickhouse_batch_request_status, get_poll_batch_inference_url},
        common::{
            check_json_mode_inference_response, check_parallel_tool_use_inference_response,
            check_simple_image_inference_response,
            check_tool_use_tool_choice_auto_unused_inference_response,
            check_tool_use_tool_choice_auto_used_inference_response,
            check_tool_use_tool_choice_none_inference_response,
            check_tool_use_tool_choice_required_inference_response,
            check_tool_use_tool_choice_specific_inference_response, E2ETestProvider,
        },
    },
};

/// Helper function to check if we're using the mock inference provider
fn is_using_mock_provider() -> bool {
    std::env::var("TENSORZERO_USE_MOCK_INFERENCE_PROVIDER").is_ok()
}

/// Unified test for simple image batch inference (OpenAI)
/// This test launches a batch, waits for it to complete, polls the result, and verifies everything in one go.
#[tokio::test]
async fn test_simple_image_batch_inference_openai_mock_unified() {
    if !is_using_mock_provider() {
        println!("Skipping unified mock test - TENSORZERO_USE_MOCK_INFERENCE_PROVIDER not set");
        return;
    }

    // Use unique episode_id for test isolation
    let episode_id = Uuid::now_v7();
    let test_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test_mock_batch",
        "variant_name": "openai",
        "episode_ids": [episode_id],
        "inputs":
            [{
               "system": {"assistant_name": "Dr. Mehta"},
               "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What kind of animal is in this image?"},
                        {
                            "type": "image",
                            "url": "https://raw.githubusercontent.com/tensorzero/tensorzero/ff3e17bbd3e32f483b027cf81b54404788c90dc1/tensorzero-internal/tests/e2e/providers/ferris.png"
                        },
                    ]
                },
            ]}],
        "tags": [{"test_type": "unified_mock", "test_id": test_id.to_string()}],
    });

    // Step 1: Launch the batch
    let response = Client::new()
        .post(get_gateway_endpoint("/batch_inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("Launch response: {response_json:#?}");

    let batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let batch_id = Uuid::parse_str(batch_id).unwrap();

    let inference_ids = response_json
        .get("inference_ids")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(inference_ids.len(), 1);

    // Step 2: Wait for the mock provider to complete the batch (~2 seconds)
    println!("Waiting for batch to complete...");
    sleep(Duration::from_secs(3)).await;

    // Step 3: Poll the batch and verify it's completed
    let url = get_poll_batch_inference_url(PollPathParams {
        batch_id,
        inference_id: None,
    });
    let response = Client::new().get(url).send().await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("Poll response: {response_json:#?}");

    // Verify the batch is completed
    assert_eq!(
        response_json.get("status").unwrap().as_str().unwrap(),
        "completed",
        "Batch should be completed after waiting"
    );

    let returned_batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let returned_batch_id = Uuid::parse_str(returned_batch_id).unwrap();
    assert_eq!(returned_batch_id, batch_id);

    // Step 4: Verify the inference response
    let inferences_json = response_json.get("inferences").unwrap().as_array().unwrap();
    assert_eq!(inferences_json.len(), 1);

    let provider = E2ETestProvider {
        variant_name: "openai".to_string(),
        model_name: "gpt-4o-mini-2024-07-18".to_string(),
        model_provider_name: "openai".to_string(),
        credentials: HashMap::new(),
        supports_batch_inference: true,
    };

    check_simple_image_inference_response(inferences_json[0].clone(), None, &provider, true, false)
        .await;

    // Step 5: Verify ClickHouse storage
    let clickhouse = get_clickhouse().await;
    check_clickhouse_batch_request_status(&clickhouse, batch_id, &provider, "completed").await;

    println!("✓ Unified mock test for simple image (OpenAI) completed successfully");
}

/// Unified test for simple image batch inference (GCP Vertex Gemini)
/// This test launches a batch, waits for it to complete, polls the result, and verifies everything in one go.
#[tokio::test]
async fn test_simple_image_batch_inference_gcp_mock_unified() {
    if !is_using_mock_provider() {
        println!("Skipping unified mock test - TENSORZERO_USE_MOCK_INFERENCE_PROVIDER not set");
        return;
    }

    // Use unique episode_id for test isolation
    let episode_id = Uuid::now_v7();
    let test_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test_mock_batch",
        "variant_name": "gcp-vertex-gemini-flash",
        "episode_ids": [episode_id],
        "inputs":
            [{
               "system": {"assistant_name": "Dr. Mehta"},
               "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What kind of animal is in this image?"},
                        {
                            "type": "image",
                            "url": "https://raw.githubusercontent.com/tensorzero/tensorzero/ff3e17bbd3e32f483b027cf81b54404788c90dc1/tensorzero-internal/tests/e2e/providers/ferris.png"
                        },
                    ]
                },
            ]}],
        "tags": [{"test_type": "unified_mock", "test_id": test_id.to_string()}],
    });

    // Step 1: Launch the batch
    let response = Client::new()
        .post(get_gateway_endpoint("/batch_inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("Launch response: {response_json:#?}");

    let batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let batch_id = Uuid::parse_str(batch_id).unwrap();

    let inference_ids = response_json
        .get("inference_ids")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(inference_ids.len(), 1);

    // Step 2: Wait for the mock provider to complete the batch (~2 seconds)
    println!("Waiting for batch to complete...");
    sleep(Duration::from_secs(3)).await;

    // Step 3: Poll the batch and verify it's completed
    let url = get_poll_batch_inference_url(PollPathParams {
        batch_id,
        inference_id: None,
    });
    let response = Client::new().get(url).send().await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("Poll response: {response_json:#?}");

    // Verify the batch is completed
    assert_eq!(
        response_json.get("status").unwrap().as_str().unwrap(),
        "completed",
        "Batch should be completed after waiting"
    );

    let returned_batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let returned_batch_id = Uuid::parse_str(returned_batch_id).unwrap();
    assert_eq!(returned_batch_id, batch_id);

    // Step 4: Verify the inference response
    let inferences_json = response_json.get("inferences").unwrap().as_array().unwrap();
    assert_eq!(inferences_json.len(), 1);

    let provider = E2ETestProvider {
        variant_name: "gcp-vertex-gemini-flash".to_string(),
        model_name: "gemini-2.0-flash-001".to_string(),
        model_provider_name: "gcp_vertex_gemini".to_string(),
        credentials: HashMap::new(),
        supports_batch_inference: true,
    };

    check_simple_image_inference_response(inferences_json[0].clone(), None, &provider, true, false)
        .await;

    // Step 5: Verify ClickHouse storage
    let clickhouse = get_clickhouse().await;
    check_clickhouse_batch_request_status(&clickhouse, batch_id, &provider, "completed").await;

    println!("✓ Unified mock test for simple image (GCP) completed successfully");
}

/// Unified test for JSON mode batch inference (OpenAI)
#[tokio::test]
async fn test_json_mode_batch_inference_openai_mock_unified() {
    if !is_using_mock_provider() {
        println!("Skipping unified mock test - TENSORZERO_USE_MOCK_INFERENCE_PROVIDER not set");
        return;
    }

    let episode_id = Uuid::now_v7();
    let test_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "json_success_mock_batch",
        "variant_name": "openai",
        "episode_ids": [episode_id],
        "inputs": [{
            "system": {"assistant_name": "Dr. Mehta"},
               "messages": [
                {
                    "role": "user",
                    "content": [{"type": "template", "name": "user", "arguments": {"country": "Japan"}}]
                }
            ]}],
        "tags": [{"test_type": "unified_mock", "test_id": test_id.to_string()}]
    });

    // Step 1: Launch the batch
    let response = Client::new()
        .post(get_gateway_endpoint("/batch_inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("Launch response: {response_json:#?}");

    let batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let batch_id = Uuid::parse_str(batch_id).unwrap();

    // Step 2: Wait for the mock provider to complete the batch
    println!("Waiting for batch to complete...");
    sleep(Duration::from_secs(3)).await;

    // Step 3: Poll the batch and verify it's completed
    let url = get_poll_batch_inference_url(PollPathParams {
        batch_id,
        inference_id: None,
    });
    let response = Client::new().get(url).send().await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("Poll response: {response_json:#?}");

    assert_eq!(
        response_json.get("status").unwrap().as_str().unwrap(),
        "completed"
    );

    // Step 4: Verify the inference response
    let inferences_json = response_json.get("inferences").unwrap().as_array().unwrap();
    assert_eq!(inferences_json.len(), 1);

    let provider = E2ETestProvider {
        variant_name: "openai".to_string(),
        model_name: "gpt-4o-mini-2024-07-18".to_string(),
        model_provider_name: "openai".to_string(),
        credentials: HashMap::new(),
        supports_batch_inference: true,
    };

    check_json_mode_inference_response(inferences_json[0].clone(), &provider, None, true).await;

    let clickhouse = get_clickhouse().await;
    check_clickhouse_batch_request_status(&clickhouse, batch_id, &provider, "completed").await;

    println!("✓ Unified mock test for JSON mode (OpenAI) completed successfully");
}

/// Unified test for JSON mode batch inference (GCP Vertex)
#[tokio::test]
async fn test_json_mode_batch_inference_gcp_mock_unified() {
    if !is_using_mock_provider() {
        println!("Skipping unified mock test - TENSORZERO_USE_MOCK_INFERENCE_PROVIDER not set");
        return;
    }

    let episode_id = Uuid::now_v7();
    let test_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "json_success_mock_batch",
        "variant_name": "gcp-vertex-gemini-flash",
        "episode_ids": [episode_id],
        "inputs": [{
            "system": {"assistant_name": "Dr. Mehta"},
               "messages": [
                {
                    "role": "user",
                    "content": [{"type": "template", "name": "user", "arguments": {"country": "Japan"}}]
                }
            ]}],
        "tags": [{"test_type": "unified_mock", "test_id": test_id.to_string()}]
    });

    // Step 1: Launch the batch
    let response = Client::new()
        .post(get_gateway_endpoint("/batch_inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("Launch response: {response_json:#?}");

    let batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let batch_id = Uuid::parse_str(batch_id).unwrap();

    // Step 2: Wait for the mock provider to complete the batch
    println!("Waiting for batch to complete...");
    sleep(Duration::from_secs(3)).await;

    // Step 3: Poll the batch and verify it's completed
    let url = get_poll_batch_inference_url(PollPathParams {
        batch_id,
        inference_id: None,
    });
    let response = Client::new().get(url).send().await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("Poll response: {response_json:#?}");

    assert_eq!(
        response_json.get("status").unwrap().as_str().unwrap(),
        "completed"
    );

    // Step 4: Verify the inference response
    let inferences_json = response_json.get("inferences").unwrap().as_array().unwrap();
    assert_eq!(inferences_json.len(), 1);

    let provider = E2ETestProvider {
        variant_name: "gcp-vertex-gemini-flash".to_string(),
        model_name: "gemini-2.0-flash-001".to_string(),
        model_provider_name: "gcp_vertex_gemini".to_string(),
        credentials: HashMap::new(),
        supports_batch_inference: true,
    };

    check_json_mode_inference_response(inferences_json[0].clone(), &provider, None, true).await;

    let clickhouse = get_clickhouse().await;
    check_clickhouse_batch_request_status(&clickhouse, batch_id, &provider, "completed").await;

    println!("✓ Unified mock test for JSON mode (GCP) completed successfully");
}

/// Unified test for tool use batch inference (OpenAI)
#[tokio::test]
async fn test_tool_use_batch_inference_openai_mock_unified() {
    if !is_using_mock_provider() {
        println!("Skipping unified mock test - TENSORZERO_USE_MOCK_INFERENCE_PROVIDER not set");
        return;
    }

    let mut episode_ids = Vec::new();
    for _ in 0..5 {
        episode_ids.push(Uuid::now_v7());
    }
    let test_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "weather_helper_mock_batch",
        "episode_ids": episode_ids,
        "inputs":
            [{
               "system": {"assistant_name": "Dr. Mehta"},
               "messages": [
                {
                    "role": "user",
                    "content": "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                }
            ]},
            {
                "system": {"assistant_name": "Dr. Mehta"},
                "messages": [
                 {
                     "role": "user",
                     "content": "What is your name?"
                 }
             ]},
             {
                "system": { "assistant_name": "Dr. Mehta" },
                "messages": [
                    {
                        "role": "user",
                        "content": "What is your name?"
                    }
                ]
            },
            {
                "system": {"assistant_name": "Dr. Mehta"},
                "messages": [
                    {
                        "role": "user",
                        "content": "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                    }
                ]},
                {
                    "system": {"assistant_name": "Dr. Mehta"},
                    "messages": [
                        {
                            "role": "user",
                            "content": "What is the temperature like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                        }
                    ]}
        ],
        "variant_name": "openai",
        "tags": [{"test_type": "unified_mock", "test_id": test_id.to_string()}]
    });

    // Step 1: Launch the batch
    let response = Client::new()
        .post(get_gateway_endpoint("/batch_inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("Launch response: {response_json:#?}");

    let batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let batch_id = Uuid::parse_str(batch_id).unwrap();

    // Step 2: Wait for the mock provider to complete the batch
    println!("Waiting for batch to complete...");
    sleep(Duration::from_secs(3)).await;

    // Step 3: Poll the batch and verify it's completed
    let url = get_poll_batch_inference_url(PollPathParams {
        batch_id,
        inference_id: None,
    });
    let response = Client::new().get(url).send().await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("Poll response: {response_json:#?}");

    assert_eq!(
        response_json.get("status").unwrap().as_str().unwrap(),
        "completed"
    );

    // Step 4: Verify the inference responses
    let inferences_json = response_json.get("inferences").unwrap().as_array().unwrap();
    assert_eq!(inferences_json.len(), 5);

    let provider = E2ETestProvider {
        variant_name: "openai".to_string(),
        model_name: "gpt-4o-mini-2024-07-18".to_string(),
        model_provider_name: "openai".to_string(),
        credentials: HashMap::new(),
        supports_batch_inference: true,
    };

    // Check tool choice required (should use tool)
    check_tool_use_tool_choice_required_inference_response(
        inferences_json[0].clone(),
        &provider,
        None,
        true,
    )
    .await;

    // Check tool choice auto unused (should not use tool)
    check_tool_use_tool_choice_auto_unused_inference_response(
        inferences_json[1].clone(),
        &provider,
        None,
        true,
    )
    .await;

    // Check tool choice none (should not use tool)
    check_tool_use_tool_choice_none_inference_response(
        inferences_json[2].clone(),
        &provider,
        None,
        true,
    )
    .await;

    // Check tool choice auto used (should use tool)
    check_tool_use_tool_choice_auto_used_inference_response(
        inferences_json[3].clone(),
        &provider,
        None,
        true,
    )
    .await;

    // Check tool choice specific (should use specific tool)
    check_tool_use_tool_choice_specific_inference_response(
        inferences_json[4].clone(),
        &provider,
        None,
        true,
    )
    .await;

    let clickhouse = get_clickhouse().await;
    check_clickhouse_batch_request_status(&clickhouse, batch_id, &provider, "completed").await;

    println!("✓ Unified mock test for tool use completed successfully");
}

/// Unified test for parallel tool use batch inference (OpenAI)
#[tokio::test]
async fn test_parallel_tool_use_batch_inference_openai_mock_unified() {
    if !is_using_mock_provider() {
        println!("Skipping unified mock test - TENSORZERO_USE_MOCK_INFERENCE_PROVIDER not set");
        return;
    }

    let episode_id = Uuid::now_v7();
    let test_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "weather_helper_mock_batch",
        "episode_ids": [episode_id],
        "inputs": [{
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather in Tokyo and San Francisco? Use the `get_temperature` tool (call it twice)."
                }
            ]
        }],
        "variant_name": "openai",
        "tags": [{"test_type": "unified_mock", "test_id": test_id.to_string()}]
    });

    // Step 1: Launch the batch
    let response = Client::new()
        .post(get_gateway_endpoint("/batch_inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("Launch response: {response_json:#?}");

    let batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let batch_id = Uuid::parse_str(batch_id).unwrap();

    // Step 2: Wait for the mock provider to complete the batch
    println!("Waiting for batch to complete...");
    sleep(Duration::from_secs(3)).await;

    // Step 3: Poll the batch and verify it's completed
    let url = get_poll_batch_inference_url(PollPathParams {
        batch_id,
        inference_id: None,
    });
    let response = Client::new().get(url).send().await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("Poll response: {response_json:#?}");

    assert_eq!(
        response_json.get("status").unwrap().as_str().unwrap(),
        "completed"
    );

    // Step 4: Verify the inference response
    let inferences_json = response_json.get("inferences").unwrap().as_array().unwrap();
    assert_eq!(inferences_json.len(), 1);

    let provider = E2ETestProvider {
        variant_name: "openai".to_string(),
        model_name: "gpt-4o-mini-2024-07-18".to_string(),
        model_provider_name: "openai".to_string(),
        credentials: HashMap::new(),
        supports_batch_inference: true,
    };

    check_parallel_tool_use_inference_response(
        inferences_json[0].clone(),
        &provider,
        None,
        true,
        true.into(),
    )
    .await;

    let clickhouse = get_clickhouse().await;
    check_clickhouse_batch_request_status(&clickhouse, batch_id, &provider, "completed").await;

    println!("✓ Unified mock test for parallel tool use (OpenAI) completed successfully");
}

/// Unified test for parallel tool use batch inference (GCP Vertex)
#[tokio::test]
async fn test_parallel_tool_use_batch_inference_gcp_mock_unified() {
    if !is_using_mock_provider() {
        println!("Skipping unified mock test - TENSORZERO_USE_MOCK_INFERENCE_PROVIDER not set");
        return;
    }

    let episode_id = Uuid::now_v7();
    let test_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "weather_helper_mock_batch",
        "episode_ids": [episode_id],
        "inputs": [{
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather in Tokyo and San Francisco? Use the `get_temperature` tool (call it twice)."
                }
            ]
        }],
        "variant_name": "gcp-vertex-gemini-flash",
        "tags": [{"test_type": "unified_mock", "test_id": test_id.to_string()}]
    });

    // Step 1: Launch the batch
    let response = Client::new()
        .post(get_gateway_endpoint("/batch_inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("Launch response: {response_json:#?}");

    let batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let batch_id = Uuid::parse_str(batch_id).unwrap();

    // Step 2: Wait for the mock provider to complete the batch
    println!("Waiting for batch to complete...");
    sleep(Duration::from_secs(3)).await;

    // Step 3: Poll the batch and verify it's completed
    let url = get_poll_batch_inference_url(PollPathParams {
        batch_id,
        inference_id: None,
    });
    let response = Client::new().get(url).send().await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("Poll response: {response_json:#?}");

    assert_eq!(
        response_json.get("status").unwrap().as_str().unwrap(),
        "completed"
    );

    // Step 4: Verify the inference response
    let inferences_json = response_json.get("inferences").unwrap().as_array().unwrap();
    assert_eq!(inferences_json.len(), 1);

    let provider = E2ETestProvider {
        variant_name: "gcp-vertex-gemini-flash".to_string(),
        model_name: "gemini-2.0-flash-001".to_string(),
        model_provider_name: "gcp_vertex_gemini".to_string(),
        credentials: HashMap::new(),
        supports_batch_inference: true,
    };

    check_parallel_tool_use_inference_response(
        inferences_json[0].clone(),
        &provider,
        None,
        true,
        true.into(),
    )
    .await;

    let clickhouse = get_clickhouse().await;
    check_clickhouse_batch_request_status(&clickhouse, batch_id, &provider, "completed").await;

    println!("✓ Unified mock test for parallel tool use (GCP) completed successfully");
}
