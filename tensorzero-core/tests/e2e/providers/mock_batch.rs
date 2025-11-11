// ============================================================================
// UNIFIED BATCH INFERENCE TESTS (Mocked Provider Only)
// ============================================================================
// These tests do launch + poll + complete in one test function.
// They are designed for the mock inference provider which auto-completes
// batches in ~2 seconds, allowing for fast, isolated testing.
//
// These tests require the mock inference provider to be running at localhost:3030
// (started via docker compose -f tensorzero-core/tests/e2e/docker-compose.yml up --wait)
//
// These tests use unique identifiers (UUIDs) for each test run to ensure
// test isolation and allow parallel execution without conflicts.
//
// For live traffic tests (real OpenAI/GCP APIs that take hours), continue
// using the 3-part pattern (test_start_*, test_poll_existing_*, test_poll_completed_*)
// ============================================================================

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
            check_json_mode_inference_response_with_function_name,
            check_parallel_tool_use_inference_response_with_function_name,
            check_simple_image_inference_response_with_function_name,
            check_tool_use_tool_choice_auto_unused_inference_response_with_function_name,
            check_tool_use_tool_choice_auto_used_inference_response_with_function_name,
            check_tool_use_tool_choice_none_inference_response_with_function_name,
            check_tool_use_tool_choice_required_inference_response_with_function_name,
            check_tool_use_tool_choice_specific_inference_response_with_function_name,
            E2ETestProvider,
        },
    },
};

/// Poll for batch completion with retries
/// Polls up to max_attempts times with delay_ms between attempts
async fn poll_until_completed(
    batch_id: Uuid,
    max_attempts: u32,
    delay_ms: u64,
) -> Result<Value, String> {
    let url = get_poll_batch_inference_url(PollPathParams {
        batch_id,
        inference_id: None,
    });

    for attempt in 1..=max_attempts {
        let response = Client::new()
            .get(url.clone())
            .send()
            .await
            .map_err(|e| format!("Failed to send request: {}", e))?;

        if response.status() != StatusCode::OK {
            return Err(format!("Expected status 200, got {}", response.status()));
        }

        let response_json = response
            .json::<Value>()
            .await
            .map_err(|e| format!("Failed to parse JSON: {}", e))?;

        let status = response_json
            .get("status")
            .and_then(|s| s.as_str())
            .ok_or_else(|| "No status field in response".to_string())?;

        println!(
            "Poll attempt {}/{}: status = {}",
            attempt, max_attempts, status
        );

        if status == "completed" {
            return Ok(response_json);
        }

        if attempt < max_attempts {
            sleep(Duration::from_millis(delay_ms)).await;
        }
    }

    Err(format!(
        "Batch did not complete after {} attempts ({} seconds)",
        max_attempts,
        (max_attempts as u64 * delay_ms) / 1000
    ))
}

/// Macro to generate unified mock batch tests for a provider
///
/// This macro generates test functions that:
/// 1. Call the provider function to get available providers
/// 2. Filter to only batch-supporting providers
/// 3. Iterate through each provider and call the appropriate helper function
///
/// Usage:
/// ```
/// crate::generate_unified_mock_batch_tests!(get_providers);
/// ```
#[macro_export]
macro_rules! generate_unified_mock_batch_tests {
    ($func:ident) => {
        // Import the helper functions
        use $crate::providers::mock_batch::{
            test_json_mode_unified_mock_batch_with_provider,
            test_parallel_tool_use_unified_mock_batch_with_provider,
            test_simple_image_unified_mock_batch_with_provider,
            test_tool_use_unified_mock_batch_with_provider,
        };

        /// Test simple image batch inference with mock provider (unified: launch + poll + verify)
        #[tokio::test]
        async fn test_simple_image_unified_mock_batch_inference() {
            let all_providers = $func().await;
            let providers = all_providers.simple_inference;
            for provider in providers {
                if provider.supports_batch_inference {
                    test_simple_image_unified_mock_batch_with_provider(
                        &provider,
                        "basic_test_mock_batch",
                    )
                    .await;
                }
            }
        }

        /// Test JSON mode batch inference with mock provider (unified: launch + poll + verify)
        #[tokio::test]
        async fn test_json_mode_unified_mock_batch_inference() {
            let all_providers = $func().await;
            let providers = all_providers.json_mode_inference;
            for provider in providers {
                if provider.supports_batch_inference {
                    test_json_mode_unified_mock_batch_with_provider(
                        &provider,
                        "json_success_mock_batch",
                    )
                    .await;
                }
            }
        }

        /// Test tool use batch inference with mock provider (unified: launch + poll + verify)
        #[tokio::test]
        async fn test_tool_use_unified_mock_batch_inference() {
            let all_providers = $func().await;
            let providers = all_providers.tool_use_inference;
            for provider in providers {
                if provider.supports_batch_inference {
                    test_tool_use_unified_mock_batch_with_provider(
                        &provider,
                        "weather_helper_mock_batch",
                    )
                    .await;
                }
            }
        }

        /// Test parallel tool use batch inference with mock provider (unified: launch + poll + verify)
        #[tokio::test]
        async fn test_parallel_tool_use_unified_mock_batch_inference() {
            let all_providers = $func().await;
            let providers = all_providers.parallel_tool_use_inference;
            for provider in providers {
                if provider.supports_batch_inference {
                    test_parallel_tool_use_unified_mock_batch_with_provider(
                        &provider,
                        "weather_helper_mock_batch",
                    )
                    .await;
                }
            }
        }
    };
}

/// Helper function for testing simple image batch inference
pub async fn test_simple_image_unified_mock_batch_with_provider(
    provider: &E2ETestProvider,
    function_name: &str,
) {
    // Use unique episode_id for test isolation
    let episode_id = Uuid::now_v7();
    let test_id = Uuid::now_v7();

    let payload = json!({
        "function_name": function_name,
        "variant_name": provider.variant_name,
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

    // Step 2: Poll until the batch completes (up to 10 seconds)
    println!("Polling for batch completion...");
    let response_json = poll_until_completed(batch_id, 20, 500)
        .await
        .expect("Batch should complete within 10 seconds");

    println!("Batch completed: {response_json:#?}");

    let returned_batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let returned_batch_id = Uuid::parse_str(returned_batch_id).unwrap();
    assert_eq!(returned_batch_id, batch_id);

    // Step 3: Verify the inference response
    let inferences_json = response_json.get("inferences").unwrap().as_array().unwrap();
    assert_eq!(inferences_json.len(), 1);

    check_simple_image_inference_response_with_function_name(
        inferences_json[0].clone(),
        None,
        provider,
        true,
        false,
        function_name,
    )
    .await;

    // Step 4: Verify ClickHouse storage
    let clickhouse = get_clickhouse().await;
    check_clickhouse_batch_request_status(&clickhouse, batch_id, provider, "completed").await;

    println!(
        "✓ Unified mock test for simple image ({}) completed successfully",
        provider.variant_name
    );
}

/// Helper function for testing JSON mode batch inference
pub async fn test_json_mode_unified_mock_batch_with_provider(
    provider: &E2ETestProvider,
    function_name: &str,
) {
    let episode_id = Uuid::now_v7();
    let test_id = Uuid::now_v7();

    let payload = json!({
        "function_name": function_name,
        "variant_name": provider.variant_name,
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

    check_json_mode_inference_response_with_function_name(
        inferences_json[0].clone(),
        provider,
        None,
        true,
        function_name,
    )
    .await;

    let clickhouse = get_clickhouse().await;
    check_clickhouse_batch_request_status(&clickhouse, batch_id, provider, "completed").await;

    println!(
        "✓ Unified mock test for JSON mode ({}) completed successfully",
        provider.variant_name
    );
}

/// Helper function for testing tool use batch inference
pub async fn test_tool_use_unified_mock_batch_with_provider(
    provider: &E2ETestProvider,
    function_name: &str,
) {
    let mut episode_ids = Vec::new();
    for _ in 0..5 {
        episode_ids.push(Uuid::now_v7());
    }
    let test_id = Uuid::now_v7();

    let payload = json!({
        "function_name": function_name,
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
        "variant_name": provider.variant_name,
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

    // Check tool choice required (should use tool)
    check_tool_use_tool_choice_required_inference_response_with_function_name(
        inferences_json[0].clone(),
        provider,
        None,
        true,
        function_name,
    )
    .await;

    // Check tool choice auto unused (should not use tool)
    check_tool_use_tool_choice_auto_unused_inference_response_with_function_name(
        inferences_json[1].clone(),
        provider,
        None,
        true,
        function_name,
    )
    .await;

    // Check tool choice none (should not use tool)
    check_tool_use_tool_choice_none_inference_response_with_function_name(
        inferences_json[2].clone(),
        provider,
        None,
        true,
        function_name,
    )
    .await;

    // Check tool choice auto used (should use tool)
    check_tool_use_tool_choice_auto_used_inference_response_with_function_name(
        inferences_json[3].clone(),
        provider,
        None,
        true,
        function_name,
    )
    .await;

    // Check tool choice specific (should use specific tool)
    check_tool_use_tool_choice_specific_inference_response_with_function_name(
        inferences_json[4].clone(),
        provider,
        None,
        true,
        function_name,
    )
    .await;

    let clickhouse = get_clickhouse().await;
    check_clickhouse_batch_request_status(&clickhouse, batch_id, provider, "completed").await;

    println!(
        "✓ Unified mock test for tool use ({}) completed successfully",
        provider.variant_name
    );
}

/// Helper function for testing parallel tool use batch inference
pub async fn test_parallel_tool_use_unified_mock_batch_with_provider(
    provider: &E2ETestProvider,
    function_name: &str,
) {
    let episode_id = Uuid::now_v7();
    let test_id = Uuid::now_v7();

    let payload = json!({
        "function_name": function_name,
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
        "variant_name": provider.variant_name,
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

    check_parallel_tool_use_inference_response_with_function_name(
        inferences_json[0].clone(),
        provider,
        None,
        true,
        true.into(),
        function_name,
    )
    .await;

    let clickhouse = get_clickhouse().await;
    check_clickhouse_batch_request_status(&clickhouse, batch_id, provider, "completed").await;

    println!(
        "✓ Unified mock test for parallel tool use ({}) completed successfully",
        provider.variant_name
    );
}
