//! E2E tests for the `/internal/action` endpoint.
//!
//! These tests verify that the action endpoint correctly executes inference
//! using historical config snapshots.
//!
//! Note: These tests only run in HTTP gateway mode because the Rust SDK's embedded
//! gateway mode doesn't support the action endpoint (it would require depending on
//! durable-tools). The action endpoint is fully functional in HTTP mode.

use durable_tools::CacheEnabledMode;
use durable_tools::action::{ActionInput, ActionInputInfo, ActionResponse, RunEvaluationParams};
use std::collections::HashMap;
use tensorzero::{
    Client, ClientInferenceParams, Input, InputMessage, InputMessageContent, Role, TensorZeroError,
};
use tensorzero_core::config::snapshot::{ConfigSnapshot, SnapshotHash};
use tensorzero_core::db::ConfigQueries;
use tensorzero_core::db::datasets::DatasetQueries;
use tensorzero_core::db::delegating_connection::DelegatingDatabaseConnection;
use tensorzero_core::db::stored_datapoint::{StoredChatInferenceDatapoint, StoredDatapoint};
use tensorzero_core::db::test_helpers::{TestDatabaseHelpers, poll_result_until_some};
use tensorzero_core::inference::types::{
    ContentBlockChatOutput, StoredInput, StoredInputMessage, StoredInputMessageContent, Text,
};
use uuid::Uuid;

// Helper to create HTTP gateway client for action tests
async fn make_http_gateway() -> Client {
    tensorzero::test_helpers::make_http_gateway().await
}

/// Helper to call the action endpoint via HTTP.
/// Since the action endpoint is internal, it's not exposed via the SDK.
async fn call_action(
    client: &Client,
    params: ActionInputInfo,
) -> Result<ActionResponse, TensorZeroError> {
    use tensorzero::ClientMode;

    match client.mode() {
        ClientMode::HTTPGateway(http_client) => {
            let url = http_client.base_url.join("internal/action").map_err(|e| {
                TensorZeroError::Other {
                    source: tensorzero_core::error::Error::new(
                        tensorzero_core::error::ErrorDetails::InvalidBaseUrl {
                            message: format!(
                                "Failed to join base URL with /internal/action endpoint: {e}"
                            ),
                        },
                    )
                    .into(),
                }
            })?;
            let builder = http_client.http_client.post(url).json(&params);
            Ok(http_client.send_and_parse_http_response(builder).await?.0)
        }
        ClientMode::EmbeddedGateway { .. } => Err(TensorZeroError::Other {
            source: tensorzero_core::error::Error::new(
                tensorzero_core::error::ErrorDetails::InternalError {
                    message: "Action endpoint is not supported for embedded gateway mode"
                        .to_string(),
                },
            )
            .into(),
        }),
    }
}

/// Test that the action endpoint can execute inference using a historical config
/// that has a function not present in the running gateway config.
async fn test_action_with_historical_config_impl(client: Client) {
    let database = DelegatingDatabaseConnection::new_for_e2e_test().await;
    let id = Uuid::now_v7();

    // Create a historical config with a unique function that doesn't exist in the running gateway.
    let historical_config = format!(
        r#"
[models.action_test_model_{id}]
routing = ["provider"]

[models.action_test_model_{id}.providers.provider]
type = "dummy"
model_name = "test"

[functions.historical_only_func_{id}]
type = "chat"

[functions.historical_only_func_{id}.variants.baseline]
type = "chat_completion"
model = "action_test_model_{id}"

[functions.historical_only_func_{id}.variants.baseline.system_template]
__tensorzero_remapped_path = "/config/functions/historical_only_func_{id}/system_template.minijinja"
__data = """
Do a historical inference successfully!
"""
"#
    );

    let snapshot =
        ConfigSnapshot::new_from_toml_string(&historical_config, HashMap::new()).unwrap();
    let snapshot_hash = snapshot.hash.clone();

    database.write_config_snapshot(&snapshot).await.unwrap();

    // Poll until config snapshot is visible in ClickHouse
    poll_result_until_some(async || {
        database
            .get_config_snapshot(snapshot_hash.clone())
            .await
            .ok()
    })
    .await;

    // Create the action request
    let params = ActionInputInfo {
        snapshot_hash,
        input: ActionInput::Inference(Box::new(ClientInferenceParams {
            function_name: Some(format!("historical_only_func_{id}")),
            input: Input {
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: "test".to_string(),
                    })],
                }],
                system: None,
            },
            ..Default::default()
        })),
    };

    // Call the action endpoint using the client
    let response = call_action(&client, params).await;

    assert!(
        response.is_ok(),
        "Action with historical config should succeed: {:?}",
        response.err()
    );

    // Verify we got an inference response
    match response.unwrap() {
        ActionResponse::Inference(_) => {}
        ActionResponse::Feedback(_) => {
            panic!("Expected inference response, got feedback response")
        }
        ActionResponse::RunEvaluation(_) => {
            panic!("Expected inference response, got run_evaluation response")
        }
    }
}

// Only HTTP gateway test - embedded mode doesn't support action in the SDK
#[tokio::test(flavor = "multi_thread")]
async fn test_action_with_historical_config_impl_http_gateway() {
    test_action_with_historical_config_impl(make_http_gateway().await).await;
}

/// Test that the action endpoint returns an error for a non-existent snapshot hash.
async fn test_action_nonexistent_snapshot_hash_impl(client: Client) {
    // Use a properly formatted hash that simply doesn't exist in the database
    let nonexistent_hash = SnapshotHash::new_test();

    let params = ActionInputInfo {
        snapshot_hash: nonexistent_hash,
        input: ActionInput::Inference(Box::new(ClientInferenceParams {
            function_name: Some("any_function".to_string()),
            input: Input {
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: "test".to_string(),
                    })],
                }],
                system: None,
            },
            ..Default::default()
        })),
    };

    let response = call_action(&client, params).await;

    // Should return an error (404 for HTTP, error for embedded)
    assert!(
        response.is_err(),
        "Should return error for non-existent snapshot hash"
    );

    // Verify it's a 404 error
    match response.unwrap_err() {
        TensorZeroError::Http { status_code, .. } => {
            assert_eq!(
                status_code, 404,
                "Should return 404 for non-existent snapshot hash"
            );
        }
        other => panic!("Expected HTTP error with 404 status, got: {other:?}"),
    }
}

// Only HTTP gateway test - embedded mode doesn't support action in the SDK
#[tokio::test(flavor = "multi_thread")]
async fn test_action_nonexistent_snapshot_hash_impl_http_gateway() {
    test_action_nonexistent_snapshot_hash_impl(make_http_gateway().await).await;
}

/// Test that the action endpoint rejects streaming requests.
async fn test_action_streaming_rejected_impl(client: Client) {
    let database = DelegatingDatabaseConnection::new_for_e2e_test().await;
    let id = Uuid::now_v7();

    // Create a minimal historical config with proper variant structure
    let historical_config = format!(
        r#"
[models.action_test_model_{id}]
routing = ["provider"]

[models.action_test_model_{id}.providers.provider]
type = "dummy"
model_name = "test"

[functions.stream_test_func_{id}]
type = "chat"

[functions.stream_test_func_{id}.variants.baseline]
type = "chat_completion"
model = "action_test_model_{id}"
"#
    );

    let snapshot =
        ConfigSnapshot::new_from_toml_string(&historical_config, HashMap::new()).unwrap();
    let snapshot_hash = snapshot.hash.clone();

    database.write_config_snapshot(&snapshot).await.unwrap();

    // Poll until config snapshot is visible in ClickHouse
    poll_result_until_some(async || {
        database
            .get_config_snapshot(snapshot_hash.clone())
            .await
            .ok()
    })
    .await;

    let params = ActionInputInfo {
        snapshot_hash,
        input: ActionInput::Inference(Box::new(ClientInferenceParams {
            function_name: Some(format!("stream_test_func_{id}")),
            stream: Some(true), // Request streaming
            input: Input {
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: "test".to_string(),
                    })],
                }],
                system: None,
            },
            ..Default::default()
        })),
    };

    let response = call_action(&client, params).await;

    // Should return an error (400 Bad Request)
    assert!(response.is_err(), "Streaming requests should be rejected");

    // Verify it's a 400 error
    match response.unwrap_err() {
        TensorZeroError::Http { status_code, .. } => {
            assert_eq!(
                status_code, 400,
                "Streaming requests should be rejected with 400 Bad Request"
            );
        }
        other => panic!("Expected HTTP error with 400 status, got: {other:?}"),
    }
}

// Only HTTP gateway test - embedded mode doesn't support action in the SDK
#[tokio::test(flavor = "multi_thread")]
async fn test_action_streaming_rejected_impl_http_gateway() {
    test_action_streaming_rejected_impl(make_http_gateway().await).await;
}

/// Test that the action endpoint returns an error when evaluation is not found in config.
async fn test_action_run_evaluation_missing_evaluation_impl(client: Client) {
    let database = DelegatingDatabaseConnection::new_for_e2e_test().await;
    let id = Uuid::now_v7();

    // Create a historical config WITHOUT any evaluations
    let historical_config = format!(
        r#"
[models.eval_test_model_{id}]
routing = ["provider"]

[models.eval_test_model_{id}.providers.provider]
type = "dummy"
model_name = "test"

[functions.eval_test_func_{id}]
type = "chat"

[functions.eval_test_func_{id}.variants.baseline]
type = "chat_completion"
model = "eval_test_model_{id}"
"#
    );

    let snapshot =
        ConfigSnapshot::new_from_toml_string(&historical_config, HashMap::new()).unwrap();
    let snapshot_hash = snapshot.hash.clone();

    database.write_config_snapshot(&snapshot).await.unwrap();

    // Poll until config snapshot is visible in ClickHouse
    poll_result_until_some(async || {
        database
            .get_config_snapshot(snapshot_hash.clone())
            .await
            .ok()
    })
    .await;

    let params = ActionInputInfo {
        snapshot_hash,
        input: ActionInput::RunEvaluation(Box::new(RunEvaluationParams {
            evaluation_name: "nonexistent_evaluation".to_string(),
            dataset_name: Some("some_dataset".to_string()),
            datapoint_ids: None,
            variant_name: "baseline".to_string(),
            concurrency: 1,
            inference_cache: CacheEnabledMode::Off,
            max_datapoints: None,
            precision_targets: HashMap::new(),
            include_datapoint_results: false,
            tags: HashMap::new(),
        })),
    };

    let response = call_action(&client, params).await;

    assert!(
        response.is_err(),
        "Should return error for missing evaluation"
    );

    match response.unwrap_err() {
        TensorZeroError::Http { status_code, .. } => {
            assert_eq!(status_code, 400, "Should return 400 for missing evaluation");
        }
        other => panic!("Expected HTTP error, got: {other:?}"),
    }
}

// Only HTTP gateway test - embedded mode doesn't support action in the SDK
#[tokio::test(flavor = "multi_thread")]
async fn test_action_run_evaluation_missing_evaluation_impl_http_gateway() {
    test_action_run_evaluation_missing_evaluation_impl(make_http_gateway().await).await;
}

/// Test that the action endpoint returns an error when validation fails
/// (neither dataset_name nor datapoint_ids provided).
async fn test_action_run_evaluation_validation_error_impl(client: Client) {
    let database = DelegatingDatabaseConnection::new_for_e2e_test().await;
    let id = Uuid::now_v7();

    // Create a historical config with an evaluation
    let historical_config = format!(
        r#"
[models.eval_test_model_{id}]
routing = ["provider"]

[models.eval_test_model_{id}.providers.provider]
type = "dummy"
model_name = "test"

[functions.eval_test_func_{id}]
type = "chat"

[functions.eval_test_func_{id}.variants.baseline]
type = "chat_completion"
model = "eval_test_model_{id}"

[evaluations.test_eval_{id}]
type = "inference"
function_name = "eval_test_func_{id}"

[evaluations.test_eval_{id}.evaluators.dummy_judge]
type = "llm_judge"
output_type = "boolean"
optimize = "max"

[evaluations.test_eval_{id}.evaluators.dummy_judge.variants.judge]
type = "chat_completion"
model = "eval_test_model_{id}"
active = true
system_instructions = {{ __tensorzero_remapped_path = "inline", __data = "Return true." }}
json_mode = "on"
"#
    );

    let snapshot =
        ConfigSnapshot::new_from_toml_string(&historical_config, HashMap::new()).unwrap();
    let snapshot_hash = snapshot.hash.clone();

    database.write_config_snapshot(&snapshot).await.unwrap();

    // Poll until config snapshot is visible in ClickHouse
    poll_result_until_some(async || {
        database
            .get_config_snapshot(snapshot_hash.clone())
            .await
            .ok()
    })
    .await;

    // Neither dataset_name nor datapoint_ids provided
    let params = ActionInputInfo {
        snapshot_hash,
        input: ActionInput::RunEvaluation(Box::new(RunEvaluationParams {
            evaluation_name: format!("test_eval_{id}"),
            dataset_name: None,
            datapoint_ids: None,
            variant_name: "baseline".to_string(),
            concurrency: 1,
            inference_cache: CacheEnabledMode::Off,
            max_datapoints: None,
            precision_targets: HashMap::new(),
            include_datapoint_results: false,
            tags: HashMap::new(),
        })),
    };

    let response = call_action(&client, params).await;

    assert!(
        response.is_err(),
        "Should return error when neither dataset_name nor datapoint_ids provided"
    );

    match response.unwrap_err() {
        TensorZeroError::Http { status_code, .. } => {
            assert_eq!(status_code, 400, "Should return 400 for validation error");
        }
        other => panic!("Expected HTTP error, got: {other:?}"),
    }
}

// Only HTTP gateway test - embedded mode doesn't support action in the SDK
#[tokio::test(flavor = "multi_thread")]
async fn test_action_run_evaluation_validation_error_impl_http_gateway() {
    test_action_run_evaluation_validation_error_impl(make_http_gateway().await).await;
}

/// Test that the action endpoint returns an error when both dataset_name AND datapoint_ids are provided.
async fn test_action_run_evaluation_both_dataset_and_ids_impl(client: Client) {
    let database = DelegatingDatabaseConnection::new_for_e2e_test().await;
    let id = Uuid::now_v7();

    // Create a historical config with an evaluation
    let historical_config = format!(
        r#"
[models.eval_test_model_{id}]
routing = ["provider"]

[models.eval_test_model_{id}.providers.provider]
type = "dummy"
model_name = "test"

[functions.eval_test_func_{id}]
type = "chat"

[functions.eval_test_func_{id}.variants.baseline]
type = "chat_completion"
model = "eval_test_model_{id}"

[evaluations.test_eval_{id}]
type = "inference"
function_name = "eval_test_func_{id}"

[evaluations.test_eval_{id}.evaluators.dummy_judge]
type = "llm_judge"
output_type = "boolean"
optimize = "max"

[evaluations.test_eval_{id}.evaluators.dummy_judge.variants.judge]
type = "chat_completion"
model = "eval_test_model_{id}"
active = true
system_instructions = {{ __tensorzero_remapped_path = "inline", __data = "Return true." }}
json_mode = "on"
"#
    );

    let snapshot =
        ConfigSnapshot::new_from_toml_string(&historical_config, HashMap::new()).unwrap();
    let snapshot_hash = snapshot.hash.clone();

    database.write_config_snapshot(&snapshot).await.unwrap();

    // Poll until config snapshot is visible in ClickHouse
    poll_result_until_some(async || {
        database
            .get_config_snapshot(snapshot_hash.clone())
            .await
            .ok()
    })
    .await;

    // Both dataset_name AND datapoint_ids provided - should fail validation
    let params = ActionInputInfo {
        snapshot_hash,
        input: ActionInput::RunEvaluation(Box::new(RunEvaluationParams {
            evaluation_name: format!("test_eval_{id}"),
            dataset_name: Some("some_dataset".to_string()),
            datapoint_ids: Some(vec![Uuid::now_v7()]),
            variant_name: "baseline".to_string(),
            concurrency: 1,
            inference_cache: CacheEnabledMode::Off,
            max_datapoints: None,
            precision_targets: HashMap::new(),
            include_datapoint_results: false,
            tags: HashMap::new(),
        })),
    };

    let response = call_action(&client, params).await;

    assert!(
        response.is_err(),
        "Should return error when both dataset_name and datapoint_ids provided"
    );

    match response.unwrap_err() {
        TensorZeroError::Http { status_code, .. } => {
            assert_eq!(status_code, 400, "Should return 400 for validation error");
        }
        other => panic!("Expected HTTP error, got: {other:?}"),
    }
}

// Only HTTP gateway test - embedded mode doesn't support action in the SDK
#[tokio::test(flavor = "multi_thread")]
async fn test_action_run_evaluation_both_dataset_and_ids_impl_http_gateway() {
    test_action_run_evaluation_both_dataset_and_ids_impl(make_http_gateway().await).await;
}

/// Test that the action endpoint can successfully run an evaluation with a historical config.
async fn test_action_run_evaluation_basic_impl(client: Client) {
    let database = DelegatingDatabaseConnection::new_for_e2e_test().await;
    let id = Uuid::now_v7();
    let dataset_name = format!("action_eval_dataset_{id}");
    let function_name = format!("eval_func_{id}");

    // Create a historical config with a function and evaluation
    // The LLM judge uses dummy::llm_judge::true which always returns true
    let historical_config = format!(
        r#"
[models.eval_model_{id}]
routing = ["provider"]

[models.eval_model_{id}.providers.provider]
type = "dummy"
model_name = "test"

[functions.{function_name}]
type = "chat"

[functions.{function_name}.variants.baseline]
type = "chat_completion"
model = "eval_model_{id}"

[evaluations.action_test_eval_{id}]
type = "inference"
function_name = "{function_name}"

[evaluations.action_test_eval_{id}.evaluators.always_true]
type = "llm_judge"
output_type = "boolean"
optimize = "max"

[evaluations.action_test_eval_{id}.evaluators.always_true.variants.judge]
type = "chat_completion"
model = "dummy::llm_judge::true"
active = true
system_instructions = {{ __tensorzero_remapped_path = "inline", __data = "Return true." }}
json_mode = "on"
"#
    );

    let snapshot =
        ConfigSnapshot::new_from_toml_string(&historical_config, HashMap::new()).unwrap();
    let snapshot_hash = snapshot.hash.clone();

    database.write_config_snapshot(&snapshot).await.unwrap();

    // Create test datapoints for the evaluation
    let datapoint_id_1 = Uuid::now_v7();
    let datapoint_id_2 = Uuid::now_v7();

    let datapoints = vec![
        StoredDatapoint::Chat(StoredChatInferenceDatapoint {
            dataset_name: dataset_name.clone(),
            function_name: function_name.clone(),
            name: Some("Test datapoint 1".to_string()),
            id: datapoint_id_1,
            episode_id: None,
            input: StoredInput {
                system: None,
                messages: vec![StoredInputMessage {
                    role: tensorzero_core::inference::types::Role::User,
                    content: vec![StoredInputMessageContent::Text(Text {
                        text: "Hello, world!".to_string(),
                    })],
                }],
            },
            output: Some(vec![ContentBlockChatOutput::Text(Text {
                text: "Hi there!".to_string(),
            })]),
            tool_params: None,
            tags: None,
            auxiliary: String::new(),
            staled_at: None,
            source_inference_id: None,
            is_custom: true,
            is_deleted: false,
            updated_at: String::new(),
            snapshot_hash: None,
        }),
        StoredDatapoint::Chat(StoredChatInferenceDatapoint {
            dataset_name: dataset_name.clone(),
            function_name: function_name.clone(),
            name: Some("Test datapoint 2".to_string()),
            id: datapoint_id_2,
            episode_id: None,
            input: StoredInput {
                system: None,
                messages: vec![StoredInputMessage {
                    role: tensorzero_core::inference::types::Role::User,
                    content: vec![StoredInputMessageContent::Text(Text {
                        text: "How are you?".to_string(),
                    })],
                }],
            },
            output: Some(vec![ContentBlockChatOutput::Text(Text {
                text: "I'm doing well!".to_string(),
            })]),
            tool_params: None,
            tags: None,
            auxiliary: String::new(),
            staled_at: None,
            source_inference_id: None,
            is_custom: true,
            is_deleted: false,
            updated_at: String::new(),
            snapshot_hash: None,
        }),
    ];

    database.insert_datapoints(&datapoints).await.unwrap();

    // Poll until config snapshot and datapoints are visible in ClickHouse
    poll_result_until_some(async || {
        database.flush_pending_writes().await;
        database
            .get_config_snapshot(snapshot_hash.clone())
            .await
            .ok()
    })
    .await;

    // Call the action endpoint with RunEvaluation using datapoint_ids
    let params = ActionInputInfo {
        snapshot_hash,
        input: ActionInput::RunEvaluation(Box::new(RunEvaluationParams {
            evaluation_name: format!("action_test_eval_{id}"),
            dataset_name: None,
            datapoint_ids: Some(vec![datapoint_id_1, datapoint_id_2]),
            variant_name: "baseline".to_string(),
            concurrency: 1,
            inference_cache: CacheEnabledMode::Off,
            max_datapoints: None,
            precision_targets: HashMap::new(),
            include_datapoint_results: true,
            tags: HashMap::new(),
        })),
    };

    let response = call_action(&client, params).await;

    assert!(
        response.is_ok(),
        "RunEvaluation action should succeed: {:?}",
        response.err()
    );

    // Verify we got a RunEvaluation response with correct structure
    match response.unwrap() {
        ActionResponse::RunEvaluation(eval_response) => {
            assert_eq!(
                eval_response.num_datapoints, 2,
                "Should have evaluated 2 datapoints"
            );
            assert_eq!(
                eval_response.num_successes, 2,
                "Both evaluations should succeed"
            );
            assert_eq!(eval_response.num_errors, 0, "Should have no errors");

            // Verify stats contain the evaluator
            assert!(
                eval_response.stats.contains_key("always_true"),
                "Stats should contain 'always_true' evaluator"
            );

            let always_true_stats = &eval_response.stats["always_true"];
            assert_eq!(always_true_stats.count, 2, "Should have 2 samples");
            // dummy::llm_judge::true always returns true (1.0)
            assert!(
                (always_true_stats.mean - 1.0).abs() < 0.01,
                "Mean should be 1.0 for always_true evaluator, got {}",
                always_true_stats.mean
            );

            // Verify datapoint results are included
            assert!(
                eval_response.datapoint_results.is_some(),
                "Should include datapoint results when requested"
            );
            let results = eval_response.datapoint_results.unwrap();
            assert_eq!(results.len(), 2, "Should have 2 datapoint results");

            for result in &results {
                assert!(result.success, "Each datapoint should be successful");
                assert!(
                    result.evaluations.contains_key("always_true"),
                    "Each result should have always_true evaluation"
                );
            }
        }
        ActionResponse::Inference(_) => {
            panic!("Expected RunEvaluation response, got Inference response")
        }
        ActionResponse::Feedback(_) => {
            panic!("Expected RunEvaluation response, got Feedback response")
        }
    }
}

// Only HTTP gateway test - embedded mode doesn't support action in the SDK
#[tokio::test(flavor = "multi_thread")]
async fn test_action_run_evaluation_basic_impl_http_gateway() {
    test_action_run_evaluation_basic_impl(make_http_gateway().await).await;
}
