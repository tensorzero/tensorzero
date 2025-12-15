//! E2E tests for the `/internal/action` endpoint.
//!
//! These tests verify that the action endpoint correctly executes inference
//! using historical config snapshots loaded from ClickHouse.

use std::collections::HashMap;
use std::time::Duration;
use tensorzero::{
    ActionInferenceParams, ActionInput, ActionInputInfo, ActionResponse, Client, ClientExt, Input,
    InputMessage, InputMessageContent, Role, TensorZeroError,
};
use tensorzero_core::config::snapshot::{ConfigSnapshot, SnapshotHash};
use tensorzero_core::config::write_config_snapshot;
use tensorzero_core::db::clickhouse::test_helpers::get_clickhouse;
use tensorzero_core::inference::types::Text;
use uuid::Uuid;

/// Test that the action endpoint can execute inference using a historical config
/// that has a function not present in the running gateway config.
async fn test_action_with_historical_config_impl(client: Client) {
    let clickhouse = get_clickhouse().await;
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
"#
    );

    let snapshot =
        ConfigSnapshot::new_from_toml_string(&historical_config, HashMap::new()).unwrap();
    let snapshot_hash = snapshot.hash.clone();

    // Write the historical snapshot to ClickHouse
    write_config_snapshot(&clickhouse, snapshot).await.unwrap();
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Create the action request
    let params = ActionInputInfo {
        snapshot_hash,
        input: ActionInput::Inference(Box::new(ActionInferenceParams {
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
    let response = client.action(params).await;

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
    }
}

tensorzero::make_gateway_test_functions!(test_action_with_historical_config_impl);

/// Test that the action endpoint returns an error for a non-existent snapshot hash.
async fn test_action_nonexistent_snapshot_hash_impl(client: Client) {
    // Use a properly formatted hash that simply doesn't exist in the database
    let nonexistent_hash = SnapshotHash::new_test();

    let params = ActionInputInfo {
        snapshot_hash: nonexistent_hash,
        input: ActionInput::Inference(Box::new(ActionInferenceParams {
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

    let response = client.action(params).await;

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

tensorzero::make_gateway_test_functions!(test_action_nonexistent_snapshot_hash_impl);

/// Test that the action endpoint rejects streaming requests.
async fn test_action_streaming_rejected_impl(client: Client) {
    let clickhouse = get_clickhouse().await;
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

    write_config_snapshot(&clickhouse, snapshot).await.unwrap();
    tokio::time::sleep(Duration::from_millis(500)).await;

    let params = ActionInputInfo {
        snapshot_hash,
        input: ActionInput::Inference(Box::new(ActionInferenceParams {
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

    let response = client.action(params).await;

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

tensorzero::make_gateway_test_functions!(test_action_streaming_rejected_impl);
