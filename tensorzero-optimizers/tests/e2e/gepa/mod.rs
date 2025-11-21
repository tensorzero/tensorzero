#![allow(clippy::unwrap_used, clippy::expect_used)]

use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use tensorzero::DynamicToolParams;
use tensorzero_core::config::{Config, ConfigFileGlob};
use tensorzero_core::db::clickhouse::test_helpers::{
    get_clickhouse, select_chat_dataset_clickhouse, select_json_dataset_clickhouse,
};
use tensorzero_core::endpoints::datasets::v1::delete_dataset;
use tensorzero_core::http::TensorzeroHttpClient;
use tensorzero_core::inference::types::{
    Arguments, ContentBlockChatOutput, JsonInferenceOutput, ModelInput, ResolvedContentBlock,
    ResolvedRequestMessage, Role, StoredInput, StoredInputMessage, StoredInputMessageContent,
    System, Template, Text,
};
use tensorzero_core::stored_inference::{RenderedSample, StoredOutput};
use tensorzero_optimizers::gepa::create_evaluation_dataset;
use uuid::Uuid;

pub mod analyze;

/// Helper function to load the e2e test config
async fn get_e2e_config() -> Arc<Config> {
    let mut config_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    config_path.push("../tensorzero-core/tests/e2e/config/tensorzero.*.toml");
    Arc::new(
        Config::load_from_path_optional_verify_credentials(
            &ConfigFileGlob::new_from_path(&config_path)
                .expect("Failed to create config glob from path"),
            false,
        )
        .await
        .expect("Failed to load e2e config")
        .config,
    )
}

/// Helper function to create a test RenderedSample for Chat functions
fn create_test_chat_rendered_sample(input: &str, output: &str) -> RenderedSample {
    let output_vec = vec![ContentBlockChatOutput::Text(Text {
        text: output.to_string(),
    })];
    RenderedSample {
        function_name: "basic_test".to_string(),
        input: ModelInput {
            system: None,
            messages: vec![ResolvedRequestMessage {
                role: Role::User,
                content: vec![ResolvedContentBlock::Text(Text {
                    text: input.to_string(),
                })],
            }],
        },
        stored_input: StoredInput {
            system: Some(System::Template(Arguments(
                json!({"assistant_name": "TestAssistant"})
                    .as_object()
                    .expect("Failed to convert JSON to object")
                    .clone(),
            ))),
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputMessageContent::Text(Text {
                    text: input.to_string(),
                })],
            }],
        },
        output: Some(output_vec.clone()),
        stored_output: Some(StoredOutput::Chat(output_vec)),
        episode_id: Some(Uuid::now_v7()),
        inference_id: Some(Uuid::now_v7()),
        tool_params: DynamicToolParams::default(),
        output_schema: None,
        dispreferred_outputs: vec![],
        tags: {
            let mut tags = HashMap::new();
            tags.insert("test_key".to_string(), "test_value".to_string());
            tags
        },
    }
}

/// Helper function to create a test RenderedSample for JSON functions
fn create_test_json_rendered_sample(input: &str, output: &str) -> RenderedSample {
    let json_output = JsonInferenceOutput {
        raw: Some(output.to_string()),
        parsed: Some(serde_json::json!({"answer": output})),
    };

    RenderedSample {
        function_name: "json_success".to_string(),
        input: ModelInput {
            system: Some("JSON system prompt".to_string()),
            messages: vec![ResolvedRequestMessage {
                role: Role::User,
                content: vec![ResolvedContentBlock::Text(Text {
                    text: input.to_string(),
                })],
            }],
        },
        stored_input: StoredInput {
            system: Some(System::Template(Arguments(
                json!({"assistant_name": "TestAssistant"})
                    .as_object()
                    .expect("Failed to convert JSON to object")
                    .clone(),
            ))),
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputMessageContent::Template(Template {
                    name: "user".to_string(),
                    arguments: Arguments(serde_json::Map::from_iter(vec![(
                        "country".to_string(),
                        json!(input),
                    )])),
                })],
            }],
        },
        output: None, // JSON functions don't have chat output
        stored_output: Some(StoredOutput::Json(json_output)),
        episode_id: Some(Uuid::now_v7()),
        inference_id: Some(Uuid::now_v7()),
        tool_params: DynamicToolParams::default(),
        output_schema: Some(serde_json::json!({
            "type": "object",
            "properties": {
                "answer": {"type": "string"}
            },
            "required": ["answer"],
            "additionalProperties": false
        })),
        dispreferred_outputs: vec![],
        tags: {
            let mut tags = HashMap::new();
            tags.insert("json_key".to_string(), "json_value".to_string());
            tags
        },
    }
}

#[tokio::test]
async fn test_create_evaluation_dataset_chat() {
    let clickhouse = get_clickhouse().await;
    let http_client = TensorzeroHttpClient::new_testing().unwrap();
    let config = get_e2e_config().await;

    // Generate unique dataset name to ensure test isolation
    let dataset_name = format!("test_eval_dataset_chat_{}", Uuid::now_v7());

    // Create test samples
    let samples = vec![
        create_test_chat_rendered_sample("test input 1", "test output 1"),
        create_test_chat_rendered_sample("test input 2", "test output 2"),
        create_test_chat_rendered_sample("test input 3", "test output 3"),
    ];

    // Call create_evaluation_dataset
    let result =
        create_evaluation_dataset(&config, &http_client, &clickhouse, samples, &dataset_name).await;

    assert!(
        result.is_ok(),
        "Failed to create evaluation dataset: {:?}",
        result.err()
    );

    // Give ClickHouse a moment to process
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    // Verify the datapoints were created in ClickHouse
    let datapoints = select_chat_dataset_clickhouse(&clickhouse, &dataset_name)
        .await
        .unwrap();

    assert_eq!(
        datapoints.len(),
        3,
        "Expected 3 datapoints to be created, but found {}",
        datapoints.len()
    );

    // Verify the structure of the first datapoint
    let first_datapoint = &datapoints[0];
    assert_eq!(first_datapoint.dataset_name, dataset_name);
    assert_eq!(first_datapoint.function_name, "basic_test");
    assert!(!first_datapoint.is_deleted);
    assert!(first_datapoint.episode_id.is_some());

    // Verify tags are preserved
    assert!(first_datapoint.tags.is_some());
    let tags = first_datapoint.tags.as_ref().unwrap();
    assert_eq!(tags.get("test_key"), Some(&"test_value".to_string()));

    // Verify output is present
    assert!(first_datapoint.output.is_some());

    // Delete the dataset
    let delete_result = delete_dataset(&clickhouse, &dataset_name).await;
    assert!(
        delete_result.is_ok(),
        "Failed to delete dataset: {:?}",
        delete_result.err()
    );
    assert_eq!(delete_result.unwrap().num_deleted_datapoints, 3);

    // Give ClickHouse a moment to process the deletion
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    // Verify the dataset is empty after deletion
    let datapoints_after_delete = select_chat_dataset_clickhouse(&clickhouse, &dataset_name)
        .await
        .unwrap();
    assert!(
        datapoints_after_delete.is_empty(),
        "Expected dataset to be empty after deletion, but found {} datapoints",
        datapoints_after_delete.len()
    );
}

#[tokio::test]
async fn test_create_evaluation_dataset_json() {
    let clickhouse = get_clickhouse().await;
    let http_client = TensorzeroHttpClient::new_testing().unwrap();
    let config = get_e2e_config().await;

    // Generate unique dataset name to ensure test isolation
    let dataset_name = format!("test_eval_dataset_json_{}", Uuid::now_v7());

    // Create test samples for JSON function
    let samples = vec![
        create_test_json_rendered_sample("input 1", r#"{"answer": "output 1"}"#),
        create_test_json_rendered_sample("input 2", r#"{"answer": "output 2"}"#),
    ];

    // Call create_evaluation_dataset
    let result =
        create_evaluation_dataset(&config, &http_client, &clickhouse, samples, &dataset_name).await;

    assert!(
        result.is_ok(),
        "Failed to create evaluation dataset: {:?}",
        result.err()
    );

    // Give ClickHouse a moment to process
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    // Verify the datapoints were created in ClickHouse
    let datapoints = select_json_dataset_clickhouse(&clickhouse, &dataset_name)
        .await
        .unwrap();

    assert_eq!(
        datapoints.len(),
        2,
        "Expected 2 datapoints to be created, but found {}",
        datapoints.len()
    );

    // Verify the structure of the first datapoint
    let first_datapoint = &datapoints[0];
    assert_eq!(first_datapoint.dataset_name, dataset_name);
    assert_eq!(first_datapoint.function_name, "json_success");
    assert!(!first_datapoint.is_deleted);
    assert!(first_datapoint.episode_id.is_some());

    // Verify tags are preserved
    assert!(first_datapoint.tags.is_some());
    let tags = first_datapoint.tags.as_ref().unwrap();
    assert_eq!(tags.get("json_key"), Some(&"json_value".to_string()));

    // Verify output is present and structured correctly
    assert!(first_datapoint.output.is_some());
    let output = first_datapoint.output.as_ref().unwrap();
    assert!(output.raw.is_some());
    assert!(output.parsed.is_some());

    // Verify output_schema is preserved
    assert!(first_datapoint.output_schema.get("type").is_some());
    assert_eq!(first_datapoint.output_schema.get("type").unwrap(), "object");

    // Delete the dataset
    let delete_result = delete_dataset(&clickhouse, &dataset_name).await;
    assert!(
        delete_result.is_ok(),
        "Failed to delete dataset: {:?}",
        delete_result.err()
    );
    assert_eq!(delete_result.unwrap().num_deleted_datapoints, 2);

    // Give ClickHouse a moment to process the deletion
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    // Verify the dataset is empty after deletion
    let datapoints_after_delete = select_json_dataset_clickhouse(&clickhouse, &dataset_name)
        .await
        .unwrap();
    assert!(
        datapoints_after_delete.is_empty(),
        "Expected dataset to be empty after deletion, but found {} datapoints",
        datapoints_after_delete.len()
    );
}
