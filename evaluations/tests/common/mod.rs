#![cfg_attr(test, allow(clippy::expect_used, clippy::unwrap_used))]
use std::{collections::HashMap, path::Path, sync::Arc};

use serde_json::Map;
use tensorzero_core::client::{Client, ClientBuilder, ClientBuilderMode};
use tensorzero_core::config::Config;
use tensorzero_core::db::clickhouse::{
    TableName,
    test_helpers::{CLICKHOUSE_URL, get_clickhouse},
};
use tensorzero_core::db::stored_datapoint::{
    StoredChatInferenceDatapoint, StoredJsonInferenceDatapoint,
};
use tensorzero_core::inference::types::stored_input::{StoredInput, StoredInputMessage};
use tensorzero_core::inference::types::{
    Arguments, ContentBlockChatOutput, Role, StoredInputMessageContent, System, Text,
};
use uuid::Uuid;

// Re-export test helpers from tensorzero-core
pub use tensorzero_core::test_helpers::get_e2e_config_path;

/// Takes a chat fixture as a path to a JSONL file and writes the fixture to the dataset.
/// To avoid trampling between tests, we use a mapping from the fixture dataset names to the actual dataset names
/// that are inserted. This way, we can have multiple tests reading the same fixtures, using the same database,
/// but run independently.
pub async fn write_chat_fixture_to_dataset(
    fixture_path: &Path,
    dataset_name_mapping: &HashMap<String, String>,
) {
    let fixture = std::fs::read_to_string(fixture_path).unwrap();
    let fixture = fixture.trim();
    let mut datapoints: Vec<StoredChatInferenceDatapoint> = Vec::new();
    // Iterate over the lines in the string
    for line in fixture.lines() {
        let mut datapoint: StoredChatInferenceDatapoint = serde_json::from_str(line).unwrap();
        datapoint.id = Uuid::now_v7();
        if let Some(dataset_name) = dataset_name_mapping.get(&datapoint.dataset_name) {
            datapoint.dataset_name = dataset_name.to_string();
        }
        datapoints.push(datapoint);
    }
    let clickhouse = get_clickhouse().await;
    clickhouse
        .write_batched(&datapoints, TableName::ChatInferenceDatapoint)
        .await
        .unwrap();
}

/// Takes a JSON fixture as a path to a JSONL file and writes the fixture to the dataset.
pub async fn write_json_fixture_to_dataset(
    fixture_path: &Path,
    dataset_name_mapping: &HashMap<String, String>,
) {
    let fixture = std::fs::read_to_string(fixture_path).unwrap();
    let fixture = fixture.trim();
    let mut datapoints: Vec<StoredJsonInferenceDatapoint> = Vec::new();
    // Iterate over the lines in the string
    for line in fixture.lines() {
        let mut datapoint: StoredJsonInferenceDatapoint = serde_json::from_str(line).unwrap();
        datapoint.id = Uuid::now_v7();
        if let Some(dataset_name) = dataset_name_mapping.get(&datapoint.dataset_name) {
            datapoint.dataset_name = dataset_name.to_string();
        }
        datapoints.push(datapoint);
    }
    let clickhouse = get_clickhouse().await;
    clickhouse
        .write_batched(&datapoints, TableName::JsonInferenceDatapoint)
        .await
        .unwrap();
}

pub async fn get_tensorzero_client() -> Client {
    ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
        config_file: Some(get_e2e_config_path()),
        clickhouse_url: Some(CLICKHOUSE_URL.clone()),
        postgres_url: None,
        timeout: None,
        verify_credentials: true,
        allow_batch_writes: true,
    })
    .build()
    .await
    .unwrap()
}

/// Loads the E2E test configuration wrapped in Arc for use in tests.
pub async fn get_config() -> Arc<Config> {
    Arc::new(tensorzero_core::test_helpers::get_e2e_config().await)
}

/// Creates deterministic test datapoints for the `basic_test` function and writes them to ClickHouse.
/// This is used for top-k tests to avoid dependency on fixture files.
///
/// # Arguments
/// * `dataset_name` - The name of the dataset to write to
/// * `count` - The number of datapoints to create
pub async fn write_basic_test_datapoints(dataset_name: &str, count: usize) {
    let messages = [
        "Hello",
        "How are you?",
        "Tell me a joke",
        "What is the weather?",
        "Good morning",
        "Goodbye",
        "Help me",
        "Thanks",
        "What can you do?",
        "Test message",
    ];

    let datapoints: Vec<StoredChatInferenceDatapoint> = (0..count)
        .map(|i| {
            let message_text = messages[i % messages.len()];
            StoredChatInferenceDatapoint {
                dataset_name: dataset_name.to_string(),
                function_name: "basic_test".to_string(),
                id: Uuid::now_v7(),
                episode_id: None,
                input: StoredInput {
                    system: Some(System::Template(Arguments(Map::from_iter([(
                        "assistant_name".to_string(),
                        serde_json::json!("TestBot"),
                    )])))),
                    messages: vec![StoredInputMessage {
                        role: Role::User,
                        content: vec![StoredInputMessageContent::Text(Text {
                            text: message_text.to_string(),
                        })],
                    }],
                },
                // Reference output equals input text - used with echo/empty models for exact_match testing
                output: Some(vec![ContentBlockChatOutput::Text(Text {
                    text: message_text.to_string(),
                })]),
                tool_params: None,
                tags: None,
                is_custom: false,
                source_inference_id: None,
                staled_at: None,
                name: None,
                snapshot_hash: None,
                is_deleted: false,
                auxiliary: String::new(),
                updated_at: String::new(),
            }
        })
        .collect();

    let clickhouse = get_clickhouse().await;
    clickhouse
        .write_batched(&datapoints, TableName::ChatInferenceDatapoint)
        .await
        .unwrap();
}
