#![cfg_attr(test, allow(clippy::expect_used, clippy::unwrap_used))]
use std::path::Path;

use tensorzero_internal::{
    clickhouse::test_helpers::get_clickhouse,
    endpoints::datasets::{
        ClickHouseChatInferenceDatapoint, ClickHouseDatapoint, ClickHouseJsonInferenceDatapoint,
        Datapoint,
    },
};

/// Takes a chat fixture as a path to a JSONL file and writes the fixture to the dataset.
pub async fn write_chat_fixture_to_dataset(fixture_path: &Path) {
    let fixture = std::fs::read_to_string(fixture_path).unwrap();
    let fixture = fixture.trim();
    let mut datapoints: Vec<Datapoint> = Vec::new();
    // Iterate over the lines in the string
    for line in fixture.lines() {
        let datapoint: ClickHouseChatInferenceDatapoint = serde_json::from_str(line).unwrap();
        datapoints.push(ClickHouseDatapoint::Chat(datapoint).into());
    }
    let clickhouse = get_clickhouse().await;
    clickhouse
        .write(&datapoints, "ChatInferenceDatapoint")
        .await
        .unwrap();
}

/// Takes a JSON fixture as a path to a JSONL file and writes the fixture to the dataset.
pub async fn write_json_fixture_to_dataset(fixture_path: &Path) {
    let fixture = std::fs::read_to_string(fixture_path).unwrap();
    let fixture = fixture.trim();
    let mut datapoints: Vec<Datapoint> = Vec::new();
    // Iterate over the lines in the string
    for line in fixture.lines() {
        let datapoint: ClickHouseJsonInferenceDatapoint = serde_json::from_str(line).unwrap();
        datapoints.push(ClickHouseDatapoint::Json(datapoint).into());
    }
    let clickhouse = get_clickhouse().await;
    clickhouse
        .write(&datapoints, "JsonInferenceDatapoint")
        .await
        .unwrap();
}
