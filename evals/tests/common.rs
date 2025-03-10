use std::path::Path;

use tensorzero_internal::{
    clickhouse::test_helpers::get_clickhouse,
    endpoints::datasets::{ChatInferenceDatapoint, JsonInferenceDatapoint},
};

/// Takes a chat fixture as a path to a JSONL file and a dataset name, and writes the fixture to the dataset.
pub async fn write_chat_fixture_to_dataset(fixture_path: &Path, dataset_name: &str) {
    let fixture = std::fs::read_to_string(fixture_path).unwrap();
    let mut datapoints = Vec::new();
    // Iterate over the lines in the string
    for line in fixture.lines() {
        let mut datapoint: ChatInferenceDatapoint = serde_json::from_str(line).unwrap();
        datapoint.dataset_name = dataset_name.to_string();
        datapoints.push(datapoint);
    }
    let clickhouse = get_clickhouse().await;
    clickhouse
        .write(&datapoints, "ChatInferenceDatapoint")
        .await
        .unwrap();
}

/// Takes a JSON fixture as a path to a JSONL file and a dataset name, and writes the fixture to the dataset.
pub async fn write_json_fixture_to_dataset(fixture_path: &Path, dataset_name: &str) {
    let fixture = std::fs::read_to_string(fixture_path).unwrap();
    let mut datapoints = Vec::new();
    // Iterate over the lines in the string
    for line in fixture.lines() {
        let mut datapoint: JsonInferenceDatapoint = serde_json::from_str(line).unwrap();
        datapoint.dataset_name = dataset_name.to_string();
        datapoints.push(datapoint);
    }
    let clickhouse = get_clickhouse().await;
    clickhouse
        .write(&datapoints, "JsonInferenceDatapoint")
        .await
        .unwrap();
}
