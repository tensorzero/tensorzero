#![cfg_attr(test, allow(clippy::expect_used, clippy::unwrap_used))]
// Helpers are shared across multiple test binaries; not all binaries use every helper.
#![allow(dead_code)]
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use tensorzero_core::config::Config;
use tensorzero_core::db::datasets::DatasetQueries;
use tensorzero_core::db::delegating_connection::DelegatingDatabaseConnection;
use tensorzero_core::db::feedback::{BooleanMetricFeedbackRow, FeedbackQueries, FeedbackRow};
use tensorzero_core::db::stored_datapoint::{StoredChatInferenceDatapoint, StoredDatapoint};
use tensorzero_core::db::test_helpers::TestDatabaseHelpers;
use uuid::Uuid;

pub use tensorzero::test_helpers::get_e2e_config_path;

pub fn init_tracing_for_tests() {
    let _ = tracing_subscriber::fmt().try_init();
}

/// Loads the E2E test configuration wrapped in Arc for use in tests.
pub async fn get_config() -> Arc<Config> {
    Arc::new(tensorzero_core::test_helpers::get_e2e_config().await)
}

/// Takes a chat fixture as a path to a JSONL file and writes the fixture to the dataset.
/// To avoid trampling between tests, we use a mapping from the fixture dataset names to the actual
/// dataset names that are inserted.
pub async fn write_chat_fixture_to_dataset(
    db: &DelegatingDatabaseConnection,
    fixture_path: &Path,
    dataset_name_mapping: &HashMap<String, String>,
) {
    let fixture = std::fs::read_to_string(fixture_path).unwrap();
    let fixture = fixture.trim();
    let mut datapoints: Vec<StoredDatapoint> = Vec::new();
    for line in fixture.lines() {
        let mut datapoint: StoredChatInferenceDatapoint = serde_json::from_str(line).unwrap();
        datapoint.id = Uuid::now_v7();
        if let Some(dataset_name) = dataset_name_mapping.get(&datapoint.dataset_name) {
            datapoint.dataset_name = dataset_name.to_string();
        }
        datapoints.push(StoredDatapoint::Chat(datapoint));
    }
    db.insert_datapoints(&datapoints).await.unwrap();
    db.flush_pending_writes().await;
}

pub async fn query_boolean_feedback(
    db: &DelegatingDatabaseConnection,
    target_id: Uuid,
    metric_name: Option<&str>,
) -> Option<BooleanMetricFeedbackRow> {
    let rows = db
        .query_feedback_by_target_id(target_id, None, None, None)
        .await
        .unwrap();
    rows.into_iter().find_map(|row| match row {
        FeedbackRow::Boolean(b)
            if metric_name.is_none() || Some(b.metric_name.as_str()) == metric_name =>
        {
            Some(b)
        }
        _ => None,
    })
}
