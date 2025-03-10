use std::path::Path;

use tensorzero_internal::endpoints::datasets::ChatInferenceDatapoint;
use tensorzero_internal::test::get_clickhouse;
pub async fn write_chat_fixture_to_dataset(fixture_path: &Path, dataset_name: &str) {
    let fixture = std::fs::read_to_string(fixture_path).unwrap();
    let mut fixture: Vec<ChatInferenceDatapoint> = serde_json::from_str(&fixture)
        .expect("Failed to parse fixture as Vec<ChatInferenceDatapoint>");
    fixture.iter_mut().for_each(|datapoint| {
        datapoint.dataset_name = dataset_name.to_string();
    });
    let clickhouse = get_clickhouse().await;
    Ok(())
}
