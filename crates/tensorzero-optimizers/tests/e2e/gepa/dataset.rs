use std::collections::HashSet;
use tensorzero_core::db::clickhouse::test_helpers::{
    get_clickhouse, select_chat_dataset_clickhouse, select_json_dataset_clickhouse,
};
use tensorzero_core::http::TensorzeroHttpClient;
use tensorzero_optimizers::gepa::evaluate::create_evaluation_dataset;
use tokio::time::{Duration, sleep};

use super::{
    TEST_CLICKHOUSE_WAIT_MS, cleanup_dataset, create_test_chat_rendered_sample,
    create_test_json_rendered_sample, get_e2e_config,
};

#[tokio::test(flavor = "multi_thread")]
async fn test_create_chat_dataset() {
    let clickhouse = get_clickhouse().await;
    let http_client = TensorzeroHttpClient::new_testing().unwrap();
    let config = get_e2e_config().await;
    let dataset_name = "test_create_chat_dataset".to_string();

    // Clean up any leftover data
    cleanup_dataset(&clickhouse, &dataset_name).await;

    // Create test samples
    let samples = vec![
        create_test_chat_rendered_sample("input 1", "output 1"),
        create_test_chat_rendered_sample("input 2", "output 2"),
        create_test_chat_rendered_sample("input 3", "output 3"),
    ];

    // Create dataset
    let response =
        create_evaluation_dataset(&config, &http_client, &clickhouse, samples, &dataset_name)
            .await
            .expect("Failed to create evaluation dataset");

    assert_eq!(
        response.ids.len(),
        3,
        "Expected 3 datapoint ids in response"
    );

    // Give ClickHouse a moment to process
    sleep(Duration::from_millis(TEST_CLICKHOUSE_WAIT_MS)).await;

    // Verify datapoints were created
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
    assert!(first_datapoint.output.is_some());

    // Verify tags are preserved
    assert!(first_datapoint.tags.is_some());
    let tags = first_datapoint.tags.as_ref().unwrap();
    assert_eq!(tags.get("test_key"), Some(&"test_value".to_string()));

    // Clean up
    cleanup_dataset(&clickhouse, &dataset_name).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_create_json_dataset() {
    let clickhouse = get_clickhouse().await;
    let http_client = TensorzeroHttpClient::new_testing().unwrap();
    let config = get_e2e_config().await;
    let dataset_name = "test_create_json_dataset".to_string();

    // Clean up any leftover data
    cleanup_dataset(&clickhouse, &dataset_name).await;

    // Create test samples
    let samples = vec![
        create_test_json_rendered_sample("input 1", r#"{"answer": "output 1"}"#),
        create_test_json_rendered_sample("input 2", r#"{"answer": "output 2"}"#),
    ];

    // Create dataset
    let response =
        create_evaluation_dataset(&config, &http_client, &clickhouse, samples, &dataset_name)
            .await
            .expect("Failed to create evaluation dataset");

    assert_eq!(
        response.ids.len(),
        2,
        "Expected 2 datapoint ids in response"
    );

    // Give ClickHouse a moment to process
    sleep(Duration::from_millis(TEST_CLICKHOUSE_WAIT_MS)).await;

    // Verify datapoints were created
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

    // Verify output is present and structured correctly
    assert!(first_datapoint.output.is_some());
    let output = first_datapoint.output.as_ref().unwrap();
    assert!(output.raw.is_some());
    assert!(output.parsed.is_some());

    // Verify output_schema is preserved
    assert!(first_datapoint.output_schema.get("type").is_some());
    assert_eq!(first_datapoint.output_schema.get("type").unwrap(), "object");

    // Clean up
    cleanup_dataset(&clickhouse, &dataset_name).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_create_dataset_returns_correct_ids() {
    let clickhouse = get_clickhouse().await;
    let http_client = TensorzeroHttpClient::new_testing().unwrap();
    let config = get_e2e_config().await;
    let dataset_name = "test_create_dataset_ids".to_string();

    // Clean up any leftover data
    cleanup_dataset(&clickhouse, &dataset_name).await;

    // Create test samples
    let samples = vec![
        create_test_chat_rendered_sample("input 1", "output 1"),
        create_test_chat_rendered_sample("input 2", "output 2"),
    ];

    // Create dataset
    let response =
        create_evaluation_dataset(&config, &http_client, &clickhouse, samples, &dataset_name)
            .await
            .expect("Failed to create evaluation dataset");

    // Give ClickHouse a moment to process
    sleep(Duration::from_millis(TEST_CLICKHOUSE_WAIT_MS)).await;

    // Verify the returned IDs match what was created
    let datapoints = select_chat_dataset_clickhouse(&clickhouse, &dataset_name)
        .await
        .unwrap();

    assert_eq!(response.ids.len(), datapoints.len());

    // Verify all returned IDs exist in the dataset
    let datapoint_ids: HashSet<_> = datapoints.iter().map(|d| d.id).collect();

    for returned_id in &response.ids {
        assert!(
            datapoint_ids.contains(returned_id),
            "Returned ID {returned_id} not found in created datapoints"
        );
    }

    // Clean up
    cleanup_dataset(&clickhouse, &dataset_name).await;
}
