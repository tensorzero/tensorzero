/// Tests for the GET /internal/datasets endpoint (list_datasets).
use reqwest::{Client, StatusCode};
use std::time::Duration;
use uuid::Uuid;

use tensorzero_core::db::clickhouse::test_helpers::get_clickhouse;
use tensorzero_core::db::datasets::{
    ChatInferenceDatapointInsert, DatapointInsert, DatasetQueries,
};
use tensorzero_core::endpoints::datasets::v1::types::ListDatasetsResponse;
use tensorzero_core::inference::types::{
    Role, StoredInput, StoredInputMessage, StoredInputContentBlock, Text,
};

use crate::common::get_gateway_endpoint;

#[tokio::test]
async fn test_list_datasets_no_params() {
    let http_client = Client::new();
    let clickhouse = get_clickhouse().await;

    // Create a test dataset with a datapoint
    let dataset_name = format!("test-list-datasets-{}", Uuid::now_v7());
    let datapoint_id = Uuid::now_v7();

    let datapoint_insert = DatapointInsert::Chat(ChatInferenceDatapointInsert {
        dataset_name: dataset_name.clone(),
        function_name: "test_function".to_string(),
        name: Some("Test Datapoint".to_string()),
        id: datapoint_id,
        episode_id: None,
        input: StoredInput {
            system: None,
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputContentBlock::Text(Text {
                    text: "Test message".to_string(),
                })],
            }],
        },
        output: Some(vec![
            tensorzero_core::inference::types::ContentBlockChatOutput::Text(Text {
                text: "Test response".to_string(),
            }),
        ]),
        tool_params: None,
        tags: None,
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: None,
        is_custom: true,
    });

    clickhouse
        .insert_datapoints(&[datapoint_insert])
        .await
        .unwrap();

    // Wait for data to be available and retry multiple times if needed
    let mut found = false;
    for _ in 0..10 {
        tokio::time::sleep(Duration::from_millis(500)).await;

        let resp = http_client
            .get(get_gateway_endpoint("/internal/datasets"))
            .send()
            .await
            .unwrap();

        if resp.status() == StatusCode::OK {
            let body: ListDatasetsResponse = resp.json().await.unwrap();
            if body.datasets.iter().any(|d| d.dataset_name == dataset_name) {
                found = true;
                break;
            }
        }
    }

    assert!(
        found,
        "Dataset '{dataset_name}' should appear in list_datasets after insertion",
    );

    // Verify the dataset details
    let resp = http_client
        .get(get_gateway_endpoint("/internal/datasets"))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body: ListDatasetsResponse = resp.json().await.unwrap();

    let our_dataset = body
        .datasets
        .iter()
        .find(|d| d.dataset_name == dataset_name)
        .expect("Dataset should be in the list");

    assert_eq!(our_dataset.datapoint_count, 1);
    assert!(!our_dataset.last_updated.is_empty());
}

#[tokio::test]
async fn test_list_datasets_with_function_filter() {
    let http_client = Client::new();
    let clickhouse = get_clickhouse().await;

    // Create two datasets with different functions
    let dataset_name_1 = format!("test-list-func-1-{}", Uuid::now_v7());
    let dataset_name_2 = format!("test-list-func-2-{}", Uuid::now_v7());
    let function_name_1 = format!("test_function_filter_1_{}", Uuid::now_v7());
    let function_name_2 = format!("test_function_filter_2_{}", Uuid::now_v7());

    let datapoint_1 = DatapointInsert::Chat(ChatInferenceDatapointInsert {
        dataset_name: dataset_name_1.clone(),
        function_name: function_name_1.clone(),
        name: None,
        id: Uuid::now_v7(),
        episode_id: None,
        input: StoredInput {
            system: None,
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputContentBlock::Text(Text {
                    text: "Test 1".to_string(),
                })],
            }],
        },
        output: None,
        tool_params: None,
        tags: None,
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: None,
        is_custom: true,
    });

    let datapoint_2 = DatapointInsert::Chat(ChatInferenceDatapointInsert {
        dataset_name: dataset_name_2.clone(),
        function_name: function_name_2.clone(),
        name: None,
        id: Uuid::now_v7(),
        episode_id: None,
        input: StoredInput {
            system: None,
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputContentBlock::Text(Text {
                    text: "Test 2".to_string(),
                })],
            }],
        },
        output: None,
        tool_params: None,
        tags: None,
        auxiliary: String::new(),
        staled_at: None,
        source_inference_id: None,
        is_custom: true,
    });

    clickhouse
        .insert_datapoints(&[datapoint_1, datapoint_2])
        .await
        .unwrap();

    // Wait for data and retry until dataset_1 appears with function filter
    let mut found = false;
    for _ in 0..10 {
        tokio::time::sleep(Duration::from_millis(500)).await;

        let resp = http_client
            .get(get_gateway_endpoint(&format!(
                "/internal/datasets?function_name={function_name_1}",
            )))
            .send()
            .await
            .unwrap();

        if resp.status() == StatusCode::OK {
            let body: ListDatasetsResponse = resp.json().await.unwrap();
            if body
                .datasets
                .iter()
                .any(|d| d.dataset_name == dataset_name_1)
            {
                found = true;
                break;
            }
        }
    }

    assert!(
        found,
        "Dataset '{dataset_name_1}' with function '{function_name_1}' should appear in filtered results",
    );

    // Verify filtering works correctly
    let resp = http_client
        .get(get_gateway_endpoint(&format!(
            "/internal/datasets?function_name={function_name_1}",
        )))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body: ListDatasetsResponse = resp.json().await.unwrap();

    let dataset_1_present = body
        .datasets
        .iter()
        .any(|d| d.dataset_name == dataset_name_1);
    let dataset_2_present = body
        .datasets
        .iter()
        .any(|d| d.dataset_name == dataset_name_2);

    assert!(
        dataset_1_present,
        "Dataset 1 should be present when filtering by its function"
    );
    assert!(
        !dataset_2_present,
        "Dataset 2 should not be present when filtering by function_name_1"
    );
}

#[tokio::test]
async fn test_list_datasets_with_pagination() {
    let http_client = Client::new();
    let clickhouse = get_clickhouse().await;

    // Create multiple datasets to test pagination
    let mut dataset_names = Vec::new();
    let mut datapoints = Vec::new();

    for i in 0..3 {
        let dataset_name = format!("test-list-paginate-{}-{}", i, Uuid::now_v7());
        dataset_names.push(dataset_name.clone());

        let datapoint = DatapointInsert::Chat(ChatInferenceDatapointInsert {
            dataset_name: dataset_name.clone(),
            function_name: "test_function".to_string(),
            name: None,
            id: Uuid::now_v7(),
            episode_id: None,
            input: StoredInput {
                system: None,
                messages: vec![StoredInputMessage {
                    role: Role::User,
                    content: vec![StoredInputContentBlock::Text(Text {
                        text: format!("Test {i}"),
                    })],
                }],
            },
            output: None,
            tool_params: None,
            tags: None,
            auxiliary: String::new(),
            staled_at: None,
            source_inference_id: None,
            is_custom: true,
        });

        datapoints.push(datapoint);
    }

    clickhouse.insert_datapoints(&datapoints).await.unwrap();

    // Wait for all datasets to be visible
    let mut all_found = false;
    for _ in 0..10 {
        tokio::time::sleep(Duration::from_millis(500)).await;

        let resp = http_client
            .get(get_gateway_endpoint("/internal/datasets"))
            .send()
            .await
            .unwrap();

        if resp.status() == StatusCode::OK {
            let body: ListDatasetsResponse = resp.json().await.unwrap();
            let found_count = dataset_names
                .iter()
                .filter(|name| body.datasets.iter().any(|d| &d.dataset_name == *name))
                .count();

            if found_count == dataset_names.len() {
                all_found = true;
                break;
            }
        }
    }

    let num_datasets = dataset_names.len();
    assert!(
        all_found,
        "All {num_datasets} test datasets should appear in list",
    );

    // Test limit
    let resp = http_client
        .get(get_gateway_endpoint("/internal/datasets?limit=1"))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body: ListDatasetsResponse = resp.json().await.unwrap();
    assert_eq!(
        body.datasets.len(),
        1,
        "Should return exactly 1 dataset with limit=1"
    );

    // Test pagination with offset
    let resp_page_1 = http_client
        .get(get_gateway_endpoint("/internal/datasets?limit=1&offset=0"))
        .send()
        .await
        .unwrap();

    let resp_page_2 = http_client
        .get(get_gateway_endpoint("/internal/datasets?limit=1&offset=1"))
        .send()
        .await
        .unwrap();

    assert_eq!(resp_page_1.status(), StatusCode::OK);
    assert_eq!(resp_page_2.status(), StatusCode::OK);

    let body_page_1: ListDatasetsResponse = resp_page_1.json().await.unwrap();
    let body_page_2: ListDatasetsResponse = resp_page_2.json().await.unwrap();

    assert_eq!(body_page_1.datasets.len(), 1);
    assert_eq!(body_page_2.datasets.len(), 1);

    // Pages should have different datasets (they're ordered by last_updated DESC)
    assert_ne!(
        body_page_1.datasets[0].dataset_name, body_page_2.datasets[0].dataset_name,
        "Different pages should return different datasets"
    );
}

#[tokio::test]
async fn test_list_datasets_empty_result() {
    let http_client = Client::new();

    // Filter by a function that doesn't exist
    let nonexistent_function = format!("nonexistent_function_{}", Uuid::now_v7());

    let resp = http_client
        .get(get_gateway_endpoint(&format!(
            "/internal/datasets?function_name={nonexistent_function}",
        )))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body: ListDatasetsResponse = resp.json().await.unwrap();

    // Should return empty list, not an error
    assert_eq!(
        body.datasets.len(),
        0,
        "Should return empty list for nonexistent function"
    );
}
