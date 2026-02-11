//! E2E tests for the datapoint stats endpoint.

use reqwest::Client;
use serde_json::{Map, Value};
use tensorzero_core::db::test_helpers::poll_result_until_some;
use uuid::Uuid;

use crate::common::get_gateway_endpoint;

use tensorzero_core::endpoints::datasets::internal::GetDatapointCountResponse;
use tensorzero_core::endpoints::datasets::v1::types::{
    CreateChatDatapointRequest, CreateDatapointRequest, CreateDatapointsRequest,
    CreateDatapointsResponse,
};
use tensorzero_core::inference::types::{
    Arguments, ContentBlockChatOutput, Input, InputMessage, InputMessageContent, Role, System, Text,
};
use tensorzero_core::tool::DynamicToolParams;

fn chat_datapoint(
    function_name: &str,
    user_text: &str,
    output_text: &str,
) -> CreateDatapointRequest {
    let mut system_args: Map<String, Value> = Map::new();
    system_args.insert(
        "assistant_name".to_string(),
        Value::String("StatsBot".to_string()),
    );

    CreateDatapointRequest::Chat(CreateChatDatapointRequest {
        function_name: function_name.to_string(),
        episode_id: None,
        input: Input {
            system: Some(System::Template(Arguments(system_args))),
            messages: vec![InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::Text(Text {
                    text: user_text.to_string(),
                })],
            }],
        },
        output: Some(vec![ContentBlockChatOutput::Text(Text {
            text: output_text.to_string(),
        })]),
        dynamic_tool_params: DynamicToolParams::default(),
        tags: None,
        name: None,
    })
}

async fn insert_datapoints(
    client: &Client,
    dataset_name: &str,
    datapoints: Vec<CreateDatapointRequest>,
) -> CreateDatapointsResponse {
    let payload = CreateDatapointsRequest { datapoints };

    let response = client
        .post(get_gateway_endpoint(&format!(
            "/v1/datasets/{dataset_name}/datapoints",
        )))
        .json(&payload)
        .send()
        .await
        .unwrap();

    let status = response.status();
    let body = response.text().await.unwrap();
    assert!(
        status.is_success(),
        "Failed to insert datapoints into {dataset_name}: {status} {body}"
    );

    serde_json::from_str(&body).unwrap()
}

async fn fetch_datapoint_count(
    client: &Client,
    dataset_name: &str,
    function_name: Option<&str>,
) -> GetDatapointCountResponse {
    let query_suffix = match function_name {
        Some(name) => format!("?function_name={name}"),
        None => String::new(),
    };
    let url = format!("/internal/datasets/{dataset_name}/datapoints/count{query_suffix}");

    let response = client.get(get_gateway_endpoint(&url)).send().await.unwrap();

    let status = response.status();
    let body = response.text().await.unwrap();
    assert!(
        status.is_success(),
        "Expected success status, got {status}: {body}"
    );

    serde_json::from_str(&body).unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_datapoint_count_counts_new_datapoints() {
    skip_for_postgres!();
    let client = Client::new();
    let dataset_name = format!("datapoints-count-{}", Uuid::now_v7());

    // Initial stats should succeed even before the dataset exists
    let initial_stats = fetch_datapoint_count(&client, &dataset_name, None).await;

    let inserted = insert_datapoints(
        &client,
        &dataset_name,
        vec![chat_datapoint(
            "basic_test",
            "Count this datapoint",
            "Here is one datapoint",
        )],
    )
    .await;
    assert_eq!(
        inserted.ids.len(),
        1,
        "Expected to insert exactly 1 datapoint"
    );

    // Poll until ClickHouse reflects the new datapoint
    let updated_stats = poll_result_until_some(async || {
        let stats = fetch_datapoint_count(&client, &dataset_name, None).await;
        (stats.datapoint_count == initial_stats.datapoint_count + 1).then_some(stats)
    })
    .await;

    assert_eq!(
        updated_stats.datapoint_count,
        initial_stats.datapoint_count + 1,
        "Expected datapoint_count to increase by 1 for dataset {dataset_name}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_datapoint_count_filters_by_function_name() {
    skip_for_postgres!();
    let client = Client::new();
    let dataset_name = format!("datapoints-count-filter-{}", Uuid::now_v7());

    let inserted = insert_datapoints(
        &client,
        &dataset_name,
        vec![
            chat_datapoint(
                "basic_test",
                "How many datapoints are here?",
                "Counting basic_test datapoint",
            ),
            chat_datapoint(
                "model_fallback_test",
                "Second datapoint for filtering",
                "model_fallback_test datapoint",
            ),
        ],
    )
    .await;
    assert_eq!(
        inserted.ids.len(),
        2,
        "Expected to insert exactly 2 datapoints"
    );

    // Poll until ClickHouse reflects both new datapoints
    let total_stats = poll_result_until_some(async || {
        let stats = fetch_datapoint_count(&client, &dataset_name, None).await;
        (stats.datapoint_count == 2).then_some(stats)
    })
    .await;
    assert_eq!(
        total_stats.datapoint_count, 2,
        "Expected exactly 2 datapoints for dataset {dataset_name}"
    );

    let basic_stats = fetch_datapoint_count(&client, &dataset_name, Some("basic_test")).await;
    assert_eq!(
        basic_stats.datapoint_count, 1,
        "Expected 1 datapoint for basic_test in dataset {dataset_name}"
    );

    let fallback_stats =
        fetch_datapoint_count(&client, &dataset_name, Some("model_fallback_test")).await;
    assert_eq!(
        fallback_stats.datapoint_count, 1,
        "Expected 1 datapoint for model_fallback_test in dataset {dataset_name}"
    );

    let unknown_stats =
        fetch_datapoint_count(&client, &dataset_name, Some("nonexistent_function")).await;
    assert_eq!(
        unknown_stats.datapoint_count, 0,
        "Expected 0 datapoints for nonexistent function in dataset {dataset_name}"
    );
}
