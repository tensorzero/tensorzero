#![expect(clippy::print_stdout)]
use std::sync::Arc;

use crate::clickhouse::get_clean_clickhouse;
use reqwest::Client;
use serde_json::json;
use tensorzero::ClientBuilder;
use tensorzero::ClientInputMessage;
use tensorzero::ClientInputMessageContent;
use tensorzero::FeedbackParams;
use tensorzero::InferenceOutput;
use tensorzero::Role;
use tensorzero::{ClientInferenceParams, ClientInput};
use tensorzero_core::clickhouse::migration_manager;
use tensorzero_core::clickhouse::test_helpers::get_clickhouse;
use tensorzero_core::clickhouse::ClickHouseConnectionInfo;
use tensorzero_core::config_parser::Config;
use tensorzero_core::gateway_util::AppStateData;
use tensorzero_core::howdy::{get_deployment_id, get_howdy_report};
use tensorzero_core::inference::types::TextKind;
use tokio::time::Duration;

#[tokio::test(flavor = "multi_thread")]
async fn test_get_deployment_id() {
    let clickhouse = get_clickhouse().await;
    let deployment_id = get_deployment_id(&clickhouse).await.unwrap();
    println!("deployment_id: {deployment_id}");
    assert!(!deployment_id.is_empty());
}

async fn get_embedded_client(clickhouse: ClickHouseConnectionInfo) -> tensorzero::Client {
    let mut config_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    config_path.push("tests/e2e/tensorzero.toml");
    let config = Arc::new(
        Config::load_from_path_optional_verify_credentials(&config_path, false)
            .await
            .unwrap(),
    );
    migration_manager::run(&clickhouse).await.unwrap();
    let state =
        AppStateData::new_with_clickhouse_and_http_client(config, clickhouse, Client::new());
    ClientBuilder::build_from_state(state).await.unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_howdy_report() {
    let (clickhouse, _guard) = get_clean_clickhouse(true);
    let client = get_embedded_client(clickhouse.clone()).await;
    tokio::time::sleep(Duration::from_secs(1)).await;
    let deployment_id = get_deployment_id(&clickhouse).await.unwrap();
    let howdy_report = get_howdy_report(&clickhouse, &deployment_id).await.unwrap();
    assert_eq!(howdy_report.inference_count, "0");
    assert_eq!(howdy_report.feedback_count, "0");
    assert_eq!(
        howdy_report.gateway_version,
        tensorzero_core::endpoints::status::TENSORZERO_VERSION
    );
    // Since we're in an e2e test, this should be true
    assert!(howdy_report.dryrun);
    // Send a chat inference and comment feedback
    let params = ClientInferenceParams {
        function_name: Some("basic_test".to_string()),
        input: ClientInput {
            system: Some(json!({"assistant_name": "AskJeeves"})),
            messages: vec![ClientInputMessage {
                role: Role::User,
                content: vec![ClientInputMessageContent::Text(TextKind::Text {
                    text: "Hello, world!".to_string(),
                })],
            }],
        },
        ..Default::default()
    };

    let InferenceOutput::NonStreaming(response) = client.inference(params).await.unwrap() else {
        panic!("Expected a non-streaming response");
    };
    // Send comment feedback
    let params = FeedbackParams {
        episode_id: Some(response.episode_id()),
        metric_name: "comment".to_string(),
        value: json!("good job!"),
        ..Default::default()
    };
    client.feedback(params).await.unwrap();
    // Send a json inference and boolean feedback
    let params = ClientInferenceParams {
        function_name: Some("json_success".to_string()),
        input: ClientInput {
            system: Some(json!({"assistant_name": "AskJeeves"})),
            messages: vec![ClientInputMessage {
                role: Role::User,
                content: vec![ClientInputMessageContent::Text(TextKind::Arguments {
                    arguments: json!({
                        "country": "Japan",
                    })
                    .as_object()
                    .unwrap()
                    .clone(),
                })],
            }],
        },
        ..Default::default()
    };

    let InferenceOutput::NonStreaming(response) = client.inference(params).await.unwrap() else {
        panic!("Expected a non-streaming response");
    };
    // Send boolean feedback
    let params = FeedbackParams {
        inference_id: Some(response.inference_id()),
        metric_name: "task_success".to_string(),
        value: json!(true),
        ..Default::default()
    };
    client.feedback(params).await.unwrap();
    // Sleep for 1 second to ensure the feedback is processed
    tokio::time::sleep(Duration::from_secs(1)).await;

    // Get the howdy report again
    let new_howdy_report = get_howdy_report(&clickhouse, &deployment_id).await.unwrap();
    assert!(!new_howdy_report.inference_count.is_empty());
    assert!(!new_howdy_report.feedback_count.is_empty());
    // Since we're in an e2e test, this should be true
    assert!(new_howdy_report.dryrun);
    println!("new_howdy_report: {new_howdy_report:?}");
    println!("howdy_report: {howdy_report:?}");
    // Assert that the parsed inference and feedback counts are greater than the old ones
    assert_eq!(new_howdy_report.inference_count, "2");
    assert_eq!(new_howdy_report.feedback_count, "2");
}
