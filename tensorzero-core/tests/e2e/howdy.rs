#![expect(clippy::print_stdout)]
use crate::common::get_gateway_endpoint;
use reqwest::Client;
use reqwest::StatusCode;
use serde_json::json;
use tensorzero_core::clickhouse::test_helpers::get_clickhouse;
use tensorzero_core::howdy::{get_deployment_id, get_howdy_report};
use tokio::time::Duration;
use uuid::Uuid;

#[tokio::test(flavor = "multi_thread")]
async fn test_get_deployment_id() {
    let clickhouse = get_clickhouse().await;
    let deployment_id = get_deployment_id(&clickhouse).await.unwrap();
    println!("deployment_id: {deployment_id}");
    assert!(!deployment_id.is_empty());
}

#[tokio::test]
async fn test_get_howdy_report() {
    let client = Client::new();
    let clickhouse = get_clickhouse().await;
    let deployment_id = get_deployment_id(&clickhouse).await.unwrap();
    let howdy_report = get_howdy_report(&clickhouse, &deployment_id).await.unwrap();
    assert!(!howdy_report.inferences.is_empty());
    assert!(!howdy_report.feedbacks.is_empty());
    // Since we're in an e2e test, this should be true
    assert!(howdy_report.dryrun);
    // Send a single inference and feedback
    let payload = json!({
        "function_name": "basic_test",
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, world!"
                }
            ]
        },
        "stream": false,
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    let response_json = response.json::<serde_json::Value>().await.unwrap();
    let episode_id = response_json.get("episode_id").unwrap().as_str().unwrap();
    let episode_id = Uuid::parse_str(episode_id).unwrap();
    // Send feedback
    let feedback_payload = json!({
        "episode_id": episode_id,
        "metric_name": "comment",
        "value": "good job!",
    });
    let feedback_response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&feedback_payload)
        .send()
        .await
        .unwrap();
    assert_eq!(feedback_response.status(), StatusCode::OK);
    // Sleep for 1 second to ensure the feedback is processed
    tokio::time::sleep(Duration::from_secs(1)).await;

    // Get the howdy report again
    let new_howdy_report = get_howdy_report(&clickhouse, &deployment_id).await.unwrap();
    assert!(!new_howdy_report.inferences.is_empty());
    assert!(!new_howdy_report.feedbacks.is_empty());
    // Since we're in an e2e test, this should be true
    assert!(new_howdy_report.dryrun);
    println!("new_howdy_report: {new_howdy_report:?}");
    println!("howdy_report: {howdy_report:?}");
    // Assert that the parsed inference and feedback counts are greater than the old ones
    let old_inferences = howdy_report.inferences.parse::<u64>().unwrap();
    let old_feedbacks = howdy_report.feedbacks.parse::<u64>().unwrap();
    let new_inferences = new_howdy_report.inferences.parse::<u64>().unwrap();
    let new_feedbacks = new_howdy_report.feedbacks.parse::<u64>().unwrap();
    assert!(new_inferences > old_inferences);
    assert!(new_feedbacks > old_feedbacks);
}
