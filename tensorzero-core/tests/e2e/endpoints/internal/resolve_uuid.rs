//! E2E tests for the resolve_uuid endpoint.

use reqwest::{Client, StatusCode};
use serde_json::{Value, json};
use tensorzero_core::db::resolve_uuid::{ResolveUuidResponse, ResolvedObject};
use uuid::Uuid;

use crate::common::get_gateway_endpoint;

/// Helper to create an inference and return its (inference_id, episode_id).
async fn create_inference() -> (Uuid, Uuid) {
    let payload = json!({
        "function_name": "basic_test",
        "episode_id": Uuid::now_v7(),
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

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::OK,
        "Inference request should succeed"
    );

    let response_json: Value = response.json().await.unwrap();
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    let episode_id = response_json.get("episode_id").unwrap().as_str().unwrap();
    let episode_id = Uuid::parse_str(episode_id).unwrap();

    // Wait for trailing writes to ClickHouse
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    (inference_id, episode_id)
}

/// Helper to create feedback and return the feedback_id.
async fn create_feedback(payload: Value) -> Uuid {
    let response = Client::new()
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::OK,
        "Feedback request should succeed"
    );

    let response_json: Value = response.json().await.unwrap();
    let feedback_id = response_json.get("feedback_id").unwrap().as_str().unwrap();
    let feedback_id = Uuid::parse_str(feedback_id).unwrap();

    // Wait for trailing writes to ClickHouse
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    feedback_id
}

/// Helper to resolve a UUID and return the response.
async fn resolve_uuid(id: &Uuid) -> ResolveUuidResponse {
    let http_client = Client::new();
    let url = get_gateway_endpoint(&format!("/internal/resolve_uuid/{id}"));

    let resp = http_client.get(url).send().await.unwrap();
    assert!(
        resp.status().is_success(),
        "resolve_uuid request failed: status={:?}",
        resp.status()
    );

    resp.json().await.unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn test_resolve_uuid_inference() {
    let (inference_id, _) = create_inference().await;

    let response = resolve_uuid(&inference_id).await;

    assert_eq!(
        response.id, inference_id,
        "Response ID should match the queried inference ID"
    );

    let has_inference = response
        .object_types
        .iter()
        .any(|obj| matches!(obj, ResolvedObject::Inference { .. }));
    assert!(
        has_inference,
        "Expected inference type in resolve_uuid response"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_resolve_uuid_episode() {
    let (_, episode_id) = create_inference().await;

    let response = resolve_uuid(&episode_id).await;

    assert_eq!(
        response.id, episode_id,
        "Response ID should match the queried episode ID"
    );

    let has_episode = response
        .object_types
        .iter()
        .any(|obj| matches!(obj, ResolvedObject::Episode));
    assert!(
        has_episode,
        "Expected episode type in resolve_uuid response"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_resolve_uuid_boolean_feedback() {
    let (inference_id, _) = create_inference().await;

    let feedback_id = create_feedback(json!({
        "inference_id": inference_id,
        "metric_name": "task_success",
        "value": true,
    }))
    .await;

    let response = resolve_uuid(&feedback_id).await;

    assert_eq!(
        response.id, feedback_id,
        "Response ID should match the queried feedback ID"
    );

    let has_boolean = response
        .object_types
        .iter()
        .any(|obj| matches!(obj, ResolvedObject::BooleanFeedback));
    assert!(
        has_boolean,
        "Expected boolean feedback type in resolve_uuid response"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_resolve_uuid_float_feedback() {
    let (inference_id, _) = create_inference().await;

    let feedback_id = create_feedback(json!({
        "inference_id": inference_id,
        "metric_name": "brevity_score",
        "value": 42.5,
    }))
    .await;

    let response = resolve_uuid(&feedback_id).await;

    assert_eq!(
        response.id, feedback_id,
        "Response ID should match the queried feedback ID"
    );

    let has_float = response
        .object_types
        .iter()
        .any(|obj| matches!(obj, ResolvedObject::FloatFeedback));
    assert!(
        has_float,
        "Expected float feedback type in resolve_uuid response"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_resolve_uuid_comment_feedback() {
    let (_, episode_id) = create_inference().await;

    let feedback_id = create_feedback(json!({
        "episode_id": episode_id,
        "metric_name": "comment",
        "value": "good job!",
    }))
    .await;

    let response = resolve_uuid(&feedback_id).await;

    assert_eq!(
        response.id, feedback_id,
        "Response ID should match the queried feedback ID"
    );

    let has_comment = response
        .object_types
        .iter()
        .any(|obj| matches!(obj, ResolvedObject::CommentFeedback));
    assert!(
        has_comment,
        "Expected comment feedback type in resolve_uuid response"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_resolve_uuid_demonstration_feedback() {
    let (inference_id, _) = create_inference().await;

    let feedback_id = create_feedback(json!({
        "inference_id": inference_id,
        "metric_name": "demonstration",
        "value": "do this!",
    }))
    .await;

    let response = resolve_uuid(&feedback_id).await;

    assert_eq!(
        response.id, feedback_id,
        "Response ID should match the queried feedback ID"
    );

    let has_demonstration = response
        .object_types
        .iter()
        .any(|obj| matches!(obj, ResolvedObject::DemonstrationFeedback));
    assert!(
        has_demonstration,
        "Expected demonstration feedback type in resolve_uuid response"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_resolve_uuid_chat_datapoint() {
    let client = Client::new();
    let dataset_name = format!("test-resolve-uuid-chat-{}", Uuid::now_v7());
    let datapoint_id = Uuid::now_v7();

    let resp = client
        .put(get_gateway_endpoint(&format!(
            "/internal/datasets/{dataset_name}/datapoints/{datapoint_id}",
        )))
        .json(&json!({
            "function_name": "basic_test",
            "input": {
                "system": {"assistant_name": "Dummy"},
                "messages": [{"role": "user", "content": [{"type": "text", "text": "test input"}]}]
            },
            "output": [{"type": "text", "text": "test output"}],
            "is_custom": true,
        }))
        .send()
        .await
        .unwrap();

    assert!(
        resp.status().is_success(),
        "Datapoint creation should succeed: {:?}",
        resp.status()
    );

    // Wait for trailing writes to ClickHouse
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    let response = resolve_uuid(&datapoint_id).await;

    assert_eq!(
        response.id, datapoint_id,
        "Response ID should match the queried datapoint ID"
    );

    let has_chat_datapoint = response
        .object_types
        .iter()
        .any(|obj| matches!(obj, ResolvedObject::ChatDatapoint { .. }));
    assert!(
        has_chat_datapoint,
        "Expected chat datapoint type in resolve_uuid response"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_resolve_uuid_json_datapoint() {
    let client = Client::new();
    let dataset_name = format!("test-resolve-uuid-json-{}", Uuid::now_v7());
    let datapoint_id = Uuid::now_v7();

    let resp = client
        .put(get_gateway_endpoint(&format!(
            "/internal/datasets/{dataset_name}/datapoints/{datapoint_id}",
        )))
        .json(&json!({
            "function_name": "json_success",
            "input": {
                "system": {"assistant_name": "Dummy"},
                "messages": [{"role": "user", "content": [{"type": "template", "name": "user", "arguments": {"country": "US"}}]}]
            },
            "output": {"answer": "Tokyo"},
            "output_schema": {},
            "is_custom": true,
        }))
        .send()
        .await
        .unwrap();

    assert!(
        resp.status().is_success(),
        "Datapoint creation should succeed: {:?}",
        resp.status()
    );

    // Wait for trailing writes to ClickHouse
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    let response = resolve_uuid(&datapoint_id).await;

    assert_eq!(
        response.id, datapoint_id,
        "Response ID should match the queried datapoint ID"
    );

    let has_json_datapoint = response
        .object_types
        .iter()
        .any(|obj| matches!(obj, ResolvedObject::JsonDatapoint { .. }));
    assert!(
        has_json_datapoint,
        "Expected json datapoint type in resolve_uuid response"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_resolve_uuid_nonexistent() {
    let nonexistent_id = Uuid::now_v7();

    let response = resolve_uuid(&nonexistent_id).await;

    assert_eq!(
        response.id, nonexistent_id,
        "Response ID should match the queried UUID"
    );
    assert!(
        response.object_types.is_empty(),
        "Expected empty object_types for nonexistent UUID"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_resolve_uuid_invalid_format() {
    let http_client = Client::new();
    let url = get_gateway_endpoint("/internal/resolve_uuid/not-a-uuid");

    let resp = http_client.get(url).send().await.unwrap();

    assert!(
        !resp.status().is_success(),
        "Expected error for invalid UUID format"
    );
}
