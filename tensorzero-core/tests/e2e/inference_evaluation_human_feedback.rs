use reqwest::{Client, StatusCode};
use serde_json::{json, Value};
use tensorzero_core::db::clickhouse::test_helpers::{
    select_feedback_clickhouse, select_feedback_tags_clickhouse,
    select_inference_evaluation_human_feedback_clickhouse,
};
use tokio::time::{sleep, Duration};
use uuid::Uuid;

use crate::common::get_gateway_endpoint;
use tensorzero_core::db::clickhouse::test_helpers::get_clickhouse;

// NOTE: for now inference evaluation human feedback is not supported on episodes
// or for demonstrations or comments as we don't have inference evaluations that do this yet.

#[tokio::test]
async fn e2e_test_float_human_feedback() {
    let client = Client::new();

    // Run inference (standard, no dryrun) to get an episode_id.
    let inference_payload = serde_json::json!({
        "function_name": "json_success",
        "input": {
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": [{"type": "template", "name": "user", "arguments": {"country": "Japan"}}]}]
        },
        "stream": false,
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&inference_payload)
        .send()
        .await
        .unwrap();

    assert!(response.status().is_success());
    let response_json = response.json::<Value>().await.unwrap();
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();
    let serialized_inference_output =
        serde_json::to_string(response_json.get("output").unwrap()).unwrap();
    // Test Float feedback on episode
    let datapoint_id = Uuid::now_v7();
    let payload = json!({
        "inference_id": inference_id,
        "metric_name": "brevity_score",
        "value": 32.8,
        "internal": true,
        "tags": {
            "tensorzero::human_feedback": "true",
            "tensorzero::datapoint_id": datapoint_id.to_string()
        }
    });
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let evaluator_inference_id = Uuid::now_v7();
    let payload = json!({
        "inference_id": inference_id,
        "metric_name": "brevity_score",
        "value": 32.8,
        "internal": true,
        "tags": {
            "tensorzero::human_feedback": "true",
            "tensorzero::datapoint_id": datapoint_id.to_string(),
            "tensorzero::evaluator_inference_id": evaluator_inference_id.to_string()
        }
    });
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    let feedback_id = response_json.get("feedback_id").unwrap();
    assert!(feedback_id.is_string());
    let feedback_id = Uuid::parse_str(feedback_id.as_str().unwrap()).unwrap();
    sleep(Duration::from_secs(1)).await;

    // Check ClickHouse FloatMetricFeedback
    let clickhouse = get_clickhouse().await;
    let result = select_feedback_clickhouse(&clickhouse, "FloatMetricFeedback", feedback_id)
        .await
        .unwrap();
    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, feedback_id);
    let retrieved_inference_id = result.get("target_id").unwrap().as_str().unwrap();
    let retrieved_inference_id_uuid = Uuid::parse_str(retrieved_inference_id).unwrap();
    assert_eq!(retrieved_inference_id_uuid, inference_id);
    let retrieved_value = result.get("value").unwrap().as_f64().unwrap();
    assert_eq!(retrieved_value, 32.8);
    let metric_name = result.get("metric_name").unwrap().as_str().unwrap();
    assert_eq!(metric_name, "brevity_score");

    // Check ClickHouse FeedbackTag
    let result = select_feedback_tags_clickhouse(
        &clickhouse,
        "brevity_score",
        "tensorzero::datapoint_id",
        &datapoint_id.to_string(),
    )
    .await
    .unwrap();
    let id = result.get("feedback_id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, feedback_id);
    // Check that data was written to StaticEvaluationHumanFeedback (database table name, retains "Static" prefix for backward compatibility)
    let human_feedback = select_inference_evaluation_human_feedback_clickhouse(
        &clickhouse,
        "brevity_score",
        datapoint_id,
        &serialized_inference_output,
    )
    .await
    .unwrap();
    assert_eq!(human_feedback.metric_name, "brevity_score");
    assert_eq!(human_feedback.datapoint_id, datapoint_id);
    assert_eq!(human_feedback.output, serialized_inference_output);
    assert_eq!(human_feedback.value, "32.8");
    assert_eq!(human_feedback.feedback_id, feedback_id);
}

#[tokio::test]
async fn e2e_test_boolean_human_feedback() {
    let client = Client::new();

    // Run inference (standard, no dryrun) to get an inference_id.
    let inference_payload = serde_json::json!({
        "function_name": "json_success",
        "input": {
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": [{"type": "template", "name": "user", "arguments": {"country": "Japan"}}]}]
        },
        "stream": false,
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&inference_payload)
        .send()
        .await
        .unwrap();

    assert!(response.status().is_success());
    let response_json = response.json::<Value>().await.unwrap();
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();
    let serialized_inference_output =
        serde_json::to_string(response_json.get("output").unwrap()).unwrap();
    let datapoint_id = Uuid::now_v7();
    // This should fail because we don't have an evaluator_inference_id
    let payload = json!({"inference_id": inference_id, "metric_name": "task_success", "internal": true, "value": true, "tags": {"tensorzero::human_feedback": "true", "tensorzero::datapoint_id": datapoint_id.to_string()}});
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    // No sleeping, we should throttle in the gateway
    let evaluator_inference_id = Uuid::now_v7();
    let payload = json!({"inference_id": inference_id, "metric_name": "task_success", "internal": true, "value": true, "tags": {"tensorzero::human_feedback": "true", "tensorzero::datapoint_id": datapoint_id.to_string(), "tensorzero::evaluator_inference_id": evaluator_inference_id.to_string()}});
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    let response_json = response.json::<Value>().await.unwrap();
    let feedback_id = response_json.get("feedback_id").unwrap();
    assert!(feedback_id.is_string());
    let feedback_id = Uuid::parse_str(feedback_id.as_str().unwrap()).unwrap();
    sleep(Duration::from_secs(1)).await;

    // Check ClickHouse BooleanMetricFeedback
    let clickhouse = get_clickhouse().await;
    let result = select_feedback_clickhouse(&clickhouse, "BooleanMetricFeedback", feedback_id)
        .await
        .unwrap();
    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, feedback_id);
    let retrieved_inference_id = result.get("target_id").unwrap().as_str().unwrap();
    let retrieved_inference_id_uuid = Uuid::parse_str(retrieved_inference_id).unwrap();
    assert_eq!(retrieved_inference_id_uuid, inference_id);
    let retrieved_value = result.get("value").unwrap().as_bool().unwrap();
    assert!(retrieved_value);
    let metric_name = result.get("metric_name").unwrap().as_str().unwrap();
    assert_eq!(metric_name, "task_success");

    // Check that data was written to StaticEvaluationHumanFeedback (database table name, retains "Static" prefix for backward compatibility)
    let human_feedback = select_inference_evaluation_human_feedback_clickhouse(
        &clickhouse,
        "task_success",
        datapoint_id,
        &serialized_inference_output,
    )
    .await
    .unwrap();
    assert_eq!(human_feedback.metric_name, "task_success");
    assert_eq!(human_feedback.datapoint_id, datapoint_id);
    assert_eq!(human_feedback.output, serialized_inference_output);
    assert_eq!(human_feedback.value, "true");
    assert_eq!(human_feedback.feedback_id, feedback_id);
}
