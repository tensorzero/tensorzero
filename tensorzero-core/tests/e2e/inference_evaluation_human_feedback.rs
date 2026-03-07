use googletest::prelude::*;
use reqwest::{Client, StatusCode};
use serde_json::{Value, json};
use tensorzero_core::db::delegating_connection::DelegatingDatabaseConnection;
use tensorzero_core::db::evaluation_queries::EvaluationQueries;
use tensorzero_core::db::feedback::{FeedbackQueries, FeedbackRow};
use tensorzero_core::db::test_helpers::TestDatabaseHelpers;
use tensorzero_core::serde_util::serialize_with_sorted_keys;
use uuid::Uuid;

use crate::common::get_gateway_endpoint;

// NOTE: for now inference evaluation human feedback is not supported on episodes
// or for demonstrations or comments as we don't have inference evaluations that do this yet.

#[gtest]
#[tokio::test]
async fn test_float_human_feedback() {
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

    expect_that!(response.status().is_success(), eq(true));
    let response_json = response.json::<Value>().await.unwrap();
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();
    // TODO(#6664): Make the lookup order-independent instead of requiring a particular
    // backend-dependent serialization implementation.
    let serialized_inference_output =
        serialize_with_sorted_keys(response_json.get("output").unwrap()).unwrap();
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
    expect_that!(response.status(), eq(StatusCode::BAD_REQUEST));
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
    expect_that!(response.status(), eq(StatusCode::OK));
    let response_json = response.json::<Value>().await.unwrap();
    let feedback_id = response_json.get("feedback_id").unwrap();
    expect_that!(feedback_id.is_string(), eq(true));
    let feedback_id = Uuid::parse_str(feedback_id.as_str().unwrap()).unwrap();
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;

    // Check FloatMetricFeedback
    let feedbacks = conn
        .query_feedback_by_target_id(inference_id, None, None, Some(100))
        .await
        .unwrap();
    let float_feedback = feedbacks
        .iter()
        .find_map(|f| match f {
            FeedbackRow::Float(f) if f.id == feedback_id => Some(f),
            _ => None,
        })
        .expect("Should find float feedback");
    expect_that!(float_feedback.target_id, eq(inference_id));
    expect_that!(float_feedback.value, eq(32.8));
    expect_that!(float_feedback.metric_name, eq("brevity_score"));

    // Check feedback tags
    expect_that!(
        float_feedback.tags.get("tensorzero::datapoint_id"),
        some(eq(&datapoint_id.to_string()))
    );

    // Check that data was written to inference evaluation human feedback
    let human_feedback = conn
        .get_inference_evaluation_human_feedback(
            "brevity_score",
            &datapoint_id,
            &serialized_inference_output,
        )
        .await
        .unwrap()
        .expect("Should find human feedback");
    expect_that!(&human_feedback.value, eq(&serde_json::json!(32.8)));
    expect_that!(
        human_feedback.evaluator_inference_id,
        eq(evaluator_inference_id)
    );
}

#[gtest]
#[tokio::test]
async fn test_boolean_human_feedback() {
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

    expect_that!(response.status().is_success(), eq(true));
    let response_json = response.json::<Value>().await.unwrap();
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();
    // TODO(#6664): Make the lookup order-independent instead of requiring a particular
    // backend-dependent serialization implementation.
    let serialized_inference_output =
        serialize_with_sorted_keys(response_json.get("output").unwrap()).unwrap();
    let datapoint_id = Uuid::now_v7();
    // This should fail because we don't have an evaluator_inference_id
    let payload = json!({"inference_id": inference_id, "metric_name": "task_success", "internal": true, "value": true, "tags": {"tensorzero::human_feedback": "true", "tensorzero::datapoint_id": datapoint_id.to_string()}});
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    expect_that!(response.status(), eq(StatusCode::BAD_REQUEST));
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
    expect_that!(feedback_id.is_string(), eq(true));
    let feedback_id = Uuid::parse_str(feedback_id.as_str().unwrap()).unwrap();
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;

    // Check BooleanMetricFeedback
    let feedbacks = conn
        .query_feedback_by_target_id(inference_id, None, None, Some(100))
        .await
        .unwrap();
    let boolean_feedback = feedbacks
        .iter()
        .find_map(|f| match f {
            FeedbackRow::Boolean(f) if f.id == feedback_id => Some(f),
            _ => None,
        })
        .expect("Should find boolean feedback");
    expect_that!(boolean_feedback.target_id, eq(inference_id));
    expect_that!(boolean_feedback.value, eq(true));
    expect_that!(boolean_feedback.metric_name, eq("task_success"));

    // Check that data was written to inference evaluation human feedback
    let human_feedback = conn
        .get_inference_evaluation_human_feedback(
            "task_success",
            &datapoint_id,
            &serialized_inference_output,
        )
        .await
        .unwrap()
        .expect("Should find human feedback");
    expect_that!(&human_feedback.value, eq(&serde_json::json!(true)));
    expect_that!(
        human_feedback.evaluator_inference_id,
        eq(evaluator_inference_id)
    );
}
