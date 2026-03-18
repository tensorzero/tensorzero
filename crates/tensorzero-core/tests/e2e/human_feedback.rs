use googletest::prelude::*;
use googletest_matchers::matches_json_literal;
use reqwest::{Client, StatusCode};
use serde_json::{Value, json};
use std::collections::HashMap;
use tensorzero_core::db::delegating_connection::DelegatingDatabaseConnection;
use tensorzero_core::db::evaluation_queries::EvaluationQueries;
use tensorzero_core::db::feedback::{FeedbackQueries, FeedbackRow};
use tensorzero_core::db::test_helpers::TestDatabaseHelpers;
use tensorzero_core::endpoints::feedback::Params;
use tensorzero_core::inference::types::{Arguments, JsonInferenceOutput, Role, System, Text};
use uuid::Uuid;

use crate::common::get_gateway_endpoint;
use crate::utils::poll_for_result::poll_for_result;

// TODO: make these write human feedback and make sure this is writing correctly.

#[gtest]
#[tokio::test]
async fn test_comment_human_feedback() {
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

    let inference_response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&inference_payload)
        .send()
        .await
        .unwrap();

    expect_that!(inference_response.status().is_success(), eq(true));
    let inference_response_json = inference_response.json::<Value>().await.unwrap();
    let episode_id = inference_response_json
        .get("episode_id")
        .unwrap()
        .as_str()
        .unwrap();
    let episode_id = Uuid::parse_str(episode_id).unwrap();
    let serialized_inference_output =
        serde_json::to_string(&inference_response_json.get("output").unwrap()).unwrap();

    let evaluator_id = Uuid::now_v7();

    let datapoint_id = Uuid::now_v7();
    // Test comment human evaluation feedback on episode
    let payload = Params {
        episode_id: Some(episode_id),
        inference_id: None,
        metric_name: String::from("comment"),
        value: serde_json::to_value("good job!").unwrap(),
        internal: true,
        tags: HashMap::from([
            (
                String::from("tensorzero::evaluator_inference_id"),
                evaluator_id.to_string(),
            ),
            (
                String::from("tensorzero::datapoint_id"),
                datapoint_id.to_string(),
            ),
            (
                String::from("tensorzero::human_feedback"),
                String::from("true"),
            ),
        ]),
        dryrun: None,
    };
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

    // Check CommentFeedback
    let feedbacks = poll_for_result(
        || async { conn.query_feedback_by_target_id(episode_id, None, None, Some(100)).await },
        |feedbacks| {
            feedbacks
                .iter()
                .any(|f| matches!(f, FeedbackRow::Comment(c) if c.id == feedback_id))
        },
        "human comment feedback should become visible",
    )
    .await;
    let comment_feedback = feedbacks
        .iter()
        .find_map(|f| match f {
            FeedbackRow::Comment(c) if c.id == feedback_id => Some(c),
            _ => None,
        })
        .expect("Should find comment feedback");
    expect_that!(comment_feedback.target_id, eq(episode_id));
    expect_that!(comment_feedback.value, eq("good job!"));

    // Check FeedbackTag
    expect_that!(
        comment_feedback.tags.get("tensorzero::datapoint_id"),
        some(eq(&datapoint_id.to_string()))
    );

    // Running without datapoint_id.
    // We generate a new datapoint_id so these don't trample. We'll only use it to check that there is no StaticEvaluationHumanFeedback
    // Note: table name retains "Static" prefix for backward compatibility (now called "Inference Evaluations")
    let new_datapoint_id = Uuid::now_v7();
    let payload = Params {
        episode_id: Some(episode_id),
        inference_id: None,
        metric_name: String::from("comment"),
        value: serde_json::to_value("bad job!").unwrap(),
        internal: true,
        tags: HashMap::from([(
            String::from("tensorzero::human_feedback"),
            String::from("true"),
        )]),
        dryrun: None,
    };
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

    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;

    // Check CommentFeedback
    let feedbacks = poll_for_result(
        || async { conn.query_feedback_by_target_id(episode_id, None, None, Some(100)).await },
        |feedbacks| {
            feedbacks
                .iter()
                .any(|f| matches!(f, FeedbackRow::Comment(c) if c.id == feedback_id))
        },
        "second human comment feedback should become visible",
    )
    .await;
    let comment_feedback = feedbacks
        .iter()
        .find_map(|f| match f {
            FeedbackRow::Comment(c) if c.id == feedback_id => Some(c),
            _ => None,
        })
        .expect("Should find comment feedback");
    expect_that!(comment_feedback.value, eq("bad job!"));

    // Check that there is no inference evaluation human feedback
    let result = conn
        .get_inference_evaluation_human_feedback(
            "comment",
            &new_datapoint_id,
            &serialized_inference_output,
        )
        .await
        .unwrap();
    expect_that!(result, none());
}

#[gtest]
#[tokio::test]
async fn test_demonstration_feedback() {
    let client = Client::new();
    // Running without valid inference_id. Should fail.
    let tag_value = Uuid::now_v7().to_string();
    let inference_id = Uuid::now_v7();
    let payload = Params {
        inference_id: Some(inference_id),
        episode_id: None,
        metric_name: String::from("demonstration"),
        value: serde_json::to_value("do this!").unwrap(),
        tags: HashMap::from([(String::from("key"), tag_value.clone())]),
        dryrun: None,
        internal: false,
    };
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    expect_that!(response.status(), eq(StatusCode::BAD_REQUEST));
    // Run inference (standard, no dryrun) to get an inference_id
    let inference_payload = serde_json::json!({
        "function_name": "basic_test",
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [{"role": "user", "content": "Hello, world!"}]
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

    // No sleeping, we should throttle in the gateway
    // Test demonstration feedback on Inference
    let tag_value = Uuid::now_v7().to_string();
    let payload = Params {
        inference_id: Some(inference_id),
        episode_id: None,
        metric_name: String::from("demonstration"),
        value: serde_json::to_value("do this!").unwrap(),
        tags: HashMap::from([(String::from("key"), tag_value.clone())]),
        dryrun: None,
        internal: false,
    };
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

    // Check DemonstrationFeedback
    let feedbacks = poll_for_result(
        || async { conn.query_feedback_by_target_id(inference_id, None, None, Some(100)).await },
        |feedbacks| {
            feedbacks.iter().any(
                |f| matches!(f, FeedbackRow::Demonstration(d) if d.id == feedback_id),
            )
        },
        "human dynamic demonstration feedback should become visible",
    )
    .await;
    let demo_feedback = feedbacks
        .iter()
        .find_map(|f| match f {
            FeedbackRow::Demonstration(d) if d.id == feedback_id => Some(d),
            _ => None,
        })
        .expect("Should find demonstration feedback");
    expect_that!(demo_feedback.inference_id, eq(inference_id));
    let retrieved_value: Value = serde_json::from_str(&demo_feedback.value).unwrap();
    expect_that!(
        retrieved_value,
        matches_json_literal!([{"type": "text", "text": "do this!"}])
    );

    // Check FeedbackTag
    expect_that!(demo_feedback.tags.get("key"), some(eq(&tag_value)));

    // Try it for an episode (should 400)
    let episode_id = Uuid::now_v7();
    let payload = Params {
        episode_id: Some(episode_id),
        inference_id: None,
        metric_name: String::from("demonstration"),
        value: serde_json::to_value("do this!").unwrap(),
        tags: HashMap::new(),
        dryrun: None,
        internal: false,
    };
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    expect_that!(response.status(), eq(StatusCode::BAD_REQUEST));
    let response_json = response.json::<Value>().await.unwrap();
    let message = response_json.get("error").unwrap().as_str().unwrap();
    expect_that!(
        message,
        eq("Correct ID was not provided for feedback level \"inference\".")
    );

    // Try a tool call demonstration
    // This should fail because the inference was made for a function that doesn't support tool calls
    let tool_call = json!({"type": "tool_call", "id": "tool_call_id", "name": "tool_name", "arguments": "tool_input"});
    let payload = Params {
        inference_id: Some(inference_id),
        episode_id: None,
        metric_name: String::from("demonstration"),
        value: serde_json::to_value(vec![tool_call]).unwrap(),
        tags: HashMap::new(),
        dryrun: None,
        internal: false,
    };
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    expect_that!(response.status(), eq(StatusCode::BAD_REQUEST));
    let response_json = response.json::<Value>().await.unwrap();
    let expected_json = json!({
        "error": "Demonstration contains invalid tool name",
        "error_json": {
            "InvalidRequest": {
                "message": "Demonstration contains invalid tool name"
            }
        }
    });
    expect_that!(&response_json, eq(&expected_json));
}

#[gtest]
#[tokio::test]
async fn test_demonstration_feedback_json() {
    let client = Client::new();
    // Running without valid inference_id. Should fail.
    let inference_id = Uuid::now_v7();
    let payload = Params {
        inference_id: Some(inference_id),
        episode_id: None,
        metric_name: String::from("demonstration"),
        value: json!({"answer": "Tokyo"}),
        tags: HashMap::new(),
        dryrun: None,
        internal: false,
    };
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    expect_that!(response.status(), eq(StatusCode::BAD_REQUEST));
    // Run inference (standard, no dryrun) to get an inference_id
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

    // No sleeping, we should throttle in the gateway
    // Test demonstration feedback on an inference
    let payload = Params {
        inference_id: Some(inference_id),
        episode_id: None,
        metric_name: String::from("demonstration"),
        value: json!({"answer": "Tokyo"}),
        tags: HashMap::new(),
        dryrun: None,
        internal: false,
    };
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

    // Check DemonstrationFeedback
    let feedbacks = poll_for_result(
        || async { conn.query_feedback_by_target_id(inference_id, None, None, Some(100)).await },
        |feedbacks| {
            feedbacks.iter().any(
                |f| matches!(f, FeedbackRow::Demonstration(d) if d.id == feedback_id),
            )
        },
        "human dynamic tool demonstration feedback should become visible",
    )
    .await;
    let demo_feedback = feedbacks
        .iter()
        .find_map(|f| match f {
            FeedbackRow::Demonstration(d) if d.id == feedback_id => Some(d),
            _ => None,
        })
        .expect("Should find demonstration feedback");
    expect_that!(demo_feedback.inference_id, eq(inference_id));
    let retrieved_value =
        serde_json::from_str::<JsonInferenceOutput>(&demo_feedback.value).unwrap();
    let expected_value = JsonInferenceOutput {
        parsed: Some(json!({"answer": "Tokyo"})),
        raw: Some("{\"answer\":\"Tokyo\"}".to_string()),
    };
    expect_that!(&retrieved_value, eq(&expected_value));

    // Try it for an episode (should 400)
    let episode_id = Uuid::now_v7();
    let payload = Params {
        episode_id: Some(episode_id),
        inference_id: None,
        metric_name: String::from("demonstration"),
        value: serde_json::to_value("do this!").unwrap(),
        tags: HashMap::new(),
        dryrun: None,
        internal: false,
    };
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    expect_that!(response.status(), eq(StatusCode::BAD_REQUEST));
    let response_json = response.json::<Value>().await.unwrap();
    let message = response_json.get("error").unwrap().as_str().unwrap();
    expect_that!(
        message,
        eq("Correct ID was not provided for feedback level \"inference\".")
    );

    // Try a tool call demonstration
    // This should fail because the inference was made for a function that doesn't support tool calls
    let tool_call = json!({"type": "tool_call", "id": "tool_call_id", "name": "tool_name", "arguments": "tool_input"});
    let payload = Params {
        inference_id: Some(inference_id),
        episode_id: None,
        metric_name: String::from("demonstration"),
        value: serde_json::to_value(vec![tool_call]).unwrap(),
        tags: HashMap::new(),
        dryrun: None,
        internal: false,
    };
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    expect_that!(response.status(), eq(StatusCode::BAD_REQUEST));
    let response_json = response.json::<Value>().await.unwrap();
    let error_message = response_json.get("error").unwrap().as_str().unwrap();
    expect_that!(
        error_message,
        starts_with("Demonstration does not fit function output schema:")
    );
}

#[gtest]
#[tokio::test]
async fn test_demonstration_feedback_dynamic_json() {
    let client = Client::new();
    // Running without valid inference_id. Should fail.
    let inference_id = Uuid::now_v7();
    let payload = Params {
        inference_id: Some(inference_id),
        episode_id: None,
        metric_name: String::from("demonstration"),
        value: json!({"answer": "Tokyo"}),
        tags: HashMap::new(),
        dryrun: None,
        internal: false,
    };
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    expect_that!(response.status(), eq(StatusCode::BAD_REQUEST));
    // Run inference (standard, no dryrun) to get an inference_id
    let new_output_schema = json!({
        "type": "object",
        "properties": {
            "answer": {
                "type": "string"
            },
            "comment": {
                "type": "string",
            }
        },
        "required": ["answer"],
        "additionalProperties": false
    });
    let inference_payload = serde_json::json!({
        "function_name": "json_success",
        "input": {
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": [{"type": "template", "name": "user", "arguments": {"country": "Japan"}}]}]
        },
        "stream": false,
        "output_schema": new_output_schema,
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

    // No sleeping, we should throttle in the gateway
    // Test demonstration feedback on an inference that requires the dynamic output schema
    let payload = Params {
        inference_id: Some(inference_id),
        episode_id: None,
        metric_name: String::from("demonstration"),
        value: json!({"answer": "Tokyo", "comment": "This is a comment"}),
        tags: HashMap::new(),
        dryrun: None,
        internal: false,
    };
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

    // Check DemonstrationFeedback
    let feedbacks = conn
        .query_feedback_by_target_id(inference_id, None, None, Some(100))
        .await
        .unwrap();
    let demo_feedback = feedbacks
        .iter()
        .find_map(|f| match f {
            FeedbackRow::Demonstration(d) if d.id == feedback_id => Some(d),
            _ => None,
        })
        .expect("Should find demonstration feedback");
    expect_that!(demo_feedback.inference_id, eq(inference_id));
    let retrieved_value =
        serde_json::from_str::<JsonInferenceOutput>(&demo_feedback.value).unwrap();
    let expected_value = JsonInferenceOutput {
        parsed: Some(json!({"answer": "Tokyo", "comment": "This is a comment"})),
        raw: Some("{\"answer\":\"Tokyo\",\"comment\":\"This is a comment\"}".to_string()),
    };
    expect_that!(&retrieved_value, eq(&expected_value));

    // Try it for an episode (should 400)
    let episode_id = Uuid::now_v7();
    let payload = Params {
        episode_id: Some(episode_id),
        inference_id: None,
        metric_name: String::from("demonstration"),
        value: serde_json::to_value("do this!").unwrap(),
        tags: HashMap::new(),
        dryrun: None,
        internal: false,
    };
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    expect_that!(response.status(), eq(StatusCode::BAD_REQUEST));
    let response_json = response.json::<Value>().await.unwrap();
    let message = response_json.get("error").unwrap().as_str().unwrap();
    expect_that!(
        message,
        eq("Correct ID was not provided for feedback level \"inference\".")
    );

    // Try a tool call demonstration
    // This should fail because the inference was made for a function that doesn't support tool calls
    let tool_call = json!({"type": "tool_call", "id": "tool_call_id", "name": "tool_name", "arguments": "tool_input"});
    let payload = Params {
        inference_id: Some(inference_id),
        episode_id: None,
        metric_name: String::from("demonstration"),
        value: serde_json::to_value(vec![tool_call]).unwrap(),
        tags: HashMap::new(),
        dryrun: None,
        internal: false,
    };
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    expect_that!(response.status(), eq(StatusCode::BAD_REQUEST));
    let response_json = response.json::<Value>().await.unwrap();
    let error_message = response_json.get("error").unwrap().as_str().unwrap();
    expect_that!(
        error_message,
        starts_with("Demonstration does not fit function output schema:")
    );

    // Try a demonstration with a value that doesn't match the output schema
    let payload = Params {
        inference_id: Some(inference_id),
        episode_id: None,
        metric_name: String::from("demonstration"),
        value: json!({"bad_key": "Tokyo"}),
        tags: HashMap::new(),
        dryrun: None,
        internal: false,
    };
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    expect_that!(response.status(), eq(StatusCode::BAD_REQUEST));
    let response_json = response.json::<Value>().await.unwrap();
    let error_message = response_json.get("error").unwrap().as_str().unwrap();
    expect_that!(
        error_message,
        starts_with("Demonstration does not fit function output schema:")
    );
}

#[gtest]
#[tokio::test]
async fn test_demonstration_feedback_tool() {
    // Running without valid inference_id. Should fail.
    let client = Client::new();
    let inference_id = Uuid::now_v7();
    let payload = Params {
        inference_id: Some(inference_id),
        episode_id: None,
        metric_name: String::from("demonstration"),
        value: serde_json::to_value("sunny").unwrap(),
        tags: HashMap::new(),
        dryrun: None,
        internal: false,
    };
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    expect_that!(response.status(), eq(StatusCode::BAD_REQUEST));
    // Run inference (standard, no dryrun) to get an inference_id
    let inference_payload = serde_json::json!({
        "function_name": "weather_helper",
        "input": {
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}]
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

    // No sleeping, we should throttle in the gateway
    // Test demonstration feedback on Inference (string shortcut)
    let payload = Params {
        inference_id: Some(inference_id),
        episode_id: None,
        metric_name: String::from("demonstration"),
        value: serde_json::to_value("sunny").unwrap(),
        tags: HashMap::new(),
        dryrun: None,
        internal: false,
    };
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

    // Check DemonstrationFeedback
    let feedbacks = conn
        .query_feedback_by_target_id(inference_id, None, None, Some(100))
        .await
        .unwrap();
    let demo_feedback = feedbacks
        .iter()
        .find_map(|f| match f {
            FeedbackRow::Demonstration(d) if d.id == feedback_id => Some(d),
            _ => None,
        })
        .expect("Should find demonstration feedback");
    expect_that!(demo_feedback.inference_id, eq(inference_id));
    let retrieved_value: Value = serde_json::from_str(&demo_feedback.value).unwrap();
    expect_that!(
        retrieved_value,
        matches_json_literal!([{"type": "text", "text": "sunny"}])
    );

    // Try it for an episode (should 400)
    let episode_id = Uuid::now_v7();
    let payload = Params {
        episode_id: Some(episode_id),
        inference_id: None,
        metric_name: String::from("demonstration"),
        value: serde_json::to_value("do this!").unwrap(),
        tags: HashMap::new(),
        dryrun: None,
        internal: false,
    };
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    expect_that!(response.status(), eq(StatusCode::BAD_REQUEST));
    let response_json = response.json::<Value>().await.unwrap();
    let message = response_json.get("error").unwrap().as_str().unwrap();
    expect_that!(
        message,
        eq("Correct ID was not provided for feedback level \"inference\".")
    );

    // Try a tool call demonstration
    // This should fail because the name is incorrect
    let tool_call = json!({"type": "tool_call", "id": "tool_call_id", "name": "tool_name", "arguments": "tool_input"});
    let payload = Params {
        inference_id: Some(inference_id),
        episode_id: None,
        metric_name: String::from("demonstration"),
        value: serde_json::to_value(vec![tool_call]).unwrap(),
        tags: HashMap::new(),
        dryrun: None,
        internal: false,
    };
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    expect_that!(response.status(), eq(StatusCode::BAD_REQUEST));
    let response_json = response.json::<Value>().await.unwrap();
    let error_message = response_json.get("error").unwrap().as_str().unwrap();
    expect_that!(
        error_message,
        eq("Demonstration contains invalid tool name")
    );

    // Try a tool call demonstration with correct name incorrect args
    let tool_call = json!({"type": "tool_call", "id": "tool_call_id", "name": "get_temperature", "arguments": "tool_input"});
    let payload = Params {
        inference_id: Some(inference_id),
        episode_id: None,
        metric_name: String::from("demonstration"),
        value: serde_json::to_value(vec![tool_call]).unwrap(),
        tags: HashMap::new(),
        dryrun: None,
        internal: false,
    };
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    expect_that!(response.status(), eq(StatusCode::BAD_REQUEST));
    let response_json = response.json::<Value>().await.unwrap();
    let error_message = response_json.get("error").unwrap().as_str().unwrap();
    expect_that!(
        error_message,
        eq("Demonstration contains invalid tool call arguments")
    );

    // Try a tool call demonstration with correct name and args
    let tool_call = json!({"type": "tool_call", "id": "tool_call_id", "name": "get_temperature", "arguments": {"location": "Tokyo", "units": "celsius"}});
    let payload = Params {
        inference_id: Some(inference_id),
        episode_id: None,
        metric_name: String::from("demonstration"),
        value: serde_json::to_value(vec![tool_call]).unwrap(),
        tags: HashMap::new(),
        dryrun: None,
        internal: false,
    };
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

    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;

    // Check DemonstrationFeedback (tool call)
    let feedbacks = conn
        .query_feedback_by_target_id(inference_id, None, None, Some(100))
        .await
        .unwrap();
    let demo_feedback = feedbacks
        .iter()
        .find_map(|f| match f {
            FeedbackRow::Demonstration(d) if d.id == feedback_id => Some(d),
            _ => None,
        })
        .expect("Should find demonstration feedback");
    expect_that!(demo_feedback.inference_id, eq(inference_id));
    let retrieved_value = serde_json::from_str::<Value>(&demo_feedback.value).unwrap();
    let expected_value = json!([{"type": "tool_call", "id": "tool_call_id", "raw_name": "get_temperature", "raw_arguments": "{\"location\":\"Tokyo\",\"units\":\"celsius\"}", "name": "get_temperature", "arguments": {"location": "Tokyo", "units": "celsius"}}]);
    expect_that!(&retrieved_value, eq(&expected_value));
}

#[gtest]
#[tokio::test]
async fn test_demonstration_feedback_dynamic_tool() {
    let client = Client::new();

    // Run inference (standard, no dryrun) to get an inference_id
    let inference_payload = serde_json::json!({
        "function_name": "weather_helper",
        "input": {
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}]
        },
        "stream": false,
        "additional_tools": [
            {
                "name": "get_humidity",
                "description": "Get the current humidity in a given location",
                "parameters": json!({
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"],
                    "additionalProperties": false
                })
            }
        ]
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

    // No sleeping, we should throttle in the gateway
    // Test demonstration feedback on Inference (string shortcut)
    let payload = Params {
        inference_id: Some(inference_id),
        episode_id: None,
        metric_name: String::from("demonstration"),
        value: serde_json::to_value("sunny").unwrap(),
        tags: HashMap::new(),
        dryrun: None,
        internal: false,
    };
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

    // Check DemonstrationFeedback
    let feedbacks = conn
        .query_feedback_by_target_id(inference_id, None, None, Some(100))
        .await
        .unwrap();
    let demo_feedback = feedbacks
        .iter()
        .find_map(|f| match f {
            FeedbackRow::Demonstration(d) if d.id == feedback_id => Some(d),
            _ => None,
        })
        .expect("Should find demonstration feedback");
    expect_that!(demo_feedback.inference_id, eq(inference_id));
    let retrieved_value: Value = serde_json::from_str(&demo_feedback.value).unwrap();
    expect_that!(
        retrieved_value,
        matches_json_literal!([{"type": "text", "text": "sunny"}])
    );

    // Try it for an episode (should 400)
    let episode_id = Uuid::now_v7();
    let payload = Params {
        episode_id: Some(episode_id),
        inference_id: None,
        metric_name: String::from("demonstration"),
        value: serde_json::to_value("do this!").unwrap(),
        tags: HashMap::new(),
        dryrun: None,
        internal: false,
    };
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    expect_that!(response.status(), eq(StatusCode::BAD_REQUEST));
    let response_json = response.json::<Value>().await.unwrap();
    let message = response_json.get("error").unwrap().as_str().unwrap();
    expect_that!(
        message,
        eq("Correct ID was not provided for feedback level \"inference\".")
    );

    // Try a tool call demonstration
    // This should fail because the name is incorrect
    let tool_call = json!({"type": "tool_call", "id": "tool_call_id", "name": "tool_name", "arguments": "tool_input"});
    let payload = Params {
        inference_id: Some(inference_id),
        episode_id: None,
        metric_name: String::from("demonstration"),
        value: serde_json::to_value(vec![tool_call]).unwrap(),
        tags: HashMap::new(),
        dryrun: None,
        internal: false,
    };
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    expect_that!(response.status(), eq(StatusCode::BAD_REQUEST));
    let response_json = response.json::<Value>().await.unwrap();
    let error_message = response_json.get("error").unwrap().as_str().unwrap();
    expect_that!(
        error_message,
        eq("Demonstration contains invalid tool name")
    );

    // Try a tool call demonstration with the dynamic tool name and incorrect args
    let tool_call = json!({"type": "tool_call", "id": "tool_call_id", "name": "get_humidity", "arguments": "tool_input"});
    let payload = Params {
        inference_id: Some(inference_id),
        episode_id: None,
        metric_name: String::from("demonstration"),
        value: serde_json::to_value(vec![tool_call]).unwrap(),
        tags: HashMap::new(),
        dryrun: None,
        internal: false,
    };
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    expect_that!(response.status(), eq(StatusCode::BAD_REQUEST));
    let response_json = response.json::<Value>().await.unwrap();
    let error_message = response_json.get("error").unwrap().as_str().unwrap();
    expect_that!(
        error_message,
        eq("Demonstration contains invalid tool call arguments")
    );

    // Try a tool call demonstration with the dynamic tool name and correct args
    let tool_call = json!({"type": "tool_call", "id": "tool_call_id", "name": "get_humidity", "arguments": {"location": "Tokyo"}});
    let payload = Params {
        inference_id: Some(inference_id),
        episode_id: None,
        metric_name: String::from("demonstration"),
        value: serde_json::to_value(vec![tool_call]).unwrap(),
        tags: HashMap::new(),
        dryrun: None,
        internal: false,
    };
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

    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;

    // Check DemonstrationFeedback (dynamic tool call)
    let feedbacks = conn
        .query_feedback_by_target_id(inference_id, None, None, Some(100))
        .await
        .unwrap();
    let demo_feedback = feedbacks
        .iter()
        .find_map(|f| match f {
            FeedbackRow::Demonstration(d) if d.id == feedback_id => Some(d),
            _ => None,
        })
        .expect("Should find demonstration feedback");
    expect_that!(demo_feedback.inference_id, eq(inference_id));
    let retrieved_value = serde_json::from_str::<Value>(&demo_feedback.value).unwrap();
    let expected_value = json!([{"type": "tool_call", "id": "tool_call_id", "raw_name": "get_humidity", "raw_arguments": "{\"location\":\"Tokyo\"}", "name": "get_humidity", "arguments": {"location": "Tokyo"}}]);
    expect_that!(&retrieved_value, eq(&expected_value));
}

#[gtest]
#[tokio::test]
async fn test_float_feedback() {
    let client = Client::new();
    let tag_value = Uuid::now_v7().to_string();
    // Running without valid episode_id. Should fail.
    let episode_id = Uuid::now_v7();
    let payload = Params {
        episode_id: Some(episode_id),
        inference_id: None,
        metric_name: String::from("user_rating"),
        value: serde_json::to_value(32.8).unwrap(),
        tags: HashMap::from([(String::from("key"), tag_value.clone())]),
        dryrun: None,
        internal: false,
    };
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    expect_that!(response.status(), eq(StatusCode::BAD_REQUEST));
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
    let episode_id = response_json.get("episode_id").unwrap().as_str().unwrap();
    let episode_id = Uuid::parse_str(episode_id).unwrap();
    // Test Float feedback on episode
    let payload = Params {
        episode_id: Some(episode_id),
        inference_id: None,
        metric_name: String::from("user_rating"),
        value: serde_json::to_value(32.8).unwrap(),
        tags: HashMap::from([(String::from("key"), tag_value.clone())]),
        dryrun: None,
        internal: false,
    };
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
        .query_feedback_by_target_id(episode_id, None, None, Some(100))
        .await
        .unwrap();
    let float_feedback = feedbacks
        .iter()
        .find_map(|f| match f {
            FeedbackRow::Float(f) if f.id == feedback_id => Some(f),
            _ => None,
        })
        .expect("Should find float feedback");
    expect_that!(float_feedback.target_id, eq(episode_id));
    expect_that!(float_feedback.value, eq(32.8));
    expect_that!(float_feedback.metric_name, eq("user_rating"));

    // Check FeedbackTag
    expect_that!(float_feedback.tags.get("key"), some(eq(&tag_value)));

    // Test boolean feedback on episode (should fail)
    let payload = Params {
        episode_id: Some(episode_id),
        inference_id: None,
        metric_name: String::from("user_rating"),
        value: serde_json::to_value(true).unwrap(),
        tags: HashMap::new(),
        dryrun: None,
        internal: false,
    };
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    expect_that!(response.status(), eq(StatusCode::BAD_REQUEST));
    let response_json = response.json::<Value>().await.unwrap();
    let error_message = response_json.get("error").unwrap().as_str().unwrap();
    expect_that!(
        error_message,
        eq("Feedback value for metric `user_rating` must be a number")
    );

    // Test float feedback on inference (should fail)
    let inference_id = Uuid::now_v7();
    let payload = Params {
        inference_id: Some(inference_id),
        episode_id: None,
        metric_name: String::from("user_rating"),
        value: serde_json::to_value(4.5).unwrap(),
        tags: HashMap::new(),
        dryrun: None,
        internal: false,
    };
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    expect_that!(response.status(), eq(StatusCode::BAD_REQUEST));
    let response_json = response.json::<Value>().await.unwrap();
    let error_message = response_json.get("error").unwrap().as_str().unwrap();
    expect_that!(
        error_message,
        contains_substring("Correct ID was not provided for feedback level"),
        "Unexpected error message: {error_message}"
    );

    // Running without valid inference_id. Should fail.
    let inference_id = Uuid::now_v7();
    let payload = Params {
        inference_id: Some(inference_id),
        episode_id: None,
        metric_name: String::from("brevity_score"),
        value: serde_json::to_value(0.5).unwrap(),
        tags: HashMap::new(),
        dryrun: None,
        internal: false,
    };
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    expect_that!(response.status(), eq(StatusCode::BAD_REQUEST));

    // Run inference (standard, no dryrun) to get an inference_id
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

    // Sleep longer than the duration of the feedback cooldown period (5s)
    // to make sure that the feedback is written after the inference.
    tokio::time::sleep(std::time::Duration::from_millis(5500)).await;

    // Test float feedback on different metric for inference.
    let payload = Params {
        inference_id: Some(inference_id),
        episode_id: None,
        metric_name: String::from("brevity_score"),
        value: serde_json::to_value(0.5).unwrap(),
        tags: HashMap::new(),
        dryrun: None,
        internal: false,
    };
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
    expect_that!(float_feedback.value, eq(0.5));
    expect_that!(float_feedback.metric_name, eq("brevity_score"));
}

#[gtest]
#[tokio::test]
async fn test_boolean_feedback() {
    let client = Client::new();
    let inference_id = Uuid::now_v7();
    let tag_value = Uuid::now_v7().to_string();
    let tag_value2 = Uuid::now_v7().to_string();
    // Running without valid inference_id. Should fail.
    let payload = Params {
        inference_id: Some(inference_id),
        episode_id: None,
        metric_name: String::from("task_success"),
        value: serde_json::to_value(true).unwrap(),
        tags: HashMap::from([
            (String::from("key"), tag_value.clone()),
            (String::from("key2"), tag_value2.clone()),
        ]),
        dryrun: None,
        internal: false,
    };
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    expect_that!(response.status(), eq(StatusCode::BAD_REQUEST));
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

    // No sleeping, we should throttle in the gateway

    let payload = Params {
        inference_id: Some(inference_id),
        episode_id: None,
        metric_name: String::from("task_success"),
        value: serde_json::to_value(true).unwrap(),
        tags: HashMap::from([
            (String::from("key"), tag_value.clone()),
            (String::from("key2"), tag_value2.clone()),
        ]),
        dryrun: None,
        internal: false,
    };
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
    let bool_feedback = feedbacks
        .iter()
        .find_map(|f| match f {
            FeedbackRow::Boolean(b) if b.id == feedback_id => Some(b),
            _ => None,
        })
        .expect("Should find boolean feedback");
    expect_that!(bool_feedback.target_id, eq(inference_id));
    expect_that!(bool_feedback.value, eq(true));
    expect_that!(bool_feedback.metric_name, eq("task_success"));

    // Check FeedbackTags
    expect_that!(bool_feedback.tags.get("key"), some(eq(&tag_value)));
    expect_that!(bool_feedback.tags.get("key2"), some(eq(&tag_value2)));

    // Try episode-level feedback (should fail)
    let episode_id = Uuid::now_v7();
    let payload = Params {
        episode_id: Some(episode_id),
        inference_id: None,
        metric_name: String::from("task_success"),
        value: serde_json::to_value(true).unwrap(),
        tags: HashMap::new(),
        dryrun: None,
        internal: false,
    };
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    expect_that!(response.status(), eq(StatusCode::BAD_REQUEST));
    let response_json = response.json::<Value>().await.unwrap();
    let error_message = response_json.get("error").unwrap().as_str().unwrap();
    expect_that!(
        error_message,
        contains_substring("Correct ID was not provided for feedback level"),
        "Unexpected error message: {error_message}"
    );

    // Try string feedback (should fail)
    let payload = Params {
        inference_id: Some(inference_id),
        episode_id: None,
        metric_name: String::from("task_success"),
        value: serde_json::to_value("true").unwrap(),
        tags: HashMap::new(),
        dryrun: None,
        internal: false,
    };
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    expect_that!(response.status(), eq(StatusCode::BAD_REQUEST));
    let response_json = response.json::<Value>().await.unwrap();
    let error_message = response_json.get("error").unwrap().as_str().unwrap();
    expect_that!(
        error_message,
        eq("Feedback value for metric `task_success` must be a boolean")
    );

    // Try episode-level feedback on different metric with invalid episode id.
    let episode_id = Uuid::now_v7();
    let payload = Params {
        episode_id: Some(episode_id),
        inference_id: None,
        metric_name: String::from("goal_achieved"),
        value: serde_json::to_value(true).unwrap(),
        tags: HashMap::new(),
        dryrun: None,
        internal: false,
    };
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    expect_that!(response.status(), eq(StatusCode::BAD_REQUEST));
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
    let episode_id = response_json.get("episode_id").unwrap().as_str().unwrap();
    let episode_id = Uuid::parse_str(episode_id).unwrap();

    let payload = Params {
        episode_id: Some(episode_id),
        inference_id: None,
        metric_name: String::from("goal_achieved"),
        value: serde_json::to_value(true).unwrap(),
        tags: HashMap::new(),
        dryrun: None,
        internal: false,
    };
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

    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;

    // Check BooleanMetricFeedback (episode-level)
    let feedbacks = conn
        .query_feedback_by_target_id(episode_id, None, None, Some(100))
        .await
        .unwrap();
    let bool_feedback = feedbacks
        .iter()
        .find_map(|f| match f {
            FeedbackRow::Boolean(b) if b.id == feedback_id => Some(b),
            _ => None,
        })
        .expect("Should find boolean feedback");
    expect_that!(bool_feedback.target_id, eq(episode_id));
    expect_that!(bool_feedback.value, eq(true));
    expect_that!(bool_feedback.metric_name, eq("goal_achieved"));
}

#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_fast_inference_then_feedback() {
    let logs_contain = tensorzero_core::utils::testing::capture_logs();
    use serde_json::json;
    use std::collections::HashMap;
    use std::sync::Arc;
    // Create the client and wrap it in an Arc for shared ownership.
    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let client = Arc::new(client);

    // Create a collection of tasks, each making an inference then a feedback call.
    let tasks: Vec<_> = (0..20)
        .map(|_| {
            let client = Arc::clone(&client);
            // TODO(https://github.com/tensorzero/tensorzero/issues/3983): Audit this callsite
            #[expect(clippy::disallowed_methods)]
            tokio::spawn(async move {
                let inference_payload = tensorzero::ClientInferenceParams {
                    function_name: Some("basic_test".to_string()),
                    model_name: None,
                    variant_name: None,
                    episode_id: None,
                    input: tensorzero::Input {
                        system: Some(System::Template(Arguments(
                            json!({"assistant_name": "Alfred Pennyworth"})
                                .as_object()
                                .unwrap()
                                .clone(),
                        ))),
                        messages: vec![tensorzero::InputMessage {
                            role: Role::User,
                            content: vec![tensorzero::InputMessageContent::Text(Text {
                                text: "What is the weather like in Tokyo (in Celsius)? Use the provided `get_temperature` tool. Do not say anything else, just call the function."
                                    .to_string()
                            })],
                        }],
                    },
                    stream: Some(false),
                    ..Default::default()
                };

                // Send the inference request.
                let response = client.inference(inference_payload).await.unwrap();
                let response = if let tensorzero::InferenceOutput::NonStreaming(response) = response {
                    response
                } else {
                    panic!("Expected non-streaming response");
                };
                let response = if let tensorzero::InferenceResponse::Chat(response) = response {
                    response
                } else {
                    panic!("Expected chat response");
                };
                let inference_id = response.inference_id;

                // Prepare and send the feedback request.
                // This also tests that the internal flag is correctly propagated.
                let feedback_payload = tensorzero::FeedbackParams {
                    inference_id: Some(inference_id),
                    episode_id: None,
                    metric_name: "task_success".to_string(),
                    value: json!(true),
                    internal: true,
                    tags: HashMap::from([("tensorzero::tag_key".to_string(), "tensorzero::tag_value".to_string())]),
                    dryrun: None,
                };
                client.feedback(feedback_payload).await.unwrap();
            })
        })
        .collect();

    // Wait for all tasks to finish.
    futures::future::join_all(tasks).await;
    expect_that!(logs_contain("does not exist"), eq(false));
}
