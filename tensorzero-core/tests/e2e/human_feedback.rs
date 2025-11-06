use reqwest::{Client, StatusCode};
use serde_json::{json, Value};
use tensorzero_core::{
    db::clickhouse::test_helpers::{
        select_feedback_clickhouse, select_feedback_tags_clickhouse,
        select_inference_evaluation_human_feedback_clickhouse,
    },
    inference::types::{
        Arguments, ContentBlockChatOutput, JsonInferenceOutput, Role, System, Text, TextKind,
    },
};
use tokio::time::{sleep, Duration};
use uuid::Uuid;

use crate::common::get_gateway_endpoint;
use tensorzero_core::db::clickhouse::test_helpers::get_clickhouse;

// TODO: make these write human feedback and make sure this is writing correctly.

#[tokio::test]
async fn e2e_test_comment_human_feedback() {
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

    assert!(inference_response.status().is_success());
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
    let payload = json!({"episode_id": episode_id, "metric_name": "comment", "value": "good job!", "internal": true, "tags": {"tensorzero::evaluator_inference_id": evaluator_id.to_string(), "tensorzero::datapoint_id": datapoint_id.to_string(), "tensorzero::human_feedback": "true"}});
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
    sleep(Duration::from_millis(200)).await;

    // Check ClickHouse CommentFeedback
    let clickhouse = get_clickhouse().await;
    let result = select_feedback_clickhouse(&clickhouse, "CommentFeedback", feedback_id)
        .await
        .unwrap();
    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, feedback_id);
    let retrieved_episode_id = result.get("target_id").unwrap().as_str().unwrap();
    let retrieved_episode_id_uuid = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id_uuid, episode_id);
    let retrieved_target_type = result.get("target_type").unwrap().as_str().unwrap();
    assert_eq!(retrieved_target_type, "episode");
    let retrieved_value = result.get("value").unwrap().as_str().unwrap();
    assert_eq!(retrieved_value, "good job!");

    // Check ClickHouse FeedbackTag
    let result = select_feedback_tags_clickhouse(
        &clickhouse,
        "comment",
        "tensorzero::datapoint_id",
        &datapoint_id.to_string(),
    )
    .await
    .unwrap();
    // Sleep for 200ms to ensure that the StaticEvaluationHumanFeedback table is written
    // Note: table name retains "Static" prefix for backward compatibility (now called "Inference Evaluations")
    sleep(Duration::from_millis(200)).await;
    let id = result.get("feedback_id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, feedback_id);

    // Running without datapoint_id.
    // We generate a new datapoint_id so these don't trample. We'll only use it to check that there is no StaticEvaluationHumanFeedback
    // Note: table name retains "Static" prefix for backward compatibility (now called "Inference Evaluations")
    let new_datapoint_id = Uuid::now_v7();
    let payload = json!({"episode_id": episode_id, "metric_name": "comment", "value": "bad job!", "internal": true, "tags": {"tensorzero::human_feedback": "true"}});
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

    // Check ClickHouse
    sleep(Duration::from_millis(200)).await;
    let clickhouse = get_clickhouse().await;
    let result = select_feedback_clickhouse(&clickhouse, "CommentFeedback", feedback_id)
        .await
        .unwrap();
    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, feedback_id);
    let retrieved_target_type = result.get("target_type").unwrap().as_str().unwrap();
    assert_eq!(retrieved_target_type, "episode");
    let retrieved_value = result.get("value").unwrap().as_str().unwrap();
    assert_eq!(retrieved_value, "bad job!");

    // Check that there is no StaticEvaluationHumanFeedback (database table name, retains "Static" prefix for backward compatibility)
    let result = select_inference_evaluation_human_feedback_clickhouse(
        &clickhouse,
        "comment",
        new_datapoint_id,
        &serialized_inference_output,
    )
    .await;
    assert!(result.is_none());
}

#[tokio::test]
async fn e2e_test_demonstration_feedback() {
    let client = Client::new();
    // Running without valid inference_id. Should fail.
    let tag_value = Uuid::now_v7().to_string();
    let inference_id = Uuid::now_v7();
    let payload = json!({
        "inference_id": inference_id,
        "metric_name": "demonstration",
        "value": "do this!",
        "tags": {"key": tag_value}
    });
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
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

    assert!(response.status().is_success());
    let response_json = response.json::<Value>().await.unwrap();
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    // No sleeping, we should throttle in the gateway
    // Test demonstration feedback on Inference
    let tag_value = Uuid::now_v7().to_string();
    let payload = json!({
        "inference_id": inference_id,
        "metric_name": "demonstration",
        "value": "do this!",
        "tags": {"key": tag_value}
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
    sleep(Duration::from_millis(200)).await;

    // Check ClickHouse DemonstrationFeedback
    let clickhouse = get_clickhouse().await;
    let result = select_feedback_clickhouse(&clickhouse, "DemonstrationFeedback", feedback_id)
        .await
        .unwrap();
    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, feedback_id);
    let retrieved_inference_id = result.get("inference_id").unwrap().as_str().unwrap();
    let retrieved_inference_id_uuid = Uuid::parse_str(retrieved_inference_id).unwrap();
    assert_eq!(retrieved_inference_id_uuid, inference_id);
    let retrieved_value = result.get("value").unwrap().as_str().unwrap();
    let expected_value = serde_json::to_string(&json!(vec![ContentBlockChatOutput::Text(Text {
        text: "do this!".to_string()
    })]))
    .unwrap();
    assert_eq!(retrieved_value, expected_value);

    // Check ClickHouse FeedbackTag
    let result = select_feedback_tags_clickhouse(&clickhouse, "demonstration", "key", &tag_value)
        .await
        .unwrap();
    let id = result.get("feedback_id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, feedback_id);

    // Try it for an episode (should 400)
    let episode_id = Uuid::now_v7();
    let payload =
        json!({"episode_id": episode_id, "metric_name": "demonstration", "value": "do this!"});
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let response_json = response.json::<Value>().await.unwrap();
    let message = response_json.get("error").unwrap().as_str().unwrap();
    assert_eq!(
        message,
        "Correct ID was not provided for feedback level \"inference\"."
    );

    // Try a tool call demonstration
    // This should fail because the inference was made for a function that doesn't support tool calls
    let tool_call = json!({"type": "tool_call", "id": "tool_call_id", "name": "tool_name", "arguments": "tool_input"});
    let payload = json!({"inference_id": inference_id, "metric_name": "demonstration", "value": vec![tool_call]});
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let response_json = response.json::<Value>().await.unwrap();
    assert_eq!(
        response_json,
        json!({
            "error": "Demonstration contains invalid tool name",
            "error_json": {
                "InvalidRequest": {
                    "message": "Demonstration contains invalid tool name"
                }
            }
        })
    );
}

#[tokio::test]
async fn e2e_test_demonstration_feedback_json() {
    let client = Client::new();
    // Running without valid inference_id. Should fail.
    let inference_id = Uuid::now_v7();
    let payload = json!({
        "inference_id": inference_id,
        "metric_name": "demonstration",
        "value": {"answer": "Tokyo"}
    });
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
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

    assert!(response.status().is_success());
    let response_json = response.json::<Value>().await.unwrap();
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    // No sleeping, we should throttle in the gateway
    // Test demonstration feedback on an inference
    let payload = json!({
        "inference_id": inference_id,
        "metric_name": "demonstration",
        "value": {"answer": "Tokyo"}
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
    sleep(Duration::from_millis(200)).await;

    // Check ClickHouse
    let clickhouse = get_clickhouse().await;
    let result = select_feedback_clickhouse(&clickhouse, "DemonstrationFeedback", feedback_id)
        .await
        .unwrap();
    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, feedback_id);
    let retrieved_inference_id = result.get("inference_id").unwrap().as_str().unwrap();
    let retrieved_inference_id_uuid = Uuid::parse_str(retrieved_inference_id).unwrap();
    assert_eq!(retrieved_inference_id_uuid, inference_id);
    let retrieved_value = result.get("value").unwrap().as_str().unwrap();
    let retrieved_value = serde_json::from_str::<JsonInferenceOutput>(retrieved_value).unwrap();
    let expected_value = JsonInferenceOutput {
        parsed: Some(json!({"answer": "Tokyo"})),
        raw: Some("{\"answer\":\"Tokyo\"}".to_string()),
    };
    assert_eq!(retrieved_value, expected_value);

    // Try it for an episode (should 400)
    let episode_id = Uuid::now_v7();
    let payload =
        json!({"episode_id": episode_id, "metric_name": "demonstration", "value": "do this!"});
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let response_json = response.json::<Value>().await.unwrap();
    let message = response_json.get("error").unwrap().as_str().unwrap();
    assert_eq!(
        message,
        "Correct ID was not provided for feedback level \"inference\"."
    );

    // Try a tool call demonstration
    // This should fail because the inference was made for a function that doesn't support tool calls
    let tool_call = json!({"type": "tool_call", "id": "tool_call_id", "name": "tool_name", "arguments": "tool_input"});
    let payload = json!({"inference_id": inference_id, "metric_name": "demonstration", "value": vec![tool_call]});
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let response_json = response.json::<Value>().await.unwrap();
    let error_message = response_json.get("error").unwrap().as_str().unwrap();
    assert!(error_message.starts_with("Demonstration does not fit function output schema:"));
}

#[tokio::test]
async fn e2e_test_demonstration_feedback_dynamic_json() {
    let client = Client::new();
    // Running without valid inference_id. Should fail.
    let inference_id = Uuid::now_v7();
    let payload = json!({
        "inference_id": inference_id,
        "metric_name": "demonstration",
        "value": {"answer": "Tokyo"}
    });
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
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

    assert!(response.status().is_success());
    let response_json = response.json::<Value>().await.unwrap();
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    // No sleeping, we should throttle in the gateway
    // Test demonstration feedback on an inference that requires the dynamic output schema
    let payload = json!({
        "inference_id": inference_id,
        "metric_name": "demonstration",
        "value": {"answer": "Tokyo", "comment": "This is a comment"}
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
    sleep(Duration::from_millis(200)).await;

    // Check ClickHouse
    let clickhouse = get_clickhouse().await;
    let result = select_feedback_clickhouse(&clickhouse, "DemonstrationFeedback", feedback_id)
        .await
        .unwrap();
    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, feedback_id);
    let retrieved_inference_id = result.get("inference_id").unwrap().as_str().unwrap();
    let retrieved_inference_id_uuid = Uuid::parse_str(retrieved_inference_id).unwrap();
    assert_eq!(retrieved_inference_id_uuid, inference_id);
    let retrieved_value = result.get("value").unwrap().as_str().unwrap();
    let retrieved_value = serde_json::from_str::<JsonInferenceOutput>(retrieved_value).unwrap();
    let expected_value = JsonInferenceOutput {
        parsed: Some(json!({"answer": "Tokyo", "comment": "This is a comment"})),
        raw: Some("{\"answer\":\"Tokyo\",\"comment\":\"This is a comment\"}".to_string()),
    };
    assert_eq!(retrieved_value, expected_value);

    // Try it for an episode (should 400)
    let episode_id = Uuid::now_v7();
    let payload =
        json!({"episode_id": episode_id, "metric_name": "demonstration", "value": "do this!"});
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let response_json = response.json::<Value>().await.unwrap();
    let message = response_json.get("error").unwrap().as_str().unwrap();
    assert_eq!(
        message,
        "Correct ID was not provided for feedback level \"inference\"."
    );

    // Try a tool call demonstration
    // This should fail because the inference was made for a function that doesn't support tool calls
    let tool_call = json!({"type": "tool_call", "id": "tool_call_id", "name": "tool_name", "arguments": "tool_input"});
    let payload = json!({"inference_id": inference_id, "metric_name": "demonstration", "value": vec![tool_call]});
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let response_json = response.json::<Value>().await.unwrap();
    let error_message = response_json.get("error").unwrap().as_str().unwrap();
    assert!(error_message.starts_with("Demonstration does not fit function output schema:"));

    // Try a demonstration with a value that doesn't match the output schema
    let payload = json!({"inference_id": inference_id, "metric_name": "demonstration", "value": {"bad_key": "Tokyo"}});
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let response_json = response.json::<Value>().await.unwrap();
    let error_message = response_json.get("error").unwrap().as_str().unwrap();
    assert!(error_message.starts_with("Demonstration does not fit function output schema:"));
}

#[tokio::test]
async fn e2e_test_demonstration_feedback_tool() {
    // Running without valid inference_id. Should fail.
    let client = Client::new();
    let inference_id = Uuid::now_v7();
    let payload = json!({
        "inference_id": inference_id,
        "metric_name": "demonstration",
        "value": "sunny",
    });
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
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

    assert!(response.status().is_success());
    let response_json = response.json::<Value>().await.unwrap();
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    // No sleeping, we should throttle in the gateway
    // Test demonstration feedback on Inference (string shortcut)
    let payload = json!({
        "inference_id": inference_id,
        "metric_name": "demonstration",
        "value": "sunny",
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
    sleep(Duration::from_millis(200)).await;

    // Check ClickHouse
    let clickhouse = get_clickhouse().await;
    let result = select_feedback_clickhouse(&clickhouse, "DemonstrationFeedback", feedback_id)
        .await
        .unwrap();
    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, feedback_id);
    let retrieved_inference_id = result.get("inference_id").unwrap().as_str().unwrap();
    let retrieved_inference_id_uuid = Uuid::parse_str(retrieved_inference_id).unwrap();
    assert_eq!(retrieved_inference_id_uuid, inference_id);
    let retrieved_value = result.get("value").unwrap().as_str().unwrap();
    let expected_value =
        serde_json::to_string(&json!([{"type": "text", "text": "sunny" }])).unwrap();
    assert_eq!(retrieved_value, expected_value);

    // Try it for an episode (should 400)
    let episode_id = Uuid::now_v7();
    let payload =
        json!({"episode_id": episode_id, "metric_name": "demonstration", "value": "do this!"});
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let response_json = response.json::<Value>().await.unwrap();
    let message = response_json.get("error").unwrap().as_str().unwrap();
    assert_eq!(
        message,
        "Correct ID was not provided for feedback level \"inference\"."
    );

    // Try a tool call demonstration
    // This should fail because the name is incorrect
    let tool_call = json!({"type": "tool_call", "id": "tool_call_id", "name": "tool_name", "arguments": "tool_input"});
    let payload = json!({"inference_id": inference_id, "metric_name": "demonstration", "value": vec![tool_call]});
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let response_json = response.json::<Value>().await.unwrap();
    let error_message = response_json.get("error").unwrap().as_str().unwrap();
    assert_eq!(error_message, "Demonstration contains invalid tool name");

    // Try a tool call demonstration with correct name incorrect args
    let tool_call = json!({"type": "tool_call", "id": "tool_call_id", "name": "get_temperature", "arguments": "tool_input"});
    let payload = json!({"inference_id": inference_id, "metric_name": "demonstration", "value": vec![tool_call]});
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let response_json = response.json::<Value>().await.unwrap();
    let error_message = response_json.get("error").unwrap().as_str().unwrap();
    assert_eq!(
        error_message,
        "Demonstration contains invalid tool call arguments"
    );

    // Try a tool call demonstration with correct name and args
    let tool_call = json!({"type": "tool_call", "id": "tool_call_id", "name": "get_temperature", "arguments": {"location": "Tokyo", "units": "celsius"}});
    let payload = json!({"inference_id": inference_id, "metric_name": "demonstration", "value": vec![tool_call]});
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
    sleep(Duration::from_millis(200)).await;

    // Check ClickHouse
    let clickhouse = get_clickhouse().await;
    let result = select_feedback_clickhouse(&clickhouse, "DemonstrationFeedback", feedback_id)
        .await
        .unwrap();
    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, feedback_id);
    let retrieved_inference_id = result.get("inference_id").unwrap().as_str().unwrap();
    let retrieved_inference_id_uuid = Uuid::parse_str(retrieved_inference_id).unwrap();
    assert_eq!(retrieved_inference_id_uuid, inference_id);
    let retrieved_value = result.get("value").unwrap().as_str().unwrap();
    let retrieved_value = serde_json::from_str::<Value>(retrieved_value).unwrap();
    let expected_value = json!([{"type": "tool_call", "id": "tool_call_id", "raw_name": "get_temperature", "raw_arguments": "{\"location\":\"Tokyo\",\"units\":\"celsius\"}", "name": "get_temperature", "arguments": {"location": "Tokyo", "units": "celsius"}}]);
    assert_eq!(retrieved_value, expected_value);
}

#[tokio::test]
async fn e2e_test_demonstration_feedback_dynamic_tool() {
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

    assert!(response.status().is_success());
    let response_json = response.json::<Value>().await.unwrap();
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    // No sleeping, we should throttle in the gateway
    // Test demonstration feedback on Inference (string shortcut)
    let payload = json!({
        "inference_id": inference_id,
        "metric_name": "demonstration",
        "value": "sunny",
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
    sleep(Duration::from_millis(200)).await;

    // Check ClickHouse
    let clickhouse = get_clickhouse().await;
    let result = select_feedback_clickhouse(&clickhouse, "DemonstrationFeedback", feedback_id)
        .await
        .unwrap();
    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, feedback_id);
    let retrieved_inference_id = result.get("inference_id").unwrap().as_str().unwrap();
    let retrieved_inference_id_uuid = Uuid::parse_str(retrieved_inference_id).unwrap();
    assert_eq!(retrieved_inference_id_uuid, inference_id);
    let retrieved_value = result.get("value").unwrap().as_str().unwrap();
    let expected_value =
        serde_json::to_string(&json!([{"type": "text", "text": "sunny" }])).unwrap();
    assert_eq!(retrieved_value, expected_value);

    // Try it for an episode (should 400)
    let episode_id = Uuid::now_v7();
    let payload =
        json!({"episode_id": episode_id, "metric_name": "demonstration", "value": "do this!"});
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let response_json = response.json::<Value>().await.unwrap();
    let message = response_json.get("error").unwrap().as_str().unwrap();
    assert_eq!(
        message,
        "Correct ID was not provided for feedback level \"inference\"."
    );

    // Try a tool call demonstration
    // This should fail because the name is incorrect
    let tool_call = json!({"type": "tool_call", "id": "tool_call_id", "name": "tool_name", "arguments": "tool_input"});
    let payload = json!({"inference_id": inference_id, "metric_name": "demonstration", "value": vec![tool_call]});
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let response_json = response.json::<Value>().await.unwrap();
    let error_message = response_json.get("error").unwrap().as_str().unwrap();
    assert_eq!(error_message, "Demonstration contains invalid tool name");

    // Try a tool call demonstration with the dynamic tool name and incorrect args
    let tool_call = json!({"type": "tool_call", "id": "tool_call_id", "name": "get_humidity", "arguments": "tool_input"});
    let payload = json!({"inference_id": inference_id, "metric_name": "demonstration", "value": vec![tool_call]});
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let response_json = response.json::<Value>().await.unwrap();
    let error_message = response_json.get("error").unwrap().as_str().unwrap();
    assert_eq!(
        error_message,
        "Demonstration contains invalid tool call arguments"
    );

    // Try a tool call demonstration with the dynamic tool name and correct args
    let tool_call = json!({"type": "tool_call", "id": "tool_call_id", "name": "get_humidity", "arguments": {"location": "Tokyo"}});
    let payload = json!({"inference_id": inference_id, "metric_name": "demonstration", "value": vec![tool_call]});
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
    sleep(Duration::from_millis(200)).await;

    // Check ClickHouse
    let clickhouse = get_clickhouse().await;
    let result = select_feedback_clickhouse(&clickhouse, "DemonstrationFeedback", feedback_id)
        .await
        .unwrap();
    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, feedback_id);
    let retrieved_inference_id = result.get("inference_id").unwrap().as_str().unwrap();
    let retrieved_inference_id_uuid = Uuid::parse_str(retrieved_inference_id).unwrap();
    assert_eq!(retrieved_inference_id_uuid, inference_id);
    let retrieved_value = result.get("value").unwrap().as_str().unwrap();
    let retrieved_value = serde_json::from_str::<Value>(retrieved_value).unwrap();
    let expected_value = json!([{"type": "tool_call", "id": "tool_call_id", "raw_name": "get_humidity", "raw_arguments": "{\"location\":\"Tokyo\"}", "name": "get_humidity", "arguments": {"location": "Tokyo"}}]);
    assert_eq!(retrieved_value, expected_value);
}

#[tokio::test]
async fn e2e_test_float_feedback() {
    let client = Client::new();
    let tag_value = Uuid::now_v7().to_string();
    // Running without valid episode_id. Should fail.
    let episode_id = Uuid::now_v7();
    let payload = json!({"episode_id": episode_id, "metric_name": "user_rating", "value": 32.8, "tags": {"key": tag_value}});
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
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
    let episode_id = response_json.get("episode_id").unwrap().as_str().unwrap();
    let episode_id = Uuid::parse_str(episode_id).unwrap();
    // Test Float feedback on episode
    let payload = json!({"episode_id": episode_id, "metric_name": "user_rating", "value": 32.8, "tags": {"key": tag_value}});
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
    sleep(Duration::from_millis(200)).await;

    // Check ClickHouse FloatMetricFeedback
    let clickhouse = get_clickhouse().await;
    let result = select_feedback_clickhouse(&clickhouse, "FloatMetricFeedback", feedback_id)
        .await
        .unwrap();
    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, feedback_id);
    let retrieved_episode_id = result.get("target_id").unwrap().as_str().unwrap();
    let retrieved_episode_id_uuid = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id_uuid, episode_id);
    let retrieved_value = result.get("value").unwrap().as_f64().unwrap();
    assert_eq!(retrieved_value, 32.8);
    let metric_name = result.get("metric_name").unwrap().as_str().unwrap();
    assert_eq!(metric_name, "user_rating");

    // Check ClickHouse FeedbackTag
    let result = select_feedback_tags_clickhouse(&clickhouse, "user_rating", "key", &tag_value)
        .await
        .unwrap();
    let id = result.get("feedback_id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, feedback_id);

    // Test boolean feedback on episode (should fail)
    let payload = json!({"episode_id": episode_id, "metric_name": "user_rating", "value": true});
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let response_json = response.json::<Value>().await.unwrap();
    let error_message = response_json.get("error").unwrap().as_str().unwrap();
    assert_eq!(
        error_message,
        "Feedback value for metric `user_rating` must be a number"
    );

    // Test float feedback on inference (should fail)
    let inference_id = Uuid::now_v7();
    let payload = json!({"inference_id": inference_id, "metric_name": "user_rating", "value": 4.5});
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let response_json = response.json::<Value>().await.unwrap();
    let error_message = response_json.get("error").unwrap().as_str().unwrap();
    assert!(
        error_message.contains("Correct ID was not provided for feedback level"),
        "Unexpected error message: {error_message}"
    );

    // Running without valid inference_id. Should fail.
    let inference_id = Uuid::now_v7();
    let payload =
        json!({"inference_id": inference_id, "metric_name": "brevity_score", "value": 0.5});
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

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

    assert!(response.status().is_success());
    let response_json = response.json::<Value>().await.unwrap();
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    // Just this once, we sleep longer than the duration of the feedback cooldown period (5s)
    // to make sure that the feedback is written after the inference.
    sleep(Duration::from_millis(5500)).await;

    // Test float feedback on different metric for inference.
    let payload =
        json!({"inference_id": inference_id, "metric_name": "brevity_score", "value": 0.5});
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
    sleep(Duration::from_millis(200)).await;

    // Check ClickHouse
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
    assert_eq!(retrieved_value, 0.5);
    let metric_name = result.get("metric_name").unwrap().as_str().unwrap();
    assert_eq!(metric_name, "brevity_score");
}

#[tokio::test]
async fn e2e_test_boolean_feedback() {
    let client = Client::new();
    let inference_id = Uuid::now_v7();
    let tag_value = Uuid::now_v7().to_string();
    let tag_value2 = Uuid::now_v7().to_string();
    // Running without valid inference_id. Should fail.
    let payload = json!({"inference_id": inference_id, "metric_name": "task_success", "value": true, "tags": {"key": tag_value, "key2": tag_value2}});
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
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

    // No sleeping, we should throttle in the gateway

    let payload = json!({"inference_id": inference_id, "metric_name": "task_success", "value": true, "tags": {"key": tag_value, "key2": tag_value2}});
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
    sleep(Duration::from_millis(200)).await;

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

    // Check ClickHouse FeedbackTag
    let result = select_feedback_tags_clickhouse(&clickhouse, "task_success", "key", &tag_value)
        .await
        .unwrap();
    let id = result.get("feedback_id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, feedback_id);

    let result = select_feedback_tags_clickhouse(&clickhouse, "task_success", "key2", &tag_value2)
        .await
        .unwrap();
    let id = result.get("feedback_id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, feedback_id);

    // Try episode-level feedback (should fail)
    let episode_id = Uuid::now_v7();
    let payload = json!({"episode_id": episode_id, "metric_name": "task_success", "value": true});
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let response_json = response.json::<Value>().await.unwrap();
    let error_message = response_json.get("error").unwrap().as_str().unwrap();
    assert!(
        error_message.contains("Correct ID was not provided for feedback level"),
        "Unexpected error message: {error_message}"
    );

    // Try string feedback (should fail)
    let payload =
        json!({"inference_id": inference_id, "metric_name": "task_success", "value": "true"});
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let response_json = response.json::<Value>().await.unwrap();
    let error_message = response_json.get("error").unwrap().as_str().unwrap();
    assert_eq!(
        error_message,
        "Feedback value for metric `task_success` must be a boolean"
    );

    // Try episode-level feedback on different metric with invalid episode id.
    let episode_id = Uuid::now_v7();
    let payload = json!({"episode_id": episode_id, "metric_name": "goal_achieved", "value": true});
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
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
    let episode_id = response_json.get("episode_id").unwrap().as_str().unwrap();
    let episode_id = Uuid::parse_str(episode_id).unwrap();

    let payload = json!({"episode_id": episode_id, "metric_name": "goal_achieved", "value": true});
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
    sleep(Duration::from_millis(200)).await;

    // Check ClickHouse
    let result = select_feedback_clickhouse(&clickhouse, "BooleanMetricFeedback", feedback_id)
        .await
        .unwrap();
    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, feedback_id);
    let retrieved_episode_id = result.get("target_id").unwrap().as_str().unwrap();
    let retrieved_episode_id_uuid = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id_uuid, episode_id);
    let retrieved_value = result.get("value").unwrap().as_bool().unwrap();
    assert!(retrieved_value);
    let metric_name = result.get("metric_name").unwrap().as_str().unwrap();
    assert_eq!(metric_name, "goal_achieved");
}

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
                    input: tensorzero::ClientInput {
                        system: Some(System::Template(Arguments(
                            json!({"assistant_name": "Alfred Pennyworth"})
                                .as_object()
                                .unwrap()
                                .clone(),
                        ))),
                        messages: vec![tensorzero::ClientInputMessage {
                            role: Role::User,
                            content: vec![tensorzero::ClientInputMessageContent::Text(TextKind::Text {
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
    assert!(!logs_contain("does not exist"));
}
