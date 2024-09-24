use gateway::inference::types::{ContentBlockOutput, JsonInferenceOutput, Text};
use reqwest::{Client, StatusCode};
use serde_json::{json, Value};
use uuid::Uuid;

use crate::common::{get_clickhouse, get_gateway_endpoint, select_feedback_clickhouse};

#[tokio::test]
async fn e2e_test_comment_feedback() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();
    // Test comment feedback on episode
    let payload = json!({"episode_id": episode_id, "metric_name": "comment", "value": "good job!"});
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

    // Test comment feedback on Inference
    let inference_id = Uuid::now_v7();
    let payload =
        json!({"inference_id": inference_id, "metric_name": "comment", "value": "bad job!"});
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
    let clickhouse = get_clickhouse().await;
    let result = select_feedback_clickhouse(&clickhouse, "CommentFeedback", feedback_id)
        .await
        .unwrap();
    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, feedback_id);
    let retrieved_inference_id = result.get("target_id").unwrap().as_str().unwrap();
    let retrieved_inference_id_uuid = Uuid::parse_str(retrieved_inference_id).unwrap();
    assert_eq!(retrieved_inference_id_uuid, inference_id);
    let retrieved_target_type = result.get("target_type").unwrap().as_str().unwrap();
    assert_eq!(retrieved_target_type, "inference");
    let retrieved_value = result.get("value").unwrap().as_str().unwrap();
    assert_eq!(retrieved_value, "bad job!");
}

#[tokio::test]
async fn e2e_test_demonstration_feedback() {
    let client = Client::new();
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
    // Sleep for 1 second to allow ClickHouse to write
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Test demonstration feedback on Inference
    let payload = json!({
        "inference_id": inference_id,
        "metric_name": "demonstration",
        "value": "do this!"
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
    let expected_value = serde_json::to_string(&json!(vec![ContentBlockOutput::Text(Text {
        text: "do this!".to_string()
    })]))
    .unwrap();
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
    let tool_call = json!({"type": "tool_call", "name": "tool_name", "arguments": "tool_input"});
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
        json!({"error": "Demonstration contains invalid tool name"})
    );
}

#[tokio::test]
async fn e2e_test_demonstration_feedback_json() {
    let client = Client::new();
    // Run inference (standard, no dryrun) to get an inference_id
    let inference_payload = serde_json::json!({
        "function_name": "json_success",
        "input": {
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": {"country": "Japan"}}]
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
    // Sleep for 1 second to allow ClickHouse to write
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Test demonstration feedback on Inference
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
    let expected_value = serde_json::to_string(&JsonInferenceOutput {
        parsed: Some(json!({"answer": "Tokyo"})),
        raw: "{\"answer\":\"Tokyo\"}".to_string(),
    })
    .unwrap();
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
    let tool_call = json!({"type": "tool_call", "name": "tool_name", "arguments": "tool_input"});
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
async fn e2e_test_demonstration_feedback_tool() {
    let client = Client::new();
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
    // Sleep for 1 second to allow ClickHouse to write
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

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
    let tool_call = json!({"type": "tool_call", "name": "tool_name", "arguments": "tool_input"});
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
    let tool_call =
        json!({"type": "tool_call", "name": "get_temperature", "arguments": "tool_input"});
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
    let tool_call = json!({"type": "tool_call", "name": "get_temperature", "arguments": {"location": "Tokyo", "units": "celsius"}});
    let payload = json!({"inference_id": inference_id, "metric_name": "demonstration", "value": vec![tool_call]});
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    // assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    let feedback_id = response_json.get("feedback_id").unwrap();
    assert!(feedback_id.is_string());
    let feedback_id = Uuid::parse_str(feedback_id.as_str().unwrap()).unwrap();

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
    let expected_value = json!([{"type": "tool_call", "name": "get_temperature", "arguments": {"location": "Tokyo", "units": "celsius"}, "raw_name": "get_temperature", "raw_arguments": "{\"location\":\"Tokyo\",\"units\":\"celsius\"}", "id": "" }]);
    assert_eq!(retrieved_value, expected_value);
}

#[tokio::test]
async fn e2e_test_float_feedback() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();
    // Test Float feedback on episode
    let payload = json!({"episode_id": episode_id, "metric_name": "user_rating", "value": 32.8});
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
        "Unexpected error message: {}",
        error_message
    );

    // Test float feedback on different metric for inference
    let inference_id = Uuid::now_v7();
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
    let payload =
        json!({"inference_id": inference_id, "metric_name": "task_success", "value": true});
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
        "Unexpected error message: {}",
        error_message
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

    // Try episode-level feedback on different metric
    let episode_id = Uuid::now_v7();
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
