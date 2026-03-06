use googletest::prelude::*;
use reqwest::{Client, StatusCode};
use serde_json::{Value, json};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};
use tensorzero_core::{
    config::{Config, MetricConfig, MetricConfigLevel, MetricConfigOptimize, MetricConfigType},
    db::{postgres::PostgresConnectionInfo, valkey::ValkeyConnectionInfo},
    endpoints::feedback::{Params, feedback},
    http::TensorzeroHttpClient,
    inference::types::{
        Arguments, ContentBlockChatOutput, JsonInferenceOutput, Role, System, Text,
    },
    utils::gateway::GatewayHandle,
};
use tokio::time::{Duration, sleep};
use uuid::Uuid;

use crate::common::get_gateway_endpoint;
use tensorzero_core::db::clickhouse::test_helpers::get_clickhouse;
use tensorzero_core::db::delegating_connection::{DelegatingDatabaseConnection, PrimaryDatastore};
use tensorzero_core::db::feedback::{FeedbackQueries, FeedbackRow};
use tensorzero_core::db::test_helpers::TestDatabaseHelpers;

#[gtest]
#[tokio::test]
async fn test_comment_feedback_normal_function() {
    test_comment_feedback_with_payload(serde_json::json!({
        "function_name": "json_success",
        "input": {
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": [{"type": "template", "name": "user", "arguments": {"country": "Japan"}}]}]
        },
        "stream": false,
    })).await;
}

#[gtest]
#[tokio::test]
async fn test_comment_feedback_default_function() {
    test_comment_feedback_with_payload(serde_json::json!({
        "model_name": "dummy::good",
        "input": {
            "messages": [{"role": "user", "content": "Hello, world!"}]
        },
        "stream": false,
    }))
    .await;
}

async fn test_comment_feedback_with_payload(inference_payload: serde_json::Value) {
    let client = Client::new();
    // // Running without valid episode_id. Should fail.
    let episode_id = Uuid::now_v7();
    // Test comment feedback on episode
    let payload = Params {
        episode_id: Some(episode_id),
        inference_id: None,
        internal: false,
        dryrun: None,
        metric_name: String::from("comment"),
        value: serde_json::to_value("good job!").unwrap(),
        tags: HashMap::from([(String::from("key"), Uuid::now_v7().to_string())]),
    };
    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    // Run inference (standard, no dryrun) to get an episode_id.
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

    // Test comment feedback on episode
    let tag_value = Uuid::now_v7().to_string();
    let payload = Params {
        episode_id: Some(episode_id),
        metric_name: String::from("comment"),
        value: serde_json::to_value("good job!").unwrap(),
        tags: HashMap::from([(String::from("key"), tag_value.clone())]),
        dryrun: None,
        internal: false,
        inference_id: None,
    };
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

    // Check CommentFeedback
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;
    let feedbacks = conn
        .query_feedback_by_target_id(episode_id, None, None, Some(100))
        .await
        .unwrap();
    let comment = feedbacks
        .iter()
        .find_map(|f| match f {
            FeedbackRow::Comment(c) if c.id == feedback_id => Some(c),
            _ => None,
        })
        .expect("Should find comment feedback");
    assert_eq!(comment.target_id, episode_id);
    assert_eq!(comment.value, "good job!");
    assert!(
        matches!(
            comment.target_type,
            tensorzero_core::db::feedback::CommentTargetType::Episode
        ),
        "Expected episode target type"
    );

    // Check FeedbackTag
    assert_eq!(comment.tags.get("key"), Some(&tag_value));

    // snapshot_hash checks are ClickHouse-specific
    if PrimaryDatastore::from_test_env() != PrimaryDatastore::Postgres {
        let clickhouse = get_clickhouse().await;
        let query = format!(
            "SELECT snapshot_hash FROM CommentFeedback WHERE id = '{feedback_id}' FORMAT JSONEachRow"
        );
        let response = clickhouse
            .run_query_synchronous_no_params(query)
            .await
            .unwrap();
        let result: serde_json::Value = serde_json::from_str(&response.response).unwrap();
        assert!(
            !result["snapshot_hash"].is_null(),
            "CommentFeedback should have snapshot_hash"
        );
    }

    // Running without valid inference_id. Should fail.
    let inference_id = Uuid::now_v7();
    let payload = Params {
        inference_id: Some(inference_id),
        metric_name: String::from("comment"),
        value: Value::String(String::from("bad job!")),
        dryrun: None,
        episode_id: None,
        internal: false,
        tags: HashMap::new(),
    };
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
    let payload = Params {
        inference_id: Some(inference_id),
        metric_name: String::from("comment"),
        value: Value::String(String::from("bad job!")),
        dryrun: None,
        episode_id: None,
        internal: false,
        tags: HashMap::new(),
    };

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

    // Check CommentFeedback
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;
    let feedbacks = conn
        .query_feedback_by_target_id(inference_id, None, None, Some(100))
        .await
        .unwrap();
    let comment = feedbacks
        .iter()
        .find_map(|f| match f {
            FeedbackRow::Comment(c) if c.id == feedback_id => Some(c),
            _ => None,
        })
        .expect("Should find comment feedback");
    assert_eq!(comment.target_id, inference_id);
    assert_eq!(comment.value, "bad job!");
    assert!(
        matches!(
            comment.target_type,
            tensorzero_core::db::feedback::CommentTargetType::Inference
        ),
        "Expected inference target type"
    );

    // snapshot_hash checks are ClickHouse-specific
    if PrimaryDatastore::from_test_env() != PrimaryDatastore::Postgres {
        let clickhouse = get_clickhouse().await;
        let query = format!(
            "SELECT snapshot_hash FROM CommentFeedback WHERE id = '{feedback_id}' FORMAT JSONEachRow"
        );
        let response = clickhouse
            .run_query_synchronous_no_params(query)
            .await
            .unwrap();
        let result: serde_json::Value = serde_json::from_str(&response.response).unwrap();
        assert!(
            !result["snapshot_hash"].is_null(),
            "CommentFeedback should have snapshot_hash"
        );

        let query = format!(
            "SELECT snapshot_hash FROM CommentFeedbackByTargetId WHERE target_id = '{inference_id}' AND id = '{feedback_id}' FORMAT JSONEachRow"
        );
        let response = clickhouse
            .run_query_synchronous_no_params(query)
            .await
            .unwrap();
        let view_result: serde_json::Value = serde_json::from_str(&response.response).unwrap();
        assert!(
            !view_result["snapshot_hash"].is_null(),
            "CommentFeedbackByTargetId should have snapshot_hash"
        );
    }
}

#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_comment_feedback_validation_disabled() {
    let mut config = Config::new_empty()
        .await
        .unwrap()
        .into_config_without_writing_for_tests();
    let clickhouse = get_clickhouse().await;
    config.gateway.unstable_disable_feedback_target_validation = true;
    let handle = GatewayHandle::new_with_database_and_http_client(
        Arc::new(config),
        clickhouse.clone(),
        PostgresConnectionInfo::Disabled,
        ValkeyConnectionInfo::Disabled,
        ValkeyConnectionInfo::Disabled,
        TensorzeroHttpClient::new_testing().unwrap(),
        None,
        HashSet::new(), // available_tools
        HashSet::new(), // tool_whitelist
    )
    .await
    .unwrap();
    let inference_id = Uuid::now_v7();
    let params = Params {
        inference_id: Some(inference_id),
        metric_name: "comment".to_string(),
        value: json!("foo bar"),
        ..Default::default()
    };
    let val = feedback(handle.app_state.clone(), params, None)
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Check that this was correctly written to ClickHouse
    let query = format!(
        "SELECT * FROM CommentFeedback WHERE target_id='{inference_id}' FORMAT JsonEachRow"
    );
    let response = clickhouse
        .run_query_synchronous_no_params(query)
        .await
        .unwrap();
    let result: Value = serde_json::from_str(&response.response).unwrap();
    let clickhouse_feedback_id = Uuid::parse_str(result["id"].as_str().unwrap()).unwrap();
    expect_that!(val.feedback_id, eq(clickhouse_feedback_id));
}

#[gtest]
#[tokio::test]
async fn test_demonstration_feedback_normal_function() {
    test_demonstration_feedback_with_payload(serde_json::json!({
        "function_name": "basic_test",
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [{"role": "user", "content": "Hello, world!"}]
        },
        "stream": false,
    }))
    .await;
}

#[gtest]
#[tokio::test]
async fn test_demonstration_feedback_default_function() {
    test_demonstration_feedback_with_payload(serde_json::json!({
        "model_name": "dummy::good",
        "input": {
            "messages": [{"role": "user", "content": "Hello, world!"}]
        },
        "stream": false,
    }))
    .await;
}

async fn test_demonstration_feedback_with_payload(inference_payload: serde_json::Value) {
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
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    // Run inference (standard, no dryrun) to get an inference_id
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
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    let feedback_id = response_json.get("feedback_id").unwrap();
    assert!(feedback_id.is_string());
    let feedback_id = Uuid::parse_str(feedback_id.as_str().unwrap()).unwrap();

    // Check DemonstrationFeedback
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;
    let feedbacks = conn
        .query_feedback_by_target_id(inference_id, None, None, Some(100))
        .await
        .unwrap();
    let demo = feedbacks
        .iter()
        .find_map(|f| match f {
            FeedbackRow::Demonstration(d) if d.id == feedback_id => Some(d),
            _ => None,
        })
        .expect("Should find demonstration feedback");
    assert_eq!(demo.inference_id, inference_id);
    let expected_value = serde_json::to_string(&json!(vec![ContentBlockChatOutput::Text(Text {
        text: "do this!".to_string()
    })]))
    .unwrap();
    assert_eq!(demo.value, expected_value);

    // Check FeedbackTag
    assert_eq!(demo.tags.get("key"), Some(&tag_value));

    // snapshot_hash checks are ClickHouse-specific
    if PrimaryDatastore::from_test_env() != PrimaryDatastore::Postgres {
        let clickhouse = get_clickhouse().await;
        let query = format!(
            "SELECT snapshot_hash FROM DemonstrationFeedback WHERE id = '{feedback_id}' FORMAT JSONEachRow"
        );
        let response = clickhouse
            .run_query_synchronous_no_params(query)
            .await
            .unwrap();
        let result: serde_json::Value = serde_json::from_str(&response.response).unwrap();
        assert!(
            !result["snapshot_hash"].is_null(),
            "DemonstrationFeedback should have snapshot_hash"
        );

        let query = format!(
            "SELECT snapshot_hash FROM DemonstrationFeedbackByInferenceId WHERE inference_id = '{inference_id}' AND id = '{feedback_id}' FORMAT JSONEachRow"
        );
        let response = clickhouse
            .run_query_synchronous_no_params(query)
            .await
            .unwrap();
        let view_result: serde_json::Value = serde_json::from_str(&response.response).unwrap();
        assert!(
            !view_result["snapshot_hash"].is_null(),
            "DemonstrationFeedbackByInferenceId should have snapshot_hash"
        );
    }

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

    assert_that!(response.status().is_success(), eq(true));
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
    assert_that!(response.status(), eq(StatusCode::OK));
    let response_json = response.json::<Value>().await.unwrap();
    let feedback_id = response_json.get("feedback_id").unwrap();
    assert_that!(feedback_id.is_string(), eq(true));
    let feedback_id = Uuid::parse_str(feedback_id.as_str().unwrap()).unwrap();

    // Check DemonstrationFeedback
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;
    let feedbacks = conn
        .query_feedback_by_target_id(inference_id, None, None, Some(100))
        .await
        .unwrap();
    let demo = feedbacks
        .iter()
        .find_map(|f| match f {
            FeedbackRow::Demonstration(d) if d.id == feedback_id => Some(d),
            _ => None,
        })
        .expect("Should find demonstration feedback");
    expect_that!(demo.inference_id, eq(inference_id));
    let retrieved_value = serde_json::from_str::<JsonInferenceOutput>(&demo.value).unwrap();
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
async fn test_demonstration_feedback_llm_judge() {
    let client = Client::new();
    // Run inference (standard, no dryrun) to get an inference_id
    let old_output_schema = json!({
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["thinking", "score"],
        "additionalProperties": false,
        "properties": {
          "thinking": {
            "type": "string",
            "description": "The reasoning or thought process behind the judgment"
          },
          "score": {
            "type": "number",
            "description": "The score assigned as a number"
          }
        }
    });
    let inference_payload = serde_json::json!({
        "function_name": "tensorzero::llm_judge::haiku_without_outputs::topic_starts_with_f",
        "input": {
            "messages": [{"role": "user", "content": [
                {"type": "template", "name": "user", "arguments": {"input": "foo", "reference_output": null, "generated_output": "A poem about a cat"}},
            ]}]
        },
        "stream": false,
        "output_schema": old_output_schema,
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&inference_payload)
        .send()
        .await
        .unwrap();
    assert_that!(response.status().is_success(), eq(true));
    let response_json = response.json::<Value>().await.unwrap();
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    // No sleeping, we should throttle in the gateway
    // Test demonstration feedback on an inference that requires the dynamic output schema
    let payload = Params {
        inference_id: Some(inference_id),
        episode_id: None,
        metric_name: String::from("demonstration"),
        value: json!({"score": 0.5}),
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
    assert_that!(response.status(), eq(StatusCode::OK));
    let response_json = response.json::<Value>().await.unwrap();
    let feedback_id = response_json.get("feedback_id").unwrap();
    assert_that!(feedback_id.is_string(), eq(true));
    let feedback_id = Uuid::parse_str(feedback_id.as_str().unwrap()).unwrap();

    // Check DemonstrationFeedback
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;
    let feedbacks = conn
        .query_feedback_by_target_id(inference_id, None, None, Some(100))
        .await
        .unwrap();
    let demo = feedbacks
        .iter()
        .find_map(|f| match f {
            FeedbackRow::Demonstration(d) if d.id == feedback_id => Some(d),
            _ => None,
        })
        .expect("Should find demonstration feedback");
    expect_that!(demo.inference_id, eq(inference_id));
    let retrieved_value = serde_json::from_str::<JsonInferenceOutput>(&demo.value).unwrap();
    let expected_value = JsonInferenceOutput {
        parsed: Some(json!({"score": 0.5})),
        raw: Some("{\"score\":0.5}".to_string()),
    };
    expect_that!(&retrieved_value, eq(&expected_value));
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

    assert_that!(response.status().is_success(), eq(true));
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
    assert_that!(response.status(), eq(StatusCode::OK));
    let response_json = response.json::<Value>().await.unwrap();
    let feedback_id = response_json.get("feedback_id").unwrap();
    assert_that!(feedback_id.is_string(), eq(true));
    let feedback_id = Uuid::parse_str(feedback_id.as_str().unwrap()).unwrap();

    // Check DemonstrationFeedback
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;
    let feedbacks = conn
        .query_feedback_by_target_id(inference_id, None, None, Some(100))
        .await
        .unwrap();
    let demo = feedbacks
        .iter()
        .find_map(|f| match f {
            FeedbackRow::Demonstration(d) if d.id == feedback_id => Some(d),
            _ => None,
        })
        .expect("Should find demonstration feedback");
    expect_that!(demo.inference_id, eq(inference_id));
    let retrieved_value = serde_json::from_str::<JsonInferenceOutput>(&demo.value).unwrap();
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

    assert_that!(response.status().is_success(), eq(true));
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
    assert_that!(response.status(), eq(StatusCode::OK));
    let response_json = response.json::<Value>().await.unwrap();
    let feedback_id = response_json.get("feedback_id").unwrap();
    assert_that!(feedback_id.is_string(), eq(true));
    let feedback_id = Uuid::parse_str(feedback_id.as_str().unwrap()).unwrap();

    // Check DemonstrationFeedback
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;
    let feedbacks = conn
        .query_feedback_by_target_id(inference_id, None, None, Some(100))
        .await
        .unwrap();
    let demo = feedbacks
        .iter()
        .find_map(|f| match f {
            FeedbackRow::Demonstration(d) if d.id == feedback_id => Some(d),
            _ => None,
        })
        .expect("Should find demonstration feedback");
    expect_that!(demo.inference_id, eq(inference_id));
    let expected_value =
        serde_json::to_string(&json!([{"type": "text", "text": "sunny" }])).unwrap();
    expect_that!(&demo.value, eq(&expected_value));

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
    assert_that!(feedback_id.is_string(), eq(true));
    let feedback_id = Uuid::parse_str(feedback_id.as_str().unwrap()).unwrap();

    // Check DemonstrationFeedback (tool call)
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;
    let feedbacks = conn
        .query_feedback_by_target_id(inference_id, None, None, Some(100))
        .await
        .unwrap();
    let demo = feedbacks
        .iter()
        .find_map(|f| match f {
            FeedbackRow::Demonstration(d) if d.id == feedback_id => Some(d),
            _ => None,
        })
        .expect("Should find demonstration feedback");
    expect_that!(demo.inference_id, eq(inference_id));
    let retrieved_value = serde_json::from_str::<Value>(&demo.value).unwrap();
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

    assert_that!(response.status().is_success(), eq(true));
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
    assert_that!(response.status(), eq(StatusCode::OK));
    let response_json = response.json::<Value>().await.unwrap();
    let feedback_id = response_json.get("feedback_id").unwrap();
    assert_that!(feedback_id.is_string(), eq(true));
    let feedback_id = Uuid::parse_str(feedback_id.as_str().unwrap()).unwrap();

    // Check DemonstrationFeedback
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;
    let feedbacks = conn
        .query_feedback_by_target_id(inference_id, None, None, Some(100))
        .await
        .unwrap();
    let demo = feedbacks
        .iter()
        .find_map(|f| match f {
            FeedbackRow::Demonstration(d) if d.id == feedback_id => Some(d),
            _ => None,
        })
        .expect("Should find demonstration feedback");
    expect_that!(demo.inference_id, eq(inference_id));
    let expected_value =
        serde_json::to_string(&json!([{"type": "text", "text": "sunny" }])).unwrap();
    expect_that!(&demo.value, eq(&expected_value));

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
    assert_that!(feedback_id.is_string(), eq(true));
    let feedback_id = Uuid::parse_str(feedback_id.as_str().unwrap()).unwrap();

    // Check DemonstrationFeedback (dynamic tool call)
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;
    let feedbacks = conn
        .query_feedback_by_target_id(inference_id, None, None, Some(100))
        .await
        .unwrap();
    let demo = feedbacks
        .iter()
        .find_map(|f| match f {
            FeedbackRow::Demonstration(d) if d.id == feedback_id => Some(d),
            _ => None,
        })
        .expect("Should find demonstration feedback");
    expect_that!(demo.inference_id, eq(inference_id));
    let retrieved_value = serde_json::from_str::<Value>(&demo.value).unwrap();
    let expected_value = json!([{"type": "tool_call", "id": "tool_call_id", "raw_name": "get_humidity", "raw_arguments": "{\"location\":\"Tokyo\"}", "name": "get_humidity", "arguments": {"location": "Tokyo"}}]);
    expect_that!(&retrieved_value, eq(&expected_value));
}

#[gtest]
#[tokio::test]
async fn test_float_feedback_normal_function() {
    test_float_feedback_with_payload(serde_json::json!({
        "function_name": "json_success",
        "input": {
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": [{"type": "template", "name": "user", "arguments": {"country": "Japan"}}]}]
        },
        "stream": false,
    })).await;
}

#[gtest]
#[tokio::test]
async fn test_float_feedback_default_function() {
    test_float_feedback_with_payload(serde_json::json!({
        "model_name": "dummy::good",
        "input": {
            "messages": [{"role": "user", "content": "Hello, world!"}]
        },
        "stream": false,
    }))
    .await;
}

async fn test_float_feedback_with_payload(inference_payload: serde_json::Value) {
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
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    // Run inference (standard, no dryrun) to get an episode_id.
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
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    let feedback_id = response_json.get("feedback_id").unwrap();
    assert!(feedback_id.is_string());
    let feedback_id = Uuid::parse_str(feedback_id.as_str().unwrap()).unwrap();

    // Check FloatMetricFeedback
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;
    let feedbacks = conn
        .query_feedback_by_target_id(episode_id, None, None, Some(100))
        .await
        .unwrap();
    let float_fb = feedbacks
        .iter()
        .find_map(|f| match f {
            FeedbackRow::Float(fl) if fl.id == feedback_id => Some(fl),
            _ => None,
        })
        .expect("Should find float feedback");
    assert_eq!(float_fb.target_id, episode_id);
    assert_eq!(float_fb.value, 32.8);
    assert_eq!(float_fb.metric_name, "user_rating");

    // Check FeedbackTag
    assert_eq!(float_fb.tags.get("key"), Some(&tag_value));

    // snapshot_hash checks are ClickHouse-specific
    if PrimaryDatastore::from_test_env() != PrimaryDatastore::Postgres {
        let clickhouse = get_clickhouse().await;
        let query = format!(
            "SELECT snapshot_hash FROM FloatMetricFeedback WHERE id = '{feedback_id}' FORMAT JSONEachRow"
        );
        let response = clickhouse
            .run_query_synchronous_no_params(query)
            .await
            .unwrap();
        let result: serde_json::Value = serde_json::from_str(&response.response).unwrap();
        assert!(
            !result["snapshot_hash"].is_null(),
            "FloatMetricFeedback should have snapshot_hash"
        );

        let query = format!(
            "SELECT snapshot_hash FROM FloatMetricFeedbackByTargetId WHERE target_id = '{episode_id}' AND id = '{feedback_id}' FORMAT JSONEachRow"
        );
        let response = clickhouse
            .run_query_synchronous_no_params(query)
            .await
            .unwrap();
        let view_result: serde_json::Value = serde_json::from_str(&response.response).unwrap();
        assert!(
            !view_result["snapshot_hash"].is_null(),
            "FloatMetricFeedbackByTargetId should have snapshot_hash"
        );
    }

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
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let response_json = response.json::<Value>().await.unwrap();
    let error_message = response_json.get("error").unwrap().as_str().unwrap();
    assert_eq!(
        error_message,
        "Feedback value for metric `user_rating` must be a number"
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
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let response_json = response.json::<Value>().await.unwrap();
    let error_message = response_json.get("error").unwrap().as_str().unwrap();
    assert!(
        error_message.contains("Correct ID was not provided for feedback level"),
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
    assert_eq!(response.status(), StatusCode::OK);

    let response_json = response.json::<Value>().await.unwrap();
    let feedback_id = response_json.get("feedback_id").unwrap();
    assert!(feedback_id.is_string());
    let feedback_id = Uuid::parse_str(feedback_id.as_str().unwrap()).unwrap();

    // Check FloatMetricFeedback (inference-level)
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;
    let feedbacks = conn
        .query_feedback_by_target_id(inference_id, None, None, Some(100))
        .await
        .unwrap();
    let float_fb = feedbacks
        .iter()
        .find_map(|f| match f {
            FeedbackRow::Float(fl) if fl.id == feedback_id => Some(fl),
            _ => None,
        })
        .expect("Should find float feedback");
    assert_eq!(float_fb.target_id, inference_id);
    assert_eq!(float_fb.value, 0.5);
    assert_eq!(float_fb.metric_name, "brevity_score");

    // snapshot_hash checks are ClickHouse-specific
    if PrimaryDatastore::from_test_env() != PrimaryDatastore::Postgres {
        let clickhouse = get_clickhouse().await;
        let query = format!(
            "SELECT snapshot_hash FROM FloatMetricFeedback WHERE id = '{feedback_id}' FORMAT JSONEachRow"
        );
        let response = clickhouse
            .run_query_synchronous_no_params(query)
            .await
            .unwrap();
        let result: serde_json::Value = serde_json::from_str(&response.response).unwrap();
        assert!(
            !result["snapshot_hash"].is_null(),
            "FloatMetricFeedback should have snapshot_hash"
        );

        let query = format!(
            "SELECT snapshot_hash FROM FloatMetricFeedbackByVariant WHERE target_id_uint = toUInt128(toUUID('{inference_id}')) AND id_uint = toUInt128(toUUID('{feedback_id}')) FORMAT JSONEachRow"
        );
        let response = clickhouse
            .run_query_synchronous_no_params(query)
            .await
            .unwrap();
        let view_result: serde_json::Value = serde_json::from_str(&response.response).unwrap();
        assert!(
            !view_result["snapshot_hash"].is_null(),
            "FloatMetricFeedbackByVariant should have snapshot_hash"
        );
    }
}

#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_float_feedback_validation_disabled() {
    let mut config = Config::new_empty()
        .await
        .unwrap()
        .into_config_without_writing_for_tests();
    let metric_config = MetricConfig {
        r#type: MetricConfigType::Float,
        optimize: MetricConfigOptimize::Max,
        level: MetricConfigLevel::Inference,
        description: None,
    };
    config
        .metrics
        .insert("user_score".to_string(), metric_config);
    let clickhouse = get_clickhouse().await;
    config.gateway.unstable_disable_feedback_target_validation = true;
    let handle = GatewayHandle::new_with_database_and_http_client(
        Arc::new(config),
        clickhouse.clone(),
        PostgresConnectionInfo::Disabled,
        ValkeyConnectionInfo::Disabled,
        ValkeyConnectionInfo::Disabled,
        TensorzeroHttpClient::new_testing().unwrap(),
        None,
        HashSet::new(), // available_tools
        HashSet::new(), // tool_whitelist
    )
    .await
    .unwrap();
    let inference_id = Uuid::now_v7();
    let params = Params {
        inference_id: Some(inference_id),
        metric_name: "user_score".to_string(),
        value: json!(3.1),
        ..Default::default()
    };
    let val = feedback(handle.app_state.clone(), params, None)
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Check that this was correctly written to ClickHouse
    let query = format!(
        "SELECT * FROM FloatMetricFeedback WHERE target_id='{inference_id}' FORMAT JsonEachRow"
    );
    let response = clickhouse
        .run_query_synchronous_no_params(query)
        .await
        .unwrap();
    let result: Value = serde_json::from_str(&response.response).unwrap();
    let clickhouse_feedback_id = Uuid::parse_str(result["id"].as_str().unwrap()).unwrap();
    expect_that!(val.feedback_id, eq(clickhouse_feedback_id));
}

#[gtest]
#[tokio::test]
async fn test_boolean_feedback_normal_function() {
    test_boolean_feedback_with_payload(serde_json::json!({
        "function_name": "json_success",
        "input": {
            "system": {"assistant_name": "Alfred Pennyworth"},
            "messages": [{"role": "user", "content": [{"type": "template", "name": "user", "arguments": {"country": "Japan"}}]}]
        },
        "stream": false,
    })).await;
}

#[gtest]
#[tokio::test]
async fn test_boolean_feedback_default_function() {
    test_boolean_feedback_with_payload(serde_json::json!({
        "model_name": "dummy::good",
        "input": {
            "messages": [{"role": "user", "content": "Hello, world!"}]
        },
        "stream": false,
    }))
    .await;
}

async fn test_boolean_feedback_with_payload(inference_payload: serde_json::Value) {
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
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    // Run inference (standard, no dryrun) to get an inference_id.
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
    assert!(feedback_id.is_string());
    let feedback_id = Uuid::parse_str(feedback_id.as_str().unwrap()).unwrap();

    // Check BooleanMetricFeedback
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;
    let feedbacks = conn
        .query_feedback_by_target_id(inference_id, None, None, Some(100))
        .await
        .unwrap();
    let bool_fb = feedbacks
        .iter()
        .find_map(|f| match f {
            FeedbackRow::Boolean(b) if b.id == feedback_id => Some(b),
            _ => None,
        })
        .expect("Should find boolean feedback");
    assert_eq!(bool_fb.target_id, inference_id);
    assert!(bool_fb.value);
    assert_eq!(bool_fb.metric_name, "task_success");

    // Check FeedbackTags
    assert_eq!(bool_fb.tags.get("key"), Some(&tag_value));
    assert_eq!(bool_fb.tags.get("key2"), Some(&tag_value2));

    // snapshot_hash checks are ClickHouse-specific
    if PrimaryDatastore::from_test_env() != PrimaryDatastore::Postgres {
        let clickhouse = get_clickhouse().await;
        let query = format!(
            "SELECT snapshot_hash FROM BooleanMetricFeedback WHERE id = '{feedback_id}' FORMAT JSONEachRow"
        );
        let response = clickhouse
            .run_query_synchronous_no_params(query)
            .await
            .unwrap();
        let result: serde_json::Value = serde_json::from_str(&response.response).unwrap();
        assert!(
            !result["snapshot_hash"].is_null(),
            "BooleanMetricFeedback should have snapshot_hash"
        );

        let query = format!(
            "SELECT snapshot_hash FROM BooleanMetricFeedbackByVariant WHERE target_id_uint = toUInt128(toUUID('{inference_id}')) AND id_uint = toUInt128(toUUID('{feedback_id}')) FORMAT JSONEachRow"
        );
        let response = clickhouse
            .run_query_synchronous_no_params(query)
            .await
            .unwrap();
        let view_result: serde_json::Value = serde_json::from_str(&response.response).unwrap();
        assert!(
            !view_result["snapshot_hash"].is_null(),
            "BooleanMetricFeedbackByVariant should have snapshot_hash"
        );
    }

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
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let response_json = response.json::<Value>().await.unwrap();
    let error_message = response_json.get("error").unwrap().as_str().unwrap();
    assert!(
        error_message.contains("Correct ID was not provided for feedback level"),
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
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let response_json = response.json::<Value>().await.unwrap();
    let error_message = response_json.get("error").unwrap().as_str().unwrap();
    assert_eq!(
        error_message,
        "Feedback value for metric `task_success` must be a boolean"
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
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    let feedback_id = response_json.get("feedback_id").unwrap();
    assert!(feedback_id.is_string());
    let feedback_id = Uuid::parse_str(feedback_id.as_str().unwrap()).unwrap();

    // Check BooleanMetricFeedback (episode-level)
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;
    let feedbacks = conn
        .query_feedback_by_target_id(episode_id, None, None, Some(100))
        .await
        .unwrap();
    let bool_fb = feedbacks
        .iter()
        .find_map(|f| match f {
            FeedbackRow::Boolean(b) if b.id == feedback_id => Some(b),
            _ => None,
        })
        .expect("Should find boolean feedback");
    assert_eq!(bool_fb.target_id, episode_id);
    assert!(bool_fb.value);
    assert_eq!(bool_fb.metric_name, "goal_achieved");

    // snapshot_hash checks are ClickHouse-specific
    if PrimaryDatastore::from_test_env() != PrimaryDatastore::Postgres {
        let clickhouse = get_clickhouse().await;
        let query = format!(
            "SELECT snapshot_hash FROM BooleanMetricFeedbackByTargetId WHERE target_id = '{episode_id}' AND id = '{feedback_id}' FORMAT JSONEachRow"
        );
        let response = clickhouse
            .run_query_synchronous_no_params(query)
            .await
            .unwrap();
        let view_result: serde_json::Value = serde_json::from_str(&response.response).unwrap();
        assert!(
            !view_result["snapshot_hash"].is_null(),
            "BooleanMetricFeedbackByTargetId should have snapshot_hash"
        );
    }
}

#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_boolean_feedback_validation_disabled() {
    let mut config = Config::new_empty()
        .await
        .unwrap()
        .into_config_without_writing_for_tests();
    let metric_config = MetricConfig {
        r#type: MetricConfigType::Boolean,
        optimize: MetricConfigOptimize::Max,
        level: MetricConfigLevel::Inference,
        description: None,
    };
    config
        .metrics
        .insert("task_success".to_string(), metric_config);
    let clickhouse = get_clickhouse().await;
    config.gateway.unstable_disable_feedback_target_validation = true;
    let handle = GatewayHandle::new_with_database_and_http_client(
        Arc::new(config),
        clickhouse.clone(),
        PostgresConnectionInfo::Disabled,
        ValkeyConnectionInfo::Disabled,
        ValkeyConnectionInfo::Disabled,
        TensorzeroHttpClient::new_testing().unwrap(),
        None,
        HashSet::new(), // available_tools
        HashSet::new(), // tool_whitelist
    )
    .await
    .unwrap();
    let inference_id = Uuid::now_v7();
    let params = Params {
        inference_id: Some(inference_id),
        metric_name: "task_success".to_string(),
        value: json!(true),
        ..Default::default()
    };
    let val = feedback(handle.app_state.clone(), params, None)
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Check that this was correctly written to ClickHouse
    let query = format!(
        "SELECT * FROM BooleanMetricFeedback WHERE target_id='{inference_id}' FORMAT JsonEachRow"
    );
    let response = clickhouse
        .run_query_synchronous_no_params(query)
        .await
        .unwrap();
    let result: Value = serde_json::from_str(&response.response).unwrap();
    let clickhouse_feedback_id = Uuid::parse_str(result["id"].as_str().unwrap()).unwrap();
    expect_that!(val.feedback_id, eq(clickhouse_feedback_id));
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
                        system: Some(System::Template(Arguments(serde_json::Map::from_iter([
                            ("assistant_name".to_string(), "Alfred Pennyworth".into()),
                        ])))),
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

#[gtest]
#[tokio::test]
async fn test_feedback_internal_tag_auto_injection() {
    let client = Client::new();

    // First, run an inference to get a valid inference_id
    let inference_payload = serde_json::json!({
        "function_name": "basic_test",
        "input": {
            "system": {"assistant_name": "Alfred"},
            "messages": [{"role": "user", "content": "Hello!"}]
        },
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&inference_payload)
        .send()
        .await
        .unwrap();

    assert_that!(response.status().is_success(), eq(true));
    let response_json = response.json::<Value>().await.unwrap();
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    sleep(Duration::from_millis(1000)).await;

    // Now send feedback with internal=true and a custom tag
    // We should NOT manually set tensorzero::internal - it should be auto-injected
    let payload = Params {
        inference_id: Some(inference_id),
        episode_id: None,
        metric_name: String::from("task_success"),
        value: serde_json::to_value(true).unwrap(),
        tags: HashMap::from([(String::from("custom_tag"), String::from("custom_value"))]),
        dryrun: None,
        internal: true,
    };

    let response = client
        .post(get_gateway_endpoint("/feedback"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_that!(response.status(), eq(StatusCode::OK));
    let response_json = response.json::<Value>().await.unwrap();
    let feedback_id = response_json.get("feedback_id").unwrap().as_str().unwrap();
    let feedback_id = Uuid::parse_str(feedback_id).unwrap();

    // Check feedback tags
    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;

    let feedbacks = conn
        .query_feedback_by_target_id(inference_id, None, None, Some(100))
        .await
        .unwrap();
    let bool_fb = feedbacks
        .iter()
        .find_map(|f| match f {
            FeedbackRow::Boolean(b) if b.id == feedback_id => Some(b),
            _ => None,
        })
        .expect("Should find boolean feedback");

    // Verify custom tag is present
    expect_that!(
        bool_fb.tags.get("custom_tag"),
        some(eq(&"custom_value".to_string()))
    );

    // Verify auto-injected tensorzero::internal tag is present
    expect_that!(
        bool_fb.tags.get("tensorzero::internal"),
        some(eq(&"true".to_string()))
    );
}
