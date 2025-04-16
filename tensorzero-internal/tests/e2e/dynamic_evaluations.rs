#![allow(clippy::print_stdout)]
use std::{
    collections::HashMap,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use crate::providers::common::{make_embedded_gateway, make_http_gateway};
use serde_json::json;
use tensorzero::{
    ClientInferenceParams, ClientInput, ClientInputMessage, ClientInputMessageContent,
    DynamicEvaluationRunParams, InferenceOutput, Role,
};
use tensorzero_internal::{
    clickhouse::test_helpers::{get_clickhouse, select_chat_inference_clickhouse},
    inference::types::TextKind,
};
use uuid::{Timestamp, Uuid};

#[tokio::test]
async fn test_dynamic_evaluation() {
    let client = make_http_gateway().await;
    let params = DynamicEvaluationRunParams {
        variants: HashMap::from([("basic_test".to_string(), "test2".to_string())]),
        tags: HashMap::from([("foo".to_string(), "bar".to_string())]),
        project_name: Some("test_project".to_string()),
        display_name: Some("test_display_name".to_string()),
    };
    let result = client.dynamic_evaluation_run(params).await.unwrap();
    let run_id = result.run_id;
    println!("Run ID: {run_id}");
    // Run an inference with the episode_id given
    let inference_params = ClientInferenceParams {
        episode_id: Some(episode_id),
        function_name: Some("basic_test".to_string()),
        input: ClientInput {
            system: Some(json!({
                "assistant_name": "AskJeeves",
            })),
            messages: vec![ClientInputMessage {
                role: Role::User,
                content: vec![ClientInputMessageContent::Text(TextKind::Text {
                    text: "Please write me a sentence about Megumin making an explosion.".into(),
                })],
            }],
        },
        tags: HashMap::from([("baz".to_string(), "bat".to_string())]),
        ..Default::default()
    };
    let response = if let InferenceOutput::NonStreaming(response) =
        client.inference(inference_params).await.unwrap()
    {
        response
    } else {
        panic!("Expected a non-streaming response");
    };
    // We won't test the output here but will grab from ClickHouse so we can check the variant name
    // and tags
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, response.inference_id())
        .await
        .unwrap();

    println!("ClickHouse - ChatInference: {result:#?}");
    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, "test2");
    let tags = result.get("tags").unwrap().as_object().unwrap();
    assert_eq!(tags.get("foo").unwrap().as_str().unwrap(), "bar");
    assert_eq!(tags.get("baz").unwrap().as_str().unwrap(), "bat");
}

#[tokio::test]
async fn test_dynamic_evaluation_nonexistent_function() {
    let client = make_http_gateway().await;
    let params = DynamicEvaluationRunParams {
        variants: HashMap::from([("nonexistent_function".to_string(), "test2".to_string())]),
        tags: HashMap::from([("foo".to_string(), "bar".to_string())]),
    };
    let result = client.dynamic_evaluation_run(params).await.unwrap_err();
    assert!(result
        .to_string()
        .contains("Unknown function: nonexistent_function"));
}

/// Test that the variant behavior is default if we use a different function name
/// But the tags are applied
#[tokio::test(flavor = "multi_thread")]
async fn test_dynamic_evaluation_other_function() {
    let client = make_embedded_gateway().await;
    let params = DynamicEvaluationRunParams {
        variants: HashMap::from([("dynamic_json".to_string(), "gcp-vertex-haiku".to_string())]),
        tags: HashMap::from([("foo".to_string(), "bar".to_string())]),
    };
    let result = client.dynamic_evaluation_run(params).await.unwrap();
    let episode_id = result.episode_id;
    println!("Episode ID: {episode_id}");
    // Run an inference with the episode_id given
    let inference_params = ClientInferenceParams {
        episode_id: Some(episode_id),
        function_name: Some("basic_test".to_string()),
        input: ClientInput {
            system: Some(json!({
                "assistant_name": "AskJeeves",
            })),
            messages: vec![ClientInputMessage {
                role: Role::User,
                content: vec![ClientInputMessageContent::Text(TextKind::Text {
                    text: "Please write me a sentence about Megumin making an explosion.".into(),
                })],
            }],
        },
        ..Default::default()
    };
    let response = if let InferenceOutput::NonStreaming(response) =
        client.inference(inference_params).await.unwrap()
    {
        response
    } else {
        panic!("Expected a non-streaming response");
    };
    // We won't test the output here but will grab from ClickHouse so we can check the variant name
    // and tags
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, response.inference_id())
        .await
        .unwrap();

    println!("ClickHouse - ChatInference: {result:#?}");
    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, "test");
    let tags = result.get("tags").unwrap().as_object().unwrap();
    assert_eq!(tags.get("foo").unwrap().as_str().unwrap(), "bar");
}

/// Test that the variant behavior is default if we use a different function name
/// But the tags are applied
#[tokio::test(flavor = "multi_thread")]
async fn test_dynamic_evaluation_override_variant_tags() {
    let client = make_embedded_gateway().await;
    let params = DynamicEvaluationRunParams {
        variants: HashMap::from([("basic_test".to_string(), "error".to_string())]),
        tags: HashMap::from([("foo".to_string(), "bar".to_string())]),
    };
    let result = client.dynamic_evaluation_run(params).await.unwrap();
    let episode_id = result.episode_id;
    println!("Episode ID: {episode_id}");
    // Run an inference with the episode_id given
    let inference_params = ClientInferenceParams {
        episode_id: Some(episode_id),
        function_name: Some("basic_test".to_string()),
        input: ClientInput {
            system: Some(json!({
                "assistant_name": "AskJeeves",
            })),
            messages: vec![ClientInputMessage {
                role: Role::User,
                content: vec![ClientInputMessageContent::Text(TextKind::Text {
                    text: "Please write me a sentence about Megumin making an explosion.".into(),
                })],
            }],
        },
        variant_name: Some("test2".to_string()),
        tags: HashMap::from([("foo".to_string(), "baz".to_string())]),
        ..Default::default()
    };
    let response = if let InferenceOutput::NonStreaming(response) =
        client.inference(inference_params).await.unwrap()
    {
        response
    } else {
        panic!("Expected a non-streaming response");
    };
    // We won't test the output here but will grab from ClickHouse so we can check the variant name
    // and tags
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, response.inference_id())
        .await
        .unwrap();

    println!("ClickHouse - ChatInference: {result:#?}");
    // Test that inference time settings override the dynamic evaluation run settings
    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, "test2");
    let tags = result.get("tags").unwrap().as_object().unwrap();
    assert_eq!(tags.get("foo").unwrap().as_str().unwrap(), "baz");
}

#[tokio::test]
async fn test_bad_dynamic_evaluation_run() {
    let client = make_http_gateway().await;
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards");
    let now_plus_offset = now + Duration::from_secs(100_000_000_000);
    let timestamp = Timestamp::from_unix_time(
        now_plus_offset.as_secs(),
        now_plus_offset.subsec_nanos(),
        0, // counter
        0, // usable_counter_bits
    );
    let episode_id = Uuid::new_v7(timestamp);
    // Run an inference with the episode_id given
    let inference_params = ClientInferenceParams {
        episode_id: Some(episode_id),
        function_name: Some("basic_test".to_string()),
        input: ClientInput {
            system: Some(json!({
                "assistant_name": "AskJeeves",
            })),
            messages: vec![ClientInputMessage {
                role: Role::User,
                content: vec![ClientInputMessageContent::Text(TextKind::Text {
                    text: "Please write me a sentence about Megumin making an explosion.".into(),
                })],
            }],
        },
        ..Default::default()
    };
    let response = client.inference(inference_params).await.unwrap_err();
    println!("Response: {response:#?}");
    assert!(response
        .to_string()
        .contains("Dynamic evaluation run not found"));
}
