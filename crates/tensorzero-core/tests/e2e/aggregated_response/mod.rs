//! E2E tests for `include_aggregated_response` parameter.
//!
//! Tests that the final streaming chunk includes the fully assembled inference result
//! when `include_aggregated_response` is set to `true`.

use futures::StreamExt;
use reqwest::Client;
use reqwest::StatusCode;
use reqwest_sse_stream::{Event, RequestBuilderExt};
use serde_json::{Value, json};
use uuid::Uuid;

use crate::common::get_gateway_endpoint;

#[tokio::test]
async fn test_aggregated_response_chat_streaming() {
    let episode_id = Uuid::now_v7();
    let random_suffix = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "variant_name": "openai",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "TestBot"},
            "messages": [
                {
                    "role": "user",
                    "content": format!("Say hello in one sentence. {random_suffix}")
                }
            ]
        },
        "stream": true,
        "include_aggregated_response": true
    });

    let mut chunks = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .await
        .expect("Failed to create eventsource for streaming request");

    let mut found_aggregated_response = false;
    let mut all_chunks: Vec<Value> = Vec::new();

    while let Some(chunk) = chunks.next().await {
        let chunk = chunk.expect("Failed to receive chunk from stream");
        let Event::Message(chunk) = chunk else {
            continue;
        };
        if chunk.data == "[DONE]" {
            break;
        }

        let chunk_json: Value =
            serde_json::from_str(&chunk.data).expect("Failed to parse chunk as JSON");

        all_chunks.push(chunk_json.clone());

        if let Some(collected) = chunk_json.get("aggregated_response") {
            found_aggregated_response = true;

            assert!(
                collected.is_array(),
                "aggregated_response should be an array for chat functions, got: {collected:?}"
            );

            let collected_array = collected.as_array().unwrap();
            assert!(
                !collected_array.is_empty(),
                "aggregated_response should have at least one content block"
            );

            // Verify at least one text block exists
            let has_text_block = collected_array.iter().any(|block| {
                block.get("type").and_then(|t| t.as_str()) == Some("text")
                    && block
                        .get("text")
                        .and_then(|t| t.as_str())
                        .is_some_and(|s| !s.is_empty())
            });
            assert!(
                has_text_block,
                "aggregated_response should contain at least one text block with non-empty text. Got: {collected_array:?}"
            );
        }
    }

    assert!(
        found_aggregated_response,
        "Streaming response should include aggregated_response in a chunk when include_aggregated_response=true.\n\
        Total chunks received: {}\n\
        Last few chunks:\n{:#?}",
        all_chunks.len(),
        all_chunks.iter().rev().take(3).collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn test_aggregated_response_json_streaming() {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "json_success",
        "variant_name": "openai",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "JsonBot"},
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "template", "name": "user", "arguments": {"country": "Japan"}}]
                }
            ]
        },
        "stream": true,
        "include_aggregated_response": true
    });

    let mut chunks = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .await
        .expect("Failed to create eventsource for streaming request");

    let mut found_aggregated_response = false;
    let mut all_chunks: Vec<Value> = Vec::new();

    while let Some(chunk) = chunks.next().await {
        let chunk = chunk.expect("Failed to receive chunk from stream");
        let Event::Message(chunk) = chunk else {
            continue;
        };
        if chunk.data == "[DONE]" {
            break;
        }

        let chunk_json: Value =
            serde_json::from_str(&chunk.data).expect("Failed to parse chunk as JSON");

        all_chunks.push(chunk_json.clone());

        if let Some(collected) = chunk_json.get("aggregated_response") {
            found_aggregated_response = true;

            assert!(
                collected.is_object(),
                "aggregated_response should be an object for JSON functions, got: {collected:?}"
            );

            // Verify `raw` field is a non-empty string
            let raw = collected
                .get("raw")
                .expect("aggregated_response should have a `raw` field");
            assert!(
                raw.as_str().is_some_and(|s| !s.is_empty()),
                "aggregated_response.raw should be a non-empty string, got: {raw:?}"
            );

            // Verify `parsed` field is a valid JSON object
            let parsed = collected
                .get("parsed")
                .expect("aggregated_response should have a `parsed` field");
            assert!(
                parsed.is_object(),
                "aggregated_response.parsed should be a JSON object, got: {parsed:?}"
            );
        }
    }

    assert!(
        found_aggregated_response,
        "Streaming JSON response should include aggregated_response in a chunk when include_aggregated_response=true.\n\
        Total chunks received: {}\n\
        Last few chunks:\n{:#?}",
        all_chunks.len(),
        all_chunks.iter().rev().take(3).collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn test_aggregated_response_rejected_in_non_stream_mode() {
    let payload = json!({
        "function_name": "basic_test",
        "variant_name": "openai",
        "episode_id": Uuid::now_v7(),
        "input": {
            "system": {"assistant_name": "TestBot"},
            "messages": [
                {
                    "role": "user",
                    "content": "Say hello"
                }
            ]
        },
        "stream": false,
        "include_aggregated_response": true
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .expect("Failed to send request");

    assert_eq!(
        response.status(),
        StatusCode::BAD_REQUEST,
        "`include_aggregated_response` with `stream: false` should return 400"
    );
    let response_json: Value = response
        .json()
        .await
        .expect("Failed to parse response JSON");
    assert_eq!(
        response_json,
        json!({
            "error": "`include_aggregated_response` is only supported in streaming mode",
            "error_json": {
                "InvalidRequest": {
                    "message": "`include_aggregated_response` is only supported in streaming mode"
                }
            }
        }),
        "Error response should indicate that `include_aggregated_response` requires streaming"
    );
}
