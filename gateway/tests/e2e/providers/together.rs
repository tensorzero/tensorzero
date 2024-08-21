use futures::StreamExt;
use reqwest::{Client, StatusCode};
use reqwest_eventsource::{Event, RequestBuilderExt};
use serde_json::{json, Value};
use uuid::Uuid;

use crate::common::{
    get_clickhouse, get_gateway_endpoint, select_inference_clickhouse,
    select_model_inferences_clickhouse,
};
use crate::providers::common::{E2ETestProvider, E2ETestProviders};

crate::generate_provider_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    E2ETestProviders::with_provider(E2ETestProvider {
        variant_name: "fireworks".to_string(),
    })
}

/// Together E2E tests
///
/// This file contains a set of tests that check that the Together integration works and the appropriate stuff is written to DB.
///
/// Currently we test:
/// - basic inference
/// - streaming inference
/// - JSON mode
///
/// TODOs (#81):
///  - tool calling
///  - other API parameters (temp, max_tokens, etc.)

#[tokio::test]
async fn test_json_request() {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "json_success",
        "episode_id": episode_id,
        "input":
            {
                "system": {"assistant_name": "AskJeeves"},
                "messages": [
                {
                    "role": "user",
                    "content": "What's the capital of Texas?"
                }
            ]},
        "stream": false,
        "variant_name": "together"
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    // Check Response is OK, then fields in order
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    let output = response_json.get("output").unwrap().as_object().unwrap();
    let parsed = output.get("parsed").unwrap().as_object().unwrap();
    parsed.get("answer").unwrap().as_str().unwrap();
    output.get("raw").unwrap().as_str().unwrap();
    // Check that inference_id is here
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();
    // Check that episode_id is here
    let response_episode_id = response_json.get("episode_id").unwrap().as_str().unwrap();
    let response_episode_id = Uuid::parse_str(response_episode_id).unwrap();
    assert_eq!(response_episode_id, episode_id);
    // Check that variant_name is here
    let variant_name = response_json.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, "together");

    // Check that usage is correct
    let usage = response_json.get("usage").unwrap();
    let usage = usage.as_object().unwrap();
    let prompt_tokens = usage.get("prompt_tokens").unwrap().as_u64().unwrap();
    let completion_tokens = usage.get("completion_tokens").unwrap().as_u64().unwrap();
    assert!(prompt_tokens > 10);
    assert!(completion_tokens > 5);
    // Sleep for 1 second to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ClickHouse
    let clickhouse = get_clickhouse().await;
    let result = select_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);
    let correct_input = json!({
        "system": {"assistant_name": "AskJeeves"},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "value": "What's the capital of Texas?"}]
            }
        ]
    });
    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(input, correct_input);
    // Check that correctly parsed output is present
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Value = serde_json::from_str(output).unwrap();
    let parsed = output.get("parsed").unwrap().as_object().unwrap();
    parsed.get("answer").unwrap().as_str().unwrap();
    output.get("raw").unwrap().as_str().unwrap();
    // Check content blocks
    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);
    // Check the variant name
    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, "together");

    // Check the ModelInference Table
    let result = select_model_inferences_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    let id = result.get("id").unwrap().as_str().unwrap();
    let _ = Uuid::parse_str(id).unwrap();
    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let input_result = result.get("input").unwrap().as_str().unwrap();
    let input_result: Value = serde_json::from_str(input_result).unwrap();
    assert_eq!(input_result, correct_input);
    let output = result.get("output").unwrap().as_str().unwrap();
    let content_blocks: Vec<Value> = serde_json::from_str(output).unwrap();
    assert_eq!(content_blocks.len(), 1);
    let content_block = content_blocks.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    content_block.get("text").unwrap().as_str().unwrap();

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 10);
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(output_tokens > 5);
    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);
    assert!(result.get("ttft_ms").unwrap().is_null());
    result.get("raw_response").unwrap().as_str().unwrap();
}

#[tokio::test]
async fn test_json_request_streaming() {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "json_success",
        "episode_id": episode_id,
        "input":
            {
                "system": {"assistant_name": "AskJeeves"},
                "messages": [
                {
                    "role": "user",
                    "content": "What's the capital of Texas?"
                }
            ]},
        "stream": true,
        "variant_name": "together"
    });

    let mut event_source = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .unwrap();
    let mut chunks = vec![];
    while let Some(event) = event_source.next().await {
        let event = event.unwrap();
        match event {
            Event::Open => continue,
            Event::Message(message) => {
                if message.data == "[DONE]" {
                    break;
                }
                chunks.push(message.data);
            }
        }
    }
    let mut inference_id = None;
    let mut content = String::new();
    let mut prompt_tokens = 0;
    let mut completion_tokens = 0;

    for chunk in chunks.iter() {
        let chunk_json: Value = serde_json::from_str(chunk).unwrap();
        if let Some(usage) = chunk_json.get("usage").and_then(|u| u.as_object()) {
            prompt_tokens += usage.get("prompt_tokens").unwrap().as_u64().unwrap();
            completion_tokens += usage.get("completion_tokens").unwrap().as_u64().unwrap();
        }
        let raw = chunk_json.get("raw").unwrap().as_str().unwrap();
        content.push_str(raw);
        let response_inference_id = chunk_json.get("inference_id").unwrap().as_str().unwrap();
        let response_inference_id = Uuid::parse_str(response_inference_id).unwrap();
        match inference_id {
            Some(inference_id) => assert_eq!(response_inference_id, inference_id),
            None => inference_id = Some(response_inference_id),
        }
    }
    let parsed: Value = serde_json::from_str(&content).unwrap();
    let inference_id = inference_id.unwrap();
    // Sleep for 1 second to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ClickHouse
    let clickhouse = get_clickhouse().await;
    let result = select_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);
    let correct_input = json!({
        "system": {"assistant_name": "AskJeeves"},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "value": "What's the capital of Texas?"}]
            }
        ]
    });
    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(input, correct_input);
    // Check that correctly parsed output is present
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Value = serde_json::from_str(output).unwrap();
    let parsed_db = output.get("parsed").unwrap();
    assert_eq!(&parsed, parsed_db);
    parsed.get("answer").unwrap().as_str().unwrap();
    let raw_db = output.get("raw").unwrap().as_str().unwrap();
    assert_eq!(content, raw_db);
    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);
    // Check the variant name
    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, "together");

    // Check the ModelInference Table
    let result = select_model_inferences_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    let id = result.get("id").unwrap().as_str().unwrap();
    let _ = Uuid::parse_str(id).unwrap();
    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let input_result = result.get("input").unwrap().as_str().unwrap();
    let input_result: Value = serde_json::from_str(input_result).unwrap();
    assert_eq!(input_result, correct_input);
    let output = result.get("output").unwrap().as_str().unwrap();
    let content_blocks: Vec<Value> = serde_json::from_str(output).unwrap();
    assert_eq!(content_blocks.len(), 1);
    let content_block = content_blocks.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    content_block.get("text").unwrap().as_str().unwrap();

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    assert_eq!(input_tokens, prompt_tokens);
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
    assert_eq!(output_tokens, completion_tokens);
    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);
    assert!(result.get("ttft_ms").unwrap().as_u64().unwrap() > 50);
    result.get("raw_response").unwrap().as_str().unwrap();
}
