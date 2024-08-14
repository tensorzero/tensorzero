use futures::StreamExt;
use reqwest::{Client, StatusCode};
use reqwest_eventsource::{Event, RequestBuilderExt};
use serde_json::{json, Value};
use uuid::Uuid;

use crate::e2e::common::{
    get_clickhouse, get_gateway_endpoint, select_inference_clickhouse,
    select_model_inferences_clickhouse,
};

/// Anthropic E2E tests
///
/// This file contains a set of tests that check that the Anthropic integration works and the appropriate stuff is written to DB.
///
/// Currently we test:
/// - basic inference
/// - streaming inference
///
/// TODOs (#81):
///  - tool calling
///  - JSON mode
///  - other API parameters (temp, max_tokens, etc.)

#[tokio::test]
async fn test_inference_basic() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "variant_name": "anthropic",
        "episode_id": episode_id,
        "input":
            {
               "system": {"assistant_name": "AskJeeves"},
               "messages": [
                {
                    "role": "user",
                    "content": "Hello, world!"
                }
            ]},
        "stream": false,
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    // Check Response is OK, then fields in order
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    let content_blocks = response_json.get("output").unwrap().as_array().unwrap();
    assert!(content_blocks.len() == 1);
    let content_block = content_blocks.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let content = content_block.get("text").unwrap().as_str().unwrap();
    // Check that created is here
    response_json.get("created").unwrap();
    // Check that inference_id is here
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();
    // Check that type is "chat"
    let r#type = response_json.get("type").unwrap().as_str().unwrap();
    assert_eq!(r#type, "chat");

    // Sleep for 1 second to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ClickHouse
    let clickhouse = get_clickhouse().await;

    // First, check Inference table
    let result = select_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);
    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input = json!({
        "system": {"assistant_name": "AskJeeves"},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "value": "Hello, world!"}]
            }
        ]
    });
    assert_eq!(input, correct_input);
    let content_blocks = result.get("output").unwrap().as_str().unwrap();
    // Check that content_blocks is a list of blocks length 1
    let content_blocks: Vec<Value> = serde_json::from_str(content_blocks).unwrap();
    assert_eq!(content_blocks.len(), 1);
    let content_block = content_blocks.first().unwrap();
    // Check the type and content in the block
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let clickhouse_content = content_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(clickhouse_content, content);
    // Check that episode_id is here and correct
    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);
    // Check the variant name
    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, "anthropic");
    // Check the processing time
    let processing_time_ms = result.get("processing_time_ms").unwrap().as_u64().unwrap();
    assert!(processing_time_ms > 0);

    // Check the ModelInference Table
    let result = select_model_inferences_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);
    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(input, correct_input);
    let output = result.get("output").unwrap().as_str().unwrap();
    let content_blocks: Vec<Value> = serde_json::from_str(output).unwrap();
    assert_eq!(content_blocks.len(), 1);
    let content_block = content_blocks.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let clickhouse_content = content_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(clickhouse_content, content);

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 5);
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(output_tokens > 5);
    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);
    assert!(result.get("ttft_ms").unwrap().is_null());
    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    let _ = serde_json::from_str::<Value>(raw_response).unwrap();
}

/// This test checks that streaming inference works as expected.
#[tokio::test]
async fn test_streaming() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "variant_name": "anthropic",
        "episode_id": episode_id,
        "input":
            {
               "system": {"assistant_name": "AskJeeves"},
               "messages": [
                {
                    "role": "user",
                    "content": "Hello, world!"
                }
            ]},
        "stream": true,
    });

    let mut event_source = client
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
    let mut full_content = String::new();
    for chunk in chunks {
        let chunk_json: Value = serde_json::from_str(&chunk).unwrap();
        let content_blocks = chunk_json.get("content").unwrap().as_array().unwrap();
        if content_blocks.is_empty() {
            continue;
        }
        let content_block = content_blocks.first().unwrap();
        let content = content_block.get("text").unwrap().as_str().unwrap();
        full_content.push_str(content);
        inference_id = Some(
            Uuid::parse_str(chunk_json.get("inference_id").unwrap().as_str().unwrap()).unwrap(),
        );
    }
    let inference_id = inference_id.unwrap();
    // Sleep for 1 second to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ClickHouse
    let clickhouse = get_clickhouse().await;

    // First, check Inference table
    let result = select_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);
    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input = json!({
        "system": {"assistant_name": "AskJeeves"},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "value": "Hello, world!"}]
            }
        ]
    });
    assert_eq!(input, correct_input);
    let content_blocks = result.get("output").unwrap().as_str().unwrap();
    // Check that content_blocks is a list of blocks length 1
    let content_blocks: Vec<Value> = serde_json::from_str(content_blocks).unwrap();
    assert_eq!(content_blocks.len(), 1);
    let content_block = content_blocks.first().unwrap();
    // Check the type and content in the block
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let clickhouse_content = content_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(clickhouse_content, full_content);
    // Check that episode_id is here and correct
    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);
    // Check the variant name
    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, "anthropic");
    // Check the processing time
    let processing_time_ms = result.get("processing_time_ms").unwrap().as_u64().unwrap();
    assert!(processing_time_ms > 0);

    // Check the ModelInference Table
    let result = select_model_inferences_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);
    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(input, correct_input);
    let output = result.get("output").unwrap().as_str().unwrap();
    let content_blocks: Vec<Value> = serde_json::from_str(output).unwrap();
    assert_eq!(content_blocks.len(), 1);
    let content_block = content_blocks.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let clickhouse_content = content_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(clickhouse_content, full_content);

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 5);
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(output_tokens > 5);
    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);
    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    // Check if raw_response is valid JSONL
    for line in raw_response.lines() {
        let _: Value = serde_json::from_str(line).expect("Each line should be valid JSON");
    }
    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 100);
    let ttft_ms = result.get("ttft_ms").unwrap().as_u64().unwrap();
    assert!(ttft_ms > 100);
    assert!(ttft_ms <= response_time_ms);
}
