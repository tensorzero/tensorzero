use futures::StreamExt;
use reqwest::{Client, StatusCode};
use reqwest_eventsource::{Event, RequestBuilderExt};
use serde_json::{json, Value};
use tensorzero_internal::inference::{
    providers::dummy::{
        DUMMY_INFER_RESPONSE_CONTENT, DUMMY_INFER_RESPONSE_RAW, DUMMY_RAW_REQUEST,
        DUMMY_STREAMING_RESPONSE,
    },
    types::{ContentBlock, RequestMessage, Role, Text},
};
use uuid::Uuid;

use crate::common::get_gateway_endpoint;
use tensorzero_internal::clickhouse::test_helpers::{
    get_clickhouse, select_chat_inference_clickhouse, select_model_inference_clickhouse,
    select_model_inferences_clickhouse,
};

/// This test calls a function which calls a model where the provider is flaky but with retries.
#[tokio::test]
async fn e2e_test_inference_flaky() {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "variant_name": "flaky",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "AskJeeves"},
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, world!"
                }
            ]},
        "stream": false,
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
    // Check that inference_id is here
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();
    // Check that raw_content is same as content
    let content_blocks: &Vec<Value> = response_json.get("content").unwrap().as_array().unwrap();
    assert_eq!(content_blocks.len(), 1);
    let content_block = content_blocks.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let content = content_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(content, DUMMY_INFER_RESPONSE_CONTENT);

    // Check that usage is correct
    let usage = response_json.get("usage").unwrap();
    let usage = usage.as_object().unwrap();
    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    assert_eq!(input_tokens, 10);
    assert_eq!(output_tokens, 10);
    // Sleep for 1 second to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ClickHouse
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);
    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input: Value = json!(
        {
            "system": {
                "assistant_name": "AskJeeves"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "value": "Hello, world!"}]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);
    // Check that content blocks are correct
    let content_blocks = result.get("output").unwrap().as_str().unwrap();
    let content_blocks: Vec<Value> = serde_json::from_str(content_blocks).unwrap();
    assert_eq!(content_blocks.len(), 1);
    let content_block = content_blocks.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let content = content_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(content, DUMMY_INFER_RESPONSE_CONTENT);
    // Check that episode_id is here and correct
    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);
    // Check the variant name
    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, "flaky");

    // Check the ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    let id = result.get("id").unwrap().as_str().unwrap();
    let _ = Uuid::parse_str(id).unwrap();
    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    assert_eq!(input_tokens, 10);
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
    assert_eq!(output_tokens, 10);
    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);
    assert!(result.get("ttft_ms").unwrap().is_null());
    assert_eq!(
        result.get("raw_response").unwrap().as_str().unwrap(),
        DUMMY_INFER_RESPONSE_RAW
    );
    assert_eq!(
        result.get("raw_request").unwrap().as_str().unwrap(),
        DUMMY_RAW_REQUEST
    );
    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named AskJeeves"
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    assert_eq!(
        input_messages,
        vec![RequestMessage {
            role: Role::User,
            content: vec!["Hello, world!".to_string().into()],
        }]
    );
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    assert_eq!(
        output,
        vec![ContentBlock::Text(Text {
            text: DUMMY_INFER_RESPONSE_CONTENT.to_string(),
        })]
    );
}

/// This test checks that streaming inference works as expected with a flaky provider and retries.
#[tokio::test]
async fn e2e_test_streaming_flaky() {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "variant_name": "flaky",
        "episode_id": episode_id,
        "input":
            {
                "system": {
                    "assistant_name": "AskJeeves"
                },
                "messages": [
                {
                    "role": "user",
                    "content": "Hello, world!"
                }
            ]},
        "stream": true,
        "params": {
            "chat_completion": {
                "temperature": 2.0,
            "max_tokens": 200,
            "seed": 420
        }}
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
    for (i, chunk) in chunks.iter().enumerate() {
        let chunk_json: Value = serde_json::from_str(chunk).unwrap();
        if i < DUMMY_STREAMING_RESPONSE.len() {
            let content = chunk_json.get("content").unwrap().as_array().unwrap();
            assert_eq!(content.len(), 1);
            let content_block = content.first().unwrap();
            let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
            assert_eq!(content_block_type, "text");
            let content = content_block.get("text").unwrap().as_str().unwrap();
            assert_eq!(content, DUMMY_STREAMING_RESPONSE[i]);
        } else {
            assert!(chunk_json
                .get("content")
                .unwrap()
                .as_array()
                .unwrap()
                .is_empty());
            let usage = chunk_json.get("usage").unwrap().as_object().unwrap();
            let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
            let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
            assert_eq!(input_tokens, 10);
            assert_eq!(output_tokens, 16);
            inference_id = Some(
                Uuid::parse_str(chunk_json.get("inference_id").unwrap().as_str().unwrap()).unwrap(),
            );
        }
    }
    let inference_id = inference_id.unwrap();
    // Sleep for 1 second to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ClickHouse
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);
    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input: Value = json!(
        {
            "system": {
                "assistant_name": "AskJeeves"
            },
            "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "value": "Hello, world!"}]
            }
        ]}
    );
    assert_eq!(input, correct_input);
    // Check content blocks
    let content_blocks = result.get("output").unwrap().as_str().unwrap();
    let content_blocks: Vec<Value> = serde_json::from_str(content_blocks).unwrap();
    assert_eq!(content_blocks.len(), 1);
    let content_block = content_blocks.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let content = content_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(content, DUMMY_STREAMING_RESPONSE.join(""));
    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);
    // Check the variant name
    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, "flaky");
    // Check the inference_params (set via payload)
    let inference_params = result.get("inference_params").unwrap().as_str().unwrap();
    let inference_params: Value = serde_json::from_str(inference_params).unwrap();
    let chat_completion_inference_params = inference_params
        .get("chat_completion")
        .unwrap()
        .as_object()
        .unwrap();
    let temperature = chat_completion_inference_params
        .get("temperature")
        .unwrap()
        .as_f64()
        .unwrap();
    assert_eq!(temperature, 2.0);
    let max_tokens = chat_completion_inference_params
        .get("max_tokens")
        .unwrap()
        .as_u64()
        .unwrap();
    assert_eq!(max_tokens, 200);
    let seed = chat_completion_inference_params
        .get("seed")
        .unwrap()
        .as_u64()
        .unwrap();
    assert_eq!(seed, 420);

    // Check the ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    assert_eq!(input_tokens, 10);
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
    assert_eq!(output_tokens, 16);
    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);
    let ttft = result.get("ttft_ms").unwrap().as_u64().unwrap();
    assert!(ttft > 0 && ttft <= response_time_ms);
    result.get("raw_response").unwrap().as_str().unwrap();
    assert_eq!(
        result.get("raw_request").unwrap().as_str().unwrap(),
        DUMMY_RAW_REQUEST
    );
    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named AskJeeves"
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    assert_eq!(
        input_messages,
        vec![RequestMessage {
            role: Role::User,
            content: vec!["Hello, world!".to_string().into(),],
        }]
    );
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    assert_eq!(
        output,
        vec![ContentBlock::Text(Text {
            text: DUMMY_STREAMING_RESPONSE.join(""),
        })]
    );
}

/// This test calls a function which currently uses best of n.
/// We call 2 models, one that gives the usual good response, one that
/// gives a JSON response, and then use a flaky judge to select the best one (it should fail the first time but succeed the second).
/// We check that the good response is selected and that the other responses are not
/// but they get stored to the ModelInference table.
#[tokio::test]
async fn e2e_test_best_of_n_dummy_candidates_flaky_judge() {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "best_of_n",
        "variant_name": "flaky_best_of_n_variant",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [
                {
                    "role": "user",
                    "content": "Please write me a sentence about Megumin making an explosion."
                }
            ]},
        "stream": false,
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
    // Check that inference_id is here
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();
    // Check that raw_content is same as content
    let content_blocks: &Vec<Value> = response_json.get("content").unwrap().as_array().unwrap();
    assert_eq!(content_blocks.len(), 1);
    let content_block = content_blocks.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let content = content_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(content, DUMMY_INFER_RESPONSE_CONTENT);

    // Check that usage is correct
    let usage = response_json.get("usage").unwrap();
    let usage = usage.as_object().unwrap();
    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 10);
    assert!(output_tokens > 20);
    // Sleep for 1 second to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ClickHouse
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);
    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input: Value = json!(
        {
            "system": {
                "assistant_name": "AskJeeves"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "value": "Please write me a sentence about Megumin making an explosion."}]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);
    // Check that content blocks are correct
    let content_blocks = result.get("output").unwrap().as_str().unwrap();
    let content_blocks: Vec<Value> = serde_json::from_str(content_blocks).unwrap();
    assert_eq!(content_blocks.len(), 1);
    let content_block = content_blocks.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let content = content_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(content, DUMMY_INFER_RESPONSE_CONTENT);
    // Check that episode_id is here and correct
    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);
    // Check the variant name
    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, "flaky_best_of_n_variant");

    // Check the ModelInference Table
    let results = select_model_inferences_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    assert_eq!(results.len(), 3);

    // Collect model names
    let mut model_names = std::collections::HashSet::new();

    for result in results {
        let id = result.get("id").unwrap().as_str().unwrap();
        let _ = Uuid::parse_str(id).unwrap();
        let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
        let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
        assert_eq!(inference_id_result, inference_id);

        // Collect model_name
        let model_name = result.get("model_name").unwrap().as_str().unwrap();
        model_names.insert(model_name.to_string());

        // Check that all expected fields are present
        assert!(result.get("model_provider_name").is_some());
        assert!(result.get("raw_request").is_some());
        assert!(result.get("raw_response").is_some());
        assert!(result.get("input_tokens").is_some());
        assert!(result.get("output_tokens").is_some());
        assert!(result.get("response_time_ms").is_some());
        assert!(result.get("ttft_ms").is_some());

        let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
        assert!(input_tokens > 0);
        let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
        assert!(output_tokens > 0);
        let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
        assert!(response_time_ms > 0);
        assert!(result.get("ttft_ms").unwrap().is_null());
    }

    // Check that all expected model names are present
    let expected_model_names: std::collections::HashSet<String> =
        ["test", "json", "flaky_best_of_n_judge"]
            .iter()
            .map(|s| s.to_string())
            .collect();
    assert_eq!(model_names, expected_model_names);
}
