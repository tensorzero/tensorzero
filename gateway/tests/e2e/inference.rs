use futures::StreamExt;
use gateway::inference::providers::dummy::{
    DUMMY_BAD_TOOL_RESPONSE, DUMMY_INFER_RESPONSE_CONTENT, DUMMY_INFER_RESPONSE_RAW,
    DUMMY_STREAMING_RESPONSE, DUMMY_STREAMING_TOOL_RESPONSE, DUMMY_TOOL_RESPONSE,
};
use reqwest::{Client, StatusCode};
use reqwest_eventsource::{Event, RequestBuilderExt};
use serde_json::{json, Value};
use uuid::Uuid;

use crate::e2e::common::{
    get_clickhouse, get_gateway_endpoint, select_inference_clickhouse,
    select_model_inferences_clickhouse,
};

#[tokio::test]
async fn e2e_test_inference_basic() {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "AskJeeves"},
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "value": "Hello, world!"}]
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
    let content_blocks = response_json.get("output").unwrap().as_array().unwrap();
    assert!(content_blocks.len() == 1);
    let content_block = content_blocks.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let content = content_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(content, DUMMY_INFER_RESPONSE_CONTENT);
    // Check that created is here
    response_json.get("created").unwrap();
    // Check that inference_id is here
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();
    // Check that parsed_output is not here
    assert!(response_json.get("parsed_output").is_none());
    // Check that type is "chat"
    let r#type = response_json.get("type").unwrap().as_str().unwrap();
    assert_eq!(r#type, "chat");

    // Check that usage is correct
    let usage = response_json.get("usage").unwrap();
    let usage = usage.as_object().unwrap();
    let prompt_tokens = usage.get("prompt_tokens").unwrap().as_u64().unwrap();
    let completion_tokens = usage.get("completion_tokens").unwrap().as_u64().unwrap();
    assert_eq!(prompt_tokens, 10);
    assert_eq!(completion_tokens, 10);
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
    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(input, payload["input"]);
    let content_blocks = result.get("output").unwrap().as_str().unwrap();
    // Check that content_blocks is a list of blocks length 1
    let content_blocks: Vec<Value> = serde_json::from_str(content_blocks).unwrap();
    assert_eq!(content_blocks.len(), 1);
    let content_block = content_blocks.first().unwrap();
    // Check the type and content in the block
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
    assert_eq!(variant_name, "test");

    // Check the ModelInference Table
    let result = select_model_inferences_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);
    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(input, payload["input"]);
    let output = result.get("output").unwrap().as_str().unwrap();
    let content_blocks: Vec<Value> = serde_json::from_str(output).unwrap();
    assert_eq!(content_blocks.len(), 1);
    let content_block = content_blocks.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let content = content_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(content, DUMMY_INFER_RESPONSE_CONTENT);

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
}

#[tokio::test]
async fn e2e_test_inference_dryrun() {
    let payload = json!({
        "function_name": "basic_test",
        "episode_id": Uuid::now_v7(),
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, world!"
                }
            ]
        },
        "stream": false,
        "dryrun": true,
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

    // Sleep for 1 second to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ClickHouse
    let clickhouse = get_clickhouse().await;
    let result = select_inference_clickhouse(&clickhouse, inference_id).await;
    assert!(result.is_none()); // No inference should be written to ClickHouse when dryrun is true
}

/// This test calls a function which calls a model where the first provider is broken but
/// then the second provider works fine. We expect this request to work despite the first provider
/// being broken.
#[tokio::test]
async fn e2e_test_inference_model_fallback() {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "model_fallback_test",
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
    // Check that created is here
    response_json.get("created").unwrap();
    // Check that inference_id is here
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();
    // Check that raw_content is same as content
    let content_blocks: &Vec<Value> = response_json.get("output").unwrap().as_array().unwrap();
    assert_eq!(content_blocks.len(), 1);
    let content_block = content_blocks.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let content = content_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(content, DUMMY_INFER_RESPONSE_CONTENT);
    // Check that type is "chat"
    let r#type = response_json.get("type").unwrap().as_str().unwrap();
    assert_eq!(r#type, "chat");

    // Check that usage is correct
    let usage = response_json.get("usage").unwrap();
    let usage = usage.as_object().unwrap();
    let prompt_tokens = usage.get("prompt_tokens").unwrap().as_u64().unwrap();
    let completion_tokens = usage.get("completion_tokens").unwrap().as_u64().unwrap();
    assert_eq!(prompt_tokens, 10);
    assert_eq!(completion_tokens, 10);
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
    assert_eq!(variant_name, "test");

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
    let content = content_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(content, DUMMY_INFER_RESPONSE_CONTENT);

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
}

#[tokio::test]
async fn e2e_test_tool_call() {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "weather_helper",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "AskJeeves"},
            "messages": [
                {
                    "role": "user",
                    "content": "Hi I'm visiting Brooklyn from Brazil. What's the weather?"
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
    // No output schema so parsed content should not be in response
    assert!(response_json.get("parsed_content").is_none());
    // Check that created is here
    response_json.get("created").unwrap();
    // Check that inference_id is here
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();
    // Check that raw_content is same as content
    let content_blocks: &Vec<Value> = response_json.get("output").unwrap().as_array().unwrap();
    assert_eq!(content_blocks.len(), 1);
    let content_block = content_blocks.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "tool_call");
    let arguments = content_block.get("arguments").unwrap().as_str().unwrap();
    let arguments: Value = serde_json::from_str(arguments).unwrap();
    assert_eq!(arguments, *DUMMY_TOOL_RESPONSE);
    let id = content_block.get("id").unwrap().as_str().unwrap();
    assert_eq!(id, "0");
    let name = content_block.get("name").unwrap().as_str().unwrap();
    assert_eq!(name, "get_weather");
    let parsed_name = content_block.get("parsed_name").unwrap().as_str().unwrap();
    assert_eq!(parsed_name, "get_weather");
    let parsed_arguments = content_block.get("parsed_arguments").unwrap();
    assert_eq!(parsed_arguments, &*DUMMY_TOOL_RESPONSE,);

    // Check that type is "chat"
    // Check that usage is correct
    let usage = response_json.get("usage").unwrap();
    let usage = usage.as_object().unwrap();
    let prompt_tokens = usage.get("prompt_tokens").unwrap().as_u64().unwrap();
    let completion_tokens = usage.get("completion_tokens").unwrap().as_u64().unwrap();
    assert_eq!(prompt_tokens, 10);
    assert_eq!(completion_tokens, 10);
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
                    "content": [{"type": "text", "value": "Hi I'm visiting Brooklyn from Brazil. What's the weather?"}]
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
    assert_eq!(content_block_type, "tool_call");
    // Check that the tool call is correctly stored
    let arguments = content_block.get("arguments").unwrap().as_str().unwrap();
    let arguments: Value = serde_json::from_str(arguments).unwrap();
    assert_eq!(arguments, *DUMMY_TOOL_RESPONSE);
    let id = content_block.get("id").unwrap().as_str().unwrap();
    assert_eq!(id, "0");
    let name = content_block.get("name").unwrap().as_str().unwrap();
    assert_eq!(name, "get_weather");
    let parsed_name = content_block.get("parsed_name").unwrap().as_str().unwrap();
    assert_eq!(parsed_name, "get_weather");
    let parsed_arguments = content_block.get("parsed_arguments").unwrap();
    assert_eq!(parsed_arguments, &*DUMMY_TOOL_RESPONSE,);
    // Check that episode_id is here and correct
    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);
    // Check the variant name
    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, "variant");
    // Check the dynamic_tool_config
    let dynamic_tool_config = result.get("dynamic_tool_config").unwrap().as_str().unwrap();
    let dynamic_tool_config: Value = serde_json::from_str(dynamic_tool_config).unwrap();
    assert!(dynamic_tool_config.get("allowed_tools").unwrap().is_null());
    assert!(dynamic_tool_config
        .get("additional_tools")
        .unwrap()
        .is_null());
    assert!(dynamic_tool_config.get("tool_choice").unwrap().is_null());
    assert!(dynamic_tool_config
        .get("parallel_tool_calls")
        .unwrap()
        .is_null());
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
    assert_eq!(content_block_type, "tool_call");
    // Check that the tool call is correctly stored (no parsing here)
    let arguments = content_block.get("arguments").unwrap().as_str().unwrap();
    let arguments: Value = serde_json::from_str(arguments).unwrap();
    assert_eq!(arguments, *DUMMY_TOOL_RESPONSE);
    let id = content_block.get("id").unwrap().as_str().unwrap();
    assert_eq!(id, "0");
    let name = content_block.get("name").unwrap().as_str().unwrap();
    assert_eq!(name, "get_weather");

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    assert_eq!(input_tokens, 10);
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
    assert_eq!(output_tokens, 10);
    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);
    assert!(result.get("ttft_ms").unwrap().is_null());
    assert_eq!(
        result.get("raw_response").unwrap().as_str().unwrap(),
        serde_json::to_string(&*DUMMY_TOOL_RESPONSE).unwrap()
    );
}

#[tokio::test]
async fn e2e_test_tool_call_malformed() {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "weather_helper",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "AskJeeves"},
            "messages": [
                {
                    "role": "user",
                    "content": "Hi I'm visiting Brooklyn from Brazil. What's the weather?"
                }
            ]},
        "stream": false,
        "variant_name": "bad_tool"
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
    // No output schema so parsed content should not be in response
    assert!(response_json.get("parsed_content").is_none());
    // Check that created is here
    response_json.get("created").unwrap();
    // Check that inference_id is here
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();
    // Check that raw_content is same as content
    let content_blocks: &Vec<Value> = response_json.get("output").unwrap().as_array().unwrap();
    assert_eq!(content_blocks.len(), 1);
    let content_block = content_blocks.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "tool_call");
    let arguments = content_block.get("arguments").unwrap().as_str().unwrap();
    let arguments: Value = serde_json::from_str(arguments).unwrap();
    assert_eq!(arguments, *DUMMY_BAD_TOOL_RESPONSE);
    let id = content_block.get("id").unwrap().as_str().unwrap();
    assert_eq!(id, "0");
    let name = content_block.get("name").unwrap().as_str().unwrap();
    assert_eq!(name, "get_weather");
    let parsed_name = content_block.get("parsed_name").unwrap().as_str().unwrap();
    assert_eq!(parsed_name, "get_weather");
    let parsed_arguments = content_block.get("parsed_arguments").unwrap();
    assert!(parsed_arguments.is_null());

    // Check that type is "chat"
    // Check that usage is correct
    let usage = response_json.get("usage").unwrap();
    let usage = usage.as_object().unwrap();
    let prompt_tokens = usage.get("prompt_tokens").unwrap().as_u64().unwrap();
    let completion_tokens = usage.get("completion_tokens").unwrap().as_u64().unwrap();
    assert_eq!(prompt_tokens, 10);
    assert_eq!(completion_tokens, 10);
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
                    "content": [{"type": "text", "value": "Hi I'm visiting Brooklyn from Brazil. What's the weather?"}]
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
    assert_eq!(content_block_type, "tool_call");
    // Check that the tool call is correctly stored
    let arguments = content_block.get("arguments").unwrap().as_str().unwrap();
    let arguments: Value = serde_json::from_str(arguments).unwrap();
    assert_eq!(arguments, *DUMMY_BAD_TOOL_RESPONSE);
    let id = content_block.get("id").unwrap().as_str().unwrap();
    assert_eq!(id, "0");
    let name = content_block.get("name").unwrap().as_str().unwrap();
    assert_eq!(name, "get_weather");
    let parsed_name = content_block.get("parsed_name").unwrap().as_str().unwrap();
    assert_eq!(parsed_name, "get_weather");
    let parsed_arguments = content_block.get("parsed_arguments").unwrap();
    assert!(parsed_arguments.is_null());
    // Check that episode_id is here and correct
    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);
    // Check the variant name
    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, "bad_tool");
    // Check the dynamic_tool_config
    let dynamic_tool_config = result.get("dynamic_tool_config").unwrap().as_str().unwrap();
    let dynamic_tool_config: Value = serde_json::from_str(dynamic_tool_config).unwrap();
    assert!(dynamic_tool_config.get("allowed_tools").unwrap().is_null());
    assert!(dynamic_tool_config
        .get("additional_tools")
        .unwrap()
        .is_null());
    assert!(dynamic_tool_config.get("tool_choice").unwrap().is_null());
    assert!(dynamic_tool_config
        .get("parallel_tool_calls")
        .unwrap()
        .is_null());
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
    assert_eq!(content_block_type, "tool_call");
    // Check that the tool call is correctly stored (no parsing here)
    let arguments = content_block.get("arguments").unwrap().as_str().unwrap();
    let arguments: Value = serde_json::from_str(arguments).unwrap();
    assert_eq!(arguments, *DUMMY_BAD_TOOL_RESPONSE);
    let id = content_block.get("id").unwrap().as_str().unwrap();
    assert_eq!(id, "0");
    let name = content_block.get("name").unwrap().as_str().unwrap();
    assert_eq!(name, "get_weather");

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    assert_eq!(input_tokens, 10);
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
    assert_eq!(output_tokens, 10);
    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);
    assert!(result.get("ttft_ms").unwrap().is_null());
    result.get("raw_response").unwrap().as_str().unwrap();
}

/// This test checks the return type and clickhouse writes for a function with an output schema and
/// a response which does not satisfy the schema.
/// We expect to see a null `parsed_content` field in the response and a null `parsed_content` field in the table.
// #[tokio::test]
// async fn e2e_test_inference_json_fail() {
//     // TODO: Fix this test
//     return;
//     let episode_id = Uuid::now_v7();

//     let payload = json!({
//         "function_name": "json_fail",
//         "episode_id": episode_id,
//         "input":
//             {
//                 "system": {"assistant_name": "AskJeeves"},
//                 "messages": [
//                 {
//                     "role": "user",
//                     "content": "Hello, world!"
//                 }
//             ]},
//         "stream": false,
//     });

//     let response = Client::new()
//         .post(get_gateway_endpoint("/inference"))
//         .json(&payload)
//         .send()
//         .await
//         .unwrap();
//     // Check Response is OK, then fields in order
//     assert_eq!(response.status(), StatusCode::OK);
//     let response_json = response.json::<Value>().await.unwrap();
//     // Check that parsed_output is present but null
//     let parsed_output = response_json.get("parsed_output").unwrap();
//     assert!(parsed_output.is_null());
//     // Check that created is here
//     response_json.get("created").unwrap();
//     // Check that inference_id is here
//     let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
//     let inference_id = Uuid::parse_str(inference_id).unwrap();
//     // Check that content_blocks are present and correct
//     let content_blocks = response_json
//         .get("content_blocks")
//         .unwrap()
//         .as_array()
//         .unwrap();
//     assert_eq!(content_blocks.len(), 1);
//     let content_block = content_blocks.first().unwrap();
//     let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
//     assert_eq!(content_block_type, "text");
//     let content = content_block.get("text").unwrap().as_str().unwrap();
//     assert_eq!(content, DUMMY_INFER_RESPONSE_CONTENT);
//     // Check that type is "chat"
//     let r#type = response_json.get("type").unwrap().as_str().unwrap();
//     assert_eq!(r#type, "chat");

//     // Check that usage is correct
//     let usage = response_json.get("usage").unwrap();
//     let usage = usage.as_object().unwrap();
//     let prompt_tokens = usage.get("prompt_tokens").unwrap().as_u64().unwrap();
//     let completion_tokens = usage.get("completion_tokens").unwrap().as_u64().unwrap();
//     assert_eq!(prompt_tokens, 10);
//     assert_eq!(completion_tokens, 10);
//     // Sleep for 1 second to allow time for data to be inserted into ClickHouse (trailing writes from API)
//     tokio::time::sleep(std::time::Duration::from_secs(1)).await;

//     // Check ClickHouse
//     let clickhouse = get_clickhouse().await;
//     let result = select_inference_clickhouse(&clickhouse, inference_id)
//         .await
//         .unwrap();
//     let id = result.get("id").unwrap().as_str().unwrap();
//     let id_uuid = Uuid::parse_str(id).unwrap();
//     assert_eq!(id_uuid, inference_id);
//     let input: Value =
//         serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
//     let correct_input = json!({
//         "system": {"assistant_name": "AskJeeves"},
//         "messages": [
//             {
//                 "role": "user",
//                 "content": [{"type": "text", "value": "Hello, world!"}]
//             }
//         ]
//     });
//     assert_eq!(input, correct_input);
//     let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
//     let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
//     assert_eq!(retrieved_episode_id, episode_id);
//     // Check the variant name
//     let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
//     assert_eq!(variant_name, "test");

//     // Check the ModelInference Table
//     let result = select_model_inferences_clickhouse(&clickhouse, inference_id)
//         .await
//         .unwrap();
//     let id = result.get("id").unwrap().as_str().unwrap();
//     let _ = Uuid::parse_str(id).unwrap();
//     let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
//     let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
//     assert_eq!(inference_id_result, inference_id);

//     let input_result = result.get("input").unwrap().as_str().unwrap();
//     let input_result: Value = serde_json::from_str(input_result).unwrap();
//     assert_eq!(input_result, correct_input);
//     let output = result.get("output").unwrap().as_str().unwrap();
//     let content_blocks: Vec<Value> = serde_json::from_str(output).unwrap();
//     assert_eq!(content_blocks.len(), 1);
//     let content_block = content_blocks.first().unwrap();
//     let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
//     assert_eq!(content_block_type, "text");
//     let content = content_block.get("text").unwrap().as_str().unwrap();
//     assert_eq!(content, DUMMY_INFER_RESPONSE_CONTENT);

//     let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
//     assert_eq!(input_tokens, 10);
//     let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
//     assert_eq!(output_tokens, 10);
//     let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
//     assert!(response_time_ms > 0);
//     assert!(result.get("ttft_ms").unwrap().is_null());
//     assert_eq!(
//         result.get("raw_response").unwrap().as_str().unwrap(),
//         DUMMY_INFER_RESPONSE_RAW
//     );
// }

/// This test checks the return type and clickhouse writes for a function with an output schema and
/// a response which satisfies the schema.
/// We expect to see a filled-out `content` field in the response and a filled-out `output` field in the table.
// #[tokio::test]
// async fn e2e_test_inference_json_succeed() {
//     let episode_id = Uuid::now_v7();

//     let payload = json!({
//         "function_name": "json_succeed",
//         "episode_id": episode_id,
//         "input":
//             {
//                 "system": {"assistant_name": "AskJeeves"},
//                 "messages": [
//                 {
//                     "role": "user",
//                     "content": "Hello, world!"
//                 }
//             ]},
//         "stream": false,
//     });

//     let response = Client::new()
//         .post(get_gateway_endpoint("/inference"))
//         .json(&payload)
//         .send()
//         .await
//         .unwrap();
//     // Check Response is OK, then fields in order
//     assert_eq!(response.status(), StatusCode::OK);
//     let response_json = response.json::<Value>().await.unwrap();
//     let parsed_output = response_json
//         .get("parsed_output")
//         .unwrap()
//         .as_object()
//         .unwrap();
//     let answer = parsed_output.get("answer").unwrap().as_str().unwrap();
//     assert_eq!(answer, "Hello");
//     // Check that created is here
//     response_json.get("created").unwrap();
//     // Check that inference_id is here
//     let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
//     let inference_id = Uuid::parse_str(inference_id).unwrap();
//     // Check that content blocks are present and correct
//     let content_blocks = response_json
//         .get("content_blocks")
//         .unwrap()
//         .as_array()
//         .unwrap();
//     assert_eq!(content_blocks.len(), 1);
//     let content_block = content_blocks.first().unwrap();
//     let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
//     assert_eq!(content_block_type, "text");
//     let content = content_block.get("text").unwrap().as_str().unwrap();
//     assert_eq!(content, "{\"answer\":\"Hello\"}");
//     // Check that type is "chat"
//     let r#type = response_json.get("type").unwrap().as_str().unwrap();
//     assert_eq!(r#type, "chat");

//     // Check that usage is correct
//     let usage = response_json.get("usage").unwrap();
//     let usage = usage.as_object().unwrap();
//     let prompt_tokens = usage.get("prompt_tokens").unwrap().as_u64().unwrap();
//     let completion_tokens = usage.get("completion_tokens").unwrap().as_u64().unwrap();
//     assert_eq!(prompt_tokens, 10);
//     assert_eq!(completion_tokens, 10);
//     // Sleep for 1 second to allow time for data to be inserted into ClickHouse (trailing writes from API)
//     tokio::time::sleep(std::time::Duration::from_secs(1)).await;

//     // Check ClickHouse
//     let clickhouse = get_clickhouse().await;
//     let result = select_inference_clickhouse(&clickhouse, inference_id)
//         .await
//         .unwrap();
//     let id = result.get("id").unwrap().as_str().unwrap();
//     let id_uuid = Uuid::parse_str(id).unwrap();
//     assert_eq!(id_uuid, inference_id);
//     let input: Value =
//         serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
//     assert_eq!(input, payload["input"]);
//     // Check that correctly parsed output is present
//     let parsed_output = result.get("parsed_output").unwrap().as_str().unwrap();
//     assert_eq!(parsed_output, "{\"answer\":\"Hello\"}");
//     // Check content blocks
//     let content_blocks = result.get("content_blocks").unwrap().as_str().unwrap();
//     let content_blocks: Vec<Value> = serde_json::from_str(content_blocks).unwrap();
//     assert_eq!(content_blocks.len(), 1);
//     let content_block = content_blocks.first().unwrap();
//     let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
//     assert_eq!(content_block_type, "text");
//     let content = content_block.get("text").unwrap().as_str().unwrap();
//     assert_eq!(content, "{\"answer\":\"Hello\"}");
//     let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
//     let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
//     assert_eq!(retrieved_episode_id, episode_id);
//     // Check the variant name
//     let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
//     assert_eq!(variant_name, "test");

//     // Check the ModelInference Table
//     let result = select_model_inferences_clickhouse(&clickhouse, inference_id)
//         .await
//         .unwrap();
//     let id = result.get("id").unwrap().as_str().unwrap();
//     let _ = Uuid::parse_str(id).unwrap();
//     let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
//     let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
//     assert_eq!(inference_id_result, inference_id);

//     let input_result = result.get("input").unwrap().as_str().unwrap();
//     let input_result: Value = serde_json::from_str(input_result).unwrap();
//     assert_eq!(input_result, payload["input"]);
//     let output = result.get("output").unwrap().as_str().unwrap();
//     let content_blocks: Vec<Value> = serde_json::from_str(output).unwrap();
//     assert_eq!(content_blocks.len(), 1);
//     let content_block = content_blocks.first().unwrap();
//     let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
//     assert_eq!(content_block_type, "text");
//     let content = content_block.get("text").unwrap().as_str().unwrap();
//     assert_eq!(content, "{\"answer\":\"Hello\"}");

//     let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
//     assert_eq!(input_tokens, 10);
//     let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
//     assert_eq!(output_tokens, 10);
//     let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
//     assert!(response_time_ms > 0);
//     assert!(result.get("ttft_ms").unwrap().is_null());
//     assert_eq!(
//         result.get("raw_response").unwrap().as_str().unwrap(),
//         DUMMY_JSON_RESPONSE_RAW
//     );
// }

/// The variant_failover function has two variants: good and error, each with weight 0.5
/// We want to make sure that this does not fail despite the error variant failing every time
/// We do this by making several requests and checking that the response is 200 in each, then checking that
/// the response is correct for the last one.
#[tokio::test]
async fn e2e_test_variant_failover() {
    let mut last_response = None;
    let mut last_episode_id = None;
    for _ in 0..50 {
        let episode_id = Uuid::now_v7();

        let payload = json!({
            "function_name": "variant_failover",
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
        last_response = Some(response);
        last_episode_id = Some(episode_id);
    }
    let response = last_response.unwrap();
    let episode_id = last_episode_id.unwrap();
    let response_json = response.json::<Value>().await.unwrap();
    let content_blocks = response_json.get("output").unwrap().as_array().unwrap();
    assert!(content_blocks.len() == 1);
    let content_block = content_blocks.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let content = content_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(content, DUMMY_INFER_RESPONSE_CONTENT);
    // Check that created is here
    response_json.get("created").unwrap();
    // Check that inference_id is here
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();
    // Check that type is "chat"
    let r#type = response_json.get("type").unwrap().as_str().unwrap();
    assert_eq!(r#type, "chat");

    // Check that usage is correct
    let usage = response_json.get("usage").unwrap();
    let usage = usage.as_object().unwrap();
    let prompt_tokens = usage.get("prompt_tokens").unwrap().as_u64().unwrap();
    let completion_tokens = usage.get("completion_tokens").unwrap().as_u64().unwrap();
    assert_eq!(prompt_tokens, 10);
    assert_eq!(completion_tokens, 10);
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
    let content_blocks = result.get("output").unwrap().as_str().unwrap();
    // Check that content_blocks is a list of blocks length 1
    let content_blocks: Vec<Value> = serde_json::from_str(content_blocks).unwrap();
    assert_eq!(content_blocks.len(), 1);
    let content_block = content_blocks.first().unwrap();
    // Check the type and content in the block
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
    assert_eq!(variant_name, "good");

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
    let content = content_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(content, DUMMY_INFER_RESPONSE_CONTENT);

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
}

/// This test checks that streaming inference works as expected.
#[tokio::test]
async fn e2e_test_streaming() {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
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
            let prompt_tokens = usage.get("prompt_tokens").unwrap().as_u64().unwrap();
            let completion_tokens = usage.get("completion_tokens").unwrap().as_u64().unwrap();
            assert_eq!(prompt_tokens, 10);
            assert_eq!(completion_tokens, 16);
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
    let result = select_inference_clickhouse(&clickhouse, inference_id)
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
    assert_eq!(variant_name, "test");

    // Check the ModelInference Table
    let result = select_model_inferences_clickhouse(&clickhouse, inference_id)
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
}

/// This test checks that streaming inference works as expected when dryrun is true.
#[tokio::test]
async fn e2e_test_streaming_dryrun() {
    let payload = json!({
        "function_name": "basic_test",
        "episode_id": Uuid::now_v7(),
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
        "dryrun": true,
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
            let prompt_tokens = usage.get("prompt_tokens").unwrap().as_u64().unwrap();
            let completion_tokens = usage.get("completion_tokens").unwrap().as_u64().unwrap();
            assert_eq!(prompt_tokens, 10);
            assert_eq!(completion_tokens, 16);
            inference_id = Some(
                Uuid::parse_str(chunk_json.get("inference_id").unwrap().as_str().unwrap()).unwrap(),
            );
        }
    }

    // Check ClickHouse
    let clickhouse = get_clickhouse().await;
    let result = select_inference_clickhouse(&clickhouse, inference_id.unwrap()).await;
    assert!(result.is_none()); // No inference should be written to ClickHouse when dryrun is true
}

#[tokio::test]
async fn e2e_test_tool_call_streaming() {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "weather_helper",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "AskJeeves"},
            "messages": [
                {
                    "role": "user",
                    "content": "Hi I'm visiting Brooklyn from Brazil. What's the weather?"
                }
            ]},
        "stream": true,
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
    let mut id: Option<String> = None;
    let mut name: Option<String> = None;

    for (i, chunk) in chunks.iter().enumerate() {
        let chunk_json: Value = serde_json::from_str(chunk).unwrap();
        if i < DUMMY_STREAMING_TOOL_RESPONSE.len() {
            let content = chunk_json.get("content").unwrap().as_array().unwrap();
            assert_eq!(content.len(), 1);
            let content_block = content.first().unwrap();
            let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
            assert_eq!(content_block_type, "tool_call");
            let new_arguments = content_block.get("arguments").unwrap().as_str().unwrap();
            assert_eq!(new_arguments, DUMMY_STREAMING_TOOL_RESPONSE[i]);
            let new_id = content_block.get("id").unwrap().as_str().unwrap();
            if i == 0 {
                id = Some(new_id.to_string());
            } else {
                assert_eq!(id, Some(new_id.to_string()));
            }
            let new_name = content_block.get("name").unwrap().as_str().unwrap();
            if i == 0 {
                name = Some(new_name.to_string());
            } else {
                assert_eq!(name, Some(new_name.to_string()));
            }
        } else {
            assert!(chunk_json
                .get("content")
                .unwrap()
                .as_array()
                .unwrap()
                .is_empty());
            let usage = chunk_json.get("usage").unwrap().as_object().unwrap();
            let prompt_tokens = usage.get("prompt_tokens").unwrap().as_u64().unwrap();
            let completion_tokens = usage.get("completion_tokens").unwrap().as_u64().unwrap();
            assert_eq!(prompt_tokens, 10);
            assert_eq!(completion_tokens, 5);
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
    let result = select_inference_clickhouse(&clickhouse, inference_id)
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
                    "content": [{"type": "text", "value": "Hi I'm visiting Brooklyn from Brazil. What's the weather?"}]
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
    assert_eq!(content_block_type, "tool_call");
    // Check that the tool call is correctly returned
    let arguments = content_block.get("arguments").unwrap().as_str().unwrap();
    let arguments: Value = serde_json::from_str(arguments).unwrap();
    assert_eq!(arguments, *DUMMY_TOOL_RESPONSE);
    let id = content_block.get("id").unwrap().as_str().unwrap();
    assert_eq!(id, "0");
    let name = content_block.get("name").unwrap().as_str().unwrap();
    assert_eq!(name, "get_weather");
    let parsed_name = content_block.get("parsed_name").unwrap().as_str().unwrap();
    assert_eq!(parsed_name, "get_weather");
    let parsed_arguments = content_block.get("parsed_arguments").unwrap();
    assert_eq!(parsed_arguments, &*DUMMY_TOOL_RESPONSE,);
    // Check that episode_id is here and correct
    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);
    // Check the variant name
    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, "variant");
    // Check the dynamic_tool_config
    let dynamic_tool_config = result.get("dynamic_tool_config").unwrap().as_str().unwrap();
    let dynamic_tool_config: Value = serde_json::from_str(dynamic_tool_config).unwrap();
    assert!(dynamic_tool_config.get("allowed_tools").unwrap().is_null());
    assert!(dynamic_tool_config
        .get("additional_tools")
        .unwrap()
        .is_null());
    assert!(dynamic_tool_config.get("tool_choice").unwrap().is_null());
    assert!(dynamic_tool_config
        .get("parallel_tool_calls")
        .unwrap()
        .is_null());
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
    assert_eq!(content_block_type, "tool_call");
    // Check that the tool call is correctly stored (no parsing here)
    let arguments = content_block.get("arguments").unwrap().as_str().unwrap();
    let arguments: Value = serde_json::from_str(arguments).unwrap();
    assert_eq!(arguments, *DUMMY_TOOL_RESPONSE);
    let id = content_block.get("id").unwrap().as_str().unwrap();
    assert_eq!(id, "0");
    let name = content_block.get("name").unwrap().as_str().unwrap();
    assert_eq!(name, "get_weather");

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    assert_eq!(input_tokens, 10);
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
    assert_eq!(output_tokens, 5);
    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);
    assert!(result.get("ttft_ms").unwrap().as_u64().unwrap() > 50);
    result.get("raw_response").unwrap().as_str().unwrap();
}
