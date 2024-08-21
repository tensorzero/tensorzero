use futures::StreamExt;
use reqwest::{Client, StatusCode};
use reqwest_eventsource::{Event, RequestBuilderExt};
use serde_json::{json, Value};
use uuid::Uuid;

use crate::common::{
    get_clickhouse, get_gateway_endpoint, select_inference_clickhouse,
    select_model_inferences_clickhouse,
};

#[derive(Clone, Debug)]
pub struct E2ETestProvider {
    pub variant_name: String,
}

/// Enforce that every provider implements a common set of tests.
///
/// To achieve that, each provider should call the `generate_provider_tests!` macro along with a
/// function that returns a `TestProviders` struct.
///
/// If some test doesn't apply to a particular provider (e.g. provider doesn't support tool use),
/// then the provider should return an empty vector for the corresponding test.
pub struct E2ETestProviders {
    pub simple_inference: Vec<E2ETestProvider>,
    pub streaming_inference: Vec<E2ETestProvider>,
    pub tool_use_inference: Vec<E2ETestProvider>,
    // pub tool_use_streaming_inference: Vec<E2ETestProvider>,
    // pub tool_result_inference: Vec<E2ETestProvider>,
    // pub tool_result_streaming_inference: Vec<E2ETestProvider>,
    // pub json_mode_inference: Vec<E2ETestProvider>,
    // pub json_mode_streaming_inference: Vec<E2ETestProvider>,
}

impl E2ETestProviders {
    pub fn with_provider(provider: E2ETestProvider) -> Self {
        let providers = vec![provider];

        Self {
            simple_inference: providers.clone(),
            streaming_inference: providers.clone(),
            tool_use_inference: providers.clone(),
            // tool_use_streaming_inference: providers.clone(),
            // tool_result_inference: providers.clone(),
            // tool_result_streaming_inference: providers.clone(),
            // json_mode_inference: providers.clone(),
            // json_mode_streaming_inference: providers,
        }
    }

    pub fn with_providers(providers: Vec<E2ETestProvider>) -> Self {
        Self {
            simple_inference: providers.clone(),
            streaming_inference: providers.clone(),
            tool_use_inference: providers.clone(),
            // tool_use_streaming_inference: providers.clone(),
            // tool_result_inference: static_providers.clone(),
            // tool_result_streaming_inference: providers.clone(),
            // json_mode_inference: providers.clone(),
            // json_mode_streaming_inference: providers,
        }
    }
}

#[macro_export]
macro_rules! generate_provider_tests {
    ($func:ident) => {
        // use $crate::providers::common::test_json_mode_inference_request_with_provider;
        // use $crate::providers::common::test_json_mode_streaming_inference_request_with_provider;
        use $crate::providers::common::test_simple_inference_request_with_provider;
        use $crate::providers::common::test_streaming_inference_request_with_provider;
        // use $crate::providers::common::test_tool_result_inference_request_with_provider;
        // use $crate::providers::common::test_tool_result_streaming_inference_request_with_provider;
        use $crate::providers::common::test_tool_use_inference_request_with_provider;
        // use $crate::providers::common::test_tool_use_streaming_inference_request_with_provider;

        #[tokio::test]
        async fn test_simple_inference_request() {
            let providers = $func().await.simple_inference;
            for provider in providers {
                test_simple_inference_request_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_streaming_inference_request() {
            let providers = $func().await.streaming_inference;
            for provider in providers {
                test_streaming_inference_request_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_tool_use_inference_request() {
            let providers = $func().await.tool_use_inference;
            for provider in providers {
                test_tool_use_inference_request_with_provider(provider).await;
            }
        }

        // #[tokio::test]
        // async fn test_tool_use_streaming_inference_request() {
        //     let providers = $func().await.tool_use_streaming_inference;
        //     for provider in providers {
        //         test_tool_use_streaming_inference_request_with_provider(provider).await;
        //     }
        // }

        // #[tokio::test]
        // async fn test_json_mode_inference_request() {
        //     let providers = $func().await.json_mode_inference;
        //     for provider in providers {
        //         test_json_mode_inference_request_with_provider(provider).await;
        //     }
        // }

        // #[tokio::test]
        // async fn test_json_mode_streaming_inference_request() {
        //     let providers = $func().await.json_mode_streaming_inference;
        //     for provider in providers {
        //         test_json_mode_streaming_inference_request_with_provider(provider).await;
        //     }
        // }

        // #[tokio::test]
        // async fn test_tool_result_inference_request() {
        //     let providers = $func().await.tool_result_inference;
        //     for provider in providers {
        //         test_tool_result_inference_request_with_provider(provider).await;
        //     }
        // }

        // #[tokio::test]
        // async fn test_tool_result_streaming_inference_request() {
        //     let providers = $func().await.tool_result_streaming_inference;
        //     for provider in providers {
        //         test_tool_result_streaming_inference_request_with_provider(provider).await;
        //     }
        // }
    };
}

pub async fn test_simple_inference_request_with_provider(provider: E2ETestProvider) {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "variant_name": provider.variant_name,
        "episode_id": episode_id,
        "input":
            {
               "system": {"assistant_name": "Professor Megumin"},
               "messages": [
                {
                    "role": "user",
                    "content": "What is the capital city of Japan?"
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

    // Check that the response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    let content_blocks = response_json.get("output").unwrap().as_array().unwrap();
    assert!(content_blocks.len() == 1);
    let content_block = content_blocks.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let content = content_block.get("text").unwrap().as_str().unwrap();
    assert!(content.contains("Tokyo"));

    // Check that inference_id is here
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    // Check that the episode_id is here
    let episode_id_response = response_json.get("episode_id").unwrap().as_str().unwrap();
    let episode_id_response = Uuid::parse_str(episode_id_response).unwrap();
    assert_eq!(episode_id_response, episode_id);

    // Sleep for 1 second to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ClickHouse
    let clickhouse = get_clickhouse().await;

    // First, check Inference table...
    let result = select_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    let id = result.get("id").unwrap().as_str().unwrap();
    let id = Uuid::parse_str(id).unwrap();
    assert_eq!(id, inference_id);
    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input = json!({
        "system": {"assistant_name": "Professor Megumin"},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "value": "What is the capital city of Japan?"}]
            }
        ]
    });
    assert_eq!(input, correct_input);

    // Check that content_blocks is a list of blocks length 1
    let content_blocks = result.get("output").unwrap().as_str().unwrap();
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
    assert_eq!(variant_name, provider.variant_name);

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

pub async fn test_streaming_inference_request_with_provider(provider: E2ETestProvider) {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "variant_name": provider.variant_name,
        "episode_id": episode_id,
        "input":
            {
               "system": {"assistant_name": "Professor Megumin"},
               "messages": [
                {
                    "role": "user",
                    "content": "What is the capital city of Japan?"
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
    let mut episode_id_response = None;
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
        if episode_id_response.is_none() {
            episode_id_response = Some(
                Uuid::parse_str(chunk_json.get("episode_id").unwrap().as_str().unwrap()).unwrap(),
            );
        }
    }
    let inference_id = inference_id.unwrap();
    assert!(
        full_content.contains("Tokyo"),
        "full_content: {}",
        full_content
    );
    assert_eq!(episode_id_response.unwrap(), episode_id);

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
        "system": {"assistant_name": "Professor Megumin"},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "value": "What is the capital city of Japan?"}]
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
    assert_eq!(variant_name, provider.variant_name);
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

    // NB: Azure doesn't support input/output tokens during streaming
    if provider.variant_name != "azure" {
        let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
        assert!(input_tokens > 5);
        let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
        assert!(output_tokens > 5);
    }

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

pub async fn test_tool_use_inference_request_with_provider(provider: E2ETestProvider) {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "weather_helper",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "Professor Aqua"},
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather like in New York City?"
                }
            ]},
        "stream": false,
        "variant_name": provider.variant_name,

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
    let _: Value = serde_json::from_str(arguments).unwrap();
    content_block.get("id").unwrap().as_str().unwrap();
    let name = content_block.get("name").unwrap().as_str().unwrap();
    assert_eq!(name, "get_weather");
    let parsed_name = content_block.get("parsed_name").unwrap().as_str().unwrap();
    assert_eq!(parsed_name, "get_weather");
    // This could fail if the LLM fails to return correct arguments (similarly in the Inference table)
    content_block.get("parsed_arguments").unwrap();

    // Check that type is "chat"
    // Check that usage is correct
    let usage = response_json.get("usage").unwrap();
    let usage = usage.as_object().unwrap();
    let prompt_tokens = usage.get("prompt_tokens").unwrap().as_u64().unwrap();
    let completion_tokens = usage.get("completion_tokens").unwrap().as_u64().unwrap();
    assert!(prompt_tokens > 10);
    assert!(completion_tokens > 0);
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
                "assistant_name": "Professor Aqua"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "value": "What is the weather like in New York City?"}]
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
    let _: Value = serde_json::from_str(arguments).unwrap();
    content_block.get("id").unwrap().as_str().unwrap();
    let name = content_block.get("name").unwrap().as_str().unwrap();
    assert_eq!(name, "get_weather");
    let parsed_name = content_block.get("parsed_name").unwrap().as_str().unwrap();
    assert_eq!(parsed_name, "get_weather");
    content_block.get("parsed_arguments").unwrap();
    // Check that episode_id is here and correct
    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);
    // Check the variant name
    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

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
    let _: Value = serde_json::from_str(arguments).unwrap();
    content_block.get("id").unwrap().as_str().unwrap();
    let name = content_block.get("name").unwrap().as_str().unwrap();
    assert_eq!(name, "get_weather");

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 10);
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(output_tokens > 0);
    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);
    assert!(result.get("ttft_ms").unwrap().is_null());
    result.get("raw_response").unwrap().as_str().unwrap();
}
