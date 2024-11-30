#![allow(clippy::print_stdout)]

use futures::StreamExt;
use gateway::{
    inference::types::{ContentBlock, RequestMessage, Role},
    tool::{ToolCall, ToolResult},
};
use reqwest::{Client, StatusCode};
use reqwest_eventsource::{Event, RequestBuilderExt};
use serde_json::{json, Value};
use uuid::Uuid;

use crate::common::{
    get_clickhouse, get_gateway_endpoint, select_chat_inference_clickhouse,
    select_inference_tags_clickhouse, select_json_inference_clickhouse,
    select_model_inference_clickhouse,
};

#[derive(Clone, Debug)]
pub struct E2ETestProvider {
    pub variant_name: String,
    pub model_name: String,
    pub model_provider_name: String,
}

/// Enforce that every provider implements a common set of tests.
///
/// To achieve that, each provider should call the `generate_provider_tests!` macro along with a
/// function that returns a `E2ETestProviders` struct.
///
/// If some test doesn't apply to a particular provider (e.g. provider doesn't support tool use),
/// then the provider should return an empty vector for the corresponding test.
pub struct E2ETestProviders {
    pub simple_inference: Vec<E2ETestProvider>,
    pub inference_params_inference: Vec<E2ETestProvider>,
    pub tool_use_inference: Vec<E2ETestProvider>,
    pub tool_multi_turn_inference: Vec<E2ETestProvider>,
    pub dynamic_tool_use_inference: Vec<E2ETestProvider>,
    pub parallel_tool_use_inference: Vec<E2ETestProvider>,
    pub json_mode_inference: Vec<E2ETestProvider>,
    pub supports_batch_inference: bool,
}

#[macro_export]
macro_rules! generate_provider_tests {
    ($func:ident) => {
        use $crate::providers::common::test_dynamic_tool_use_inference_request_with_provider;
        use $crate::providers::common::test_dynamic_tool_use_streaming_inference_request_with_provider;
        use $crate::providers::common::test_inference_params_inference_request_with_provider;
        use $crate::providers::common::test_inference_params_streaming_inference_request_with_provider;
        use $crate::providers::common::test_json_mode_inference_request_with_provider;
        use $crate::providers::common::test_json_mode_streaming_inference_request_with_provider;
        use $crate::providers::common::test_dynamic_json_mode_inference_request_with_provider;
        use $crate::providers::common::test_parallel_tool_use_inference_request_with_provider;
        use $crate::providers::common::test_parallel_tool_use_streaming_inference_request_with_provider;
        use $crate::providers::common::test_simple_inference_request_with_provider;
        use $crate::providers::common::test_simple_streaming_inference_request_with_provider;
        use $crate::providers::common::test_tool_multi_turn_inference_request_with_provider;
        use $crate::providers::common::test_tool_multi_turn_streaming_inference_request_with_provider;
        use $crate::providers::common::test_tool_use_allowed_tools_inference_request_with_provider;
        use $crate::providers::common::test_tool_use_allowed_tools_streaming_inference_request_with_provider;
        use $crate::providers::common::test_tool_use_tool_choice_auto_unused_inference_request_with_provider;
        use $crate::providers::common::test_tool_use_tool_choice_auto_unused_streaming_inference_request_with_provider;
        use $crate::providers::common::test_tool_use_tool_choice_auto_used_inference_request_with_provider;
        use $crate::providers::common::test_tool_use_tool_choice_auto_used_streaming_inference_request_with_provider;
        use $crate::providers::common::test_tool_use_tool_choice_none_inference_request_with_provider;
        use $crate::providers::common::test_tool_use_tool_choice_none_streaming_inference_request_with_provider;
        use $crate::providers::common::test_tool_use_tool_choice_required_inference_request_with_provider;
        use $crate::providers::common::test_tool_use_tool_choice_required_streaming_inference_request_with_provider;
        use $crate::providers::common::test_tool_use_tool_choice_specific_inference_request_with_provider;
        use $crate::providers::common::test_tool_use_tool_choice_specific_streaming_inference_request_with_provider;

        #[tokio::test]
        async fn test_simple_inference_request() {
            let providers = $func().await.simple_inference;
            for provider in providers {
                test_simple_inference_request_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_simple_streaming_inference_request() {
            let providers = $func().await.simple_inference;
            for provider in providers {
                test_simple_streaming_inference_request_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_inference_params_inference_request() {
            let providers = $func().await.inference_params_inference;
            for provider in providers {
                test_inference_params_inference_request_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_inference_params_streaming_inference_request() {
            let providers = $func().await.inference_params_inference;
            for provider in providers {
                test_inference_params_streaming_inference_request_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_tool_use_tool_choice_auto_used_inference_request() {
            let providers = $func().await.tool_use_inference;
            for provider in providers {
                test_tool_use_tool_choice_auto_used_inference_request_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_tool_use_tool_choice_auto_used_streaming_inference_request() {
            let providers = $func().await.tool_use_inference;
            for provider in providers {
                test_tool_use_tool_choice_auto_used_streaming_inference_request_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_tool_use_tool_choice_auto_unused_inference_request() {
            let providers = $func().await.tool_use_inference;
            for provider in providers {
                test_tool_use_tool_choice_auto_unused_inference_request_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_tool_use_tool_choice_auto_unused_streaming_inference_request() {
            let providers = $func().await.tool_use_inference;
            for provider in providers {
                test_tool_use_tool_choice_auto_unused_streaming_inference_request_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_tool_use_tool_choice_required_inference_request() {
            let providers = $func().await.tool_use_inference;
            for provider in providers {
                test_tool_use_tool_choice_required_inference_request_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_tool_use_tool_choice_required_streaming_inference_request() {
            let providers = $func().await.tool_use_inference;
            for provider in providers {
                test_tool_use_tool_choice_required_streaming_inference_request_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_tool_use_tool_choice_none_inference_request() {
            let providers = $func().await.tool_use_inference;
            for provider in providers {
                test_tool_use_tool_choice_none_inference_request_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_tool_use_tool_choice_none_streaming_inference_request() {
            let providers = $func().await.tool_use_inference;
            for provider in providers {
                test_tool_use_tool_choice_none_streaming_inference_request_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_tool_use_tool_choice_specific_inference_request() {
            let providers = $func().await.tool_use_inference;
            for provider in providers {
                test_tool_use_tool_choice_specific_inference_request_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_tool_use_tool_choice_specific_streaming_inference_request() {
            let providers = $func().await.tool_use_inference;
            for provider in providers {
                test_tool_use_tool_choice_specific_streaming_inference_request_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_tool_use_allowed_tools_inference_request() {
            let providers = $func().await.tool_use_inference;
            for provider in providers {
                test_tool_use_allowed_tools_inference_request_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_tool_use_allowed_tools_streaming_inference_request() {
            let providers = $func().await.tool_use_inference;
            for provider in providers {
                test_tool_use_allowed_tools_streaming_inference_request_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_tool_multi_turn_inference_request() {
            let providers = $func().await.tool_multi_turn_inference;
            for provider in providers {
                test_tool_multi_turn_inference_request_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_tool_multi_turn_streaming_inference_request() {
            let providers = $func().await.tool_multi_turn_inference;
            for provider in providers {
                test_tool_multi_turn_streaming_inference_request_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_dynamic_tool_use_inference_request() {
            let providers = $func().await.dynamic_tool_use_inference;
            for provider in providers {
                test_dynamic_tool_use_inference_request_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_dynamic_tool_use_streaming_inference_request() {
            let providers = $func().await.dynamic_tool_use_inference;
            for provider in providers {
                test_dynamic_tool_use_streaming_inference_request_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_parallel_tool_use_inference_request() {
            let providers = $func().await.parallel_tool_use_inference;
            for provider in providers {
                test_parallel_tool_use_inference_request_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_parallel_tool_use_streaming_inference_request() {
            let providers = $func().await.parallel_tool_use_inference;
            for provider in providers {
                test_parallel_tool_use_streaming_inference_request_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_json_mode_inference_request() {
            let providers = $func().await.json_mode_inference;
            for provider in providers {
                test_json_mode_inference_request_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_dynamic_json_mode_inference_request() {
            let providers = $func().await.json_mode_inference;
            for provider in providers {
                test_dynamic_json_mode_inference_request_with_provider(provider).await;
            }
        }

        #[tokio::test]
        async fn test_json_mode_streaming_inference_request() {
            let providers = $func().await.json_mode_inference;
            for provider in providers {
                test_json_mode_streaming_inference_request_with_provider(provider).await;
            }
        }
    };
}

pub async fn test_simple_inference_request_with_provider(provider: E2ETestProvider) {
    let episode_id = Uuid::now_v7();
    let tag_value = Uuid::now_v7().to_string();

    let payload = json!({
        "function_name": "basic_test",
        "variant_name": provider.variant_name,
        "episode_id": episode_id,
        "input":
            {
               "system": {"assistant_name": "Dr. Mehta"},
               "messages": [
                {
                    "role": "user",
                    "content": "What is the capital city of Japan?"
                }
            ]},
        "stream": false,
        "tags": {"key": tag_value},
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    println!("API response: {response_json:#?}");

    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    let episode_id_response = response_json.get("episode_id").unwrap().as_str().unwrap();
    let episode_id_response = Uuid::parse_str(episode_id_response).unwrap();
    assert_eq!(episode_id_response, episode_id);

    let variant_name = response_json.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let content = response_json.get("content").unwrap().as_array().unwrap();
    assert_eq!(content.len(), 1);
    let content_block = content.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let content = content_block.get("text").unwrap().as_str().unwrap();
    assert!(content.to_lowercase().contains("tokyo"));

    let usage = response_json.get("usage").unwrap();
    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(output_tokens > 0);

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check if ClickHouse is ok - ChatInference Table
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ChatInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id = Uuid::parse_str(id).unwrap();
    assert_eq!(id, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, payload["function_name"]);

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input = json!({
        "system": {"assistant_name": "Dr. Mehta"},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "value": "What is the capital city of Japan?"}]
            }
        ]
    });
    assert_eq!(input, correct_input);

    let content_blocks = result.get("output").unwrap().as_str().unwrap();
    let content_blocks: Vec<Value> = serde_json::from_str(content_blocks).unwrap();
    assert_eq!(content_blocks.len(), 1);
    let content_block = content_blocks.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let clickhouse_content = content_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(clickhouse_content, content);

    let tags = result.get("tags").unwrap().as_object().unwrap();
    assert_eq!(tags.len(), 1);
    assert_eq!(tags.get("key").unwrap().as_str().unwrap(), tag_value);

    let tool_params = result.get("tool_params").unwrap().as_str().unwrap();
    assert!(tool_params.is_empty());

    let inference_params = result.get("inference_params").unwrap().as_str().unwrap();
    let inference_params: Value = serde_json::from_str(inference_params).unwrap();
    let inference_params = inference_params.get("chat_completion").unwrap();
    assert!(inference_params.get("temperature").is_none());
    assert!(inference_params.get("seed").is_none());
    assert_eq!(
        inference_params
            .get("max_tokens")
            .unwrap()
            .as_u64()
            .unwrap(),
        100
    );

    let processing_time_ms = result.get("processing_time_ms").unwrap().as_u64().unwrap();
    assert!(processing_time_ms > 0);

    // Check the ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let model_inference_id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(model_inference_id).is_ok());

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.to_lowercase().contains("japan"));
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );

    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    assert!(raw_response.to_lowercase().contains("tokyo"));
    assert!(serde_json::from_str::<Value>(raw_response).is_ok());

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(output_tokens > 0);
    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);
    assert!(result.get("ttft_ms").unwrap().is_null());
    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta"
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![RequestMessage {
        role: Role::User,
        content: vec!["What is the capital city of Japan?".to_string().into()],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    assert_eq!(output.len(), 1);

    // Check the InferenceTag Table
    let result = select_inference_tags_clickhouse(&clickhouse, "basic_test", "key", &tag_value)
        .await
        .unwrap();
    let id = result.get("inference_id").unwrap().as_str().unwrap();
    let id = Uuid::parse_str(id).unwrap();
    assert_eq!(id, inference_id);
}

pub async fn test_simple_streaming_inference_request_with_provider(provider: E2ETestProvider) {
    let episode_id = Uuid::now_v7();
    let tag_value = Uuid::now_v7().to_string();

    let payload = json!({
        "function_name": "basic_test",
        "variant_name": provider.variant_name,
        "episode_id": episode_id,
        "input":
            {
               "system": {"assistant_name": "Dr. Mehta"},
               "messages": [
                {
                    "role": "user",
                    "content": "What is the capital city of Japan?"
                }
            ]},
        "stream": true,
        "tags": {"key": tag_value},
    });

    let mut event_source = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .unwrap();

    let mut chunks = vec![];
    let mut found_done_chunk = false;
    while let Some(event) = event_source.next().await {
        let event = event.unwrap();
        match event {
            Event::Open => continue,
            Event::Message(message) => {
                if message.data == "[DONE]" {
                    found_done_chunk = true;
                    break;
                }
                chunks.push(message.data);
            }
        }
    }
    assert!(found_done_chunk);

    let mut inference_id: Option<Uuid> = None;
    let mut full_content = String::new();
    let mut input_tokens = 0;
    let mut output_tokens = 0;
    for chunk in chunks.clone() {
        let chunk_json: Value = serde_json::from_str(&chunk).unwrap();

        println!("API response chunk: {chunk_json:#?}");

        let chunk_inference_id = chunk_json.get("inference_id").unwrap().as_str().unwrap();
        let chunk_inference_id = Uuid::parse_str(chunk_inference_id).unwrap();
        match inference_id {
            Some(inference_id) => {
                assert_eq!(inference_id, chunk_inference_id);
            }
            None => {
                inference_id = Some(chunk_inference_id);
            }
        }

        let chunk_episode_id = chunk_json.get("episode_id").unwrap().as_str().unwrap();
        let chunk_episode_id = Uuid::parse_str(chunk_episode_id).unwrap();
        assert_eq!(chunk_episode_id, episode_id);

        let content_blocks = chunk_json.get("content").unwrap().as_array().unwrap();
        if !content_blocks.is_empty() {
            let content_block = content_blocks.first().unwrap();
            let content = content_block.get("text").unwrap().as_str().unwrap();
            full_content.push_str(content);
        }

        if let Some(usage) = chunk_json.get("usage") {
            input_tokens += usage.get("input_tokens").unwrap().as_u64().unwrap();
            output_tokens += usage.get("output_tokens").unwrap().as_u64().unwrap();
        }
    }

    let inference_id = inference_id.unwrap();
    assert!(full_content.to_lowercase().contains("tokyo"));

    // NB: Azure doesn't support input/output tokens during streaming
    if provider.variant_name.contains("azure") {
        assert_eq!(input_tokens, 0);
        assert_eq!(output_tokens, 0);
    } else {
        assert!(input_tokens > 0);
        assert!(output_tokens > 0);
    }

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ClickHouse - ChatInference Table
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ChatInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, payload["function_name"]);

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let episode_id_result = result.get("episode_id").unwrap().as_str().unwrap();
    let episode_id_result = Uuid::parse_str(episode_id_result).unwrap();
    assert_eq!(episode_id_result, episode_id);

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input = json!({
        "system": {"assistant_name": "Dr. Mehta"},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "value": "What is the capital city of Japan?"}]
            }
        ]
    });
    assert_eq!(input, correct_input);

    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<Value> = serde_json::from_str(output).unwrap();
    assert_eq!(output.len(), 1);
    let content_block = output.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let clickhouse_content = content_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(clickhouse_content, full_content);

    let tool_params = result.get("tool_params").unwrap().as_str().unwrap();
    assert!(tool_params.is_empty());

    let inference_params = result.get("inference_params").unwrap().as_str().unwrap();
    let inference_params: Value = serde_json::from_str(inference_params).unwrap();
    let inference_params = inference_params.get("chat_completion").unwrap();
    assert!(inference_params.get("temperature").is_none());
    assert!(inference_params.get("seed").is_none());
    assert_eq!(
        inference_params
            .get("max_tokens")
            .unwrap()
            .as_u64()
            .unwrap(),
        100
    );

    let processing_time_ms = result.get("processing_time_ms").unwrap().as_u64().unwrap();
    assert!(processing_time_ms > 0);

    let tags = result.get("tags").unwrap().as_object().unwrap();
    assert_eq!(tags.len(), 1);
    assert_eq!(tags.get("key").unwrap().as_str().unwrap(), tag_value);

    // Check ClickHouse - ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let model_inference_id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(model_inference_id).is_ok());

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.to_lowercase().contains("japan"));
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );

    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();

    // Check if raw_response is valid JSONL
    for line in raw_response.lines() {
        assert!(serde_json::from_str::<Value>(line).is_ok());
    }

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();

    // NB: Azure doesn't support input/output tokens during streaming
    if provider.variant_name.contains("azure") {
        assert_eq!(input_tokens, 0);
        assert_eq!(output_tokens, 0);
    } else {
        assert!(input_tokens > 0);
        assert!(output_tokens > 0);
    }

    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);

    let ttft_ms = result.get("ttft_ms").unwrap().as_u64().unwrap();
    assert!(ttft_ms > 50);
    assert!(ttft_ms <= response_time_ms);

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta"
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![RequestMessage {
        role: Role::User,
        content: vec!["What is the capital city of Japan?".to_string().into()],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    assert_eq!(output.len(), 1);
    // Check the InferenceTag Table
    let result = select_inference_tags_clickhouse(&clickhouse, "basic_test", "key", &tag_value)
        .await
        .unwrap();
    let id = result.get("inference_id").unwrap().as_str().unwrap();
    let id = Uuid::parse_str(id).unwrap();
    assert_eq!(id, inference_id);
}

pub async fn test_inference_params_inference_request_with_provider(provider: E2ETestProvider) {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "variant_name": provider.variant_name,
        "episode_id": episode_id,
        "input":
            {
               "system": {"assistant_name": "Dr. Mehta"},
               "messages": [
                {
                    "role": "user",
                    "content": "What is the capital city of Japan?"
                }
            ]},
        "params": {
            "chat_completion": {
                "temperature": 0.9,
                "seed": 1337,
                "max_tokens": 120,
                "top_p": 0.9,
                "presence_penalty": 0.1,
                "frequency_penalty": 0.2,
            },
            "fake_variant_type": {
                "temperature": 0.8,
                "seed": 7331,
                "max_tokens": 80,
            }
        },
        "stream": false,
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    println!("API response: {response_json:#?}");

    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    let episode_id_response = response_json.get("episode_id").unwrap().as_str().unwrap();
    let episode_id_response = Uuid::parse_str(episode_id_response).unwrap();
    assert_eq!(episode_id_response, episode_id);

    let variant_name = response_json.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let content = response_json.get("content").unwrap().as_array().unwrap();
    assert_eq!(content.len(), 1);
    let content_block = content.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let content = content_block.get("text").unwrap().as_str().unwrap();
    assert!(content.to_lowercase().contains("tokyo"));

    let usage = response_json.get("usage").unwrap();
    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(output_tokens > 0);

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check if ClickHouse is ok - ChatInference Table
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ChatInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id = Uuid::parse_str(id).unwrap();
    assert_eq!(id, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, payload["function_name"]);

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input = json!({
        "system": {"assistant_name": "Dr. Mehta"},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "value": "What is the capital city of Japan?"}]
            }
        ]
    });
    assert_eq!(input, correct_input);

    let content_blocks = result.get("output").unwrap().as_str().unwrap();
    let content_blocks: Vec<Value> = serde_json::from_str(content_blocks).unwrap();
    assert_eq!(content_blocks.len(), 1);
    let content_block = content_blocks.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let clickhouse_content = content_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(clickhouse_content, content);

    let tool_params = result.get("tool_params").unwrap().as_str().unwrap();
    assert!(tool_params.is_empty());

    let inference_params = result.get("inference_params").unwrap().as_str().unwrap();
    let inference_params: Value = serde_json::from_str(inference_params).unwrap();
    let inference_params = inference_params.get("chat_completion").unwrap();
    let temperature = inference_params
        .get("temperature")
        .unwrap()
        .as_f64()
        .unwrap();
    assert_eq!(temperature, 0.9);
    let seed = inference_params.get("seed").unwrap().as_u64().unwrap();
    assert_eq!(seed, 1337);
    let max_tokens = inference_params
        .get("max_tokens")
        .unwrap()
        .as_u64()
        .unwrap();
    assert_eq!(max_tokens, 120);
    let top_p = inference_params.get("top_p").unwrap().as_f64().unwrap();
    assert_eq!(top_p, 0.9);
    let presence_penalty = inference_params
        .get("presence_penalty")
        .unwrap()
        .as_f64()
        .unwrap();
    assert_eq!(presence_penalty, 0.1);
    let frequency_penalty = inference_params
        .get("frequency_penalty")
        .unwrap()
        .as_f64()
        .unwrap();
    assert_eq!(frequency_penalty, 0.2);

    let processing_time_ms = result.get("processing_time_ms").unwrap().as_u64().unwrap();
    assert!(processing_time_ms > 0);

    // Check the ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let model_inference_id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(model_inference_id).is_ok());

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.to_lowercase().contains("japan"));
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );

    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    assert!(raw_response.to_lowercase().contains("tokyo"));
    assert!(serde_json::from_str::<Value>(raw_response).is_ok());

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(output_tokens > 0);
    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);
    assert!(result.get("ttft_ms").unwrap().is_null());
    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta"
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![RequestMessage {
        role: Role::User,
        content: vec!["What is the capital city of Japan?".to_string().into()],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    assert_eq!(output.len(), 1);
}

pub async fn test_inference_params_streaming_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "variant_name": provider.variant_name,
        "episode_id": episode_id,
        "input":
            {
               "system": {"assistant_name": "Dr. Mehta"},
               "messages": [
                {
                    "role": "user",
                    "content": "What is the capital city of Japan?"
                }
            ]},
        "params": {
            "chat_completion": {
                "temperature": 0.9,
                "seed": 1337,
                "max_tokens": 120,
                "top_p": 0.9,
                "presence_penalty": 0.1,
                "frequency_penalty": 0.2,
            },
            "fake_variant_type": {
                "temperature": 0.8,
                "seed": 7331,
                "max_tokens": 80,
                "top_p": 0.9,
                "presence_penalty": 0.1,
                "frequency_penalty": 0.2,
            }
        },
        "stream": true,
    });

    let mut event_source = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .unwrap();

    let mut chunks = vec![];
    let mut found_done_chunk = false;
    while let Some(event) = event_source.next().await {
        let event = event.unwrap();
        match event {
            Event::Open => continue,
            Event::Message(message) => {
                if message.data == "[DONE]" {
                    found_done_chunk = true;
                    break;
                }
                chunks.push(message.data);
            }
        }
    }
    assert!(found_done_chunk);

    let mut inference_id: Option<Uuid> = None;
    let mut full_content = String::new();
    let mut input_tokens = 0;
    let mut output_tokens = 0;
    for chunk in chunks.clone() {
        let chunk_json: Value = serde_json::from_str(&chunk).unwrap();

        println!("API response chunk: {chunk_json:#?}");

        let chunk_inference_id = chunk_json.get("inference_id").unwrap().as_str().unwrap();
        let chunk_inference_id = Uuid::parse_str(chunk_inference_id).unwrap();
        match inference_id {
            Some(inference_id) => {
                assert_eq!(inference_id, chunk_inference_id);
            }
            None => {
                inference_id = Some(chunk_inference_id);
            }
        }

        let chunk_episode_id = chunk_json.get("episode_id").unwrap().as_str().unwrap();
        let chunk_episode_id = Uuid::parse_str(chunk_episode_id).unwrap();
        assert_eq!(chunk_episode_id, episode_id);

        let content_blocks = chunk_json.get("content").unwrap().as_array().unwrap();
        if !content_blocks.is_empty() {
            let content_block = content_blocks.first().unwrap();
            let content = content_block.get("text").unwrap().as_str().unwrap();
            full_content.push_str(content);
        }

        if let Some(usage) = chunk_json.get("usage") {
            input_tokens += usage.get("input_tokens").unwrap().as_u64().unwrap();
            output_tokens += usage.get("output_tokens").unwrap().as_u64().unwrap();
        }
    }

    let inference_id = inference_id.unwrap();
    assert!(full_content.to_lowercase().contains("tokyo"));

    // NB: Azure doesn't support input/output tokens during streaming
    if provider.variant_name.contains("azure") {
        assert_eq!(input_tokens, 0);
        assert_eq!(output_tokens, 0);
    } else {
        assert!(input_tokens > 0);
        assert!(output_tokens > 0);
    }

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ClickHouse - ChatInference Table
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ChatInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, payload["function_name"]);

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let episode_id_result = result.get("episode_id").unwrap().as_str().unwrap();
    let episode_id_result = Uuid::parse_str(episode_id_result).unwrap();
    assert_eq!(episode_id_result, episode_id);

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input = json!({
        "system": {"assistant_name": "Dr. Mehta"},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "value": "What is the capital city of Japan?"}]
            }
        ]
    });
    assert_eq!(input, correct_input);

    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<Value> = serde_json::from_str(output).unwrap();
    assert_eq!(output.len(), 1);
    let content_block = output.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let clickhouse_content = content_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(clickhouse_content, full_content);

    let tool_params = result.get("tool_params").unwrap().as_str().unwrap();
    assert!(tool_params.is_empty());

    let inference_params = result.get("inference_params").unwrap().as_str().unwrap();
    let inference_params: Value = serde_json::from_str(inference_params).unwrap();
    let inference_params = inference_params.get("chat_completion").unwrap();
    let temperature = inference_params
        .get("temperature")
        .unwrap()
        .as_f64()
        .unwrap();
    assert_eq!(temperature, 0.9);
    let seed = inference_params.get("seed").unwrap().as_u64().unwrap();
    assert_eq!(seed, 1337);
    let max_tokens = inference_params
        .get("max_tokens")
        .unwrap()
        .as_u64()
        .unwrap();
    assert_eq!(max_tokens, 120);
    let top_p = inference_params.get("top_p").unwrap().as_f64().unwrap();
    assert_eq!(top_p, 0.9);
    let presence_penalty = inference_params
        .get("presence_penalty")
        .unwrap()
        .as_f64()
        .unwrap();
    assert_eq!(presence_penalty, 0.1);
    let frequency_penalty = inference_params
        .get("frequency_penalty")
        .unwrap()
        .as_f64()
        .unwrap();
    assert_eq!(frequency_penalty, 0.2);

    let processing_time_ms = result.get("processing_time_ms").unwrap().as_u64().unwrap();
    assert!(processing_time_ms > 0);

    // Check ClickHouse - ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let model_inference_id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(model_inference_id).is_ok());

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.to_lowercase().contains("japan"));
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );

    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();

    // Check if raw_response is valid JSONL
    for line in raw_response.lines() {
        assert!(serde_json::from_str::<Value>(line).is_ok());
    }

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();

    // NB: Azure doesn't support input/output tokens during streaming
    if provider.variant_name.contains("azure") {
        assert_eq!(input_tokens, 0);
        assert_eq!(output_tokens, 0);
    } else {
        assert!(input_tokens > 0);
        assert!(output_tokens > 0);
    }

    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);

    let ttft_ms = result.get("ttft_ms").unwrap().as_u64().unwrap();
    assert!(ttft_ms > 50);
    assert!(ttft_ms <= response_time_ms);

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta"
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![RequestMessage {
        role: Role::User,
        content: vec!["What is the capital city of Japan?".to_string().into()],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    assert_eq!(output.len(), 1);
}

pub async fn test_tool_use_tool_choice_auto_used_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "weather_helper",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."
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

    // Check if the API response is fine
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    println!("API response: {response_json:#?}");

    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    let episode_id_response = response_json.get("episode_id").unwrap().as_str().unwrap();
    let episode_id_response = Uuid::parse_str(episode_id_response).unwrap();
    assert_eq!(episode_id_response, episode_id);

    let variant_name = response_json.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let content = response_json.get("content").unwrap().as_array().unwrap();
    assert!(!content.is_empty()); // could be > 1 if the model returns text as well
    let content_block = content
        .iter()
        .find(|block| block["type"] == "tool_call")
        .unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "tool_call");

    assert!(content_block.get("id").unwrap().as_str().is_some());

    let raw_name = content_block.get("raw_name").unwrap().as_str().unwrap();
    assert_eq!(raw_name, "get_temperature");
    let name = content_block.get("name").unwrap().as_str().unwrap();
    assert_eq!(name, "get_temperature");

    let raw_arguments = content_block
        .get("raw_arguments")
        .unwrap()
        .as_str()
        .unwrap();
    let raw_arguments: Value = serde_json::from_str(raw_arguments).unwrap();
    let raw_arguments = raw_arguments.as_object().unwrap();
    assert!(raw_arguments.len() == 2);
    let location = raw_arguments.get("location").unwrap().as_str().unwrap();
    assert_eq!(location.to_lowercase(), "tokyo");
    let units = raw_arguments.get("units").unwrap().as_str().unwrap();
    assert!(units == "celsius");

    let arguments = content_block.get("arguments").unwrap();
    let arguments = arguments.as_object().unwrap();
    assert!(arguments.len() == 2);
    let location = arguments.get("location").unwrap().as_str().unwrap();
    assert_eq!(location.to_lowercase(), "tokyo");
    let units = arguments.get("units").unwrap().as_str().unwrap();
    assert!(units == "celsius");

    let usage = response_json.get("usage").unwrap();
    let usage = usage.as_object().unwrap();
    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    assert!(output_tokens > 0);

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check if ClickHouse is correct - ChatInference table
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ChatInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, "weather_helper");

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let episode_id_result = result.get("episode_id").unwrap().as_str().unwrap();
    let episode_id_result = Uuid::parse_str(episode_id_result).unwrap();
    assert_eq!(episode_id_result, episode_id);

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input: Value = json!(
        {
            "system": {
                "assistant_name": "Dr. Mehta"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "value": "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."}]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);

    let output_clickhouse: Vec<Value> =
        serde_json::from_str(result.get("output").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(output_clickhouse, *content);

    let tool_params: Value =
        serde_json::from_str(result.get("tool_params").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(tool_params["tool_choice"], "auto");
    assert_eq!(tool_params["parallel_tool_calls"], false);

    let tools_available = tool_params["tools_available"].as_array().unwrap();
    assert_eq!(tools_available.len(), 1);
    let tool = tools_available.first().unwrap();
    assert_eq!(tool["name"], "get_temperature");
    assert_eq!(
        tool["description"],
        "Get the current temperature in a given location"
    );
    assert_eq!(tool["strict"], false);

    let tool_parameters = tool["parameters"].as_object().unwrap();
    assert_eq!(tool_parameters["type"], "object");
    assert!(tool_parameters.get("properties").is_some());
    assert!(tool_parameters.get("required").is_some());
    assert_eq!(tool_parameters["additionalProperties"], false);

    let properties = tool_parameters["properties"].as_object().unwrap();
    assert!(properties.contains_key("location"));
    assert!(properties.contains_key("units"));

    let location = properties["location"].as_object().unwrap();
    assert_eq!(location["type"], "string");
    assert_eq!(
        location["description"],
        "The location to get the temperature for (e.g. \"New York\")"
    );

    let units = properties["units"].as_object().unwrap();
    assert_eq!(units["type"], "string");
    assert_eq!(
        units["description"],
        "The units to get the temperature in (must be \"fahrenheit\" or \"celsius\")"
    );
    let units_enum = units["enum"].as_array().unwrap();
    assert_eq!(units_enum.len(), 2);
    assert!(units_enum.contains(&json!("fahrenheit")));
    assert!(units_enum.contains(&json!("celsius")));

    let required = tool_parameters["required"].as_array().unwrap();
    assert!(required.contains(&json!("location")));

    // Check if ClickHouse is correct - ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(id).is_ok());

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.to_lowercase().contains("tokyo"));
    assert!(raw_request.to_lowercase().contains("celsius"));
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );

    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    assert!(raw_response.to_lowercase().contains("tokyo"));
    assert!(raw_response.contains("get_temperature"));

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(output_tokens > 0);
    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);
    assert!(result.get("ttft_ms").unwrap().is_null());

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta.\n\nPeople will ask you questions about the weather.\n\nIf asked about the weather, just respond with the tool call. Use the \"get_temperature\" tool.\n\nIf provided with a tool result, use it to respond to the user (e.g. \"The weather in New York is 55 degrees Fahrenheit.\")."
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![RequestMessage {
        role: Role::User,
        content: vec![
            "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                .to_string()
                .into(),
        ],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    let tool_call_blocks: Vec<_> = output
        .iter()
        .filter(|block| matches!(block, ContentBlock::ToolCall(_)))
        .collect();

    // Assert exactly one tool call
    assert_eq!(tool_call_blocks.len(), 1, "Expected exactly one tool call");

    let tool_call_block = tool_call_blocks[0];
    match tool_call_block {
        ContentBlock::ToolCall(tool_call) => {
            assert_eq!(tool_call.name, "get_temperature");
            let arguments =
                serde_json::from_str::<Value>(&tool_call.arguments.to_lowercase()).unwrap();
            let expected_arguments = json!({
                "location": "tokyo",
                "units": "celsius"
            });
            assert_eq!(arguments, expected_arguments);
        }
        _ => panic!("Unreachable"),
    }
}

pub async fn test_tool_use_tool_choice_auto_used_streaming_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "weather_helper",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                }
            ]},
        "stream": true,
        "variant_name": provider.variant_name,
    });

    let mut event_source = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .unwrap();

    let mut chunks = vec![];
    let mut found_done_chunk = false;
    while let Some(event) = event_source.next().await {
        let event = event.unwrap();
        match event {
            Event::Open => continue,
            Event::Message(message) => {
                if message.data == "[DONE]" {
                    found_done_chunk = true;
                    break;
                }
                chunks.push(message.data);
            }
        }
    }
    assert!(found_done_chunk);

    let mut inference_id = None;
    let mut tool_id: Option<String> = None;
    let mut arguments = String::new();
    let mut input_tokens = 0;
    let mut output_tokens = 0;

    for chunk in chunks {
        let chunk_json: Value = serde_json::from_str(&chunk).unwrap();

        println!("API response chunk: {chunk_json:#?}");

        let chunk_inference_id = chunk_json.get("inference_id").unwrap().as_str().unwrap();
        let chunk_inference_id = Uuid::parse_str(chunk_inference_id).unwrap();
        match inference_id {
            None => inference_id = Some(chunk_inference_id),
            Some(inference_id) => assert_eq!(inference_id, chunk_inference_id),
        }

        let chunk_episode_id = chunk_json.get("episode_id").unwrap().as_str().unwrap();
        let chunk_episode_id = Uuid::parse_str(chunk_episode_id).unwrap();
        assert_eq!(chunk_episode_id, episode_id);

        for block in chunk_json.get("content").unwrap().as_array().unwrap() {
            assert!(block.get("id").is_some());

            let block_type = block.get("type").unwrap().as_str().unwrap();

            match block_type {
                "tool_call" => {
                    assert_eq!(
                        block.get("raw_name").unwrap().as_str().unwrap(),
                        "get_temperature"
                    );

                    let block_tool_id = block.get("id").unwrap().as_str().unwrap();
                    match &tool_id {
                        None => tool_id = Some(block_tool_id.to_string()),
                        Some(tool_id) => assert_eq!(tool_id, block_tool_id),
                    }

                    let chunk_arguments = block.get("raw_arguments").unwrap().as_str().unwrap();
                    arguments.push_str(chunk_arguments);
                }
                "text" => {
                    // Sometimes the model will also return some text
                    // (e.g. "Sure, here's the weather in Tokyo:" + tool call)
                    // We mostly care about the tool call, so we'll ignore the text.
                }
                _ => {
                    panic!("Unexpected block type: {}", block_type);
                }
            }
        }

        if let Some(usage) = chunk_json.get("usage").and_then(|u| u.as_object()) {
            input_tokens += usage.get("input_tokens").unwrap().as_u64().unwrap();
            output_tokens += usage.get("output_tokens").unwrap().as_u64().unwrap();
        }
    }

    // NB: Azure doesn't return usage during streaming
    if provider.variant_name.contains("azure") {
        assert_eq!(input_tokens, 0);
        assert_eq!(output_tokens, 0);
    } else {
        assert!(input_tokens > 0);
        assert!(output_tokens > 0);
    }

    let inference_id = inference_id.unwrap();
    let tool_id = tool_id.unwrap();
    assert!(serde_json::from_str::<Value>(&arguments).is_ok());

    // Sleep for 1 second to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ClickHouse - ChatInference Table
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ChatInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, "weather_helper");

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let episode_id_result = result.get("episode_id").unwrap().as_str().unwrap();
    let episode_id_result = Uuid::parse_str(episode_id_result).unwrap();
    assert_eq!(episode_id_result, episode_id);

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input: Value = json!(
        {
            "system": {
                "assistant_name": "Dr. Mehta"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "value": "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."}]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);

    let output_clickhouse: Vec<Value> =
        serde_json::from_str(result.get("output").unwrap().as_str().unwrap()).unwrap();
    assert!(!output_clickhouse.is_empty()); // could be > 1 if the model returns text as well
    let content_block = output_clickhouse.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "tool_call");
    assert_eq!(content_block.get("id").unwrap().as_str().unwrap(), tool_id);
    assert_eq!(
        content_block.get("name").unwrap().as_str().unwrap(),
        "get_temperature"
    );
    assert_eq!(
        content_block.get("arguments").unwrap().as_object().unwrap(),
        &serde_json::from_str::<serde_json::Map<String, Value>>(&arguments).unwrap()
    );
    assert_eq!(
        content_block.get("raw_name").unwrap().as_str().unwrap(),
        "get_temperature"
    );
    assert_eq!(
        content_block
            .get("raw_arguments")
            .unwrap()
            .as_str()
            .unwrap(),
        arguments
    );

    let tool_params: Value =
        serde_json::from_str(result.get("tool_params").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(tool_params["tool_choice"], "auto");
    assert_eq!(tool_params["parallel_tool_calls"], false);

    let tools_available = tool_params["tools_available"].as_array().unwrap();
    assert_eq!(tools_available.len(), 1);
    let tool = tools_available.first().unwrap();
    assert_eq!(tool["name"], "get_temperature");
    assert_eq!(
        tool["description"],
        "Get the current temperature in a given location"
    );
    assert_eq!(tool["strict"], false);

    let tool_parameters = tool["parameters"].as_object().unwrap();
    assert_eq!(tool_parameters["type"], "object");
    assert!(tool_parameters.get("properties").is_some());
    assert!(tool_parameters.get("required").is_some());
    assert_eq!(tool_parameters["additionalProperties"], false);

    let properties = tool_parameters["properties"].as_object().unwrap();
    assert!(properties.contains_key("location"));
    assert!(properties.contains_key("units"));

    let location = properties["location"].as_object().unwrap();
    assert_eq!(location["type"], "string");
    assert_eq!(
        location["description"],
        "The location to get the temperature for (e.g. \"New York\")"
    );

    let units = properties["units"].as_object().unwrap();
    assert_eq!(units["type"], "string");
    assert_eq!(
        units["description"],
        "The units to get the temperature in (must be \"fahrenheit\" or \"celsius\")"
    );
    let units_enum = units["enum"].as_array().unwrap();
    assert_eq!(units_enum.len(), 2);
    assert!(units_enum.contains(&json!("fahrenheit")));
    assert!(units_enum.contains(&json!("celsius")));

    let required = tool_parameters["required"].as_array().unwrap();
    assert!(required.contains(&json!("location")));

    // Check if ClickHouse is correct - ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(id).is_ok());

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.contains("get_temperature"));
    assert!(raw_request.to_lowercase().contains("tokyo"));
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );

    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    assert!(raw_response.contains("get_temperature"));
    // Check if raw_response is valid JSONL
    for line in raw_response.lines() {
        assert!(serde_json::from_str::<Value>(line).is_ok());
    }

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();

    // NB: Azure doesn't support input/output tokens during streaming
    if provider.variant_name.contains("azure") {
        assert_eq!(input_tokens, 0);
        assert_eq!(output_tokens, 0);
    } else {
        assert!(input_tokens > 0);
        assert!(output_tokens > 0);
    }

    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);

    let ttft_ms = result.get("ttft_ms").unwrap().as_u64().unwrap();
    assert!(ttft_ms > 50);
    assert!(ttft_ms <= response_time_ms);

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta.\n\nPeople will ask you questions about the weather.\n\nIf asked about the weather, just respond with the tool call. Use the \"get_temperature\" tool.\n\nIf provided with a tool result, use it to respond to the user (e.g. \"The weather in New York is 55 degrees Fahrenheit.\")."
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![RequestMessage {
        role: Role::User,
        content: vec![
            "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                .to_string()
                .into(),
        ],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    let tool_call_blocks: Vec<_> = output
        .iter()
        .filter(|block| matches!(block, ContentBlock::ToolCall(_)))
        .collect();

    // Assert exactly one tool call
    assert_eq!(tool_call_blocks.len(), 1, "Expected exactly one tool call");

    let tool_call_block = tool_call_blocks[0];
    match tool_call_block {
        ContentBlock::ToolCall(tool_call) => {
            assert_eq!(tool_call.name, "get_temperature");
            let arguments =
                serde_json::from_str::<Value>(&tool_call.arguments.to_lowercase()).unwrap();
            let expected_arguments = json!({
                "location": "tokyo",
                "units": "celsius"
            });
            assert_eq!(arguments, expected_arguments);
        }
        _ => panic!("Unreachable"),
    }
}

/// This test is similar to `test_tool_use_tool_choice_auto_used_inference_request_with_provider`, but it steers the model to not use the tool.
/// This ensures that ToolChoice::Auto is working as expected.
pub async fn test_tool_use_tool_choice_auto_unused_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "weather_helper",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [
                {
                    "role": "user",
                    "content": "What is your name?"
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

    // Check if the API response is fine
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    println!("API response: {response_json:#?}");

    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    let episode_id_response = response_json.get("episode_id").unwrap().as_str().unwrap();
    let episode_id_response = Uuid::parse_str(episode_id_response).unwrap();
    assert_eq!(episode_id_response, episode_id);

    let variant_name = response_json.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let content = response_json.get("content").unwrap().as_array().unwrap();
    assert!(!content.iter().any(|block| block["type"] == "tool_call"));
    let content_block = content
        .iter()
        .find(|block| block["type"] == "text")
        .unwrap();
    let content_block_text = content_block.get("text").unwrap().as_str().unwrap();
    assert!(content_block_text.to_lowercase().contains("mehta"));

    let usage = response_json.get("usage").unwrap();
    let usage = usage.as_object().unwrap();
    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    assert!(output_tokens > 0);

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check if ClickHouse is correct - ChatInference table
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ChatInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, "weather_helper");

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let episode_id_result = result.get("episode_id").unwrap().as_str().unwrap();
    let episode_id_result = Uuid::parse_str(episode_id_result).unwrap();
    assert_eq!(episode_id_result, episode_id);

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input: Value = json!(
        {
            "system": {
                "assistant_name": "Dr. Mehta"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "value": "What is your name?"}]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);

    let output_clickhouse: Vec<Value> =
        serde_json::from_str(result.get("output").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(output_clickhouse, *content);

    let tool_params: Value =
        serde_json::from_str(result.get("tool_params").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(tool_params["tool_choice"], "auto");
    assert_eq!(tool_params["parallel_tool_calls"], false);

    let tools_available = tool_params["tools_available"].as_array().unwrap();
    assert_eq!(tools_available.len(), 1);
    let tool = tools_available.first().unwrap();
    assert_eq!(tool["name"], "get_temperature");
    assert_eq!(
        tool["description"],
        "Get the current temperature in a given location"
    );
    assert_eq!(tool["strict"], false);

    let tool_parameters = tool["parameters"].as_object().unwrap();
    assert_eq!(tool_parameters["type"], "object");
    assert!(tool_parameters.get("properties").is_some());
    assert!(tool_parameters.get("required").is_some());
    assert_eq!(tool_parameters["additionalProperties"], false);

    let properties = tool_parameters["properties"].as_object().unwrap();
    assert!(properties.contains_key("location"));
    assert!(properties.contains_key("units"));

    let location = properties["location"].as_object().unwrap();
    assert_eq!(location["type"], "string");
    assert_eq!(
        location["description"],
        "The location to get the temperature for (e.g. \"New York\")"
    );

    let units = properties["units"].as_object().unwrap();
    assert_eq!(units["type"], "string");
    assert_eq!(
        units["description"],
        "The units to get the temperature in (must be \"fahrenheit\" or \"celsius\")"
    );
    let units_enum = units["enum"].as_array().unwrap();
    assert_eq!(units_enum.len(), 2);
    assert!(units_enum.contains(&json!("fahrenheit")));
    assert!(units_enum.contains(&json!("celsius")));

    let required = tool_parameters["required"].as_array().unwrap();
    assert!(required.contains(&json!("location")));

    // Check if ClickHouse is correct - ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(id).is_ok());

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.to_lowercase().contains("what is your name"));
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );

    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    assert!(raw_response.to_lowercase().contains("mehta"));

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(output_tokens > 0);
    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);
    assert!(result.get("ttft_ms").unwrap().is_null());

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta.\n\nPeople will ask you questions about the weather.\n\nIf asked about the weather, just respond with the tool call. Use the \"get_temperature\" tool.\n\nIf provided with a tool result, use it to respond to the user (e.g. \"The weather in New York is 55 degrees Fahrenheit.\")."
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![RequestMessage {
        role: Role::User,
        content: vec!["What is your name?".to_string().into()],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    let first = output.first().unwrap();
    match first {
        ContentBlock::Text(_text) => {}
        _ => {
            panic!("Expected a text block, got {:?}", first);
        }
    }
}

/// This test is similar to `test_tool_use_tool_choice_auto_used_streaming_inference_request_with_provider`, but it steers the model to not use the tool.
/// This ensures that ToolChoice::Auto is working as expected.
pub async fn test_tool_use_tool_choice_auto_unused_streaming_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "weather_helper",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [
                {
                    "role": "user",
                    "content": "What is your name?"
                }
            ]},
        "stream": true,
        "variant_name": provider.variant_name,
    });

    let mut event_source = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .unwrap();

    let mut chunks = vec![];
    let mut found_done_chunk = false;
    while let Some(event) = event_source.next().await {
        let event = event.unwrap();
        match event {
            Event::Open => continue,
            Event::Message(message) => {
                if message.data == "[DONE]" {
                    found_done_chunk = true;
                    break;
                }
                chunks.push(message.data);
            }
        }
    }
    assert!(found_done_chunk);

    let mut inference_id = None;
    let mut full_text = String::new();
    let mut input_tokens = 0;
    let mut output_tokens = 0;

    for chunk in chunks {
        let chunk_json: Value = serde_json::from_str(&chunk).unwrap();

        println!("API response chunk: {chunk_json:#?}");

        let chunk_inference_id = chunk_json.get("inference_id").unwrap().as_str().unwrap();
        let chunk_inference_id = Uuid::parse_str(chunk_inference_id).unwrap();
        match inference_id {
            None => inference_id = Some(chunk_inference_id),
            Some(inference_id) => assert_eq!(inference_id, chunk_inference_id),
        }

        let chunk_episode_id = chunk_json.get("episode_id").unwrap().as_str().unwrap();
        let chunk_episode_id = Uuid::parse_str(chunk_episode_id).unwrap();
        assert_eq!(chunk_episode_id, episode_id);

        for block in chunk_json.get("content").unwrap().as_array().unwrap() {
            assert!(block.get("id").is_some());

            let block_type = block.get("type").unwrap().as_str().unwrap();

            match block_type {
                "tool_call" => {
                    panic!("Tool call found in streaming inference response");
                }
                "text" => {
                    full_text.push_str(block.get("text").unwrap().as_str().unwrap());
                }
                _ => {
                    panic!("Unexpected block type: {}", block_type);
                }
            }
        }

        if let Some(usage) = chunk_json.get("usage").and_then(|u| u.as_object()) {
            input_tokens += usage.get("input_tokens").unwrap().as_u64().unwrap();
            output_tokens += usage.get("output_tokens").unwrap().as_u64().unwrap();
        }
    }

    // NB: Azure doesn't return usage during streaming
    if provider.variant_name.contains("azure") {
        assert_eq!(input_tokens, 0);
        assert_eq!(output_tokens, 0);
    } else {
        assert!(input_tokens > 0);
        assert!(output_tokens > 0);
    }

    let inference_id = inference_id.unwrap();

    assert!(full_text.to_lowercase().contains("mehta"));

    // Sleep for 1 second to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ClickHouse - ChatInference Table
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ChatInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, "weather_helper");

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let episode_id_result = result.get("episode_id").unwrap().as_str().unwrap();
    let episode_id_result = Uuid::parse_str(episode_id_result).unwrap();
    assert_eq!(episode_id_result, episode_id);

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input: Value = json!(
        {
            "system": {
                "assistant_name": "Dr. Mehta"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "value": "What is your name?"}]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);

    let output_clickhouse: Vec<Value> =
        serde_json::from_str(result.get("output").unwrap().as_str().unwrap()).unwrap();

    assert!(!output_clickhouse
        .iter()
        .any(|block| block["type"] == "tool_call"));

    let content_block = output_clickhouse.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    assert!(content_block
        .get("text")
        .unwrap()
        .as_str()
        .unwrap()
        .to_lowercase()
        .contains("mehta"));

    let tool_params: Value =
        serde_json::from_str(result.get("tool_params").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(tool_params["tool_choice"], "auto");
    assert_eq!(tool_params["parallel_tool_calls"], false);

    let tools_available = tool_params["tools_available"].as_array().unwrap();
    assert_eq!(tools_available.len(), 1);
    let tool = tools_available.first().unwrap();
    assert_eq!(tool["name"], "get_temperature");
    assert_eq!(
        tool["description"],
        "Get the current temperature in a given location"
    );
    assert_eq!(tool["strict"], false);

    let tool_parameters = tool["parameters"].as_object().unwrap();
    assert_eq!(tool_parameters["type"], "object");
    assert!(tool_parameters.get("properties").is_some());
    assert!(tool_parameters.get("required").is_some());
    assert_eq!(tool_parameters["additionalProperties"], false);

    let properties = tool_parameters["properties"].as_object().unwrap();
    assert!(properties.contains_key("location"));
    assert!(properties.contains_key("units"));

    let location = properties["location"].as_object().unwrap();
    assert_eq!(location["type"], "string");
    assert_eq!(
        location["description"],
        "The location to get the temperature for (e.g. \"New York\")"
    );

    let units = properties["units"].as_object().unwrap();
    assert_eq!(units["type"], "string");
    assert_eq!(
        units["description"],
        "The units to get the temperature in (must be \"fahrenheit\" or \"celsius\")"
    );
    let units_enum = units["enum"].as_array().unwrap();
    assert_eq!(units_enum.len(), 2);
    assert!(units_enum.contains(&json!("fahrenheit")));
    assert!(units_enum.contains(&json!("celsius")));

    let required = tool_parameters["required"].as_array().unwrap();
    assert!(required.contains(&json!("location")));

    // Check if ClickHouse is correct - ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(id).is_ok());

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.to_lowercase().contains("what is your name"));
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );

    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    // Check if raw_response is valid JSONL
    for line in raw_response.lines() {
        assert!(serde_json::from_str::<Value>(line).is_ok());
    }

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();

    // NB: Azure doesn't support input/output tokens during streaming
    if provider.variant_name.contains("azure") {
        assert_eq!(input_tokens, 0);
        assert_eq!(output_tokens, 0);
    } else {
        assert!(input_tokens > 0);
        assert!(output_tokens > 0);
    }

    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);

    let ttft_ms = result.get("ttft_ms").unwrap().as_u64().unwrap();
    assert!(ttft_ms > 50);
    assert!(ttft_ms <= response_time_ms);

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta.\n\nPeople will ask you questions about the weather.\n\nIf asked about the weather, just respond with the tool call. Use the \"get_temperature\" tool.\n\nIf provided with a tool result, use it to respond to the user (e.g. \"The weather in New York is 55 degrees Fahrenheit.\")."
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![RequestMessage {
        role: Role::User,
        content: vec!["What is your name?".to_string().into()],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    let first = output.first().unwrap();
    match first {
        ContentBlock::Text(_text) => {}
        _ => {
            panic!("Expected a text block, got {:?}", first);
        }
    }
}

pub async fn test_tool_use_tool_choice_required_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    // Azure doesn't support `tool_choice: "required"`
    if provider.model_provider_name == "azure" {
        return;
    }

    // GCP Vertex doesn't support `tool_choice: "required"` for Gemini 1.5 Flash
    if provider.model_provider_name.contains("gcp_vertex")
        && provider.model_name == "gemini-1.5-flash-001"
    {
        return;
    }

    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "weather_helper",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [
                {
                    "role": "user",
                    "content": "What is your name?"
                }
            ]},
        "tool_choice": "required",
        "stream": false,
        "variant_name": provider.variant_name,
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check if the API response is fine
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    println!("API response: {response_json:#?}");

    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    let episode_id_response = response_json.get("episode_id").unwrap().as_str().unwrap();
    let episode_id_response = Uuid::parse_str(episode_id_response).unwrap();
    assert_eq!(episode_id_response, episode_id);

    let variant_name = response_json.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let content = response_json.get("content").unwrap().as_array().unwrap();
    assert!(!content.is_empty()); // could be > 1 if the model returns text as well
    let content_block = content
        .iter()
        .find(|block| block["type"] == "tool_call")
        .unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "tool_call");

    assert!(content_block.get("id").unwrap().as_str().is_some());

    let raw_name = content_block.get("raw_name").unwrap().as_str().unwrap();
    assert_eq!(raw_name, "get_temperature");
    let name = content_block.get("name").unwrap().as_str().unwrap();
    assert_eq!(name, "get_temperature");

    let raw_arguments = content_block
        .get("raw_arguments")
        .unwrap()
        .as_str()
        .unwrap();
    let raw_arguments: Value = serde_json::from_str(raw_arguments).unwrap();
    let raw_arguments = raw_arguments.as_object().unwrap();
    assert!(raw_arguments.len() == 1 || raw_arguments.len() == 2);
    assert!(raw_arguments.get("location").unwrap().as_str().is_some());
    if raw_arguments.len() == 2 {
        let units = raw_arguments.get("units").unwrap().as_str().unwrap();
        assert!(units == "celsius" || units == "fahrenheit");
    }

    let arguments = content_block.get("arguments").unwrap();
    let arguments = arguments.as_object().unwrap();
    assert!(arguments.len() == 1 || arguments.len() == 2);
    assert!(arguments.get("location").unwrap().as_str().is_some());
    if arguments.len() == 2 {
        let units = arguments.get("units").unwrap().as_str().unwrap();
        assert!(units == "celsius" || units == "fahrenheit");
    }

    let usage = response_json.get("usage").unwrap();
    let usage = usage.as_object().unwrap();
    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    assert!(output_tokens > 0);

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check if ClickHouse is correct - ChatInference table
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ChatInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, "weather_helper");

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let episode_id_result = result.get("episode_id").unwrap().as_str().unwrap();
    let episode_id_result = Uuid::parse_str(episode_id_result).unwrap();
    assert_eq!(episode_id_result, episode_id);

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input: Value = json!(
        {
            "system": {
                "assistant_name": "Dr. Mehta"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "value": "What is your name?"}]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);

    let output_clickhouse: Vec<Value> =
        serde_json::from_str(result.get("output").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(output_clickhouse, *content);

    let tool_params: Value =
        serde_json::from_str(result.get("tool_params").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(tool_params["tool_choice"], "required");
    assert_eq!(tool_params["parallel_tool_calls"], false);

    let tools_available = tool_params["tools_available"].as_array().unwrap();
    assert_eq!(tools_available.len(), 1);
    let tool = tools_available.first().unwrap();
    assert_eq!(tool["name"], "get_temperature");
    assert_eq!(
        tool["description"],
        "Get the current temperature in a given location"
    );
    assert_eq!(tool["strict"], false);

    let tool_parameters = tool["parameters"].as_object().unwrap();
    assert_eq!(tool_parameters["type"], "object");
    assert!(tool_parameters.get("properties").is_some());
    assert!(tool_parameters.get("required").is_some());
    assert_eq!(tool_parameters["additionalProperties"], false);

    let properties = tool_parameters["properties"].as_object().unwrap();
    assert!(properties.contains_key("location"));
    assert!(properties.contains_key("units"));

    let location = properties["location"].as_object().unwrap();
    assert_eq!(location["type"], "string");
    assert_eq!(
        location["description"],
        "The location to get the temperature for (e.g. \"New York\")"
    );

    let units = properties["units"].as_object().unwrap();
    assert_eq!(units["type"], "string");
    assert_eq!(
        units["description"],
        "The units to get the temperature in (must be \"fahrenheit\" or \"celsius\")"
    );
    let units_enum = units["enum"].as_array().unwrap();
    assert_eq!(units_enum.len(), 2);
    assert!(units_enum.contains(&json!("fahrenheit")));
    assert!(units_enum.contains(&json!("celsius")));

    let required = tool_parameters["required"].as_array().unwrap();
    assert!(required.contains(&json!("location")));

    // Check if ClickHouse is correct - ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(id).is_ok());

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.to_lowercase().contains("what is your name"));
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );

    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    assert!(raw_response.contains("get_temperature"));

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(output_tokens > 0);
    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);
    assert!(result.get("ttft_ms").unwrap().is_null());

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta.\n\nPeople will ask you questions about the weather.\n\nIf asked about the weather, just respond with the tool call. Use the \"get_temperature\" tool.\n\nIf provided with a tool result, use it to respond to the user (e.g. \"The weather in New York is 55 degrees Fahrenheit.\")."
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![RequestMessage {
        role: Role::User,
        content: vec!["What is your name?".to_string().into()],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    let tool_call_blocks: Vec<_> = output
        .iter()
        .filter(|block| matches!(block, ContentBlock::ToolCall(_)))
        .collect();

    // Assert exactly one tool call
    assert_eq!(tool_call_blocks.len(), 1, "Expected exactly one tool call");

    let tool_call_block = tool_call_blocks[0];
    match tool_call_block {
        ContentBlock::ToolCall(tool_call) => {
            assert_eq!(tool_call.name, "get_temperature");
            serde_json::from_str::<Value>(&tool_call.arguments.to_lowercase()).unwrap();
        }
        _ => panic!("Unreachable"),
    }
}

pub async fn test_tool_use_tool_choice_required_streaming_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    // Azure doesn't support `tool_choice: "required"`
    if provider.model_provider_name == "azure" {
        return;
    }

    // GCP Vertex doesn't support `tool_choice: "required"` for Gemini 1.5 Flash
    if provider.model_provider_name.contains("gcp_vertex")
        && provider.model_name == "gemini-1.5-flash-001"
    {
        return;
    }

    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "weather_helper",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [
                {
                    "role": "user",
                    "content": "What is your name?"
                }
            ]},
        "tool_choice": "required",
        "stream": true,
        "variant_name": provider.variant_name,
    });

    let mut event_source = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .unwrap();

    let mut chunks = vec![];
    let mut found_done_chunk = false;
    while let Some(event) = event_source.next().await {
        let event = event.unwrap();
        match event {
            Event::Open => continue,
            Event::Message(message) => {
                if message.data == "[DONE]" {
                    found_done_chunk = true;
                    break;
                }
                chunks.push(message.data);
            }
        }
    }
    assert!(found_done_chunk);

    let mut inference_id = None;
    let mut tool_id: Option<String> = None;
    let mut arguments = String::new();
    let mut input_tokens = 0;
    let mut output_tokens = 0;

    for chunk in chunks {
        let chunk_json: Value = serde_json::from_str(&chunk).unwrap();

        println!("API response chunk: {chunk_json:#?}");

        let chunk_inference_id = chunk_json.get("inference_id").unwrap().as_str().unwrap();
        let chunk_inference_id = Uuid::parse_str(chunk_inference_id).unwrap();
        match inference_id {
            None => inference_id = Some(chunk_inference_id),
            Some(inference_id) => assert_eq!(inference_id, chunk_inference_id),
        }

        let chunk_episode_id = chunk_json.get("episode_id").unwrap().as_str().unwrap();
        let chunk_episode_id = Uuid::parse_str(chunk_episode_id).unwrap();
        assert_eq!(chunk_episode_id, episode_id);

        for block in chunk_json.get("content").unwrap().as_array().unwrap() {
            assert!(block.get("id").is_some());

            let block_type = block.get("type").unwrap().as_str().unwrap();

            match block_type {
                "tool_call" => {
                    assert_eq!(
                        block.get("raw_name").unwrap().as_str().unwrap(),
                        "get_temperature"
                    );

                    let block_tool_id = block.get("id").unwrap().as_str().unwrap();
                    match &tool_id {
                        None => tool_id = Some(block_tool_id.to_string()),
                        Some(tool_id) => assert_eq!(tool_id, block_tool_id),
                    }

                    let chunk_arguments = block.get("raw_arguments").unwrap().as_str().unwrap();
                    arguments.push_str(chunk_arguments);
                }
                "text" => {
                    // Sometimes the model will also return some text
                    // (e.g. "Sure, here's the weather in Tokyo:" + tool call)
                    // We mostly care about the tool call, so we'll ignore the text.
                }
                _ => {
                    panic!("Unexpected block type: {}", block_type);
                }
            }
        }

        if let Some(usage) = chunk_json.get("usage").and_then(|u| u.as_object()) {
            input_tokens += usage.get("input_tokens").unwrap().as_u64().unwrap();
            output_tokens += usage.get("output_tokens").unwrap().as_u64().unwrap();
        }
    }

    // NB: Azure doesn't return usage during streaming
    if provider.variant_name.contains("azure") {
        assert_eq!(input_tokens, 0);
        assert_eq!(output_tokens, 0);
    } else {
        assert!(input_tokens > 0);
        assert!(output_tokens > 0);
    }

    let inference_id = inference_id.unwrap();
    let tool_id = tool_id.unwrap();
    assert!(serde_json::from_str::<Value>(&arguments).is_ok());

    // Sleep for 1 second to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ClickHouse - ChatInference Table
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ChatInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, "weather_helper");

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let episode_id_result = result.get("episode_id").unwrap().as_str().unwrap();
    let episode_id_result = Uuid::parse_str(episode_id_result).unwrap();
    assert_eq!(episode_id_result, episode_id);

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input: Value = json!(
        {
            "system": {
                "assistant_name": "Dr. Mehta"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "value": "What is your name?"}]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);

    let output_clickhouse: Vec<Value> =
        serde_json::from_str(result.get("output").unwrap().as_str().unwrap()).unwrap();
    assert!(!output_clickhouse.is_empty()); // could be > 1 if the model returns text as well
    let content_block = output_clickhouse.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "tool_call");
    assert_eq!(content_block.get("id").unwrap().as_str().unwrap(), tool_id);
    assert_eq!(
        content_block.get("raw_name").unwrap().as_str().unwrap(),
        "get_temperature"
    );
    assert_eq!(
        content_block
            .get("raw_arguments")
            .unwrap()
            .as_str()
            .unwrap(),
        arguments
    );
    assert_eq!(
        content_block.get("name").unwrap().as_str().unwrap(),
        "get_temperature"
    );
    assert_eq!(
        content_block.get("arguments").unwrap().as_object().unwrap(),
        &serde_json::from_str::<serde_json::Map<String, Value>>(&arguments).unwrap()
    );

    let tool_params: Value =
        serde_json::from_str(result.get("tool_params").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(tool_params["tool_choice"], "required");
    assert_eq!(tool_params["parallel_tool_calls"], false);

    let tools_available = tool_params["tools_available"].as_array().unwrap();
    assert_eq!(tools_available.len(), 1);
    let tool = tools_available.first().unwrap();
    assert_eq!(tool["name"], "get_temperature");
    assert_eq!(
        tool["description"],
        "Get the current temperature in a given location"
    );
    assert_eq!(tool["strict"], false);

    let tool_parameters = tool["parameters"].as_object().unwrap();
    assert_eq!(tool_parameters["type"], "object");
    assert!(tool_parameters.get("properties").is_some());
    assert!(tool_parameters.get("required").is_some());
    assert_eq!(tool_parameters["additionalProperties"], false);

    let properties = tool_parameters["properties"].as_object().unwrap();
    assert!(properties.contains_key("location"));
    assert!(properties.contains_key("units"));

    let location = properties["location"].as_object().unwrap();
    assert_eq!(location["type"], "string");
    assert_eq!(
        location["description"],
        "The location to get the temperature for (e.g. \"New York\")"
    );

    let units = properties["units"].as_object().unwrap();
    assert_eq!(units["type"], "string");
    assert_eq!(
        units["description"],
        "The units to get the temperature in (must be \"fahrenheit\" or \"celsius\")"
    );
    let units_enum = units["enum"].as_array().unwrap();
    assert_eq!(units_enum.len(), 2);
    assert!(units_enum.contains(&json!("fahrenheit")));
    assert!(units_enum.contains(&json!("celsius")));

    let required = tool_parameters["required"].as_array().unwrap();
    assert!(required.contains(&json!("location")));

    // Check if ClickHouse is correct - ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(id).is_ok());

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.to_lowercase().contains("what is your name"));
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );

    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    assert!(raw_response.contains("get_temperature"));
    // Check if raw_response is valid JSONL
    for line in raw_response.lines() {
        assert!(serde_json::from_str::<Value>(line).is_ok());
    }

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();

    // NB: Azure doesn't support input/output tokens during streaming
    if provider.variant_name.contains("azure") {
        assert_eq!(input_tokens, 0);
        assert_eq!(output_tokens, 0);
    } else {
        assert!(input_tokens > 0);
        assert!(output_tokens > 0);
    }

    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);

    let ttft_ms = result.get("ttft_ms").unwrap().as_u64().unwrap();
    assert!(ttft_ms > 50);
    assert!(ttft_ms <= response_time_ms);

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta.\n\nPeople will ask you questions about the weather.\n\nIf asked about the weather, just respond with the tool call. Use the \"get_temperature\" tool.\n\nIf provided with a tool result, use it to respond to the user (e.g. \"The weather in New York is 55 degrees Fahrenheit.\")."
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![RequestMessage {
        role: Role::User,
        content: vec!["What is your name?".to_string().into()],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    let tool_call_blocks: Vec<_> = output
        .iter()
        .filter(|block| matches!(block, ContentBlock::ToolCall(_)))
        .collect();

    // Assert exactly one tool call
    assert_eq!(tool_call_blocks.len(), 1, "Expected exactly one tool call");

    let tool_call_block = tool_call_blocks[0];
    match tool_call_block {
        ContentBlock::ToolCall(tool_call) => {
            assert_eq!(tool_call.name, "get_temperature");
            serde_json::from_str::<Value>(&tool_call.arguments.to_lowercase()).unwrap();
        }
        _ => panic!("Unreachable"),
    }
}

pub async fn test_tool_use_tool_choice_none_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    // TODO (#204): Implement ToolChoice::None workaround for AWS Bedrock.
    if provider.model_provider_name.contains("aws-bedrock") {
        return;
    }

    // TODO (#205): Implement ToolChoice::None workaround for Anthropic.
    if provider.model_provider_name.contains("anthropic") {
        return;
    }

    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "weather_helper",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                }
            ]},
        "tool_choice": "none",
        "stream": false,
        "variant_name": provider.variant_name,
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check if the API response is fine
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    println!("API response: {response_json:#?}");

    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    let episode_id_response = response_json.get("episode_id").unwrap().as_str().unwrap();
    let episode_id_response = Uuid::parse_str(episode_id_response).unwrap();
    assert_eq!(episode_id_response, episode_id);

    let variant_name = response_json.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let content = response_json.get("content").unwrap().as_array().unwrap();
    assert!(!content.iter().any(|block| block["type"] == "tool_call"));
    let content_block = content
        .iter()
        .find(|block| block["type"] == "text")
        .unwrap();
    assert!(content_block.get("text").unwrap().as_str().is_some());

    let usage = response_json.get("usage").unwrap();
    let usage = usage.as_object().unwrap();
    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    assert!(output_tokens > 0);

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check if ClickHouse is correct - ChatInference table
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ChatInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, "weather_helper");

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let episode_id_result = result.get("episode_id").unwrap().as_str().unwrap();
    let episode_id_result = Uuid::parse_str(episode_id_result).unwrap();
    assert_eq!(episode_id_result, episode_id);

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input: Value = json!(
        {
            "system": {
                "assistant_name": "Dr. Mehta"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "value": "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."}]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);

    let output_clickhouse: Vec<Value> =
        serde_json::from_str(result.get("output").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(output_clickhouse, *content);

    let tool_params: Value =
        serde_json::from_str(result.get("tool_params").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(tool_params["tool_choice"], "none");
    assert_eq!(tool_params["parallel_tool_calls"], false);

    let tools_available = tool_params["tools_available"].as_array().unwrap();
    assert_eq!(tools_available.len(), 1);
    let tool = tools_available.first().unwrap();
    assert_eq!(tool["name"], "get_temperature");
    assert_eq!(
        tool["description"],
        "Get the current temperature in a given location"
    );
    assert_eq!(tool["strict"], false);

    let tool_parameters = tool["parameters"].as_object().unwrap();
    assert_eq!(tool_parameters["type"], "object");
    assert!(tool_parameters.get("properties").is_some());
    assert!(tool_parameters.get("required").is_some());
    assert_eq!(tool_parameters["additionalProperties"], false);

    let properties = tool_parameters["properties"].as_object().unwrap();
    assert!(properties.contains_key("location"));
    assert!(properties.contains_key("units"));

    let location = properties["location"].as_object().unwrap();
    assert_eq!(location["type"], "string");
    assert_eq!(
        location["description"],
        "The location to get the temperature for (e.g. \"New York\")"
    );

    let units = properties["units"].as_object().unwrap();
    assert_eq!(units["type"], "string");
    assert_eq!(
        units["description"],
        "The units to get the temperature in (must be \"fahrenheit\" or \"celsius\")"
    );
    let units_enum = units["enum"].as_array().unwrap();
    assert_eq!(units_enum.len(), 2);
    assert!(units_enum.contains(&json!("fahrenheit")));
    assert!(units_enum.contains(&json!("celsius")));

    let required = tool_parameters["required"].as_array().unwrap();
    assert!(required.contains(&json!("location")));

    // Check if ClickHouse is correct - ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(id).is_ok());

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.to_lowercase().contains("tokyo"));
    assert!(raw_request.to_lowercase().contains("celsius"));
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );

    assert!(result.get("raw_response").unwrap().as_str().is_some());

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(output_tokens > 0);
    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);
    assert!(result.get("ttft_ms").unwrap().is_null());

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta.\n\nPeople will ask you questions about the weather.\n\nIf asked about the weather, just respond with the tool call. Use the \"get_temperature\" tool.\n\nIf provided with a tool result, use it to respond to the user (e.g. \"The weather in New York is 55 degrees Fahrenheit.\")."
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![RequestMessage {
        role: Role::User,
        content: vec![
            "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                .to_string()
                .into(),
        ],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    let first = output.first().unwrap();
    match first {
        ContentBlock::Text(_text) => {}
        _ => {
            panic!("Expected a text block, got {:?}", first);
        }
    }
}

pub async fn test_tool_use_tool_choice_none_streaming_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    // TODO (#204): Implement ToolChoice::None workaround for AWS Bedrock.
    if provider.model_provider_name.contains("aws-bedrock") {
        return;
    }

    // TODO (#205): Implement ToolChoice::None workaround for Anthropic.
    if provider.model_provider_name.contains("anthropic") {
        return;
    }

    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "weather_helper",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                }
            ]},
        "tool_choice": "none",
        "stream": true,
        "variant_name": provider.variant_name,
    });

    let mut event_source = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .unwrap();

    let mut chunks = vec![];
    let mut found_done_chunk = false;
    while let Some(event) = event_source.next().await {
        let event = event.unwrap();
        match event {
            Event::Open => continue,
            Event::Message(message) => {
                if message.data == "[DONE]" {
                    found_done_chunk = true;
                    break;
                }
                chunks.push(message.data);
            }
        }
    }
    assert!(found_done_chunk);

    let mut inference_id = None;
    let mut full_text = String::new();
    let mut input_tokens = 0;
    let mut output_tokens = 0;

    for chunk in chunks {
        let chunk_json: Value = serde_json::from_str(&chunk).unwrap();

        println!("API response chunk: {chunk_json:#?}");

        let chunk_inference_id = chunk_json.get("inference_id").unwrap().as_str().unwrap();
        let chunk_inference_id = Uuid::parse_str(chunk_inference_id).unwrap();
        match inference_id {
            None => inference_id = Some(chunk_inference_id),
            Some(inference_id) => assert_eq!(inference_id, chunk_inference_id),
        }

        let chunk_episode_id = chunk_json.get("episode_id").unwrap().as_str().unwrap();
        let chunk_episode_id = Uuid::parse_str(chunk_episode_id).unwrap();
        assert_eq!(chunk_episode_id, episode_id);

        for block in chunk_json.get("content").unwrap().as_array().unwrap() {
            assert!(block.get("id").is_some());

            let block_type = block.get("type").unwrap().as_str().unwrap();

            match block_type {
                "tool_call" => {
                    panic!("Tool call found in streaming inference response");
                }
                "text" => {
                    full_text.push_str(block.get("text").unwrap().as_str().unwrap());
                }
                _ => {
                    panic!("Unexpected block type: {}", block_type);
                }
            }
        }

        if let Some(usage) = chunk_json.get("usage").and_then(|u| u.as_object()) {
            input_tokens += usage.get("input_tokens").unwrap().as_u64().unwrap();
            output_tokens += usage.get("output_tokens").unwrap().as_u64().unwrap();
        }
    }

    // NB: Azure doesn't return usage during streaming
    if provider.variant_name.contains("azure") {
        assert_eq!(input_tokens, 0);
        assert_eq!(output_tokens, 0);
    } else {
        assert!(input_tokens > 0);
        assert!(output_tokens > 0);
    }

    let inference_id = inference_id.unwrap();

    // Sleep for 1 second to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ClickHouse - ChatInference Table
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ChatInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, "weather_helper");

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let episode_id_result = result.get("episode_id").unwrap().as_str().unwrap();
    let episode_id_result = Uuid::parse_str(episode_id_result).unwrap();
    assert_eq!(episode_id_result, episode_id);

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input: Value = json!(
        {
            "system": {
                "assistant_name": "Dr. Mehta"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "value": "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."}]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);

    let output_clickhouse: Vec<Value> =
        serde_json::from_str(result.get("output").unwrap().as_str().unwrap()).unwrap();

    assert!(!output_clickhouse
        .iter()
        .any(|block| block["type"] == "tool_call"));

    let content_block = output_clickhouse.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    assert!(content_block.get("text").unwrap().as_str().is_some());

    let tool_params: Value =
        serde_json::from_str(result.get("tool_params").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(tool_params["tool_choice"], "none");
    assert_eq!(tool_params["parallel_tool_calls"], false);

    let tools_available = tool_params["tools_available"].as_array().unwrap();
    assert_eq!(tools_available.len(), 1);

    let tool = tools_available
        .iter()
        .find(|tool| tool["name"] == "get_temperature")
        .unwrap();
    assert_eq!(
        tool["description"],
        "Get the current temperature in a given location"
    );
    assert_eq!(tool["strict"], false);

    let tool_parameters = tool["parameters"].as_object().unwrap();
    assert_eq!(tool_parameters["type"], "object");
    assert!(tool_parameters.get("properties").is_some());
    assert!(tool_parameters.get("required").is_some());
    assert_eq!(tool_parameters["additionalProperties"], false);

    let properties = tool_parameters["properties"].as_object().unwrap();
    assert!(properties.contains_key("location"));
    assert!(properties.contains_key("units"));

    let location = properties["location"].as_object().unwrap();
    assert_eq!(location["type"], "string");
    assert_eq!(
        location["description"],
        "The location to get the temperature for (e.g. \"New York\")"
    );

    let units = properties["units"].as_object().unwrap();
    assert_eq!(units["type"], "string");
    assert_eq!(
        units["description"],
        "The units to get the temperature in (must be \"fahrenheit\" or \"celsius\")"
    );
    let units_enum = units["enum"].as_array().unwrap();
    assert_eq!(units_enum.len(), 2);
    assert!(units_enum.contains(&json!("fahrenheit")));
    assert!(units_enum.contains(&json!("celsius")));

    let required = tool_parameters["required"].as_array().unwrap();
    assert!(required.contains(&json!("location")));

    // Check if ClickHouse is correct - ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(id).is_ok());

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request
        .to_lowercase()
        .contains("what is the weather like in tokyo (in celsius)"));
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );

    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    // Check if raw_response is valid JSONL
    for line in raw_response.lines() {
        assert!(serde_json::from_str::<Value>(line).is_ok());
    }

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();

    // NB: Azure doesn't support input/output tokens during streaming
    if provider.variant_name.contains("azure") {
        assert_eq!(input_tokens, 0);
        assert_eq!(output_tokens, 0);
    } else {
        assert!(input_tokens > 0);
        assert!(output_tokens > 0);
    }

    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);

    let ttft_ms = result.get("ttft_ms").unwrap().as_u64().unwrap();
    assert!(ttft_ms > 50);
    assert!(ttft_ms <= response_time_ms);

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta.\n\nPeople will ask you questions about the weather.\n\nIf asked about the weather, just respond with the tool call. Use the \"get_temperature\" tool.\n\nIf provided with a tool result, use it to respond to the user (e.g. \"The weather in New York is 55 degrees Fahrenheit.\")."
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![RequestMessage {
        role: Role::User,
        content: vec![
            "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                .to_string()
                .into(),
        ],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    let first = output.first().unwrap();
    match first {
        ContentBlock::Text(_text) => {}
        _ => {
            panic!("Expected a text block, got {:?}", first);
        }
    }
}

pub async fn test_tool_use_tool_choice_specific_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    // Mistral and GCP Vertex don't support ToolChoice::Specific.
    // In those cases, we use ToolChoice::Any with a single tool under the hood.
    // Even then, they seem to hallucinate a new tool.
    if provider.model_provider_name == "mistral"
        || provider.model_provider_name.contains("gcp_vertex")
    {
        return;
    }

    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "weather_helper",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [
                {
                    "role": "user",
                    "content": "What is the temperature like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                }
            ]},
        "tool_choice": {"specific": "self_destruct"},
        "additional_tools": [
            {
                "name": "self_destruct",
                "description": "Do not call this function under any circumstances.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "fast": {
                            "type": "boolean",
                            "description": "Whether to use a fast method to self-destruct."
                        },
                    },
                    "required": ["fast"],
                    "additionalProperties": false
                },
            }
        ],
        "stream": false,
        "variant_name": provider.variant_name,
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check if the API response is fine
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    println!("API response: {response_json:#?}");

    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    let episode_id_response = response_json.get("episode_id").unwrap().as_str().unwrap();
    let episode_id_response = Uuid::parse_str(episode_id_response).unwrap();
    assert_eq!(episode_id_response, episode_id);

    let variant_name = response_json.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let content = response_json.get("content").unwrap().as_array().unwrap();
    assert!(!content.is_empty()); // could be > 1 if the model returns text as well
    let content_block = content
        .iter()
        .find(|block| block["type"] == "tool_call")
        .unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "tool_call");

    assert!(content_block.get("id").unwrap().as_str().is_some());

    let raw_name = content_block.get("raw_name").unwrap().as_str().unwrap();
    assert_eq!(raw_name, "self_destruct");
    let name = content_block.get("name").unwrap().as_str().unwrap();
    assert_eq!(name, "self_destruct");

    let raw_arguments = content_block
        .get("raw_arguments")
        .unwrap()
        .as_str()
        .unwrap();
    let raw_arguments: Value = serde_json::from_str(raw_arguments).unwrap();
    let raw_arguments = raw_arguments.as_object().unwrap();
    assert!(raw_arguments.len() == 1);
    assert!(raw_arguments.get("fast").unwrap().as_bool().is_some());

    let arguments = content_block.get("arguments").unwrap();
    let arguments = arguments.as_object().unwrap();
    assert!(arguments.len() == 1);
    assert!(arguments.get("fast").unwrap().as_bool().is_some());

    let usage = response_json.get("usage").unwrap();
    let usage = usage.as_object().unwrap();
    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    assert!(output_tokens > 0);

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check if ClickHouse is correct - ChatInference table
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ChatInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, "weather_helper");

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let episode_id_result = result.get("episode_id").unwrap().as_str().unwrap();
    let episode_id_result = Uuid::parse_str(episode_id_result).unwrap();
    assert_eq!(episode_id_result, episode_id);

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input: Value = json!(
        {
            "system": {
                "assistant_name": "Dr. Mehta"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "value": "What is the temperature like in Tokyo (in Celsius)? Use the `get_temperature` tool."}]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);

    let output_clickhouse: Vec<Value> =
        serde_json::from_str(result.get("output").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(output_clickhouse, *content);

    let tool_params: Value =
        serde_json::from_str(result.get("tool_params").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(
        tool_params["tool_choice"],
        json!({"specific": "self_destruct"})
    );
    assert_eq!(tool_params["parallel_tool_calls"], false);

    let tools_available = tool_params["tools_available"].as_array().unwrap();
    assert_eq!(tools_available.len(), 2);
    let tool = tools_available
        .iter()
        .find(|tool| tool["name"] == "get_temperature")
        .unwrap();
    assert_eq!(tool["name"], "get_temperature");
    assert_eq!(
        tool["description"],
        "Get the current temperature in a given location"
    );
    assert_eq!(tool["strict"], false);

    let tool_parameters = tool["parameters"].as_object().unwrap();
    assert_eq!(tool_parameters["type"], "object");
    assert!(tool_parameters.get("properties").is_some());
    assert!(tool_parameters.get("required").is_some());
    assert_eq!(tool_parameters["additionalProperties"], false);

    let properties = tool_parameters["properties"].as_object().unwrap();
    assert!(properties.contains_key("location"));
    assert!(properties.contains_key("units"));

    let location = properties["location"].as_object().unwrap();
    assert_eq!(location["type"], "string");
    assert_eq!(
        location["description"],
        "The location to get the temperature for (e.g. \"New York\")"
    );

    let units = properties["units"].as_object().unwrap();
    assert_eq!(units["type"], "string");
    assert_eq!(
        units["description"],
        "The units to get the temperature in (must be \"fahrenheit\" or \"celsius\")"
    );
    let units_enum = units["enum"].as_array().unwrap();
    assert_eq!(units_enum.len(), 2);
    assert!(units_enum.contains(&json!("fahrenheit")));
    assert!(units_enum.contains(&json!("celsius")));

    let required = tool_parameters["required"].as_array().unwrap();
    assert!(required.contains(&json!("location")));

    let tool = tools_available
        .iter()
        .find(|tool| tool["name"] == "self_destruct")
        .unwrap();
    assert_eq!(
        tool["description"],
        "Do not call this function under any circumstances."
    );
    assert_eq!(tool["strict"], false);

    let tool_parameters = tool["parameters"].as_object().unwrap();
    assert_eq!(tool_parameters["type"], "object");
    assert!(tool_parameters.get("properties").is_some());
    assert!(tool_parameters
        .get("required")
        .unwrap()
        .as_array()
        .unwrap()
        .contains(&json!("fast")));
    assert_eq!(tool_parameters["additionalProperties"], false);

    let properties = tool_parameters["properties"].as_object().unwrap();
    println!("Properties: {properties:#?}");
    assert!(properties.get("fast").is_some());

    // Check if ClickHouse is correct - ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(id).is_ok());

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.contains("self_destruct"));
    assert!(raw_request.to_lowercase().contains("tokyo"));
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );

    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    assert!(raw_response.contains("self_destruct"));

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(output_tokens > 0);
    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);
    assert!(result.get("ttft_ms").unwrap().is_null());

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta.\n\nPeople will ask you questions about the weather.\n\nIf asked about the weather, just respond with the tool call. Use the \"get_temperature\" tool.\n\nIf provided with a tool result, use it to respond to the user (e.g. \"The weather in New York is 55 degrees Fahrenheit.\")."
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![RequestMessage {
        role: Role::User,
        content: vec![
            "What is the temperature like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                .to_string()
                .into(),
        ],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    let tool_call_blocks: Vec<_> = output
        .iter()
        .filter(|block| matches!(block, ContentBlock::ToolCall(_)))
        .collect();

    // Assert exactly one tool call
    assert_eq!(tool_call_blocks.len(), 1, "Expected exactly one tool call");

    let tool_call_block = tool_call_blocks[0];
    match tool_call_block {
        ContentBlock::ToolCall(tool_call) => {
            assert_eq!(tool_call.name, "self_destruct");
            serde_json::from_str::<Value>(&tool_call.arguments.to_lowercase()).unwrap();
        }
        _ => panic!("Unreachable"),
    }
}

pub async fn test_tool_use_tool_choice_specific_streaming_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    // Mistral and GCP Vertex don't support ToolChoice::Specific.
    // In those cases, we use ToolChoice::Any with a single tool under the hood.
    // Even then, they seem to hallucinate a new tool.
    if provider.model_provider_name == "mistral"
        || provider.model_provider_name.contains("gcp_vertex")
    {
        return;
    }

    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "weather_helper",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [
                {
                    "role": "user",
                    "content": "What is the temperature like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                }
            ]},
        "additional_tools": [
            {
                "name": "self_destruct",
                "description": "Do not call this function under any circumstances.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "fast": {
                            "type": "boolean",
                            "description": "Whether to use a fast method to self-destruct."
                        },
                    },
                    "required": ["fast"],
                    "additionalProperties": false
                },
            }
        ],
        "tool_choice": {"specific": "self_destruct"},
        "stream": true,
        "variant_name": provider.variant_name,
    });

    let mut event_source = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .unwrap();

    let mut chunks = vec![];
    let mut found_done_chunk = false;
    while let Some(event) = event_source.next().await {
        let event = event.unwrap();
        match event {
            Event::Open => continue,
            Event::Message(message) => {
                if message.data == "[DONE]" {
                    found_done_chunk = true;
                    break;
                }
                chunks.push(message.data);
            }
        }
    }
    assert!(found_done_chunk);

    let mut inference_id = None;
    let mut tool_id: Option<String> = None;
    let mut arguments = String::new();
    let mut input_tokens = 0;
    let mut output_tokens = 0;

    for chunk in chunks {
        let chunk_json: Value = serde_json::from_str(&chunk).unwrap();

        println!("API response chunk: {chunk_json:#?}");

        let chunk_inference_id = chunk_json.get("inference_id").unwrap().as_str().unwrap();
        let chunk_inference_id = Uuid::parse_str(chunk_inference_id).unwrap();
        match inference_id {
            None => inference_id = Some(chunk_inference_id),
            Some(inference_id) => assert_eq!(inference_id, chunk_inference_id),
        }

        let chunk_episode_id = chunk_json.get("episode_id").unwrap().as_str().unwrap();
        let chunk_episode_id = Uuid::parse_str(chunk_episode_id).unwrap();
        assert_eq!(chunk_episode_id, episode_id);

        for block in chunk_json.get("content").unwrap().as_array().unwrap() {
            assert!(block.get("id").is_some());

            let block_type = block.get("type").unwrap().as_str().unwrap();

            match block_type {
                "tool_call" => {
                    assert_eq!(
                        block.get("raw_name").unwrap().as_str().unwrap(),
                        "self_destruct"
                    );

                    let block_tool_id = block.get("id").unwrap().as_str().unwrap();
                    match &tool_id {
                        None => tool_id = Some(block_tool_id.to_string()),
                        Some(tool_id) => assert_eq!(
                            tool_id, block_tool_id,
                            "Provider returned multiple tool calls"
                        ),
                    }

                    let chunk_arguments = block.get("raw_arguments").unwrap().as_str().unwrap();
                    arguments.push_str(chunk_arguments);
                }
                "text" => {
                    // Sometimes the model will also return some text
                    // (e.g. "Sure, here's the weather in Tokyo:" + tool call)
                    // We mostly care about the tool call, so we'll ignore the text.
                }
                _ => {
                    panic!("Unexpected block type: {}", block_type);
                }
            }
        }

        if let Some(usage) = chunk_json.get("usage").and_then(|u| u.as_object()) {
            input_tokens += usage.get("input_tokens").unwrap().as_u64().unwrap();
            output_tokens += usage.get("output_tokens").unwrap().as_u64().unwrap();
        }
    }

    // NB: Azure doesn't return usage during streaming
    if provider.variant_name.contains("azure") {
        assert_eq!(input_tokens, 0);
        assert_eq!(output_tokens, 0);
    } else {
        assert!(input_tokens > 0);
        assert!(output_tokens > 0);
    }

    let inference_id = inference_id.unwrap();
    let tool_id = tool_id.unwrap();
    assert!(
        serde_json::from_str::<Value>(&arguments).is_ok(),
        "Arguments: {arguments}"
    );

    // Sleep for 1 second to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ClickHouse - ChatInference Table
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ChatInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, "weather_helper");

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let episode_id_result = result.get("episode_id").unwrap().as_str().unwrap();
    let episode_id_result = Uuid::parse_str(episode_id_result).unwrap();
    assert_eq!(episode_id_result, episode_id);

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input: Value = json!(
        {
            "system": {
                "assistant_name": "Dr. Mehta"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "value": "What is the temperature like in Tokyo (in Celsius)? Use the `get_temperature` tool."}]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);

    let output_clickhouse: Vec<Value> =
        serde_json::from_str(result.get("output").unwrap().as_str().unwrap()).unwrap();
    assert!(!output_clickhouse.is_empty()); // could be > 1 if the model returns text as well
    let content_block = output_clickhouse.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "tool_call");
    assert_eq!(content_block.get("id").unwrap().as_str().unwrap(), tool_id);
    assert_eq!(
        content_block.get("raw_name").unwrap().as_str().unwrap(),
        "self_destruct"
    );
    assert_eq!(
        content_block
            .get("raw_arguments")
            .unwrap()
            .as_str()
            .unwrap(),
        arguments
    );
    assert_eq!(
        content_block.get("name").unwrap().as_str().unwrap(),
        "self_destruct"
    );
    assert_eq!(
        content_block
            .get("raw_arguments")
            .unwrap()
            .as_str()
            .unwrap(),
        arguments
    );

    let tool_params: Value =
        serde_json::from_str(result.get("tool_params").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(
        tool_params["tool_choice"],
        json!({"specific": "self_destruct"})
    );
    assert_eq!(tool_params["parallel_tool_calls"], false);

    let tools_available = tool_params["tools_available"].as_array().unwrap();
    assert_eq!(tools_available.len(), 2);
    let tool = tools_available
        .iter()
        .find(|t| t["name"] == "get_temperature")
        .unwrap();
    assert_eq!(tool["name"], "get_temperature");
    assert_eq!(
        tool["description"],
        "Get the current temperature in a given location"
    );
    assert_eq!(tool["strict"], false);

    let tool_parameters = tool["parameters"].as_object().unwrap();
    assert_eq!(tool_parameters["type"], "object");
    assert!(tool_parameters.get("properties").is_some());
    assert!(tool_parameters.get("required").is_some());
    assert_eq!(tool_parameters["additionalProperties"], false);

    let properties = tool_parameters["properties"].as_object().unwrap();
    assert!(properties.contains_key("location"));
    assert!(properties.contains_key("units"));

    let location = properties["location"].as_object().unwrap();
    assert_eq!(location["type"], "string");
    assert_eq!(
        location["description"],
        "The location to get the temperature for (e.g. \"New York\")"
    );

    let units = properties["units"].as_object().unwrap();
    assert_eq!(units["type"], "string");
    assert_eq!(
        units["description"],
        "The units to get the temperature in (must be \"fahrenheit\" or \"celsius\")"
    );
    let units_enum = units["enum"].as_array().unwrap();
    assert_eq!(units_enum.len(), 2);
    assert!(units_enum.contains(&json!("fahrenheit")));
    assert!(units_enum.contains(&json!("celsius")));

    let required = tool_parameters["required"].as_array().unwrap();
    assert!(required.contains(&json!("location")));

    let tool = tools_available
        .iter()
        .find(|t| t["name"] == "self_destruct")
        .unwrap();

    assert_eq!(
        tool["description"],
        "Do not call this function under any circumstances."
    );
    assert_eq!(tool["strict"], false);

    let tool_parameters = tool["parameters"].as_object().unwrap();
    assert_eq!(tool_parameters["type"], "object");
    assert!(tool_parameters.get("properties").is_some());
    assert!(tool_parameters
        .get("required")
        .unwrap()
        .as_array()
        .unwrap()
        .contains(&json!("fast")));
    assert_eq!(tool_parameters["additionalProperties"], false);

    let properties = tool_parameters["properties"].as_object().unwrap();
    assert!(properties.contains_key("fast"));

    // Check if ClickHouse is correct - ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(id).is_ok());

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.contains("self_destruct"));
    assert!(raw_request.to_lowercase().contains("tokyo"));
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );

    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    assert!(raw_response.contains("self_destruct"));
    // Check if raw_response is valid JSONL
    for line in raw_response.lines() {
        assert!(serde_json::from_str::<Value>(line).is_ok());
    }

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();

    // NB: Azure doesn't support input/output tokens during streaming
    if provider.variant_name.contains("azure") {
        assert_eq!(input_tokens, 0);
        assert_eq!(output_tokens, 0);
    } else {
        assert!(input_tokens > 0);
        assert!(output_tokens > 0);
    }

    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);

    let ttft_ms = result.get("ttft_ms").unwrap().as_u64().unwrap();
    assert!(ttft_ms > 50);
    assert!(ttft_ms <= response_time_ms);

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta.\n\nPeople will ask you questions about the weather.\n\nIf asked about the weather, just respond with the tool call. Use the \"get_temperature\" tool.\n\nIf provided with a tool result, use it to respond to the user (e.g. \"The weather in New York is 55 degrees Fahrenheit.\")."
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![RequestMessage {
        role: Role::User,
        content: vec![
            "What is the temperature like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                .to_string()
                .into(),
        ],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    let tool_call_blocks: Vec<_> = output
        .iter()
        .filter(|block| matches!(block, ContentBlock::ToolCall(_)))
        .collect();

    // Assert exactly one tool call
    assert_eq!(tool_call_blocks.len(), 1, "Expected exactly one tool call");

    let tool_call_block = tool_call_blocks[0];
    match tool_call_block {
        ContentBlock::ToolCall(tool_call) => {
            assert_eq!(tool_call.name, "self_destruct");
            serde_json::from_str::<Value>(&tool_call.arguments.to_lowercase()).unwrap();
        }
        _ => panic!("Unreachable"),
    }
}

pub async fn test_tool_use_allowed_tools_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    // GCP Vertex doesn't support `tool_choice: "required"` for Gemini 1.5 Flash,
    // and it won't call `get_humidity` on auto.
    if provider.model_provider_name.contains("gcp_vertex")
        && provider.model_name == "gemini-1.5-flash-001"
    {
        return;
    }

    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather like in Tokyo? Call a function."
                }
            ]},
        "tool_choice": "required",
        "allowed_tools": ["get_humidity"],
        "stream": false,
        "variant_name": provider.variant_name,
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check if the API response is fine
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    println!("API response: {response_json:#?}");

    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    let episode_id_response = response_json.get("episode_id").unwrap().as_str().unwrap();
    let episode_id_response = Uuid::parse_str(episode_id_response).unwrap();
    assert_eq!(episode_id_response, episode_id);

    let variant_name = response_json.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let content = response_json.get("content").unwrap().as_array().unwrap();
    assert!(!content.is_empty()); // could be > 1 if the model returns text as well
    let content_block = content
        .iter()
        .find(|block| block["type"] == "tool_call")
        .unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "tool_call");

    assert!(content_block.get("id").unwrap().as_str().is_some());

    let raw_name = content_block.get("raw_name").unwrap().as_str().unwrap();
    assert_eq!(raw_name, "get_humidity");
    let name = content_block.get("name").unwrap().as_str().unwrap();
    assert_eq!(name, "get_humidity");

    let raw_arguments = content_block
        .get("raw_arguments")
        .unwrap()
        .as_str()
        .unwrap();
    let raw_arguments: Value = serde_json::from_str(raw_arguments).unwrap();
    let raw_arguments = raw_arguments.as_object().unwrap();
    assert!(raw_arguments.len() == 1);
    assert!(raw_arguments.get("location").unwrap().as_str().is_some());

    let arguments = content_block.get("arguments").unwrap();
    let arguments = arguments.as_object().unwrap();
    assert!(arguments.len() == 1);
    assert!(arguments.get("location").unwrap().as_str().is_some());

    let usage = response_json.get("usage").unwrap();
    let usage = usage.as_object().unwrap();
    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    assert!(output_tokens > 0);

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check if ClickHouse is correct - ChatInference table
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ChatInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, "basic_test");

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let episode_id_result = result.get("episode_id").unwrap().as_str().unwrap();
    let episode_id_result = Uuid::parse_str(episode_id_result).unwrap();
    assert_eq!(episode_id_result, episode_id);

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input: Value = json!(
        {
            "system": {
                "assistant_name": "Dr. Mehta"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "value": "What is the weather like in Tokyo? Call a function."}]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);

    let output_clickhouse: Vec<Value> =
        serde_json::from_str(result.get("output").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(output_clickhouse, *content);

    let tool_params: Value =
        serde_json::from_str(result.get("tool_params").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(tool_params["tool_choice"], "required");
    assert_eq!(tool_params["parallel_tool_calls"], false);

    let tools_available = tool_params["tools_available"].as_array().unwrap();
    assert_eq!(tools_available.len(), 1);

    let tool = tools_available
        .iter()
        .find(|tool| tool["name"] == "get_humidity")
        .unwrap();
    assert_eq!(
        tool["description"],
        "Get the current humidity in a given location"
    );
    assert_eq!(tool["strict"], false);

    let tool_parameters = tool["parameters"].as_object().unwrap();
    assert_eq!(tool_parameters["type"], "object");
    assert!(tool_parameters.get("properties").is_some());
    assert!(tool_parameters
        .get("required")
        .unwrap()
        .as_array()
        .unwrap()
        .contains(&json!("location")));
    assert_eq!(tool_parameters["additionalProperties"], false);

    let properties = tool_parameters["properties"].as_object().unwrap();
    println!("Properties: {properties:#?}");
    assert!(properties.get("location").is_some());

    // Check if ClickHouse is correct - ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(id).is_ok());

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.contains("get_humidity"));
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );

    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    assert!(raw_response.contains("get_humidity"));

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(output_tokens > 0);
    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);
    assert!(result.get("ttft_ms").unwrap().is_null());

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta"
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![RequestMessage {
        role: Role::User,
        content: vec!["What is the weather like in Tokyo? Call a function."
            .to_string()
            .into()],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    let tool_call_blocks: Vec<_> = output
        .iter()
        .filter(|block| matches!(block, ContentBlock::ToolCall(_)))
        .collect();

    // Assert exactly one tool call
    assert_eq!(tool_call_blocks.len(), 1, "Expected exactly one tool call");

    let tool_call_block = tool_call_blocks[0];
    match tool_call_block {
        ContentBlock::ToolCall(tool_call) => {
            assert_eq!(tool_call.name, "get_humidity");
            serde_json::from_str::<Value>(&tool_call.arguments.to_lowercase()).unwrap();
        }
        _ => panic!("Unreachable"),
    }
}

pub async fn test_tool_use_allowed_tools_streaming_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    // GCP Vertex doesn't support `tool_choice: "required"` for Gemini 1.5 Flash,
    // and it won't call `get_humidity` on auto.
    if provider.model_provider_name.contains("gcp_vertex")
        && provider.model_name == "gemini-1.5-flash-001"
    {
        return;
    }

    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather like in Tokyo? Call a function."
                }
            ]},
        "tool_choice": "required",
        "allowed_tools": ["get_humidity"],
        "stream": true,
        "variant_name": provider.variant_name,
    });

    let mut event_source = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .unwrap();

    let mut chunks = vec![];
    let mut found_done_chunk = false;
    while let Some(event) = event_source.next().await {
        let event = event.unwrap();
        match event {
            Event::Open => continue,
            Event::Message(message) => {
                if message.data == "[DONE]" {
                    found_done_chunk = true;
                    break;
                }
                chunks.push(message.data);
            }
        }
    }
    assert!(found_done_chunk);

    let mut inference_id = None;
    let mut tool_id: Option<String> = None;
    let mut arguments = String::new();
    let mut input_tokens = 0;
    let mut output_tokens = 0;

    for chunk in chunks {
        let chunk_json: Value = serde_json::from_str(&chunk).unwrap();

        println!("API response chunk: {chunk_json:#?}");

        let chunk_inference_id = chunk_json.get("inference_id").unwrap().as_str().unwrap();
        let chunk_inference_id = Uuid::parse_str(chunk_inference_id).unwrap();
        match inference_id {
            None => inference_id = Some(chunk_inference_id),
            Some(inference_id) => assert_eq!(inference_id, chunk_inference_id),
        }

        let chunk_episode_id = chunk_json.get("episode_id").unwrap().as_str().unwrap();
        let chunk_episode_id = Uuid::parse_str(chunk_episode_id).unwrap();
        assert_eq!(chunk_episode_id, episode_id);

        for block in chunk_json.get("content").unwrap().as_array().unwrap() {
            assert!(block.get("id").is_some());

            let block_type = block.get("type").unwrap().as_str().unwrap();

            match block_type {
                "tool_call" => {
                    assert_eq!(
                        block.get("raw_name").unwrap().as_str().unwrap(),
                        "get_humidity"
                    );

                    let block_tool_id = block.get("id").unwrap().as_str().unwrap();
                    match &tool_id {
                        None => tool_id = Some(block_tool_id.to_string()),
                        Some(tool_id) => assert_eq!(tool_id, block_tool_id),
                    }

                    let chunk_arguments = block.get("raw_arguments").unwrap().as_str().unwrap();
                    arguments.push_str(chunk_arguments);
                }
                "text" => {
                    // Sometimes the model will also return some text
                    // (e.g. "Sure, here's the weather in Tokyo:" + tool call)
                    // We mostly care about the tool call, so we'll ignore the text.
                }
                _ => {
                    panic!("Unexpected block type: {}", block_type);
                }
            }
        }

        if let Some(usage) = chunk_json.get("usage").and_then(|u| u.as_object()) {
            input_tokens += usage.get("input_tokens").unwrap().as_u64().unwrap();
            output_tokens += usage.get("output_tokens").unwrap().as_u64().unwrap();
        }
    }

    // NB: Azure doesn't return usage during streaming
    if provider.variant_name.contains("azure") {
        assert_eq!(input_tokens, 0);
        assert_eq!(output_tokens, 0);
    } else {
        assert!(input_tokens > 0);
        assert!(output_tokens > 0);
    }

    let inference_id = inference_id.unwrap();
    let tool_id = tool_id.unwrap();
    assert!(serde_json::from_str::<Value>(&arguments).is_ok());

    // Sleep for 1 second to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ClickHouse - ChatInference Table
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ChatInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, "basic_test");

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let episode_id_result = result.get("episode_id").unwrap().as_str().unwrap();
    let episode_id_result = Uuid::parse_str(episode_id_result).unwrap();
    assert_eq!(episode_id_result, episode_id);

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input: Value = json!(
        {
            "system": {
                "assistant_name": "Dr. Mehta"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "value": "What is the weather like in Tokyo? Call a function."}]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);

    let output_clickhouse: Vec<Value> =
        serde_json::from_str(result.get("output").unwrap().as_str().unwrap()).unwrap();
    assert!(!output_clickhouse.is_empty()); // could be > 1 if the model returns text as well
    let content_block = output_clickhouse.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "tool_call");
    assert_eq!(content_block.get("id").unwrap().as_str().unwrap(), tool_id);
    assert_eq!(
        content_block.get("raw_name").unwrap().as_str().unwrap(),
        "get_humidity"
    );
    assert_eq!(
        content_block
            .get("raw_arguments")
            .unwrap()
            .as_str()
            .unwrap(),
        arguments
    );
    assert_eq!(
        content_block.get("name").unwrap().as_str().unwrap(),
        "get_humidity"
    );
    assert_eq!(
        content_block.get("arguments").unwrap().as_object().unwrap(),
        &serde_json::from_str::<serde_json::Map<String, Value>>(&arguments).unwrap()
    );

    let tool_params: Value =
        serde_json::from_str(result.get("tool_params").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(tool_params["tool_choice"], "required");
    assert_eq!(tool_params["parallel_tool_calls"], false);

    let tools_available = tool_params["tools_available"].as_array().unwrap();
    assert_eq!(tools_available.len(), 1);
    let tool = tools_available.first().unwrap();
    assert_eq!(tool["name"], "get_humidity");
    assert_eq!(
        tool["description"],
        "Get the current humidity in a given location"
    );
    assert_eq!(tool["strict"], false);

    let tool_parameters = tool["parameters"].as_object().unwrap();
    assert_eq!(tool_parameters["type"], "object");
    assert!(tool_parameters.get("properties").is_some());
    assert!(tool_parameters.get("required").is_some());
    assert_eq!(tool_parameters["additionalProperties"], false);

    let properties = tool_parameters["properties"].as_object().unwrap();
    assert!(properties.contains_key("location"));

    let location = properties["location"].as_object().unwrap();
    assert_eq!(location["type"], "string");
    assert_eq!(
        location["description"],
        "The location to get the humidity for (e.g. \"New York\")"
    );

    let required = tool_parameters["required"].as_array().unwrap();
    assert!(required.contains(&json!("location")));

    // Check if ClickHouse is correct - ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(id).is_ok());

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.contains("get_humidity"));
    assert!(raw_request.to_lowercase().contains("tokyo"));
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );

    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    assert!(raw_response.contains("get_humidity"));
    // Check if raw_response is valid JSONL
    for line in raw_response.lines() {
        assert!(serde_json::from_str::<Value>(line).is_ok());
    }

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();

    // NB: Azure doesn't support input/output tokens during streaming
    if provider.variant_name.contains("azure") {
        assert_eq!(input_tokens, 0);
        assert_eq!(output_tokens, 0);
    } else {
        assert!(input_tokens > 0);
        assert!(output_tokens > 0);
    }

    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);

    let ttft_ms = result.get("ttft_ms").unwrap().as_u64().unwrap();
    assert!(ttft_ms > 50);
    assert!(ttft_ms <= response_time_ms);

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta"
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![RequestMessage {
        role: Role::User,
        content: vec!["What is the weather like in Tokyo? Call a function."
            .to_string()
            .into()],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    let tool_call_blocks: Vec<_> = output
        .iter()
        .filter(|block| matches!(block, ContentBlock::ToolCall(_)))
        .collect();

    // Assert exactly one tool call
    assert_eq!(tool_call_blocks.len(), 1, "Expected exactly one tool call");

    let tool_call_block = tool_call_blocks[0];
    match tool_call_block {
        ContentBlock::ToolCall(tool_call) => {
            assert_eq!(tool_call.name, "get_humidity");
            serde_json::from_str::<Value>(&tool_call.arguments.to_lowercase()).unwrap();
        }
        _ => panic!("Unreachable"),
    }
}

pub async fn test_tool_multi_turn_inference_request_with_provider(provider: E2ETestProvider) {
    let episode_id = Uuid::now_v7();

    let payload = json!({
       "function_name": "weather_helper",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "id": "123456789",
                            "name": "get_temperature",
                            "arguments": "{\"location\": \"Tokyo\"}"
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "id": "123456789",
                            "name": "get_temperature",
                            "result": "70"
                        }
                    ]
                }
            ]},
        "variant_name": provider.variant_name,
        "stream": false,
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    println!("API response: {response_json:#?}");

    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    let episode_id_response = response_json.get("episode_id").unwrap().as_str().unwrap();
    let episode_id_response = Uuid::parse_str(episode_id_response).unwrap();
    assert_eq!(episode_id_response, episode_id);

    let variant_name = response_json.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let content = response_json.get("content").unwrap().as_array().unwrap();
    assert_eq!(content.len(), 1);
    let content_block = content.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let content = content_block.get("text").unwrap().as_str().unwrap();
    assert!(content.to_lowercase().contains("tokyo"));

    let usage = response_json.get("usage").unwrap();
    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(output_tokens > 0);

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check if ClickHouse is ok - ChatInference Table
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ChatInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id = Uuid::parse_str(id).unwrap();
    assert_eq!(id, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, "weather_helper");

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input = json!({
        "system": {"assistant_name": "Dr. Mehta"},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "value": "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."}]
            },
            {
                "role": "assistant",
                "content": [{"type": "tool_call", "id": "123456789", "name": "get_temperature", "arguments": "{\"location\": \"Tokyo\"}"}]
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "id": "123456789", "name": "get_temperature", "result": "70"}]
            }
        ]
    });
    assert_eq!(input, correct_input);

    let content_blocks = result.get("output").unwrap().as_str().unwrap();
    let content_blocks: Vec<Value> = serde_json::from_str(content_blocks).unwrap();
    assert_eq!(content_blocks.len(), 1);
    let content_block = content_blocks.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let clickhouse_content = content_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(clickhouse_content, content);

    let tool_params: Value =
        serde_json::from_str(result.get("tool_params").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(tool_params["tool_choice"], "auto");
    assert_eq!(tool_params["parallel_tool_calls"], false);

    let tools_available = tool_params["tools_available"].as_array().unwrap();
    assert_eq!(tools_available.len(), 1);
    let tool = tools_available.first().unwrap();
    assert_eq!(tool["name"], "get_temperature");
    assert_eq!(
        tool["description"],
        "Get the current temperature in a given location"
    );
    assert_eq!(tool["strict"], false);

    let inference_params = result.get("inference_params").unwrap().as_str().unwrap();
    let inference_params: Value = serde_json::from_str(inference_params).unwrap();
    let inference_params = inference_params.get("chat_completion").unwrap();
    assert!(inference_params.get("temperature").is_none());
    assert!(inference_params.get("seed").is_none());
    assert_eq!(
        inference_params
            .get("max_tokens")
            .unwrap()
            .as_u64()
            .unwrap(),
        100
    );

    let processing_time_ms = result.get("processing_time_ms").unwrap().as_u64().unwrap();
    assert!(processing_time_ms > 0);

    // Check the ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let model_inference_id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(model_inference_id).is_ok());

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.contains("get_temperature"));
    assert!(raw_request.to_lowercase().contains("tokyo"));
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );

    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    assert!(raw_response.to_lowercase().contains("tokyo"));
    assert!(serde_json::from_str::<Value>(raw_response).is_ok());

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(output_tokens > 0);
    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);
    assert!(result.get("ttft_ms").unwrap().is_null());

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta.\n\nPeople will ask you questions about the weather.\n\nIf asked about the weather, just respond with the tool call. Use the \"get_temperature\" tool.\n\nIf provided with a tool result, use it to respond to the user (e.g. \"The weather in New York is 55 degrees Fahrenheit.\")."
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![
        RequestMessage {
            role: Role::User,
            content: vec![
                "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                    .to_string()
                    .into(),
            ],
        },
        RequestMessage {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolCall(ToolCall {
                id: "123456789".to_string(),
                name: "get_temperature".to_string(),
                arguments: "{\"location\": \"Tokyo\"}".to_string(),
            })],
        },
        RequestMessage {
            role: Role::User,
            content: vec![ContentBlock::ToolResult(ToolResult {
                id: "123456789".to_string(),
                name: "get_temperature".to_string(),
                result: "70".to_string(),
            })],
        },
    ];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    assert_eq!(output.len(), 1);
    let first = output.first().unwrap();
    match first {
        ContentBlock::Text(text) => {
            assert!(text.text.to_lowercase().contains("tokyo"));
        }
        _ => {
            panic!("Expected a text block, got {:?}", first);
        }
    }
}

pub async fn test_tool_multi_turn_streaming_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    let episode_id = Uuid::now_v7();

    let payload = json!({
       "function_name": "weather_helper",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "id": "123456789",
                            "name": "get_temperature",
                            "arguments": "{\"location\": \"Tokyo\"}"
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "id": "123456789",
                            "name": "get_temperature",
                            "result": "70"
                        }
                    ]
                }
            ]},
        "variant_name": provider.variant_name,
        "stream": true,
    });

    let mut event_source = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .unwrap();

    let mut chunks = vec![];
    let mut found_done_chunk = false;
    while let Some(event) = event_source.next().await {
        let event = event.unwrap();
        match event {
            Event::Open => continue,
            Event::Message(message) => {
                if message.data == "[DONE]" {
                    found_done_chunk = true;
                    break;
                }
                chunks.push(message.data);
            }
        }
    }
    assert!(found_done_chunk);

    let mut inference_id: Option<Uuid> = None;
    let mut full_content = String::new();
    let mut input_tokens = 0;
    let mut output_tokens = 0;
    for chunk in chunks.clone() {
        let chunk_json: Value = serde_json::from_str(&chunk).unwrap();

        println!("API response chunk: {chunk_json:#?}");

        let chunk_inference_id = chunk_json.get("inference_id").unwrap().as_str().unwrap();
        let chunk_inference_id = Uuid::parse_str(chunk_inference_id).unwrap();
        match inference_id {
            Some(inference_id) => {
                assert_eq!(inference_id, chunk_inference_id);
            }
            None => {
                inference_id = Some(chunk_inference_id);
            }
        }

        let chunk_episode_id = chunk_json.get("episode_id").unwrap().as_str().unwrap();
        let chunk_episode_id = Uuid::parse_str(chunk_episode_id).unwrap();
        assert_eq!(chunk_episode_id, episode_id);

        let content_blocks = chunk_json.get("content").unwrap().as_array().unwrap();
        if !content_blocks.is_empty() {
            let content_block = content_blocks.first().unwrap();
            let content = content_block.get("text").unwrap().as_str().unwrap();
            full_content.push_str(content);
        }

        if let Some(usage) = chunk_json.get("usage") {
            input_tokens += usage.get("input_tokens").unwrap().as_u64().unwrap();
            output_tokens += usage.get("output_tokens").unwrap().as_u64().unwrap();
        }
    }

    let inference_id = inference_id.unwrap();
    assert!(full_content.to_lowercase().contains("tokyo"));

    // NB: Azure doesn't support input/output tokens during streaming
    if provider.variant_name.contains("azure") {
        assert_eq!(input_tokens, 0);
        assert_eq!(output_tokens, 0);
    } else {
        assert!(input_tokens > 0);
        assert!(output_tokens > 0);
    }

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ClickHouse - ChatInference Table
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ChatInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, "weather_helper");

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let episode_id_result = result.get("episode_id").unwrap().as_str().unwrap();
    let episode_id_result = Uuid::parse_str(episode_id_result).unwrap();
    assert_eq!(episode_id_result, episode_id);

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input = json!({
        "system": {"assistant_name": "Dr. Mehta"},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "value": "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."}]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "id": "123456789",
                        "name": "get_temperature",
                        "arguments": "{\"location\": \"Tokyo\"}"
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "id": "123456789",
                        "name": "get_temperature",
                        "result": "70"
                    }
                ]
            }
        ]
    });
    assert_eq!(input, correct_input);

    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<Value> = serde_json::from_str(output).unwrap();
    assert_eq!(output.len(), 1);
    let content_block = output.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let clickhouse_content = content_block.get("text").unwrap().as_str().unwrap();
    assert_eq!(clickhouse_content, full_content);

    let tool_params: Value =
        serde_json::from_str(result.get("tool_params").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(tool_params["tool_choice"], "auto");
    assert_eq!(tool_params["parallel_tool_calls"], false);

    let inference_params = result.get("inference_params").unwrap().as_str().unwrap();
    let inference_params: Value = serde_json::from_str(inference_params).unwrap();
    let inference_params = inference_params.get("chat_completion").unwrap();
    assert!(inference_params.get("temperature").is_none());
    assert!(inference_params.get("seed").is_none());
    assert_eq!(
        inference_params
            .get("max_tokens")
            .unwrap()
            .as_u64()
            .unwrap(),
        100
    );

    let processing_time_ms = result.get("processing_time_ms").unwrap().as_u64().unwrap();
    assert!(processing_time_ms > 0);

    // Check ClickHouse - ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let model_inference_id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(model_inference_id).is_ok());

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.contains("get_temperature"));
    assert!(raw_request.to_lowercase().contains("tokyo"));
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );

    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();

    // Check if raw_response is valid JSONL
    for line in raw_response.lines() {
        assert!(serde_json::from_str::<Value>(line).is_ok());
    }

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();

    // NB: Azure doesn't support input/output tokens during streaming
    if provider.variant_name.contains("azure") {
        assert_eq!(input_tokens, 0);
        assert_eq!(output_tokens, 0);
    } else {
        assert!(input_tokens > 0);
        assert!(output_tokens > 0);
    }

    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);

    let ttft_ms = result.get("ttft_ms").unwrap().as_u64().unwrap();
    assert!(ttft_ms > 50);
    assert!(ttft_ms <= response_time_ms);

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta.\n\nPeople will ask you questions about the weather.\n\nIf asked about the weather, just respond with the tool call. Use the \"get_temperature\" tool.\n\nIf provided with a tool result, use it to respond to the user (e.g. \"The weather in New York is 55 degrees Fahrenheit.\")."
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![
        RequestMessage {
            role: Role::User,
            content: vec![
                "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                    .to_string()
                    .into(),
            ],
        },
        RequestMessage {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolCall(ToolCall {
                id: "123456789".to_string(),
                name: "get_temperature".to_string(),
                arguments: "{\"location\": \"Tokyo\"}".to_string(),
            })],
        },
        RequestMessage {
            role: Role::User,
            content: vec![ContentBlock::ToolResult(ToolResult {
                id: "123456789".to_string(),
                name: "get_temperature".to_string(),
                result: "70".to_string(),
            })],
        },
    ];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    assert_eq!(output.len(), 1);
    let first = output.first().unwrap();
    match first {
        ContentBlock::Text(text) => {
            assert!(text.text.to_lowercase().contains("tokyo"));
        }
        _ => {
            panic!("Expected a text block, got {:?}", first);
        }
    }
}

pub async fn test_dynamic_tool_use_inference_request_with_provider(provider: E2ETestProvider) {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather like in Tokyo (in Celsius)? Use the provided `get_temperature` tool. Do not say anything else, just call the function."
                }
            ]},
        "stream": false,
        "additional_tools": [
            {
                "name": "get_temperature",
                "description": "Get the current temperature in a given location",
                "parameters": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The location to get the temperature for (e.g. \"New York\")"
                        },
                        "units": {
                            "type": "string",
                            "description": "The units to get the temperature in (must be \"fahrenheit\" or \"celsius\")",
                            "enum": ["fahrenheit", "celsius"]
                        }
                    },
                    "required": ["location"],
                    "additionalProperties": false
                }
            }
        ],
        "variant_name": provider.variant_name,
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check if the API response is fine
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    println!("API response: {response_json:#?}");

    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    let episode_id_response = response_json.get("episode_id").unwrap().as_str().unwrap();
    let episode_id_response = Uuid::parse_str(episode_id_response).unwrap();
    assert_eq!(episode_id_response, episode_id);

    let variant_name = response_json.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let content = response_json.get("content").unwrap().as_array().unwrap();
    assert!(!content.is_empty()); // could be > 1 if the model returns text as well
    let content_block = content
        .iter()
        .find(|block| block["type"] == "tool_call")
        .unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "tool_call");

    assert!(content_block.get("id").unwrap().as_str().is_some());

    let raw_name = content_block.get("raw_name").unwrap().as_str().unwrap();
    assert_eq!(raw_name, "get_temperature");
    let name = content_block.get("name").unwrap().as_str().unwrap();
    assert_eq!(name, "get_temperature");

    let raw_arguments = content_block
        .get("raw_arguments")
        .unwrap()
        .as_str()
        .unwrap();
    let raw_arguments: Value = serde_json::from_str(raw_arguments).unwrap();
    let raw_arguments = raw_arguments.as_object().unwrap();
    assert!(raw_arguments.len() == 2);
    let location = raw_arguments.get("location").unwrap().as_str().unwrap();
    assert_eq!(location.to_lowercase(), "tokyo");
    let units = raw_arguments.get("units").unwrap().as_str().unwrap();
    assert!(units == "celsius");

    let arguments = content_block.get("arguments").unwrap();
    let arguments = arguments.as_object().unwrap();
    assert!(arguments.len() == 2);
    let location = arguments.get("location").unwrap().as_str().unwrap();
    assert_eq!(location.to_lowercase(), "tokyo");
    let units = arguments.get("units").unwrap().as_str().unwrap();
    assert!(units == "celsius");

    let usage = response_json.get("usage").unwrap();
    let usage = usage.as_object().unwrap();
    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    assert!(output_tokens > 0);

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check if ClickHouse is correct - ChatInference table
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ChatInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, payload["function_name"]);

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let episode_id_result = result.get("episode_id").unwrap().as_str().unwrap();
    let episode_id_result = Uuid::parse_str(episode_id_result).unwrap();
    assert_eq!(episode_id_result, episode_id);

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input: Value = json!(
        {
            "system": {
                "assistant_name": "Dr. Mehta"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "value": "What is the weather like in Tokyo (in Celsius)? Use the provided `get_temperature` tool. Do not say anything else, just call the function."}]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);

    let output_clickhouse: Vec<Value> =
        serde_json::from_str(result.get("output").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(output_clickhouse, *content);

    let tool_params: Value =
        serde_json::from_str(result.get("tool_params").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(tool_params["tool_choice"], "auto");
    assert_eq!(tool_params["parallel_tool_calls"], false);

    let tools_available = tool_params["tools_available"].as_array().unwrap();
    assert_eq!(tools_available.len(), 1);
    let tool = tools_available.first().unwrap();
    assert_eq!(tool["name"], "get_temperature");
    assert_eq!(
        tool["description"],
        "Get the current temperature in a given location"
    );
    assert_eq!(tool["strict"], false);

    let tool_parameters = tool["parameters"].as_object().unwrap();
    assert_eq!(tool_parameters["type"], "object");
    assert!(tool_parameters.get("properties").is_some());
    assert!(tool_parameters.get("required").is_some());

    let properties = tool_parameters["properties"].as_object().unwrap();
    assert!(properties.contains_key("location"));
    assert!(properties.contains_key("units"));

    let location = properties["location"].as_object().unwrap();
    assert_eq!(location["type"], "string");
    assert_eq!(
        location["description"],
        "The location to get the temperature for (e.g. \"New York\")"
    );

    let units = properties["units"].as_object().unwrap();
    assert_eq!(units["type"], "string");
    assert_eq!(
        units["description"],
        "The units to get the temperature in (must be \"fahrenheit\" or \"celsius\")"
    );
    let units_enum = units["enum"].as_array().unwrap();
    assert_eq!(units_enum.len(), 2);
    assert!(units_enum.contains(&json!("fahrenheit")));
    assert!(units_enum.contains(&json!("celsius")));

    let required = tool_parameters["required"].as_array().unwrap();
    assert!(required.contains(&json!("location")));

    // Check if ClickHouse is correct - ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(id).is_ok());

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.contains("get_temperature"));
    assert!(raw_request.to_lowercase().contains("tokyo"));
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );

    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    assert!(raw_response.to_lowercase().contains("tokyo"));
    assert!(raw_response.contains("get_temperature"));

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(output_tokens > 0);
    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);
    assert!(result.get("ttft_ms").unwrap().is_null());

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta"
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![RequestMessage {
        role: Role::User,
        content: vec!["What is the weather like in Tokyo (in Celsius)? Use the provided `get_temperature` tool. Do not say anything else, just call the function."
            .to_string()
            .into()],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    let tool_call_blocks: Vec<_> = output
        .iter()
        .filter(|block| matches!(block, ContentBlock::ToolCall(_)))
        .collect();

    // Assert exactly one tool call
    assert_eq!(tool_call_blocks.len(), 1, "Expected exactly one tool call");

    let tool_call_block = tool_call_blocks[0];
    match tool_call_block {
        ContentBlock::ToolCall(tool_call) => {
            assert_eq!(tool_call.name, "get_temperature");
            serde_json::from_str::<Value>(&tool_call.arguments.to_lowercase()).unwrap();
        }
        _ => panic!("Unreachable"),
    }
}

pub async fn test_dynamic_tool_use_streaming_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather like in Tokyo (in Celsius)? Use the provided `get_temperature` tool. Do not say anything else, just call the function."
                }
            ]},
        "stream": true,
        "additional_tools": [
            {
                "name": "get_temperature",
                "description": "Get the current temperature in a given location",
                "parameters": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The location to get the temperature for (e.g. \"New York\")"
                        },
                        "units": {
                            "type": "string",
                            "description": "The units to get the temperature in (must be \"fahrenheit\" or \"celsius\")",
                            "enum": ["fahrenheit", "celsius"]
                        }
                    },
                    "required": ["location"],
                    "additionalProperties": false
                }
            }
        ],
        "variant_name": provider.variant_name,
    });

    let mut event_source = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .unwrap();

    let mut chunks = vec![];
    let mut found_done_chunk = false;
    while let Some(event) = event_source.next().await {
        let event = event.unwrap();
        match event {
            Event::Open => continue,
            Event::Message(message) => {
                if message.data == "[DONE]" {
                    found_done_chunk = true;
                    break;
                }
                chunks.push(message.data);
            }
        }
    }
    assert!(found_done_chunk);

    let mut inference_id = None;
    let mut tool_id: Option<String> = None;
    let mut arguments = String::new();
    let mut input_tokens = 0;
    let mut output_tokens = 0;

    for chunk in chunks {
        let chunk_json: Value = serde_json::from_str(&chunk).unwrap();

        println!("API response chunk: {chunk_json:#?}");

        let chunk_inference_id = chunk_json.get("inference_id").unwrap().as_str().unwrap();
        let chunk_inference_id = Uuid::parse_str(chunk_inference_id).unwrap();
        match inference_id {
            None => inference_id = Some(chunk_inference_id),
            Some(inference_id) => assert_eq!(inference_id, chunk_inference_id),
        }

        let chunk_episode_id = chunk_json.get("episode_id").unwrap().as_str().unwrap();
        let chunk_episode_id = Uuid::parse_str(chunk_episode_id).unwrap();
        assert_eq!(chunk_episode_id, episode_id);

        for block in chunk_json.get("content").unwrap().as_array().unwrap() {
            assert!(block.get("id").is_some());

            let block_type = block.get("type").unwrap().as_str().unwrap();

            match block_type {
                "tool_call" => {
                    assert_eq!(
                        block.get("raw_name").unwrap().as_str().unwrap(),
                        "get_temperature"
                    );

                    let block_tool_id = block.get("id").unwrap().as_str().unwrap();
                    match &tool_id {
                        None => tool_id = Some(block_tool_id.to_string()),
                        Some(tool_id) => assert_eq!(tool_id, block_tool_id),
                    }

                    let chunk_arguments = block.get("raw_arguments").unwrap().as_str().unwrap();
                    arguments.push_str(chunk_arguments);
                }
                "text" => {
                    // Sometimes the model will also return some text
                    // (e.g. "Sure, here's the weather in Tokyo:" + tool call)
                    // We mostly care about the tool call, so we'll ignore the text.
                }
                _ => {
                    panic!("Unexpected block type: {}", block_type);
                }
            }
        }

        if let Some(usage) = chunk_json.get("usage").and_then(|u| u.as_object()) {
            input_tokens += usage.get("input_tokens").unwrap().as_u64().unwrap();
            output_tokens += usage.get("output_tokens").unwrap().as_u64().unwrap();
        }
    }

    // NB: Azure doesn't return usage during streaming
    if provider.variant_name.contains("azure") {
        assert_eq!(input_tokens, 0);
        assert_eq!(output_tokens, 0);
    } else {
        assert!(input_tokens > 0);
        assert!(output_tokens > 0);
    }

    let inference_id = inference_id.unwrap();
    let tool_id = tool_id.unwrap();
    assert!(serde_json::from_str::<Value>(&arguments).is_ok());

    // Sleep for 1 second to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ClickHouse - ChatInference Table
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ChatInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, payload["function_name"]);

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let episode_id_result = result.get("episode_id").unwrap().as_str().unwrap();
    let episode_id_result = Uuid::parse_str(episode_id_result).unwrap();
    assert_eq!(episode_id_result, episode_id);

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input: Value = json!(
        {
            "system": {
                "assistant_name": "Dr. Mehta"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "value": "What is the weather like in Tokyo (in Celsius)? Use the provided `get_temperature` tool. Do not say anything else, just call the function."}]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);

    let output_clickhouse: Vec<Value> =
        serde_json::from_str(result.get("output").unwrap().as_str().unwrap()).unwrap();
    assert!(!output_clickhouse.is_empty()); // could be > 1 if the model returns text as well
    let content_block = output_clickhouse.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "tool_call");
    assert_eq!(content_block.get("id").unwrap().as_str().unwrap(), tool_id);
    assert_eq!(
        content_block.get("raw_name").unwrap().as_str().unwrap(),
        "get_temperature"
    );
    assert_eq!(
        content_block.get("name").unwrap().as_str().unwrap(),
        "get_temperature"
    );
    assert_eq!(
        content_block
            .get("raw_arguments")
            .unwrap()
            .as_str()
            .unwrap(),
        arguments
    );
    assert_eq!(
        content_block.get("arguments").unwrap().as_object().unwrap(),
        &serde_json::from_str::<serde_json::Map<String, Value>>(&arguments).unwrap()
    );

    let tool_params: Value =
        serde_json::from_str(result.get("tool_params").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(tool_params["tool_choice"], "auto");
    assert_eq!(tool_params["parallel_tool_calls"], false);

    let tools_available = tool_params["tools_available"].as_array().unwrap();
    assert_eq!(tools_available.len(), 1);
    let tool = tools_available.first().unwrap();
    assert_eq!(tool["name"], "get_temperature");
    assert_eq!(
        tool["description"],
        "Get the current temperature in a given location"
    );
    assert_eq!(tool["strict"], false);

    let tool_parameters = tool["parameters"].as_object().unwrap();
    assert_eq!(tool_parameters["type"], "object");
    assert!(tool_parameters.get("properties").is_some());
    assert!(tool_parameters.get("required").is_some());

    let properties = tool_parameters["properties"].as_object().unwrap();
    assert!(properties.contains_key("location"));
    assert!(properties.contains_key("units"));

    let location = properties["location"].as_object().unwrap();
    assert_eq!(location["type"], "string");
    assert_eq!(
        location["description"],
        "The location to get the temperature for (e.g. \"New York\")"
    );

    let units = properties["units"].as_object().unwrap();
    assert_eq!(units["type"], "string");
    assert_eq!(
        units["description"],
        "The units to get the temperature in (must be \"fahrenheit\" or \"celsius\")"
    );
    let units_enum = units["enum"].as_array().unwrap();
    assert_eq!(units_enum.len(), 2);
    assert!(units_enum.contains(&json!("fahrenheit")));
    assert!(units_enum.contains(&json!("celsius")));

    let required = tool_parameters["required"].as_array().unwrap();
    assert!(required.contains(&json!("location")));

    // Check if ClickHouse is correct - ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(id).is_ok());

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.contains("get_temperature"));
    assert!(raw_request.to_lowercase().contains("tokyo"));
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );

    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    assert!(raw_response.contains("get_temperature"));
    // Check if raw_response is valid JSONL
    for line in raw_response.lines() {
        assert!(serde_json::from_str::<Value>(line).is_ok());
    }

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();

    // NB: Azure doesn't support input/output tokens during streaming
    if provider.variant_name.contains("azure") {
        assert_eq!(input_tokens, 0);
        assert_eq!(output_tokens, 0);
    } else {
        assert!(input_tokens > 0);
        assert!(output_tokens > 0);
    }

    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);

    let ttft_ms = result.get("ttft_ms").unwrap().as_u64().unwrap();
    assert!(ttft_ms > 50);
    assert!(ttft_ms <= response_time_ms);

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta"
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![RequestMessage {
        role: Role::User,
        content: vec!["What is the weather like in Tokyo (in Celsius)? Use the provided `get_temperature` tool. Do not say anything else, just call the function."
            .to_string()
            .into()],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    let tool_call_blocks: Vec<_> = output
        .iter()
        .filter(|block| matches!(block, ContentBlock::ToolCall(_)))
        .collect();

    // Assert exactly one tool call
    assert_eq!(tool_call_blocks.len(), 1, "Expected exactly one tool call");

    let tool_call_block = tool_call_blocks[0];
    match tool_call_block {
        ContentBlock::ToolCall(tool_call) => {
            assert_eq!(tool_call.name, "get_temperature");
            serde_json::from_str::<Value>(&tool_call.arguments.to_lowercase()).unwrap();
        }
        _ => panic!("Unreachable"),
    }
}

pub async fn test_parallel_tool_use_inference_request_with_provider(provider: E2ETestProvider) {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "weather_helper_parallel",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather like in Tokyo (in Celsius)? Use both the provided `get_temperature` and `get_humidity` tools. Do not say anything else, just call the two functions."
                }
            ]},
        "parallel_tool_calls": true,
        "stream": false,
        "variant_name": provider.variant_name,
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check if the API response is fine
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    println!("API response: {response_json:#?}");

    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    let episode_id_response = response_json.get("episode_id").unwrap().as_str().unwrap();
    let episode_id_response = Uuid::parse_str(episode_id_response).unwrap();
    assert_eq!(episode_id_response, episode_id);

    let variant_name = response_json.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let content = response_json.get("content").unwrap().as_array().unwrap();

    // Validate the `get_temperature` tool call
    let content_block = content
        .iter()
        .find(|block| block["name"] == "get_temperature")
        .unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "tool_call");

    assert!(content_block.get("id").unwrap().as_str().is_some());

    let raw_name = content_block.get("raw_name").unwrap().as_str().unwrap();
    assert_eq!(raw_name, "get_temperature");
    let name = content_block.get("name").unwrap().as_str().unwrap();
    assert_eq!(name, "get_temperature");
    let raw_arguments = content_block
        .get("raw_arguments")
        .unwrap()
        .as_str()
        .unwrap();
    let raw_arguments: Value = serde_json::from_str(raw_arguments).unwrap();
    let raw_arguments = raw_arguments.as_object().unwrap();
    assert!(raw_arguments.len() == 2);
    let location = raw_arguments.get("location").unwrap().as_str().unwrap();
    assert_eq!(location.to_lowercase(), "tokyo");
    let units = raw_arguments.get("units").unwrap().as_str().unwrap();
    assert!(units == "celsius");

    let arguments = content_block.get("arguments").unwrap().as_object().unwrap();
    assert!(arguments.len() == 2);
    let location = arguments.get("location").unwrap().as_str().unwrap();
    assert_eq!(location.to_lowercase(), "tokyo");
    let units = arguments.get("units").unwrap().as_str().unwrap();
    assert!(units == "celsius");

    // Validate the `get_humidity` tool call
    let content_block = content
        .iter()
        .find(|block| block["name"] == "get_humidity")
        .unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "tool_call");

    assert!(content_block.get("id").unwrap().as_str().is_some());

    let raw_name = content_block.get("raw_name").unwrap().as_str().unwrap();
    assert_eq!(raw_name, "get_humidity");
    let name = content_block.get("name").unwrap().as_str().unwrap();
    assert_eq!(name, "get_humidity");

    let raw_arguments = content_block
        .get("raw_arguments")
        .unwrap()
        .as_str()
        .unwrap();
    let raw_arguments: Value = serde_json::from_str(raw_arguments).unwrap();
    let raw_arguments = raw_arguments.as_object().unwrap();
    assert!(raw_arguments.len() == 1);
    let location = raw_arguments.get("location").unwrap().as_str().unwrap();
    assert_eq!(location.to_lowercase(), "tokyo");

    let arguments = content_block.get("arguments").unwrap();
    let arguments = arguments.as_object().unwrap();
    assert!(arguments.len() == 1);
    let location = arguments.get("location").unwrap().as_str().unwrap();
    assert_eq!(location.to_lowercase(), "tokyo");

    let usage = response_json.get("usage").unwrap();
    let usage = usage.as_object().unwrap();
    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    assert!(output_tokens > 0);

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check if ClickHouse is correct - ChatInference table
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ChatInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, payload["function_name"]);

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let episode_id_result = result.get("episode_id").unwrap().as_str().unwrap();
    let episode_id_result = Uuid::parse_str(episode_id_result).unwrap();
    assert_eq!(episode_id_result, episode_id);

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input: Value = json!(
        {
            "system": {
                "assistant_name": "Dr. Mehta"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "value": "What is the weather like in Tokyo (in Celsius)? Use both the provided `get_temperature` and `get_humidity` tools. Do not say anything else, just call the two functions."
                    }]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);

    let output_clickhouse: Vec<Value> =
        serde_json::from_str(result.get("output").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(output_clickhouse, *content);

    let tool_params: Value =
        serde_json::from_str(result.get("tool_params").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(tool_params["tool_choice"], "auto");
    assert_eq!(tool_params["parallel_tool_calls"], true);

    let tools_available = tool_params["tools_available"].as_array().unwrap();

    // Validate the `get_temperature` tool
    assert_eq!(tools_available.len(), 2);
    let tool = tools_available
        .iter()
        .find(|tool| tool["name"] == "get_temperature")
        .unwrap();
    assert_eq!(
        tool["description"],
        "Get the current temperature in a given location"
    );
    assert_eq!(tool["strict"], false);

    let tool_parameters = tool["parameters"].as_object().unwrap();
    assert_eq!(tool_parameters["type"], "object");
    assert!(tool_parameters.get("properties").is_some());
    assert!(tool_parameters.get("required").is_some());

    let properties = tool_parameters["properties"].as_object().unwrap();
    assert!(properties.contains_key("location"));
    assert!(properties.contains_key("units"));

    let location = properties["location"].as_object().unwrap();
    assert_eq!(location["type"], "string");
    assert_eq!(
        location["description"],
        "The location to get the temperature for (e.g. \"New York\")"
    );

    let units = properties["units"].as_object().unwrap();
    assert_eq!(units["type"], "string");
    assert_eq!(
        units["description"],
        "The units to get the temperature in (must be \"fahrenheit\" or \"celsius\")"
    );
    let units_enum = units["enum"].as_array().unwrap();
    assert_eq!(units_enum.len(), 2);
    assert!(units_enum.contains(&json!("fahrenheit")));
    assert!(units_enum.contains(&json!("celsius")));

    let required = tool_parameters["required"].as_array().unwrap();
    assert!(required.contains(&json!("location")));

    // Validate the `get_humidity` tool
    let tool = tools_available
        .iter()
        .find(|tool| tool["name"] == "get_humidity")
        .unwrap();
    assert_eq!(
        tool["description"],
        "Get the current humidity in a given location"
    );
    assert_eq!(tool["strict"], false);

    let tool_parameters = tool["parameters"].as_object().unwrap();
    assert_eq!(tool_parameters["type"], "object");
    assert!(tool_parameters.get("properties").is_some());
    assert!(tool_parameters.get("required").is_some());

    let properties = tool_parameters["properties"].as_object().unwrap();
    assert!(properties.contains_key("location"));

    let location = properties["location"].as_object().unwrap();
    assert_eq!(location["type"], "string");
    assert_eq!(
        location["description"],
        "The location to get the humidity for (e.g. \"New York\")"
    );

    let required = tool_parameters["required"].as_array().unwrap();
    assert!(required.contains(&json!("location")));

    // Check if ClickHouse is correct - ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(id).is_ok());

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.contains("get_temperature"));
    assert!(raw_request.to_lowercase().contains("tokyo"));
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );

    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    assert!(raw_response.contains("get_temperature"));
    assert!(raw_response.to_lowercase().contains("tokyo"));

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(output_tokens > 0);
    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);
    assert!(result.get("ttft_ms").unwrap().is_null());

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta.\n\nPeople will ask you questions about the weather.\n\nIf asked about the weather, just respond with two tool calls. Use BOTH the \"get_temperature\" and \"get_humidity\" tools.\n\nIf provided with a tool result, use it to respond to the user (e.g. \"The weather in New York is 55 degrees Fahrenheit with 50% humidity.\")."
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![RequestMessage {
        role: Role::User,
        content: vec!["What is the weather like in Tokyo (in Celsius)? Use both the provided `get_temperature` and `get_humidity` tools. Do not say anything else, just call the two functions."
            .to_string()
            .into()],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    assert_eq!(output.len(), 2);
    let mut tool_call_names = vec![];
    for block in output {
        match block {
            ContentBlock::ToolCall(tool_call) => {
                tool_call_names.push(tool_call.name);
                serde_json::from_str::<Value>(&tool_call.arguments).unwrap();
            }
            _ => {
                panic!("Expected a tool call, got {:?}", block);
            }
        }
    }
    assert!(tool_call_names.contains(&"get_temperature".to_string()));
    assert!(tool_call_names.contains(&"get_humidity".to_string()));
}

pub async fn test_parallel_tool_use_streaming_inference_request_with_provider(
    provider: E2ETestProvider,
) {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "weather_helper_parallel",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather like in Tokyo (in Celsius)? Use both the provided `get_temperature` and `get_humidity` tools. Do not say anything else, just call the two functions."
                }
            ]},
        "parallel_tool_calls": true,
        "stream": true,
        "variant_name": provider.variant_name,
    });

    let mut event_source = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .unwrap();

    let mut chunks = vec![];
    let mut found_done_chunk = false;
    while let Some(event) = event_source.next().await {
        let event = event.unwrap();
        match event {
            Event::Open => continue,
            Event::Message(message) => {
                if message.data == "[DONE]" {
                    found_done_chunk = true;
                    break;
                }
                chunks.push(message.data);
            }
        }
    }
    assert!(found_done_chunk);

    let mut inference_id = None;
    let mut get_temperature_tool_id: Option<String> = None;
    let mut get_temperature_arguments = String::new();
    let mut get_humidity_tool_id: Option<String> = None;
    let mut get_humidity_arguments = String::new();
    let mut input_tokens = 0;
    let mut output_tokens = 0;

    for chunk in chunks {
        let chunk_json: Value = serde_json::from_str(&chunk).unwrap();

        println!("API response chunk: {chunk_json:#?}");

        let chunk_inference_id = chunk_json.get("inference_id").unwrap().as_str().unwrap();
        let chunk_inference_id = Uuid::parse_str(chunk_inference_id).unwrap();
        match inference_id {
            None => inference_id = Some(chunk_inference_id),
            Some(inference_id) => assert_eq!(inference_id, chunk_inference_id),
        }

        let chunk_episode_id = chunk_json.get("episode_id").unwrap().as_str().unwrap();
        let chunk_episode_id = Uuid::parse_str(chunk_episode_id).unwrap();
        assert_eq!(chunk_episode_id, episode_id);

        for block in chunk_json.get("content").unwrap().as_array().unwrap() {
            assert!(block.get("id").is_some());

            let block_type = block.get("type").unwrap().as_str().unwrap();

            match block_type {
                "tool_call" => {
                    let block_tool_id = block.get("id").unwrap().as_str().unwrap();
                    let tool_name = block.get("raw_name").unwrap().as_str().unwrap();
                    let chunk_arguments = block.get("raw_arguments").unwrap().as_str().unwrap();

                    match tool_name {
                        "get_temperature" => {
                            match &get_temperature_tool_id {
                                None => get_temperature_tool_id = Some(block_tool_id.to_string()),
                                Some(tool_id) => assert_eq!(tool_id, block_tool_id),
                            };
                            get_temperature_arguments.push_str(chunk_arguments);
                        }
                        "get_humidity" => {
                            match &get_humidity_tool_id {
                                None => get_humidity_tool_id = Some(block_tool_id.to_string()),
                                Some(tool_id) => assert_eq!(tool_id, block_tool_id),
                            };
                            get_humidity_arguments.push_str(chunk_arguments);
                        }
                        _ => {
                            panic!("Unexpected tool name: {}", tool_name);
                        }
                    }
                }
                "text" => {
                    // Sometimes the model will also return some text
                    // (e.g. "Sure, here's the weather in Tokyo:" + tool call)
                    // We mostly care about the tool call, so we'll ignore the text.
                }
                _ => {
                    panic!("Unexpected block type: {}", block_type);
                }
            }
        }

        if let Some(usage) = chunk_json.get("usage").and_then(|u| u.as_object()) {
            input_tokens += usage.get("input_tokens").unwrap().as_u64().unwrap();
            output_tokens += usage.get("output_tokens").unwrap().as_u64().unwrap();
        }
    }

    // NB: Azure doesn't return usage during streaming
    if provider.variant_name.contains("azure") {
        assert_eq!(input_tokens, 0);
        assert_eq!(output_tokens, 0);
    } else {
        assert!(input_tokens > 0);
        assert!(output_tokens > 0);
    }

    let inference_id = inference_id.unwrap();
    let get_temperature_tool_id = get_temperature_tool_id.unwrap();
    let get_humidity_tool_id = get_humidity_tool_id.unwrap();
    assert!(serde_json::from_str::<Value>(&get_temperature_arguments).is_ok());
    assert!(serde_json::from_str::<Value>(&get_humidity_arguments).is_ok());

    // Sleep for 1 second to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ClickHouse - Inference Table
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ChatInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, "weather_helper_parallel");

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let episode_id_result = result.get("episode_id").unwrap().as_str().unwrap();
    let episode_id_result = Uuid::parse_str(episode_id_result).unwrap();
    assert_eq!(episode_id_result, episode_id);

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input: Value = json!(
        {
            "system": {
                "assistant_name": "Dr. Mehta"
            },
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "value": "What is the weather like in Tokyo (in Celsius)? Use both the provided `get_temperature` and `get_humidity` tools. Do not say anything else, just call the two functions."}]
                }
            ]
        }
    );
    assert_eq!(input, correct_input);

    let output_clickhouse: Vec<Value> =
        serde_json::from_str(result.get("output").unwrap().as_str().unwrap()).unwrap();
    assert!(!output_clickhouse.is_empty()); // could be > 1 if the model returns text as well

    // Validate the `get_temperature` tool call
    let content_block = output_clickhouse
        .iter()
        .find(|block| block["name"] == "get_temperature")
        .unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "tool_call");
    assert_eq!(
        content_block.get("id").unwrap().as_str().unwrap(),
        get_temperature_tool_id
    );
    assert_eq!(
        content_block
            .get("raw_arguments")
            .unwrap()
            .as_str()
            .unwrap(),
        get_temperature_arguments
    );
    assert_eq!(
        content_block.get("raw_name").unwrap().as_str().unwrap(),
        "get_temperature"
    );
    assert_eq!(
        content_block.get("arguments").unwrap().as_object().unwrap(),
        &serde_json::from_str::<serde_json::Map<String, Value>>(&get_temperature_arguments)
            .unwrap()
    );

    // Validate the `get_humidity` tool call
    let content_block = output_clickhouse
        .iter()
        .find(|block| block["name"] == "get_humidity")
        .unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "tool_call");
    assert_eq!(
        content_block.get("id").unwrap().as_str().unwrap(),
        get_humidity_tool_id
    );
    assert_eq!(
        content_block
            .get("raw_arguments")
            .unwrap()
            .as_str()
            .unwrap(),
        get_humidity_arguments
    );
    assert_eq!(
        content_block.get("arguments").unwrap().as_object().unwrap(),
        &serde_json::from_str::<serde_json::Map<String, Value>>(&get_humidity_arguments).unwrap()
    );
    assert_eq!(
        content_block.get("raw_name").unwrap().as_str().unwrap(),
        "get_humidity"
    );

    let tool_params: Value =
        serde_json::from_str(result.get("tool_params").unwrap().as_str().unwrap()).unwrap();
    assert_eq!(tool_params["tool_choice"], "auto");
    assert_eq!(tool_params["parallel_tool_calls"], true);

    let tools_available = tool_params["tools_available"].as_array().unwrap();
    assert_eq!(tools_available.len(), 2);

    let tool = tools_available
        .iter()
        .find(|tool| tool["name"] == "get_temperature")
        .unwrap();
    assert_eq!(
        tool["description"],
        "Get the current temperature in a given location"
    );
    assert_eq!(tool["strict"], false);

    let tool_parameters = tool["parameters"].as_object().unwrap();
    assert_eq!(tool_parameters["type"], "object");
    assert!(tool_parameters.get("properties").is_some());
    assert!(tool_parameters.get("required").is_some());
    assert_eq!(tool_parameters["additionalProperties"], false);

    let properties = tool_parameters["properties"].as_object().unwrap();
    assert!(properties.contains_key("location"));
    assert!(properties.contains_key("units"));

    let location = properties["location"].as_object().unwrap();
    assert_eq!(location["type"], "string");
    assert_eq!(
        location["description"],
        "The location to get the temperature for (e.g. \"New York\")"
    );

    let units = properties["units"].as_object().unwrap();
    assert_eq!(units["type"], "string");
    assert_eq!(
        units["description"],
        "The units to get the temperature in (must be \"fahrenheit\" or \"celsius\")"
    );
    let units_enum = units["enum"].as_array().unwrap();
    assert_eq!(units_enum.len(), 2);
    assert!(units_enum.contains(&json!("fahrenheit")));
    assert!(units_enum.contains(&json!("celsius")));

    let required = tool_parameters["required"].as_array().unwrap();
    assert!(required.contains(&json!("location")));

    let tool = tools_available
        .iter()
        .find(|tool| tool["name"] == "get_humidity")
        .unwrap();
    assert_eq!(
        tool["description"],
        "Get the current humidity in a given location"
    );
    assert_eq!(tool["strict"], false);

    let tool_parameters = tool["parameters"].as_object().unwrap();
    assert_eq!(tool_parameters["type"], "object");
    assert!(tool_parameters.get("properties").is_some());
    assert!(tool_parameters.get("required").is_some());
    assert_eq!(tool_parameters["additionalProperties"], false);

    let properties = tool_parameters["properties"].as_object().unwrap();
    assert!(properties.contains_key("location"));

    let location = properties["location"].as_object().unwrap();
    assert_eq!(location["type"], "string");
    assert_eq!(
        location["description"],
        "The location to get the humidity for (e.g. \"New York\")"
    );

    let required = tool_parameters["required"].as_array().unwrap();
    assert!(required.contains(&json!("location")));

    // Check if ClickHouse is correct - ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(id).is_ok());

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.to_lowercase().contains("get_temperature"));
    assert!(raw_request.to_lowercase().contains("get_humidity"));
    assert!(raw_request.to_lowercase().contains("tokyo"));
    assert!(raw_request.to_lowercase().contains("celsius"));
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );

    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    assert!(raw_response.contains("get_temperature"));
    // Check if raw_response is valid JSONL
    for line in raw_response.lines() {
        assert!(serde_json::from_str::<Value>(line).is_ok());
    }

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();

    // NB: Azure doesn't support input/output tokens during streaming
    if provider.variant_name.contains("azure") {
        assert_eq!(input_tokens, 0);
        assert_eq!(output_tokens, 0);
    } else {
        assert!(input_tokens > 0);
        assert!(output_tokens > 0);
    }

    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);

    let ttft_ms = result.get("ttft_ms").unwrap().as_u64().unwrap();
    assert!(ttft_ms > 50);
    assert!(ttft_ms <= response_time_ms);

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta.\n\nPeople will ask you questions about the weather.\n\nIf asked about the weather, just respond with two tool calls. Use BOTH the \"get_temperature\" and \"get_humidity\" tools.\n\nIf provided with a tool result, use it to respond to the user (e.g. \"The weather in New York is 55 degrees Fahrenheit with 50% humidity.\")."
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![RequestMessage {
        role: Role::User,
        content: vec!["What is the weather like in Tokyo (in Celsius)? Use both the provided `get_temperature` and `get_humidity` tools. Do not say anything else, just call the two functions."
            .to_string()
            .into()],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    assert_eq!(output.len(), 2);
    let mut tool_call_names = vec![];
    for block in output {
        match block {
            ContentBlock::ToolCall(tool_call) => {
                tool_call_names.push(tool_call.name);
                serde_json::from_str::<Value>(&tool_call.arguments).unwrap();
            }
            _ => {
                panic!("Expected a tool call, got {:?}", block);
            }
        }
    }
    assert!(tool_call_names.contains(&"get_temperature".to_string()));
    assert!(tool_call_names.contains(&"get_humidity".to_string()));
}

pub async fn test_json_mode_inference_request_with_provider(provider: E2ETestProvider) {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "json_success",
        "variant_name": provider.variant_name,
        "episode_id": episode_id,
        "input":
            {
               "system": {"assistant_name": "Dr. Mehta"},
               "messages": [
                {
                    "role": "user",
                    "content": {"country": "Japan"}
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

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    println!("API response: {response_json:#?}");

    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    let episode_id_response = response_json.get("episode_id").unwrap().as_str().unwrap();
    let episode_id_response = Uuid::parse_str(episode_id_response).unwrap();
    assert_eq!(episode_id_response, episode_id);

    let variant_name = response_json.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let output = response_json.get("output").unwrap().as_object().unwrap();
    assert!(output.keys().len() == 2);
    let parsed_output = output.get("parsed").unwrap().as_object().unwrap();
    assert!(parsed_output
        .get("answer")
        .unwrap()
        .as_str()
        .unwrap()
        .to_lowercase()
        .contains("tokyo"));
    let raw_output = output.get("raw").unwrap().as_str().unwrap();
    let raw_output: Value = serde_json::from_str(raw_output).unwrap();
    assert_eq!(&raw_output, output.get("parsed").unwrap());

    let usage = response_json.get("usage").unwrap();
    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(output_tokens > 0);

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check if ClickHouse is ok - JsonInference Table
    let clickhouse = get_clickhouse().await;
    let result = select_json_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - JsonInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id = Uuid::parse_str(id).unwrap();
    assert_eq!(id, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, "json_success");

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input = json!({
        "system": {"assistant_name": "Dr. Mehta"},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "value": {"country": "Japan"}}]
            }
        ]
    });
    assert_eq!(input, correct_input);

    let output_clickhouse = result.get("output").unwrap().as_str().unwrap();
    let output_clickhouse: Value = serde_json::from_str(output_clickhouse).unwrap();
    let output_clickhouse = output_clickhouse.as_object().unwrap();
    assert_eq!(output_clickhouse, output);

    let inference_params = result.get("inference_params").unwrap().as_str().unwrap();
    let inference_params: Value = serde_json::from_str(inference_params).unwrap();
    let inference_params = inference_params.get("chat_completion").unwrap();
    assert!(inference_params.get("temperature").is_none());
    assert!(inference_params.get("seed").is_none());
    assert_eq!(
        inference_params
            .get("max_tokens")
            .unwrap()
            .as_u64()
            .unwrap(),
        100
    );

    let processing_time_ms = result.get("processing_time_ms").unwrap().as_u64().unwrap();
    assert!(processing_time_ms > 0);
    let retrieved_output_schema = result.get("output_schema").unwrap().as_str().unwrap();
    let retrieved_output_schema: Value = serde_json::from_str(retrieved_output_schema).unwrap();
    let expected_output_schema = json!({
        "type": "object",
        "properties": {
          "answer": {
            "type": "string"
          }
        },
        "required": ["answer"],
        "additionalProperties": false
      }
    );
    assert_eq!(retrieved_output_schema, expected_output_schema);

    // Check the ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let model_inference_id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(model_inference_id).is_ok());

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.to_lowercase().contains("japan"));
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );

    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    assert!(raw_response.to_lowercase().contains("tokyo"));
    assert!(serde_json::from_str::<Value>(raw_response).is_ok());

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(output_tokens > 0);
    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);
    assert!(result.get("ttft_ms").unwrap().is_null());

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta.\n\nPlease answer the questions in a JSON with key \"answer\".\n\nDo not include any other text than the JSON object. Do not include \"```json\" or \"```\" or anything else.\n\nExample Response:\n\n{\n    \"answer\": \"42\"\n}"
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![RequestMessage {
        role: Role::User,
        content: vec!["What is the capital city of Japan?".to_string().into()],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    assert_eq!(output.len(), 1);
    match &output[0] {
        ContentBlock::Text(text) => {
            let parsed: Value = serde_json::from_str(&text.text).unwrap();
            let answer = parsed.get("answer").unwrap().as_str().unwrap();
            assert!(answer.to_lowercase().contains("tokyo"));
        }
        ContentBlock::ToolCall(tool_call) => {
            // Handles implicit tool calls
            assert_eq!(tool_call.name, "respond");
            let arguments: Value = serde_json::from_str(&tool_call.arguments).unwrap();
            let answer = arguments.get("answer").unwrap().as_str().unwrap();
            assert!(answer.to_lowercase().contains("tokyo"));
        }
        _ => {
            panic!("Expected a text block, got {:?}", output[0]);
        }
    }
}

pub async fn test_dynamic_json_mode_inference_request_with_provider(provider: E2ETestProvider) {
    let episode_id = Uuid::now_v7();
    let output_schema = json!({
      "type": "object",
      "properties": {
        "response": {
          "type": "string"
        }
      },
      "required": ["response"],
      "additionalProperties": false
    });
    let serialized_output_schema = serde_json::to_string(&output_schema).unwrap();

    let payload = json!({
        "function_name": "dynamic_json",
        "variant_name": provider.variant_name,
        "episode_id": episode_id,
        "input":
            {
               "system": {"assistant_name": "Dr. Mehta", "schema": serialized_output_schema},
               "messages": [
                {
                    "role": "user",
                    "content": {"country": "Japan"}
                }
            ]},
        "stream": false,
        "output_schema": output_schema.clone(),
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check that the API response is ok
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    println!("API response: {response_json:#?}");

    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    let episode_id_response = response_json.get("episode_id").unwrap().as_str().unwrap();
    let episode_id_response = Uuid::parse_str(episode_id_response).unwrap();
    assert_eq!(episode_id_response, episode_id);

    let variant_name = response_json.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let output = response_json.get("output").unwrap().as_object().unwrap();
    assert!(output.keys().len() == 2);
    let parsed_output = output.get("parsed").unwrap().as_object().unwrap();
    assert!(parsed_output
        .get("response")
        .unwrap()
        .as_str()
        .unwrap()
        .to_lowercase()
        .contains("tokyo"));
    let raw_output = output.get("raw").unwrap().as_str().unwrap();
    let raw_output: Value = serde_json::from_str(raw_output).unwrap();
    assert_eq!(&raw_output, output.get("parsed").unwrap());

    let usage = response_json.get("usage").unwrap();
    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(output_tokens > 0);

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check if ClickHouse is ok - JsonInference Table
    let clickhouse = get_clickhouse().await;
    let result = select_json_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - JsonInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id = Uuid::parse_str(id).unwrap();
    assert_eq!(id, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, "dynamic_json");

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input = json!({
        "system": {"assistant_name": "Dr. Mehta", "schema": serialized_output_schema},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "value": {"country": "Japan"}}]
            }
        ]
    });
    assert_eq!(input, correct_input);

    let output_clickhouse = result.get("output").unwrap().as_str().unwrap();
    let output_clickhouse: Value = serde_json::from_str(output_clickhouse).unwrap();
    let output_clickhouse = output_clickhouse.as_object().unwrap();
    assert_eq!(output_clickhouse, output);

    let inference_params = result.get("inference_params").unwrap().as_str().unwrap();
    let inference_params: Value = serde_json::from_str(inference_params).unwrap();
    let inference_params = inference_params.get("chat_completion").unwrap();
    assert!(inference_params.get("temperature").is_none());
    assert!(inference_params.get("seed").is_none());
    assert_eq!(
        inference_params
            .get("max_tokens")
            .unwrap()
            .as_u64()
            .unwrap(),
        100
    );

    let processing_time_ms = result.get("processing_time_ms").unwrap().as_u64().unwrap();
    assert!(processing_time_ms > 0);

    let retrieved_output_schema = result.get("output_schema").unwrap().as_str().unwrap();
    let retrieved_output_schema: Value = serde_json::from_str(retrieved_output_schema).unwrap();
    assert_eq!(retrieved_output_schema, output_schema);

    // Check the ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let model_inference_id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(model_inference_id).is_ok());

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.to_lowercase().contains("japan"));
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );

    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    assert!(raw_response.to_lowercase().contains("tokyo"));
    assert!(serde_json::from_str::<Value>(raw_response).is_ok());

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(output_tokens > 0);
    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);
    assert!(result.get("ttft_ms").unwrap().is_null());

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta.\n\nDo not include any other text than the JSON object.  Do not include \"```json\" or \"```\" or anything else.\n\nPlease answer the questions in a JSON with the following schema:\n\n{\"type\":\"object\",\"properties\":{\"response\":{\"type\":\"string\"}},\"required\":[\"response\"],\"additionalProperties\":false}"
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![RequestMessage {
        role: Role::User,
        content: vec!["What is the capital city of Japan?".to_string().into()],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    assert_eq!(output.len(), 1);
    match &output[0] {
        ContentBlock::Text(text) => {
            let parsed: Value = serde_json::from_str(&text.text).unwrap();
            let answer = parsed.get("response").unwrap().as_str().unwrap();
            assert!(answer.to_lowercase().contains("tokyo"));
        }
        ContentBlock::ToolCall(tool_call) => {
            // Handles implicit tool calls
            assert_eq!(tool_call.name, "respond");
            let arguments: Value = serde_json::from_str(&tool_call.arguments).unwrap();
            let answer = arguments.get("response").unwrap().as_str().unwrap();
            assert!(answer.to_lowercase().contains("tokyo"));
        }
        _ => {
            panic!("Expected a text block, got {:?}", output[0]);
        }
    }
}

pub async fn test_json_mode_streaming_inference_request_with_provider(provider: E2ETestProvider) {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "json_success",
        "variant_name": provider.variant_name,
        "episode_id": episode_id,
        "input":
            {
               "system": {"assistant_name": "Dr. Mehta"},
               "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "value": {"country": "Japan"}}]
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
    let mut found_done_chunk = false;
    while let Some(event) = event_source.next().await {
        let event = event.unwrap();
        match event {
            Event::Open => continue,
            Event::Message(message) => {
                if message.data == "[DONE]" {
                    found_done_chunk = true;
                    break;
                }
                chunks.push(message.data);
            }
        }
    }
    assert!(found_done_chunk);

    let mut inference_id: Option<Uuid> = None;
    let mut full_content = String::new();
    let mut input_tokens = 0;
    let mut output_tokens = 0;
    for chunk in chunks.clone() {
        let chunk_json: Value = serde_json::from_str(&chunk).unwrap();

        println!("API response chunk: {chunk_json:#?}");

        let chunk_inference_id = chunk_json.get("inference_id").unwrap().as_str().unwrap();
        let chunk_inference_id = Uuid::parse_str(chunk_inference_id).unwrap();
        match inference_id {
            Some(inference_id) => {
                assert_eq!(inference_id, chunk_inference_id);
            }
            None => {
                inference_id = Some(chunk_inference_id);
            }
        }

        let chunk_episode_id = chunk_json.get("episode_id").unwrap().as_str().unwrap();
        let chunk_episode_id = Uuid::parse_str(chunk_episode_id).unwrap();
        assert_eq!(chunk_episode_id, episode_id);

        let raw = chunk_json.get("raw").unwrap().as_str().unwrap();
        if !raw.is_empty() {
            full_content.push_str(raw);
        }

        if let Some(usage) = chunk_json.get("usage") {
            input_tokens += usage.get("input_tokens").unwrap().as_u64().unwrap();
            output_tokens += usage.get("output_tokens").unwrap().as_u64().unwrap();
        }
    }

    let inference_id = inference_id.unwrap();
    assert!(full_content.to_lowercase().contains("tokyo"));

    // NB: Azure doesn't support input/output tokens during streaming
    if provider.variant_name.contains("azure") {
        assert_eq!(input_tokens, 0);
        assert_eq!(output_tokens, 0);
    } else {
        assert!(input_tokens > 0);
        assert!(output_tokens > 0);
    }

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ClickHouse - JsonInference Table
    let clickhouse = get_clickhouse().await;
    let result = select_json_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - JsonInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, "json_success");

    let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(variant_name, provider.variant_name);

    let episode_id_result = result.get("episode_id").unwrap().as_str().unwrap();
    let episode_id_result = Uuid::parse_str(episode_id_result).unwrap();
    assert_eq!(episode_id_result, episode_id);

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input = json!({
        "system": {"assistant_name": "Dr. Mehta"},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "value": {"country": "Japan"}}]
            }
        ]
    });
    assert_eq!(input, correct_input);

    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Value = serde_json::from_str(output).unwrap();
    let output = output.as_object().unwrap();
    assert_eq!(output.keys().len(), 2);
    let clickhouse_parsed = output.get("parsed").unwrap().as_object().unwrap();
    let clickhouse_raw = output.get("parsed").unwrap().as_object().unwrap();
    assert_eq!(clickhouse_parsed, clickhouse_raw);
    let full_content_parsed: Value = serde_json::from_str(&full_content).unwrap();
    let full_content_parsed = full_content_parsed.as_object().unwrap();
    assert_eq!(clickhouse_parsed, full_content_parsed);

    let inference_params = result.get("inference_params").unwrap().as_str().unwrap();
    let inference_params: Value = serde_json::from_str(inference_params).unwrap();
    let inference_params = inference_params.get("chat_completion").unwrap();
    assert!(inference_params.get("temperature").is_none());
    assert!(inference_params.get("seed").is_none());
    assert_eq!(
        inference_params
            .get("max_tokens")
            .unwrap()
            .as_u64()
            .unwrap(),
        100
    );

    let processing_time_ms = result.get("processing_time_ms").unwrap().as_u64().unwrap();
    assert!(processing_time_ms > 0);

    let retrieved_output_schema = result.get("output_schema").unwrap().as_str().unwrap();
    let retrieved_output_schema: Value = serde_json::from_str(retrieved_output_schema).unwrap();
    let expected_output_schema = json!({
        "type": "object",
        "properties": {
          "answer": {
            "type": "string"
          }
        },
        "required": ["answer"],
        "additionalProperties": false
    });
    assert_eq!(retrieved_output_schema, expected_output_schema);

    // Check ClickHouse - ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");

    let model_inference_id = result.get("id").unwrap().as_str().unwrap();
    assert!(Uuid::parse_str(model_inference_id).is_ok());

    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.to_lowercase().contains("japan"));
    assert!(raw_request.to_lowercase().contains("mehta"));
    assert!(
        serde_json::from_str::<Value>(raw_request).is_ok(),
        "raw_request is not a valid JSON"
    );

    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();

    // Check if raw_response is valid JSONL
    for line in raw_response.lines() {
        assert!(serde_json::from_str::<Value>(line).is_ok());
    }

    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();

    // NB: Azure doesn't support input/output tokens during streaming
    if provider.variant_name.contains("azure") {
        assert_eq!(input_tokens, 0);
        assert_eq!(output_tokens, 0);
    } else {
        assert!(input_tokens > 0);
        assert!(output_tokens > 0);
    }

    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);

    let ttft_ms = result.get("ttft_ms").unwrap().as_u64().unwrap();
    assert!(ttft_ms > 50);
    assert!(ttft_ms <= response_time_ms);

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta.\n\nPlease answer the questions in a JSON with key \"answer\".\n\nDo not include any other text than the JSON object. Do not include \"```json\" or \"```\" or anything else.\n\nExample Response:\n\n{\n    \"answer\": \"42\"\n}"
    );
    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![RequestMessage {
        role: Role::User,
        content: vec!["What is the capital city of Japan?".to_string().into()],
    }];
    assert_eq!(input_messages, expected_input_messages);
    let output = result.get("output").unwrap().as_str().unwrap();
    let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    assert_eq!(output.len(), 1);
    match &output[0] {
        ContentBlock::Text(text) => {
            let parsed: Value = serde_json::from_str(&text.text).unwrap();
            let answer = parsed.get("answer").unwrap().as_str().unwrap();
            assert!(answer.to_lowercase().contains("tokyo"));
        }
        ContentBlock::ToolCall(tool_call) => {
            // Handles implicit tool calls
            assert_eq!(tool_call.name, "respond");
            let arguments: Value = serde_json::from_str(&tool_call.arguments).unwrap();
            let answer = arguments.get("answer").unwrap().as_str().unwrap();
            assert!(answer.to_lowercase().contains("tokyo"));
        }
        _ => {
            panic!("Expected a text block, got {:?}", output[0]);
        }
    }
}
