#![allow(clippy::print_stdout)]

use reqwest::{Client, StatusCode};
use serde_json::{json, Value};
use uuid::Uuid;

use crate::common::get_gateway_endpoint;

use super::common::E2ETestProvider;

#[macro_export]
macro_rules! generate_batch_inference_tests {
    ($func:ident) => {
        use $crate::providers::batch::test_simple_batch_inference_request_with_provider;

        #[tokio::test]
        async fn test_simple_batch_inference_request() {
            let all_providers = $func().await;
            let providers = all_providers.simple_inference;
            println!(
                "supports_batch_inference: {}",
                all_providers.supports_batch_inference
            );
            if all_providers.supports_batch_inference {
                for provider in providers {
                    test_simple_batch_inference_request_with_provider(provider).await;
                }
            }
        }
    };
}

pub async fn test_simple_batch_inference_request_with_provider(provider: E2ETestProvider) {
    let episode_id = Uuid::now_v7();
    let tag_value = Uuid::now_v7().to_string();

    let payload = json!({
        "function_name": "basic_test",
        "variant_name": provider.variant_name,
        "episode_ids": [episode_id],
        "inputs":
            [{
               "system": {"assistant_name": "Dr. Mehta"},
               "messages": [
                {
                    "role": "user",
                    "content": "What is the capital city of Japan?"
                }
            ]}],
        "tags": [{"key": tag_value}],
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/batch_inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check that the API response is ok
    // assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    println!("API response: {response_json:#?}");
    assert!(false);

    // let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    // let inference_id = Uuid::parse_str(inference_id).unwrap();

    // let episode_id_response = response_json.get("episode_id").unwrap().as_str().unwrap();
    // let episode_id_response = Uuid::parse_str(episode_id_response).unwrap();
    // assert_eq!(episode_id_response, episode_id);

    // let variant_name = response_json.get("variant_name").unwrap().as_str().unwrap();
    // assert_eq!(variant_name, provider.variant_name);

    // let content = response_json.get("content").unwrap().as_array().unwrap();
    // assert_eq!(content.len(), 1);
    // let content_block = content.first().unwrap();
    // let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    // assert_eq!(content_block_type, "text");
    // let content = content_block.get("text").unwrap().as_str().unwrap();
    // assert!(content.to_lowercase().contains("tokyo"));

    // let usage = response_json.get("usage").unwrap();
    // let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    // assert!(input_tokens > 0);
    // let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    // assert!(output_tokens > 0);

    // // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    // tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // // Check if ClickHouse is ok - ChatInference Table
    // let clickhouse = get_clickhouse().await;
    // let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
    //     .await
    //     .unwrap();

    // println!("ClickHouse - ChatInference: {result:#?}");

    // let id = result.get("id").unwrap().as_str().unwrap();
    // let id = Uuid::parse_str(id).unwrap();
    // assert_eq!(id, inference_id);

    // let function_name = result.get("function_name").unwrap().as_str().unwrap();
    // assert_eq!(function_name, payload["function_name"]);

    // let variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    // assert_eq!(variant_name, provider.variant_name);

    // let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    // let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    // assert_eq!(retrieved_episode_id, episode_id);

    // let input: Value =
    //     serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    // let correct_input = json!({
    //     "system": {"assistant_name": "Dr. Mehta"},
    //     "messages": [
    //         {
    //             "role": "user",
    //             "content": [{"type": "text", "value": "What is the capital city of Japan?"}]
    //         }
    //     ]
    // });
    // assert_eq!(input, correct_input);

    // let content_blocks = result.get("output").unwrap().as_str().unwrap();
    // let content_blocks: Vec<Value> = serde_json::from_str(content_blocks).unwrap();
    // assert_eq!(content_blocks.len(), 1);
    // let content_block = content_blocks.first().unwrap();
    // let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    // assert_eq!(content_block_type, "text");
    // let clickhouse_content = content_block.get("text").unwrap().as_str().unwrap();
    // assert_eq!(clickhouse_content, content);

    // let tags = result.get("tags").unwrap().as_object().unwrap();
    // assert_eq!(tags.len(), 1);
    // assert_eq!(tags.get("key").unwrap().as_str().unwrap(), tag_value);

    // let tool_params = result.get("tool_params").unwrap().as_str().unwrap();
    // assert!(tool_params.is_empty());

    // let inference_params = result.get("inference_params").unwrap().as_str().unwrap();
    // let inference_params: Value = serde_json::from_str(inference_params).unwrap();
    // let inference_params = inference_params.get("chat_completion").unwrap();
    // assert!(inference_params.get("temperature").is_none());
    // assert!(inference_params.get("seed").is_none());
    // assert_eq!(
    //     inference_params
    //         .get("max_tokens")
    //         .unwrap()
    //         .as_u64()
    //         .unwrap(),
    //     100
    // );

    // let processing_time_ms = result.get("processing_time_ms").unwrap().as_u64().unwrap();
    // assert!(processing_time_ms > 0);

    // // Check the ModelInference Table
    // let result = select_model_inference_clickhouse(&clickhouse, inference_id)
    //     .await
    //     .unwrap();

    // println!("ClickHouse - ModelInference: {result:#?}");

    // let model_inference_id = result.get("id").unwrap().as_str().unwrap();
    // assert!(Uuid::parse_str(model_inference_id).is_ok());

    // let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    // let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    // assert_eq!(inference_id_result, inference_id);

    // let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    // assert!(raw_request.to_lowercase().contains("japan"));
    // assert!(
    //     serde_json::from_str::<Value>(raw_request).is_ok(),
    //     "raw_request is not a valid JSON"
    // );

    // let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    // assert!(raw_response.to_lowercase().contains("tokyo"));
    // assert!(serde_json::from_str::<Value>(raw_response).is_ok());

    // let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    // assert!(input_tokens > 0);
    // let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
    // assert!(output_tokens > 0);
    // let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    // assert!(response_time_ms > 0);
    // assert!(result.get("ttft_ms").unwrap().is_null());
    // let system = result.get("system").unwrap().as_str().unwrap();
    // assert_eq!(
    //     system,
    //     "You are a helpful and friendly assistant named Dr. Mehta"
    // );
    // let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    // let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    // let expected_input_messages = vec![RequestMessage {
    //     role: Role::User,
    //     content: vec!["What is the capital city of Japan?".to_string().into()],
    // }];
    // assert_eq!(input_messages, expected_input_messages);
    // let output = result.get("output").unwrap().as_str().unwrap();
    // let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
    // assert_eq!(output.len(), 1);

    // // Check the InferenceTag Table
    // let result = select_inference_tags_clickhouse(&clickhouse, "basic_test", "key", &tag_value)
    //     .await
    //     .unwrap();
    // let id = result.get("inference_id").unwrap().as_str().unwrap();
    // let id = Uuid::parse_str(id).unwrap();
    // assert_eq!(id, inference_id);
}
