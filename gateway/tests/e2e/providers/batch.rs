#![allow(clippy::print_stdout)]

use gateway::inference::types::{RequestMessage, Role};
use reqwest::{Client, StatusCode};
use serde_json::{json, Value};
use uuid::Uuid;

use crate::common::{
    get_clickhouse, get_gateway_endpoint, select_batch_model_inference_clickhouse,
    select_latest_batch_request_clickhouse,
};

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
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    println!("API response: {response_json:#?}");
    let batch_id = response_json.get("batch_id").unwrap().as_str().unwrap();
    let batch_id = Uuid::parse_str(batch_id).unwrap();

    let inference_ids = response_json
        .get("inference_ids")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(inference_ids.len(), 1);
    let inference_id = inference_ids.first().unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    let episode_ids = response_json
        .get("episode_ids")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(episode_ids.len(), 1);
    let returned_episode_id = episode_ids.first().unwrap().as_str().unwrap();
    let returned_episode_id = Uuid::parse_str(returned_episode_id).unwrap();
    assert_eq!(returned_episode_id, episode_id);

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // Check if ClickHouse is ok - BatchModelInference Table
    let clickhouse = get_clickhouse().await;
    let result = select_batch_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - BatchModelInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id = Uuid::parse_str(id).unwrap();
    assert_eq!(id, inference_id);

    let retrieved_batch_id = result.get("batch_id").unwrap().as_str().unwrap();
    let retrieved_batch_id = Uuid::parse_str(retrieved_batch_id).unwrap();
    assert_eq!(retrieved_batch_id, batch_id);

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

    let input_messages = result.get("input_messages").unwrap().as_str().unwrap();
    let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
    let expected_input_messages = vec![RequestMessage {
        role: Role::User,
        content: vec!["What is the capital city of Japan?".to_string().into()],
    }];
    assert_eq!(input_messages, expected_input_messages);

    let system = result.get("system").unwrap().as_str().unwrap();
    assert_eq!(
        system,
        "You are a helpful and friendly assistant named Dr. Mehta"
    );

    let tool_params = result.get("tool_params");
    assert_eq!(tool_params, Some(&Value::Null));

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

    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);

    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let output_schema = result.get("output_schema");
    assert_eq!(output_schema, Some(&Value::Null));

    let tags = result.get("tags").unwrap().as_object().unwrap();
    assert_eq!(tags.len(), 1);
    assert_eq!(tags.get("key").unwrap().as_str().unwrap(), tag_value);

    // Check if ClickHouse is ok - BatchRequest Table
    let result = select_latest_batch_request_clickhouse(&clickhouse, batch_id)
        .await
        .unwrap();

    println!("ClickHouse - BatchRequest: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    Uuid::parse_str(id).unwrap();

    let retrieved_batch_id = result.get("batch_id").unwrap().as_str().unwrap();
    let retrieved_batch_id = Uuid::parse_str(retrieved_batch_id).unwrap();
    assert_eq!(retrieved_batch_id, batch_id);
    let batch_params = result.get("batch_params").unwrap().as_str().unwrap();
    let _batch_params: Value = serde_json::from_str(batch_params).unwrap();
    // We can't check that the batch params are exactly the same because they vary per-provider
    // We will check that they are valid by using them instead.
    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, provider.model_name);

    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, provider.model_provider_name);

    let status = result.get("status").unwrap().as_str().unwrap();
    assert_eq!(status, "pending");

    let errors = result.get("errors").unwrap().as_object().unwrap();
    assert_eq!(errors.len(), 0);
}
