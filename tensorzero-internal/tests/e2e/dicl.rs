#![allow(clippy::print_stdout)]

use futures::StreamExt;
use reqwest::{Client, StatusCode};
use reqwest_eventsource::{Event, RequestBuilderExt};
use serde_json::{json, Value};
use std::time::Duration;
use tensorzero_internal::{
    clickhouse::{test_helpers::select_json_inference_clickhouse, ClickHouseConnectionInfo},
    embeddings::{EmbeddingProvider, EmbeddingProviderConfig, EmbeddingRequest},
    endpoints::inference::InferenceCredentials,
    inference::types::{
        ContentBlock, ContentBlockChatOutput, JsonInferenceOutput, RequestMessage, ResolvedInput,
        ResolvedInputMessage, ResolvedInputMessageContent, Role,
    },
};
use tokio::time::sleep;
use uuid::Uuid;

use crate::common::get_gateway_endpoint;
use tensorzero_internal::clickhouse::test_helpers::{
    get_clickhouse, select_chat_inference_clickhouse, select_model_inferences_clickhouse,
};

#[tokio::test]
pub async fn test_dicl_inference_request_no_examples_empty_dicl() {
    test_dicl_inference_request_no_examples("empty_dicl").await;
}

#[tokio::test]
pub async fn test_dicl_inference_request_no_examples_empty_dicl_extra_body() {
    test_dicl_inference_request_no_examples("empty_dicl_extra_body").await;
}

// This model is identical to `empty_dicl`, but it specified the embedding model
// using shorthand
#[tokio::test]
pub async fn test_dicl_inference_request_no_examples_empty_dicl_shorthand() {
    test_dicl_inference_request_no_examples("empty_dicl_shorthand").await;
}

#[tokio::test]
async fn test_dicl_reject_unknown_content_block() {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "variant_name": "empty_dicl",
        "episode_id": episode_id,
        "input":
            {
               "system": {"assistant_name": "Dr. Mehta"},
               "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "value": "What is the name of the capital city of Japan?"},
                        {"type": "unknown", "model_provider_name": "tensorzero::model_name::gpt-4o-mini-2024-07-18::provider_name::openai", "data": {"type": "text", "text": "My extra openai text"}}
                    ]
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

    // Check that the API response is correct
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let response_json = response.json::<Value>().await.unwrap();
    assert!(
        response_json
            .get("error")
            .unwrap()
            .as_str()
            .unwrap()
            .contains(" Unsupported content block type `unknown` for provider `dicl`"),
        "Unexpected error message: {response_json:#?}"
    );

    println!("API response: {response_json:#?}");
}

#[tokio::test]
async fn test_dicl_reject_image_content_block() {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "variant_name": "empty_dicl",
        "episode_id": episode_id,
        "input":
            {
               "system": {"assistant_name": "Dr. Mehta"},
               "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "value": "What is the name of the capital city of Japan?"},
                        {"type": "image", "mime_type": "image/jpeg", "data": "abc"}
                    ]
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
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let response_json = response.json::<Value>().await.unwrap();
    assert!(
        response_json
            .get("error")
            .unwrap()
            .as_str()
            .unwrap()
            .contains(" Unsupported content block type `image` for provider `dicl`"),
        "Unexpected error message: {response_json:#?}"
    );

    println!("API response: {response_json:#?}");
}

pub async fn test_dicl_inference_request_no_examples(dicl_variant_name: &str) {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "variant_name": dicl_variant_name,
        "episode_id": episode_id,
        "input":
            {
               "system": {"assistant_name": "Dr. Mehta"},
               "messages": [
                {
                    "role": "user",
                    "content": "What is the name of the capital city of Japan?"
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
    assert_eq!(variant_name, dicl_variant_name);

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
    sleep(Duration::from_secs(1)).await;

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
    assert_eq!(variant_name, dicl_variant_name);

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
                "content": [{"type": "text", "value": "What is the name of the capital city of Japan?"}]
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
    let result = select_model_inferences_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    println!("ClickHouse - ModelInference: {result:#?}");
    assert_eq!(result.len(), 2);
    for model_inference in result {
        let model_name = model_inference.get("model_name").unwrap().as_str().unwrap();
        match model_name {
            "gpt-4o-mini-2024-07-18" => {
                // The LLM call should generate output tokens
                assert!(
                    model_inference
                        .get("output_tokens")
                        .unwrap()
                        .as_u64()
                        .unwrap()
                        > 0
                );

                let raw_response = model_inference
                    .get("raw_response")
                    .unwrap()
                    .as_str()
                    .unwrap();
                assert!(raw_response.to_lowercase().contains("tokyo"));

                if dicl_variant_name == "empty_dicl_extra_body" {
                    let raw_request = model_inference
                        .get("raw_request")
                        .unwrap()
                        .as_str()
                        .unwrap();
                    let raw_request: Value = serde_json::from_str(raw_request).unwrap();
                    let temperature = raw_request.get("temperature").unwrap().as_f64().unwrap();
                    assert_eq!(temperature, 0.123);
                }
            }
            "openai::text-embedding-3-small" | "text-embedding-3-small" => {
                // The embedding call should not generate any output tokens
                assert!(model_inference.get("output_tokens").unwrap().is_null());
            }
            _ => {
                panic!("Unexpected model: {}", model_name);
            }
        }
        let model_inference_id = model_inference.get("id").unwrap().as_str().unwrap();
        assert!(Uuid::parse_str(model_inference_id).is_ok());

        let inference_id_result = model_inference
            .get("inference_id")
            .unwrap()
            .as_str()
            .unwrap();
        let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
        assert_eq!(inference_id_result, inference_id);

        let raw_request = model_inference
            .get("raw_request")
            .unwrap()
            .as_str()
            .unwrap();
        assert!(raw_request.to_lowercase().contains("japan"));
        assert!(
            serde_json::from_str::<Value>(raw_request).is_ok(),
            "raw_request is not a valid JSON"
        );

        let raw_response = model_inference
            .get("raw_response")
            .unwrap()
            .as_str()
            .unwrap();
        assert!(serde_json::from_str::<Value>(raw_response).is_ok());

        let input_tokens = model_inference
            .get("input_tokens")
            .unwrap()
            .as_u64()
            .unwrap();
        assert!(input_tokens > 0);
        let response_time_ms = model_inference
            .get("response_time_ms")
            .unwrap()
            .as_u64()
            .unwrap();
        assert!(response_time_ms > 0);
        assert!(model_inference.get("ttft_ms").unwrap().is_null());
    }
}

// Stick an embedding example into the database
async fn embed_insert_example(
    clickhouse: &ClickHouseConnectionInfo,
    input: ResolvedInput,
    output: String,
    function_name: &str,
    variant_name: &str,
) {
    let provider_config_serialized = r#"
    type = "openai"
    model_name = "text-embedding-3-small"
    "#;
    let provider_config: EmbeddingProviderConfig = toml::from_str(provider_config_serialized)
        .expect("Failed to deserialize EmbeddingProviderConfig");

    let client = Client::new();
    let request = EmbeddingRequest {
        input: serde_json::to_string(&input).unwrap(),
    };
    let api_keys = InferenceCredentials::default();
    let response = provider_config
        .embed(&request, &client, &api_keys)
        .await
        .unwrap();

    let id = Uuid::now_v7();
    let embedding = response.embedding;

    let input_string = serde_json::to_string(&input).unwrap();
    let row = serde_json::json!({
        "id": id,
        "function_name": function_name,
        "variant_name": variant_name,
        "input": input_string,
        "output": output,
        "embedding": embedding,
    });

    let query = format!(
        "INSERT INTO DynamicInContextLearningExample\n\
        SETTINGS async_insert=1, wait_for_async_insert=1\n\
        FORMAT JSONEachRow\n\
        {}",
        serde_json::to_string(&row).unwrap()
    );

    clickhouse.run_query(query, None).await.unwrap();
}

/// Testing a DICL variant
/// Trying to get the LLM to learn that Pinocchio is a liar from examples
#[tokio::test]
pub async fn test_dicl_inference_request_simple() {
    let clickhouse = get_clickhouse().await;
    let episode_id = Uuid::now_v7();
    let variant_name = "dicl";
    let function_name = "basic_test";
    // Delete any existing examples for this function and variant
    let delete_query = format!(
        "ALTER TABLE DynamicInContextLearningExample DELETE WHERE function_name = '{}' AND variant_name = '{}'",
            function_name, variant_name
        );
    clickhouse.run_query(delete_query, None).await.unwrap();

    // Insert examples into the database
    let mut tasks = Vec::new();

    let input = ResolvedInput {
        system: Some(json!({"assistant_name": "Dr. Mehta"})),
        messages: vec![ResolvedInputMessage {
            role: Role::User,
            content: vec![ResolvedInputMessageContent::Text {
                value: json!("What is the boiling point of water?"),
            }],
        }],
    };
    let output: Vec<ContentBlockChatOutput> = vec!["100 degrees Celsius".to_string().into()];
    let output_string = serde_json::to_string(&output).unwrap();

    tasks.push(embed_insert_example(
        &clickhouse,
        input,
        output_string,
        function_name,
        variant_name,
    ));

    let input = ResolvedInput {
        system: Some(json!({"assistant_name": "Pinocchio"})),
        messages: vec![ResolvedInputMessage {
            role: Role::User,
            content: vec![ResolvedInputMessageContent::Text {
                value: json!("What the capital city of India?"),
            }],
        }],
    };
    let output: Vec<ContentBlockChatOutput> =
        vec!["Ahmedabad (nose grows 3 inches)".to_string().into()];
    let output_string = serde_json::to_string(&output).unwrap();

    tasks.push(embed_insert_example(
        &clickhouse,
        input,
        output_string,
        function_name,
        variant_name,
    ));

    let input = ResolvedInput {
        system: Some(json!({"assistant_name": "Pinocchio"})),
        messages: vec![ResolvedInputMessage {
            role: Role::User,
            content: vec![ResolvedInputMessageContent::Text {
                value: json!("What is an example of a computationally hard problem?"),
            }],
        }],
    };
    let output: Vec<ContentBlockChatOutput> = vec![
        "Finding the median of an unsorted list of numbers (nose grows 4 inches)"
            .to_string()
            .into(),
    ];
    let output_string = serde_json::to_string(&output).unwrap();
    tasks.push(embed_insert_example(
        &clickhouse,
        input,
        output_string,
        function_name,
        variant_name,
    ));

    let input = ResolvedInput {
        system: Some(json!({"assistant_name": "Pinocchio"})),
        messages: vec![ResolvedInputMessage {
            role: Role::User,
            content: vec![ResolvedInputMessageContent::Text {
                value: json!("Who wrote Lord of the Rings?"),
            }],
        }],
    };
    let output: Vec<ContentBlockChatOutput> =
        vec!["J.K. Rowling (nose grows 5 inches)".to_string().into()];
    let output_string = serde_json::to_string(&output).unwrap();
    tasks.push(embed_insert_example(
        &clickhouse,
        input,
        output_string,
        function_name,
        variant_name,
    ));

    // Join all tasks and wait for them to complete
    futures::future::join_all(tasks).await;

    // Wait for 1 second
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Launch the dicl inference request
    let payload = json!({
        "function_name": function_name,
        "variant_name": variant_name,
        "episode_id": episode_id,
        "input":
            {
               "system": {"assistant_name": "Pinocchio"},
               "messages": [
                {
                    "role": "user",
                    "content": "Who was the author of the Harry Potter series?"
                }
            ]},
        "stream": false,
        "include_original_response": true,
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

    let response_variant_name = response_json.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(response_variant_name, variant_name);

    let content = response_json.get("content").unwrap().as_array().unwrap();
    assert_eq!(content.len(), 1);
    let content_block = content.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let content = content_block.get("text").unwrap().as_str().unwrap();
    assert!(!content.to_lowercase().contains("rowling"));
    assert!(content.to_lowercase().contains("nose"));

    let usage = response_json.get("usage").unwrap();
    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(output_tokens > 0);

    let original_response = response_json.get("original_response").unwrap();
    let original_response_json: serde_json::Value =
        serde_json::from_str(original_response.as_str().unwrap()).unwrap();
    assert_eq!(original_response_json["model"], "gpt-4o-mini-2024-07-18");
    assert!(
        original_response_json.get("choices").is_some(),
        "Unexpected original_response: {original_response}"
    );

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

    let retrieved_variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(retrieved_variant_name, variant_name);

    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input = json!({
        "system": {"assistant_name": "Pinocchio"},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "value": "Who was the author of the Harry Potter series?"}]
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
    let result = select_model_inferences_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    println!("ClickHouse - ModelInference: {result:#?}");
    assert_eq!(result.len(), 2);
    for model_inference in result {
        let model_name = model_inference.get("model_name").unwrap().as_str().unwrap();
        let input_messages = model_inference
            .get("input_messages")
            .unwrap()
            .as_str()
            .unwrap();
        let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
        let output = model_inference.get("output").unwrap().as_str().unwrap();
        let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
        match model_name {
            "gpt-4o-mini-2024-07-18" => {
                // The LLM call should generate output tokens
                assert!(
                    model_inference
                        .get("output_tokens")
                        .unwrap()
                        .as_u64()
                        .unwrap()
                        > 0
                );

                let raw_response = model_inference
                    .get("raw_response")
                    .unwrap()
                    .as_str()
                    .unwrap();
                assert!(!raw_response.to_lowercase().contains("rowling"));
                assert!(raw_response.to_lowercase().contains("nose"));
                let system = model_inference.get("system").unwrap().as_str().unwrap();
                assert_eq!(system, "You are tasked with learning by induction and then solving a problem below. You will be shown several examples of inputs followed by outputs. Then, in the same format you will be given one last set of inputs. Your job is to use the provided examples to inform your response to the last set of inputs.");
                assert_eq!(input_messages.len(), 7);
                assert_eq!(output.len(), 1);
                match &output[0] {
                    ContentBlock::Text(text) => {
                        assert!(text.text.to_lowercase().contains("nose"));
                    }
                    _ => {
                        panic!("Expected a text block, got {:?}", output[0]);
                    }
                }
            }
            "text-embedding-3-small" => {
                // The embedding call should not generate any output tokens
                assert!(model_inference.get("output_tokens").unwrap().is_null());
                assert!(model_inference.get("system").unwrap().is_null());
                assert_eq!(input_messages.len(), 1);
                assert_eq!(output.len(), 0);
            }
            _ => {
                panic!("Unexpected model: {}", model_name);
            }
        }
        let model_inference_id = model_inference.get("id").unwrap().as_str().unwrap();
        assert!(Uuid::parse_str(model_inference_id).is_ok());

        let inference_id_result = model_inference
            .get("inference_id")
            .unwrap()
            .as_str()
            .unwrap();
        let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
        assert_eq!(inference_id_result, inference_id);

        let raw_request = model_inference
            .get("raw_request")
            .unwrap()
            .as_str()
            .unwrap();
        assert!(
            serde_json::from_str::<Value>(raw_request).is_ok(),
            "raw_request is not a valid JSON"
        );

        let raw_response = model_inference
            .get("raw_response")
            .unwrap()
            .as_str()
            .unwrap();
        assert!(serde_json::from_str::<Value>(raw_response).is_ok());

        let input_tokens = model_inference
            .get("input_tokens")
            .unwrap()
            .as_u64()
            .unwrap();
        assert!(input_tokens > 0);
        let response_time_ms = model_inference
            .get("response_time_ms")
            .unwrap()
            .as_u64()
            .unwrap();
        assert!(response_time_ms > 0);
        assert!(model_inference.get("ttft_ms").unwrap().is_null());
    }

    // Launch the dicl inference request with streaming
    let payload = json!({
        "function_name": function_name,
        "variant_name": variant_name,
        "episode_id": episode_id,
        "stream": true,
        "input": {
            "system": {"assistant_name": "Pinocchio"},
            "messages": [
                {
                    "role": "user",
                    "content": "Who was the author of the Harry Potter series?"
                }
            ]
        }
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
    for chunk in chunks.clone() {
        let chunk_json: Value = serde_json::from_str(&chunk).unwrap();

        println!("API response chunk: {chunk_json:#?}");

        let chunk_inference_id = chunk_json.get("inference_id").unwrap().as_str().unwrap();
        let chunk_inference_id = Uuid::parse_str(chunk_inference_id).unwrap();
        match inference_id {
            Some(existing_id) => {
                assert_eq!(existing_id, chunk_inference_id);
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
    }

    let inference_id = inference_id.unwrap();
    assert!(!full_content.to_lowercase().contains("rowling"));
    assert!(full_content.to_lowercase().contains("nose"));

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

    let retrieved_variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(retrieved_variant_name, variant_name);

    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input = json!({
        "system": {"assistant_name": "Pinocchio"},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "value": "Who was the author of the Harry Potter series?"}]
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
    // Check the ModelInference Table
    let result = select_model_inferences_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    println!("ClickHouse - ModelInference: {result:#?}");
    assert_eq!(result.len(), 2);
    for model_inference in result {
        let model_name = model_inference.get("model_name").unwrap().as_str().unwrap();
        let input_messages = model_inference
            .get("input_messages")
            .unwrap()
            .as_str()
            .unwrap();
        let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
        let output = model_inference.get("output").unwrap().as_str().unwrap();
        let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
        match model_name {
            "gpt-4o-mini-2024-07-18" => {
                // The LLM call should generate output tokens
                assert!(
                    model_inference
                        .get("output_tokens")
                        .unwrap()
                        .as_u64()
                        .unwrap()
                        > 0
                );

                let raw_response = model_inference
                    .get("raw_response")
                    .unwrap()
                    .as_str()
                    .unwrap();
                assert!(!raw_response.to_lowercase().contains("rowling"));
                assert!(raw_response.to_lowercase().contains("nose"));
                let system = model_inference.get("system").unwrap().as_str().unwrap();
                assert_eq!(system, "You are tasked with learning by induction and then solving a problem below. You will be shown several examples of inputs followed by outputs. Then, in the same format you will be given one last set of inputs. Your job is to use the provided examples to inform your response to the last set of inputs.");
                assert_eq!(input_messages.len(), 7);
                assert_eq!(output.len(), 1);
                match &output[0] {
                    ContentBlock::Text(text) => {
                        assert!(text.text.to_lowercase().contains("nose"));
                    }
                    _ => {
                        panic!("Expected a text block, got {:?}", output[0]);
                    }
                }
                assert!(!model_inference.get("ttft_ms").unwrap().is_null());
            }
            "text-embedding-3-small" => {
                // The embedding call should not generate any output tokens
                assert!(model_inference.get("output_tokens").unwrap().is_null());
                assert!(model_inference.get("system").unwrap().is_null());
                assert_eq!(input_messages.len(), 1);
                assert_eq!(output.len(), 0);
            }
            _ => {
                panic!("Unexpected model: {}", model_name);
            }
        }
        let model_inference_id = model_inference.get("id").unwrap().as_str().unwrap();
        assert!(Uuid::parse_str(model_inference_id).is_ok());

        let inference_id_result = model_inference
            .get("inference_id")
            .unwrap()
            .as_str()
            .unwrap();
        let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
        assert_eq!(inference_id_result, inference_id);

        let raw_request = model_inference
            .get("raw_request")
            .unwrap()
            .as_str()
            .unwrap();
        assert!(raw_request.to_lowercase().contains("potter"));
        assert!(
            serde_json::from_str::<Value>(raw_request).is_ok(),
            "raw_request is not a valid JSON"
        );
        // Raw response is going to be json lines for streaming responses, so we'll skip this here

        let input_tokens = model_inference
            .get("input_tokens")
            .unwrap()
            .as_u64()
            .unwrap();
        assert!(input_tokens > 0);
        let response_time_ms = model_inference
            .get("response_time_ms")
            .unwrap()
            .as_u64()
            .unwrap();
        assert!(response_time_ms > 0);
    }
}

#[tokio::test]
async fn test_dicl_json_request() {
    let clickhouse = get_clickhouse().await;
    let episode_id = Uuid::now_v7();
    let variant_name = "dicl";
    let function_name = "json_success";
    // Delete any existing examples for this function and variant
    let delete_query = format!(
        "ALTER TABLE DynamicInContextLearningExample DELETE WHERE function_name = '{}' AND variant_name = '{}'",
            function_name, variant_name
        );
    clickhouse.run_query(delete_query, None).await.unwrap();

    // Insert examples into the database
    let mut tasks = Vec::new();

    let input = ResolvedInput {
        system: Some(json!({"assistant_name": "Dr. Mehta"})),
        messages: vec![ResolvedInputMessage {
            role: Role::User,
            content: vec![ResolvedInputMessageContent::Text {
                value: json!({"country": "Canada"}),
            }],
        }],
    };
    let output = JsonInferenceOutput {
        raw: "{\"answer\": \"Ottawa\"}".to_string(),
        parsed: Some(json!({"answer": "Ottawa"})),
    };
    let output_string = serde_json::to_string(&output).unwrap();

    tasks.push(embed_insert_example(
        &clickhouse,
        input,
        output_string,
        function_name,
        variant_name,
    ));

    let input = ResolvedInput {
        system: Some(json!({"assistant_name": "Pinocchio"})),
        messages: vec![ResolvedInputMessage {
            role: Role::User,
            content: vec![ResolvedInputMessageContent::Text {
                value: json!({"country": "India"}),
            }],
        }],
    };
    let output = JsonInferenceOutput {
        raw: "{\"answer\": \"Ahmedabad (nose grows 3 inches)\"}".to_string(),
        parsed: Some(json!({"answer": "Ahmedabad (nose grows 3 inches)"})),
    };
    let output_string = serde_json::to_string(&output).unwrap();

    tasks.push(embed_insert_example(
        &clickhouse,
        input,
        output_string,
        function_name,
        variant_name,
    ));

    let input = ResolvedInput {
        system: Some(json!({"assistant_name": "Pinocchio"})),
        messages: vec![ResolvedInputMessage {
            role: Role::User,
            content: vec![ResolvedInputMessageContent::Text {
                value: json!({"country": "USA"}),
            }],
        }],
    };
    let output = JsonInferenceOutput {
        raw: "{\"answer\": \"New York City (nose grows 4 inches)\"}".to_string(),
        parsed: Some(json!({"answer": "New York City (nose grows 4 inches)"})),
    };
    let output_string = serde_json::to_string(&output).unwrap();
    tasks.push(embed_insert_example(
        &clickhouse,
        input,
        output_string,
        function_name,
        variant_name,
    ));

    let input = ResolvedInput {
        system: Some(json!({"assistant_name": "Pinocchio"})),
        messages: vec![ResolvedInputMessage {
            role: Role::User,
            content: vec![ResolvedInputMessageContent::Text {
                value: json!({"country": "England"}),
            }],
        }],
    };
    let output = JsonInferenceOutput {
        raw: "{\"answer\": \"Liverpool (nose grows 5 inches)\"}".to_string(),
        parsed: Some(json!({"answer": "Liverpool (nose grows 5 inches)"})),
    };
    let output_string = serde_json::to_string(&output).unwrap();
    tasks.push(embed_insert_example(
        &clickhouse,
        input,
        output_string,
        function_name,
        variant_name,
    ));

    // Join all tasks and wait for them to complete
    futures::future::join_all(tasks).await;

    // Wait for 1 second
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Launch the dicl inference request
    let payload = json!({
        "function_name": function_name,
        "variant_name": variant_name,
        "episode_id": episode_id,
        "input":
            {
               "system": {"assistant_name": "Pinocchio"},
               "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "value": {"country": "Brazil"}}]
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

    let response_variant_name = response_json.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(response_variant_name, variant_name);

    let content = response_json.get("output").unwrap().get("parsed").unwrap();
    let answer = content.get("answer").unwrap().as_str().unwrap();
    assert!(!answer.to_lowercase().contains("brasilia"));
    assert!(answer.to_lowercase().contains("nose"));

    let usage = response_json.get("usage").unwrap();
    let input_tokens = usage.get("input_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 0);
    let output_tokens = usage.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(output_tokens > 0);

    // Sleep to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check if ClickHouse is ok - ChatInference Table
    let clickhouse = get_clickhouse().await;
    let result = select_json_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - JsonInference: {result:#?}");

    let id = result.get("id").unwrap().as_str().unwrap();
    let id = Uuid::parse_str(id).unwrap();
    assert_eq!(id, inference_id);

    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, payload["function_name"]);

    let retrieved_variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(retrieved_variant_name, variant_name);

    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, episode_id);

    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input = json!({
        "system": {"assistant_name": "Pinocchio"},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "value": {"country": "Brazil"}}]
            }
        ]
    });
    assert_eq!(input, correct_input);

    let json_output = serde_json::from_str::<JsonInferenceOutput>(
        result.get("output").unwrap().as_str().unwrap(),
    )
    .unwrap();
    let clickhouse_answer = json_output.parsed.unwrap();
    let clickhouse_answer = clickhouse_answer.get("answer").unwrap().as_str().unwrap();
    assert_eq!(clickhouse_answer, answer);

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
    let result = select_model_inferences_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    println!("ClickHouse - ModelInference: {result:#?}");
    assert_eq!(result.len(), 2);
    for model_inference in result {
        let model_name = model_inference.get("model_name").unwrap().as_str().unwrap();
        let input_messages = model_inference
            .get("input_messages")
            .unwrap()
            .as_str()
            .unwrap();
        let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
        let output = model_inference.get("output").unwrap().as_str().unwrap();
        let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();
        match model_name {
            "gpt-4o-mini-2024-07-18" => {
                // The LLM call should generate output tokens
                assert!(
                    model_inference
                        .get("output_tokens")
                        .unwrap()
                        .as_u64()
                        .unwrap()
                        > 0
                );

                let raw_response = model_inference
                    .get("raw_response")
                    .unwrap()
                    .as_str()
                    .unwrap();
                assert!(!raw_response.to_lowercase().contains("brasilia"));
                assert!(raw_response.to_lowercase().contains("nose"));
                let system = model_inference.get("system").unwrap().as_str().unwrap();
                assert_eq!(system, "You are a helpful and friendly assistant with a name that will be provided to you.\n\nPlease answer the questions in a JSON with key \"answer\".\n\nDo not include any other text than the JSON object. Do not include \"```json\" or \"```\" or anything else.\n\nExample Response:\n\n{\n    \"answer\": \"42\"\n}\n");
                assert_eq!(input_messages.len(), 7);
                assert_eq!(output.len(), 1);
                match &output[0] {
                    ContentBlock::Text(text) => {
                        assert!(!text.text.to_lowercase().contains("brasilia"));
                        assert!(text.text.to_lowercase().contains("nose"));
                    }
                    _ => {
                        panic!("Expected a text block, got {:?}", output[0]);
                    }
                }
            }
            "text-embedding-3-small" => {
                // The embedding call should not generate any output tokens
                assert!(model_inference.get("output_tokens").unwrap().is_null());
                assert!(model_inference.get("system").unwrap().is_null());
                assert_eq!(input_messages.len(), 1);
                assert_eq!(output.len(), 0);
            }
            _ => {
                panic!("Unexpected model: {}", model_name);
            }
        }
        let model_inference_id = model_inference.get("id").unwrap().as_str().unwrap();
        assert!(Uuid::parse_str(model_inference_id).is_ok());

        let inference_id_result = model_inference
            .get("inference_id")
            .unwrap()
            .as_str()
            .unwrap();
        let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
        assert_eq!(inference_id_result, inference_id);

        let raw_request = model_inference
            .get("raw_request")
            .unwrap()
            .as_str()
            .unwrap();
        assert!(
            serde_json::from_str::<Value>(raw_request).is_ok(),
            "raw_request is not a valid JSON"
        );

        let raw_response = model_inference
            .get("raw_response")
            .unwrap()
            .as_str()
            .unwrap();
        assert!(serde_json::from_str::<Value>(raw_response).is_ok());

        let input_tokens = model_inference
            .get("input_tokens")
            .unwrap()
            .as_u64()
            .unwrap();
        assert!(input_tokens > 0);
        let response_time_ms = model_inference
            .get("response_time_ms")
            .unwrap()
            .as_u64()
            .unwrap();
        assert!(response_time_ms > 0);
        assert!(model_inference.get("ttft_ms").unwrap().is_null());
    }
}
