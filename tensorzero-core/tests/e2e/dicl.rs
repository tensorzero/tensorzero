use futures::StreamExt;
use reqwest::{Client, StatusCode};
use reqwest_eventsource::{Event, RequestBuilderExt};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tensorzero_core::{
    cache::{CacheEnabledMode, CacheOptions},
    config::provider_types::ProviderTypesConfig,
    db::{
        clickhouse::{test_helpers::select_json_inference_clickhouse, ClickHouseConnectionInfo},
        postgres::PostgresConnectionInfo,
    },
    embeddings::{EmbeddingEncodingFormat, EmbeddingRequest, UninitializedEmbeddingProviderConfig},
    endpoints::inference::{InferenceClients, InferenceCredentials},
    http::TensorzeroHttpClient,
    inference::types::{
        Arguments, ContentBlockChatOutput, JsonInferenceOutput, ResolvedInput,
        ResolvedInputMessage, ResolvedInputMessageContent, Role, StoredContentBlock,
        StoredRequestMessage, System, Template, Text,
    },
    model_table::ProviderTypeDefaultCredentials,
    rate_limiting::ScopeInfo,
};
use tokio::time::sleep;
use uuid::Uuid;

use crate::common::get_gateway_endpoint;
use tensorzero::{
    ClientInferenceParams, ClientInput, ClientInputMessage, ClientInputMessageContent,
    InferenceOutput, InferenceResponse,
};
use tensorzero_core::db::clickhouse::test_helpers::{
    get_clickhouse, select_chat_inference_clickhouse, select_model_inferences_clickhouse,
};
use tensorzero_core::inference::types::TextKind;

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
                        {"type": "text", "text": "What is the name of the capital city of Japan?"},
                        {"type": "unknown", "model_name": "gpt-4o-mini-2024-07-18", "provider_name": "openai", "data": {"type": "text", "text": "My extra openai text"}}
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
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let response_json = response.json::<Value>().await.unwrap();
    assert!(
        response_json
            .get("error")
            .unwrap()
            .as_str()
            .unwrap()
            .contains("Unsupported content block type `unknown` for provider `dicl`"),
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
                        {"type": "text", "text": "What is the name of the capital city of Japan?"},
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

    // Check that the API response is as expected
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let response_json = response.json::<Value>().await.unwrap();
    assert!(
        response_json
            .get("error")
            .unwrap()
            .as_str()
            .unwrap()
            .contains("Unsupported content block type `image` for provider `dicl`"),
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
                "content": [{"type": "text", "text": "What is the name of the capital city of Japan?"}]
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
                panic!("Unexpected model: {model_name}");
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
    let provider_config =
        toml::from_str::<UninitializedEmbeddingProviderConfig>(provider_config_serialized)
            .expect("Failed to deserialize EmbeddingProviderConfig")
            .load(
                &ProviderTypesConfig::default(),
                Arc::from("good".to_string()),
                &ProviderTypeDefaultCredentials::default(),
                TensorzeroHttpClient::new_testing().unwrap(),
            )
            .await
            .unwrap();

    let client = TensorzeroHttpClient::new_testing().unwrap();
    let request = EmbeddingRequest {
        input: serde_json::to_string(&input.clone().into_stored_input())
            .unwrap()
            .into(),
        dimensions: None,
        encoding_format: EmbeddingEncodingFormat::Float,
    };
    let api_keys = InferenceCredentials::default();
    let clients = InferenceClients {
        http_client: client.clone(),
        clickhouse_connection_info: clickhouse.clone(),
        postgres_connection_info: PostgresConnectionInfo::Disabled,
        credentials: Arc::new(api_keys),
        cache_options: CacheOptions {
            max_age_s: None,
            enabled: CacheEnabledMode::On,
        },
        tags: Arc::new(Default::default()),
        rate_limiting_config: Arc::new(Default::default()),
        otlp_config: Default::default(),
        deferred_tasks: tokio_util::task::TaskTracker::new(),
        scope_info: ScopeInfo {
            tags: Arc::new(HashMap::new()),
            api_key_public_id: None,
        },
    };
    let response = provider_config
        .embed(&request, &clients, &(&provider_config).into())
        .await
        .unwrap();

    let id = Uuid::now_v7();
    let embedding = &response.embeddings[0];

    let input_string = serde_json::to_string(&input.clone().into_stored_input()).unwrap();
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

    clickhouse
        .run_query_synchronous_no_params(query)
        .await
        .unwrap();
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
        "ALTER TABLE DynamicInContextLearningExample DELETE WHERE function_name = '{function_name}' AND variant_name = '{variant_name}'"
        );
    clickhouse
        .run_query_synchronous_no_params(delete_query)
        .await
        .unwrap();

    // Insert examples into the database
    let mut tasks = Vec::new();

    let input = ResolvedInput {
        system: Some(System::Template(Arguments(serde_json::Map::from_iter([(
            "assistant_name".to_string(),
            "Dr. Mehta".into(),
        )])))),
        messages: vec![ResolvedInputMessage {
            role: Role::User,
            content: vec![ResolvedInputMessageContent::Text(Text {
                text: "What is the boiling point of water?".to_string(),
            })],
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
        system: Some(System::Template(Arguments(serde_json::Map::from_iter([(
            "assistant_name".to_string(),
            "Pinocchio".into(),
        )])))),
        messages: vec![ResolvedInputMessage {
            role: Role::User,
            content: vec![ResolvedInputMessageContent::Text(Text {
                text: "What the capital city of India?".to_string(),
            })],
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
        system: Some(System::Template(Arguments(serde_json::Map::from_iter([(
            "assistant_name".to_string(),
            "Pinocchio".into(),
        )])))),
        messages: vec![ResolvedInputMessage {
            role: Role::User,
            content: vec![ResolvedInputMessageContent::Text(Text {
                text: "What is an example of a computationally hard problem?".to_string(),
            })],
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
        system: Some(System::Template(Arguments(serde_json::Map::from_iter([(
            "assistant_name".to_string(),
            "Pinocchio".into(),
        )])))),
        messages: vec![ResolvedInputMessage {
            role: Role::User,
            content: vec![ResolvedInputMessageContent::Text(Text {
                text: "Who wrote Lord of the Rings?".to_string(),
            })],
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
                "content": [{"type": "text", "text": "Who was the author of the Harry Potter series?"}]
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
        let input_messages: Vec<StoredRequestMessage> =
            serde_json::from_str(input_messages).unwrap();
        let output = model_inference.get("output").unwrap().as_str().unwrap();
        let output: Vec<StoredContentBlock> = serde_json::from_str(output).unwrap();
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
                    StoredContentBlock::Text(text) => {
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
                panic!("Unexpected model: {model_name}");
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
                "content": [{"type": "text", "text": "Who was the author of the Harry Potter series?"}]
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
        let input_messages: Vec<StoredRequestMessage> =
            serde_json::from_str(input_messages).unwrap();
        let output = model_inference.get("output").unwrap().as_str().unwrap();
        let output: Vec<StoredContentBlock> = serde_json::from_str(output).unwrap();
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
                    StoredContentBlock::Text(text) => {
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
                panic!("Unexpected model: {model_name}");
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
        "ALTER TABLE DynamicInContextLearningExample DELETE WHERE function_name = '{function_name}' AND variant_name = '{variant_name}'"
        );
    clickhouse
        .run_query_synchronous_no_params(delete_query)
        .await
        .unwrap();

    // Insert examples into the database
    let mut tasks = Vec::new();

    let input = ResolvedInput {
        system: Some(System::Template(Arguments(serde_json::Map::from_iter([(
            "assistant_name".to_string(),
            "Dr. Mehta".into(),
        )])))),
        messages: vec![ResolvedInputMessage {
            role: Role::User,
            content: vec![ResolvedInputMessageContent::Template(Template {
                name: "user".to_string(),
                arguments: Arguments(serde_json::Map::from_iter([(
                    "country".to_string(),
                    "Canada".into(),
                )])),
            })],
        }],
    };
    let output = JsonInferenceOutput {
        raw: Some("{\"answer\": \"Ottawa\"}".to_string()),
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
        system: Some(System::Template(Arguments(serde_json::Map::from_iter([(
            "assistant_name".to_string(),
            "Pinocchio".into(),
        )])))),
        messages: vec![ResolvedInputMessage {
            role: Role::User,
            content: vec![ResolvedInputMessageContent::Template(Template {
                name: "user".to_string(),
                arguments: Arguments(serde_json::Map::from_iter([(
                    "country".to_string(),
                    "India".into(),
                )])),
            })],
        }],
    };
    let output = JsonInferenceOutput {
        raw: Some("{\"answer\": \"Ahmedabad (nose grows 3 inches)\"}".to_string()),
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
        system: Some(System::Template(Arguments(serde_json::Map::from_iter([(
            "assistant_name".to_string(),
            "Pinocchio".into(),
        )])))),
        messages: vec![ResolvedInputMessage {
            role: Role::User,
            content: vec![ResolvedInputMessageContent::Template(Template {
                name: "user".to_string(),
                arguments: Arguments(serde_json::Map::from_iter([(
                    "country".to_string(),
                    "USA".into(),
                )])),
            })],
        }],
    };
    let output = JsonInferenceOutput {
        raw: Some("{\"answer\": \"New York City (nose grows 4 inches)\"}".to_string()),
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
        system: Some(System::Template(Arguments(serde_json::Map::from_iter([(
            "assistant_name".to_string(),
            "Pinocchio".into(),
        )])))),
        messages: vec![ResolvedInputMessage {
            role: Role::User,
            content: vec![ResolvedInputMessageContent::Template(Template {
                name: "user".to_string(),
                arguments: Arguments(serde_json::Map::from_iter([(
                    "country".to_string(),
                    "England".into(),
                )])),
            })],
        }],
    };
    let output = JsonInferenceOutput {
        raw: Some("{\"answer\": \"Liverpool (nose grows 5 inches)\"}".to_string()),
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
                    "content": [{"type": "template", "name": "user", "arguments": {"country": "Brazil"}}]
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
                "content": [{"type": "template", "name" : "user", "arguments": {"country": "Brazil"}}]
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
        let input_messages: Vec<StoredRequestMessage> =
            serde_json::from_str(input_messages).unwrap();
        let output = model_inference.get("output").unwrap().as_str().unwrap();
        let output: Vec<StoredContentBlock> = serde_json::from_str(output).unwrap();
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
                    StoredContentBlock::Text(text) => {
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
                panic!("Unexpected model: {model_name}");
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

/// Test that max_distance filters out all irrelevant examples, falling back to vanilla chat completion
#[tokio::test]
pub async fn test_dicl_max_distance_filters_all_examples() {
    let clickhouse = get_clickhouse().await;
    let episode_id = Uuid::now_v7();
    let variant_name = "dicl_max_distance_strict";
    let function_name = "basic_test";

    // Create embedded gateway with DICL variant that has strict max_distance
    let config = r#"
[functions.basic_test]
type = "chat"

[functions.basic_test.variants.dicl_max_distance_strict]
type = "experimental_dynamic_in_context_learning"
model = "openai::gpt-4o-mini-2024-07-18"
embedding_model = "openai::text-embedding-3-small"
k = 3
max_distance = 0.15
max_tokens = 100
"#;

    let gateway = tensorzero::test_helpers::make_embedded_gateway_with_config(config).await;

    // Delete any existing examples for this function and variant
    let delete_query = format!(
        "ALTER TABLE DynamicInContextLearningExample DELETE WHERE function_name = '{function_name}' AND variant_name = '{variant_name}'"
    );
    clickhouse
        .run_query_synchronous_no_params(delete_query)
        .await
        .unwrap();

    // Insert geography examples (countries and capitals)
    let mut tasks = Vec::new();

    let input = ResolvedInput {
        system: None,
        messages: vec![ResolvedInputMessage {
            role: Role::User,
            content: vec![ResolvedInputMessageContent::Text(Text {
                text: "What is the capital of France?".to_string(),
            })],
        }],
    };
    let output: Vec<ContentBlockChatOutput> = vec!["Paris".to_string().into()];
    let output_string = serde_json::to_string(&output).unwrap();
    tasks.push(embed_insert_example(
        &clickhouse,
        input,
        output_string,
        function_name,
        variant_name,
    ));

    let input = ResolvedInput {
        system: None,
        messages: vec![ResolvedInputMessage {
            role: Role::User,
            content: vec![ResolvedInputMessageContent::Text(Text {
                text: "What is the capital of Germany?".to_string(),
            })],
        }],
    };
    let output: Vec<ContentBlockChatOutput> = vec!["Berlin".to_string().into()];
    let output_string = serde_json::to_string(&output).unwrap();
    tasks.push(embed_insert_example(
        &clickhouse,
        input,
        output_string,
        function_name,
        variant_name,
    ));

    let input = ResolvedInput {
        system: None,
        messages: vec![ResolvedInputMessage {
            role: Role::User,
            content: vec![ResolvedInputMessageContent::Text(Text {
                text: "What is the capital of Italy?".to_string(),
            })],
        }],
    };
    let output: Vec<ContentBlockChatOutput> = vec!["Rome".to_string().into()];
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

    // Wait for 1 second for ClickHouse to process
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Query about a completely unrelated topic (programming/software)
    // The max_distance should filter out all geography examples due to high cosine distance
    let params = ClientInferenceParams {
        function_name: Some(function_name.to_string()),
        variant_name: Some(variant_name.to_string()),
        episode_id: Some(episode_id),
        input: ClientInput {
            system: None,
            messages: vec![ClientInputMessage {
                role: Role::User,
                content: vec![ClientInputMessageContent::Text(TextKind::Text {
                    text: "What programming language is used for web development?".to_string(),
                })],
            }],
        },
        ..Default::default()
    };

    let response = gateway.inference(params).await.unwrap();
    let InferenceOutput::NonStreaming(InferenceResponse::Chat(response)) = response else {
        panic!("Expected non-streaming chat response");
    };

    println!("API response: {response:#?}");

    let inference_id = response.inference_id;

    // Sleep to allow time for data to be inserted into ClickHouse
    sleep(Duration::from_secs(1)).await;

    // Check the ModelInference Table
    let clickhouse = get_clickhouse().await;
    let result = select_model_inferences_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");
    assert_eq!(result.len(), 2); // embedding + chat completion

    for model_inference in result {
        let model_name = model_inference.get("model_name").unwrap().as_str().unwrap();
        let input_messages = model_inference
            .get("input_messages")
            .unwrap()
            .as_str()
            .unwrap();
        let input_messages: Vec<StoredRequestMessage> =
            serde_json::from_str(input_messages).unwrap();

        match model_name {
            "openai::gpt-4o-mini-2024-07-18" => {
                // When all examples are filtered, should behave like vanilla chat completion
                // This means short input_messages (1-2 messages, not 7+ with examples)
                assert!(
                    input_messages.len() <= 2,
                    "Expected short input_messages for vanilla chat completion, got {}",
                    input_messages.len()
                );

                // System should always contain DICL system instructions
                let system = model_inference.get("system").unwrap().as_str().unwrap();
                assert!(system.contains("learning by induction"));
            }
            "openai::text-embedding-3-small" => {
                // The embedding call should have 1 input message
                assert_eq!(input_messages.len(), 1);
            }
            _ => {
                panic!("Unexpected model: {model_name}");
            }
        }
    }
}

/// Test that max_distance keeps relevant examples when cosine distance is below threshold
#[tokio::test]
pub async fn test_dicl_max_distance_keeps_relevant_examples() {
    let clickhouse = get_clickhouse().await;
    let episode_id = Uuid::now_v7();
    let variant_name = "dicl_max_distance_moderate";
    let function_name = "basic_test";

    // Create embedded gateway with DICL variant that has moderate max_distance
    let config = r#"
[functions.basic_test]
type = "chat"

[functions.basic_test.variants.dicl_max_distance_moderate]
type = "experimental_dynamic_in_context_learning"
model = "openai::gpt-4o-mini-2024-07-18"
embedding_model = "openai::text-embedding-3-small"
k = 3
max_distance = 0.6
max_tokens = 100
"#;

    let gateway = tensorzero::test_helpers::make_embedded_gateway_with_config(config).await;

    // Delete any existing examples for this function and variant
    let delete_query = format!(
        "ALTER TABLE DynamicInContextLearningExample DELETE WHERE function_name = '{function_name}' AND variant_name = '{variant_name}'"
    );
    clickhouse
        .run_query_synchronous_no_params(delete_query)
        .await
        .unwrap();

    let mut tasks = Vec::new();

    let input = ResolvedInput {
        system: None,
        messages: vec![ResolvedInputMessage {
            role: Role::User,
            content: vec![ResolvedInputMessageContent::Text(Text {
                text: "What the capital city of India?".to_string(),
            })],
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
        system: None,
        messages: vec![ResolvedInputMessage {
            role: Role::User,
            content: vec![ResolvedInputMessageContent::Text(Text {
                text: "What is an example of a computationally hard problem?".to_string(),
            })],
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
        system: None,
        messages: vec![ResolvedInputMessage {
            role: Role::User,
            content: vec![ResolvedInputMessageContent::Text(Text {
                text: "Who wrote Lord of the Rings?".to_string(),
            })],
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

    // Wait for 1 second for ClickHouse to process
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Query about a similar topic (Harry Potter author, similar to Lord of the Rings question)
    // The max_distance=0.6 should keep relevant examples
    let params = ClientInferenceParams {
        function_name: Some(function_name.to_string()),
        variant_name: Some(variant_name.to_string()),
        episode_id: Some(episode_id),
        input: ClientInput {
            system: None,
            messages: vec![ClientInputMessage {
                role: Role::User,
                content: vec![ClientInputMessageContent::Text(TextKind::Text {
                    text: "Who was the author of the Harry Potter series?".to_string(),
                })],
            }],
        },
        ..Default::default()
    };

    let response = gateway.inference(params).await.unwrap();
    let InferenceOutput::NonStreaming(InferenceResponse::Chat(response)) = response else {
        panic!("Expected non-streaming chat response");
    };

    println!("API response: {response:#?}");

    let inference_id = response.inference_id;

    // Sleep to allow time for data to be inserted into ClickHouse
    sleep(Duration::from_secs(1)).await;

    // Check the ModelInference Table
    let clickhouse = get_clickhouse().await;
    let result = select_model_inferences_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {result:#?}");
    assert_eq!(result.len(), 2); // embedding + chat completion

    for model_inference in result {
        let model_name = model_inference.get("model_name").unwrap().as_str().unwrap();
        let input_messages = model_inference
            .get("input_messages")
            .unwrap()
            .as_str()
            .unwrap();
        let input_messages: Vec<StoredRequestMessage> =
            serde_json::from_str(input_messages).unwrap();

        match model_name {
            "openai::gpt-4o-mini-2024-07-18" => {
                // When relevant examples are kept, should have DICL behavior with examples
                // This means long input_messages (7 messages: 3 examples * 2 + 1 query)
                assert_eq!(
                    input_messages.len(),
                    7,
                    "Expected 7 input_messages with DICL examples, got {}",
                    input_messages.len()
                );

                // System should contain DICL instructions
                let system = model_inference.get("system").unwrap().as_str().unwrap();
                assert!(system.contains("learning by induction"));
            }
            "openai::text-embedding-3-small" => {
                // The embedding call should have 1 input message
                assert_eq!(input_messages.len(), 1);
            }
            _ => {
                panic!("Unexpected model: {model_name}");
            }
        }
    }
}
