#![expect(clippy::print_stdout)]

use std::collections::HashSet;
use tensorzero::ClientExt;

use axum::extract::State;
use http_body_util::BodyExt;
use reqwest::{Client, StatusCode};
use serde_json::{json, Value};
use uuid::Uuid;

use crate::common::get_gateway_endpoint;

use tensorzero_core::endpoints::openai_compatible::chat_completions::chat_completions_handler;
use tensorzero_core::{
    db::clickhouse::test_helpers::{
        get_clickhouse, select_chat_inference_clickhouse, select_json_inference_clickhouse,
        select_model_inference_clickhouse,
    },
    utils::gateway::StructuredJson,
};

#[tokio::test(flavor = "multi_thread")]
async fn test_openai_compatible_route_new_format() {
    test_openai_compatible_route_with_function_name_as_model(
        "tensorzero::function_name::basic_test_no_system_schema",
    )
    .await;
}

async fn test_openai_compatible_route_with_function_name_as_model(model: &str) {
    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let state = client.get_app_state_data().unwrap().clone();
    let episode_id = Uuid::now_v7();

    let response = chat_completions_handler(
        State(state),
        None,
        StructuredJson(
            serde_json::from_value(serde_json::json!({
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "TensorBot"
                    },
                    {
                        "role": "user",
                        "content": "What is the capital of Japan?"
                    }
                ],
                "stream": false,
                "tensorzero::tags": {
                    "foo": "bar"
                },
                "tensorzero::episode_id": episode_id.to_string(),
            }))
            .unwrap(),
        ),
    )
    .await
    .unwrap();

    // Check Response is OK, then fields in order
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.into_body().collect().await.unwrap().to_bytes();
    let response_json: Value = serde_json::from_slice(&response_json).unwrap();
    println!("response: {response_json:?}");
    let choices = response_json.get("choices").unwrap().as_array().unwrap();
    assert!(choices.len() == 1);
    let choice = choices.first().unwrap();
    assert_eq!(choice.get("index").unwrap().as_u64().unwrap(), 0);
    let message = choice.get("message").unwrap();
    assert_eq!(message.get("role").unwrap().as_str().unwrap(), "assistant");
    let content = message.get("content").unwrap().as_str().unwrap();
    assert_eq!(content, "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake.");
    let finish_reason = choice.get("finish_reason").unwrap().as_str().unwrap();
    assert_eq!(finish_reason, "stop");
    let response_model = response_json.get("model").unwrap().as_str().unwrap();
    assert_eq!(
        response_model,
        "tensorzero::function_name::basic_test_no_system_schema::variant_name::test"
    );

    let inference_id: Uuid = response_json
        .get("id")
        .unwrap()
        .as_str()
        .unwrap()
        .parse()
        .unwrap();

    // Sleep for 1 second to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ClickHouse
    let clickhouse = get_clickhouse().await;

    // First, check Inference table
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);
    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, "basic_test_no_system_schema");
    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input = json!({
        "system": "TensorBot",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is the capital of Japan?"}]
            }
        ]
    });
    assert_eq!(input, correct_input);
    let tags = result.get("tags").unwrap().as_object().unwrap();
    assert_eq!(tags.get("foo").unwrap().as_str().unwrap(), "bar");
    assert_eq!(tags.len(), 1);
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
    assert_eq!(variant_name, "test");
    // Check the processing time
    let _processing_time_ms = result.get("processing_time_ms").unwrap().as_u64().unwrap();

    // Check the ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    println!("ModelInference result: {result:?}");
    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);
    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, "test");
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, "good");
    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert_eq!(raw_request, "raw request");
    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 5);
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(output_tokens > 0);
    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);
    assert!(result.get("ttft_ms").unwrap().is_null());
    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    let _raw_response_json: Value = serde_json::from_str(raw_response).unwrap();
    let finish_reason = result.get("finish_reason").unwrap().as_str().unwrap();
    assert_eq!(finish_reason, "stop");
}

#[tokio::test]
async fn test_openai_compatible_matches_response_fields() {
    let client = Client::new();

    let tensorzero_payload = json!({
        "model": "tensorzero::model_name::openai::gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": "What is the capital of Japan?"
            }
        ],
    });

    let openai_payload = json!({
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": "What is the capital of Japan?"
            }
        ],
    });

    let tensorzero_response_fut = client
        .post(get_gateway_endpoint("/openai/v1/chat/completions"))
        .json(&tensorzero_payload)
        .send();

    let openai_response_fut = client
        .post("https://api.openai.com/v1/chat/completions")
        .bearer_auth(std::env::var("OPENAI_API_KEY").unwrap())
        .json(&openai_payload)
        .send();

    let (tensorzero_response, openai_response) =
        tokio::try_join!(tensorzero_response_fut, openai_response_fut).unwrap();

    assert_eq!(
        tensorzero_response.status(),
        StatusCode::OK,
        "TensorZero request failed"
    );
    assert_eq!(
        openai_response.status(),
        StatusCode::OK,
        "OpenAI request failed"
    );

    let openai_json: serde_json::Value = openai_response.json().await.unwrap();
    let tensorzero_json: serde_json::Value = tensorzero_response.json().await.unwrap();

    let openai_keys: HashSet<_> = openai_json.as_object().unwrap().keys().collect();
    let tensorzero_keys: HashSet<_> = tensorzero_json.as_object().unwrap().keys().collect();

    let missing_keys: Vec<_> = openai_keys.difference(&tensorzero_keys).collect();
    assert!(
        missing_keys.is_empty(),
        "Missing keys in TensorZero response: {missing_keys:?}"
    );
}

#[tokio::test]
async fn test_openai_compatible_dryrun() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "model": "tensorzero::model_name::json",
        "messages": [
            {
                "role": "system",
                "content": "TensorBot"
            },
            {
                "role": "user",
                "content": "What is the capital of Japan?"
            }
        ],
        "stream": false,
        "tensorzero::episode_id": episode_id.to_string(),
        "tensorzero::dryrun": true
    });

    let response = client
        .post(get_gateway_endpoint("/openai/v1/chat/completions"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    // Check Response is OK, then fields in order
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("response_json: {response_json:?}");
    let choices = response_json.get("choices").unwrap().as_array().unwrap();
    assert!(choices.len() == 1);
    let choice = choices.first().unwrap();
    assert_eq!(choice.get("index").unwrap().as_u64().unwrap(), 0);
    let message = choice.get("message").unwrap();
    assert_eq!(message.get("role").unwrap().as_str().unwrap(), "assistant");
    let content = message.get("content").unwrap().as_str().unwrap();
    assert_eq!(content, "{\"answer\":\"Hello\"}");
    let finish_reason = choice.get("finish_reason").unwrap().as_str().unwrap();
    assert_eq!(finish_reason, "stop");
    let response_model = response_json.get("model").unwrap().as_str().unwrap();
    assert_eq!(response_model, "tensorzero::model_name::json");

    let inference_id: Uuid = response_json
        .get("id")
        .unwrap()
        .as_str()
        .unwrap()
        .parse()
        .unwrap();

    // Sleep for 1 second to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ClickHouse
    let clickhouse = get_clickhouse().await;

    let chat_result = select_chat_inference_clickhouse(&clickhouse, inference_id).await;
    let json_result = select_json_inference_clickhouse(&clickhouse, inference_id).await;
    // No inference should be written to ClickHouse when dryrun is true
    assert!(chat_result.is_none());
    assert!(json_result.is_none());
}

#[tokio::test]
async fn test_openai_compatible_route_model_name_shorthand() {
    test_openai_compatible_route_with_default_function("tensorzero::model_name::dummy::good", "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake.").await;
}

#[tokio::test]
async fn test_openai_compatible_route_model_name_toml() {
    test_openai_compatible_route_with_default_function(
        "tensorzero::model_name::json",
        "{\"answer\":\"Hello\"}",
    )
    .await;
}

async fn test_openai_compatible_route_with_default_function(
    prefixed_model_name: &str,
    expected_content: &str,
) {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "model": prefixed_model_name,
        "messages": [
            {
                "role": "system",
                "content": "TensorBot"
            },
            {
                "role": "user",
                "content": "What is the capital of Japan?"
            }
        ],
        "tensorzero::episode_id": episode_id.to_string(),
        "stream": false,
    });

    let response = client
        .post(get_gateway_endpoint("/openai/v1/chat/completions"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    // Check Response is OK, then fields in order
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("response_json: {response_json:?}");
    let choices = response_json.get("choices").unwrap().as_array().unwrap();
    assert!(choices.len() == 1);
    let choice = choices.first().unwrap();
    assert_eq!(choice.get("index").unwrap().as_u64().unwrap(), 0);
    let message = choice.get("message").unwrap();
    assert_eq!(message.get("role").unwrap().as_str().unwrap(), "assistant");
    let content = message.get("content").unwrap().as_str().unwrap();
    assert_eq!(content, expected_content);
    let finish_reason = choice.get("finish_reason").unwrap().as_str().unwrap();
    assert_eq!(finish_reason, "stop");
    let response_model = response_json.get("model").unwrap().as_str().unwrap();
    assert_eq!(response_model, prefixed_model_name);

    let inference_id: Uuid = response_json
        .get("id")
        .unwrap()
        .as_str()
        .unwrap()
        .parse()
        .unwrap();

    // Sleep for 1 second to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ClickHouse
    let clickhouse = get_clickhouse().await;

    // First, check Inference table
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);
    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, "tensorzero::default");
    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input = json!({
        "system": "TensorBot",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is the capital of Japan?"}]
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
    // Check the processing time
    let _processing_time_ms = result.get("processing_time_ms").unwrap().as_u64().unwrap();

    // Check the ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);
    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(
        model_name,
        prefixed_model_name
            .strip_prefix("tensorzero::model_name::")
            .unwrap()
    );
    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert_eq!(raw_request, "raw request");
    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 5);
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(output_tokens > 0);
    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);
    assert!(result.get("ttft_ms").unwrap().is_null());
    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    let _raw_response_json: Value = serde_json::from_str(raw_response).unwrap();
    let finish_reason = result.get("finish_reason").unwrap().as_str().unwrap();
    assert_eq!(finish_reason, "stop");
}

#[tokio::test]
async fn test_openai_compatible_route_bad_model_name() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "model": "tensorzero::model_name::my_missing_model",
        "messages": [
            {
                "role": "system",
                "content": "TensorBot"
            },
            {
                "role": "user",
                "content": "What is the capital of Japan?"
            }
        ],
        "stream": false,
        "tensorzero::episode_id": episode_id.to_string(),
    });

    let response = client
        .post(get_gateway_endpoint("/openai/v1/chat/completions"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let response_json = response.json::<Value>().await.unwrap();
    assert_eq!(
        response_json,
        json!({
            "error": "Invalid inference target: Invalid model name: Model name 'my_missing_model' not found in model table",
            "error_json": {
                "InvalidInferenceTarget": {
                    "message": "Invalid model name: Model name 'my_missing_model' not found in model table"
                }
            }
        })
    );
}

#[tokio::test]
async fn test_openai_compatible_route_with_json_mode_on() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "model": "tensorzero::function_name::basic_test_no_system_schema",
        "messages": [
            {
                "role": "system",
                "content": "TensorBot"
            },
            {
                "role": "user",
                "content": "What is the capital of Japan?"
            }
        ],
        "stream": false,
        "response_format":{"type":"json_object"},
        "tensorzero::episode_id": episode_id.to_string(),
    });

    let response = client
        .post(get_gateway_endpoint("/openai/v1/chat/completions"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    // Check Response is OK, then fields in order
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    let choices = response_json.get("choices").unwrap().as_array().unwrap();
    assert!(choices.len() == 1);
    let choice = choices.first().unwrap();
    assert_eq!(choice.get("index").unwrap().as_u64().unwrap(), 0);
    let message = choice.get("message").unwrap();
    assert_eq!(message.get("role").unwrap().as_str().unwrap(), "assistant");
    let content = message.get("content").unwrap().as_str().unwrap();
    assert_eq!(content, "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake.");
    let response_model = response_json.get("model").unwrap().as_str().unwrap();
    assert_eq!(
        response_model,
        "tensorzero::function_name::basic_test_no_system_schema::variant_name::test"
    );

    let inference_id: Uuid = response_json
        .get("id")
        .unwrap()
        .as_str()
        .unwrap()
        .parse()
        .unwrap();

    // Sleep for 1 second to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ClickHouse
    let clickhouse = get_clickhouse().await;

    // First, check Inference table
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);
    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, "basic_test_no_system_schema");
    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input = json!({
        "system": "TensorBot",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is the capital of Japan?"}]
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
    assert_eq!(variant_name, "test");
    // Check the processing time
    let _processing_time_ms = result.get("processing_time_ms").unwrap().as_u64().unwrap();
    let inference_params = result.get("inference_params").unwrap().as_str().unwrap();
    let inference_params: Value = serde_json::from_str(inference_params).unwrap();
    let clickhouse_json_mode = inference_params
        .get("chat_completion")
        .unwrap()
        .get("json_mode")
        .unwrap()
        .as_str()
        .unwrap();
    assert_eq!("on", clickhouse_json_mode);

    // Check the ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);
    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, "test");
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, "good");
    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert_eq!(raw_request, "raw request");
    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 5);
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(output_tokens > 0);
    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);
    assert!(result.get("ttft_ms").unwrap().is_null());
    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    let _raw_response_json: Value = serde_json::from_str(raw_response).unwrap();
}

#[tokio::test]
async fn test_openai_compatible_route_with_json_schema() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "model": "tensorzero::function_name::basic_test_no_system_schema",
        "messages": [
            {
                "role": "system",
                "content": "TensorBot"
            },
            {
                "role": "user",
                "content": "What is the capital of Japan?"
            }
        ],
        "stream": false,
        "tensorzero::episode_id": episode_id.to_string(),
        "response_format":{"type":"json_schema", "json_schema":{"name":"test", "strict":true, "schema":{}}}
    });

    let response = client
        .post(get_gateway_endpoint("/openai/v1/chat/completions"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    // Check Response is OK, then fields in order
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("response_json: {response_json:?}");
    let choices = response_json.get("choices").unwrap().as_array().unwrap();
    assert!(choices.len() == 1);
    let choice = choices.first().unwrap();
    assert_eq!(choice.get("index").unwrap().as_u64().unwrap(), 0);
    let message = choice.get("message").unwrap();
    assert_eq!(message.get("role").unwrap().as_str().unwrap(), "assistant");
    let content = message.get("content").unwrap().as_str().unwrap();
    assert_eq!(content, "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake.");
    let finish_reason = choice.get("finish_reason").unwrap().as_str().unwrap();
    assert_eq!(finish_reason, "stop");
    let response_model = response_json.get("model").unwrap().as_str().unwrap();
    assert_eq!(
        response_model,
        "tensorzero::function_name::basic_test_no_system_schema::variant_name::test"
    );

    let inference_id: Uuid = response_json
        .get("id")
        .unwrap()
        .as_str()
        .unwrap()
        .parse()
        .unwrap();

    // Sleep for 1 second to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ClickHouse
    let clickhouse = get_clickhouse().await;

    // First, check Inference table
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    let id = result.get("id").unwrap().as_str().unwrap();
    let id_uuid = Uuid::parse_str(id).unwrap();
    assert_eq!(id_uuid, inference_id);
    let function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(function_name, "basic_test_no_system_schema");
    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input = json!({
        "system": "TensorBot",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is the capital of Japan?"}]
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
    assert_eq!(variant_name, "test");
    // Check the processing time
    let _processing_time_ms = result.get("processing_time_ms").unwrap().as_u64().unwrap();
    let inference_params = result.get("inference_params").unwrap().as_str().unwrap();
    let inference_params: Value = serde_json::from_str(inference_params).unwrap();
    let clickhouse_json_mode = inference_params
        .get("chat_completion")
        .unwrap()
        .get("json_mode")
        .unwrap()
        .as_str()
        .unwrap();
    assert_eq!("strict", clickhouse_json_mode);

    // Check the ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);
    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, "test");
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, "good");
    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert_eq!(raw_request, "raw request");
    let input_tokens = result.get("input_tokens").unwrap().as_u64().unwrap();
    assert!(input_tokens > 5);
    let output_tokens = result.get("output_tokens").unwrap().as_u64().unwrap();
    assert!(output_tokens > 0);
    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);
    assert!(result.get("ttft_ms").unwrap().is_null());
    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    let _raw_response_json: Value = serde_json::from_str(raw_response).unwrap();
}

#[tokio::test]
async fn test_openai_compatible_streaming_tool_call() {
    use futures::StreamExt;
    use reqwest_eventsource::{Event, RequestBuilderExt};

    let client = Client::new();
    let episode_id = Uuid::now_v7();
    let body = json!({
        "stream": true,
        "stream_options": {
            "include_usage": true
        },
        "model": "tensorzero::model_name::openai::gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": "What's the weather like in Boston today?"
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA"
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"]
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ],
        "tool_choice": "auto",
        "tensorzero::episode_id": episode_id.to_string(),
    });

    let mut response = client
        .post(get_gateway_endpoint("/openai/v1/chat/completions"))
        .header("Content-Type", "application/json")
        .json(&body)
        .eventsource()
        .unwrap();

    let mut chunks = vec![];
    let mut found_done_chunk = false;
    while let Some(event) = response.next().await {
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
    let first_chunk = chunks.first().unwrap();
    let parsed_chunk: Value = serde_json::from_str(first_chunk).unwrap();
    assert_eq!(parsed_chunk["choices"][0]["index"].as_i64().unwrap(), 0);
    assert_eq!(
        parsed_chunk["choices"][0]["delta"]["role"]
            .as_str()
            .unwrap(),
        "assistant"
    );
    assert!(parsed_chunk["choices"][0]["delta"].get("content").is_none());
    println!("parsed_chunk: {parsed_chunk:?}");
    let tool_calls = parsed_chunk["choices"][0]["delta"]["tool_calls"]
        .as_array()
        .unwrap();
    assert_eq!(tool_calls.len(), 1);
    let tool_call = tool_calls[0].as_object().unwrap();
    assert_eq!(tool_call["index"].as_i64().unwrap(), 0);
    assert_eq!(
        tool_call["function"]["name"].as_str().unwrap(),
        "get_current_weather"
    );
    assert_eq!(tool_call["function"]["arguments"].as_str().unwrap(), "");
    for (i, chunk) in chunks.iter().enumerate() {
        let parsed_chunk: Value = serde_json::from_str(chunk).unwrap();
        if let Some(tool_calls) = parsed_chunk["choices"][0]["delta"]["tool_calls"].as_array() {
            for tool_call in tool_calls {
                let index = tool_call["index"].as_i64().unwrap();
                assert_eq!(index, 0);
            }
        }
        if let Some(finish_reason) = parsed_chunk["choices"][0]["delta"]["finish_reason"].as_str() {
            assert_eq!(finish_reason, "tool_calls");
            assert_eq!(i, chunks.len() - 2);
        }
        if i == chunks.len() - 2 {
            assert!(parsed_chunk["choices"][0]["delta"].get("content").is_none());
            assert!(parsed_chunk["choices"][0]["delta"]
                .get("tool_calls")
                .is_none());
        }
        if i == chunks.len() - 1 {
            let usage = parsed_chunk["usage"].as_object().unwrap();
            assert!(usage["prompt_tokens"].as_i64().unwrap() > 0);
            assert!(usage["completion_tokens"].as_i64().unwrap() > 0);
        }
        let response_model = parsed_chunk.get("model").unwrap().as_str().unwrap();
        assert_eq!(response_model, "tensorzero::model_name::openai::gpt-4o");
    }
}

#[tokio::test]
async fn test_openai_compatible_warn_unknown_fields() {
    let logs_contain = tensorzero_core::utils::testing::capture_logs();
    let client = tensorzero::test_helpers::make_embedded_gateway_no_config().await;
    let state = client.get_app_state_data().unwrap().clone();
    chat_completions_handler(
        State(state),
        None,
        StructuredJson(
            serde_json::from_value(serde_json::json!({
                "messages": [],
                "model": "tensorzero::model_name::dummy::good",
                "my_fake_param": "fake_value"
            }))
            .unwrap(),
        ),
    )
    .await
    .unwrap();

    assert!(logs_contain(
        "Ignoring unknown fields in OpenAI-compatible request: [\"my_fake_param\"]"
    ));
}

#[tokio::test]
async fn test_openai_compatible_deny_unknown_fields() {
    let client = tensorzero::test_helpers::make_embedded_gateway_no_config().await;
    let state = client.get_app_state_data().unwrap().clone();
    let err = chat_completions_handler(
        State(state),
        None,
        StructuredJson(
            serde_json::from_value(serde_json::json!({
                "messages": [],
                "model": "tensorzero::model_name::dummy::good",
                "tensorzero::deny_unknown_fields": true,
                "my_fake_param": "fake_value",
                "my_other_fake_param": "fake_value_2"
            }))
            .unwrap(),
        ),
    )
    .await
    .unwrap_err();
    assert_eq!(
        err.to_string(),
        "Invalid request to OpenAI-compatible endpoint: `tensorzero::deny_unknown_fields` is set to true, but found unknown fields in the request: [my_fake_param, my_other_fake_param]"
    );
}

#[tokio::test]
async fn test_openai_compatible_streaming() {
    use futures::StreamExt;
    use reqwest_eventsource::{Event, RequestBuilderExt};

    let client = Client::new();
    let episode_id = Uuid::now_v7();
    let body = json!({
        "stream": true,
        "model": "tensorzero::model_name::openai::gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": "What's the reason for why we use AC not DC?"
            }
        ],
        "tensorzero::episode_id": episode_id.to_string(),
    });

    let mut response = client
        .post(get_gateway_endpoint("/openai/v1/chat/completions"))
        .header("Content-Type", "application/json")
        .json(&body)
        .eventsource()
        .unwrap();

    let mut chunks = vec![];
    let mut found_done_chunk = false;
    while let Some(event) = response.next().await {
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
    let first_chunk = chunks.first().unwrap();
    let parsed_chunk: Value = serde_json::from_str(first_chunk).unwrap();
    assert_eq!(parsed_chunk["choices"][0]["index"].as_i64().unwrap(), 0);
    assert_eq!(
        parsed_chunk["choices"][0]["delta"]["role"]
            .as_str()
            .unwrap(),
        "assistant"
    );
    let _content = parsed_chunk["choices"][0]["delta"]["content"]
        .as_str()
        .unwrap();
    assert!(parsed_chunk["choices"][0]["delta"]
        .get("tool_calls")
        .is_none());
    for (i, chunk) in chunks.iter().enumerate() {
        let parsed_chunk: Value = serde_json::from_str(chunk).unwrap();
        assert!(parsed_chunk["choices"][0]["delta"]
            .get("tool_calls")
            .is_none());
        if i < chunks.len() - 2 {
            let _content = parsed_chunk["choices"][0]["delta"]["content"]
                .as_str()
                .unwrap();
        }
        assert!(parsed_chunk["service_tier"].is_null());
        assert!(parsed_chunk["choices"][0]["logprobs"].is_null());
        if let Some(finish_reason) = parsed_chunk["choices"][0]["delta"]["finish_reason"].as_str() {
            assert_eq!(finish_reason, "stop");
            assert_eq!(i, chunks.len() - 2);
        }

        let response_model = parsed_chunk.get("model").unwrap().as_str().unwrap();
        assert_eq!(response_model, "tensorzero::model_name::openai::gpt-4o");
    }
}

// Test using 'stop' parameter in the openai-compatible endpoint
#[tokio::test]
async fn test_openai_compatible_stop_sequence() {
    let client = Client::new();

    let payload = json!({
        "model": "tensorzero::model_name::anthropic::claude-3-7-sonnet-20250219",
        "messages": [
            {
                "role": "user",
                "content": "Output 'Hello world' followed by either '0' or '1'. Do not output anything else"
            }
        ],
        "stop": ["0", "1"],
        "stream": false,
    });

    let response = client
        .post(get_gateway_endpoint("/openai/v1/chat/completions"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    let response_json = response.json::<Value>().await.unwrap();

    println!("API response: {response_json}");
    let finish_reason = response_json["choices"][0]["finish_reason"]
        .as_str()
        .unwrap();
    assert_eq!(finish_reason, "stop");
    let output = response_json["choices"][0]["message"]["content"]
        .as_str()
        .unwrap();
    assert!(output.contains("Hello"), "Unexpected output: {output}");
    assert!(
        !output.contains("zero") && !output.contains("one"),
        "Unexpected output: {output}"
    );

    // We don't bother checking ClickHouse, as we do that in lots of other tests
}

#[tokio::test]
async fn test_openai_compatible_file_with_custom_filename() {
    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    let state = client.get_app_state_data().unwrap().clone();
    let episode_id = Uuid::now_v7();

    let response = chat_completions_handler(
        State(state),
        None,
        StructuredJson(
            serde_json::from_value(serde_json::json!({
                "model": "tensorzero::function_name::basic_test_no_system_schema",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "What is in this file?"
                            },
                            {
                                "type": "file",
                                "file": {
                                    "file_data": "data:application/pdf;base64,JVBERi0xLjQK",
                                    "filename": "myfile.pdf"
                                }
                            }
                        ]
                    }
                ],
                "stream": false,
                "tensorzero::episode_id": episode_id.to_string(),
            }))
            .unwrap(),
        ),
    )
    .await
    .unwrap();

    // Check Response is OK
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.into_body().collect().await.unwrap().to_bytes();
    let response_json: Value = serde_json::from_slice(&response_json).unwrap();
    let inference_id: Uuid = response_json
        .get("id")
        .unwrap()
        .as_str()
        .unwrap()
        .parse()
        .unwrap();

    // Sleep for 1 second to allow time for data to be inserted into ClickHouse
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ClickHouse
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    // Verify the input was stored correctly with the custom filename
    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();

    // Check that the file content block has the custom filename
    let messages = input.get("messages").unwrap().as_array().unwrap();
    assert_eq!(messages.len(), 1);
    let content = messages[0].get("content").unwrap().as_array().unwrap();
    assert_eq!(content.len(), 2);

    // Second content block should be the file
    let file_block = &content[1];
    assert_eq!(file_block.get("type").unwrap().as_str().unwrap(), "file");

    // Verify filename is present in the stored file (fields are at top level, not nested)
    assert_eq!(
        file_block.get("filename").unwrap().as_str().unwrap(),
        "myfile.pdf"
    );
}

#[tokio::test]
async fn test_openai_compatible_parallel_tool_calls_multi_turn() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    // First request: Get parallel tool calls
    let body = json!({
        "stream": false,
        "model": "tensorzero::function_name::weather_helper_parallel",
        "messages": [
            { "role": "system", "content": [{"type": "tensorzero::template", "name": "system", "arguments": {"assistant_name": "Dr.Mehta"}}]},
            {
                "role": "user",
                "content": "What is the weather like in Tokyo (in Celsius)? Use both the provided `get_temperature` and `get_humidity` tools. Do not say anything else, just call the two functions."
            }
        ],
        "parallel_tool_calls": true,
        "tensorzero::episode_id": episode_id.to_string(),
        "tensorzero::variant_name": "gpt-5-mini",
    });

    let response = client
        .post(get_gateway_endpoint("/openai/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    println!("First request response: {response_json:#?}");
    // Sleep to allow ClickHouse writes
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    // Extract inference_id from response
    let inference_id: Uuid = response_json
        .get("id")
        .unwrap()
        .as_str()
        .unwrap()
        .parse()
        .unwrap();

    // ClickHouse validation
    let clickhouse = get_clickhouse().await;

    // Validate ChatInference table
    let chat_inference = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ChatInference: {chat_inference:#?}");

    // Validate ModelInference table
    let model_inference = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    println!("ClickHouse - ModelInference: {model_inference:#?}");

    // Extract tool calls from response
    let first_message = response_json["choices"][0]["message"].clone();
    let tool_calls = first_message["tool_calls"].as_array().unwrap();
    assert_eq!(tool_calls.len(), 2);

    // Build messages with tool results (one tool message per tool call)
    let mut messages = vec![
        json!({ "role": "system", "content": [{"type": "tensorzero::template", "name": "system", "arguments": {"assistant_name": "Dr.Mehta"}}]}),
        json!({
            "role": "user",
            "content": "What is the weather like in Tokyo (in Celsius)? Use both the provided `get_temperature` and `get_humidity` tools. Do not say anything else, just call the two functions."
        }),
        first_message.clone(),
    ];

    // Add one tool message for each tool call
    for tool_call in tool_calls {
        let tool_id = tool_call["id"].as_str().unwrap();
        let tool_name = tool_call["function"]["name"].as_str().unwrap();

        let result_content = match tool_name {
            "get_temperature" => "22°C",
            "get_humidity" => "65%",
            _ => panic!("Unexpected tool: {tool_name}"),
        };

        messages.push(json!({
            "role": "tool",
            "tool_call_id": tool_id,
            "content": result_content
        }));
    }

    // Second request: Submit tool results and get final response
    let second_body = json!({
        "stream": false,
        "model": "tensorzero::function_name::weather_helper_parallel",
        "messages": messages,
        "tensorzero::episode_id": episode_id.to_string(),
        "tensorzero::variant_name": "openai",
    });

    let response = client
        .post(get_gateway_endpoint("/openai/v1/chat/completions"))
        .json(&second_body)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let final_response_json = response.json::<Value>().await.unwrap();

    println!("Final response: {final_response_json:#?}");

    // Validate final response
    let final_choice = &final_response_json["choices"][0];
    let finish_reason = final_choice["finish_reason"].as_str().unwrap();
    // Should be "stop" (normal completion) not "tool_calls" since we provided results
    assert_eq!(finish_reason, "stop");

    // Should have text content in the response
    let content = final_choice["message"]["content"].as_str();
    assert!(content.is_some());
    assert!(!content.unwrap().is_empty());

    // Should not have tool_calls in final response
    let tool_calls = final_choice["message"]["tool_calls"].as_array().unwrap();
    assert!(tool_calls.is_empty());

    println!(
        "Multi-turn test passed! Got final response: {}",
        content.unwrap()
    );
}
