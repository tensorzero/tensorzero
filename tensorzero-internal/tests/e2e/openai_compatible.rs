#![allow(clippy::print_stdout)]

use reqwest::{Client, StatusCode};
use serde_json::{json, Value};
use uuid::Uuid;

use crate::common::{
    get_clickhouse, get_gateway_endpoint, select_chat_inference_clickhouse,
    select_model_inference_clickhouse,
};

#[cfg(feature = "e2e_tests")]
#[tokio::test]
async fn test_openai_compatible_route() {
    // Test that both the old and new formats work.
    test_openai_compatible_route_with_function_name_asmodel(
        "tensorzero::function_name::basic_test_no_system_schema",
    )
    .await;
    test_openai_compatible_route_with_function_name_asmodel(
        "tensorzero::basic_test_no_system_schema",
    )
    .await;
}

async fn test_openai_compatible_route_with_function_name_asmodel(model: &str) {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
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
    });

    let response = client
        .post(get_gateway_endpoint("/openai/v1/chat/completions"))
        .header("episode_id", episode_id.to_string())
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
                "content": [{"type": "text", "value": "What is the capital of Japan?"}]
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
    assert!(output_tokens > 5);
    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);
    assert!(result.get("ttft_ms").unwrap().is_null());
    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    let _raw_response_json: Value = serde_json::from_str(raw_response).unwrap();
}

#[cfg(feature = "e2e_tests")]
#[tokio::test]
async fn test_openai_compatible_route_model_name_shorthand() {
    test_openai_compatible_route_with_default_function("tensorzero::model_name::dummy::good", "Megumin gleefully chanted her spell, unleashing a thunderous explosion that lit up the sky and left a massive crater in its wake.").await;
}

#[cfg(feature = "e2e_tests")]
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
        "stream": false,
    });

    let response = client
        .post(get_gateway_endpoint("/openai/v1/chat/completions"))
        .header("episode_id", episode_id.to_string())
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
    assert_eq!(content, expected_content);

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
                "content": [{"type": "text", "value": "What is the capital of Japan?"}]
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
    assert!(output_tokens > 5);
    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);
    assert!(result.get("ttft_ms").unwrap().is_null());
    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    let _raw_response_json: Value = serde_json::from_str(raw_response).unwrap();
}

#[cfg(feature = "e2e_tests")]
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
    });

    let response = client
        .post(get_gateway_endpoint("/openai/v1/chat/completions"))
        .header("episode_id", episode_id.to_string())
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let response_json = response.json::<Value>().await.unwrap();
    assert_eq!(
        response_json,
        json!({
            "error": "Invalid inference target: Invalid model name: Model name 'my_missing_model' not found in model table"
        })
    )
}

#[cfg(feature = "e2e_tests")]
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
        "response_format":{"type":"json_object"}
    });

    let response = client
        .post(get_gateway_endpoint("/openai/v1/chat/completions"))
        .header("episode_id", episode_id.to_string())
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
                "content": [{"type": "text", "value": "What is the capital of Japan?"}]
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
    assert!(output_tokens > 5);
    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);
    assert!(result.get("ttft_ms").unwrap().is_null());
    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    let _raw_response_json: Value = serde_json::from_str(raw_response).unwrap();
}

#[cfg(feature = "e2e_tests")]
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
        "response_format":{"type":"json_schema", "json_schema":{"name":"test", "strict":true, "schema":{}}}
    });

    let response = client
        .post(get_gateway_endpoint("/openai/v1/chat/completions"))
        .header("episode_id", episode_id.to_string())
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
                "content": [{"type": "text", "value": "What is the capital of Japan?"}]
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
    assert!(output_tokens > 5);
    let response_time_ms = result.get("response_time_ms").unwrap().as_u64().unwrap();
    assert!(response_time_ms > 0);
    assert!(result.get("ttft_ms").unwrap().is_null());
    let raw_response = result.get("raw_response").unwrap().as_str().unwrap();
    let _raw_response_json: Value = serde_json::from_str(raw_response).unwrap();
}
