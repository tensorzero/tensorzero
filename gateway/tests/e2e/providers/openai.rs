use gateway::{
    embeddings::{EmbeddingProvider, EmbeddingProviderConfig, EmbeddingRequest},
    endpoints::inference::InferenceCredentials,
    inference::types::Latency,
};
use reqwest::{Client, StatusCode};
use serde_json::{json, Value};
use uuid::Uuid;

use crate::{
    common::{
        get_clickhouse, get_gateway_endpoint, select_chat_inference_clickhouse,
        select_model_inference_clickhouse,
    },
    providers::common::{E2ETestProvider, E2ETestProviders},
};

crate::generate_provider_tests!(get_providers);
crate::generate_batch_inference_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let standard_providers = vec![E2ETestProvider {
        variant_name: "openai".to_string(),
        model_name: "gpt-4o-mini-2024-07-18".to_string(),
        model_provider_name: "openai".to_string(),
    }];

    let json_providers = vec![
        E2ETestProvider {
            variant_name: "openai".to_string(),
            model_name: "gpt-4o-mini-2024-07-18".to_string(),
            model_provider_name: "openai".to_string(),
        },
        E2ETestProvider {
            variant_name: "openai-implicit".to_string(),
            model_name: "gpt-4o-mini-2024-07-18".to_string(),
            model_provider_name: "openai".to_string(),
        },
        E2ETestProvider {
            variant_name: "openai-strict".to_string(),
            model_name: "gpt-4o-mini-2024-07-18".to_string(),
            model_provider_name: "openai".to_string(),
        },
    ];

    let shorthand_providers = vec![E2ETestProvider {
        variant_name: "openai-shorthand".to_string(),
        model_name: "openai::gpt-4o-mini-2024-07-18".to_string(),
        model_provider_name: "openai".to_string(),
    }];

    E2ETestProviders {
        simple_inference: standard_providers.clone(),
        inference_params_inference: standard_providers.clone(),
        tool_use_inference: standard_providers.clone(),
        tool_multi_turn_inference: standard_providers.clone(),
        dynamic_tool_use_inference: standard_providers.clone(),
        parallel_tool_use_inference: standard_providers.clone(),
        json_mode_inference: json_providers.clone(),
        shorthand_inference: shorthand_providers.clone(),
        supports_batch_inference: true,
    }
}

#[tokio::test]
async fn test_o1_mini_inference() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "variant_name": "o1-mini",
        "episode_id": episode_id,
        "input":
            {"system": {"assistant_name": "AskJeeves"},
            "messages": [
                {
                    "role": "user",
                    "content": "What is the capital of Japan?"
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
    // Check Response is OK, then fields in order
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    let content_blocks = response_json.get("content").unwrap().as_array().unwrap();
    assert!(content_blocks.len() == 1);
    let content_block = content_blocks.first().unwrap();
    let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
    assert_eq!(content_block_type, "text");
    let content = content_block.get("text").unwrap().as_str().unwrap();
    // Assert that Tokyo is in the content
    assert!(content.contains("Tokyo"), "Content should mention Tokyo");
    // Check that inference_id is here
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

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
    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input = json!({
        "system": {"assistant_name": "AskJeeves"},
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
    assert_eq!(variant_name, "o1-mini");
    // Check the processing time
    let processing_time_ms = result.get("processing_time_ms").unwrap().as_u64().unwrap();
    assert!(processing_time_ms > 0);

    // Check the ModelInference Table
    let result = select_model_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    let inference_id_result = result.get("inference_id").unwrap().as_str().unwrap();
    let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
    assert_eq!(inference_id_result, inference_id);
    let model_name = result.get("model_name").unwrap().as_str().unwrap();
    assert_eq!(model_name, "o1-mini");
    let model_provider_name = result.get("model_provider_name").unwrap().as_str().unwrap();
    assert_eq!(model_provider_name, "openai");
    let raw_request = result.get("raw_request").unwrap().as_str().unwrap();
    assert!(raw_request.to_lowercase().contains("japan"));
    // Check that raw_request is valid JSON
    let _: Value = serde_json::from_str(raw_request).expect("raw_request should be valid JSON");
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

#[tokio::test]
async fn test_embedding_request() {
    let provider_config_serialized = r#"
    type = "openai"
    model_name = "text-embedding-3-small"
    "#;
    let provider_config: EmbeddingProviderConfig = toml::from_str(provider_config_serialized)
        .expect("Failed to deserialize EmbeddingProviderConfig");
    assert!(matches!(
        provider_config,
        EmbeddingProviderConfig::OpenAI(_)
    ));

    let client = Client::new();
    let request = EmbeddingRequest {
        input: "This is a test input".to_string(),
    };
    let api_keys = InferenceCredentials::default();
    let response = provider_config
        .embed(&request, &client, &api_keys)
        .await
        .unwrap();
    assert_eq!(response.embedding.len(), 1536);
    // Calculate the L2 norm of the embedding
    let norm: f32 = response
        .embedding
        .iter()
        .map(|&x| x.powi(2))
        .sum::<f32>()
        .sqrt();

    // Assert that the norm is approximately 1 (allowing for small floating-point errors)
    assert!(
        (norm - 1.0).abs() < 1e-6,
        "The L2 norm of the embedding should be 1, but it is {}",
        norm
    );
    // Check that the timestamp in created is within 1 second of the current time
    let created = response.created;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("Time went backwards")
        .as_secs() as i64;
    assert!(
        (created as i64 - now).abs() <= 1,
        "The created timestamp should be within 1 second of the current time, but it is {}",
        created
    );
    let parsed_raw_response: Value = serde_json::from_str(&response.raw_response).unwrap();
    assert!(
        !parsed_raw_response.is_null(),
        "Parsed raw response should not be null"
    );
    let parsed_raw_request: Value = serde_json::from_str(&response.raw_request).unwrap();
    assert!(
        !parsed_raw_request.is_null(),
        "Parsed raw request should not be null"
    );
    // Hardcoded since the input is 5 tokens
    assert_eq!(response.usage.input_tokens, 5);
    assert_eq!(response.usage.output_tokens, 0);
    match response.latency {
        Latency::NonStreaming { response_time } => {
            assert!(response_time.as_millis() > 100);
        }
        _ => panic!("Latency should be non-streaming"),
    }
}

#[tokio::test]
async fn test_embedding_sanity_check() {
    let provider_config_serialized = r#"
    type = "openai"
    model_name = "text-embedding-3-small"
    "#;
    let provider_config: EmbeddingProviderConfig = toml::from_str(provider_config_serialized)
        .expect("Failed to deserialize EmbeddingProviderConfig");
    let client = Client::new();
    let embedding_request_a = EmbeddingRequest {
        input: "Joe Biden is the president of the United States".to_string(),
    };

    let embedding_request_b = EmbeddingRequest {
        input: "Kamala Harris is the vice president of the United States".to_string(),
    };

    let embedding_request_c = EmbeddingRequest {
        input: "My favorite systems programming language is Rust".to_string(),
    };
    let api_keys = InferenceCredentials::default();

    // Compute all 3 embeddings concurrently
    let (response_a, response_b, response_c) = tokio::join!(
        provider_config.embed(&embedding_request_a, &client, &api_keys),
        provider_config.embed(&embedding_request_b, &client, &api_keys),
        provider_config.embed(&embedding_request_c, &client, &api_keys)
    );

    // Unwrap the results
    let response_a = response_a.expect("Failed to get embedding for request A");
    let response_b = response_b.expect("Failed to get embedding for request B");
    let response_c = response_c.expect("Failed to get embedding for request C");

    // Calculate cosine similarities
    let similarity_ab = cosine_similarity(&response_a.embedding, &response_b.embedding);
    let similarity_ac = cosine_similarity(&response_a.embedding, &response_c.embedding);
    let similarity_bc = cosine_similarity(&response_b.embedding, &response_c.embedding);

    // Assert that semantically similar sentences have higher similarity (with a margin of 0.3)
    // We empirically determined this by staring at it (no science to it)
    assert!(
        similarity_ab - similarity_ac > 0.3,
        "Similarity between A and B should be higher than between A and C"
    );
    assert!(
        similarity_ab - similarity_bc > 0.3,
        "Similarity between A and B should be higher than between B and C"
    );
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let magnitude_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let magnitude_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot_product / (magnitude_a * magnitude_b)
}
