#![expect(clippy::print_stdout)]
use super::common::EmbeddingTestProvider;
use crate::common::get_gateway_endpoint;
use reqwest::{Client, StatusCode};
use serde_json::{json, Value};

pub async fn test_basic_embedding_with_provider(provider: EmbeddingTestProvider) {
    let payload = json!({
        "input": "Hello, world!",
        "model": format!("tensorzero::embedding_model_name::{}", provider.model_name),
    });
    let response = Client::new()
        .post(get_gateway_endpoint("/openai/v1/embeddings"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("API response: {response_json:?}");
    assert_eq!(response_json["object"].as_str().unwrap(), "list");
    assert_eq!(
        response_json["data"][0]["embedding"]
            .as_array()
            .unwrap()
            .len(),
        1536
    );
    assert_eq!(response_json["data"].as_array().unwrap().len(), 1);
    assert_eq!(response_json["data"][0]["index"].as_u64().unwrap(), 0);
    assert_eq!(
        response_json["data"][0]["object"].as_str().unwrap(),
        "embedding"
    );
    assert!(response_json["usage"]["prompt_tokens"].as_u64().unwrap() > 0);
    assert!(response_json["usage"]["total_tokens"].as_u64().unwrap() > 0);
}

pub async fn test_bulk_embedding_with_provider(provider: EmbeddingTestProvider) {
    let inputs = vec![
        "Hello, world!",
        "How are you today?",
        "This is a test of batch embeddings.",
    ];
    let payload = json!({
        "input": inputs,
        "model": format!("tensorzero::embedding_model_name::{}", provider.model_name),
    });
    let response = Client::new()
        .post(get_gateway_endpoint("/openai/v1/embeddings"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("Batch API response: {response_json:?}");
    assert_eq!(response_json["object"].as_str().unwrap(), "list");
    assert_eq!(
        response_json["model"].as_str().unwrap(),
        format!("tensorzero::embedding_model_name::{}", provider.model_name)
    );
    assert_eq!(
        response_json["data"].as_array().unwrap().len(),
        inputs.len()
    );

    for (i, embedding_data) in response_json["data"].as_array().unwrap().iter().enumerate() {
        assert_eq!(embedding_data["index"].as_u64().unwrap(), i as u64);
        assert_eq!(embedding_data["object"].as_str().unwrap(), "embedding");
        assert!(!embedding_data["embedding"].as_array().unwrap().is_empty());
    }

    assert!(response_json["usage"]["prompt_tokens"].as_u64().unwrap() > 0);
    assert!(response_json["usage"]["total_tokens"].as_u64().unwrap() > 0);
}

pub async fn test_embedding_with_dimensions_with_provider(provider: EmbeddingTestProvider) {
    let payload = json!({
        "input": "Test with specific dimensions",
        "model": format!("tensorzero::embedding_model_name::{}", provider.model_name),
        "dimensions": 512,
    });
    let response = Client::new()
        .post(get_gateway_endpoint("/openai/v1/embeddings"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("Dimensions API response: {response_json:?}");
    assert_eq!(response_json["object"].as_str().unwrap(), "list");
    assert_eq!(
        response_json["model"].as_str().unwrap(),
        format!("tensorzero::embedding_model_name::{}", provider.model_name)
    );
    assert_eq!(response_json["data"].as_array().unwrap().len(), 1);
    assert_eq!(
        response_json["data"][0]["embedding"]
            .as_array()
            .unwrap()
            .len(),
        512
    );
}

pub async fn test_embedding_with_encoding_format_with_provider(provider: EmbeddingTestProvider) {
    let payload = json!({
        "input": "Test encoding format",
        "model": format!("tensorzero::embedding_model_name::{}", provider.model_name),
        "encoding_format": "float",
    });
    let response = Client::new()
        .post(get_gateway_endpoint("/openai/v1/embeddings"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("Encoding format API response: {response_json:?}");
    assert_eq!(response_json["object"].as_str().unwrap(), "list");
    assert_eq!(
        response_json["model"].as_str().unwrap(),
        format!("tensorzero::embedding_model_name::{}", provider.model_name)
    );
    assert_eq!(response_json["data"].as_array().unwrap().len(), 1);
    let embedding = &response_json["data"][0]["embedding"];
    assert!(!embedding.as_array().unwrap().is_empty());
    // Verify that the first element is a float
    assert!(embedding[0].as_f64().is_some());
}

pub async fn test_embedding_with_user_parameter_with_provider(provider: EmbeddingTestProvider) {
    let user_id = "test_user_123";
    let payload = json!({
        "input": "Test with user parameter",
        "model": format!("tensorzero::embedding_model_name::{}", provider.model_name),
        "user": user_id,
    });
    let response = Client::new()
        .post(get_gateway_endpoint("/openai/v1/embeddings"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("User parameter API response: {response_json:?}");
    assert_eq!(response_json["object"].as_str().unwrap(), "list");
    assert_eq!(
        response_json["model"].as_str().unwrap(),
        format!("tensorzero::embedding_model_name::{}", provider.model_name)
    );
    assert_eq!(response_json["data"].as_array().unwrap().len(), 1);
    assert!(!response_json["data"][0]["embedding"]
        .as_array()
        .unwrap()
        .is_empty());
}

pub async fn test_embedding_invalid_model_error_with_provider(_provider: EmbeddingTestProvider) {
    let payload = json!({
        "input": "Test invalid model",
        "model": "tensorzero::embedding_model_name::nonexistent_model",
    });
    let response = Client::new()
        .post(get_gateway_endpoint("/openai/v1/embeddings"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

pub async fn test_embedding_large_bulk_with_provider(provider: EmbeddingTestProvider) {
    let inputs: Vec<String> = (1..=10)
        .map(|i| format!("This is test input number {i}"))
        .collect();
    let payload = json!({
        "input": inputs,
        "model": format!("tensorzero::embedding_model_name::{}", provider.model_name),
    });
    let response = Client::new()
        .post(get_gateway_endpoint("/openai/v1/embeddings"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("Large batch API response: {response_json:?}");
    assert_eq!(response_json["object"].as_str().unwrap(), "list");
    assert_eq!(
        response_json["model"].as_str().unwrap(),
        format!("tensorzero::embedding_model_name::{}", provider.model_name)
    );
    assert_eq!(response_json["data"].as_array().unwrap().len(), 10);

    for (i, embedding_data) in response_json["data"].as_array().unwrap().iter().enumerate() {
        assert_eq!(embedding_data["index"].as_u64().unwrap(), i as u64);
        assert_eq!(embedding_data["object"].as_str().unwrap(), "embedding");
        assert!(!embedding_data["embedding"].as_array().unwrap().is_empty());
    }

    assert!(response_json["usage"]["prompt_tokens"].as_u64().unwrap() > 0);
    assert!(response_json["usage"]["total_tokens"].as_u64().unwrap() > 0);
}

pub async fn test_embedding_consistency_with_provider(provider: EmbeddingTestProvider) {
    let input_text = "This is a consistency test";

    // Generate embeddings twice with the same input
    let payload = json!({
        "input": input_text,
        "model": format!("tensorzero::embedding_model_name::{}", provider.model_name),
    });

    let response1 = Client::new()
        .post(get_gateway_endpoint("/openai/v1/embeddings"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response1.status(), StatusCode::OK);
    let response1_json = response1.json::<Value>().await.unwrap();

    let response2 = Client::new()
        .post(get_gateway_endpoint("/openai/v1/embeddings"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response2.status(), StatusCode::OK);
    let response2_json = response2.json::<Value>().await.unwrap();

    println!("Consistency test responses: {response1_json:?} vs {response2_json:?}");

    // Both should have the same model and structure
    assert_eq!(response1_json["model"], response2_json["model"]);
    assert_eq!(response1_json["data"].as_array().unwrap().len(), 1);
    assert_eq!(response2_json["data"].as_array().unwrap().len(), 1);

    let embedding1 = response1_json["data"][0]["embedding"].as_array().unwrap();
    let embedding2 = response2_json["data"][0]["embedding"].as_array().unwrap();
    assert_eq!(embedding1.len(), embedding2.len());

    // Check that embeddings are similar (allowing for small numerical differences)
    for i in 0..std::cmp::min(10, embedding1.len()) {
        let val1 = embedding1[i].as_f64().unwrap();
        let val2 = embedding2[i].as_f64().unwrap();
        assert!(
            (val1 - val2).abs() < 0.01,
            "Embeddings differ significantly at index {i}: {val1} vs {val2}"
        );
    }
}

/// Test basic embedding with fallback model
/// That model should have a slow model run first, time out, and then succeed with the OpenAI model
/// For now this test is underspecified since we can't run the embeddings through the embedded client and check logs
/// or check ClickHouse
#[tokio::test]
pub async fn test_basic_embedding_fallback() {
    let payload = json!({
        "input": "Hello, world!",
        "model": "tensorzero::embedding_model_name::fallback",
    });
    let response = Client::new()
        .post(get_gateway_endpoint("/openai/v1/embeddings"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("API response: {response_json:?}");
    assert_eq!(response_json["object"].as_str().unwrap(), "list");
    assert_eq!(
        response_json["data"][0]["embedding"]
            .as_array()
            .unwrap()
            .len(),
        1536
    );
    assert_eq!(response_json["data"].as_array().unwrap().len(), 1);
    assert_eq!(response_json["data"][0]["index"].as_u64().unwrap(), 0);
    assert_eq!(
        response_json["data"][0]["object"].as_str().unwrap(),
        "embedding"
    );
    assert!(response_json["usage"]["prompt_tokens"].as_u64().unwrap() > 0);
    assert!(response_json["usage"]["total_tokens"].as_u64().unwrap() > 0);
}

#[tokio::test]
pub async fn test_basic_embedding_timeout() {
    let payload = json!({
        "input": "Hello, world!",
        "model": "tensorzero::embedding_model_name::timeout",
    });
    let response = Client::new()
        .post(get_gateway_endpoint("/openai/v1/embeddings"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::REQUEST_TIMEOUT);
}

pub async fn test_embedding_cache_with_provider(provider: EmbeddingTestProvider) {
    let input_text = format!(
        "This is a cache test for embeddings (test_embedding_cache_with_provider) - {}",
        rand::random::<u32>()
    );

    // First request with cache enabled to populate cache
    let payload = json!({
        "input": input_text,
        "model": format!("tensorzero::embedding_model_name::{}", provider.model_name),
        "tensorzero::cache_options": {
            "enabled": "on",
            "max_age_s": 60
        }
    });
    let response = Client::new()
        .post(get_gateway_endpoint("/openai/v1/embeddings"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    // Store original response for comparison
    let original_usage = response_json["usage"].clone();
    assert!(original_usage["prompt_tokens"].as_u64().unwrap() > 0);
    assert!(original_usage["total_tokens"].as_u64().unwrap() > 0);
    let original_embedding = response_json["data"][0]["embedding"].clone();

    // Wait a moment for cache write to complete
    tokio::time::sleep(std::time::Duration::from_secs(2)).await;

    // Second request with same cache options - should hit cache
    let payload_cached = json!({
        "input": input_text,
        "model": format!("tensorzero::embedding_model_name::{}", provider.model_name),
        "tensorzero::cache_options": {
            "enabled": "on",
            "max_age_s": 60
        }
    });
    let response_cached = Client::new()
        .post(get_gateway_endpoint("/openai/v1/embeddings"))
        .json(&payload_cached)
        .send()
        .await
        .unwrap();
    assert_eq!(response_cached.status(), StatusCode::OK);
    let response_cached_json = response_cached.json::<Value>().await.unwrap();

    // Check that cached response has zero tokens (indicating cache hit)
    let cached_usage = response_cached_json["usage"].clone();
    assert_eq!(cached_usage["prompt_tokens"].as_u64().unwrap(), 0);
    assert_eq!(cached_usage["total_tokens"].as_u64().unwrap(), 0);

    // Check that embeddings are identical
    let cached_embedding = response_cached_json["data"][0]["embedding"].clone();
    assert_eq!(original_embedding, cached_embedding);

    // Check other response fields are consistent
    assert_eq!(response_json["model"], response_cached_json["model"]);
    assert_eq!(response_json["object"], response_cached_json["object"]);
    assert_eq!(
        response_json["data"][0]["index"],
        response_cached_json["data"][0]["index"]
    );
    assert_eq!(
        response_json["data"][0]["object"],
        response_cached_json["data"][0]["object"]
    );
}

pub async fn test_embedding_cache_options_with_provider(provider: EmbeddingTestProvider) {
    let input_text = format!(
        "This is a cache options test for embeddings (test_embedding_cache_options_with_provider) - {}",
        rand::random::<u32>()
    );

    // First, make a request that will be cached
    let payload_initial = json!({
        "input": input_text,
        "model": format!("tensorzero::embedding_model_name::{}", provider.model_name),
        "tensorzero::cache_options": {
            "enabled": "on",
        }
    });
    let response_initial = Client::new()
        .post(get_gateway_endpoint("/openai/v1/embeddings"))
        .json(&payload_initial)
        .send()
        .await
        .unwrap();
    assert_eq!(response_initial.status(), StatusCode::OK);
    let response_initial_json = response_initial.json::<Value>().await.unwrap();
    assert!(
        response_initial_json["usage"]["prompt_tokens"]
            .as_u64()
            .unwrap()
            > 0
    );

    // Wait for cache write to complete
    tokio::time::sleep(std::time::Duration::from_secs(2)).await;

    // Test with cache disabled - should not use cache
    let payload_disabled = json!({
        "input": input_text,
        "model": format!("tensorzero::embedding_model_name::{}", provider.model_name),
        "tensorzero::cache_options": {
            "enabled": "off"
        }
    });
    let response_disabled = Client::new()
        .post(get_gateway_endpoint("/openai/v1/embeddings"))
        .json(&payload_disabled)
        .send()
        .await
        .unwrap();
    assert_eq!(response_disabled.status(), StatusCode::OK);
    let response_disabled_json = response_disabled.json::<Value>().await.unwrap();
    assert!(
        response_disabled_json["usage"]["prompt_tokens"]
            .as_u64()
            .unwrap()
            > 0
    );

    // Test with cache enabled and valid max_age - should use cache
    let payload_enabled = json!({
        "input": input_text,
        "model": format!("tensorzero::embedding_model_name::{}", provider.model_name),
        "tensorzero::cache_options": {
            "enabled": "on",
            "max_age_s": 60
        }
    });
    let response_enabled = Client::new()
        .post(get_gateway_endpoint("/openai/v1/embeddings"))
        .json(&payload_enabled)
        .send()
        .await
        .unwrap();
    assert_eq!(response_enabled.status(), StatusCode::OK);
    let response_enabled_json = response_enabled.json::<Value>().await.unwrap();
    assert_eq!(
        response_enabled_json["usage"]["prompt_tokens"]
            .as_u64()
            .unwrap(),
        0
    );
    assert_eq!(
        response_enabled_json["usage"]["total_tokens"]
            .as_u64()
            .unwrap(),
        0
    );

    // Test with cache enabled but expired max_age - should miss cache
    let payload_expired = json!({
        "input": input_text,
        "model": format!("tensorzero::embedding_model_name::{}", provider.model_name),
        "tensorzero::cache_options": {
            "enabled": "on",
            "max_age_s": 1
        }
    });
    let response_expired = Client::new()
        .post(get_gateway_endpoint("/openai/v1/embeddings"))
        .json(&payload_expired)
        .send()
        .await
        .unwrap();
    assert_eq!(response_expired.status(), StatusCode::OK);
    let response_expired_json = response_expired.json::<Value>().await.unwrap();
    assert!(
        response_expired_json["usage"]["prompt_tokens"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert!(
        response_expired_json["usage"]["total_tokens"]
            .as_u64()
            .unwrap()
            > 0
    );
}

pub async fn test_embedding_dryrun_with_provider(provider: EmbeddingTestProvider) {
    let input_text = "This is a dryrun test for embeddings (test_embedding_dryrun_with_provider).";

    // Test with dryrun enabled (should not store to database)
    let payload = json!({
        "input": input_text,
        "model": format!("tensorzero::embedding_model_name::{}", provider.model_name),
        "tensorzero::dryrun": true
    });
    let response = Client::new()
        .post(get_gateway_endpoint("/openai/v1/embeddings"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();

    // Should still return valid embedding data
    assert_eq!(response_json["object"].as_str().unwrap(), "list");
    assert_eq!(response_json["data"].as_array().unwrap().len(), 1);
    assert!(!response_json["data"][0]["embedding"]
        .as_array()
        .unwrap()
        .is_empty());
    assert!(response_json["usage"]["prompt_tokens"].as_u64().unwrap() > 0);
    assert!(response_json["usage"]["total_tokens"].as_u64().unwrap() > 0);

    // Test dryrun combined with cache options
    let payload_dryrun_cache = json!({
        "input": input_text,
        "model": format!("tensorzero::embedding_model_name::{}",  provider.model_name),
        "tensorzero::dryrun": true,
        "tensorzero::cache_options": {
            "enabled": "on",
            "max_age_s": 60
        }
    });
    let response_dryrun_cache = Client::new()
        .post(get_gateway_endpoint("/openai/v1/embeddings"))
        .json(&payload_dryrun_cache)
        .send()
        .await
        .unwrap();
    assert_eq!(response_dryrun_cache.status(), StatusCode::OK);
    let response_dryrun_cache_json = response_dryrun_cache.json::<Value>().await.unwrap();

    // Should still return valid embedding data even with dryrun + cache
    assert_eq!(
        response_dryrun_cache_json["object"].as_str().unwrap(),
        "list"
    );
    assert_eq!(
        response_dryrun_cache_json["data"].as_array().unwrap().len(),
        1
    );
    assert!(!response_dryrun_cache_json["data"][0]["embedding"]
        .as_array()
        .unwrap()
        .is_empty());
}
