#![expect(clippy::print_stdout)]
use super::common::EmbeddingTestProvider;
use crate::common::get_gateway_endpoint;
use reqwest::{Client, StatusCode};
use serde_json::{json, Value};

pub async fn test_basic_embedding_with_provider(provider: EmbeddingTestProvider) {
    let payload = json!({
        "input": "Hello, world!",
        "model": provider.model_name,
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

pub async fn test_shorthand_embedding_with_provider(provider: EmbeddingTestProvider) {
    let shorthand_model = format!("openai::{}", provider.model_name);
    let payload = json!({
        "input": "Hello, world!",
        "model": shorthand_model,
    });
    let response = Client::new()
        .post(get_gateway_endpoint("/openai/v1/embeddings"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("Shorthand API response: {response_json:?}");
    assert_eq!(response_json["object"].as_str().unwrap(), "list");
    assert_eq!(response_json["model"].as_str().unwrap(), shorthand_model);
    assert_eq!(response_json["data"].as_array().unwrap().len(), 1);
    assert_eq!(response_json["data"][0]["index"].as_u64().unwrap(), 0);
    assert_eq!(
        response_json["data"][0]["object"].as_str().unwrap(),
        "embedding"
    );
    assert!(!response_json["data"][0]["embedding"]
        .as_array()
        .unwrap()
        .is_empty());
    assert!(response_json["usage"]["prompt_tokens"].as_u64().unwrap() > 0);
    assert!(response_json["usage"]["total_tokens"].as_u64().unwrap() > 0);
}

pub async fn test_batch_embedding_with_provider(provider: EmbeddingTestProvider) {
    let inputs = vec![
        "Hello, world!",
        "How are you today?",
        "This is a test of batch embeddings.",
    ];
    let payload = json!({
        "input": inputs,
        "model": provider.model_name,
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
        provider.model_name
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
        "model": provider.model_name,
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
        provider.model_name
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
        "model": provider.model_name,
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
        provider.model_name
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
        "model": provider.model_name,
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
        provider.model_name
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
        "model": "tensorzero::model_name::nonexistent_model",
    });
    let response = Client::new()
        .post(get_gateway_endpoint("/openai/v1/embeddings"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

pub async fn test_embedding_large_batch_with_provider(provider: EmbeddingTestProvider) {
    let inputs: Vec<String> = (1..=10)
        .map(|i| format!("This is test input number {i}"))
        .collect();
    let payload = json!({
        "input": inputs,
        "model": provider.model_name,
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
        provider.model_name
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
        "model": provider.model_name,
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
        "model": "fallback",
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
        "model": "timeout",
    });
    let response = Client::new()
        .post(get_gateway_endpoint("/openai/v1/embeddings"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::REQUEST_TIMEOUT);
}
