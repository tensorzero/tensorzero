#![expect(clippy::print_stdout)]
use super::common::EmbeddingTestProvider;
use crate::common::get_gateway_endpoint;
use reqwest::{Client, StatusCode};
use serde_json::{json, Value};

/*
*
* #[derive(Debug, Deserialize)]
pub struct OpenAICompatibleEmbeddingParams {
    input: EmbeddingInput,
    model: String,
    dimensions: Option<u32>,
    // Since we only support one format, this field is not used.
    #[expect(dead_code)]
    encoding_format: EmbeddingEncodingFormat,
    #[serde(default, rename = "tensorzero::credentials")]
    tensorzero_credentials: InferenceCredentials,
}
*
*/

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
}
