use serde_json::Value;

use crate::common::start_gateway_on_random_port;

mod common;

#[tokio::test]
async fn test_no_error_json() {
    let child_data = start_gateway_on_random_port("", None).await;
    let inference_response = reqwest::Client::new()
        .post(format!("http://{}/inference", child_data.addr))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert_eq!(
        inference_response,
        r#"{"error":"Failed to parse the request body as JSON: EOF while parsing a value at line 1 column 0 (400 Bad Request)"}"#
    );
}

#[tokio::test]
async fn test_error_json() {
    let child_data = start_gateway_on_random_port("unstable_error_json = true", None).await;
    let inference_response = reqwest::Client::new()
        .post(format!("http://{}/inference", child_data.addr))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert_eq!(
        inference_response,
        r#"{"error":"Failed to parse the request body as JSON: EOF while parsing a value at line 1 column 0 (400 Bad Request)","error_json":{"JsonRequest":{"message":"Failed to parse the request body as JSON: EOF while parsing a value at line 1 column 0 (400 Bad Request)"}}}"#
    );
}

/// Test that OpenAI-compatible chat completions endpoint returns errors in OpenAI format
/// OpenAI format: {"error": {"message": "..."}}
#[tokio::test]
async fn test_openai_compatible_error_format() {
    let child_data = start_gateway_on_random_port("", None).await;
    let response = reqwest::Client::new()
        .post(format!(
            "http://{}/openai/v1/chat/completions",
            child_data.addr
        ))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();

    let json: Value = serde_json::from_str(&response).expect("Response should be valid JSON");

    // Verify OpenAI error format: {"error": {"message": "..."}}
    assert!(
        json.get("error").is_some(),
        "Response should have an 'error' field"
    );
    let error_obj = json.get("error").unwrap();
    assert!(
        error_obj.is_object(),
        "The 'error' field should be an object, not a string"
    );
    assert!(
        error_obj.get("message").is_some(),
        "The 'error' object should have a 'message' field"
    );
    assert!(
        error_obj.get("message").unwrap().is_string(),
        "The 'message' field should be a string"
    );
}

/// Test that OpenAI-compatible embeddings endpoint returns errors in OpenAI format
#[tokio::test]
async fn test_openai_compatible_embeddings_error_format() {
    let child_data = start_gateway_on_random_port("", None).await;
    let response = reqwest::Client::new()
        .post(format!("http://{}/openai/v1/embeddings", child_data.addr))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();

    let json: Value = serde_json::from_str(&response).expect("Response should be valid JSON");

    // Verify OpenAI error format: {"error": {"message": "..."}}
    assert!(
        json.get("error").is_some(),
        "Response should have an 'error' field"
    );
    let error_obj = json.get("error").unwrap();
    assert!(
        error_obj.is_object(),
        "The 'error' field should be an object, not a string"
    );
    assert!(
        error_obj.get("message").is_some(),
        "The 'error' object should have a 'message' field"
    );
    assert!(
        error_obj.get("message").unwrap().is_string(),
        "The 'message' field should be a string"
    );
}
