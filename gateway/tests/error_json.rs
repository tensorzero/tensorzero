use http::StatusCode;
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

    let json: Value =
        serde_json::from_str(&inference_response).expect("Response should be valid JSON");

    // Verify TensorZero error format with both `error_json` and `unstable_error_json`
    assert_eq!(
        json.get("error").and_then(|v| v.as_str()),
        Some(
            "Failed to parse the request body as JSON: EOF while parsing a value at line 1 column 0 (400 Bad Request)"
        ),
        "Response should have correct error message"
    );
    assert!(
        json.get("error_json").is_some(),
        "Response should have `error_json` field"
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
        "Response should have an `error` field"
    );
    let error_obj = json.get("error").unwrap();
    assert!(
        error_obj.is_object(),
        "The `error` field should be an object, not a string"
    );
    assert!(
        error_obj.get("message").is_some(),
        "The `error` object should have a `message` field"
    );
    assert!(
        error_obj.get("message").unwrap().is_string(),
        "The `message` field should be a string"
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
        "Response should have an `error` field"
    );
    let error_obj = json.get("error").unwrap();
    assert!(
        error_obj.is_object(),
        "The `error` field should be an object, not a string"
    );
    assert!(
        error_obj.get("message").is_some(),
        "The `error` object should have a `message` field"
    );
    assert!(
        error_obj.get("message").unwrap().is_string(),
        "The `message` field should be a string"
    );
}

/// Test that OpenAI-compatible endpoint includes `error_json` when `UNSTABLE_ERROR_JSON` is enabled
#[tokio::test]
async fn test_openai_compatible_error_json() {
    let child_data = start_gateway_on_random_port("unstable_error_json = true", None).await;
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

    // Verify OpenAI error format with `error_json` and `tensorzero_error_json`:
    // {"error": {"message": "...", "error_json": {...}, "tensorzero_error_json": {...}}}
    let error_obj = json
        .get("error")
        .expect("Response should have an `error` field");
    assert!(
        error_obj.is_object(),
        "The `error` field should be an object"
    );
    assert!(
        error_obj.get("message").is_some(),
        "The `error` object should have a `message` field"
    );
    assert!(
        error_obj.get("error_json").is_some(),
        "The `error` object should have an `error_json` field when `UNSTABLE_ERROR_JSON` is enabled"
    );
    assert!(
        error_obj.get("error_json").unwrap().is_object(),
        "The `error_json` field should be an object"
    );
    assert!(
        error_obj.get("tensorzero_error_json").is_some(),
        "The `error` object should have a `tensorzero_error_json` field when `UNSTABLE_ERROR_JSON` is enabled"
    );
    assert!(
        error_obj.get("tensorzero_error_json").unwrap().is_object(),
        "The `tensorzero_error_json` field should be an object"
    );
    // Both fields should have identical content
    assert_eq!(
        error_obj.get("error_json"),
        error_obj.get("tensorzero_error_json"),
        "`error_json` and `tensorzero_error_json` should be identical"
    );
}

/// Test that auth errors on TensorZero endpoints return TensorZero format
#[tokio::test]
async fn test_auth_error_tensorzero_format() {
    let child_data = start_gateway_on_random_port(
        "
    [gateway.auth]
    enabled = true
    ",
        None,
    )
    .await;

    let response = reqwest::Client::new()
        .post(format!("http://{}/inference", child_data.addr))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    let text = response.text().await.unwrap();
    let json: Value = serde_json::from_str(&text).expect("Response should be valid JSON");

    // Verify TensorZero error format: {"error": "..."}
    assert!(
        json.get("error").is_some(),
        "Response should have an `error` field"
    );
    assert!(
        json.get("error").unwrap().is_string(),
        "The `error` field should be a string for TensorZero format"
    );
    assert!(
        json.get("error")
            .unwrap()
            .as_str()
            .unwrap()
            .contains("TensorZero authentication error"),
        "Error message should mention authentication"
    );
}

/// Test that auth errors on OpenAI endpoints return OpenAI format
#[tokio::test]
async fn test_auth_error_openai_format() {
    let child_data = start_gateway_on_random_port(
        "
    [gateway.auth]
    enabled = true
    ",
        None,
    )
    .await;

    let response = reqwest::Client::new()
        .post(format!(
            "http://{}/openai/v1/chat/completions",
            child_data.addr
        ))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    let text = response.text().await.unwrap();
    let json: Value = serde_json::from_str(&text).expect("Response should be valid JSON");

    // Verify OpenAI error format: {"error": {"message": "..."}}
    assert!(
        json.get("error").is_some(),
        "Response should have an `error` field"
    );
    let error_obj = json.get("error").unwrap();
    assert!(
        error_obj.is_object(),
        "The `error` field should be an object for OpenAI format"
    );
    assert!(
        error_obj.get("message").is_some(),
        "The `error` object should have a `message` field"
    );
    assert!(
        error_obj
            .get("message")
            .unwrap()
            .as_str()
            .unwrap()
            .contains("TensorZero authentication error"),
        "Error message should mention authentication"
    );
}

/// Test that auth errors on OpenAI endpoints use OpenAI format with a base path
#[tokio::test]
async fn test_auth_error_openai_format_with_base_path() {
    let child_data = start_gateway_on_random_port(
        r#"
    base_path = "/my/prefix"

    [gateway.auth]
    enabled = true
    "#,
        None,
    )
    .await;

    let response = reqwest::Client::new()
        .post(format!(
            "http://{}/my/prefix/openai/v1/chat/completions",
            child_data.addr
        ))
        .send()
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::UNAUTHORIZED,
        "Auth errors should return 401 for OpenAI endpoints with a base path"
    );
    let text = response.text().await.unwrap();
    let json: Value = serde_json::from_str(&text).expect("Response should be valid JSON");

    assert!(
        json.get("error").is_some(),
        "Response should have an `error` field"
    );
    let error_obj = json.get("error").unwrap();
    assert!(
        error_obj.is_object(),
        "The `error` field should be an object for OpenAI format with a base path"
    );
    assert!(
        error_obj.get("message").and_then(Value::as_str).is_some(),
        "The `error` object should have a string `message` field"
    );
}

/// Test that auth errors on TensorZero endpoints include `error_json` when enabled
#[tokio::test]
async fn test_auth_error_tensorzero_format_with_error_json() {
    let child_data = start_gateway_on_random_port(
        "
    unstable_error_json = true

    [gateway.auth]
    enabled = true
    ",
        None,
    )
    .await;

    let response = reqwest::Client::new()
        .post(format!("http://{}/inference", child_data.addr))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    let text = response.text().await.unwrap();
    let json: Value = serde_json::from_str(&text).expect("Response should be valid JSON");

    // Verify TensorZero error format with `error_json`
    assert!(
        json.get("error").is_some() && json.get("error").unwrap().is_string(),
        "Response should have a string `error` field"
    );
    assert!(
        json.get("error_json").is_some(),
        "Response should have `error_json` field when `unstable_error_json` is enabled"
    );
}

/// Test that auth errors on OpenAI endpoints include `error_json` when enabled
#[tokio::test]
async fn test_auth_error_openai_format_with_error_json() {
    let child_data = start_gateway_on_random_port(
        "
    unstable_error_json = true

    [gateway.auth]
    enabled = true
    ",
        None,
    )
    .await;

    let response = reqwest::Client::new()
        .post(format!(
            "http://{}/openai/v1/chat/completions",
            child_data.addr
        ))
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    let text = response.text().await.unwrap();
    let json: Value = serde_json::from_str(&text).expect("Response should be valid JSON");

    // Verify OpenAI error format with `error_json` and `tensorzero_error_json`
    let error_obj = json
        .get("error")
        .expect("Response should have an `error` field");
    assert!(
        error_obj.is_object(),
        "The `error` field should be an object"
    );
    assert!(
        error_obj.get("message").is_some(),
        "The `error` object should have a `message` field"
    );
    assert!(
        error_obj.get("error_json").is_some(),
        "The `error` object should have an `error_json` field when `unstable_error_json` is enabled"
    );
    assert!(
        error_obj.get("tensorzero_error_json").is_some(),
        "The `error` object should have a `tensorzero_error_json` field when `unstable_error_json` is enabled"
    );
    assert_eq!(
        error_obj.get("error_json"),
        error_obj.get("tensorzero_error_json"),
        "`error_json` and `tensorzero_error_json` should be identical"
    );
}
