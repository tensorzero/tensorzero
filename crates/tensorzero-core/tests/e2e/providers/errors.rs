use futures::StreamExt;
use reqwest::{Client, StatusCode};
use serde_json::{Value, json};
use tensorzero::{
    ClientInferenceParams, InferenceOutput, Input, InputMessage, InputMessageContent, Role,
};
use tensorzero_core::endpoints::inference::{ChatCompletionInferenceParams, InferenceParams};
use tensorzero_core::inference::types::Text;

use crate::common::get_gateway_endpoint;

async fn test_payload_produces_error(payload: Value, expected_err: &str) {
    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    let status = response.status();
    let response_json = response.json::<Value>().await.unwrap();
    let error_msg = response_json["error"].as_str().unwrap();
    assert_eq!(expected_err, error_msg);
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_bad_text_input() {
    test_payload_produces_error(
        json!({
            "function_name": "basic_test",
            "input":
                {
                   "system": {"assistant_name": "Dr. Mehta"},
                   "messages": [
                    {
                        "role": "user",
                        "content": [{
                            "type": "text",
                            "bad_field": "Blah"
                        }]
                    }
                ]},

        }),
        "input.messages[0].content[0]: Unknown key `bad_field` in text content",
    )
    .await;

    test_payload_produces_error(
        json!({
            "function_name": "basic_test",
            "input":
                {
                   "system": {"assistant_name": "Dr. Mehta"},
                   "messages": [
                    {
                        "role": "user",
                        "content": [{
                            "type": "text",
                            "bad_field": "Blah",
                            "bad_field_2": "Other",
                        }]
                    }
                ]},

        }),
        "input.messages[0].content[0]: Expected exactly one other key in text content, found 2 other keys",
    )
    .await;

    test_payload_produces_error(
        json!({
            "function_name": "basic_test",
            "input":
                {
                   "system": {"assistant_name": "Dr. Mehta"},
                   "messages": [
                    {
                        "role": "user",
                        "content": [{
                            "type": "text",
                        }]
                    }
                ]},

        }),
        "input.messages[0].content[0]: Expected exactly one other key in text content, found 0 other keys",
    )
    .await;

    test_payload_produces_error(
        json!({
            "function_name": "basic_test",
            "input":
                {
                   "system": {"assistant_name": "Dr. Mehta"},
                   "messages": [
                    {
                        "role": "user",
                        "content": [{
                            "type": "text",
                            "text": ["Not", "a", "string"]
                        }]
                    }
                ]},

        }),
        "input.messages[0].content[0]: Error deserializing `text`: invalid type: sequence, expected a string",
    )
    .await;

    test_payload_produces_error(
        json!({
            "function_name": "basic_test",
            "input":
                {
                   "system": {"assistant_name": "Dr. Mehta"},
                   "messages": [
                    {
                        "role": "user",
                        "content": [{
                            "type": "text",
                            "arguments": "Not an object"
                        }]
                    }
                ]},

        }),
        "input.messages[0].content[0]: Error deserializing `arguments`: invalid type: string \"Not an object\", expected a map",
    )
    .await;
}

/// Test that provider error messages are included in the error response (non-streaming).
/// This test sends an invalid `thinking_budget_tokens` value to Anthropic which should
/// return an error message containing "budget_tokens".
#[tokio::test]
async fn test_error_propagates_non_streaming_http_gateway() {
    let payload = json!({
        "model_name": "anthropic::claude-haiku-4-5",
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": "Hello"
                }
            ]
        },
        "params": {
            "chat_completion": {
                "thinking_budget_tokens": -9999
            }
        },
        "stream": false
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    let status = response.status();
    let response_json = response.json::<Value>().await.unwrap();
    let error_msg = response_json["error"].as_str().unwrap();

    assert!(!status.is_success(), "Expected error status, got: {status}");
    assert!(
        error_msg.contains("budget_tokens"),
        "Error message should contain 'budget_tokens' from Anthropic's error response. Got: {error_msg}"
    );
}

/// Test that provider error messages are included in the error response (streaming).
/// This test sends an invalid `thinking_budget_tokens` value to Anthropic which should
/// return an error message containing "budget_tokens".
#[tokio::test]
async fn test_error_propagates_streaming_http_gateway() {
    let payload = json!({
        "model_name": "anthropic::claude-haiku-4-5",
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": "Hello"
                }
            ]
        },
        "params": {
            "chat_completion": {
                "thinking_budget_tokens": -9999
            }
        },
        "stream": true
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    let status = response.status();
    let response_text = response.text().await.unwrap();

    assert!(!status.is_success(), "Expected error status, got: {status}");
    assert!(
        response_text.contains("budget_tokens"),
        "Error response should contain 'budget_tokens' from Anthropic's error response. Got: {response_text}"
    );
}

/// Test that provider error messages are included in the error response (non-streaming)
/// using the embedded gateway.
#[tokio::test(flavor = "multi_thread")]
async fn test_error_propagates_non_streaming_embedded_gateway() {
    let client = tensorzero::test_helpers::make_embedded_gateway().await;

    let result = client
        .inference(ClientInferenceParams {
            model_name: Some("anthropic::claude-haiku-4-5".to_string()),
            input: Input {
                system: None,
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: "Hello".into(),
                    })],
                }],
            },
            stream: Some(false),
            params: InferenceParams {
                chat_completion: ChatCompletionInferenceParams {
                    thinking_budget_tokens: Some(-9999),
                    ..Default::default()
                },
            },
            ..Default::default()
        })
        .await;

    let err = result.expect_err("Expected an error from Anthropic");
    let err_str = err.to_string();

    assert!(
        err_str.contains("budget_tokens"),
        "Error message should contain 'budget_tokens' from Anthropic's error response. Got: {err_str}"
    );
}

/// Test that provider error messages are included in the error response (streaming)
/// using the embedded gateway.
#[tokio::test(flavor = "multi_thread")]
async fn test_error_propagates_streaming_embedded_gateway() {
    let client = tensorzero::test_helpers::make_embedded_gateway().await;

    let result = client
        .inference(ClientInferenceParams {
            model_name: Some("anthropic::claude-haiku-4-5".to_string()),
            input: Input {
                system: None,
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: "Hello".into(),
                    })],
                }],
            },
            stream: Some(true),
            params: InferenceParams {
                chat_completion: ChatCompletionInferenceParams {
                    thinking_budget_tokens: Some(-9999),
                    ..Default::default()
                },
            },
            ..Default::default()
        })
        .await;

    // For streaming, the error may come back as either:
    // 1. An error from the initial connection (if the provider rejects immediately)
    // 2. An error in the stream (if the provider starts streaming then errors)
    match result {
        Err(err) => {
            let err_str = err.to_string();
            assert!(
                err_str.contains("budget_tokens"),
                "Error message should contain 'budget_tokens' from Anthropic's error response. Got: {err_str}"
            );
        }
        Ok(InferenceOutput::Streaming(stream)) => {
            let chunks: Vec<_> = stream.collect().await;
            let has_error_with_budget_tokens = chunks.iter().any(|chunk| {
                if let Err(err) = chunk {
                    err.to_string().contains("budget_tokens")
                } else {
                    false
                }
            });
            assert!(
                has_error_with_budget_tokens,
                "Expected an error chunk containing 'budget_tokens' in the stream"
            );
        }
        Ok(InferenceOutput::NonStreaming(_)) => {
            panic!("Expected streaming output or error, got non-streaming output");
        }
    }
}
