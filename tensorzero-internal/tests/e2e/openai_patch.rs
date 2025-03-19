use axum::http::StatusCode;
use futures::StreamExt;
use reqwest::Client;
use reqwest_eventsource::{Event, RequestBuilderExt};
use serde_json::json;
use uuid::Uuid;

use crate::common::get_gateway_endpoint;

#[cfg(feature = "e2e_tests")]
#[tokio::test]
async fn test_openai_patch_basic() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "model": "tensorzero::model_name::dummy::json",
        "messages": [
            {
                "role": "user",
                "content": "Write a haiku about artificial intelligence."
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

    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<serde_json::Value>().await.unwrap();
    let choices = response_json.get("choices").unwrap().as_array().unwrap();
    assert_eq!(choices.len(), 1);
    let choice = choices.first().unwrap();
    let message = choice.get("message").unwrap();
    let content = message.get("content").unwrap().as_str().unwrap();
    assert_eq!(content, r#"{"answer":"Hello"}"#);
    let finish_reason = choice.get("finish_reason").unwrap().as_str().unwrap();
    assert_eq!(finish_reason, "stop");
}

#[cfg(feature = "e2e_tests")]
#[tokio::test]
async fn test_openai_patch_streaming() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "model": "tensorzero::model_name::dummy::json",
        "messages": [
            {
                "role": "user",
                "content": "Write a haiku about artificial intelligence."
            }
        ],
        "stream": true,
    });

    let mut response = client
        .post(get_gateway_endpoint("/openai/v1/chat/completions"))
        .header("episode_id", episode_id.to_string())
        .json(&payload)
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

    // Verify chunk properties
    let mut previous_inference_id = None;
    for chunk in chunks {
        let chunk_json: serde_json::Value = serde_json::from_str(&chunk).unwrap();
        let inference_id = chunk_json.get("id").unwrap().as_str().unwrap().to_string();
        
        if let Some(prev_id) = previous_inference_id {
            assert_eq!(inference_id, prev_id);
        }
        previous_inference_id = Some(inference_id);

        let model = chunk_json.get("model").unwrap().as_str().unwrap();
        assert_eq!(model, "tensorzero::model_name::dummy::json::variant_name::test");

        let choices = chunk_json.get("choices").unwrap().as_array().unwrap();
        let choice = choices.first().unwrap();
        let delta = choice.get("delta").unwrap();
        
        if choice.get("finish_reason").is_none() {
            assert!(delta.get("content").is_some());
        } else {
            assert!(delta.get("content").is_none());
            let usage = chunk_json.get("usage").unwrap();
            assert_eq!(usage.get("prompt_tokens").unwrap().as_u64().unwrap(), 10);
            assert_eq!(usage.get("completion_tokens").unwrap().as_u64().unwrap(), 16);
            assert_eq!(usage.get("total_tokens").unwrap().as_u64().unwrap(), 26);
            assert_eq!(choice.get("finish_reason").unwrap().as_str().unwrap(), "stop");
        }
    }
}

#[cfg(feature = "e2e_tests")]
#[tokio::test]
async fn test_openai_patch_with_tools() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "model": "tensorzero::function_name::weather_helper",
        "messages": [
            {
                "role": "user",
                "content": "What's the weather in Boston?"
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_temperature",
                    "description": "Get the temperature in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA"
                            },
                            "units": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"]
                            }
                        },
                        "required": ["location", "units"]
                    }
                }
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

    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<serde_json::Value>().await.unwrap();
    let choices = response_json.get("choices").unwrap().as_array().unwrap();
    let choice = choices.first().unwrap();
    assert_eq!(choice.get("finish_reason").unwrap().as_str().unwrap(), "tool_calls");
    
    let message = choice.get("message").unwrap();
    let tool_calls = message.get("tool_calls").unwrap().as_array().unwrap();
    assert_eq!(tool_calls.len(), 1);
    
    let tool_call = tool_calls.first().unwrap();
    assert_eq!(tool_call.get("type").unwrap().as_str().unwrap(), "function");
    assert_eq!(tool_call.get("function").unwrap().get("name").unwrap().as_str().unwrap(), "get_temperature");
    
    let arguments = tool_call.get("function").unwrap().get("arguments").unwrap().as_str().unwrap();
    let arguments_json: serde_json::Value = serde_json::from_str(arguments).unwrap();
    assert!(arguments_json.get("location").is_some());
    assert!(arguments_json.get("units").is_some());
}

#[cfg(feature = "e2e_tests")]
#[tokio::test]
async fn test_openai_patch_with_json_mode() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "model": "tensorzero::function_name::json_success",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "value": {"country": "Japan"}}]
            }
        ],
        "response_format": {
            "type": "json_object"
        },
        "stream": false,
    });

    let response = client
        .post(get_gateway_endpoint("/openai/v1/chat/completions"))
        .header("episode_id", episode_id.to_string())
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<serde_json::Value>().await.unwrap();
    let choices = response_json.get("choices").unwrap().as_array().unwrap();
    let choice = choices.first().unwrap();
    let message = choice.get("message").unwrap();
    let content = message.get("content").unwrap().as_str().unwrap();
    assert_eq!(content, r#"{"answer":"Hello"}"#);
}

#[cfg(feature = "e2e_tests")]
#[tokio::test]
async fn test_openai_patch_error_handling() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "model": "tensorzero::function_name::does_not_exist",
        "messages": [
            {
                "role": "user",
                "content": "Write a haiku about artificial intelligence."
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

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let error_json = response.json::<serde_json::Value>().await.unwrap();
    let error = error_json.get("error").unwrap().as_str().unwrap();
    assert!(error.contains("Unknown function: does_not_exist"));
} 