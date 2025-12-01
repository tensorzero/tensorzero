use futures::StreamExt;
use http::StatusCode;
use reqwest::Client;
use reqwest_eventsource::{Event, RequestBuilderExt};
use serde_json::{json, Value};
use std::collections::HashMap;
use tensorzero_core::db::clickhouse::test_helpers::{
    get_clickhouse, select_chat_inference_clickhouse,
};
use uuid::Uuid;

use crate::{
    common::get_gateway_endpoint,
    providers::common::{E2ETestProvider, E2ETestProviders},
};

use super::common::ModelTestProvider;

crate::generate_provider_tests!(get_providers);
crate::generate_batch_inference_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let credentials = match std::env::var("GOOGLE_AI_STUDIO_API_KEY") {
        Ok(key) => HashMap::from([("google_ai_studio_api_key".to_string(), key)]),
        Err(_) => HashMap::new(),
    };

    let standard_providers = vec![
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "google-ai-studio-gemini-flash-8b".to_string(),
            model_name: "gemini-2.0-flash-lite".into(),
            model_provider_name: "google_ai_studio_gemini".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "google-ai-studio-gemini-2_5-pro".to_string(),
            model_name: "gemini-2.5-pro".into(),
            model_provider_name: "google_ai_studio_gemini".into(),
            credentials: HashMap::new(),
        },
    ];

    let image_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "google_ai_studio".to_string(),
        model_name: "google_ai_studio_gemini::gemini-2.0-flash-lite".into(),
        model_provider_name: "google_ai_studio_gemini".into(),
        credentials: HashMap::new(),
    }];
    let extra_body_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "google-ai-studio-gemini-flash-8b-extra-body".to_string(),
        model_name: "gemini-2.0-flash-lite".into(),
        model_provider_name: "google_ai_studio_gemini".into(),
        credentials: HashMap::new(),
    }];

    let bad_auth_extra_headers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "google-ai-studio-gemini-flash-8b-extra-headers".to_string(),
        model_name: "gemini-2.0-flash-lite".into(),
        model_provider_name: "google_ai_studio_gemini".into(),
        credentials: HashMap::new(),
    }];

    let inference_params_dynamic_providers = vec![
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "google-ai-studio-gemini-flash-8b-dynamic".to_string(),
            model_name: "gemini-2.0-flash-lite-dynamic".into(),
            model_provider_name: "google_ai_studio_gemini".into(),
            credentials: credentials.clone(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "google-ai-studio-gemini-2_5-pro-dynamic".to_string(),
            model_name: "gemini-2.5-pro-dynamic".into(),
            model_provider_name: "google_ai_studio_gemini".into(),
            credentials,
        },
    ];

    let tool_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "google-ai-studio-gemini-flash-8b".to_string(),
        model_name: "gemini-2.0-flash-lite".into(),
        model_provider_name: "google_ai_studio_gemini".into(),
        credentials: HashMap::new(),
    }];

    let json_providers = vec![
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "google-ai-studio-gemini-flash-8b".to_string(),
            model_name: "gemini-2.0-flash-lite".into(),
            model_provider_name: "google_ai_studio_gemini".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "google-ai-studio-gemini-flash-8b-implicit".to_string(),
            model_name: "gemini-2.0-flash-lite".into(),
            model_provider_name: "google_ai_studio_gemini".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "google-ai-studio-gemini-2_5-pro".to_string(),
            model_name: "gemini-2.5-pro".into(),
            model_provider_name: "google_ai_studio_gemini".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "google-ai-studio-gemini-2_5-pro-implicit".to_string(),
            model_name: "gemini-2.5-pro".into(),
            model_provider_name: "google_ai_studio_gemini".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "google-ai-studio-gemini-flash-8b-strict".to_string(),
            model_name: "gemini-2.0-flash-lite".into(),
            model_provider_name: "google_ai_studio_gemini".into(),
            credentials: HashMap::new(),
        },
    ];

    let json_mode_off_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "google_ai_studio_gemini_flash_8b_json_mode_off".to_string(),
        model_name: "gemini-2.0-flash-lite".into(),
        model_provider_name: "google_ai_studio_gemini".into(),
        credentials: HashMap::new(),
    }];

    let shorthand_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "google-ai-studio-gemini-flash-8b-shorthand".to_string(),
        model_name: "google_ai_studio_gemini::gemini-2.0-flash-lite".into(),
        model_provider_name: "google_ai_studio_gemini".into(),
        credentials: HashMap::new(),
    }];

    let provider_type_default_credentials_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "google-ai-studio-gemini-flash-8b".to_string(),
        model_name: "gemini-2.0-flash-lite".into(),
        model_provider_name: "google_ai_studio_gemini".into(),
        credentials: HashMap::new(),
    }];

    let provider_type_default_credentials_shorthand_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "google-ai-studio-gemini-flash-8b-shorthand".to_string(),
        model_name: "google_ai_studio_gemini::gemini-2.0-flash-lite".into(),
        model_provider_name: "google_ai_studio_gemini".into(),
        credentials: HashMap::new(),
    }];

    let credential_fallbacks = vec![ModelTestProvider {
        provider_type: "google_ai_studio_gemini".to_string(),
        model_info: HashMap::from([(
            "model_name".to_string(),
            "gemini-2.0-flash-lite".to_string(),
        )]),
        use_modal_headers: false,
    }];

    let input_audio_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "google_ai_studio_gemini".to_string(),
        model_name: "gemini-2.5-flash-lite".into(),
        model_provider_name: "google_ai_studio_gemini".into(),
        credentials: HashMap::new(),
    }];

    E2ETestProviders {
        simple_inference: standard_providers.clone(),
        extra_body_inference: extra_body_providers,
        bad_auth_extra_headers,
        reasoning_inference: vec![],
        embeddings: vec![],
        inference_params_inference: standard_providers.clone(),
        inference_params_dynamic_credentials: inference_params_dynamic_providers,
        provider_type_default_credentials: provider_type_default_credentials_providers,
        provider_type_default_credentials_shorthand:
            provider_type_default_credentials_shorthand_providers,
        tool_use_inference: tool_providers.clone(),
        tool_multi_turn_inference: tool_providers.clone(),
        dynamic_tool_use_inference: tool_providers.clone(),
        parallel_tool_use_inference: vec![],
        json_mode_inference: json_providers.clone(),
        json_mode_off_inference: json_mode_off_providers.clone(),
        image_inference: image_providers.clone(),
        pdf_inference: image_providers,
        input_audio: input_audio_providers,
        shorthand_inference: shorthand_providers.clone(),
        credential_fallbacks,
    }
}

#[tokio::test]
async fn test_gemini_multi_turn_thought_non_streaming() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();
    let payload = json!({
        "function_name": "weather_helper",
        "variant_name": "google-ai-studio-gemini-2_5-pro",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "AskJeeves"},
            "messages": [
                {
                    "role": "user",
                    "content": "Hi I'm visiting Brooklyn from Brazil. What's the weather?"
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

    println!("Original Content blocks: {content_blocks:?}");
    assert!(
        content_blocks.len() == 2,
        "Unexpected content blocks: {content_blocks:?}"
    );
    let signature = content_blocks[0]
        .get("signature")
        .unwrap()
        .as_str()
        .unwrap();
    assert_eq!(
        content_blocks[0],
        json!({
            "type": "thought",
            "text": null,
            "signature": signature,
            "_internal_provider_type": "google_ai_studio_gemini",
        })
    );
    assert_eq!(content_blocks[1]["type"], "tool_call");
    let tool_id = content_blocks[1]["id"].as_str().unwrap();

    let tensorzero_content_blocks = content_blocks.clone();

    let mut new_messages = vec![
        serde_json::json!({
            "role": "user",
            "content": "Hi I'm visiting Brooklyn from Brazil. What's the weather?"
        }),
        serde_json::json!({
            "role": "assistant",
            "content": tensorzero_content_blocks,
        }),
    ];

    new_messages.push(serde_json::json!({
        "role": "user",
        "content": [{"type": "tool_result", "name": "My result", "result": "13", "id": tool_id}],
    }));

    let payload = json!({
        "function_name": "weather_helper",
        "variant_name": "google-ai-studio-gemini-2_5-pro",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "AskJeeves"},
            "messages": new_messages
        },
        "stream": false,
    });
    println!("New payload: {payload}");

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    let response_json = response.json::<Value>().await.unwrap();
    let new_content_blocks = response_json.get("content").unwrap().as_array().unwrap();
    assert_eq!(
        new_content_blocks.len(),
        1,
        "Unexpected new content blocks: {new_content_blocks:?}"
    );
    assert_eq!(new_content_blocks[0]["type"], "text");

    // Don't bother checking ClickHouse, as we do that in lots of other tests
}

#[tokio::test]
async fn test_gemini_multi_turn_thought_streaming() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();
    let payload = json!({
        "function_name": "weather_helper",
        "variant_name": "google-ai-studio-gemini-2_5-pro",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "AskJeeves"},
            "messages": [
                {
                    "role": "user",
                    "content": "Hi I'm visiting Brooklyn from Brazil. What's the weather?"
                }
            ]},
        "stream": true,
    });

    let mut event_source = client
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .eventsource()
        .unwrap();
    let mut chunks = vec![];
    while let Some(event) = event_source.next().await {
        let event = event.unwrap();
        match event {
            Event::Open => continue,
            Event::Message(message) => {
                if message.data == "[DONE]" {
                    break;
                }
                chunks.push(message.data);
            }
        }
    }

    let mut inference_id = None;

    // Just validate that all of the chunks are valid JSON,
    // and then check the `collect_chunks` result stored in the databsae
    for chunk in chunks {
        let chunk_json: Value = serde_json::from_str(&chunk).unwrap();
        println!("Chunk: {chunk_json}");
        inference_id = Some(
            chunk_json
                .get("inference_id")
                .unwrap()
                .as_str()
                .unwrap()
                .to_string(),
        );
    }
    let inference_id = inference_id.unwrap().parse::<Uuid>().unwrap();

    // Sleep for 1 second to allow time for data to be inserted into ClickHouse (trailing writes from API)
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    // Check ClickHouse
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    let id = result.get("id").unwrap().as_str().unwrap();
    let _id_uuid = Uuid::parse_str(id).unwrap();

    let clickhouse_content_blocks = result.get("output").unwrap().as_str().unwrap();
    let clickhouse_content_blocks: Vec<Value> =
        serde_json::from_str(clickhouse_content_blocks).unwrap();
    assert_eq!(clickhouse_content_blocks.len(), 2);
    let signature = clickhouse_content_blocks[0]
        .get("signature")
        .unwrap()
        .as_str()
        .unwrap();
    assert_eq!(
        clickhouse_content_blocks[0],
        json!({
            "type": "thought",
            "text": null,
            "signature": signature,
            "_internal_provider_type": "google_ai_studio_gemini",
        })
    );
    assert_eq!(clickhouse_content_blocks[1]["type"], "tool_call");
}

#[tokio::test]
async fn test_gemini_invalid_thought() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();
    let payload = json!({
        "model_name": "google_ai_studio_gemini::gemini-2.5-pro",
        "episode_id": episode_id,
        "input":{
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thought",
                            "signature": "===",
                        },
                        {
                            "type": "text",
                            "text": "Fake assistant message",
                        }
                    ]
                },
            ]},
        "stream": false,
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    // Google AI Studio should reject the request due to an invalid thought signature
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let response_json = response.json::<Value>().await.unwrap();
    let error = response_json.get("error").unwrap().as_str().unwrap();
    assert!(
        error.contains("Base64 decoding failed"),
        "Unexpected error message: {error}"
    );
}

#[tokio::test]
async fn test_gemini_trailing_thought() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();
    let payload = json!({
        "model_name": "google_ai_studio_gemini::gemini-2.5-pro",
        "episode_id": episode_id,
        "input":{
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thought",
                            "signature": "fake_signature",
                        },
                    ]
                },
            ]},
        "stream": false,
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    // Google AI Studio should reject the request due to an invalid thought signature
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let response_json = response.json::<Value>().await.unwrap();
    let error = response_json.get("error").unwrap().as_str().unwrap();
    assert!(
        error
            .contains("Thought block with signature must be followed by a content block in Gemini"),
        "Unexpected error message: {error}"
    );
}

#[tokio::test]
async fn test_gemini_double_thought() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();
    let payload = json!({
        "model_name": "google_ai_studio_gemini::gemini-2.5-pro",
        "episode_id": episode_id,
        "input":{
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thought",
                            "signature": "fake_signature",
                        },
                        {
                            "type": "thought",
                            "signature": "other_signature",
                        },
                    ]
                },
            ]},
        "stream": false,
    });

    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    // Google AI Studio should reject the request due to an invalid thought signature
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let response_json = response.json::<Value>().await.unwrap();
    let error = response_json.get("error").unwrap().as_str().unwrap();
    assert!(
        error.contains(
            "Thought block with signature cannot be followed by another thought block in Gemini"
        ),
        "Unexpected error message: {error}"
    );
}

/// Test that when tool_choice is "auto" but allowed_tools is set,
/// the model is forced to use tools (because internally we set mode to Any).
#[tokio::test]
async fn test_google_ai_studio_gemini_tool_choice_auto_with_allowed_tools() {
    let episode_id = Uuid::now_v7();

    let payload = json!({
        "function_name": "basic_test",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [
                {
                    "role": "user",
                    "content": "What can you tell me about the weather in Tokyo (e.g. temperature, humidity, wind)? Use the provided tools and return what you can (not necessarily everything)."
                }
            ]},
        "tool_choice": "auto",
        "allowed_tools": ["get_humidity"],
        "stream": false,
        "variant_name": "google-ai-studio-gemini-flash-8b",
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let response_json = response.json::<Value>().await.unwrap();
    println!("API response: {response_json:#?}");

    // Verify the response contains a tool call to get_humidity
    let inference_id = response_json.get("inference_id").unwrap().as_str().unwrap();
    let inference_id = Uuid::parse_str(inference_id).unwrap();

    let content = response_json.get("content").unwrap().as_array().unwrap();
    assert!(
        !content.is_empty(),
        "Response should contain content blocks"
    );

    let tool_call = content
        .iter()
        .find(|block| block["type"] == "tool_call")
        .expect("Response should contain a tool_call block");
    assert_eq!(
        tool_call.get("name").unwrap().as_str().unwrap(),
        "get_humidity"
    );

    // Check ClickHouse ChatInference
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    let clickhouse = get_clickhouse().await;
    let result = select_chat_inference_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();

    let tool_params = result.get("tool_params").unwrap().as_str().unwrap();
    let tool_params: Value = serde_json::from_str(tool_params).unwrap();
    assert_eq!(tool_params.get("tool_choice").unwrap(), "auto");

    // tools_available should only contain get_humidity since we specified allowed_tools
    let tools_available = tool_params
        .get("tools_available")
        .unwrap()
        .as_array()
        .unwrap();
    assert_eq!(
        tools_available.len(),
        1,
        "Should only have one tool available"
    );
    assert_eq!(tools_available[0].get("name").unwrap(), "get_humidity");
}
