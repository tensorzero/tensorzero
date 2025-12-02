use std::collections::HashMap;

use http::StatusCode;
use reqwest::Client;
use serde_json::{json, Value};
use uuid::Uuid;

use crate::{
    common::get_gateway_endpoint,
    providers::common::{E2ETestProvider, E2ETestProviders},
};

use super::common::ModelTestProvider;

crate::generate_provider_tests!(get_providers);
crate::generate_batch_inference_tests!(get_providers);
crate::generate_unified_mock_batch_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    // TODO - fine-tune a better model and add it back to our tests
    let _tuned = E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "gcp-vertex-gemini-flash-lite-tuned".to_string(),
        model_name: "gemini-2.0-flash-lite-tuned".into(),
        model_provider_name: "gcp_vertex_gemini".into(),
        credentials: HashMap::new(),
    };
    let standard_providers = vec![
        E2ETestProvider {
            supports_batch_inference: true,
            variant_name: "gcp-vertex-gemini-flash".to_string(),
            model_name: "gemini-2.0-flash-001".into(),
            model_provider_name: "gcp_vertex_gemini".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "gcp-vertex-gemini-pro".to_string(),
            model_name: "gcp-gemini-2.5-pro".into(),
            model_provider_name: "gcp_vertex_gemini".into(),
            credentials: HashMap::new(),
        },
    ];

    let tool_providers = vec![
        E2ETestProvider {
            supports_batch_inference: true,
            variant_name: "gcp-vertex-gemini-flash".to_string(),
            model_name: "gemini-2.0-flash-001".into(),
            model_provider_name: "gcp_vertex_gemini".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "gcp-vertex-gemini-pro".to_string(),
            model_name: "gcp-gemini-2.5-pro".into(),
            model_provider_name: "gcp_vertex_gemini".into(),
            credentials: HashMap::new(),
        },
    ];

    let image_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "gcp_vertex".to_string(),
        model_name: "gcp-gemini-2.5-pro".into(),
        model_provider_name: "gcp_vertex_gemini".into(),
        credentials: HashMap::new(),
    }];

    let pdf_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "gcp_vertex_gemini".to_string(),
        model_name: "gcp_vertex_gemini::projects/tensorzero-public/locations/us-central1/publishers/google/models/gemini-2.0-flash-lite".into(),
        model_provider_name: "gcp_vertex_gemini".into(),
        credentials: HashMap::new(),
    }];

    let input_audio_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "gcp_vertex_gemini".to_string(),
        model_name: "gemini-2.5-flash-lite".into(),
        model_provider_name: "gcp_vertex_gemini".into(),
        credentials: HashMap::new(),
    }];

    let extra_body_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "gcp-vertex-gemini-flash-extra-body".to_string(),
        model_name: "gemini-2.0-flash-001".into(),
        model_provider_name: "gcp_vertex_gemini".into(),
        credentials: HashMap::new(),
    }];

    let bad_auth_extra_headers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "gcp-vertex-gemini-flash-extra-headers".to_string(),
        model_name: "gemini-2.0-flash-001".into(),
        model_provider_name: "gcp_vertex_gemini".into(),
        credentials: HashMap::new(),
    }];

    let json_providers = vec![
        E2ETestProvider {
            supports_batch_inference: true,
            variant_name: "gcp-vertex-gemini-flash".to_string(),
            model_name: "gemini-2.0-flash-001".into(),
            model_provider_name: "gcp_vertex_gemini".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "gcp-vertex-gemini-pro".to_string(),
            model_name: "gcp-gemini-2.5-pro".into(),
            model_provider_name: "gcp_vertex_gemini".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "gcp-vertex-gemini-pro-implicit".to_string(),
            model_name: "gcp-gemini-2.5-pro".into(),
            model_provider_name: "gcp_vertex_gemini".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "gcp-vertex-gemini-flash-strict".to_string(),
            model_name: "gemini-2.0-flash-001".into(),
            model_provider_name: "gcp_vertex_gemini".into(),
            credentials: HashMap::new(),
        },
    ];

    let json_mode_off_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "gcp_vertex_gemini_flash_json_mode_off".to_string(),
        model_name: "gemini-2.0-flash-001".into(),
        model_provider_name: "gcp_vertex_gemini".into(),
        credentials: HashMap::new(),
    }];

    let shorthand_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "gcp_vertex_gemini_shorthand".to_string(),
        model_name: "gcp_vertex_gemini::projects/tensorzero-public/locations/us-central1/publishers/google/models/gemini-2.0-flash-001".into(),
        model_provider_name: "gcp_vertex_gemini".into(),
        credentials: HashMap::new(),
    }];

    let credential_fallbacks = vec![ModelTestProvider {
        provider_type: "gcp_vertex_gemini".to_string(),
        model_info: HashMap::from([
            ("model_id".to_string(), "gemini-2.0-flash-001".to_string()),
            ("location".to_string(), "us-central1".to_string()),
            ("project_id".to_string(), "tensorzero-public".to_string()),
        ]),
        use_modal_headers: false,
    }];

    E2ETestProviders {
        simple_inference: standard_providers.clone(),
        extra_body_inference: extra_body_providers,
        bad_auth_extra_headers,
        reasoning_inference: vec![],
        embeddings: vec![],
        inference_params_inference: standard_providers.clone(),
        inference_params_dynamic_credentials: vec![],
        provider_type_default_credentials: vec![],
        provider_type_default_credentials_shorthand: vec![],
        tool_use_inference: tool_providers.clone(),
        tool_multi_turn_inference: tool_providers.clone(),
        dynamic_tool_use_inference: tool_providers.clone(),
        parallel_tool_use_inference: vec![],
        json_mode_inference: json_providers.clone(),
        json_mode_off_inference: json_mode_off_providers.clone(),
        image_inference: image_providers.clone(),
        pdf_inference: pdf_providers,
        input_audio: input_audio_providers,
        shorthand_inference: shorthand_providers.clone(),
        credential_fallbacks,
    }
}

// Specifying `tool_choice: none` causes Gemini 2.5 to emit an 'UNEXPECTED_TOOL_CALL'
// error. For now, we disable this test until we decide how to handle this:
// https://github.com/tensorzero/tensorzero/issues/2329
#[tokio::test]
#[ignore]
async fn test_gcp_pro_tool_choice_none() {
    let episode_id = Uuid::now_v7();
    let payload = json!({
        "function_name": "weather_helper",
        "episode_id": episode_id,
        "input":{
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather like in Tokyo (in Celsius)? Use the `get_temperature` tool."
                }
            ]},
        "tool_choice": "none",
        "stream": false,
        "variant_name": "gcp-vertex-gemini-pro",
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    let status = response.status();

    let response_json: Value = response.json().await.unwrap();
    println!("API response: {response_json:#?}");

    assert_eq!(status, StatusCode::OK);
    let output = response_json.get("content").unwrap();
    let output = output.as_array().unwrap();
    let unknown_block = output
        .iter()
        .find(|block| block.get("type").unwrap() == "unknown")
        .expect("Missing 'unknown' block in output: {output}");
    assert_eq!(
        unknown_block.get("model_name").unwrap().as_str(),
        Some("gcp-gemini-2.5-pro")
    );
    assert_eq!(
        unknown_block.get("provider_name").unwrap().as_str(),
        Some("gcp_vertex_gemini")
    );
    assert!(unknown_block
        .get("data")
        .unwrap()
        .as_object()
        .unwrap()
        .get("executableCode")
        .is_some());
}

/// There are fields for both model_name and endpoint_id and we can't know a priori which one to use.
/// We test here that the error message is correct and helpful.
#[tokio::test]
async fn test_gcp_vertex_gemini_bad_model_id() {
    let episode_id = Uuid::now_v7();
    let payload = json!({
        "function_name": "basic_test",
        "episode_id": episode_id,
        "input":
            {
               "system": {"assistant_name": "Dr. Mehta"},
               "messages": [
                {
                    "role": "user",
                    "content": "What is the name of the capital city of Japan?"
                }
            ]},
        "stream": false,
        "variant_name": "gemini-bad-model-name",
    });

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();
    let status = response.status();

    let response_json: Value = response.json().await.unwrap();
    println!("API response: {response_json:#?}");

    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    let error = response_json.get("error").unwrap().as_str().unwrap();
    assert!(error.contains("Model or endpoint not found. You may be specifying the wrong one of these. Standard GCP models should use a `model_id` and not an `endpoint_id`, while fine-tuned models should use an `endpoint_id`."));
}

/// Test that thought signatures with tool calls can round-trip correctly
/// This mirrors test_gemini_multi_turn_thought_non_streaming in google_ai_studio_gemini.rs
#[tokio::test]
async fn test_gcp_vertex_multi_turn_thought_non_streaming() {
    let client = Client::new();
    let episode_id = Uuid::now_v7();
    let payload = json!({
        "function_name": "weather_helper",
        "variant_name": "gcp-vertex-gemini-pro",
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
            "_internal_provider_type": "gcp_vertex_gemini",
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
        "variant_name": "gcp-vertex-gemini-pro",
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

/// Test that when tool_choice is "auto" but allowed_tools is set,
/// the model is forced to use tools (because internally we set mode to Any).
#[tokio::test]
async fn test_gcp_vertex_gemini_tool_choice_auto_with_allowed_tools() {
    use tensorzero_core::db::clickhouse::test_helpers::{
        get_clickhouse, select_chat_inference_clickhouse,
    };

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
        "variant_name": "gcp-vertex-gemini-flash",
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
