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
        unknown_block.get("model_provider_name").unwrap().as_str(),
        Some("tensorzero::model_name::gcp-gemini-2.5-pro::provider_name::gcp_vertex_gemini")
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
