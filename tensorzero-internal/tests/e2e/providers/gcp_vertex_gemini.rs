#![allow(clippy::print_stdout)]

use std::collections::HashMap;

use http::StatusCode;
use reqwest::Client;
use serde_json::{json, Value};
use uuid::Uuid;

use crate::{
    common::get_gateway_endpoint,
    providers::common::{E2ETestProvider, E2ETestProviders},
};

crate::generate_provider_tests!(get_providers);
crate::generate_batch_inference_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
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
            model_name: "gemini-2.5-pro-preview-05-06".into(),
            model_provider_name: "gcp_vertex_gemini".into(),
            credentials: HashMap::new(),
        },
    ];

    let image_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "gcp_vertex".to_string(),
        model_name: "gemini-2.5-pro-preview-05-06".into(),
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
            model_name: "gemini-2.5-pro-preview-05-06".into(),
            model_provider_name: "gcp_vertex_gemini".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "gcp-vertex-gemini-pro-implicit".to_string(),
            model_name: "gemini-2.5-pro-preview-05-06".into(),
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
    }, E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "gcp_vertex_gemini_shorthand_endpoint".to_string(),
        model_name: "gcp_vertex_gemini::projects/tensorzero-public/locations/us-central1/endpoints/945488740422254592".into(),
        model_provider_name: "gcp_vertex_gemini".into(),
        credentials: HashMap::new(),
    }];

    E2ETestProviders {
        simple_inference: standard_providers.clone(),
        extra_body_inference: extra_body_providers,
        bad_auth_extra_headers,
        reasoning_inference: vec![],
        inference_params_inference: standard_providers.clone(),
        inference_params_dynamic_credentials: vec![],
        tool_use_inference: standard_providers.clone(),
        tool_multi_turn_inference: standard_providers.clone(),
        dynamic_tool_use_inference: standard_providers.clone(),
        parallel_tool_use_inference: vec![],
        json_mode_inference: json_providers.clone(),
        json_mode_off_inference: json_mode_off_providers.clone(),
        image_inference: image_providers.clone(),
        pdf_inference: pdf_providers,
        shorthand_inference: shorthand_providers.clone(),
    }
}

// Specifying `tool_choice: none` causes Gemini 2.5 Pro to produce an 'executableCode' block.
// We test that we properly construct an 'unknown' content block in this case.
#[tokio::test]
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
    assert_eq!(unknown_block.get("model_provider_name").unwrap().as_str(), Some("tensorzero::model_name::gemini-2.5-pro-preview-05-06::provider_name::gcp_vertex_gemini"));
    assert!(unknown_block
        .get("data")
        .unwrap()
        .as_object()
        .unwrap()
        .get("executableCode")
        .is_some());
}
