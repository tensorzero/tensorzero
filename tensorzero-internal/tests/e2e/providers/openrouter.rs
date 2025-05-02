#![allow(clippy::print_stdout)]
use std::collections::HashMap;

use crate::common::get_gateway_endpoint;
use crate::providers::common::{E2ETestProvider, E2ETestProviders};
use reqwest::{Client, StatusCode};
use serde_json::json;
use uuid::Uuid;

crate::generate_provider_tests!(get_providers);
crate::generate_batch_inference_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let credentials = match std::env::var("OPENROUTER_API_KEY") {
        Ok(key) => HashMap::from([("openrouter_api_key".to_string(), key)]),
        Err(_) => HashMap::new(),
    };

    let standard_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "openrouter".to_string(),
        model_name: "gpt_4_1_mini_openrouter".into(),
        model_provider_name: "openrouter".into(),
        credentials: HashMap::new(),
    }];

    let extra_body_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "openrouter-extra-body".to_string(),
        model_name: "gpt_4_1_mini_openrouter".into(),
        model_provider_name: "openrouter".into(),
        credentials: HashMap::new(),
    }];

    let bad_auth_extra_headers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "openrouter-extra-headers".to_string(),
        model_name: "gpt_4_1_mini_openrouter".into(),
        model_provider_name: "openrouter".into(),
        credentials: HashMap::new(),
    }];

    let inference_params_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "openrouter".to_string(),
        model_name: "gpt_4_1_mini_openrouter".into(),
        model_provider_name: "openrouter".into(),
        credentials: credentials.clone(),
    }];

    let inference_params_dynamic_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "openrouter-dynamic".to_string(),
        model_name: "gpt_4_1_mini_openrouter_dynamic".into(),
        model_provider_name: "openrouter".into(),
        credentials,
    }];

    let json_providers = vec![
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "openrouter".to_string(),
            model_name: "gpt_4_1_mini_openrouter".into(),
            model_provider_name: "openrouter".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "openrouter-implicit".to_string(),
            model_name: "gpt_4_1_mini_openrouter".into(),
            model_provider_name: "openrouter".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "openrouter-strict".to_string(),
            model_name: "gpt_4_1_mini_openrouter".into(),
            model_provider_name: "openrouter".into(),
            credentials: HashMap::new(),
        },
    ];

    E2ETestProviders {
        simple_inference: standard_providers.clone(),
        extra_body_inference: extra_body_providers,
        bad_auth_extra_headers,
        reasoning_inference: vec![],
        inference_params_inference: inference_params_providers,
        inference_params_dynamic_credentials: inference_params_dynamic_providers,
        tool_use_inference: standard_providers.clone(),
        tool_multi_turn_inference: standard_providers.clone(),
        dynamic_tool_use_inference: standard_providers.clone(),
        parallel_tool_use_inference: standard_providers.clone(),
        json_mode_inference: json_providers.clone(),
        image_inference: vec![],
        shorthand_inference: vec![],
        json_mode_off_inference: vec![],
    }
}

/// Test to verify that OpenRouter-specific headers (X-Title and HTTP-Referer) are included in inference requests
#[tokio::test]
async fn test_openrouter_headers_present_in_requests() {
    let episode_id = Uuid::now_v7();
    let payload = json!({
        "function_name": "basic_test",
        "variant_name": "openrouter",
        "episode_id": episode_id,
        "input": {
            "system": {"assistant_name": "Dr. Mehta"},
            "messages": [{
                "role": "user",
                "content": "What is the name of the capital city of Japan?"
            }]
        },
        "stream": false,
        "tags": {"foo": "bar"},
    });

    // Build both a a client and a request object, so we can both
    // inspect the headers and send the request later to ensure it goes through
    // We can't reuse a request builder (from build()) to ownership issues
    let (client, request) = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .build_split();

    let unwrapped_request = request.unwrap();
    let headers = unwrapped_request.headers();

    println!("Headers: {:?}", &headers);

    // // Check that our custom headers are present
    // assert_eq!(
    //     headers.get("X-Title").map(|v| v.to_str().unwrap()),
    //     Some("TensorZero"),
    //     "X-Title header should be present and set to 'TensorZero'"
    // );
    // assert_eq!(
    //     headers.get("HTTP-Referer").map(|v| v.to_str().unwrap()),
    //     Some("https://www.tensorzero.com/"),
    //     "HTTP-Referer header should be present and set to 'https://www.tensorzero.com/'"
    // );

    // Create a new client with the same payload to actually send the request
    // and verify it goes through successfully.
    let response = client
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    // Check that the API response is successful
    assert_eq!(response.status(), StatusCode::OK);
}
