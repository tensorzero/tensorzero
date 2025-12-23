use std::collections::HashMap;

use crate::common::get_gateway_endpoint;
use crate::providers::common::{
    E2ETestProvider, E2ETestProviders, EmbeddingTestProvider, ModelTestProvider,
};
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
        model_name: "gpt_4_o_mini_openrouter".into(),
        model_provider_name: "openrouter".into(),
        credentials: HashMap::new(),
    }];

    let extra_body_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "openrouter-extra-body".to_string(),
        model_name: "gpt_4_o_mini_openrouter".into(),
        model_provider_name: "openrouter".into(),
        credentials: HashMap::new(),
    }];

    let bad_auth_extra_headers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "openrouter-extra-headers".to_string(),
        model_name: "gpt_4_o_mini_openrouter".into(),
        model_provider_name: "openrouter".into(),
        credentials: HashMap::new(),
    }];

    let inference_params_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "openrouter".to_string(),
        model_name: "gpt_4_o_mini_openrouter".into(),
        model_provider_name: "openrouter".into(),
        credentials: credentials.clone(),
    }];

    let inference_params_dynamic_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "openrouter-dynamic".to_string(),
        model_name: "gpt_4_o_mini_openrouter_dynamic".into(),
        model_provider_name: "openrouter".into(),
        credentials,
    }];

    let json_providers = vec![
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "openrouter".to_string(),
            model_name: "gpt_4_o_mini_openrouter".into(),
            model_provider_name: "openrouter".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "openrouter-implicit".to_string(),
            model_name: "gpt_4_o_mini_openrouter".into(),
            model_provider_name: "openrouter".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "openrouter-strict".to_string(),
            model_name: "gpt_4_o_mini_openrouter".into(),
            model_provider_name: "openrouter".into(),
            credentials: HashMap::new(),
        },
    ];

    let shorthand_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "openrouter-shorthand".to_string(),
        model_name: "openrouter::openai/gpt-4o-mini".to_string(),
        model_provider_name: "openrouter".into(),
        credentials: HashMap::new(),
    }];

    let provider_type_default_credentials_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "openrouter".to_string(),
        model_name: "openai/gpt-4.1-mini".into(),
        model_provider_name: "openrouter".into(),
        credentials: HashMap::new(),
    }];

    let provider_type_default_credentials_shorthand_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "openrouter-shorthand".to_string(),
        model_name: "openrouter::openai/gpt-4.1-mini".into(),
        model_provider_name: "openrouter".into(),
        credentials: HashMap::new(),
    }];

    let input_audio_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "openrouter".to_string(),
        model_name: "openai/gpt-4o-audio-preview".into(),
        model_provider_name: "openrouter".into(),
        credentials: HashMap::new(),
    }];

    let credential_fallbacks = vec![ModelTestProvider {
        provider_type: "openrouter".to_string(),
        model_info: HashMap::from([("model_name".to_string(), "openai/gpt-4o-mini".to_string())]),
        use_modal_headers: false,
    }];

    let embedding_providers = vec![EmbeddingTestProvider {
        model_name: "gemini_embedding_001_openrouter".to_string(),
        dimensions: 3072,
    }];

    E2ETestProviders {
        simple_inference: standard_providers.clone(),
        extra_body_inference: extra_body_providers,
        bad_auth_extra_headers,
        reasoning_inference: vec![],
        embeddings: embedding_providers,
        inference_params_inference: inference_params_providers,
        inference_params_dynamic_credentials: inference_params_dynamic_providers,
        provider_type_default_credentials: provider_type_default_credentials_providers,
        provider_type_default_credentials_shorthand:
            provider_type_default_credentials_shorthand_providers,
        tool_use_inference: standard_providers.clone(),
        tool_multi_turn_inference: standard_providers.clone(),
        dynamic_tool_use_inference: standard_providers.clone(),
        parallel_tool_use_inference: standard_providers.clone(),
        json_mode_inference: json_providers.clone(),
        image_inference: vec![],
        pdf_inference: vec![],
        input_audio: input_audio_providers,
        shorthand_inference: shorthand_providers,
        json_mode_off_inference: vec![],
        credential_fallbacks,
    }
}

/// Test to verify that OpenRouter-specific headers (X-Title and HTTP-Referer) are included in inference requests
/// On the CI the provider-proxy will check for the presence and correct values for these headers, and return
/// a 400 when they're not there.
///
/// This test will only fail when:
/// - TENSORZERO_E2E_PROXY is set and the proxy is running
/// - X-Title and HTTP-Referer headers are missing or have the wrong values (TensorZero and https://www.tensorzero.com/, respectively)
///   in the gateway request to the OpenRouter API.
///
/// Note: other E2E tests will also break if the headers are missing or incorrect when running the proxy, so this test is really here for
/// explicitness / documentation.
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

    let response = Client::new()
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}
