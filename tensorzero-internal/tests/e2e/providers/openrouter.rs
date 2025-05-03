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
