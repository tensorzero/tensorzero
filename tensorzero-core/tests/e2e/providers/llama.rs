#![allow(clippy::print_stdout)]
use std::collections::HashMap;

use crate::providers::common::{E2ETestProvider, E2ETestProviders};

crate::generate_provider_tests!(get_providers);
crate::generate_batch_inference_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let credentials = match std::env::var("LLAMA_API_KEY") {
        Ok(key) => HashMap::from([("llama_api_key".to_string(), key)]),
        Err(_) => HashMap::new(),
    };

    let standard_providers = vec![E2ETestProvider {
        supports_batch_inference: true,
        variant_name: "llama".to_string(),
        model_name: "llama-scout-llama".into(),
        model_provider_name: "llama".into(),
        credentials: HashMap::new(),
    }];

    let extra_body_providers = vec![E2ETestProvider {
        supports_batch_inference: true,
        variant_name: "llama-extra-body".to_string(),
        model_name: "llama-scout-llama".into(),
        model_provider_name: "llama".into(),
        credentials: HashMap::new(),
    }];

    let bad_auth_extra_headers = vec![E2ETestProvider {
        supports_batch_inference: true,
        variant_name: "llama-extra-headers".to_string(),
        model_name: "llama-scout-llama".into(),
        model_provider_name: "llama".into(),
        credentials: HashMap::new(),
    }];

    let inference_params_providers = vec![E2ETestProvider {
        supports_batch_inference: true,
        variant_name: "llama".to_string(),
        model_name: "llama-scout-llama".into(),
        model_provider_name: "llama".into(),
        credentials: credentials.clone(),
    }];

    let inference_params_dynamic_providers = vec![E2ETestProvider {
        supports_batch_inference: true,
        variant_name: "llama-dynamic".to_string(),
        model_name: "llama-scout-llama-dynamic".into(),
        model_provider_name: "llama".into(),
        credentials: credentials.clone(),
    }];

    let shorthand_providers = vec![E2ETestProvider {
        supports_batch_inference: true,
        variant_name: "llama-shorthand".to_string(),
        model_name: "llama-scout-llama".into(),
        model_provider_name: "llama".into(),
        credentials: HashMap::new(),
    }];

    let json_providers = vec![
        E2ETestProvider {
            supports_batch_inference: true,
            variant_name: "llama".to_string(),
            model_name: "llama-scout-llama".into(),
            model_provider_name: "llama".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: true,
            variant_name: "llama-strict".to_string(),
            model_name: "llama-scout-llama".into(),
            model_provider_name: "llama".into(),
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
        dynamic_tool_use_inference: vec![E2ETestProvider {
            supports_batch_inference: true,
            variant_name: "llama".to_string(),
            model_name: "llama-scout-llama".into(),
            model_provider_name: "llama".into(),
            credentials: credentials.clone(),
        }],
        parallel_tool_use_inference: standard_providers.clone(),
        json_mode_inference: json_providers.clone(),
        image_inference: vec![],
        pdf_inference: vec![],
        shorthand_inference: shorthand_providers,
        json_mode_off_inference: vec![],
        embeddings: vec![],
    }
}
