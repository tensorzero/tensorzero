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
        variant_name: "llama_api".to_string(),
        model_name: "llama-scout-llama".into(),
        model_provider_name: "llama_api".into(),
            credentials: HashMap::from([
                ("temperature".to_string(), "1.0".to_string()),
                ("top_p".to_string(), "0.95".to_string()),
            ]),
    }];

    let extra_body_providers = vec![E2ETestProvider {
        supports_batch_inference: true,
        variant_name: "llama_api-extra-body".to_string(),
        model_name: "llama-scout-llama".into(),
        model_provider_name: "llama_api".into(),
            credentials: HashMap::from([
                ("temperature".to_string(), "1.0".to_string()),
                ("top_p".to_string(), "0.95".to_string()),
            ]),
    }];

    let bad_auth_extra_headers = vec![E2ETestProvider {
        supports_batch_inference: true,
        variant_name: "llama_api-extra-headers".to_string(),
        model_name: "llama-scout-llama".into(),
        model_provider_name: "llama_api".into(),
            credentials: HashMap::from([
                ("temperature".to_string(), "1.0".to_string()),
                ("top_p".to_string(), "0.95".to_string()),
            ]),
    }];

    let inference_params_providers = vec![E2ETestProvider {
        supports_batch_inference: true,
        variant_name: "llama_api".to_string(),
        model_name: "llama-scout-llama".into(),
        model_provider_name: "llama_api".into(),
            credentials: {
                let mut creds = credentials.clone();
                creds.insert("temperature".to_string(), "1.0".to_string());
                creds.insert("top_p".to_string(), "0.95".to_string());
                creds
            },
    }];

    let inference_params_dynamic_providers = vec![E2ETestProvider {
        supports_batch_inference: true,
        variant_name: "llama_api-dynamic".to_string(),
        model_name: "llama-scout-llama-dynamic".into(),
        model_provider_name: "llama_api".into(),
            credentials: {
                let mut creds = credentials.clone();
                creds.insert("temperature".to_string(), "1.0".to_string());
                creds.insert("top_p".to_string(), "0.95".to_string());
                creds
            },
    }];

    let shorthand_providers = vec![E2ETestProvider {
        supports_batch_inference: true,
        variant_name: "llama_api-shorthand".to_string(),
        model_name: "llama-scout-llama".into(),
        model_provider_name: "llama_api".into(),
            credentials: HashMap::from([
                ("temperature".to_string(), "1.0".to_string()),
                ("top_p".to_string(), "0.95".to_string()),
            ]),
    }];

    let json_providers = vec![
        E2ETestProvider {
            supports_batch_inference: true,
            variant_name: "llama_api".to_string(),
            model_name: "llama-scout-llama".into(),
            model_provider_name: "llama_api".into(),
                credentials: HashMap::from([
                    ("temperature".to_string(), "1.0".to_string()),
                    ("top_p".to_string(), "0.95".to_string()),
                ]),
        },
        E2ETestProvider {
            supports_batch_inference: true,
            variant_name: "llama_api-strict".to_string(),
            model_name: "llama-scout-llama".into(),
            model_provider_name: "llama_api".into(),
                credentials: HashMap::from([
                    ("temperature".to_string(), "1.0".to_string()),
                    ("top_p".to_string(), "0.95".to_string()),
                ]),
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
            variant_name: "llama_api".to_string(),
            model_name: "llama-scout-llama".into(),
            model_provider_name: "llama_api".into(),
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
