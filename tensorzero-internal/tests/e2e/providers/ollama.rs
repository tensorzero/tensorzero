use std::collections::HashMap;

use crate::providers::common::{E2ETestProvider, E2ETestProviders};

crate::generate_provider_tests!(get_providers);
crate::generate_batch_inference_tests!(get_providers);

async fn get_providers()-> E2ETestProviders {
    let standard_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "ollama".to_string(),
        model_name: "llama3.2".into(),
        model_provider_name: "ollama".into(),
        credentials: HashMap::new()
    }];

    let extra_body_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "ollama-extra-body".to_string(),
        model_name: "llama3.2".into(),
        model_provider_name: "ollama".into(),
        credentials: HashMap::new(),
    }];


    let bad_auth_extra_headers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "ollama-extra-headers".to_string(),
        model_name: "llama3.2".into(),
        model_provider_name: "ollama".into(),
        credentials: HashMap::new(),
    }];


    let json_providers = vec![
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "ollama".to_string(),
            model_name: "llama3.2".into(),
            model_provider_name: "ollama".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "ollama-strict".to_string(),
            model_name: "llama3.2".into(),
            model_provider_name: "ollama".into(),
            credentials: HashMap::new(),
        },
    ];


    let json_mode_off_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "ollama_json_mode_off".to_string(),
        model_name: "llama3.2".into(),
        model_provider_name: "ollama".into(),
        credentials: HashMap::new(),
    }];

    let inference_params_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "ollama".to_string(),
        model_name: "llama3.2".into(),
        model_provider_name: "ollama".into(),
        credentials: HashMap::new(),
    }];

    let shorthand_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "ollama-shorthand".to_string(),
        model_name: "ollama::llama3.2".into(),
        model_provider_name: "ollama".into(),
        credentials: HashMap::new(),
    }];


    E2ETestProviders {
        simple_inference: standard_providers.clone(),
        extra_body_inference: extra_body_providers,
        bad_auth_extra_headers,
        reasoning_inference: vec![],
        inference_params_inference: inference_params_providers,
        inference_params_dynamic_credentials: vec![],
        tool_use_inference: standard_providers.clone(),
        tool_multi_turn_inference: standard_providers.clone(),
        dynamic_tool_use_inference: standard_providers.clone(),
        parallel_tool_use_inference: vec![],
        json_mode_inference: json_providers,
        json_mode_off_inference: json_mode_off_providers.clone(),
        image_inference: vec![],

        shorthand_inference: shorthand_providers.clone(),
    }
}
