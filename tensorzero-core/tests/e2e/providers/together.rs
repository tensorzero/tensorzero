use std::collections::HashMap;

use crate::providers::common::{E2ETestProvider, E2ETestProviders, ModelTestProvider};

crate::generate_provider_tests!(get_providers);
crate::generate_batch_inference_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let credentials = match std::env::var("TOGETHER_API_KEY") {
        Ok(key) => HashMap::from([("together_api_key".to_string(), key)]),
        Err(_) => HashMap::new(),
    };

    let standard_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "together".to_string(),
        model_name: "llama3.1-8b-instruct-together".into(),
        model_provider_name: "together".into(),
        credentials: HashMap::new(),
    }];

    let extra_body_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "together-extra-body".to_string(),
        model_name: "llama3.1-8b-instruct-together".into(),
        model_provider_name: "together".into(),
        credentials: HashMap::new(),
    }];

    let bad_auth_extra_headers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "together-extra-headers".to_string(),
        model_name: "llama3.1-8b-instruct-together".into(),
        model_provider_name: "together".into(),
        credentials: HashMap::new(),
    }];

    let inference_params_dynamic_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "together-dynamic".to_string(),
        model_name: "llama3.1-8b-instruct-together-dynamic".into(),
        model_provider_name: "together".into(),
        credentials,
    }];

    let json_providers = vec![
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "together".to_string(),
            model_name: "llama3.1-8b-instruct-together".into(),
            model_provider_name: "together".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "together-strict".to_string(),
            model_name: "llama3.1-8b-instruct-together".into(),
            model_provider_name: "together".into(),
            credentials: HashMap::new(),
        },
    ];

    let json_mode_off_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "together_json_mode_off".to_string(),
        model_name: "llama3.1-8b-instruct-together".into(),
        model_provider_name: "together".into(),
        credentials: HashMap::new(),
    }];

    let tool_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "together-tool".to_string(),
        model_name: "llama3.1-405b-instruct-turbo-together".into(),
        model_provider_name: "together".into(),
        credentials: HashMap::new(),
    }];

    let reasoning_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "together-deepseek-r1".to_string(),
        model_name: "together-deepseek-r1".to_string(),
        model_provider_name: "together".to_string(),
        credentials: HashMap::new(),
    }];

    let shorthand_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "together-shorthand".to_string(),
        model_name: "together::meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo".into(),
        model_provider_name: "together".into(),
        credentials: HashMap::new(),
    }];

    let provider_type_default_credentials_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "together".to_string(),
        model_name: "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo".into(),
        model_provider_name: "together".into(),
        credentials: HashMap::new(),
    }];

    let provider_type_default_credentials_shorthand_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "together-shorthand".to_string(),
        model_name: "together::meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo".into(),
        model_provider_name: "together".into(),
        credentials: HashMap::new(),
    }];

    let credential_fallbacks = vec![ModelTestProvider {
        provider_type: "together".into(),
        model_info: HashMap::from([(
            "model_name".to_string(),
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo".to_string(),
        )]),
        use_modal_headers: false,
    }];

    E2ETestProviders {
        simple_inference: standard_providers.clone(),
        extra_body_inference: extra_body_providers,
        bad_auth_extra_headers,
        reasoning_inference: reasoning_providers.clone(),
        embeddings: vec![],
        inference_params_inference: standard_providers,
        inference_params_dynamic_credentials: inference_params_dynamic_providers,
        provider_type_default_credentials: provider_type_default_credentials_providers,
        provider_type_default_credentials_shorthand:
            provider_type_default_credentials_shorthand_providers,
        tool_use_inference: tool_providers.clone(),
        tool_multi_turn_inference: tool_providers.clone(),
        dynamic_tool_use_inference: tool_providers.clone(),
        parallel_tool_use_inference: tool_providers.clone(),
        json_mode_inference: json_providers.clone(),
        json_mode_off_inference: json_mode_off_providers.clone(),
        image_inference: vec![],
        pdf_inference: vec![],
        input_audio: vec![],
        shorthand_inference: shorthand_providers.clone(),
        credential_fallbacks,
    }
}
