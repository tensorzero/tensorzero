use std::collections::HashMap;

use crate::providers::common::{E2ETestProvider, E2ETestProviders};

use super::common::ModelTestProvider;

crate::generate_provider_tests!(get_providers);
crate::generate_batch_inference_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let credentials = match std::env::var("DEEPSEEK_API_KEY") {
        Ok(key) => HashMap::from([("deepseek_api_key".to_string(), key)]),
        Err(_) => HashMap::new(),
    };
    let standard_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "deepseek-chat".to_string(),
        model_name: "deepseek-chat".to_string(),
        model_provider_name: "deepseek".to_string(),
        credentials: HashMap::new(),
    }];

    let extra_body_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "deepseek-chat-extra-body".to_string(),
        model_name: "deepseek-chat".to_string(),
        model_provider_name: "deepseek".to_string(),
        credentials: HashMap::new(),
    }];

    let bad_auth_extra_headers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "deepseek-chat-extra-headers".to_string(),
        model_name: "deepseek-chat".to_string(),
        model_provider_name: "deepseek".to_string(),
        credentials: HashMap::new(),
    }];

    let reasoning_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "deepseek-reasoner".to_string(),
        model_name: "deepseek-reasoner".to_string(),
        model_provider_name: "deepseek".to_string(),
        credentials: HashMap::new(),
    }];

    let inference_params_dynamic_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "deepseek-dynamic".to_string(),
        model_name: "deepseek-chat-dynamic".to_string(),
        model_provider_name: "deepseek".to_string(),
        credentials: credentials.clone(),
    }];

    let json_providers = vec![
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "deepseek-chat".to_string(),
            model_name: "deepseek-chat".to_string(),
            model_provider_name: "deepseek".to_string(),
            credentials: credentials.clone(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "deepseek-chat-strict".to_string(),
            model_name: "deepseek-chat".to_string(),
            model_provider_name: "deepseek".to_string(),
            credentials: credentials.clone(),
        },
    ];

    let json_mode_off_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "deepseek_chat_json_mode_off".to_string(),
        model_name: "deepseek-chat".to_string(),
        model_provider_name: "deepseek".to_string(),
        credentials: credentials.clone(),
    }];

    let shorthand_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "deepseek-shorthand".to_string(),
        model_name: "deepseek::deepseek-chat".to_string(),
        model_provider_name: "deepseek".to_string(),
        credentials: HashMap::new(),
    }];

    let provider_type_default_credentials_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "deepseek-chat".to_string(),
        model_name: "deepseek-chat".to_string(),
        model_provider_name: "deepseek".to_string(),
        credentials: HashMap::new(),
    }];

    let provider_type_default_credentials_shorthand_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "deepseek-shorthand".to_string(),
        model_name: "deepseek::deepseek-chat".to_string(),
        model_provider_name: "deepseek".to_string(),
        credentials: HashMap::new(),
    }];

    let credential_fallbacks = vec![ModelTestProvider {
        provider_type: "deepseek".to_string(),
        model_info: HashMap::from([("model_name".to_string(), "deepseek-chat".to_string())]),
        use_modal_headers: false,
    }];

    E2ETestProviders {
        simple_inference: standard_providers.clone(),
        extra_body_inference: extra_body_providers,
        bad_auth_extra_headers,
        reasoning_inference: reasoning_providers.clone(),
        embeddings: vec![],
        inference_params_inference: standard_providers.clone(),
        inference_params_dynamic_credentials: inference_params_dynamic_providers,
        provider_type_default_credentials: provider_type_default_credentials_providers,
        provider_type_default_credentials_shorthand:
            provider_type_default_credentials_shorthand_providers,
        tool_use_inference: vec![],
        tool_multi_turn_inference: vec![],
        dynamic_tool_use_inference: vec![],
        parallel_tool_use_inference: vec![],
        json_mode_inference: json_providers.clone(),
        json_mode_off_inference: json_mode_off_providers.clone(),
        image_inference: vec![],
        pdf_inference: vec![],
        input_audio: vec![],
        shorthand_inference: shorthand_providers.clone(),
        credential_fallbacks,
    }
}
