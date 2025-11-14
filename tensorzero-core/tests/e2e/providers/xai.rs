use std::collections::HashMap;

use crate::providers::common::{E2ETestProvider, E2ETestProviders, ModelTestProvider};

crate::generate_provider_tests!(get_providers);
crate::generate_batch_inference_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let credentials = match std::env::var("XAI_API_KEY") {
        Ok(key) => HashMap::from([("xai_api_key".to_string(), key)]),
        Err(_) => HashMap::new(),
    };

    let standard_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "xai".to_string(),
        model_name: "grok_2_1212".into(),
        model_provider_name: "xai".into(),
        credentials: HashMap::new(),
    }];

    let extra_body_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "xai-extra-body".to_string(),
        model_name: "grok_2_1212".into(),
        model_provider_name: "xai".into(),
        credentials: HashMap::new(),
    }];

    let bad_auth_extra_headers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "xai-extra-headers".to_string(),
        model_name: "grok_2_1212".into(),
        model_provider_name: "xai".into(),
        credentials: HashMap::new(),
    }];

    let json_providers = vec![
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "xai".to_string(),
            model_name: "grok_2_1212".into(),
            model_provider_name: "xai".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "xai-strict".to_string(),
            model_name: "grok_2_1212".into(),
            model_provider_name: "xai".into(),
            credentials: HashMap::new(),
        },
    ];

    let json_mode_off_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "xai_json_mode_off".to_string(),
        model_name: "grok_2_1212".into(),
        model_provider_name: "xai".into(),
        credentials: HashMap::new(),
    }];

    let inference_params_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "xai".to_string(),
        model_name: "grok_2_1212".into(),
        model_provider_name: "xai".into(),
        credentials,
    }];

    let shorthand_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "xai-shorthand".to_string(),
        model_name: "xai::grok-2-1212".into(),
        model_provider_name: "xai".into(),
        credentials: HashMap::new(),
    }];

    let provider_type_default_credentials_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "xai".to_string(),
        model_name: "grok-2-1212".into(),
        model_provider_name: "xai".into(),
        credentials: HashMap::new(),
    }];

    let provider_type_default_credentials_shorthand_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "xai-shorthand".to_string(),
        model_name: "xai::grok-2-1212".into(),
        model_provider_name: "xai".into(),
        credentials: HashMap::new(),
    }];

    let credential_fallbacks = vec![ModelTestProvider {
        provider_type: "xai".into(),
        model_info: HashMap::from([("model_name".to_string(), "grok-2-1212".to_string())]),
        use_modal_headers: false,
    }];

    E2ETestProviders {
        simple_inference: standard_providers.clone(),
        extra_body_inference: extra_body_providers,
        bad_auth_extra_headers,
        reasoning_inference: vec![],
        embeddings: vec![],
        inference_params_inference: inference_params_providers,
        inference_params_dynamic_credentials: vec![],
        provider_type_default_credentials: provider_type_default_credentials_providers,
        provider_type_default_credentials_shorthand:
            provider_type_default_credentials_shorthand_providers,
        tool_use_inference: standard_providers.clone(),
        tool_multi_turn_inference: standard_providers.clone(),
        dynamic_tool_use_inference: standard_providers.clone(),
        parallel_tool_use_inference: vec![],
        json_mode_inference: json_providers,
        json_mode_off_inference: json_mode_off_providers.clone(),
        image_inference: vec![],
        pdf_inference: vec![],
        input_audio: vec![],
        shorthand_inference: shorthand_providers.clone(),
        credential_fallbacks,
    }
}
