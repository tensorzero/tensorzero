use std::collections::HashMap;

use crate::providers::common::{E2ETestProvider, E2ETestProviders};

crate::generate_provider_tests!(get_providers);
crate::generate_batch_inference_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let credentials = match std::env::var("XAI_API_KEY") {
        Ok(key) => HashMap::from([("xai_api_key".to_string(), key)]),
        Err(_) => HashMap::new(),
    };

    let standard_providers = vec![E2ETestProvider {
        variant_name: "xai".to_string(),
        model_name: "grok_2_1212".into(),
        model_provider_name: "xai".into(),
        credentials: HashMap::new(),
    }];

    let extra_body_providers = vec![E2ETestProvider {
        variant_name: "xai-extra-body".to_string(),
        model_name: "grok_2_1212".into(),
        model_provider_name: "xai".into(),
        credentials: HashMap::new(),
    }];

    let bad_auth_extra_headers = vec![E2ETestProvider {
        variant_name: "xai-extra-headers".to_string(),
        model_name: "grok_2_1212".into(),
        model_provider_name: "xai".into(),
        credentials: HashMap::new(),
    }];

    let json_providers = vec![
        E2ETestProvider {
            variant_name: "xai".to_string(),
            model_name: "grok_2_1212".into(),
            model_provider_name: "xai".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            variant_name: "xai-default".to_string(),
            model_name: "grok_2_1212".into(),
            model_provider_name: "xai".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            variant_name: "xai-strict".to_string(),
            model_name: "grok_2_1212".into(),
            model_provider_name: "xai".into(),
            credentials: HashMap::new(),
        },
    ];
    let inference_params_providers = vec![E2ETestProvider {
        variant_name: "xai".to_string(),
        model_name: "grok_2_1212".into(),
        model_provider_name: "xai".into(),
        credentials,
    }];

    let shorthand_providers = vec![E2ETestProvider {
        variant_name: "xai-shorthand".to_string(),
        model_name: "xai::grok-2-1212".into(),
        model_provider_name: "xai".into(),
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
        image_inference: vec![],

        shorthand_inference: shorthand_providers.clone(),
        supports_batch_inference: false,
    }
}
