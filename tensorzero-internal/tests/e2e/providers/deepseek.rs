use std::collections::HashMap;

use crate::providers::common::{E2ETestProvider, E2ETestProviders};

#[cfg(feature = "e2e_tests")]
crate::generate_provider_tests!(get_providers);
#[cfg(feature = "batch_tests")]
crate::generate_batch_inference_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let credentials = match std::env::var("DEEPSEEK_API_KEY") {
        Ok(key) => HashMap::from([("deepseek_api_key".to_string(), key)]),
        Err(_) => HashMap::new(),
    };
    let standard_providers = vec![E2ETestProvider {
        variant_name: "deepseek-chat".to_string(),
        model_name: "deepseek-chat".to_string(),
        model_provider_name: "deepseek".to_string(),
        credentials: HashMap::new(),
    }];

    let extra_body_providers = vec![E2ETestProvider {
        variant_name: "deepseek-chat-extra-body".to_string(),
        model_name: "deepseek-chat".to_string(),
        model_provider_name: "deepseek".to_string(),
        credentials: HashMap::new(),
    }];

    let reasoning_providers = vec![E2ETestProvider {
        variant_name: "deepseek-reasoner".to_string(),
        model_name: "deepseek-reasoner".to_string(),
        model_provider_name: "deepseek".to_string(),
        credentials: HashMap::new(),
    }];

    let inference_params_dynamic_providers = vec![E2ETestProvider {
        variant_name: "deepseek-dynamic".to_string(),
        model_name: "deepseek-chat-dynamic".to_string(),
        model_provider_name: "deepseek".to_string(),
        credentials: credentials.clone(),
    }];

    let json_providers = vec![
        E2ETestProvider {
            variant_name: "deepseek-chat".to_string(),
            model_name: "deepseek-chat".to_string(),
            model_provider_name: "deepseek".to_string(),
            credentials: credentials.clone(),
        },
        E2ETestProvider {
            variant_name: "deepseek-chat-default".to_string(),
            model_name: "deepseek-chat".to_string(),
            model_provider_name: "deepseek".to_string(),
            credentials: credentials.clone(),
        },
    ];
    #[cfg(feature = "e2e_tests")]
    let shorthand_providers = vec![E2ETestProvider {
        variant_name: "deepseek-shorthand".to_string(),
        model_name: "deepseek::deepseek-chat".to_string(),
        model_provider_name: "deepseek".to_string(),
        credentials: HashMap::new(),
    }];

    E2ETestProviders {
        simple_inference: standard_providers.clone(),
        extra_body_inference: extra_body_providers,
        reasoning_inference: reasoning_providers.clone(),
        inference_params_inference: standard_providers.clone(),
        inference_params_dynamic_credentials: inference_params_dynamic_providers,
        tool_use_inference: vec![],
        tool_multi_turn_inference: vec![],
        dynamic_tool_use_inference: vec![],
        parallel_tool_use_inference: vec![],
        json_mode_inference: json_providers.clone(),
        image_inference: vec![],
        #[cfg(feature = "e2e_tests")]
        shorthand_inference: shorthand_providers.clone(),
        #[cfg(feature = "batch_tests")]
        supports_batch_inference: false,
    }
}
