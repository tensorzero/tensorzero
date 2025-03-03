use std::collections::HashMap;

use crate::providers::common::{E2ETestProvider, E2ETestProviders};

#[cfg(feature = "e2e_tests")]
crate::generate_provider_tests!(get_providers);
#[cfg(feature = "batch_tests")]
crate::generate_batch_inference_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let credentials = match std::env::var("TOGETHER_API_KEY") {
        Ok(key) => HashMap::from([("together_api_key".to_string(), key)]),
        Err(_) => HashMap::new(),
    };

    let standard_providers = vec![E2ETestProvider {
        variant_name: "together".to_string(),
        model_name: "llama3.1-8b-instruct-together".into(),
        model_provider_name: "together".into(),
        credentials: HashMap::new(),
    }];

    let extra_body_providers = vec![E2ETestProvider {
        variant_name: "together-extra-body".to_string(),
        model_name: "llama3.1-8b-instruct-together".into(),
        model_provider_name: "together".into(),
        credentials: HashMap::new(),
    }];

    let inference_params_providers = vec![E2ETestProvider {
        variant_name: "together-dynamic".to_string(),
        model_name: "llama3.1-8b-instruct-together-dynamic".into(),
        model_provider_name: "together".into(),
        credentials,
    }];

    let json_providers = vec![
        E2ETestProvider {
            variant_name: "together".to_string(),
            model_name: "llama3.1-8b-instruct-together".into(),
            model_provider_name: "together".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            variant_name: "together-default".to_string(),
            model_name: "llama3.1-8b-instruct-together".into(),
            model_provider_name: "together".into(),
            credentials: HashMap::new(),
        },
    ];

    let tool_providers = vec![E2ETestProvider {
        variant_name: "together-tool".to_string(),
        model_name: "llama3.1-405b-instruct-turbo-together".into(),
        model_provider_name: "together".into(),
        credentials: HashMap::new(),
    }];

    let reasoning_providers = vec![E2ETestProvider {
        variant_name: "together-deepseek-r1".to_string(),
        model_name: "together-deepseek-r1".to_string(),
        model_provider_name: "together".to_string(),
        credentials: HashMap::new(),
    }];

    #[cfg(feature = "e2e_tests")]
    let shorthand_providers = vec![E2ETestProvider {
        variant_name: "together-shorthand".to_string(),
        model_name: "together::meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo".into(),
        model_provider_name: "together".into(),
        credentials: HashMap::new(),
    }];

    E2ETestProviders {
        simple_inference: standard_providers.clone(),
        extra_body_inference: extra_body_providers,
        reasoning_inference: reasoning_providers.clone(),
        inference_params_inference: inference_params_providers,
        tool_use_inference: tool_providers.clone(),
        tool_multi_turn_inference: tool_providers.clone(),
        dynamic_tool_use_inference: tool_providers.clone(),
        parallel_tool_use_inference: tool_providers.clone(),
        json_mode_inference: json_providers.clone(),
        image_inference: vec![],
        #[cfg(feature = "e2e_tests")]
        shorthand_inference: shorthand_providers.clone(),
        #[cfg(feature = "batch_tests")]
        supports_batch_inference: false,
    }
}
