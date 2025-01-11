use std::collections::HashMap;

use crate::providers::common::{E2ETestProvider, E2ETestProviders};

#[cfg(feature = "e2e_tests")]
crate::generate_provider_tests!(get_providers);
#[cfg(feature = "batch_tests")]
crate::generate_batch_inference_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let credentials = match std::env::var("ANTHROPIC_API_KEY") {
        Ok(key) => HashMap::from([("anthropic_api_key".to_string(), key)]),
        Err(_) => HashMap::new(),
    };

    let standard_providers = vec![E2ETestProvider {
        variant_name: "anthropic".to_string(),
        model_name: "claude-3-haiku-20240307-anthropic".to_string(),
        model_provider_name: "anthropic".to_string(),
        credentials: HashMap::new(),
    }];

    let inference_params_providers = vec![E2ETestProvider {
        variant_name: "anthropic-dynamic".to_string(),
        model_name: "claude-3-haiku-20240307-anthropic-dynamic".to_string(),
        model_provider_name: "anthropic".to_string(),
        credentials,
    }];

    let json_providers = vec![
        E2ETestProvider {
            variant_name: "anthropic".to_string(),
            model_name: "claude-3-haiku-20240307-anthropic".to_string(),
            model_provider_name: "anthropic".to_string(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            variant_name: "anthropic-implicit".to_string(),
            model_name: "claude-3-haiku-20240307-anthropic".to_string(),
            model_provider_name: "anthropic".to_string(),
            credentials: HashMap::new(),
        },
    ];

    #[cfg(feature = "e2e_tests")]
    let shorthand_providers = vec![E2ETestProvider {
        variant_name: "anthropic-shorthand".to_string(),
        model_name: "anthropic::claude-3-haiku-20240307".to_string(),
        model_provider_name: "anthropic".to_string(),
        credentials: HashMap::new(),
    }];

    E2ETestProviders {
        simple_inference: standard_providers.clone(),
        inference_params_inference: inference_params_providers,
        tool_use_inference: standard_providers.clone(),
        tool_multi_turn_inference: standard_providers.clone(),
        dynamic_tool_use_inference: standard_providers.clone(),
        parallel_tool_use_inference: standard_providers.clone(),
        json_mode_inference: json_providers.clone(),
        #[cfg(feature = "e2e_tests")]
        shorthand_inference: shorthand_providers.clone(),
        #[cfg(feature = "batch_tests")]
        supports_batch_inference: false,
    }
}
