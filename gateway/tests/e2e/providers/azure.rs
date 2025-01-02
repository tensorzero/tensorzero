use std::collections::HashMap;

use crate::providers::common::{E2ETestProvider, E2ETestProviders};

crate::generate_provider_tests!(get_providers);
crate::generate_batch_inference_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let credentials = match std::env::var("AZURE_OPENAI_API_KEY") {
        Ok(key) => HashMap::from([("azure_openai_api_key".to_string(), key)]),
        Err(_) => HashMap::new(),
    };

    let standard_providers = vec![E2ETestProvider {
        variant_name: "azure".to_string(),
        model_name: "gpt-4o-mini-azure".to_string(),
        model_provider_name: "azure".to_string(),
        credentials: HashMap::new(),
    }];

    let inference_params_providers = vec![E2ETestProvider {
        variant_name: "azure-dynamic".to_string(),
        model_name: "gpt-4o-mini-azure-dynamic".to_string(),
        model_provider_name: "azure".to_string(),
        credentials,
    }];

    let json_providers = vec![
        E2ETestProvider {
            variant_name: "azure".to_string(),
            model_name: "gpt-4o-mini-azure".to_string(),
            model_provider_name: "azure".to_string(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            variant_name: "azure-implicit".to_string(),
            model_name: "gpt-4o-mini-azure".to_string(),
            model_provider_name: "azure".to_string(),
            credentials: HashMap::new(),
        },
    ];

    E2ETestProviders {
        simple_inference: standard_providers.clone(),
        inference_params_inference: inference_params_providers,
        tool_use_inference: standard_providers.clone(),
        tool_multi_turn_inference: standard_providers.clone(),
        dynamic_tool_use_inference: standard_providers.clone(),
        parallel_tool_use_inference: vec![],
        json_mode_inference: json_providers.clone(),
        shorthand_inference: vec![],
        supports_batch_inference: false,
    }
}
