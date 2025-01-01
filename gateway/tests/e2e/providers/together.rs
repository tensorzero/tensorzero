use std::collections::HashMap;

use crate::providers::common::{E2ETestProvider, E2ETestProviders};

crate::generate_provider_tests!(get_providers);
crate::generate_batch_inference_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let mut map = HashMap::new();
    if let Ok(api_key) = std::env::var("TOGETHER_API_KEY") {
        map.insert("TOGETHER_API_KEY".to_string(), api_key);
    }
    let credentials = if map.is_empty() { None } else { Some(map) };

    let standard_providers = vec![E2ETestProvider {
        variant_name: "together".to_string(),
        model_name: "llama3.1-8b-instruct-together".to_string(),
        model_provider_name: "together".to_string(),
        credentials: None,
    }];

    let inference_params_providers = vec![E2ETestProvider {
        variant_name: "together".to_string(),
        model_name: "llama3.1-8b-instruct-together".to_string(),
        model_provider_name: "together".to_string(),
        credentials,
    }];

    let json_providers = vec![
        E2ETestProvider {
            variant_name: "together".to_string(),
            model_name: "llama3.1-8b-instruct-together".to_string(),
            model_provider_name: "together".to_string(),
            credentials: None,
        },
        // TODOs (#80): see below
        // E2ETestProvider {
        //     variant_name: "together-implicit".to_string(),
        // },
    ];

    let shorthand_providers = vec![E2ETestProvider {
        variant_name: "together-shorthand".to_string(),
        model_name: "together::meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo".to_string(),
        model_provider_name: "together".to_string(),
        credentials: None,
    }];

    // TODOs (#80):
    // - Together seems to have a different format for tool use responses compared to OpenAI (breaking)
    // - Together's function calling for Llama 3.1 is different from Llama 3.0 (breaking) - we should test both
    E2ETestProviders {
        simple_inference: standard_providers.clone(),
        inference_params_inference: inference_params_providers,
        tool_use_inference: vec![],
        tool_multi_turn_inference: vec![],
        dynamic_tool_use_inference: vec![],
        parallel_tool_use_inference: vec![],
        json_mode_inference: json_providers.clone(),
        shorthand_inference: shorthand_providers.clone(),
        supports_batch_inference: false,
    }
}
