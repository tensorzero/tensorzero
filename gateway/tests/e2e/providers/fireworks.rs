use std::collections::HashMap;

use crate::providers::common::{E2ETestProvider, E2ETestProviders};

crate::generate_provider_tests!(get_providers);
crate::generate_batch_inference_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let mut map = HashMap::new();

    if let Ok(api_key) = std::env::var("FIREWORKS_API_KEY") {
        map.insert("FIREWORKS_API_KEY".to_string(), api_key);
    }
    let credentials = if map.is_empty() { None } else { Some(map) };

    let providers = vec![E2ETestProvider {
        variant_name: "fireworks".to_string(),
        model_name: "llama3.1-8b-instruct-fireworks".to_string(),
        model_provider_name: "fireworks".to_string(),
        credentials: None,
    }];

    let inference_params_providers = vec![E2ETestProvider {
        variant_name: "fireworks".to_string(),
        model_name: "llama3.1-8b-instruct-fireworks".to_string(),
        model_provider_name: "fireworks".to_string(),
        credentials,
    }];

    let tool_providers = vec![E2ETestProvider {
        variant_name: "fireworks-firefunction".to_string(),
        model_name: "firefunction-v2".to_string(),
        model_provider_name: "fireworks".to_string(),
        credentials: None,
    }];

    let json_providers = vec![
        E2ETestProvider {
            variant_name: "fireworks".to_string(),
            model_name: "llama3.1-8b-instruct-fireworks".to_string(),
            model_provider_name: "fireworks".to_string(),
            credentials: None,
        },
        E2ETestProvider {
            variant_name: "fireworks-implicit".to_string(),
            model_name: "firefunction-v2".to_string(),
            model_provider_name: "fireworks".to_string(),
            credentials: None,
        },
    ];

    let shorthand_providers = vec![E2ETestProvider {
        variant_name: "fireworks-shorthand".to_string(),
        model_name: "fireworks::accounts/fireworks/models/llama-v3p1-8b-instruct".to_string(),
        model_provider_name: "fireworks".to_string(),
        credentials: None,
    }];

    E2ETestProviders {
        simple_inference: providers,
        inference_params_inference: inference_params_providers,
        tool_use_inference: tool_providers.clone(),
        tool_multi_turn_inference: tool_providers.clone(),
        dynamic_tool_use_inference: tool_providers.clone(),
        parallel_tool_use_inference: tool_providers,
        json_mode_inference: json_providers,
        shorthand_inference: shorthand_providers,
        supports_batch_inference: false,
    }
}
