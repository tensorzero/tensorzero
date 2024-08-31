use crate::providers::common::{E2ETestProvider, E2ETestProviders};

crate::generate_provider_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let providers = vec![E2ETestProvider {
        variant_name: "fireworks".to_string(),
        model_name: "llama3.1-8b-instruct-fireworks".to_string(),
        model_provider_name: "fireworks".to_string(),
    }];

    let tool_providers = vec![E2ETestProvider {
        variant_name: "fireworks-firefunction".to_string(),
        model_name: "firefunction-v2".to_string(),
        model_provider_name: "fireworks".to_string(),
    }];

    let json_providers = vec![
        E2ETestProvider {
            variant_name: "fireworks".to_string(),
            model_name: "llama3.1-8b-instruct-fireworks".to_string(),
            model_provider_name: "fireworks".to_string(),
        },
        E2ETestProvider {
            variant_name: "fireworks-implicit".to_string(),
            model_name: "firefunction-v2".to_string(),
            model_provider_name: "fireworks".to_string(),
        },
    ];

    E2ETestProviders {
        simple_inference: providers.clone(),
        inference_params_inference: providers,
        tool_use_inference: tool_providers.clone(),
        tool_multi_turn_inference: tool_providers.clone(),
        dynamic_tool_use_inference: tool_providers.clone(),
        parallel_tool_use_inference: tool_providers,
        json_mode_inference: json_providers,
    }
}
