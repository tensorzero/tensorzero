use crate::providers::common::{E2ETestProvider, E2ETestProviders};

crate::generate_provider_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let providers = vec![E2ETestProvider {
        variant_name: "fireworks".to_string(),
        model_name: "llama3.1-8b-instruct-fireworks".to_string(),
        model_provider_name: "fireworks".to_string(),
    }];

    let tool_providers = vec![E2ETestProvider {
        variant_name: "fireworks".to_string(),
        model_name: "firefunction-v2".to_string(),
        model_provider_name: "fireworks".to_string(),
    }];

    let providers_dynamic_tool_use = vec![E2ETestProvider {
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
        simple_streaming_inference: providers.clone(),
        inference_params_inference: providers.clone(),
        inference_params_streaming_inference: providers.clone(),
        tool_use_inference: tool_providers.clone(),
        tool_use_streaming_inference: tool_providers.clone(),
        tool_multi_turn_inference: tool_providers.clone(),
        tool_multi_turn_streaming_inference: tool_providers.clone(),
        dynamic_tool_use_inference: providers_dynamic_tool_use.clone(),
        dynamic_tool_use_streaming_inference: providers_dynamic_tool_use,
        parallel_tool_use_inference: vec![],
        parallel_tool_use_streaming_inference: vec![],
        json_mode_inference: json_providers.clone(),
        json_mode_streaming_inference: json_providers,
    }
}
