use crate::providers::common::{E2ETestProvider, E2ETestProviders};

crate::generate_provider_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let provider = vec![E2ETestProvider {
        variant_name: "fireworks".to_string(),
    }];

    let provider_dynamic_tool_use = vec![E2ETestProvider {
        variant_name: "fireworks-firefunction".to_string(),
    }];

    let json_providers = vec![
        E2ETestProvider {
            variant_name: "fireworks".to_string(),
        },
        E2ETestProvider {
            variant_name: "fireworks-implicit".to_string(),
        },
    ];

    E2ETestProviders {
        simple_inference: provider.clone(),
        simple_streaming_inference: provider.clone(),
        inference_params_inference: provider.clone(),
        inference_params_streaming_inference: provider.clone(),
        tool_use_inference: provider.clone(),
        tool_use_streaming_inference: provider.clone(),
        tool_multi_turn_inference: provider.clone(),
        tool_multi_turn_streaming_inference: provider.clone(),
        dynamic_tool_use_inference: provider_dynamic_tool_use.clone(),
        dynamic_tool_use_streaming_inference: provider_dynamic_tool_use,
        json_mode_inference: json_providers.clone(),
        json_mode_streaming_inference: json_providers,
    }
}
