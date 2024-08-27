use crate::providers::common::{E2ETestProvider, E2ETestProviders};

crate::generate_provider_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let standard_providers = vec![E2ETestProvider {
        variant_name: "openai".to_string(),
        model_name: "gpt-4o-mini-2024-07-18".to_string(),
        model_provider_name: "openai".to_string(),
    }];

    let json_providers = vec![
        E2ETestProvider {
            variant_name: "openai".to_string(),
            model_name: "gpt-4o-mini-2024-07-18".to_string(),
            model_provider_name: "openai".to_string(),
        },
        E2ETestProvider {
            variant_name: "openai-implicit".to_string(),
            model_name: "gpt-4o-mini-2024-07-18".to_string(),
            model_provider_name: "openai".to_string(),
        },
        E2ETestProvider {
            variant_name: "openai-strict".to_string(),
            model_name: "gpt-4o-mini-2024-07-18".to_string(),
            model_provider_name: "openai".to_string(),
        },
    ];

    E2ETestProviders {
        simple_inference: standard_providers.clone(),
        inference_params_inference: standard_providers.clone(),
        tool_use_inference: standard_providers.clone(),
        tool_multi_turn_inference: standard_providers.clone(),
        dynamic_tool_use_inference: standard_providers.clone(),
        parallel_tool_use_inference: standard_providers.clone(),
        json_mode_inference: json_providers.clone(),
    }
}
