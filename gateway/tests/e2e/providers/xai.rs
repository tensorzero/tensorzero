use crate::providers::common::{E2ETestProvider, E2ETestProviders};

crate::generate_provider_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let standard_providers = vec![
        E2ETestProvider {
            variant_name: "xai".to_string(),
            model_name: "grok-beta".to_string(),
            model_provider_name: "xai".to_string(),
        },
        E2ETestProvider {
            variant_name: "xai".to_string(),
            model_name: "grok-vision-beta".to_string(),
            model_provider_name: "xai".to_string(),
        },
        E2ETestProvider {
            variant_name: "xai".to_string(),
            model_name: "grok-2-1212".to_string(),
            model_provider_name: "xai".to_string(),
        },
        E2ETestProvider {
            variant_name: "xai".to_string(),
            model_name: "grok-2-vision-1212".to_string(),
            model_provider_name: "xai".to_string(),
        },
    ];

    E2ETestProviders {
        simple_inference: standard_providers.clone(),
        inference_params_inference: standard_providers.clone(),
        tool_use_inference: standard_providers.clone(),
        tool_multi_turn_inference: standard_providers.clone(),
        dynamic_tool_use_inference: standard_providers.clone(),
        parallel_tool_use_inference: vec![],
        json_mode_inference: vec![],
    }
}
