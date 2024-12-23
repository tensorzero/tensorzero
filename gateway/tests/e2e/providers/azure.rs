use crate::providers::common::{E2ETestProvider, E2ETestProviders};

#[cfg(feature = "e2e_tests")]
crate::generate_provider_tests!(get_providers);
#[cfg(feature = "batch_tests")]
crate::generate_batch_inference_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let standard_providers = vec![E2ETestProvider {
        variant_name: "azure".to_string(),
        model_name: "gpt-4o-mini-azure".to_string(),
        model_provider_name: "azure".to_string(),
    }];

    let json_providers = vec![
        E2ETestProvider {
            variant_name: "azure".to_string(),
            model_name: "gpt-4o-mini-azure".to_string(),
            model_provider_name: "azure".to_string(),
        },
        E2ETestProvider {
            variant_name: "azure-implicit".to_string(),
            model_name: "gpt-4o-mini-azure".to_string(),
            model_provider_name: "azure".to_string(),
        },
    ];

    E2ETestProviders {
        simple_inference: standard_providers.clone(),
        inference_params_inference: standard_providers.clone(),
        tool_use_inference: standard_providers.clone(),
        tool_multi_turn_inference: standard_providers.clone(),
        dynamic_tool_use_inference: standard_providers.clone(),
        parallel_tool_use_inference: vec![],
        json_mode_inference: json_providers.clone(),
        shorthand_inference: vec![],
        #[cfg(feature = "batch_tests")]
        supports_batch_inference: false,
    }
}
