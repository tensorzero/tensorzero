use crate::providers::common::{E2ETestProvider, E2ETestProviders};

#[cfg(feature = "e2e_tests")]
crate::generate_provider_tests!(get_providers);
#[cfg(feature = "batch_tests")]
crate::generate_batch_inference_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let providers = vec![E2ETestProvider {
        variant_name: "mistral".to_string(),
        model_name: "open-mistral-nemo-2407".to_string(),
        model_provider_name: "mistral".to_string(),
    }];

    #[cfg(feature = "e2e_tests")]
    let shorthand_providers = vec![E2ETestProvider {
        variant_name: "mistral-shorthand".to_string(),
        model_name: "mistral::open-mistral-nemo-2407".to_string(),
        model_provider_name: "mistral".to_string(),
    }];

    E2ETestProviders {
        simple_inference: providers.clone(),
        inference_params_inference: providers.clone(),
        tool_use_inference: providers.clone(),
        tool_multi_turn_inference: providers.clone(),
        dynamic_tool_use_inference: providers.clone(),
        parallel_tool_use_inference: vec![],
        json_mode_inference: providers.clone(),
        #[cfg(feature = "e2e_tests")]
        shorthand_inference: shorthand_providers.clone(),
        #[cfg(feature = "batch_tests")]
        supports_batch_inference: false,
    }
}
