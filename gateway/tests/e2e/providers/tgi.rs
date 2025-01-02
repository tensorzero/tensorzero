use crate::providers::common::{E2ETestProvider, E2ETestProviders};

crate::generate_provider_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let model_name = "tgi".to_string();
    let standard_providers = vec![E2ETestProvider {
        variant_name: model_name.clone(),
        model_name: model_name.clone(),
        model_provider_name: model_name.clone(),
    }];

    E2ETestProviders {
        simple_inference: standard_providers.clone(),
        inference_params_inference: standard_providers.clone(),
        tool_use_inference: standard_providers.clone(),
        tool_multi_turn_inference: standard_providers.clone(),
        dynamic_tool_use_inference: standard_providers.clone(),
        parallel_tool_use_inference: vec![],
        json_mode_inference: vec![],
        shorthand_inference: vec![],
        supports_batch_inference: false,
    }
}
