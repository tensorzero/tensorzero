use crate::providers::common::{E2ETestProvider, E2ETestProviders};
use std::collections::HashMap;
#[cfg(feature = "e2e_tests")]
crate::generate_provider_tests!(get_providers);
#[cfg(feature = "batch_tests")]
crate::generate_batch_inference_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let model_name = "phi-3.5-mini-instruct-tgi".to_string();
    let standard_providers = vec![E2ETestProvider {
        variant_name: "tgi".to_string(),
        model_name: model_name.clone(),
        model_provider_name: "tgi".into(),
        credentials: HashMap::new(),
    }];
    E2ETestProviders {
        simple_inference: standard_providers.clone(),
        inference_params_inference: standard_providers.clone(),
        tool_use_inference: vec![],
        tool_multi_turn_inference: vec![],
        dynamic_tool_use_inference: vec![],
        parallel_tool_use_inference: vec![],
        json_mode_inference: standard_providers.clone(),
        #[cfg(feature = "e2e_tests")]
        shorthand_inference: vec![],
        #[cfg(feature = "batch_tests")]
        supports_batch_inference: false,
    }
}
