use std::collections::HashMap;

use crate::providers::common::{E2ETestProvider, E2ETestProviders};

crate::generate_provider_tests!(get_providers);
crate::generate_batch_inference_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let credentials = match std::env::var("VLLM_API_KEY") {
        Ok(key) => HashMap::from([("vllm_api_key".to_string(), key)]),
        Err(_) => HashMap::new(),
    };

    let providers = vec![E2ETestProvider {
        variant_name: "vllm".to_string(),
        model_name: "microsoft/Phi-3.5-mini-instruct".to_string(),
        model_provider_name: "vllm".to_string(),
        credentials: HashMap::new(),
    }];

    let inference_params_providers = vec![E2ETestProvider {
        variant_name: "vllm-dynamic".to_string(),
        model_name: "microsoft/Phi-3.5-mini-instruct-dynamic".to_string(),
        model_provider_name: "vllm".to_string(),
        credentials,
    }];

    // TODOs (#169): Implement a solution for vLLM tool use
    E2ETestProviders {
        simple_inference: providers.clone(),
        inference_params_inference: inference_params_providers,
        tool_use_inference: vec![],
        tool_multi_turn_inference: vec![],
        dynamic_tool_use_inference: vec![],
        parallel_tool_use_inference: vec![],
        json_mode_inference: providers.clone(),
        shorthand_inference: vec![],
        supports_batch_inference: false,
    }
}
