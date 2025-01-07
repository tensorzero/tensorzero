use std::collections::HashMap;

use crate::providers::common::{E2ETestProvider, E2ETestProviders};

crate::generate_provider_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let credentials = match std::env::var("HYPERBOLIC_API_KEY") {
        Ok(key) => HashMap::from([("hyperbolic_api_key".to_string(), key)]),
        Err(_) => HashMap::new(),
    };

    let standard_providers = vec![E2ETestProvider {
        variant_name: "hyperbolic".to_string(),
        model_name: "meta-llama/Meta-Llama-3-70B-Instruct".to_string(),
        model_provider_name: "hyperbolic".to_string(),
        credentials: HashMap::new(),
    }];

    let inference_params_providers = vec![E2ETestProvider {
        variant_name: "hyperbolic-dynamic".to_string(),
        model_name: "meta-llama/Meta-Llama-3-70B-Instruct-dynamic".to_string(),
        model_provider_name: "hyperbolic".to_string(),
        credentials,
    }];

    let shorthand_providers = vec![E2ETestProvider {
        variant_name: "hyperbolic-shorthand".to_string(),
        model_name: "hyperbolic::meta-llama/Meta-Llama-3-70B-Instruct".to_string(),
        model_provider_name: "hyperbolic".to_string(),
        credentials: HashMap::new(),
    }];

    E2ETestProviders {
        simple_inference: standard_providers,
        inference_params_inference: inference_params_providers,
        tool_use_inference: vec![],
        tool_multi_turn_inference: vec![],
        dynamic_tool_use_inference: vec![],
        parallel_tool_use_inference: vec![],
        json_mode_inference: vec![],
        shorthand_inference: shorthand_providers.clone(),
        supports_batch_inference: false,
    }
}
