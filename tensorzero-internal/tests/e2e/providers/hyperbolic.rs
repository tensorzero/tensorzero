use std::collections::HashMap;

use crate::providers::common::{E2ETestProvider, E2ETestProviders};

crate::generate_provider_tests!(get_providers);
crate::generate_batch_inference_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let credentials = match std::env::var("HYPERBOLIC_API_KEY") {
        Ok(key) => HashMap::from([("hyperbolic_api_key".to_string(), key)]),
        Err(_) => HashMap::new(),
    };

    let standard_providers = vec![E2ETestProvider {
        variant_name: "hyperbolic".to_string(),
        model_name: "meta-llama/Meta-Llama-3-70B-Instruct".into(),
        model_provider_name: "hyperbolic".into(),
        credentials: HashMap::new(),
    }];

    let extra_body_providers = vec![E2ETestProvider {
        variant_name: "hyperbolic-extra-body".to_string(),
        model_name: "meta-llama/Meta-Llama-3-70B-Instruct".into(),
        model_provider_name: "hyperbolic".into(),
        credentials: HashMap::new(),
    }];

    let bad_auth_extra_headers = vec![E2ETestProvider {
        variant_name: "hyperbolic-extra-headers".to_string(),
        model_name: "meta-llama/Meta-Llama-3-70B-Instruct".into(),
        model_provider_name: "hyperbolic".into(),
        credentials: HashMap::new(),
    }];

    let inference_params_dynamic_providers = vec![E2ETestProvider {
        variant_name: "hyperbolic-dynamic".to_string(),
        model_name: "meta-llama/Meta-Llama-3-70B-Instruct-dynamic".into(),
        model_provider_name: "hyperbolic".into(),
        credentials,
    }];

    let shorthand_providers = vec![E2ETestProvider {
        variant_name: "hyperbolic-shorthand".to_string(),
        model_name: "hyperbolic::meta-llama/Meta-Llama-3-70B-Instruct".into(),
        model_provider_name: "hyperbolic".into(),
        credentials: HashMap::new(),
    }];

    E2ETestProviders {
        simple_inference: standard_providers.clone(),
        bad_auth_extra_headers,
        reasoning_inference: vec![],
        extra_body_inference: extra_body_providers,
        inference_params_inference: standard_providers,
        inference_params_dynamic_credentials: inference_params_dynamic_providers,
        tool_use_inference: vec![],
        tool_multi_turn_inference: vec![],
        dynamic_tool_use_inference: vec![],
        parallel_tool_use_inference: vec![],
        json_mode_inference: vec![],
        image_inference: vec![],

        shorthand_inference: shorthand_providers.clone(),
        supports_batch_inference: false,
    }
}
