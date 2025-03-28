use std::collections::HashMap;

use crate::providers::common::{E2ETestProvider, E2ETestProviders};

crate::generate_provider_tests!(get_providers);
crate::generate_batch_inference_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let standard_providers = vec![E2ETestProvider {
        variant_name: "sglang".to_string(),
        model_name: "HuggingFaceTB/SmolLM-1.7B-Instruct".to_string(),
        model_provider_name: "sglang".to_string(),
        credentials: HashMap::new(),
    }];

    let extra_body_providers = vec![E2ETestProvider {
        variant_name: "sglang-extra-body".to_string(),
        model_name: "HuggingFaceTB/SmolLM-1.7B-Instruct".to_string(),
        model_provider_name: "sglang".to_string(),
        credentials: HashMap::new(),
    }];

    let bad_auth_extra_headers = vec![E2ETestProvider {
        variant_name: "sglang-extra-headers".to_string(),
        model_name: "HuggingFaceTB/SmolLM-1.7B-Instruct".to_string(),
        model_provider_name: "sglang".to_string(),
        credentials: HashMap::new(),
    }];

    let json_providers = vec![
        E2ETestProvider {
            variant_name: "sglang".to_string(),
            model_name: "HuggingFaceTB/SmolLM-1.7B-Instruct".to_string(),
            model_provider_name: "sglang".to_string(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            variant_name: "sglang-default".to_string(),
            model_name: "HuggingFaceTB/SmolLM-1.7B-Instruct".to_string(),
            model_provider_name: "sglang".to_string(),
            credentials: HashMap::new(),
        },
    ];

    E2ETestProviders {
        simple_inference: standard_providers.clone(),
        extra_body_inference: extra_body_providers,
        bad_auth_extra_headers,
        reasoning_inference: vec![],
        inference_params_inference: standard_providers.clone(),
        inference_params_dynamic_credentials: vec![],
        // TODO: Support tool use once SGLang makes a releawe with https://github.com/sgl-project/sglang/pull/2544
        tool_use_inference: vec![],
        tool_multi_turn_inference: vec![],
        dynamic_tool_use_inference: vec![],
        parallel_tool_use_inference: vec![],
        json_mode_inference: json_providers.clone(),
        image_inference: vec![],

        shorthand_inference: vec![],
        supports_batch_inference: false,
    }
}
