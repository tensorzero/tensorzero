use std::collections::HashMap;

use crate::providers::common::{E2ETestProvider, E2ETestProviders};

crate::generate_provider_tests!(get_providers);
crate::generate_batch_inference_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let standard_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "sglang".to_string(),
        model_name: "Qwen/Qwen2.5-1.5B-Instruct".to_string(),
        model_provider_name: "sglang".to_string(),
        credentials: HashMap::new(),
    }];

    let extra_body_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "sglang-extra-body".to_string(),
        model_name: "Qwen/Qwen2.5-1.5B-Instruct".to_string(),
        model_provider_name: "sglang".to_string(),
        credentials: HashMap::new(),
    }];

    let json_providers = vec![
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "sglang".to_string(),
            model_name: "Qwen/Qwen2.5-1.5B-Instruct".to_string(),
            model_provider_name: "sglang".to_string(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "sglang-strict".to_string(),
            model_name: "Qwen/Qwen2.5-1.5B-Instruct".to_string(),
            model_provider_name: "sglang".to_string(),
            credentials: HashMap::new(),
        },
    ];

    let tool_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "sglang".to_string(),
        model_name: "Qwen/Qwen2.5-1.5B-Instruct".to_string(),
        model_provider_name: "sglang".to_string(),
        credentials: HashMap::new(),
    }];

    let json_mode_off_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "sglang_json_mode_off".to_string(),
        model_name: "Qwen/Qwen2.5-1.5B-Instruct".to_string(),
        model_provider_name: "sglang".to_string(),
        credentials: HashMap::new(),
    }];

    E2ETestProviders {
        simple_inference: standard_providers.clone(),
        extra_body_inference: extra_body_providers,
        bad_auth_extra_headers: vec![],
        reasoning_inference: vec![],
        inference_params_inference: standard_providers.clone(),
        inference_params_dynamic_credentials: vec![],
        tool_use_inference: tool_providers.clone(),
        tool_multi_turn_inference: vec![],
        dynamic_tool_use_inference: vec![],
        parallel_tool_use_inference: vec![],
        json_mode_inference: json_providers.clone(),
        json_mode_off_inference: json_mode_off_providers.clone(),
        image_inference: vec![],
        pdf_inference: vec![],
        shorthand_inference: vec![],
    }
}
