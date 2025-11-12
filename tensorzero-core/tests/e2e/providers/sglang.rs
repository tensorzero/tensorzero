use std::collections::HashMap;

use crate::providers::common::{E2ETestProvider, E2ETestProviders, ModelTestProvider};

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

    let credential_fallbacks = vec![ModelTestProvider {
        provider_type: "sglang".to_string(),
        model_info: HashMap::from([
            (
                "model_name".to_string(),
                "Qwen/Qwen2.5-1.5B-Instruct".to_string(),
            ),
            (
                "api_base".to_string(),
                "https://tensorzero--sglang-0-4-10-inference-sglang-inference.modal.run/v1/"
                    .to_string(),
            ),
        ]),
        use_modal_headers: true,
    }];

    E2ETestProviders {
        simple_inference: standard_providers.clone(),
        extra_body_inference: extra_body_providers,
        bad_auth_extra_headers: vec![],
        reasoning_inference: vec![],
        embeddings: vec![],
        inference_params_inference: standard_providers.clone(),
        inference_params_dynamic_credentials: vec![],
        provider_type_default_credentials: vec![],
        provider_type_default_credentials_shorthand: vec![],
        tool_use_inference: tool_providers.clone(),
        tool_multi_turn_inference: vec![],
        dynamic_tool_use_inference: vec![],
        parallel_tool_use_inference: vec![],
        json_mode_inference: json_providers.clone(),
        json_mode_off_inference: json_mode_off_providers.clone(),
        image_inference: vec![],
        pdf_inference: vec![],
        input_audio: vec![],
        shorthand_inference: vec![],
        credential_fallbacks,
    }
}
