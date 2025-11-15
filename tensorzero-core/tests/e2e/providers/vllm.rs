use std::collections::HashMap;

use crate::providers::common::{E2ETestProvider, E2ETestProviders, ModelTestProvider};

crate::generate_provider_tests!(get_providers);
crate::generate_batch_inference_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let credentials = match std::env::var("VLLM_API_KEY") {
        Ok(key) => HashMap::from([("vllm_api_key".to_string(), key)]),
        Err(_) => HashMap::new(),
    };

    let providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "vllm".to_string(),
        model_name: "qwen2.5-0.5b-instruct-vllm".into(),
        model_provider_name: "vllm".into(),
        credentials: HashMap::new(),
    }];

    let extra_body_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "vllm-extra-body".to_string(),
        model_name: "qwen2.5-0.5b-instruct-vllm".into(),
        model_provider_name: "vllm".into(),
        credentials: HashMap::new(),
    }];

    let bad_auth_extra_headers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "vllm-extra-headers".to_string(),
        model_name: "qwen2.5-0.5b-instruct-vllm".into(),
        model_provider_name: "vllm".into(),
        credentials: HashMap::new(),
    }];

    let json_providers = vec![
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "vllm".to_string(),
            model_name: "qwen2.5-0.5b-instruct-vllm".into(),
            model_provider_name: "vllm".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "vllm-strict".to_string(),
            model_name: "qwen2.5-0.5b-instruct-vllm".into(),
            model_provider_name: "vllm".into(),
            credentials: HashMap::new(),
        },
    ];

    let inference_params_dynamic_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "vllm-dynamic".to_string(),
        model_name: "qwen2.5-0.5b-instruct-vllm-dynamic".into(),
        model_provider_name: "vllm".into(),
        credentials,
    }];

    let tool_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "vllm".to_string(),
        model_name: "qwen2.5-0.5b-instruct-vllm".to_string(),
        model_provider_name: "vllm".to_string(),
        credentials: HashMap::new(),
    }];

    // TODO: Re-enable once we can switch to a T4 GPU
    /*let reasoning_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "vllm-gpt-oss-20b".to_string(),
        model_name: "gpt-oss-20b-vllm".to_string(),
        model_provider_name: "vllm".to_string(),
        credentials: HashMap::new(),
    }];*/
    let reasoning_providers = vec![];

    // vllm requires api_base parameter, so it can't be tested with just default credentials
    let provider_type_default_credentials_providers = vec![];

    let credential_fallbacks = vec![ModelTestProvider {
        provider_type: "vllm".into(),
        model_info: HashMap::from([
            (
                "model_name".to_string(),
                "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
            ),
            (
                "api_base".to_string(),
                "https://tensorzero--vllm-inference-qwen-vllm-inference.modal.run/v1/".to_string(),
            ),
        ]),
        use_modal_headers: true,
    }];

    E2ETestProviders {
        simple_inference: providers.clone(),
        extra_body_inference: extra_body_providers,
        bad_auth_extra_headers,
        reasoning_inference: reasoning_providers,
        embeddings: vec![],
        inference_params_inference: providers.clone(),
        inference_params_dynamic_credentials: inference_params_dynamic_providers,
        provider_type_default_credentials: provider_type_default_credentials_providers,
        provider_type_default_credentials_shorthand: vec![],
        tool_use_inference: tool_providers.clone(),
        tool_multi_turn_inference: tool_providers.clone(),
        dynamic_tool_use_inference: tool_providers.clone(),
        parallel_tool_use_inference: tool_providers.clone(),
        json_mode_inference: json_providers.clone(),
        json_mode_off_inference: vec![],
        image_inference: vec![],
        pdf_inference: vec![],
        input_audio: vec![],
        shorthand_inference: vec![],
        credential_fallbacks,
    }
}
