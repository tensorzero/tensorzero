use crate::providers::common::{E2ETestProvider, E2ETestProviders, ModelTestProvider};
use std::collections::HashMap;

crate::generate_provider_tests!(get_providers);
crate::generate_batch_inference_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let model_name = "phi-3.5-mini-instruct-tgi".to_string();
    let standard_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "tgi".to_string(),
        model_name: model_name.clone(),
        model_provider_name: "tgi".into(),
        credentials: HashMap::new(),
    }];

    let extra_body_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "tgi-extra-body".to_string(),
        model_name: model_name.clone(),
        model_provider_name: "tgi".into(),
        credentials: HashMap::new(),
    }];

    let bad_auth_extra_headers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "tgi-extra-headers".to_string(),
        model_name: model_name.clone(),
        model_provider_name: "tgi".into(),
        credentials: HashMap::new(),
    }];

    let json_providers = vec![
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "tgi".to_string(),
            model_name: model_name.clone(),
            model_provider_name: "tgi".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "tgi-strict".to_string(),
            model_name: model_name.clone(),
            model_provider_name: "tgi".into(),
            credentials: HashMap::new(),
        },
    ];

    let json_mode_off_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "tgi_json_mode_off".to_string(),
        model_name: model_name.clone(),
        model_provider_name: "tgi".into(),
        credentials: HashMap::new(),
    }];

    let credential_fallbacks = vec![ModelTestProvider {
        provider_type: "tgi".into(),
        model_info: HashMap::from([(
            "api_base".to_string(),
            "https://zr0gj152lrhnrr-80.proxy.runpod.net/v1/".to_string(),
        )]),
        use_modal_headers: false,
    }];

    E2ETestProviders {
        simple_inference: standard_providers.clone(),
        extra_body_inference: extra_body_providers,
        bad_auth_extra_headers,
        reasoning_inference: vec![],
        embeddings: vec![],
        inference_params_inference: standard_providers.clone(),
        inference_params_dynamic_credentials: vec![],
        provider_type_default_credentials: vec![],
        provider_type_default_credentials_shorthand: vec![],
        tool_use_inference: vec![],
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
