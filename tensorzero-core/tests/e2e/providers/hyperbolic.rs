use std::collections::HashMap;

use crate::providers::common::{E2ETestProvider, E2ETestProviders, ModelTestProvider};

crate::generate_provider_tests!(get_providers);
crate::generate_batch_inference_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let credentials = match std::env::var("HYPERBOLIC_API_KEY") {
        Ok(key) => HashMap::from([("hyperbolic_api_key".to_string(), key)]),
        Err(_) => HashMap::new(),
    };

    let standard_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "hyperbolic".to_string(),
        model_name: "meta-llama/Meta-Llama-3-70B-Instruct".into(),
        model_provider_name: "hyperbolic".into(),
        credentials: HashMap::new(),
    }];

    let extra_body_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "hyperbolic-extra-body".to_string(),
        model_name: "meta-llama/Meta-Llama-3-70B-Instruct".into(),
        model_provider_name: "hyperbolic".into(),
        credentials: HashMap::new(),
    }];

    let bad_auth_extra_headers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "hyperbolic-extra-headers".to_string(),
        model_name: "meta-llama/Meta-Llama-3-70B-Instruct".into(),
        model_provider_name: "hyperbolic".into(),
        credentials: HashMap::new(),
    }];

    let inference_params_dynamic_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "hyperbolic-dynamic".to_string(),
        model_name: "meta-llama/Meta-Llama-3-70B-Instruct-dynamic".into(),
        model_provider_name: "hyperbolic".into(),
        credentials,
    }];

    let shorthand_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "hyperbolic-shorthand".to_string(),
        model_name: "hyperbolic::meta-llama/Meta-Llama-3-70B-Instruct".into(),
        model_provider_name: "hyperbolic".into(),
        credentials: HashMap::new(),
    }];

    let provider_type_default_credentials_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "hyperbolic".to_string(),
        model_name: "meta-llama/Meta-Llama-3-70B-Instruct".into(),
        model_provider_name: "hyperbolic".into(),
        credentials: HashMap::new(),
    }];

    let provider_type_default_credentials_shorthand_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "hyperbolic-shorthand".to_string(),
        model_name: "hyperbolic::meta-llama/Meta-Llama-3-70B-Instruct".into(),
        model_provider_name: "hyperbolic".into(),
        credentials: HashMap::new(),
    }];

    let credential_fallbacks = vec![ModelTestProvider {
        provider_type: "hyperbolic".to_string(),
        model_info: HashMap::from([(
            "model_name".to_string(),
            "meta-llama/Meta-Llama-3-70B-Instruct".to_string(),
        )]),
        use_modal_headers: false,
    }];

    E2ETestProviders {
        simple_inference: standard_providers.clone(),
        bad_auth_extra_headers,
        reasoning_inference: vec![],
        extra_body_inference: extra_body_providers,
        embeddings: vec![],
        inference_params_inference: standard_providers,
        inference_params_dynamic_credentials: inference_params_dynamic_providers,
        provider_type_default_credentials: provider_type_default_credentials_providers,
        provider_type_default_credentials_shorthand:
            provider_type_default_credentials_shorthand_providers,
        tool_use_inference: vec![],
        tool_multi_turn_inference: vec![],
        dynamic_tool_use_inference: vec![],
        parallel_tool_use_inference: vec![],
        json_mode_inference: vec![],
        json_mode_off_inference: vec![],
        image_inference: vec![],
        pdf_inference: vec![],
        input_audio: vec![],
        shorthand_inference: shorthand_providers.clone(),
        credential_fallbacks,
    }
}
