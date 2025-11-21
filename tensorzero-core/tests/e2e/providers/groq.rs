use std::collections::HashMap;

use crate::providers::common::{E2ETestProvider, E2ETestProviders, ModelTestProvider};

crate::generate_provider_tests!(get_providers);
crate::generate_batch_inference_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let credentials = match std::env::var("GROQ_API_KEY") {
        Ok(key) => HashMap::from([("groq_api_key".to_string(), key)]),
        Err(_) => HashMap::new(),
    };

    let standard_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "groq".to_string(),
        model_name: "groq-qwen".into(),
        model_provider_name: "groq".into(),
        credentials: HashMap::new(),
    }];

    let extra_body_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "groq-extra-body".to_string(),
        model_name: "groq-qwen".into(),
        model_provider_name: "groq".into(),
        credentials: HashMap::new(),
    }];

    let bad_auth_extra_headers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "groq-extra-headers".to_string(),
        model_name: "groq-qwen".into(),
        model_provider_name: "groq".into(),
        credentials: HashMap::new(),
    }];

    let inference_params_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "groq".to_string(),
        model_name: "groq-qwen".into(),
        model_provider_name: "groq".into(),
        credentials: credentials.clone(),
    }];

    let inference_params_dynamic_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "groq-dynamic".to_string(),
        model_name: "groq-qwen-dynamic".into(),
        model_provider_name: "groq".into(),
        credentials,
    }];

    let shorthand_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "groq-shorthand".to_string(),
        model_name: "groq::meta-llama/llama-4-scout-17b-16e-instruct".into(),
        model_provider_name: "groq".into(),
        credentials: HashMap::new(),
    }];

    let json_providers = vec![
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "groq".to_string(),
            model_name: "groq-qwen".into(),
            model_provider_name: "groq".into(),
            credentials: HashMap::new(),
        },
        // Groq gives 500 errors when tool calls fail to parse. We don't test
        // json_mode = "tool" here for that reason.
        // We don't recommend its use with Groq.
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "groq-strict".to_string(),
            model_name: "groq-qwen".into(),
            model_provider_name: "groq".into(),
            credentials: HashMap::new(),
        },
    ];

    let provider_type_default_credentials_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "groq".to_string(),
        model_name: "qwen/qwen3-32b".into(),
        model_provider_name: "groq".into(),
        credentials: HashMap::new(),
    }];

    let provider_type_default_credentials_shorthand_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "groq-shorthand".to_string(),
        model_name: "groq::qwen/qwen3-32b".into(),
        model_provider_name: "groq".into(),
        credentials: HashMap::new(),
    }];

    let credential_fallbacks = vec![ModelTestProvider {
        provider_type: "groq".to_string(),
        model_info: HashMap::from([("model_name".to_string(), "qwen/qwen3-32b".to_string())]),
        use_modal_headers: false,
    }];

    E2ETestProviders {
        simple_inference: standard_providers.clone(),
        extra_body_inference: extra_body_providers,
        bad_auth_extra_headers,
        reasoning_inference: vec![],
        embeddings: vec![],
        inference_params_inference: inference_params_providers,
        inference_params_dynamic_credentials: inference_params_dynamic_providers,
        provider_type_default_credentials: provider_type_default_credentials_providers,
        provider_type_default_credentials_shorthand:
            provider_type_default_credentials_shorthand_providers,
        tool_use_inference: standard_providers.clone(),
        tool_multi_turn_inference: standard_providers.clone(),
        dynamic_tool_use_inference: standard_providers.clone(),
        parallel_tool_use_inference: standard_providers.clone(),
        json_mode_inference: json_providers.clone(),
        image_inference: vec![],
        pdf_inference: vec![],
        input_audio: vec![],
        shorthand_inference: shorthand_providers,
        json_mode_off_inference: vec![],
        credential_fallbacks,
    }
}
