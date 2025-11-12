use std::collections::HashMap;

use crate::providers::common::{E2ETestProvider, E2ETestProviders, ModelTestProvider};

crate::generate_provider_tests!(get_providers);
crate::generate_batch_inference_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let credentials = match std::env::var("FIREWORKS_API_KEY") {
        Ok(key) => HashMap::from([("fireworks_api_key".to_string(), key)]),
        Err(_) => HashMap::new(),
    };

    let providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "fireworks".to_string(),
        model_name: "fireworks::accounts/fireworks/models/deepseek-v3p1".into(),
        model_provider_name: "fireworks".into(),
        credentials: HashMap::new(),
    }];

    let extra_body_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "fireworks-extra-body".to_string(),
        model_name: "fireworks::accounts/fireworks/models/deepseek-v3p1".into(),
        model_provider_name: "fireworks".into(),
        credentials: HashMap::new(),
    }];

    let bad_auth_extra_headers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "fireworks-extra-headers".to_string(),
        model_name: "fireworks::accounts/fireworks/models/deepseek-r1-0528".into(),
        model_provider_name: "fireworks".into(),
        credentials: HashMap::new(),
    }];

    let inference_params_dynamic_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "fireworks-dynamic".to_string(),
        model_name: "fireworks::accounts/fireworks/models/deepseek-v3p1".into(),
        model_provider_name: "fireworks".into(),
        credentials,
    }];

    let tool_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "fireworks".to_string(),
        model_name: "fireworks::accounts/fireworks/models/deepseek-v3p1".into(),
        model_provider_name: "fireworks".into(),
        credentials: HashMap::new(),
    }];

    let json_providers = vec![
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "fireworks".to_string(),
            model_name: "fireworks::accounts/fireworks/models/deepseek-v3p1".into(),
            model_provider_name: "fireworks".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "fireworks-implicit".to_string(),
            model_name: "fireworks::accounts/fireworks/models/deepseek-v3p1".into(),
            model_provider_name: "fireworks".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "fireworks-strict".to_string(),
            model_name: "fireworks::accounts/fireworks/models/deepseek-v3p1".into(),
            model_provider_name: "fireworks".into(),
            credentials: HashMap::new(),
        },
    ];

    let json_mode_off_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "fireworks_json_mode_off".to_string(),
        model_name: "fireworks::accounts/fireworks/models/deepseek-v3p1".into(),
        model_provider_name: "fireworks".into(),
        credentials: HashMap::new(),
    }];

    let shorthand_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "fireworks-shorthand".to_string(),
        model_name: "fireworks::accounts/fireworks/models/llama-v3p1-8b-instruct".into(),
        model_provider_name: "fireworks".into(),
        credentials: HashMap::new(),
    }];

    let thinking_block_providers = vec![
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "fireworks-deepseek".to_string(),
            model_name: "deepseek-r1".into(),
            model_provider_name: "fireworks".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "fireworks-gpt-oss-20b".to_string(),
            model_name: "gpt-oss-20b-fireworks".into(),
            model_provider_name: "fireworks".into(),
            credentials: HashMap::new(),
        },
    ];

    let provider_type_default_credentials_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "fireworks".to_string(),
        model_name: "accounts/fireworks/models/deepseek-v3p1".into(),
        model_provider_name: "fireworks".into(),
        credentials: HashMap::new(),
    }];

    let provider_type_default_credentials_shorthand_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "fireworks-shorthand".to_string(),
        model_name: "fireworks::accounts/fireworks/models/deepseek-v3p1".into(),
        model_provider_name: "fireworks".into(),
        credentials: HashMap::new(),
    }];

    let credential_fallbacks = vec![ModelTestProvider {
        provider_type: "fireworks".to_string(),
        model_info: HashMap::from([(
            "model_name".to_string(),
            "accounts/fireworks/models/deepseek-v3p1".to_string(),
        )]),
        use_modal_headers: false,
    }];

    E2ETestProviders {
        simple_inference: providers.clone(),
        extra_body_inference: extra_body_providers,
        bad_auth_extra_headers,
        reasoning_inference: thinking_block_providers.clone(),
        embeddings: vec![],
        inference_params_inference: providers.clone(),
        inference_params_dynamic_credentials: inference_params_dynamic_providers,
        provider_type_default_credentials: provider_type_default_credentials_providers,
        provider_type_default_credentials_shorthand:
            provider_type_default_credentials_shorthand_providers,
        tool_use_inference: tool_providers.clone(),
        tool_multi_turn_inference: tool_providers.clone(),
        dynamic_tool_use_inference: tool_providers.clone(),
        parallel_tool_use_inference: vec![],
        json_mode_inference: json_providers,
        json_mode_off_inference: json_mode_off_providers.clone(),
        image_inference: vec![],
        pdf_inference: vec![],
        input_audio: vec![],
        shorthand_inference: shorthand_providers,
        credential_fallbacks,
    }
}
