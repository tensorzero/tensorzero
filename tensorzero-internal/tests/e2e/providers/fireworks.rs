use std::collections::HashMap;

use crate::providers::common::{E2ETestProvider, E2ETestProviders};

crate::generate_provider_tests!(get_providers);
crate::generate_batch_inference_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let credentials = match std::env::var("FIREWORKS_API_KEY") {
        Ok(key) => HashMap::from([("fireworks_api_key".to_string(), key)]),
        Err(_) => HashMap::new(),
    };

    let providers = vec![E2ETestProvider {
        variant_name: "fireworks".to_string(),
        model_name: "qwen2p5-72b-instruct".into(),
        model_provider_name: "fireworks".into(),
        credentials: HashMap::new(),
    }];

    let extra_body_providers = vec![E2ETestProvider {
        variant_name: "fireworks-extra-body".to_string(),
        model_name: "qwen2p5-72b-instruct".into(),
        model_provider_name: "fireworks".into(),
        credentials: HashMap::new(),
    }];

    let bad_auth_extra_headers = vec![E2ETestProvider {
        variant_name: "fireworks-extra-headers".to_string(),
        model_name: "qwen2p5-72b-instruct".into(),
        model_provider_name: "fireworks".into(),
        credentials: HashMap::new(),
    }];

    let inference_params_dynamic_providers = vec![E2ETestProvider {
        variant_name: "fireworks-dynamic".to_string(),
        model_name: "llama3.3-70b-instruct-fireworks-dynamic".into(),
        model_provider_name: "fireworks".into(),
        credentials,
    }];

    let tool_providers = vec![E2ETestProvider {
        variant_name: "fireworks".to_string(),
        model_name: "qwen2p5-72b-instruct".into(),
        model_provider_name: "fireworks".into(),
        credentials: HashMap::new(),
    }];

    let json_providers = vec![
        E2ETestProvider {
            variant_name: "fireworks".to_string(),
            model_name: "llama3.3-70b-instruct-fireworks".into(),
            model_provider_name: "fireworks".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            variant_name: "fireworks-implicit".to_string(),
            model_name: "qwen2p5-72b-instruct".into(),
            model_provider_name: "fireworks".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            variant_name: "fireworks-default".to_string(),
            model_name: "llama3.3-70b-instruct-fireworks".into(),
            model_provider_name: "fireworks".into(),
            credentials: HashMap::new(),
        },
    ];

    let shorthand_providers = vec![E2ETestProvider {
        variant_name: "fireworks-shorthand".to_string(),
        model_name: "fireworks::accounts/fireworks/models/llama-v3p1-8b-instruct".into(),
        model_provider_name: "fireworks".into(),
        credentials: HashMap::new(),
    }];

    E2ETestProviders {
        simple_inference: providers.clone(),
        extra_body_inference: extra_body_providers,
        bad_auth_extra_headers,
        reasoning_inference: vec![],
        inference_params_inference: providers.clone(),
        inference_params_dynamic_credentials: inference_params_dynamic_providers,
        tool_use_inference: tool_providers.clone(),
        tool_multi_turn_inference: tool_providers.clone(),
        dynamic_tool_use_inference: tool_providers.clone(),
        parallel_tool_use_inference: vec![],
        json_mode_inference: json_providers,
        image_inference: vec![],

        shorthand_inference: shorthand_providers,
        supports_batch_inference: false,
    }
}
