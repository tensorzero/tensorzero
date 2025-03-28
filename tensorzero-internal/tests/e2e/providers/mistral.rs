use std::collections::HashMap;

use crate::providers::common::{E2ETestProvider, E2ETestProviders};

crate::generate_provider_tests!(get_providers);
crate::generate_batch_inference_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let credentials = match std::env::var("MISTRAL_API_KEY") {
        Ok(key) => HashMap::from([("mistral_api_key".to_string(), key)]),
        Err(_) => HashMap::new(),
    };

    let providers = vec![E2ETestProvider {
        variant_name: "mistral".to_string(),
        model_name: "open-mistral-nemo-2407".into(),
        model_provider_name: "mistral".into(),
        credentials: HashMap::new(),
    }];

    let extra_body_providers = vec![E2ETestProvider {
        variant_name: "mistral-extra-body".to_string(),
        model_name: "open-mistral-nemo-2407".into(),
        model_provider_name: "mistral".into(),
        credentials: HashMap::new(),
    }];

    let bad_auth_extra_headers = vec![E2ETestProvider {
        variant_name: "mistral-extra-headers".to_string(),
        model_name: "open-mistral-nemo-2407".into(),
        model_provider_name: "mistral".into(),
        credentials: HashMap::new(),
    }];

    let json_providers = vec![
        E2ETestProvider {
            variant_name: "mistral".to_string(),
            model_name: "open-mistral-nemo-2407".into(),
            model_provider_name: "mistral".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            variant_name: "mistral-default".to_string(),
            model_name: "open-mistral-nemo-2407".into(),
            model_provider_name: "mistral".into(),
            credentials: HashMap::new(),
        },
    ];

    let inference_params_dynamic_providers = vec![E2ETestProvider {
        variant_name: "mistral-dynamic".to_string(),
        model_name: "open-mistral-nemo-2407-dynamic".into(),
        model_provider_name: "mistral".into(),
        credentials,
    }];

    let shorthand_providers = vec![E2ETestProvider {
        variant_name: "mistral-shorthand".to_string(),
        model_name: "mistral::open-mistral-nemo-2407".into(),
        model_provider_name: "mistral".into(),
        credentials: HashMap::new(),
    }];

    E2ETestProviders {
        simple_inference: providers.clone(),
        extra_body_inference: extra_body_providers,
        bad_auth_extra_headers,
        reasoning_inference: vec![],
        inference_params_inference: providers.clone(),
        inference_params_dynamic_credentials: inference_params_dynamic_providers,
        tool_use_inference: providers.clone(),
        tool_multi_turn_inference: providers.clone(),
        dynamic_tool_use_inference: providers.clone(),
        parallel_tool_use_inference: vec![],
        json_mode_inference: json_providers.clone(),
        image_inference: vec![],

        shorthand_inference: shorthand_providers.clone(),
        supports_batch_inference: false,
    }
}
