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
        model_name: "open-mistral-nemo-2407".to_string(),
        model_provider_name: "mistral".to_string(),
        credentials: HashMap::new(),
    }];

    let inference_params_providers = vec![E2ETestProvider {
        variant_name: "mistral-dynamic".to_string(),
        model_name: "open-mistral-nemo-2407-dynamic".to_string(),
        model_provider_name: "mistral".to_string(),
        credentials,
    }];

    let shorthand_providers = vec![E2ETestProvider {
        variant_name: "mistral-shorthand".to_string(),
        model_name: "mistral::open-mistral-nemo-2407".to_string(),
        model_provider_name: "mistral".to_string(),
        credentials: HashMap::new(),
    }];

    E2ETestProviders {
        simple_inference: providers.clone(),
        inference_params_inference: inference_params_providers,
        tool_use_inference: providers.clone(),
        tool_multi_turn_inference: providers.clone(),
        dynamic_tool_use_inference: providers.clone(),
        parallel_tool_use_inference: vec![],
        json_mode_inference: providers.clone(),
        shorthand_inference: shorthand_providers.clone(),
        supports_batch_inference: false,
    }
}
