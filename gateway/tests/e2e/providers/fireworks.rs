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
        model_name: "llama3.1-8b-instruct-fireworks".to_string(),
        model_provider_name: "fireworks".to_string(),
        credentials: HashMap::new(),
    }];

    let inference_params_providers = vec![E2ETestProvider {
        variant_name: "fireworks-dynamic".to_string(),
        model_name: "llama3.1-8b-instruct-fireworks-dynamic".to_string(),
        model_provider_name: "fireworks".to_string(),
        credentials,
    }];

    // NOTE: FireFunction might not be available serverlessly anymore so we have temporarily disabled it
    // let tool_providers = vec![E2ETestProvider {
    //     variant_name: "fireworks-firefunction".to_string(),
    //     model_name: "firefunction-v2".to_string(),
    //     model_provider_name: "fireworks".to_string(),
    //     credentials: HashMap::new(),
    // }];
    let tool_providers = vec![];

    let json_providers = vec![
        E2ETestProvider {
            variant_name: "fireworks".to_string(),
            model_name: "llama3.1-8b-instruct-fireworks".to_string(),
            model_provider_name: "fireworks".to_string(),
            credentials: HashMap::new(),
        },
        // E2ETestProvider {
        //     variant_name: "fireworks-implicit".to_string(),
        //     model_name: "firefunction-v2".to_string(),
        //     model_provider_name: "fireworks".to_string(),
        //     credentials: HashMap::new(),
        // },
    ];

    let shorthand_providers = vec![E2ETestProvider {
        variant_name: "fireworks-shorthand".to_string(),
        model_name: "fireworks::accounts/fireworks/models/llama-v3p1-8b-instruct".to_string(),
        model_provider_name: "fireworks".to_string(),
        credentials: HashMap::new(),
    }];

    E2ETestProviders {
        simple_inference: providers,
        inference_params_inference: inference_params_providers,
        tool_use_inference: tool_providers.clone(),
        tool_multi_turn_inference: tool_providers.clone(),
        dynamic_tool_use_inference: tool_providers.clone(),
        parallel_tool_use_inference: tool_providers,
        json_mode_inference: json_providers,
        shorthand_inference: shorthand_providers,
        supports_batch_inference: false,
    }
}
