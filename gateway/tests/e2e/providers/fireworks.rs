use std::collections::HashMap;

use crate::providers::common::{E2ETestProvider, E2ETestProviders};

#[cfg(feature = "e2e_tests")]
crate::generate_provider_tests!(get_providers);
#[cfg(feature = "batch_tests")]
crate::generate_batch_inference_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let credentials = match std::env::var("FIREWORKS_API_KEY") {
        Ok(key) => HashMap::from([("fireworks_api_key".to_string(), key)]),
        Err(_) => HashMap::new(),
    };

    let providers = vec![E2ETestProvider {
        variant_name: "fireworks".to_string(),
        model_name: "llama3.1-8b-instruct-fireworks".into(),
        model_provider_name: "fireworks".into(),
        credentials: HashMap::new(),
    }];

    let inference_params_providers = vec![E2ETestProvider {
        variant_name: "fireworks-dynamic".to_string(),
        model_name: "llama3.1-8b-instruct-fireworks-dynamic".into(),
        model_provider_name: "fireworks".into(),
        credentials,
    }];

    // NOTE: FireFunction might not be available serverlessly anymore so we have temporarily disabled it
    // let tool_providers = vec![E2ETestProvider {
    //     variant_name: "fireworks-firefunction".to_string(),
    //     model_name: "firefunction-v2".into(),
    //     model_provider_name: "fireworks".into(),
    //     credentials: HashMap::new(),
    // }];
    let tool_providers = vec![];

    let json_providers = vec![
        E2ETestProvider {
            variant_name: "fireworks".to_string(),
            model_name: "llama3.1-8b-instruct-fireworks".into(),
            model_provider_name: "fireworks".into(),
            credentials: HashMap::new(),
        },
        // E2ETestProvider {
        //     variant_name: "fireworks-implicit".to_string(),
        //     model_name: "firefunction-v2".into(),
        //     model_provider_name: "fireworks".into(),
        //     credentials: HashMap::new(),
        // },
    ];

    #[cfg(feature = "e2e_tests")]
    let shorthand_providers = vec![E2ETestProvider {
        variant_name: "fireworks-shorthand".to_string(),
        model_name: "fireworks::accounts/fireworks/models/llama-v3p1-8b-instruct".into(),
        model_provider_name: "fireworks".into(),
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
        #[cfg(feature = "e2e_tests")]
        shorthand_inference: shorthand_providers,
        #[cfg(feature = "batch_tests")]
        supports_batch_inference: false,
    }
}
