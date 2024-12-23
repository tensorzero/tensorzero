use crate::providers::common::{E2ETestProvider, E2ETestProviders};

#[cfg(feature = "e2e_tests")]
crate::generate_provider_tests!(get_providers);
#[cfg(feature = "batch_tests")]
crate::generate_batch_inference_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let standard_providers = vec![E2ETestProvider {
        variant_name: "together".to_string(),
        model_name: "llama3.1-8b-instruct-together".to_string(),
        model_provider_name: "together".to_string(),
    }];

    let json_providers = vec![
        E2ETestProvider {
            variant_name: "together".to_string(),
            model_name: "llama3.1-8b-instruct-together".to_string(),
            model_provider_name: "together".to_string(),
        },
        // TODOs (#80): see below
        // E2ETestProvider {
        //     variant_name: "together-implicit".to_string(),
        // },
    ];

    let shorthand_providers = vec![E2ETestProvider {
        variant_name: "together-shorthand".to_string(),
        model_name: "together::meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo".to_string(),
        model_provider_name: "together".to_string(),
    }];

    // TODOs (#80):
    // - Together seems to have a different format for tool use responses compared to OpenAI (breaking)
    // - Together's function calling for Llama 3.1 is different from Llama 3.0 (breaking) - we should test both
    E2ETestProviders {
        simple_inference: standard_providers.clone(),
        inference_params_inference: standard_providers.clone(),
        tool_use_inference: vec![],
        tool_multi_turn_inference: vec![],
        dynamic_tool_use_inference: vec![],
        parallel_tool_use_inference: vec![],
        json_mode_inference: json_providers.clone(),
        shorthand_inference: shorthand_providers.clone(),
        #[cfg(feature = "batch_tests")]
        supports_batch_inference: false,
    }
}
