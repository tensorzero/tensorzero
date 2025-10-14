use std::collections::HashMap;

use crate::providers::common::{E2ETestProvider, E2ETestProviders};

crate::generate_provider_tests!(get_providers);
crate::generate_batch_inference_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let credentials = match std::env::var("AZURE_OPENAI_API_KEY") {
        Ok(k) => HashMap::from([("azure_openai_api_key".to_string(), k)]),
        Err(_) => HashMap::new(),
    };

    let standard_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "azure-llama".to_string(),
        model_name: "llama-3.3-70b-instruct-azure".into(),
        model_provider_name: "azure".into(),
        credentials: HashMap::new(),
    }];

    let extra_body_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "azure-llama-extra-body".to_string(),
        model_name: "llama-3.3-70b-instruct-azure".into(),
        model_provider_name: "azure".into(),
        credentials: HashMap::new(),
    }];

    let bad_auth_extra_headers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "azure-llama-extra-headers".to_string(),
        model_name: "llama-3.3-70b-instruct-azure".into(),
        model_provider_name: "azure".into(),
        credentials: HashMap::new(),
    }];

    let inference_params_dynamic_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "azure-llama-dynamic".to_string(),
        model_name: "llama-3.3-70b-instruct-azure-dynamic".into(),
        model_provider_name: "azure".into(),
        credentials,
    }];

    /*
    let json_mode_inference = vec![E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "azure-llama".to_string(), // json_mode=on
            model_name: "llama-3.3-70b-instruct-azure".into(),
            model_provider_name: "azure".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "azure-llama-implicit".to_string(),
            model_name: "llama-3.3-70b-instruct-azure".into(),
            model_provider_name: "azure".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "azure-llama-strict".to_string(),
            model_name: "llama-3.3-70b-instruct-azure".into(),
            model_provider_name: "azure".into(),
            credentials: HashMap::new(),
        },
    ];

    let json_mode_off_inference = vec![
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "azure-llama_json_mode_off".to_string(),
            model_name: "llama-3.3-70b-instruct-azure".into(),
            model_provider_name: "azure".into(),
            credentials: HashMap::new(),
        },
    ];
    */

    E2ETestProviders {
        simple_inference: standard_providers.clone(),
        extra_body_inference: extra_body_providers,
        bad_auth_extra_headers,
        reasoning_inference: vec![],
        embeddings: vec![], // No embeddings for Llama on Azure yet
        inference_params_inference: standard_providers.clone(),
        inference_params_dynamic_credentials: inference_params_dynamic_providers,
        tool_use_inference: vec![],
        tool_multi_turn_inference: vec![],
        dynamic_tool_use_inference: vec![],
        parallel_tool_use_inference: vec![],
        json_mode_inference: vec![],
        json_mode_off_inference: vec![],
        image_inference: vec![],
        pdf_inference: vec![],
        shorthand_inference: vec![],
    }
}
