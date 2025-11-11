/// This file contains test cases that target the `azure` provider against Azure AI Foundry models instead of Azure OpenAI Service models.
use std::collections::HashMap;

use crate::providers::common::{E2ETestProvider, E2ETestProviders};

crate::generate_provider_tests!(get_providers);
crate::generate_batch_inference_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let credentials = match std::env::var("AZURE_AI_FOUNDRY_API_KEY") {
        Ok(k) => HashMap::from([("azure_ai_foundry_api_key".to_string(), k)]),
        Err(_) => HashMap::new(),
    };

    let standard_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "azure-ai-foundry".to_string(),
        model_name: "llama-3.3-70b-instruct-azure".into(),
        model_provider_name: "azure".into(),
        credentials: HashMap::new(),
    }];

    let extra_body_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "azure-ai-foundry-extra-body".to_string(),
        model_name: "llama-3.3-70b-instruct-azure".into(),
        model_provider_name: "azure".into(),
        credentials: HashMap::new(),
    }];

    let bad_auth_extra_headers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "azure-ai-foundry-extra-headers".to_string(),
        model_name: "llama-3.3-70b-instruct-azure".into(),
        model_provider_name: "azure".into(),
        credentials: HashMap::new(),
    }];

    let inference_params_dynamic_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "azure-ai-foundry-dynamic".to_string(),
        model_name: "llama-3.3-70b-instruct-azure-dynamic".into(),
        model_provider_name: "azure".into(),
        credentials,
    }];

    let json_mode_inference = vec![
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "azure-ai-foundry".to_string(), // json_mode=on
            model_name: "llama-3.3-70b-instruct-azure".into(),
            model_provider_name: "azure".into(),
            credentials: HashMap::new(),
        },
        // NB: As of 2025-09-04, Azure AI Foundry doesn't support tool calling for Llama models.
        // TODO: Use `grok-3-mini` model instead for this test case (supports tool calling)
        // E2ETestProvider {
        //     supports_batch_inference: false,
        //     variant_name: "azure-ai-foundry-implicit".to_string(),
        //     model_name: "llama-3.3-70b-instruct-azure".into(),
        //     model_provider_name: "azure".into(),
        //     credentials: HashMap::new(),
        // },
        // NB: As of 2025-09-04, Azure AI Foundry only supports strict mode for OpenAI models
        // E2ETestProvider {
        //     supports_batch_inference: false,
        //     variant_name: "azure-ai-foundry-strict".to_string(),
        //     model_name: "llama-3.3-70b-instruct-azure".into(),
        //     model_provider_name: "azure".into(),
        //     credentials: HashMap::new(),
        // },
    ];

    let json_mode_off_inference = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "azure-ai-foundry_json_mode_off".to_string(),
        model_name: "llama-3.3-70b-instruct-azure".into(),
        model_provider_name: "azure".into(),
        credentials: HashMap::new(),
    }];

    // TODO: Use `grok-3-mini` to run the tool use tests.

    // azure requires deployment_id and endpoint parameters, so it can't be tested with just default credentials
    let provider_type_default_credentials_providers = vec![];

    E2ETestProviders {
        simple_inference: standard_providers.clone(),
        extra_body_inference: extra_body_providers,
        bad_auth_extra_headers,
        reasoning_inference: vec![],
        embeddings: vec![],
        inference_params_inference: standard_providers.clone(),
        inference_params_dynamic_credentials: inference_params_dynamic_providers,
        provider_type_default_credentials: provider_type_default_credentials_providers,
        provider_type_default_credentials_shorthand: vec![],
        tool_use_inference: vec![],
        tool_multi_turn_inference: vec![],
        dynamic_tool_use_inference: vec![],
        parallel_tool_use_inference: vec![],
        json_mode_inference,
        json_mode_off_inference,
        image_inference: vec![],
        pdf_inference: vec![],
        input_audio: vec![],
        shorthand_inference: vec![],
        // Azure AI foundry uses same code as Azure regular for fallbacks so we don't
        // need to double test
        credential_fallbacks: vec![],
    }
}
