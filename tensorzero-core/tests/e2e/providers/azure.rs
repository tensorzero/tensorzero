use std::collections::HashMap;

use crate::providers::common::{E2ETestProvider, E2ETestProviders};

use super::common::{EmbeddingTestProvider, ModelTestProvider};

crate::generate_provider_tests!(get_providers);
crate::generate_batch_inference_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let credentials = match std::env::var("AZURE_OPENAI_API_KEY") {
        Ok(key) => HashMap::from([("azure_openai_api_key".to_string(), key)]),
        Err(_) => HashMap::new(),
    };

    let standard_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "azure".to_string(),
        model_name: "gpt-4o-mini-azure".into(),
        model_provider_name: "azure".into(),
        credentials: HashMap::new(),
    }];

    let extra_body_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "azure-extra-body".to_string(),
        model_name: "gpt-4o-mini-azure".into(),
        model_provider_name: "azure".into(),
        credentials: HashMap::new(),
    }];

    let bad_auth_extra_headers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "azure-extra-headers".to_string(),
        model_name: "gpt-4o-mini-azure".into(),
        model_provider_name: "azure".into(),
        credentials: HashMap::new(),
    }];

    let inference_params_dynamic_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "azure-dynamic".to_string(),
        model_name: "gpt-4o-mini-azure-dynamic".into(),
        model_provider_name: "azure".into(),
        credentials,
    }];

    let json_providers = vec![
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "azure".to_string(),
            model_name: "gpt-4o-mini-azure".into(),
            model_provider_name: "azure".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "azure-implicit".to_string(),
            model_name: "gpt-4o-mini-azure".into(),
            model_provider_name: "azure".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "azure-strict".to_string(),
            model_name: "gpt-4o-mini-azure".into(),
            model_provider_name: "azure".into(),
            credentials: HashMap::new(),
        },
    ];

    let json_mode_off_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "azure_json_mode_off".to_string(),
        model_name: "gpt-4o-mini-azure".into(),
        model_provider_name: "azure".into(),
        credentials: HashMap::new(),
    }];

    let embedding_providers = vec![EmbeddingTestProvider {
        model_name: "azure-text-embedding-3-small".into(),
        dimensions: 1536,
    }];

    // azure requires deployment_id and endpoint parameters, so it can't be tested with just default credentials
    let provider_type_default_credentials_providers = vec![];

    let input_audio_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "azure".to_string(),
        model_name: "azure-gpt-4o-audio-preview".into(),
        model_provider_name: "azure".into(),
        credentials: HashMap::new(),
    }];

    let credential_fallbacks = vec![ModelTestProvider {
        provider_type: "azure".to_string(),
        model_info: HashMap::from([
            (
                "deployment_id".to_string(),
                "gpt4o-mini-20240718".to_string(),
            ),
            (
                "endpoint".to_string(),
                "https://t0-azure-openai-east.openai.azure.com".to_string(),
            ),
        ]),
        use_modal_headers: false,
    }];

    E2ETestProviders {
        simple_inference: standard_providers.clone(),
        extra_body_inference: extra_body_providers,
        bad_auth_extra_headers,
        reasoning_inference: vec![],
        embeddings: embedding_providers,
        inference_params_inference: standard_providers.clone(),
        inference_params_dynamic_credentials: inference_params_dynamic_providers,
        provider_type_default_credentials: provider_type_default_credentials_providers,
        provider_type_default_credentials_shorthand: vec![],
        tool_use_inference: standard_providers.clone(),
        tool_multi_turn_inference: standard_providers.clone(),
        dynamic_tool_use_inference: standard_providers.clone(),
        parallel_tool_use_inference: vec![],
        json_mode_inference: json_providers.clone(),
        json_mode_off_inference: json_mode_off_providers.clone(),
        image_inference: vec![],
        pdf_inference: vec![],
        input_audio: input_audio_providers,
        shorthand_inference: vec![],
        credential_fallbacks,
    }
}
