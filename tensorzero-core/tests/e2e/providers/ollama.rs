use std::collections::HashMap;

use crate::providers::common::{E2ETestProvider, E2ETestProviders};

use super::common::ModelTestProvider;

crate::generate_provider_tests!(get_providers);
crate::generate_batch_inference_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let ollama_available = reqwest::Client::new()
        .get("http://ollama:11434/api/tags")
        .send()
        .await
        .is_ok();

    let standard_providers = if ollama_available {
        vec![E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "qwen2-0.5b-ollama".to_string(),
            model_name: "qwen2-0.5b-ollama".to_string(),
            model_provider_name: "ollama".to_string(),
            credentials: HashMap::new(),
        }]
    } else {
        vec![]
    };

    let json_providers = if ollama_available {
        vec![E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "qwen2-0.5b-ollama".to_string(),
            model_name: "qwen2-0.5b-ollama".to_string(),
            model_provider_name: "ollama".to_string(),
            credentials: HashMap::new(),
        }]
    } else {
        vec![]
    };

    let credential_fallbacks = if ollama_available {
        vec![ModelTestProvider {
            provider_type: "ollama".to_string(),
            model_info: HashMap::from([
                ("model_name".to_string(), "qwen2:0.5b".to_string()),
                ("api_base".to_string(), "http://ollama:11434".to_string()),
            ]),
            use_modal_headers: false,
        }]
    } else {
        vec![]
    };

    let shorthand_providers = if ollama_available {
        vec![E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "ollama-shorthand".to_string(),
            model_name: "ollama::qwen2:0.5b".to_string(),
            model_provider_name: "ollama".to_string(),
            credentials: HashMap::new(),
        }]
    } else {
        vec![]
    };

    E2ETestProviders {
        simple_inference: standard_providers.clone(),
        extra_body_inference: vec![],
        bad_auth_extra_headers: vec![],
        reasoning_inference: vec![],
        embeddings: vec![],
        inference_params_inference: standard_providers.clone(),
        inference_params_dynamic_credentials: vec![],
        provider_type_default_credentials: standard_providers,
        provider_type_default_credentials_shorthand: shorthand_providers.clone(),
        tool_use_inference: vec![],
        tool_multi_turn_inference: vec![],
        dynamic_tool_use_inference: vec![],
        parallel_tool_use_inference: vec![],
        json_mode_inference: json_providers,
        json_mode_off_inference: vec![],
        image_inference: vec![],
        pdf_inference: vec![],
        input_audio: vec![],
        shorthand_inference: shorthand_providers,
        credential_fallbacks,
    }
}
