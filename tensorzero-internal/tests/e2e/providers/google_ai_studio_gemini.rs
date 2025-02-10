use std::collections::HashMap;

use crate::providers::common::{E2ETestProvider, E2ETestProviders};

#[cfg(feature = "e2e_tests")]
crate::generate_provider_tests!(get_providers);
#[cfg(feature = "batch_tests")]
crate::generate_batch_inference_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let credentials = match std::env::var("GOOGLE_AI_STUDIO_API_KEY") {
        Ok(key) => HashMap::from([("google_ai_studio_api_key".to_string(), key)]),
        Err(_) => HashMap::new(),
    };

    let standard_providers = vec![
        E2ETestProvider {
            variant_name: "google-ai-studio-gemini-flash-8b".to_string(),
            model_name: "gemini-1.5-flash-8b".into(),
            model_provider_name: "google_ai_studio_gemini".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            variant_name: "google-ai-studio-gemini-pro-002".to_string(),
            model_name: "gemini-1.5-pro-002".into(),
            model_provider_name: "google_ai_studio_gemini".into(),
            credentials: HashMap::new(),
        },
    ];

    let inference_params_providers = vec![
        E2ETestProvider {
            variant_name: "google-ai-studio-gemini-flash-8b-dynamic".to_string(),
            model_name: "gemini-1.5-flash-8b-dynamic".into(),
            model_provider_name: "google_ai_studio_gemini".into(),
            credentials: credentials.clone(),
        },
        E2ETestProvider {
            variant_name: "google-ai-studio-gemini-pro-002-dynamic".to_string(),
            model_name: "gemini-1.5-pro-002-dynamic".into(),
            model_provider_name: "google_ai_studio_gemini".into(),
            credentials,
        },
    ];

    let tool_providers = vec![E2ETestProvider {
        variant_name: "google-ai-studio-gemini-flash-8b".to_string(),
        model_name: "gemini-1.5-flash-8b".into(),
        model_provider_name: "google_ai_studio_gemini".into(),
        credentials: HashMap::new(),
    }];

    let json_providers = vec![
        E2ETestProvider {
            variant_name: "google-ai-studio-gemini-flash-8b".to_string(),
            model_name: "gemini-1.5-flash-8b".into(),
            model_provider_name: "google_ai_studio_gemini".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            variant_name: "google-ai-studio-gemini-flash-8b-implicit".to_string(),
            model_name: "gemini-1.5-flash-8b".into(),
            model_provider_name: "google_ai_studio_gemini".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            variant_name: "google-ai-studio-gemini-pro-002".to_string(),
            model_name: "gemini-1.5-pro-002".into(),
            model_provider_name: "google_ai_studio_gemini".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            variant_name: "google-ai-studio-gemini-pro-002-implicit".to_string(),
            model_name: "gemini-1.5-pro-002".into(),
            model_provider_name: "google_ai_studio_gemini".into(),
            credentials: HashMap::new(),
        },
    ];

    #[cfg(feature = "e2e_tests")]
    let shorthand_providers = vec![E2ETestProvider {
        variant_name: "google-ai-studio-gemini-flash-8b-shorthand".to_string(),
        model_name: "google_ai_studio_gemini::gemini-1.5-flash-8b".into(),
        model_provider_name: "google_ai_studio_gemini".into(),
        credentials: HashMap::new(),
    }];

    E2ETestProviders {
        simple_inference: standard_providers.clone(),
        inference_params_inference: inference_params_providers,
        tool_use_inference: tool_providers.clone(),
        tool_multi_turn_inference: tool_providers.clone(),
        dynamic_tool_use_inference: tool_providers.clone(),
        parallel_tool_use_inference: vec![],
        json_mode_inference: json_providers.clone(),
        #[cfg(feature = "e2e_tests")]
        shorthand_inference: shorthand_providers.clone(),
        #[cfg(feature = "batch_tests")]
        supports_batch_inference: false,
    }
}
