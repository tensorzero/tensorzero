use crate::providers::common::{E2ETestProvider, E2ETestProviders};

crate::generate_provider_tests!(get_providers);
crate::generate_batch_inference_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let standard_providers = vec![
        E2ETestProvider {
            variant_name: "google-ai-studio-gemini-flash-8b".to_string(),
            model_name: "gemini-1.5-flash-8b".to_string(),
            model_provider_name: "google_ai_studio_gemini".to_string(),
        },
        E2ETestProvider {
            variant_name: "google-ai-studio-gemini-pro-002".to_string(),
            model_name: "gemini-1.5-pro-002".to_string(),
            model_provider_name: "google_ai_studio_gemini".to_string(),
        },
    ];

    let tool_providers = vec![E2ETestProvider {
        variant_name: "google-ai-studio-gemini-flash-8b".to_string(),
        model_name: "gemini-1.5-flash-8b".to_string(),
        model_provider_name: "google_ai_studio_gemini".to_string(),
    }];

    let json_providers = vec![
        E2ETestProvider {
            variant_name: "google-ai-studio-gemini-flash-8b".to_string(),
            model_name: "gemini-1.5-flash-8b".to_string(),
            model_provider_name: "google_ai_studio_gemini".to_string(),
        },
        E2ETestProvider {
            variant_name: "google-ai-studio-gemini-flash-8b-implicit".to_string(),
            model_name: "gemini-1.5-flash-8b".to_string(),
            model_provider_name: "google_ai_studio_gemini".to_string(),
        },
        E2ETestProvider {
            variant_name: "google-ai-studio-gemini-pro-002".to_string(),
            model_name: "gemini-1.5-pro-002".to_string(),
            model_provider_name: "google_ai_studio_gemini".to_string(),
        },
        E2ETestProvider {
            variant_name: "google-ai-studio-gemini-pro-002-implicit".to_string(),
            model_name: "gemini-1.5-pro-002".to_string(),
            model_provider_name: "google_ai_studio_gemini".to_string(),
        },
    ];

    let shorthand_providers = vec![E2ETestProvider {
        variant_name: "google-ai-studio-gemini-flash-8b-shorthand".to_string(),
        model_name: "google_ai_studio_gemini::gemini-1.5-flash-8b".to_string(),
        model_provider_name: "google_ai_studio_gemini".to_string(),
    }];

    E2ETestProviders {
        simple_inference: standard_providers.clone(),
        inference_params_inference: standard_providers.clone(),
        tool_use_inference: tool_providers.clone(),
        tool_multi_turn_inference: tool_providers.clone(),
        dynamic_tool_use_inference: tool_providers.clone(),
        parallel_tool_use_inference: vec![],
        json_mode_inference: json_providers.clone(),
        shorthand_inference: shorthand_providers.clone(),
        supports_batch_inference: false,
    }
}
