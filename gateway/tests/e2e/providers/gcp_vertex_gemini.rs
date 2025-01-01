use std::collections::HashMap;

use crate::providers::common::{E2ETestProvider, E2ETestProviders};

crate::generate_provider_tests!(get_providers);
crate::generate_batch_inference_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let mut map = HashMap::new();
    if let Ok(credential_path) = std::env::var("GCP_VERTEX_CREDENTIALS_PATH") {
        map.insert("GCP_VERTEX_CREDENTIALS_PATH".to_string(), credential_path);
    }
    let credentials = if map.is_empty() { None } else { Some(map) };

    let standard_providers = vec![
        E2ETestProvider {
            variant_name: "gcp-vertex-gemini-flash".to_string(),
            model_name: "gemini-1.5-flash-001".to_string(),
            model_provider_name: "gcp_vertex_gemini".to_string(),
            credentials: None,
        },
        E2ETestProvider {
            variant_name: "gcp-vertex-gemini-pro".to_string(),
            model_name: "gemini-1.5-pro-001".to_string(),
            model_provider_name: "gcp_vertex_gemini".to_string(),
            credentials: None,
        },
    ];

    let inference_params_providers = vec![
        E2ETestProvider {
            variant_name: "gcp-vertex-gemini-flash".to_string(),
            model_name: "gemini-1.5-flash-001".to_string(),
            model_provider_name: "gcp_vertex_gemini".to_string(),
            credentials: credentials.clone(),
        },
        E2ETestProvider {
            variant_name: "gcp-vertex-gemini-pro".to_string(),
            model_name: "gemini-1.5-pro-001".to_string(),
            model_provider_name: "gcp_vertex_gemini".to_string(),
            credentials,
        },
    ];

    let json_providers = vec![
        E2ETestProvider {
            variant_name: "gcp-vertex-gemini-flash".to_string(),
            model_name: "gemini-1.5-flash-001".to_string(),
            model_provider_name: "gcp_vertex_gemini".to_string(),
            credentials: None,
        },
        E2ETestProvider {
            variant_name: "gcp-vertex-gemini-flash-implicit".to_string(),
            model_name: "gemini-1.5-flash-001".to_string(),
            model_provider_name: "gcp_vertex_gemini".to_string(),
            credentials: None,
        },
        E2ETestProvider {
            variant_name: "gcp-vertex-gemini-pro".to_string(),
            model_name: "gemini-1.5-pro-001".to_string(),
            model_provider_name: "gcp_vertex_gemini".to_string(),
            credentials: None,
        },
        E2ETestProvider {
            variant_name: "gcp-vertex-gemini-pro-implicit".to_string(),
            model_name: "gemini-1.5-pro-001".to_string(),
            model_provider_name: "gcp_vertex_gemini".to_string(),
            credentials: None,
        },
    ];

    E2ETestProviders {
        simple_inference: standard_providers.clone(),
        inference_params_inference: inference_params_providers,
        tool_use_inference: standard_providers.clone(),
        tool_multi_turn_inference: standard_providers.clone(),
        dynamic_tool_use_inference: standard_providers.clone(),
        parallel_tool_use_inference: vec![],
        json_mode_inference: json_providers.clone(),
        shorthand_inference: vec![],
        supports_batch_inference: false,
    }
}
