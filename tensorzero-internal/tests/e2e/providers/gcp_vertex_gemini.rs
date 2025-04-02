use std::collections::HashMap;

use crate::providers::common::{E2ETestProvider, E2ETestProviders};

crate::generate_provider_tests!(get_providers);
crate::generate_batch_inference_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let standard_providers = vec![
        E2ETestProvider {
            variant_name: "gcp-vertex-gemini-flash".to_string(),
            model_name: "gemini-2.0-flash-001".into(),
            model_provider_name: "gcp_vertex_gemini".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            variant_name: "gcp-vertex-gemini-pro".to_string(),
            model_name: "gemini-1.5-pro-001".into(),
            model_provider_name: "gcp_vertex_gemini".into(),
            credentials: HashMap::new(),
        },
    ];

    let image_providers = vec![E2ETestProvider {
        variant_name: "gcp_vertex".to_string(),
        model_name: "gemini-1.5-pro-001".into(),
        model_provider_name: "gcp_vertex_gemini".into(),
        credentials: HashMap::new(),
    }];

    let extra_body_providers = vec![E2ETestProvider {
        variant_name: "gcp-vertex-gemini-flash-extra-body".to_string(),
        model_name: "gemini-2.0-flash-001".into(),
        model_provider_name: "gcp_vertex_gemini".into(),
        credentials: HashMap::new(),
    }];

    let bad_auth_extra_headers = vec![E2ETestProvider {
        variant_name: "gcp-vertex-gemini-flash-extra-headers".to_string(),
        model_name: "gemini-2.0-flash-001".into(),
        model_provider_name: "gcp_vertex_gemini".into(),
        credentials: HashMap::new(),
    }];

    let json_providers = vec![
        E2ETestProvider {
            variant_name: "gcp-vertex-gemini-flash".to_string(),
            model_name: "gemini-2.0-flash-001".into(),
            model_provider_name: "gcp_vertex_gemini".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            variant_name: "gcp-vertex-gemini-flash-implicit".to_string(),
            model_name: "gemini-2.0-flash-001".into(),
            model_provider_name: "gcp_vertex_gemini".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            variant_name: "gcp-vertex-gemini-pro".to_string(),
            model_name: "gemini-1.5-pro-001".into(),
            model_provider_name: "gcp_vertex_gemini".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            variant_name: "gcp-vertex-gemini-pro-implicit".to_string(),
            model_name: "gemini-1.5-pro-001".into(),
            model_provider_name: "gcp_vertex_gemini".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            variant_name: "gcp-vertex-gemini-flash-default".to_string(),
            model_name: "gemini-2.0-flash-001".into(),
            model_provider_name: "gcp_vertex_gemini".into(),
            credentials: HashMap::new(),
        },
    ];

    E2ETestProviders {
        simple_inference: standard_providers.clone(),
        extra_body_inference: extra_body_providers,
        bad_auth_extra_headers,
        reasoning_inference: vec![],
        inference_params_inference: standard_providers.clone(),
        inference_params_dynamic_credentials: vec![],
        tool_use_inference: standard_providers.clone(),
        tool_multi_turn_inference: standard_providers.clone(),
        dynamic_tool_use_inference: standard_providers.clone(),
        parallel_tool_use_inference: vec![],
        json_mode_inference: json_providers.clone(),
        image_inference: image_providers,

        shorthand_inference: vec![],
        supports_batch_inference: false,
    }
}
