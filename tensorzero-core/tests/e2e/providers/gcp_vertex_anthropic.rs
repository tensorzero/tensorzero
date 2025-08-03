use std::collections::HashMap;

use crate::providers::common::{E2ETestProvider, E2ETestProviders};

crate::generate_provider_tests!(get_providers);
crate::generate_batch_inference_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let standard_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "gcp-vertex-haiku".to_string(),
        model_name: "claude-3-haiku-20240307-gcp-vertex".into(),
        model_provider_name: "gcp_vertex_anthropic".into(),
        credentials: HashMap::new(),
    }];

    let image_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "gcp-vertex-haiku".to_string(),
        model_name: "claude-3-haiku-20240307-gcp-vertex".into(),
        model_provider_name: "gcp_vertex_anthropic".into(),
        credentials: HashMap::new(),
    }];

    let extra_body_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "gcp-vertex-haiku-extra-body".to_string(),
        model_name: "claude-3-haiku-20240307-gcp-vertex".into(),
        model_provider_name: "gcp_vertex_anthropic".into(),
        credentials: HashMap::new(),
    }];

    let bad_auth_extra_headers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "gcp-vertex-haiku-extra-headers".to_string(),
        model_name: "claude-3-haiku-20240307-gcp-vertex".into(),
        model_provider_name: "gcp_vertex_anthropic".into(),
        credentials: HashMap::new(),
    }];

    let json_providers = vec![
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "gcp-vertex-haiku".to_string(),
            model_name: "claude-3-haiku-20240307-gcp-vertex".into(),
            model_provider_name: "gcp_vertex_anthropic".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "gcp-vertex-haiku-implicit".to_string(),
            model_name: "claude-3-haiku-20240307-gcp-vertex".into(),
            model_provider_name: "gcp_vertex_anthropic".into(),
            credentials: HashMap::new(),
        },
        E2ETestProvider {
            supports_batch_inference: false,
            variant_name: "gcp-vertex-haiku-strict".to_string(),
            model_name: "claude-3-haiku-20240307-gcp-vertex".into(),
            model_provider_name: "gcp_vertex_anthropic".into(),
            credentials: HashMap::new(),
        },
    ];

    let json_mode_off_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "gcp_vertex_haiku_json_mode_off".to_string(),
        model_name: "claude-3-haiku-20240307-gcp-vertex".into(),
        model_provider_name: "gcp_vertex_anthropic".into(),
        credentials: HashMap::new(),
    }];

    let shorthand_providers = vec![E2ETestProvider {
        supports_batch_inference: false,
        variant_name: "gcp_vertex_anthropic_shorthand".to_string(),
        model_name: "gcp_vertex_anthropic::projects/tensorzero-public/locations/us-central1/publishers/anthropic/models/claude-3-haiku@20240307".into(),
        model_provider_name: "gcp_vertex_anthropic".into(),
        credentials: HashMap::new(),
    }];

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
        json_mode_off_inference: json_mode_off_providers.clone(),
        image_inference: image_providers,
        pdf_inference: vec![],
        shorthand_inference: shorthand_providers,
    }
}
