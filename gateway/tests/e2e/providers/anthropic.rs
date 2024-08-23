use crate::providers::common::{E2ETestProvider, E2ETestProviders};

crate::generate_provider_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let standard_providers = vec![E2ETestProvider {
        variant_name: "anthropic".to_string(),
    }];

    let json_providers = vec![
        E2ETestProvider {
            variant_name: "anthropic".to_string(),
        },
        E2ETestProvider {
            variant_name: "anthropic-implicit".to_string(),
        },
    ];

    E2ETestProviders {
        simple_inference: standard_providers.clone(),
        streaming_inference: standard_providers.clone(),
        tool_use_inference: standard_providers.clone(),
        tool_use_streaming_inference: standard_providers.clone(),
        tool_multi_turn_inference: standard_providers.clone(),
        tool_multi_turn_streaming_inference: standard_providers.clone(),
        dynamic_tool_use_inference: standard_providers.clone(),
        dynamic_tool_use_streaming_inference: standard_providers.clone(),
        json_mode_inference: json_providers.clone(),
        json_mode_streaming_inference: json_providers,
    }
}
