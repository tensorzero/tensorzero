use crate::providers::common::{E2ETestProvider, E2ETestProviders};

crate::generate_provider_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let provider = E2ETestProvider {
        variant_name: "fireworks".to_string(),
    };

    let provider_firefunction = E2ETestProvider {
        variant_name: "fireworks-firefunction".to_string(),
    };

    E2ETestProviders {
        simple_inference: vec![provider.clone()],
        streaming_inference: vec![provider.clone()],
        tool_use_inference: vec![provider.clone()],
        tool_use_streaming_inference: vec![provider.clone()],
        tool_multi_turn_inference: vec![provider.clone()],
        tool_multi_turn_streaming_inference: vec![provider.clone()],
        dynamic_tool_use_inference: vec![provider_firefunction.clone()],
        dynamic_tool_use_streaming_inference: vec![provider_firefunction],
        json_mode_inference: vec![provider.clone()],
        json_mode_streaming_inference: vec![provider],
    }
}
