use crate::providers::common::{E2ETestProvider, E2ETestProviders};

crate::generate_provider_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let providers = vec![E2ETestProvider {
        variant_name: "mistral".to_string(),
        model_name: "open-mistral-nemo-2407".to_string(),
        model_provider_name: "mistral".to_string(),
    }];
    E2ETestProviders {
        simple_inference: providers.clone(),
        streaming_inference: providers.clone(),
        tool_use_inference: providers.clone(),
        tool_use_streaming_inference: providers.clone(),
        tool_multi_turn_inference: providers.clone(),
        tool_multi_turn_streaming_inference: providers.clone(),
        dynamic_tool_use_inference: providers.clone(),
        dynamic_tool_use_streaming_inference: providers.clone(),
        json_mode_inference: providers.clone(),
        json_mode_streaming_inference: providers.clone(),
    }
}
