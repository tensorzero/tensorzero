use crate::providers::common::{E2ETestProvider, E2ETestProviders};

crate::generate_provider_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let providers = vec![E2ETestProvider {
        variant_name: "vllm".to_string(),
        model_name: "microsoft/Phi-3.5-mini-instruct".to_string(),
        model_provider_name: "vllm".to_string(),
    }];

    // TODOs (#169): Implement a solution for vLLM tool use
    E2ETestProviders {
        simple_inference: providers.clone(),
        streaming_inference: providers.clone(),
        tool_use_inference: vec![],
        tool_use_streaming_inference: vec![],
        tool_multi_turn_inference: vec![],
        tool_multi_turn_streaming_inference: vec![],
        json_mode_inference: providers.clone(),
        json_mode_streaming_inference: providers,
    }
}
