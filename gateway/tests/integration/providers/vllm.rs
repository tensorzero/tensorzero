use secrecy::SecretString;
use std::env;

use gateway::inference::providers::vllm::VLLMProvider;
use gateway::model::ProviderConfig;

use super::common::IntegrationTestProviders;

crate::generate_provider_tests!(get_providers);

/// Get a generic provider for testing
async fn get_providers() -> IntegrationTestProviders {
    let provider = Box::leak(Box::new(get_provider()));

    // TODOs (#169): support tool use and tool result inference
    IntegrationTestProviders {
        simple_inference: vec![provider],
        tool_use_inference: vec![],
        tool_multi_turn_inference: vec![],
        json_mode_inference: vec![provider],
    }
}

fn get_provider() -> ProviderConfig {
    let api_key = env::var("VLLM_API_KEY").expect("VLLM_API_KEY must be set");
    let api_key = Some(SecretString::new(api_key));
    let api_base = env::var("VLLM_API_BASE").expect("VLLM_API_BASE must be set");
    let model_name = env::var("VLLM_MODEL_NAME").expect("VLLM_MODEL_NAME must be set");

    ProviderConfig::VLLM(VLLMProvider {
        model_name,
        api_key,
        api_base,
    })
}
