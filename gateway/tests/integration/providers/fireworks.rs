use secrecy::SecretString;
use std::env;

use gateway::inference::providers::fireworks::FireworksProvider;
use gateway::model::ProviderConfig;

use crate::providers::common::IntegrationTestProviders;

crate::generate_provider_tests!(get_providers);

async fn get_providers() -> IntegrationTestProviders {
    let provider = Box::leak(Box::new(get_provider()));
    let provider_tool_use = Box::leak(Box::new(get_provider_tool_use()));

    IntegrationTestProviders {
        simple_inference: vec![provider],
        simple_streaming_inference: vec![provider],
        tool_use_inference: vec![provider_tool_use],
        tool_use_streaming_inference: vec![provider_tool_use],
        tool_multi_turn_inference: vec![provider_tool_use],
        tool_multi_turn_streaming_inference: vec![provider_tool_use],
        json_mode_inference: vec![provider],
        json_mode_streaming_inference: vec![provider],
    }
}

/// Get a generic provider for testing
fn get_provider() -> ProviderConfig {
    let api_key = env::var("FIREWORKS_API_KEY").expect("FIREWORKS_API_KEY must be set");
    let api_key = Some(SecretString::new(api_key));
    let model_name = "accounts/fireworks/models/llama-v3-8b-instruct".to_string();

    ProviderConfig::Fireworks(FireworksProvider {
        model_name,
        api_key,
    })
}

// Get a generic provider for tool use
// NOTE: Fireworks doesn't support tool use with vanilla Llama 3 8b yet
fn get_provider_tool_use() -> ProviderConfig {
    let api_key = env::var("FIREWORKS_API_KEY").expect("FIREWORKS_API_KEY must be set");
    let api_key = Some(SecretString::new(api_key));
    let model_name = "accounts/fireworks/models/firefunction-v2".to_string();

    ProviderConfig::Fireworks(FireworksProvider {
        model_name,
        api_key,
    })
}
