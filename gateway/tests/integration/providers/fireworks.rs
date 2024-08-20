use secrecy::SecretString;
use std::env;

use gateway::inference::providers::fireworks::FireworksProvider;
use gateway::model::ProviderConfig;

use crate::providers::common::TestableProviderConfig;

crate::generate_provider_tests!(FireworksProvider);

impl TestableProviderConfig for FireworksProvider {
    async fn get_simple_inference_request_provider() -> Option<ProviderConfig> {
        Some(get_provider())
    }

    async fn get_streaming_inference_request_provider() -> Option<ProviderConfig> {
        Some(get_provider())
    }

    async fn get_tool_use_inference_request_provider() -> Option<ProviderConfig> {
        Some(get_provider_tool_use())
    }

    async fn get_tool_use_streaming_inference_request_provider() -> Option<ProviderConfig> {
        Some(get_provider_tool_use())
    }

    async fn get_json_mode_inference_request_provider() -> Option<ProviderConfig> {
        Some(get_provider())
    }

    async fn get_json_mode_streaming_inference_request_provider() -> Option<ProviderConfig> {
        Some(get_provider())
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
