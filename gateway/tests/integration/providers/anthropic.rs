use gateway::{inference::providers::anthropic::AnthropicProvider, model::ProviderConfig};
use secrecy::SecretString;
use std::env;

use crate::providers::common::TestableProviderConfig;

crate::enforce_provider_tests!(AnthropicProvider);

impl TestableProviderConfig for AnthropicProvider {
    async fn get_simple_inference_request_provider() -> Option<ProviderConfig> {
        Some(get_provider())
    }

    async fn get_streaming_inference_request_provider() -> Option<ProviderConfig> {
        Some(get_provider())
    }

    async fn get_tool_use_inference_request_provider() -> Option<ProviderConfig> {
        Some(get_provider())
    }

    async fn get_tool_use_streaming_inference_request_provider() -> Option<ProviderConfig> {
        Some(get_provider())
    }
}

/// Get a generic provider for testing
fn get_provider() -> ProviderConfig {
    let api_key = env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY must be set");
    let api_key = Some(SecretString::new(api_key));

    ProviderConfig::Anthropic(AnthropicProvider {
        model_name: "claude-3-haiku-20240307".to_string(),
        api_key,
    })
}
