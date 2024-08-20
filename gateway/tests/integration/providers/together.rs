use secrecy::SecretString;
use std::env;

use gateway::inference::providers::together::TogetherProvider;
use gateway::model::ProviderConfig;

use crate::providers::common::TestableProviderConfig;

crate::generate_provider_tests!(TogetherProvider);

impl TestableProviderConfig for TogetherProvider {
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

    async fn get_json_mode_inference_request_provider() -> Option<ProviderConfig> {
        Some(get_provider())
    }

    async fn get_json_mode_streaming_inference_request_provider() -> Option<ProviderConfig> {
        Some(get_provider())
    }
}

/// Get a generic provider for testing
fn get_provider() -> ProviderConfig {
    let api_key = env::var("TOGETHER_API_KEY").expect("TOGETHER_API_KEY must be set");
    let api_key = Some(SecretString::new(api_key));
    let model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo".to_string();

    ProviderConfig::Together(TogetherProvider {
        model_name,
        api_key,
    })
}
