use secrecy::SecretString;
use std::env;

use gateway::inference::providers::azure::AzureProvider;
use gateway::model::ProviderConfig;

use crate::providers::common::TestableProviderConfig;

crate::generate_provider_tests!(AzureProvider);

impl TestableProviderConfig for AzureProvider {
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
    let api_key = env::var("AZURE_OPENAI_API_KEY")
        .expect("Environment variable AZURE_OPENAI_API_KEY must be set");
    let api_base = env::var("AZURE_OPENAI_API_BASE")
        .expect("Environment variable AZURE_OPENAI_API_BASE must be set");
    let deployment_id = env::var("AZURE_OPENAI_DEPLOYMENT_ID")
        .expect("Environment variable AZURE_OPENAI_DEPLOYMENT_ID must be set");
    let api_key = Some(SecretString::new(api_key));
    let model_name = "gpt-4o-mini".to_string();

    ProviderConfig::Azure(AzureProvider {
        model_name,
        api_base,
        api_key,
        deployment_id,
    })
}
