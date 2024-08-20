use secrecy::SecretString;
use std::env;

use gateway::inference::providers::together::TogetherProvider;
use gateway::model::ProviderConfig;

use crate::providers::common::TestProviders;

crate::generate_provider_tests!(get_providers);

async fn get_providers() -> TestProviders {
    let provider = Box::leak(Box::new(get_provider()));
    let provider_tool_use = Box::leak(Box::new(get_provider_tool_use()));

    // TODO (#80): Together's function calling format seems slightly different from OpenAI (breaking)
    TestProviders {
        simple_inference: vec![provider],
        streaming_inference: vec![provider],
        tool_use_inference: vec![provider_tool_use],
        tool_use_streaming_inference: vec![provider_tool_use],
        tool_result_inference: vec![],
        tool_result_streaming_inference: vec![],
        json_mode_inference: vec![provider],
        json_mode_streaming_inference: vec![provider],
    }
}

/// Get a generic provider for testing
fn get_provider() -> ProviderConfig {
    // Get a generic provider for testing
    let api_key = env::var("TOGETHER_API_KEY").expect("TOGETHER_API_KEY must be set");
    let api_key = Some(SecretString::new(api_key));
    let model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo".to_string();

    ProviderConfig::Together(TogetherProvider {
        model_name,
        api_key,
    })
}

// Get a generic provider for tool use
// TODO (#80): Together has a different function calling format for Llama 3.
fn get_provider_tool_use() -> ProviderConfig {
    // Get a generic provider for testing
    let api_key = env::var("TOGETHER_API_KEY").expect("TOGETHER_API_KEY must be set");
    let api_key = Some(SecretString::new(api_key));
    let model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1".to_string();

    ProviderConfig::Together(TogetherProvider {
        model_name,
        api_key,
    })
}
