use secrecy::SecretString;
use std::env;

use gateway::inference::providers::mistral::MistralProvider;
use gateway::model::ProviderConfig;

use crate::providers::common::IntegrationTestProviders;

crate::generate_provider_tests!(get_providers);

async fn get_providers() -> IntegrationTestProviders {
    let provider = Box::leak(Box::new(get_provider()));

    IntegrationTestProviders {
        simple_inference: vec![provider],
        simple_streaming_inference: vec![provider],
        tool_use_inference: vec![provider],
        tool_use_streaming_inference: vec![provider],
        tool_multi_turn_inference: vec![provider],
        tool_multi_turn_streaming_inference: vec![provider],
        json_mode_inference: vec![provider],
        json_mode_streaming_inference: vec![provider],
    }
}

/// Get a generic provider for testing
fn get_provider() -> ProviderConfig {
    let api_key = env::var("MISTRAL_API_KEY").expect("MISTRAL_API_KEY must be set");
    let api_key = Some(SecretString::new(api_key));
    let model_name = "open-mistral-nemo-2407".to_string();

    ProviderConfig::Mistral(MistralProvider {
        model_name,
        api_key,
    })
}
