use secrecy::SecretString;
use std::env;

use gateway::{inference::providers::anthropic::AnthropicProvider, model::ProviderConfig};

use crate::providers::common::TestProviders;

crate::generate_provider_tests!(get_providers);

async fn get_providers() -> TestProviders {
    // Generic provider for testing
    let api_key = env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY must be set");
    let api_key = Some(SecretString::new(api_key));

    let provider = ProviderConfig::Anthropic(AnthropicProvider {
        model_name: "claude-3-haiku-20240307".to_string(),
        api_key,
    });

    TestProviders::with_provider(provider)
}
