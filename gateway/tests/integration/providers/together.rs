use secrecy::SecretString;
use std::env;

use gateway::inference::providers::together::TogetherProvider;
use gateway::model::ProviderConfig;

use crate::providers::common::TestProviders;

crate::generate_provider_tests!(get_providers);

async fn get_providers() -> TestProviders {
    // Get a generic provider for testing
    let api_key = env::var("TOGETHER_API_KEY").expect("TOGETHER_API_KEY must be set");
    let api_key = Some(SecretString::new(api_key));
    let model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo".to_string();

    let provider = ProviderConfig::Together(TogetherProvider {
        model_name,
        api_key,
    });

    TestProviders::with_provider(provider)
}
