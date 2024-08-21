use secrecy::SecretString;
use std::env;

use gateway::inference::providers::azure::AzureProvider;
use gateway::model::ProviderConfig;

use crate::providers::common::IntegrationTestProviders;

crate::generate_provider_tests!(get_providers);

async fn get_providers() -> IntegrationTestProviders {
    // Generic provider for testing
    let api_key = env::var("AZURE_OPENAI_API_KEY")
        .expect("Environment variable AZURE_OPENAI_API_KEY must be set");
    let api_base = env::var("AZURE_OPENAI_API_BASE")
        .expect("Environment variable AZURE_OPENAI_API_BASE must be set");
    let deployment_id = env::var("AZURE_OPENAI_DEPLOYMENT_ID")
        .expect("Environment variable AZURE_OPENAI_DEPLOYMENT_ID must be set");
    let api_key = Some(SecretString::new(api_key));
    let model_name = "gpt-4o-mini".to_string();

    let provider = ProviderConfig::Azure(AzureProvider {
        model_name,
        api_base,
        api_key,
        deployment_id,
    });

    IntegrationTestProviders::with_provider(provider)
}
