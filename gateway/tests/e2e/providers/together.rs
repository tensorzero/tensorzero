use crate::providers::common::{E2ETestProvider, E2ETestProviders};

crate::generate_provider_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    E2ETestProviders::with_provider(E2ETestProvider {
        variant_name: "fireworks".to_string(),
    })
}
