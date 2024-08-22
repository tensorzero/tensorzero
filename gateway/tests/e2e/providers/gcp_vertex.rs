use crate::providers::common::{E2ETestProvider, E2ETestProviders};

crate::generate_provider_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let provider_flash = E2ETestProvider {
        variant_name: "gcp-vertex-gemini-flash".to_string(),
    };

    let provider_pro = E2ETestProvider {
        variant_name: "gcp-vertex-gemini-pro".to_string(),
    };

    E2ETestProviders::with_providers(vec![provider_flash, provider_pro])
}
