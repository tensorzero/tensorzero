use crate::providers::common::{E2ETestProvider, E2ETestProviders};

crate::generate_provider_tests!(get_providers);

async fn get_providers() -> E2ETestProviders {
    let provider_flash = E2ETestProvider {
        variant_name: "gcp-vertex-gemini-flash".to_string(),
        model_name: "gemini-1.5-flash-001".to_string(),
        model_provider_name: "gcp_vertex_gemini".to_string(),
    };

    let provider_pro = E2ETestProvider {
        variant_name: "gcp-vertex-gemini-pro".to_string(),
        model_name: "gemini-1.5-pro-001".to_string(),
        model_provider_name: "gcp_vertex_gemini".to_string(),
    };

    E2ETestProviders::with_providers(vec![provider_flash, provider_pro])
}
