//! Mock API helpers for testing with mock inference providers.
//!
//! These functions read from the `TENSORZERO_INTERNAL_MOCK_PROVIDER_API` environment variable
//! to determine if we're in mock mode and to construct mock API URLs.

use url::Url;

/// Returns true if we're in mock mode (TENSORZERO_INTERNAL_MOCK_PROVIDER_API is set and non-empty).
pub fn is_mock_mode() -> bool {
    std::env::var("TENSORZERO_INTERNAL_MOCK_PROVIDER_API")
        .ok()
        .filter(|s| !s.is_empty())
        .is_some()
}

/// Returns the mock API base URL with the provider suffix appended.
/// Reads from TENSORZERO_INTERNAL_MOCK_PROVIDER_API env var.
/// Handles trailing slash normalization. Maps empty string to None.
pub fn get_mock_provider_api_base(provider_suffix: &str) -> Option<Url> {
    std::env::var("TENSORZERO_INTERNAL_MOCK_PROVIDER_API")
        .ok()
        .filter(|s| !s.is_empty())
        .and_then(|base| {
            // Normalize: ensure base ends with / if suffix doesn't start with /
            let needs_slash = !base.ends_with('/')
                && !provider_suffix.starts_with('/')
                && !provider_suffix.is_empty();
            let base = if needs_slash {
                format!("{base}/")
            } else {
                base
            };
            Url::parse(&format!("{base}{provider_suffix}")).ok()
        })
}
