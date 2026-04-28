//! Small utilities shared between providers and core.

use url::Url;

/// Emits a deprecation warning.
/// All deprecation warnings should be emitted using this function so that we can detect
/// unintentional use of deprecated behavior in our e2e tests.
pub fn deprecation_warning(message: &str) {
    tracing::warn!("Deprecation Warning: {message}");
}

/// Returns true if we're in mock mode (`TENSORZERO_INTERNAL_MOCK_PROVIDER_API` is set and non-empty).
pub fn is_mock_mode() -> bool {
    std::env::var("TENSORZERO_INTERNAL_MOCK_PROVIDER_API")
        .ok()
        .filter(|s| !s.is_empty())
        .is_some()
}

/// Returns the mock API base URL with the provider suffix appended.
/// Reads from `TENSORZERO_INTERNAL_MOCK_PROVIDER_API` env var.
/// Handles trailing slash normalization. Maps empty string to None.
pub fn get_mock_provider_api_base(provider_suffix: &str) -> Option<Url> {
    std::env::var("TENSORZERO_INTERNAL_MOCK_PROVIDER_API")
        .ok()
        .filter(|s| !s.is_empty())
        .and_then(|base| {
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
