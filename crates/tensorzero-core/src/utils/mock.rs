//! Mock API helpers for testing with mock inference providers.
//!
//! These functions read from the `TENSORZERO_INTERNAL_MOCK_PROVIDER_API` environment variable
//! to determine if we're in mock mode and to construct mock API URLs.

pub use tensorzero_inference_types::utils::{get_mock_provider_api_base, is_mock_mode};
