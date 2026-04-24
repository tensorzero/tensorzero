//! REST bootstrap helper for E2E tests that start from an empty gateway.
//!
//! The long-term plan is to populate gateway config through narrow per-object
//! endpoints (Phase 2B in the design doc). Until those exist, the most direct
//! way for a test to install a full config is via `POST /internal/config_toml/apply`
//! — the same endpoint the UI's config editor uses. This module wraps that
//! fetch-then-apply dance into a single call so tests can focus on what they
//! want the gateway to look like, not on the CAS protocol.

use std::collections::HashMap;

use googletest::prelude::*;
use reqwest::{Client, StatusCode};
use serde_json::json;
use tensorzero_core::endpoints::internal::config_toml::{
    ApplyConfigTomlResponse, GetConfigTomlResponse,
};

use crate::common::get_gateway_endpoint;

/// Handle to a gateway that has just been bootstrapped with some config.
/// Subsequent calls (e.g. follow-on `apply`s in multi-step tests) need the
/// up-to-date `base_signature` to pass the CAS check, so we expose it here
/// along with the canonical TOML the gateway recorded.
#[derive(Debug, Clone)]
#[expect(
    dead_code,
    reason = "path_contents and base_signature are exposed for chained apply tests that land in follow-up Phase 2A work"
)]
pub struct BootstrappedConfig {
    pub toml: String,
    pub path_contents: HashMap<String, String>,
    pub hash: String,
    pub base_signature: String,
}

impl From<ApplyConfigTomlResponse> for BootstrappedConfig {
    fn from(resp: ApplyConfigTomlResponse) -> Self {
        Self {
            toml: resp.toml,
            path_contents: resp.path_contents,
            hash: resp.hash,
            base_signature: resp.base_signature,
        }
    }
}

/// Fetch the current editable config from the gateway. Tests call this to get
/// the initial `base_signature` — for a just-booted empty gateway it'll be
/// the signature of the default-singletons document, not the empty string.
pub async fn fetch_current_config(client: &Client) -> GetConfigTomlResponse {
    let response = client
        .get(get_gateway_endpoint("/internal/config_toml"))
        .send()
        .await
        .expect("GET /internal/config_toml should send");
    assert_that!(
        response.status(),
        eq(StatusCode::OK),
        "GET /internal/config_toml did not return 200"
    );
    response
        .json::<GetConfigTomlResponse>()
        .await
        .expect("GET /internal/config_toml should deserialize as GetConfigTomlResponse")
}

/// Install `toml` (plus any referenced template files in `path_contents`) as
/// the gateway's active config. Performs the fetch-then-apply CAS dance so
/// callers don't have to thread `base_signature` manually. Panics with an
/// informative error message on any non-200 response.
pub async fn bootstrap_gateway_with_toml(
    client: &Client,
    toml: &str,
    path_contents: &HashMap<String, String>,
) -> BootstrappedConfig {
    let current = fetch_current_config(client).await;
    apply_with_signature(client, toml, path_contents, &current.base_signature).await
}

/// Apply `toml` to the gateway using an explicit `base_signature`. Useful for
/// tests that want to make multiple sequential edits without re-fetching the
/// full document between each one — they can chain the returned
/// `base_signature` into the next call.
pub async fn apply_with_signature(
    client: &Client,
    toml: &str,
    path_contents: &HashMap<String, String>,
    base_signature: &str,
) -> BootstrappedConfig {
    let response = client
        .post(get_gateway_endpoint("/internal/config_toml/apply"))
        .json(&json!({
            "toml": toml,
            "path_contents": path_contents,
            "base_signature": base_signature,
        }))
        .send()
        .await
        .expect("POST /internal/config_toml/apply should send");
    let status = response.status();
    let body_text = response
        .text()
        .await
        .expect("apply response body should decode");
    assert_that!(
        status,
        eq(StatusCode::OK),
        "POST /internal/config_toml/apply failed with status {status}: {body_text}"
    );
    let apply: ApplyConfigTomlResponse = serde_json::from_str(&body_text).expect(
        "apply response should deserialize as ApplyConfigTomlResponse on a successful apply",
    );
    BootstrappedConfig::from(apply)
}
