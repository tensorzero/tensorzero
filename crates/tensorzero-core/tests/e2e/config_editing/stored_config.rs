//! E2E tests for the stored-config TOML endpoints.
//!
//! These tests cover the routes registered behind the
//! `TENSORZERO_INTERNAL_FLAG_ENABLE_CONFIG_IN_DATABASE` feature flag:
//!
//! - `GET  /internal/config_toml`          (`get_live_config_toml_handler`)
//! - `POST /internal/config_toml/validate` (`validate_config_toml_handler`)
//!
//! These endpoints are part of the "config in database" stack: they return
//! human-editable TOML (with file-backed contents extracted into a separate
//! `path_contents` map) and validate user-edited TOML by running the shared
//! config-loading pipeline.
//!
//! NOTE: This file is intentionally distinct from `config.rs` in this same
//! directory. `config.rs` covers the always-on JSON snapshot endpoints
//! (`/internal/config`, `/internal/config/{hash}`, and the `POST` write
//! handler), which return a `GetConfigResponse` whose `config` field is an
//! arbitrary JSON value. The endpoints exercised here instead return
//! `GetConfigTomlResponse` — a TOML string plus a `path_contents` map — and
//! are only mounted when the `enable_config_in_database` feature flag is set.
//! Tests live in a separate file so the JSON and TOML surfaces stay easy to
//! navigate independently.

use std::collections::HashMap;

use reqwest::Client;
use tensorzero_core::endpoints::internal::config_toml::{
    GetConfigTomlResponse, ValidateConfigTomlRequest, ValidateConfigTomlResponse,
};

use crate::common::get_gateway_endpoint;

#[tokio::test(flavor = "multi_thread")]
async fn test_get_live_config_toml() {
    let http_client = Client::new();
    let url = get_gateway_endpoint("/internal/config_toml");

    let resp = http_client
        .get(url)
        .send()
        .await
        .expect("GET /internal/config_toml should reach the gateway");
    let status = resp.status();
    let body = resp.text().await.expect("response body should be readable");

    assert!(
        status.is_success(),
        "GET /internal/config_toml should succeed: status={status}, body={body}"
    );

    let response: GetConfigTomlResponse = serde_json::from_str(&body)
        .expect("GET /internal/config_toml response should parse as GetConfigTomlResponse");

    assert!(
        !response.hash.is_empty(),
        "live config TOML response should include a non-empty hash"
    );
    assert!(
        !response.toml.is_empty(),
        "live config TOML response should include non-empty TOML body"
    );
    // The live e2e gateway config has no file-backed paths in the
    // `path_contents` extraction set, so an empty map is fine — we only
    // require that the field is present and well-formed (already enforced
    // by the typed deserialization above).
    assert!(
        toml::from_str::<toml::Table>(&response.toml).is_ok(),
        "rendered TOML body should itself be parseable as TOML"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_validate_config_toml_accepts_round_trip() {
    let http_client = Client::new();

    // Round-trip the live config through `GET /internal/config_toml` and
    // feed it back to the validate endpoint. This is the canonical "valid"
    // payload the UI will send: when the `enable_config_in_database` flag
    // is on, the live endpoint reads from the stored-config tables (not the
    // snapshot table) and therefore omits the runtime-injected `tensorzero::`
    // built-in functions, so the loader will accept what comes back out.
    let live_url = get_gateway_endpoint("/internal/config_toml");
    let live_resp = http_client
        .get(live_url)
        .send()
        .await
        .expect("GET /internal/config_toml should reach the gateway");
    assert!(
        live_resp.status().is_success(),
        "live GET /internal/config_toml should succeed"
    );
    let live: GetConfigTomlResponse = live_resp
        .json()
        .await
        .expect("live response should parse as GetConfigTomlResponse");

    let request = ValidateConfigTomlRequest {
        toml: live.toml,
        path_contents: live.path_contents,
    };

    let url = get_gateway_endpoint("/internal/config_toml/validate");
    let resp = http_client
        .post(url)
        .json(&request)
        .send()
        .await
        .expect("POST /internal/config_toml/validate should reach the gateway");
    let status = resp.status();
    let body = resp.text().await.expect("response body should be readable");
    assert!(
        status.is_success(),
        "validate should accept a live round-tripped config: status={status}, body={body}"
    );

    let response: ValidateConfigTomlResponse =
        serde_json::from_str(&body).expect("response should parse as ValidateConfigTomlResponse");
    assert!(
        response.valid,
        "validate response should report `valid = true` for a round-tripped live config"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_validate_config_toml_rejects_unparseable_toml() {
    let http_client = Client::new();

    let request = ValidateConfigTomlRequest {
        // Stray bracket — not a valid TOML document.
        toml: "[functions.bad".to_string(),
        path_contents: HashMap::new(),
    };

    let url = get_gateway_endpoint("/internal/config_toml/validate");
    let resp = http_client
        .post(url)
        .json(&request)
        .send()
        .await
        .expect("POST /internal/config_toml/validate should reach the gateway");

    assert!(
        !resp.status().is_success(),
        "validate should reject unparseable TOML, got status={}",
        resp.status()
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_validate_config_toml_rejects_invalid_model_ref() {
    let http_client = Client::new();

    // Parses fine as TOML and as `UninitializedConfig`, but the variant
    // points at a model that does not exist — the shared loading pipeline
    // should reject this.
    let toml = r#"
[functions.test_invalid_func_toml]
type = "chat"

[functions.test_invalid_func_toml.variants.bad_variant]
type = "chat_completion"
model = "nonexistent_model_for_validate_toml"
"#
    .to_string();

    let request = ValidateConfigTomlRequest {
        toml,
        path_contents: HashMap::new(),
    };

    let url = get_gateway_endpoint("/internal/config_toml/validate");
    let resp = http_client
        .post(url)
        .json(&request)
        .send()
        .await
        .expect("POST /internal/config_toml/validate should reach the gateway");

    assert!(
        !resp.status().is_success(),
        "validate should reject a config whose variant references a nonexistent model, \
         got status={}",
        resp.status()
    );
}
