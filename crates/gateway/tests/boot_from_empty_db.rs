//! End-to-end scenario proving the gateway boots from Postgres alone, with
//! the config-in-database feature flag on and no `--config-file`, and that
//! the DB-authoritative config endpoints serve back the default config.
//!
//! This is the baseline the zero-config UI deployment will build on top of:
//! schema migrated, no config rows, no TOML, gateway up, `/status` healthy,
//! `/internal/config_toml` returns the defaulted editable TOML, and the
//! hash reported across the two endpoints agrees.
//!
//! Runs against the shared live-tests Postgres (the one the CI stack
//! migrates and configures `pg_cron` for) rather than an ephemeral DB,
//! because `pg_cron` is wired to a single cluster-configured database.

use http::StatusCode;
use serde_json::Value;
use tensorzero_core::endpoints::status::{StatusResponse, TENSORZERO_VERSION};

use crate::common::start_gateway_from_db_url_on_random_port;

mod common;

#[tokio::test]
async fn gateway_boots_from_db_only_and_serves_default_config() {
    let db_url = std::env::var("TENSORZERO_POSTGRES_URL")
        .expect("TENSORZERO_POSTGRES_URL must be set for this integration test");

    let child_data = start_gateway_from_db_url_on_random_port(
        &db_url,
        &[("TENSORZERO_INTERNAL_FLAG_ENABLE_CONFIG_IN_DATABASE", "true")],
    )
    .await;

    // Basic liveness.
    let health = child_data.call_health_endpoint().await;
    assert_eq!(health.status(), StatusCode::OK, "/health should be 200");

    // /status sanity + capture the config_hash for cross-endpoint comparison.
    let status = reqwest::Client::new()
        .get(format!("http://{}/status", child_data.addr))
        .send()
        .await
        .expect("status request should succeed");
    assert_eq!(status.status(), StatusCode::OK);
    let status_body: StatusResponse = status
        .json()
        .await
        .expect("status response should deserialize");
    assert_eq!(status_body.status, "ok");
    assert_eq!(status_body.version, TENSORZERO_VERSION);
    assert!(
        !status_body.config_hash.is_empty(),
        "config_hash should be non-empty even for an empty/default config"
    );

    // The DB-authoritative editable TOML endpoint should be available
    // (feature flag is on) and return a config whose hash matches /status.
    let config_toml = reqwest::Client::new()
        .get(format!("http://{}/internal/config_toml", child_data.addr))
        .send()
        .await
        .expect("config_toml request should succeed");
    assert_eq!(
        config_toml.status(),
        StatusCode::OK,
        "/internal/config_toml should be 200 with config-in-database enabled"
    );
    let config_toml_body: Value = config_toml
        .json()
        .await
        .expect("config_toml response should deserialize");

    let returned_hash = config_toml_body
        .get("hash")
        .and_then(|v| v.as_str())
        .expect("config_toml response should carry a hash");
    assert_eq!(
        returned_hash, status_body.config_hash,
        "/status and /internal/config_toml should agree on the live config hash"
    );

    // The default config has no user-provided functions/variants/models, so
    // the editable TOML has no file-backed content referenced.
    let path_contents = config_toml_body
        .get("path_contents")
        .and_then(|v| v.as_object())
        .expect("config_toml response should carry path_contents");
    assert!(
        path_contents.is_empty(),
        "empty-DB config should have no path-backed template content, got: {path_contents:?}"
    );

    // And the TOML body should parse as TOML (sanity — not empty string).
    let toml_body = config_toml_body
        .get("toml")
        .and_then(|v| v.as_str())
        .expect("config_toml response should carry a toml string");
    let parsed: toml::Value = toml::from_str(toml_body)
        .unwrap_or_else(|e| panic!("editable TOML should parse: {e}\nbody:\n{toml_body}"));
    assert!(
        parsed.is_table(),
        "editable TOML should parse to a table at the root"
    );
}
