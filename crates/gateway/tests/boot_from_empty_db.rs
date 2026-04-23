//! Integration test proving the gateway actually boots and serves HTTP when
//! only `TENSORZERO_POSTGRES_URL` is set: no `--config-file`, no
//! `--default-config`, and no `ENABLE_CONFIG_IN_DATABASE` feature flag. The
//! unit-level test in `tensorzero-core/tests/e2e/db/postgres/stored_configs`
//! exercises `load_config_from_db` against an empty DB; this test exercises
//! the full binary boot path end-to-end and makes a real HTTP request.
//!
//! Runs against the shared live-tests Postgres (the one the CI stack
//! migrates and configures `pg_cron` for) rather than an ephemeral DB,
//! because `pg_cron` is wired to a single cluster-configured database and
//! the gateway's extension validation fails outside it.

use http::StatusCode;
use tensorzero_core::endpoints::status::{StatusResponse, TENSORZERO_VERSION};

use crate::common::start_gateway_from_db_url_on_random_port;

mod common;

#[tokio::test]
async fn gateway_boots_from_postgres_only_and_serves_status() {
    let db_url = std::env::var("TENSORZERO_POSTGRES_URL")
        .expect("TENSORZERO_POSTGRES_URL must be set for this integration test");

    let child_data = start_gateway_from_db_url_on_random_port(&db_url).await;

    let health = child_data.call_health_endpoint().await;
    assert_eq!(
        health.status(),
        StatusCode::OK,
        "gateway booted with only Postgres should serve a healthy /health"
    );

    let status = reqwest::Client::new()
        .get(format!("http://{}/status", child_data.addr))
        .send()
        .await
        .expect("status request should succeed");
    assert_eq!(status.status(), StatusCode::OK);

    let body: StatusResponse = status
        .json()
        .await
        .expect("status response should deserialize");
    assert_eq!(body.status, "ok");
    assert_eq!(body.version, TENSORZERO_VERSION);
    assert!(
        !body.config_hash.is_empty(),
        "config_hash should be non-empty even for an empty/default config"
    );
}
