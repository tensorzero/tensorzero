//! E2E coverage for the zero-config boot path: a migrated-but-empty
//! Postgres + ClickHouse stack with `ENABLE_CONFIG_IN_DATABASE=true`
//! and no `--config-file`. This is the deploy shape the
//! configure-via-UI story builds on — schema present, no config rows,
//! no files on disk — so these tests lock in the contract that the
//! gateway boots and surfaces the defaulted config through the same
//! REST endpoints the UI will use.
//!
//! These tests run in their own `db-only-boot` nextest profile against
//! a dedicated docker-compose stack
//! (`crates/tensorzero-core/tests/e2e/docker-compose.db-only-boot.yml`)
//! that omits the `gateway-migrate-config` step and drops the on-disk
//! config/template bind mounts. They do not run in the regular live
//! suite, which uses the TOML-config gateway.

use googletest::prelude::*;
use reqwest::{Client, StatusCode};
use tensorzero_core::endpoints::internal::config_toml::GetConfigTomlResponse;
use tensorzero_core::endpoints::status::{StatusResponse, TENSORZERO_VERSION};

use crate::common::get_gateway_endpoint;

#[gtest]
#[tokio::test]
async fn db_only_boot_serves_status_with_defaulted_config() {
    let client = Client::new();

    let response = client
        .get(get_gateway_endpoint("/status"))
        .send()
        .await
        .expect("status request should succeed");
    expect_that!(response.status(), eq(StatusCode::OK));

    let status: StatusResponse = response
        .json()
        .await
        .expect("status response should deserialize");
    expect_that!(
        status,
        matches_pattern!(StatusResponse {
            status: eq("ok"),
            version: eq(TENSORZERO_VERSION),
            config_hash: not(eq("")),
        })
    );
}

#[gtest]
#[tokio::test]
async fn db_only_boot_returns_default_config_via_config_toml_endpoint() {
    let client = Client::new();

    let status: StatusResponse = client
        .get(get_gateway_endpoint("/status"))
        .send()
        .await
        .expect("status request should succeed")
        .json()
        .await
        .expect("status response should deserialize");

    let config: GetConfigTomlResponse = client
        .get(get_gateway_endpoint("/internal/config_toml"))
        .send()
        .await
        .expect("config_toml request should succeed")
        .json()
        .await
        .expect("config_toml response should deserialize");

    // The two endpoints must agree on the live config hash; otherwise the
    // UI could read one version and write against another.
    expect_that!(&config.hash, eq(&status.config_hash));

    // No user config means no file-backed templates referenced by the TOML.
    expect_that!(&config.path_contents, is_empty());

    // The serialized editable TOML should still be non-empty — every config
    // singleton renders its defaulted section header — and it must parse
    // back as a valid TOML table.
    expect_that!(&config.toml, not(eq("")));
    let parsed = toml::from_str::<toml::Value>(&config.toml);
    expect_that!(parsed, ok(predicate(toml::Value::is_table)));
}
