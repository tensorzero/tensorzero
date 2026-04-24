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

mod bootstrap;

use std::collections::HashMap;

use googletest::prelude::*;
use reqwest::{Client, StatusCode};
use tensorzero_core::endpoints::internal::config_toml::GetConfigTomlResponse;
use tensorzero_core::endpoints::status::{StatusResponse, TENSORZERO_VERSION};

use crate::common::get_gateway_endpoint;
use bootstrap::{apply_with_signature, bootstrap_gateway_with_toml, fetch_current_config};

/// Minimal self-contained bootstrap config: a single metric definition. Kept
/// provider-free on purpose — the `db-only-boot` CI job runs against the
/// production gateway image, which does not compile in the `dummy` provider,
/// and we don't want this test to depend on a live upstream credential.
const MINIMAL_BOOTSTRAP_TOML: &str = r#"
[metrics.db_only_boot_bootstrap_metric]
type = "boolean"
optimize = "max"
level = "episode"
"#;

const BOOTSTRAP_METRIC_NAME: &str = "db_only_boot_bootstrap_metric";

/// Reset the gateway back to an empty default config after a test mutates it,
/// so the next test starts from a clean state.
async fn reset_to_empty_config(client: &Client) {
    let current = fetch_current_config(client).await;
    // Re-applying an empty TOML won't quite produce the same document as a
    // truly-empty DB (the gateway's default-filled singletons come along
    // for the ride), but `/apply` treats "empty functions/models/etc." as
    // a tombstone signal and clears the collections. That's what we need.
    apply_with_signature(client, "", &HashMap::new(), &current.base_signature).await;
}

async fn get_status(client: &Client) -> StatusResponse {
    client
        .get(get_gateway_endpoint("/status"))
        .send()
        .await
        .expect("GET /status should send")
        .json()
        .await
        .expect("GET /status should deserialize as StatusResponse")
}

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
    let parsed: toml::Table = toml::from_str(&config.toml)
        .expect("GET /internal/config_toml body should parse as a TOML table");

    // The collection tables (functions, models, tools, metrics) must be
    // absent or empty on a zero-config gateway. Parse structurally rather
    // than grepping substrings so the assertion doesn't encode a specific
    // header layout.
    for key in ["functions", "models", "tools", "metrics"] {
        let Some(value) = parsed.get(key) else {
            continue;
        };
        let table = value
            .as_table()
            .unwrap_or_else(|| panic!("expected `{key}` to be a table, got {value:?}"));
        assert!(
            table.is_empty(),
            "expected `{key}` table to be empty on a zero-config gateway, got {table:?}",
        );
    }

    // `base_signature` is the CAS token callers echo back to
    // `/internal/config_toml/apply`. It must be populated even when the
    // database is empty, otherwise the first apply would fail validation.
    expect_that!(&config.base_signature, not(eq("")));
}

/// End-to-end exercise of the REST bootstrap flow: take an empty gateway,
/// install a minimal config via `/internal/config_toml/apply`, and confirm
/// that (a) the new config is visible via `/internal/config_toml` and (b)
/// the live runtime was hot-swapped. `/apply` returns the hash of the
/// config it prepared for hot-swap, and `/status` reports the live
/// runtime's hash — so comparing the two after apply catches the
/// failure mode where the DB write lands but the in-memory swap never
/// runs.
#[gtest]
#[tokio::test]
async fn db_only_boot_bootstrap_installs_metric_and_hot_swaps_runtime() {
    let client = Client::new();

    let metric_header = format!("[metrics.{BOOTSTRAP_METRIC_NAME}]");

    // Precondition: gateway starts with no user-defined metric, and the
    // live runtime agrees with the DB-read endpoint on the current config
    // hash. This half of the invariant exists on a fresh empty DB already
    // (the baseline `..._returns_default_config_via_config_toml_endpoint`
    // test covers it), we re-assert here so a test-order swap would still
    // catch drift.
    let initial_status = get_status(&client).await;
    let initial_config = fetch_current_config(&client).await;
    assert_that!(&initial_config.hash, eq(&initial_status.config_hash));
    assert_that!(
        initial_config.toml.as_str(),
        not(contains_substring(metric_header.as_str())),
        "bootstrap metric must not exist before the test applies it"
    );

    // Bootstrap: install the minimal config via REST.
    let bootstrapped =
        bootstrap_gateway_with_toml(&client, MINIMAL_BOOTSTRAP_TOML, &HashMap::new()).await;
    assert_that!(bootstrapped.hash.as_str(), not(eq("")));
    assert_that!(
        &bootstrapped.hash,
        not(eq(&initial_status.config_hash)),
        "apply should have produced a new config hash"
    );
    assert_that!(
        bootstrapped.toml.as_str(),
        contains_substring(metric_header.as_str())
    );

    // DB-read confirmation: the metric is reachable through
    // `/internal/config_toml`, AND the hash computed from the re-read matches
    // the hash `/apply` returned. The latter is the invariant that guarantees
    // the UI sees a stable hash on save vs. refresh.
    let after_config = fetch_current_config(&client).await;
    assert_that!(
        after_config.toml.as_str(),
        contains_substring(metric_header.as_str())
    );
    assert_that!(
        &after_config.hash,
        eq(&bootstrapped.hash),
        "DB-read hash must match /apply hash after a round-trip"
    );

    // Hot-swap confirmation: `/status` reports the live runtime's config
    // hash, which is the same thing `/apply` returns (both come from
    // `app_state.config.hash` after the swap). Mismatch means the DB
    // write landed but the runtime swap didn't.
    let after_status = get_status(&client).await;
    assert_that!(
        &after_status.config_hash,
        eq(&bootstrapped.hash),
        "runtime config hash must match the applied config hash — otherwise the DB write did not hot-swap"
    );

    // Cleanup: next test in this profile expects an empty gateway.
    reset_to_empty_config(&client).await;
}
