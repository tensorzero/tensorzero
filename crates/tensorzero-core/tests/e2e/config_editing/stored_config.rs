//! E2E tests for the stored-config TOML endpoints.
//!
//! These tests cover the routes registered behind the
//! `TENSORZERO_INTERNAL_FLAG_ENABLE_CONFIG_IN_DATABASE` feature flag:
//!
//! - `GET  /internal/config_toml`          (`get_live_config_toml_handler`)
//! - `POST /internal/config_toml/validate` (`validate_config_toml_handler`)
//! - `POST /internal/config_toml/apply`    (`apply_config_toml_handler`)
//!   (free-file behavior only; full apply coverage lives in `mod.rs`)
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

use googletest::prelude::*;
use reqwest::{Client, StatusCode};
use serde_json::json;
use tensorzero_core::endpoints::internal::config_toml::{
    ApplyConfigTomlResponse, GetConfigTomlResponse, ValidateConfigTomlRequest,
    ValidateConfigTomlResponse,
};

use crate::common::get_gateway_endpoint;

/// Helper: GET /internal/config_toml and return the parsed response.
async fn get_config_toml(client: &Client) -> GetConfigTomlResponse {
    let resp = client
        .get(get_gateway_endpoint("/internal/config_toml"))
        .send()
        .await
        .expect("GET /internal/config_toml should reach the gateway");
    assert_that!(resp.status(), eq(StatusCode::OK));
    resp.json::<GetConfigTomlResponse>()
        .await
        .expect("GET /internal/config_toml should deserialize as GetConfigTomlResponse")
}

/// Helper: POST /internal/config_toml/apply and return the raw response.
async fn apply_config_toml(
    client: &Client,
    toml: &str,
    path_contents: &HashMap<String, String>,
    base_signature: &str,
) -> reqwest::Response {
    client
        .post(get_gateway_endpoint("/internal/config_toml/apply"))
        .json(&json!({
            "toml": toml,
            "path_contents": path_contents,
            "base_signature": base_signature,
        }))
        .send()
        .await
        .expect("POST /internal/config_toml/apply should send")
}

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

// ---------------------------------------------------------------------------
// Free-file tests (files in path_contents not referenced by the TOML config)
// ---------------------------------------------------------------------------

const FREE_FILE_PATH: &str = "e2e_stored_config/free_file.txt";

/// Applying a config with a file in `path_contents` that is not referenced by
/// any TOML entry (a "free file") should succeed, and a subsequent GET should
/// include that file in `path_contents`.
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_free_file_appears_in_get_after_apply() {
    let client = Client::new();
    let original = get_config_toml(&client).await;

    let mut path_contents = original.path_contents.clone();
    path_contents.insert(FREE_FILE_PATH.to_string(), "free file v1".to_string());

    let apply_resp = apply_config_toml(
        &client,
        &original.toml,
        &path_contents,
        &original.base_signature,
    )
    .await;
    let apply_status = apply_resp.status();
    let apply_body = apply_resp.text().await.expect("apply body should decode");
    assert_that!(
        apply_status,
        eq(StatusCode::OK),
        "apply with free file should succeed; body={apply_body}"
    );

    // The file should now show up in a fresh GET.
    let after = get_config_toml(&client).await;
    expect_that!(
        after.path_contents.get(FREE_FILE_PATH),
        some(eq("free file v1")),
        "GET after apply should include the free file in path_contents"
    );

    // Restore: remove the free file by not including it in path_contents.
    let restore_resp = apply_config_toml(
        &client,
        &original.toml,
        &original.path_contents,
        &after.base_signature,
    )
    .await;
    assert_that!(
        restore_resp.status(),
        eq(StatusCode::OK),
        "restore (remove free file) should succeed"
    );
}

/// The `base_signature` returned by `apply` must match the `base_signature`
/// returned by a subsequent GET, ensuring the client can continue editing
/// without fetching a fresh snapshot.
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_free_file_apply_response_base_signature_matches_get() {
    let client = Client::new();
    let original = get_config_toml(&client).await;

    let mut path_contents = original.path_contents.clone();
    path_contents.insert(FREE_FILE_PATH.to_string(), "sig-check content".to_string());

    let apply_resp = apply_config_toml(
        &client,
        &original.toml,
        &path_contents,
        &original.base_signature,
    )
    .await;
    assert_that!(
        apply_resp.status(),
        eq(StatusCode::OK),
        "apply with free file should succeed"
    );
    let apply_parsed: ApplyConfigTomlResponse = apply_resp
        .json()
        .await
        .expect("apply response should deserialize as ApplyConfigTomlResponse");

    // The apply response must contain the free file so the client does not
    // need to re-fetch.
    expect_that!(
        apply_parsed.path_contents.get(FREE_FILE_PATH),
        some(eq("sig-check content")),
        "apply response path_contents should include the free file"
    );

    // A fresh GET should return the same base_signature.
    let after = get_config_toml(&client).await;
    expect_that!(
        apply_parsed.base_signature.as_str(),
        eq(after.base_signature.as_str()),
        "apply response base_signature should match the subsequent GET base_signature"
    );

    // Restore.
    let restore_resp = apply_config_toml(
        &client,
        &original.toml,
        &original.path_contents,
        &after.base_signature,
    )
    .await;
    assert_that!(
        restore_resp.status(),
        eq(StatusCode::OK),
        "restore should succeed"
    );
}

/// Omitting a previously-applied free file from `path_contents` in a
/// subsequent apply should tombstone it, so a follow-up GET no longer returns
/// it.
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_free_file_tombstoned_when_omitted_from_apply() {
    let client = Client::new();
    let original = get_config_toml(&client).await;

    // Apply #1: add the free file.
    let mut with_free_file = original.path_contents.clone();
    with_free_file.insert(FREE_FILE_PATH.to_string(), "to be removed".to_string());

    let apply1_resp = apply_config_toml(
        &client,
        &original.toml,
        &with_free_file,
        &original.base_signature,
    )
    .await;
    assert_that!(
        apply1_resp.status(),
        eq(StatusCode::OK),
        "apply #1 (add free file) should succeed"
    );

    let after_add = get_config_toml(&client).await;
    assert_that!(
        after_add.path_contents.contains_key(FREE_FILE_PATH),
        eq(true),
        "free file must be present after apply #1 before testing removal"
    );

    // Apply #2: omit the free file (use original path_contents, which does
    // not include it).
    let apply2_resp = apply_config_toml(
        &client,
        &original.toml,
        &original.path_contents,
        &after_add.base_signature,
    )
    .await;
    assert_that!(
        apply2_resp.status(),
        eq(StatusCode::OK),
        "apply #2 (remove free file) should succeed"
    );

    let after_remove = get_config_toml(&client).await;
    expect_that!(
        after_remove.path_contents.contains_key(FREE_FILE_PATH),
        eq(false),
        "tombstoned free file must not appear in GET after removal"
    );
}

/// Applying the same TOML twice with updated free-file content should result
/// in the latest content being returned by GET.
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_free_file_content_updated_on_edit() {
    let client = Client::new();
    let original = get_config_toml(&client).await;

    // Apply #1: insert v1.
    let mut with_v1 = original.path_contents.clone();
    with_v1.insert(FREE_FILE_PATH.to_string(), "version 1".to_string());

    let apply1_resp =
        apply_config_toml(&client, &original.toml, &with_v1, &original.base_signature).await;
    assert_that!(
        apply1_resp.status(),
        eq(StatusCode::OK),
        "apply #1 (v1) should succeed"
    );

    let after_v1 = get_config_toml(&client).await;

    // Apply #2: update to v2.
    let mut with_v2 = original.path_contents.clone();
    with_v2.insert(FREE_FILE_PATH.to_string(), "version 2".to_string());

    let apply2_resp =
        apply_config_toml(&client, &original.toml, &with_v2, &after_v1.base_signature).await;
    assert_that!(
        apply2_resp.status(),
        eq(StatusCode::OK),
        "apply #2 (v2) should succeed"
    );

    let after_v2 = get_config_toml(&client).await;
    expect_that!(
        after_v2.path_contents.get(FREE_FILE_PATH),
        some(eq("version 2")),
        "GET after edit should return the updated free file content"
    );

    // Restore.
    let restore_resp = apply_config_toml(
        &client,
        &original.toml,
        &original.path_contents,
        &after_v2.base_signature,
    )
    .await;
    assert_that!(
        restore_resp.status(),
        eq(StatusCode::OK),
        "restore should succeed"
    );
}

/// Validate should accept a config TOML paired with extra free files in
/// `path_contents` — free files are not referenced by the config but must not
/// cause validation to fail.
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_validate_config_toml_accepts_free_files_in_path_contents() {
    let client = Client::new();
    let live = get_config_toml(&client).await;

    // Add a file that is not referenced by any TOML entry.
    let mut path_contents_with_free = live.path_contents.clone();
    path_contents_with_free.insert(
        "e2e_stored_config/extra_validate_free_file.txt".to_string(),
        "this file is not referenced by any config entry".to_string(),
    );

    let request = ValidateConfigTomlRequest {
        toml: live.toml,
        path_contents: path_contents_with_free,
    };

    let resp = client
        .post(get_gateway_endpoint("/internal/config_toml/validate"))
        .json(&request)
        .send()
        .await
        .expect("POST /internal/config_toml/validate should reach the gateway");
    let status = resp.status();
    let body = resp.text().await.expect("response body should be readable");
    assert_that!(
        status,
        eq(StatusCode::OK),
        "validate should accept a config with free files in path_contents; body={body}"
    );

    let parsed: ValidateConfigTomlResponse =
        serde_json::from_str(&body).expect("response should parse as ValidateConfigTomlResponse");
    expect_that!(
        parsed.valid,
        eq(true),
        "validate should report `valid = true` when path_contents includes unreferenced free files"
    );
}
