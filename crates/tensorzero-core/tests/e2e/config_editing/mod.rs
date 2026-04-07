//! End-to-end coverage for writing config via `/internal/config_toml/apply`.
//!
//! These tests mutate gateway-wide config state, so they live in their own
//! `config-editing` nextest profile (see `crates/.config/nextest.toml`) and
//! run after the main live test suite finishes. Each test fetches the live
//! editable TOML on entry and re-applies it on the way out so the fixture is
//! restored regardless of which test runs next.

// Read-only coverage of `/internal/config_toml` and
// `/internal/config_toml/validate`. These routes are only mounted when the
// `enable_config_in_database` feature flag is on, so they share the
// `config-editing` profile's config-in-database gateway even though they
// don't mutate state.
mod stored_config;

use std::collections::HashMap;

use googletest::prelude::*;
use reqwest::{Client, StatusCode};
use serde_json::{Value, json};
use tensorzero_core::db::delegating_connection::DelegatingDatabaseConnection;
use tensorzero_core::db::model_inferences::ModelInferenceQueries;
use tensorzero_core::db::postgres::test_helpers::get_postgres;
use tensorzero_core::db::test_helpers::TestDatabaseHelpers;
use tensorzero_core::endpoints::internal::config_toml::{
    ApplyConfigTomlResponse, GetConfigTomlResponse,
};
use uuid::Uuid;

use crate::common::get_gateway_endpoint;

/// Name used by the add/remove round-trip test.
const THROWAWAY_FUNCTION_NAME: &str = "e2e_config_editing_throwaway_function";

/// Name used by the add-variant round-trip test.
const ADD_VARIANT_FUNCTION_NAME: &str = "e2e_config_editing_add_variant_function";
const EXTRA_VARIANT_NAME: &str = "extra_variant";

async fn get_config_toml(client: &Client) -> GetConfigTomlResponse {
    let response = client
        .get(get_gateway_endpoint("/internal/config_toml"))
        .send()
        .await
        .expect("GET /internal/config_toml should succeed");
    assert_that!(response.status(), eq(StatusCode::OK));
    response
        .json::<GetConfigTomlResponse>()
        .await
        .expect("GET /internal/config_toml should deserialize as GetConfigTomlResponse")
}

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

async fn run_inference_with_variant(
    client: &Client,
    function_name: &str,
    variant_name: Option<&str>,
) -> reqwest::Response {
    let mut payload = json!({
        "function_name": function_name,
        "episode_id": Uuid::now_v7(),
        "input": {
            "messages": [
                { "role": "user", "content": "Hello, world!" }
            ]
        },
        "stream": false,
    });
    if let Some(variant) = variant_name {
        payload["variant_name"] = json!(variant);
    }
    client
        .post(get_gateway_endpoint("/inference"))
        .json(&payload)
        .send()
        .await
        .expect("POST /inference should send")
}

async fn run_inference(client: &Client, function_name: &str) -> reqwest::Response {
    run_inference_with_variant(client, function_name, None).await
}

/// Appends a minimal chat function (no schema, no templates) with a single
/// `test` variant that points at the `test` dummy model. Keeping the block
/// textually trivial avoids any TOML serialization round-trip issues.
fn append_new_function_block(original_toml: &str, name: &str) -> String {
    let block = format!(
        r#"

[functions.{name}]
type = "chat"

[functions.{name}.variants.test]
type = "chat_completion"
model = "test"
"#
    );
    format!("{original_toml}{block}")
}

/// Appends an extra variant on the function added via
/// `append_new_function_block`, also pointing at the `test` dummy model.
fn append_extra_variant_block(toml: &str, function_name: &str, variant_name: &str) -> String {
    let block = format!(
        r#"

[functions.{function_name}.variants.{variant_name}]
type = "chat_completion"
model = "test"
"#
    );
    format!("{toml}{block}")
}

/// Appends a function whose only variant references a model that does not
/// exist. Used to trigger validation failure inside `apply_config_toml`.
fn append_invalid_function_block(original_toml: &str) -> String {
    let block = r#"

[functions.e2e_config_editing_invalid_function]
type = "chat"

[functions.e2e_config_editing_invalid_function.variants.invalid]
type = "chat_completion"
model = "this_model_definitely_does_not_exist_e2e"
"#;
    format!("{original_toml}{block}")
}

#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_config_editing_add_then_remove_function() {
    let client = Client::new();
    let original = get_config_toml(&client).await;

    assert_that!(
        original
            .toml
            .contains(&format!("[functions.{THROWAWAY_FUNCTION_NAME}]")),
        eq(false),
        "throwaway function name must not already exist in base e2e config"
    );

    // --- Add a new function and confirm inference works against it ---
    let toml_with_new_function = append_new_function_block(&original.toml, THROWAWAY_FUNCTION_NAME);
    let apply_response = apply_config_toml(
        &client,
        &toml_with_new_function,
        &original.path_contents,
        &original.base_signature,
    )
    .await;
    let apply_status = apply_response.status();
    let apply_body = apply_response
        .text()
        .await
        .expect("apply response body should decode");
    assert_that!(
        apply_status,
        eq(StatusCode::OK),
        "apply (add) failed: {apply_body}"
    );

    let inference_after_add = run_inference(&client, THROWAWAY_FUNCTION_NAME).await;
    let add_status = inference_after_add.status();
    let add_body = inference_after_add
        .text()
        .await
        .expect("inference body should decode");
    expect_that!(
        add_status,
        eq(StatusCode::OK),
        "inference against newly-added function should succeed; body={add_body}"
    );

    // --- Remove the function and confirm inference now fails ---
    let post_add = get_config_toml(&client).await;
    let restore_response = apply_config_toml(
        &client,
        &original.toml,
        &original.path_contents,
        &post_add.base_signature,
    )
    .await;
    assert_that!(
        restore_response.status(),
        eq(StatusCode::OK),
        "restoring original config failed"
    );

    let inference_after_remove = run_inference(&client, THROWAWAY_FUNCTION_NAME).await;
    let remove_status = inference_after_remove.status();
    let remove_body = inference_after_remove
        .text()
        .await
        .expect("inference body should decode");
    expect_that!(
        remove_status.is_success(),
        eq(false),
        "inference against removed function should fail; status={remove_status}, body={remove_body}"
    );
}

/// Exercises the "add a variant to an already-stored function" flow. Because
/// the base e2e config has no unschema'd function we can safely graft onto,
/// the test first applies a throwaway function (apply #1) and then adds a
/// second variant to that now-existing function (apply #2).
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_config_editing_add_variant_to_existing_function() {
    let client = Client::new();
    let original = get_config_toml(&client).await;

    // Apply #1: create the host function with one variant.
    let toml_with_function = append_new_function_block(&original.toml, ADD_VARIANT_FUNCTION_NAME);
    let apply1 = apply_config_toml(
        &client,
        &toml_with_function,
        &original.path_contents,
        &original.base_signature,
    )
    .await;
    let apply1_status = apply1.status();
    let apply1_body = apply1.text().await.expect("apply #1 body should decode");
    assert_that!(
        apply1_status,
        eq(StatusCode::OK),
        "apply #1 (create host function) failed: {apply1_body}"
    );

    // Apply #2: add a second variant to the now-existing function.
    let after_first_apply = get_config_toml(&client).await;
    let toml_with_extra_variant = append_extra_variant_block(
        &toml_with_function,
        ADD_VARIANT_FUNCTION_NAME,
        EXTRA_VARIANT_NAME,
    );
    let apply2 = apply_config_toml(
        &client,
        &toml_with_extra_variant,
        &after_first_apply.path_contents,
        &after_first_apply.base_signature,
    )
    .await;
    let apply2_status = apply2.status();
    let apply2_body = apply2.text().await.expect("apply #2 body should decode");
    assert_that!(
        apply2_status,
        eq(StatusCode::OK),
        "apply #2 (add variant) failed: {apply2_body}"
    );

    // Inference against the newly-added variant should succeed and should
    // report the variant name back in the response.
    let inference_response =
        run_inference_with_variant(&client, ADD_VARIANT_FUNCTION_NAME, Some(EXTRA_VARIANT_NAME))
            .await;
    let status = inference_response.status();
    let body = inference_response
        .text()
        .await
        .expect("inference body should decode");
    assert_that!(
        status,
        eq(StatusCode::OK),
        "inference against newly-added variant should succeed; body={body}"
    );
    let parsed: Value = serde_json::from_str(&body).expect("inference body should be JSON");
    expect_that!(
        parsed["variant_name"].as_str(),
        some(eq(EXTRA_VARIANT_NAME)),
        "inference should have used the newly-added variant"
    );

    // Restore the original config.
    let post_apply = get_config_toml(&client).await;
    let restore_response = apply_config_toml(
        &client,
        &original.toml,
        &original.path_contents,
        &post_apply.base_signature,
    )
    .await;
    assert_that!(
        restore_response.status(),
        eq(StatusCode::OK),
        "restoring original config failed"
    );
}

/// An apply with a stale base_signature that also doesn't match the canonical
/// signature of the submitted TOML must be rejected with 409 Conflict, and
/// must not mutate the stored config.
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_config_editing_cas_conflict_rejects_stale_signature() {
    let client = Client::new();
    let original = get_config_toml(&client).await;

    // Build a TOML that introduces a real change so the canonical signature
    // of the submitted TOML also won't match the current DB signature — this
    // forces the apply handler down the conflict path rather than the no-op
    // short-circuit.
    let toml_with_new_function =
        append_new_function_block(&original.toml, "e2e_config_editing_cas_conflict_function");

    let stale_signature = "0000000000000000000000000000000000000000000000000000000000000000";
    let response = apply_config_toml(
        &client,
        &toml_with_new_function,
        &original.path_contents,
        stale_signature,
    )
    .await;
    let status = response.status();
    let body = response
        .text()
        .await
        .expect("cas conflict response body should decode");
    expect_that!(
        status,
        eq(StatusCode::CONFLICT),
        "stale base_signature should trigger 409 Conflict; body={body}"
    );

    // The rejected apply must not have mutated stored state.
    let after = get_config_toml(&client).await;
    expect_that!(
        after.hash.as_str(),
        eq(original.hash.as_str()),
        "rejected CAS apply should leave the config hash unchanged"
    );
    expect_that!(
        after.base_signature.as_str(),
        eq(original.base_signature.as_str()),
        "rejected CAS apply should leave the base_signature unchanged"
    );
}

/// A semantically invalid TOML (e.g. a variant pointing at a nonexistent
/// model) must be rejected before any state is written. A follow-up GET must
/// return the same hash and base_signature as before the failed apply.
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_config_editing_validation_failure_leaves_state_unchanged() {
    let client = Client::new();
    let original = get_config_toml(&client).await;

    let invalid_toml = append_invalid_function_block(&original.toml);
    let response = apply_config_toml(
        &client,
        &invalid_toml,
        &original.path_contents,
        &original.base_signature,
    )
    .await;
    let status = response.status();
    let body = response
        .text()
        .await
        .expect("validation failure response body should decode");
    expect_that!(
        status.is_success(),
        eq(false),
        "apply with invalid model reference should fail; status={status}, body={body}"
    );

    let after = get_config_toml(&client).await;
    expect_that!(
        after.hash.as_str(),
        eq(original.hash.as_str()),
        "failed validation apply should leave config hash unchanged"
    );
    expect_that!(
        after.base_signature.as_str(),
        eq(original.base_signature.as_str()),
        "failed validation apply should leave base_signature unchanged"
    );
}

/// Modifies a variant's system prompt via `/internal/config_toml/apply` and
/// verifies the rendered system prompt on the resulting `ModelInference` row
/// contains the new marker, proving the gateway actually re-loaded and used
/// the modified template for subsequent inferences.
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_config_editing_modify_prompt_updates_inference_system() {
    let client = Client::new();
    let original = get_config_toml(&client).await;

    let function_name = "e2e_config_editing_prompt_modify_function";
    let template_path = "e2e_config_editing/prompt_modify_system_template.minijinja";
    let marker = "PROMPT_MARKER_E2E_CONFIG_EDITING";
    let template_body = format!("You are a helpful assistant. {marker}");

    // The path_contents map is a content-addressed store keyed by the path
    // string in the TOML body — no filesystem resolution happens for keys
    // that are present in the map, so we can register a synthetic path.
    let mut path_contents = original.path_contents.clone();
    path_contents.insert(template_path.to_string(), template_body.clone());

    let new_function_block = format!(
        r#"

[functions.{function_name}]
type = "chat"

[functions.{function_name}.variants.test]
type = "chat_completion"
model = "test"
system_template = "{template_path}"
"#
    );
    let new_toml = format!("{}{}", original.toml, new_function_block);

    let apply_response =
        apply_config_toml(&client, &new_toml, &path_contents, &original.base_signature).await;
    let apply_status = apply_response.status();
    let apply_body = apply_response
        .text()
        .await
        .expect("apply body should decode");
    assert_that!(
        apply_status,
        eq(StatusCode::OK),
        "apply (modify prompt) failed: {apply_body}"
    );

    // Run inference against the new function and capture the inference id so
    // we can look up the rendered system prompt on the `ModelInference` row.
    let inference_response = run_inference(&client, function_name).await;
    let inference_status = inference_response.status();
    let inference_body = inference_response
        .text()
        .await
        .expect("inference body should decode");
    assert_that!(
        inference_status,
        eq(StatusCode::OK),
        "inference against prompt-modified function should succeed; body={inference_body}"
    );
    let parsed: Value =
        serde_json::from_str(&inference_body).expect("inference body should be JSON");
    let inference_id: Uuid = parsed["inference_id"]
        .as_str()
        .expect("inference_id should be a string")
        .parse()
        .expect("inference_id should be a UUID");

    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;
    let model_inferences = conn
        .get_model_inferences_by_inference_id(inference_id)
        .await
        .expect("model inferences lookup should succeed");
    assert_that!(
        model_inferences.len(),
        eq(1),
        "expected exactly one model inference for the test call"
    );
    let rendered_system = model_inferences[0]
        .system
        .as_deref()
        .expect("model inference should have a rendered system prompt");
    expect_that!(
        rendered_system,
        contains_substring(marker),
        "rendered system prompt should contain the marker from the modified template"
    );

    // Restore the original config.
    let post_apply = get_config_toml(&client).await;
    let restore_response = apply_config_toml(
        &client,
        &original.toml,
        &original.path_contents,
        &post_apply.base_signature,
    )
    .await;
    assert_that!(
        restore_response.status(),
        eq(StatusCode::OK),
        "restoring original config failed"
    );
}

/// Re-applying the canonical editable TOML should be a 200 no-op that
/// preserves the content-addressed hash. The apply handler no longer
/// short-circuits when `canonical_signature == current_signature` — it now
/// always goes through `write_stored_config_in_tx` — so this test also
/// exercises that the bulk write path is idempotent in the face of an
/// already-present identical config.
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_config_editing_noop_apply_preserves_hash() {
    let client = Client::new();
    let original = get_config_toml(&client).await;

    let response = apply_config_toml(
        &client,
        &original.toml,
        &original.path_contents,
        &original.base_signature,
    )
    .await;
    let status = response.status();
    let body_text = response
        .text()
        .await
        .expect("noop apply response body should decode");
    assert_that!(
        status,
        eq(StatusCode::OK),
        "noop apply should succeed; body={body_text}"
    );
    let parsed: ApplyConfigTomlResponse = serde_json::from_str(&body_text)
        .expect("noop apply body should deserialize as ApplyConfigTomlResponse");
    expect_that!(
        parsed.hash.as_str(),
        eq(original.hash.as_str()),
        "noop apply should return the same hash as the pre-apply GET"
    );
    expect_that!(
        parsed.base_signature.as_str(),
        eq(original.base_signature.as_str()),
        "noop apply should return the same base_signature as the pre-apply GET"
    );
}

/// File path of a template that exists on the e2e gateway's filesystem and is
/// used by the transitive-include regression tests below. Its content is a
/// single line (`assistant`) so a successful render for a template that
/// `{% include %}`s it will contain the word `assistant`.
const TRANSITIVE_INCLUDE_FS_TEMPLATE: &str = "extra_templates/foo.minijinja";

/// `creation_source` value the apply-TOML handler uses when writing stored
/// config rows. We query `stored_files` with this filter so the regression
/// test only counts rows written by `apply_config_toml_handler` (and not rows
/// left behind by other test fixtures, `--store-config`, or the single-
/// function edit path).
const APPLY_TOML_CREATION_SOURCE: &str = "ui-config-editor";

/// Regression test for PR #7185 review: transitively-included templates must
/// be persisted.
///
/// Before the fix, `apply_config_toml_handler` forwarded only
/// `request.path_contents` into `write_stored_config_in_tx` as
/// `extra_templates`, so any template discovered during validation via
/// MiniJinja `{% include %}` (from the gateway's `template_filesystem_access`
/// base path) was dropped on the floor. The apply would succeed and the
/// in-process gateway would continue to render correctly, but the persisted
/// config in the stored-config tables was no longer self-contained: a cold
/// reload on another gateway instance (or after a restart without filesystem
/// access) would fail to find the included template.
///
/// The fix merges `unwritten.extra_templates()` (the templates the validation
/// walker actually loaded) with the caller-provided path_contents inside
/// `write_stored_config_in_tx`, so the included file ends up in
/// `stored_files`. This test submits a top-level template whose body
/// transitively includes a filesystem-backed template and then asserts that
/// the included file is present in `stored_files` with the `ui-config-editor`
/// creation source — which is only possible if the merge happened.
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_config_editing_apply_persists_transitive_include_templates() {
    let client = Client::new();
    let original = get_config_toml(&client).await;

    // Name the function and the synthetic top-level template uniquely so
    // parallel e2e runs don't collide on either the function name or the
    // `stored_files` file path.
    let function_name = "e2e_config_editing_transitive_include_function";
    let top_template_path = "e2e_config_editing/transitive_include_system_template.minijinja";
    let top_template_body = format!(
        "You are a helpful {{% include '{TRANSITIVE_INCLUDE_FS_TEMPLATE}' %}} named {{{{ name }}}}."
    );

    // Intentionally do NOT put `extra_templates/foo.minijinja` in
    // `path_contents` — the whole point is that the MiniJinja walker should
    // discover it from the filesystem during validation and the fix should
    // then persist it.
    let mut path_contents = original.path_contents.clone();
    path_contents.insert(top_template_path.to_string(), top_template_body);

    let new_function_block = format!(
        r#"

[functions.{function_name}]
type = "chat"

[functions.{function_name}.variants.test]
type = "chat_completion"
model = "test"
system_template = "{top_template_path}"
"#
    );
    let new_toml = format!("{}{}", original.toml, new_function_block);

    let apply_response =
        apply_config_toml(&client, &new_toml, &path_contents, &original.base_signature).await;
    let apply_status = apply_response.status();
    let apply_body = apply_response
        .text()
        .await
        .expect("apply response body should decode");
    assert_that!(
        apply_status,
        eq(StatusCode::OK),
        "apply (transitive include) failed: {apply_body}"
    );

    // Query `stored_files` directly. Filter by `creation_source` so we only
    // see rows written by `apply_config_toml_handler` — the `--store-config`
    // CLI that bootstraps the e2e gateway uses a different creation source,
    // so a hit here can only come from the apply path under test.
    let postgres = get_postgres().await;
    let pool = postgres
        .get_pool()
        .expect("Postgres pool should be available for the e2e test");
    let count: i64 = sqlx::query_scalar(
        r"SELECT COUNT(*)::BIGINT
          FROM tensorzero.stored_files
          WHERE file_path = $1
            AND creation_source = $2",
    )
    .bind(TRANSITIVE_INCLUDE_FS_TEMPLATE)
    .bind(APPLY_TOML_CREATION_SOURCE)
    .fetch_one(pool)
    .await
    .expect("stored_files count query should succeed");
    expect_that!(
        count,
        ge(1),
        "transitive-include template `{TRANSITIVE_INCLUDE_FS_TEMPLATE}` should be \
         persisted in stored_files with creation_source=`{APPLY_TOML_CREATION_SOURCE}` \
         after apply (the review-fix for PR #7185 merges discovered templates \
         into the write path)"
    );

    // Sanity-check that an inference against the new function also renders
    // the transitively-included template — a second-level assertion that the
    // in-memory swap wired the discovered template through correctly.
    let inference_response = run_inference(&client, function_name).await;
    let inference_status = inference_response.status();
    let inference_body = inference_response
        .text()
        .await
        .expect("inference body should decode");
    assert_that!(
        inference_status,
        eq(StatusCode::OK),
        "inference against transitive-include function should succeed; body={inference_body}"
    );
    let parsed: Value =
        serde_json::from_str(&inference_body).expect("inference body should be JSON");
    let inference_id: Uuid = parsed["inference_id"]
        .as_str()
        .expect("inference_id should be a string")
        .parse()
        .expect("inference_id should be a UUID");

    let conn = DelegatingDatabaseConnection::new_for_e2e_test().await;
    conn.flush_pending_writes().await;
    conn.sleep_for_writes_to_be_visible().await;
    let model_inferences = conn
        .get_model_inferences_by_inference_id(inference_id)
        .await
        .expect("model inferences lookup should succeed");
    assert_that!(
        model_inferences.len(),
        eq(1),
        "expected exactly one model inference for the transitive-include test call"
    );
    let rendered_system = model_inferences[0]
        .system
        .as_deref()
        .expect("model inference should have a rendered system prompt");
    expect_that!(
        rendered_system,
        contains_substring("assistant"),
        "rendered system prompt should contain the body of the transitively included template"
    );

    // Restore the original config so later tests see a clean slate.
    let post_apply = get_config_toml(&client).await;
    let restore_response = apply_config_toml(
        &client,
        &original.toml,
        &original.path_contents,
        &post_apply.base_signature,
    )
    .await;
    assert_that!(
        restore_response.status(),
        eq(StatusCode::OK),
        "restoring original config failed"
    );
}

/// Validate endpoint should accept a config whose template transitively
/// includes another template from the gateway's `template_filesystem_access`
/// base path. Exercises the validate handler's happy path for filesystem
/// discovery — a client-side precheck before the user clicks `Apply`.
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_config_editing_validate_accepts_transitive_include() {
    let client = Client::new();

    let function_name = "e2e_config_editing_validate_transitive_include_function";
    let top_template_path =
        "e2e_config_editing/validate_transitive_include_system_template.minijinja";
    let top_template_body =
        format!("Top level template body {{% include '{TRANSITIVE_INCLUDE_FS_TEMPLATE}' %}}");

    let mut path_contents: HashMap<String, String> = HashMap::new();
    path_contents.insert(top_template_path.to_string(), top_template_body);

    let toml = format!(
        r#"
[functions.{function_name}]
type = "chat"

[functions.{function_name}.variants.test]
type = "chat_completion"
model = "test"
system_template = "{top_template_path}"
"#
    );

    let response = client
        .post(get_gateway_endpoint("/internal/config_toml/validate"))
        .json(&json!({
            "toml": toml,
            "path_contents": path_contents,
        }))
        .send()
        .await
        .expect("POST /internal/config_toml/validate should send");
    let status = response.status();
    let body = response
        .text()
        .await
        .expect("validate response body should decode");
    expect_that!(
        status,
        eq(StatusCode::OK),
        "validate should accept transitive-include template; body={body}"
    );
}

/// Regression test for PR #7185 review: the apply CAS check only compares the
/// caller's `base_signature` against the current DB signature — the earlier
/// special case that also accepted a matching canonical signature was removed
/// in the same commit. This test re-submits a *different* TOML against a
/// stale `base_signature` and verifies we get a 409, confirming the single-
/// comparison CAS still rejects edits built on top of stale snapshots.
#[gtest]
#[tokio::test(flavor = "multi_thread")]
async fn test_config_editing_cas_rejects_stale_base_signature_for_new_edit() {
    let client = Client::new();
    let original = get_config_toml(&client).await;

    // First, push an unrelated change so the live base signature advances
    // past `original.base_signature`.
    let churn_function = "e2e_config_editing_cas_churn_function";
    let churn_toml = append_new_function_block(&original.toml, churn_function);
    let churn_apply = apply_config_toml(
        &client,
        &churn_toml,
        &original.path_contents,
        &original.base_signature,
    )
    .await;
    assert_that!(
        churn_apply.status(),
        eq(StatusCode::OK),
        "churn apply (advance base signature) failed"
    );

    // Now attempt to apply a *different* edit using the now-stale
    // `original.base_signature`. This is the canonical "UI was editing off a
    // stale snapshot" case. With the simplified CAS check, this must 409.
    let stale_edit_function = "e2e_config_editing_cas_stale_edit_function";
    let post_churn = get_config_toml(&client).await;
    let stale_toml = append_new_function_block(&post_churn.toml, stale_edit_function);
    let stale_apply = apply_config_toml(
        &client,
        &stale_toml,
        &original.path_contents,
        &original.base_signature,
    )
    .await;
    let stale_status = stale_apply.status();
    let stale_body = stale_apply
        .text()
        .await
        .expect("stale apply body should decode");
    expect_that!(
        stale_status,
        eq(StatusCode::CONFLICT),
        "stale base_signature on a new edit must 409; body={stale_body}"
    );

    // Restore the original config.
    let post_stale = get_config_toml(&client).await;
    let restore_response = apply_config_toml(
        &client,
        &original.toml,
        &original.path_contents,
        &post_stale.base_signature,
    )
    .await;
    assert_that!(
        restore_response.status(),
        eq(StatusCode::OK),
        "restoring original config failed"
    );
}
