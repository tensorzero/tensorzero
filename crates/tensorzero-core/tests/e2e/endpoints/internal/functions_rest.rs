//! E2E tests for the narrow per-object REST endpoints
//! (`/internal/functions`, `/internal/functions/{name}/variants`).
//!
//! These run against an HTTP gateway booted in **config-in-database mode**
//! with an empty initial DB — the same shape the UI walkthrough uses.
//! The CI lane that runs them is `live-tests-config-in-database-empty.yml`
//! (added separately).
//!
//! ## What we exercise
//!
//! 1. `POST /internal/functions` — first write seeds the DB
//! 2. `GET /internal/functions` — appears in the list
//! 3. `GET /internal/functions/{name}` — current shape
//! 4. `POST /internal/functions/{name}/variants` — second variant
//! 5. `GET /internal/functions/{name}/variants` — both variants returned
//! 6. `POST /inference` — using the dummy provider, both variants serve traffic
//! 7. `PATCH /internal/functions/{name}/variants/{variant}` — version auto-bumps
//! 8. `DELETE /internal/functions/{name}/variants/{variant}` — variant gone
//! 9. `DELETE /internal/functions/{name}` — function tombstoned
//!
//! Each test uses unique random names (`Uuid`-suffixed) so it doesn't
//! collide with other tests' state in a shared DB. Tests are
//! self-cleaning via DELETE.
//!
//! ## Skip conditions
//!
//! Tests skip if `/status` reports the gateway isn't in
//! config-in-database mode (file mode rejects these endpoints with 400
//! by design). This means the same test file can be linked into a
//! suite that runs against either kind of gateway, and the
//! file-mode runs simply skip rather than fail.

use std::time::Duration;
use tensorzero::test_helpers::get_gateway_endpoint;
use tensorzero_core::endpoints::internal::functions_rest::ConfigEditResult;
use uuid::Uuid;

// Each mutating endpoint here triggers a global hot-swap (read all
// functions+variants from Postgres → write snapshot to the observability
// backend → swap the live config). When tests run in parallel they can
// race on that pipeline, surfacing as occasional 500s — particularly in
// the two-backend setup (config in PG, observability in CH) where the
// snapshot write hops to a different DB. The
// `e2e_functions_rest` test-group in `crates/.config/nextest.toml` pins
// these to `max-threads = 1` so they serialize.

/// Probe the gateway's `/status` endpoint and return whether it appears
/// to be in config-in-database mode. We use a heuristic: the
/// `/internal/functions` endpoint returns 200 in DB mode and 400 in file
/// mode, so we just probe that directly. Cheap and unambiguous.
async fn config_in_database_mode(client: &reqwest::Client) -> bool {
    let url = get_gateway_endpoint("/internal/functions");
    match client
        .get(&url)
        .timeout(Duration::from_secs(5))
        .send()
        .await
    {
        Ok(response) => response.status().as_u16() == 200,
        Err(_) => false,
    }
}

fn unique_function_name(prefix: &str) -> String {
    format!("{prefix}_{}", Uuid::now_v7().simple())
}

fn http() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .expect("reqwest client should build")
}

/// `POST /internal/functions` against an empty DB → 200, the function
/// appears in `GET /internal/functions`.
#[tokio::test(flavor = "multi_thread")]
async fn create_function_then_list() {
    let client = http();
    if !config_in_database_mode(&client).await {
        // Silently skip when the gateway isn't in config-in-database mode.
        // The CI lane that does run them is `live-tests-config-in-database`;
        // other lanes link this file but treat it as a no-op.
        return;
    }

    let name = unique_function_name("e2e_fn");
    let create_body = serde_json::json!({
        "name": name,
        "config": {
            "type": "chat",
            "variants": {
                "default": { "type": "chat_completion", "model": "dummy::good" }
            }
        }
    });
    let response = client
        .post(get_gateway_endpoint("/internal/functions"))
        .json(&create_body)
        .send()
        .await
        .expect("POST /internal/functions should succeed");
    assert_eq!(response.status(), 200, "create body: {create_body:?}");
    let result: ConfigEditResult = response.json().await.expect("parse create response");
    assert_eq!(result.function.function_name, name);

    // List should include the new function.
    let list = client
        .get(get_gateway_endpoint("/internal/functions"))
        .send()
        .await
        .unwrap();
    assert_eq!(list.status(), 200);
    let body: serde_json::Value = list.json().await.unwrap();
    let functions = body.get("functions").and_then(|v| v.as_array()).unwrap();
    let has_ours = functions
        .iter()
        .any(|f| f.get("name").and_then(|n| n.as_str()) == Some(name.as_str()));
    assert!(has_ours, "list should include {name}; got {body}");

    // Cleanup.
    let _ = client
        .delete(get_gateway_endpoint(&format!("/internal/functions/{name}")))
        .json(&serde_json::json!({
            "expected_current_function_version_id": result.function.function_version_id,
        }))
        .send()
        .await;
}

/// `POST /internal/functions/{name}/variants` adds a second variant
/// and `GET /internal/functions/{name}/variants` returns both. The
/// inference half of the round-trip lives in `empty_bootstrap_to_inference`,
/// which uses a real provider — the `db-only-boot` CI lane runs the
/// production gateway image, which has no `dummy::*` provider.
#[tokio::test(flavor = "multi_thread")]
async fn add_variant_then_list_both() {
    let client = http();
    if !config_in_database_mode(&client).await {
        // Silently skip when the gateway isn't in config-in-database mode.
        // The CI lane that does run them is `live-tests-config-in-database`;
        // other lanes link this file but treat it as a no-op.
        return;
    }

    let name = unique_function_name("e2e_var");
    let create = client
        .post(get_gateway_endpoint("/internal/functions"))
        .json(&serde_json::json!({
            "name": name,
            "config": {
                "type": "chat",
                "variants": {
                    "default": { "type": "chat_completion", "model": "dummy::good" }
                }
            }
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(create.status(), 200);
    let create: ConfigEditResult = create.json().await.unwrap();

    // Add a second variant. CAS must match the current version_id.
    let add_variant = client
        .post(get_gateway_endpoint(&format!(
            "/internal/functions/{name}/variants"
        )))
        .json(&serde_json::json!({
            "variant_name": "alt",
            "config": { "type": "chat_completion", "model": "dummy::good" },
            "expected_current_function_version_id": create.function.function_version_id,
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        add_variant.status(),
        200,
        "add variant body: {:?}",
        add_variant.text().await.ok()
    );
    let add_variant: ConfigEditResult = add_variant.json().await.unwrap();

    // Both variants visible via GET.
    let variants = client
        .get(get_gateway_endpoint(&format!(
            "/internal/functions/{name}/variants"
        )))
        .send()
        .await
        .unwrap();
    let body: serde_json::Value = variants.json().await.unwrap();
    let names: Vec<&str> = body
        .get("variants")
        .and_then(|v| v.as_array())
        .unwrap()
        .iter()
        .filter_map(|v| v.get("name").and_then(|n| n.as_str()))
        .collect();
    assert!(
        names.contains(&"default") && names.contains(&"alt"),
        "expected both default and alt; got {names:?}"
    );

    // Cleanup — delete function (CAS).
    let _ = client
        .delete(get_gateway_endpoint(&format!("/internal/functions/{name}")))
        .json(&serde_json::json!({
            "expected_current_function_version_id": add_variant.function.function_version_id,
        }))
        .send()
        .await;
}

/// `PATCH /internal/functions/{name}/variants/{variant}` auto-bumps the
/// variant's `version` field. List returns the new version.
#[tokio::test(flavor = "multi_thread")]
async fn patch_variant_auto_bumps_version() {
    let client = http();
    if !config_in_database_mode(&client).await {
        // Silently skip when the gateway isn't in config-in-database mode.
        // The CI lane that does run them is `live-tests-config-in-database`;
        // other lanes link this file but treat it as a no-op.
        return;
    }

    let name = unique_function_name("e2e_patch");
    let create = client
        .post(get_gateway_endpoint("/internal/functions"))
        .json(&serde_json::json!({
            "name": name,
            "config": {
                "type": "chat",
                "variants": {
                    "default": { "type": "chat_completion", "model": "dummy::good" }
                }
            }
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(create.status(), 200);
    let create: ConfigEditResult = create.json().await.unwrap();

    // PATCH the variant. The body's `version` is intentionally 99 — the
    // server should ignore it and write `current + 1` (i.e. 1).
    let patch = client
        .patch(get_gateway_endpoint(&format!(
            "/internal/functions/{name}/variants/default"
        )))
        .json(&serde_json::json!({
            "config": { "type": "chat_completion", "model": "dummy::good", "version": 99 },
            "expected_current_function_version_id": create.function.function_version_id,
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        patch.status(),
        200,
        "patch should 200; got body: {:?}",
        patch.text().await.ok()
    );
    let patched: ConfigEditResult = patch.json().await.unwrap();

    // List the variant — `version` must be 1, not 99 (server overrides).
    let variants = client
        .get(get_gateway_endpoint(&format!(
            "/internal/functions/{name}/variants"
        )))
        .send()
        .await
        .unwrap();
    let body: serde_json::Value = variants.json().await.unwrap();
    let default_variant = body
        .get("variants")
        .and_then(|v| v.as_array())
        .unwrap()
        .iter()
        .find(|v| v.get("name").and_then(|n| n.as_str()) == Some("default"))
        .expect("default variant should be in list");
    let version = default_variant
        .get("version")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    assert_eq!(
        version, 1,
        "PATCH should bump version 0 → 1, ignoring client-supplied 99",
    );

    // PATCH again — version should bump to 2.
    let patch2 = client
        .patch(get_gateway_endpoint(&format!(
            "/internal/functions/{name}/variants/default"
        )))
        .json(&serde_json::json!({
            "config": { "type": "chat_completion", "model": "dummy::good" },
            "expected_current_function_version_id": patched.function.function_version_id,
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(patch2.status(), 200);

    let variants = client
        .get(get_gateway_endpoint(&format!(
            "/internal/functions/{name}/variants"
        )))
        .send()
        .await
        .unwrap();
    let body: serde_json::Value = variants.json().await.unwrap();
    let version = body
        .get("variants")
        .and_then(|v| v.as_array())
        .unwrap()
        .iter()
        .find(|v| v.get("name").and_then(|n| n.as_str()) == Some("default"))
        .and_then(|v| v.get("version"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    assert_eq!(version, 2, "second PATCH should bump 1 → 2");
}

/// Lifecycle: create → add variant → delete variant → delete function.
/// Each delete advances the function's version_id and we CAS against
/// the latest one.
#[tokio::test(flavor = "multi_thread")]
async fn full_lifecycle() {
    let client = http();
    if !config_in_database_mode(&client).await {
        // Silently skip when the gateway isn't in config-in-database mode.
        // The CI lane that does run them is `live-tests-config-in-database`;
        // other lanes link this file but treat it as a no-op.
        return;
    }

    let name = unique_function_name("e2e_life");
    let create = client
        .post(get_gateway_endpoint("/internal/functions"))
        .json(&serde_json::json!({
            "name": name,
            "config": {
                "type": "chat",
                "variants": {
                    "default": { "type": "chat_completion", "model": "dummy::good" }
                }
            }
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(create.status(), 200);
    let create: ConfigEditResult = create.json().await.unwrap();

    let add = client
        .post(get_gateway_endpoint(&format!(
            "/internal/functions/{name}/variants"
        )))
        .json(&serde_json::json!({
            "variant_name": "alt",
            "config": { "type": "chat_completion", "model": "dummy::good" },
            "expected_current_function_version_id": create.function.function_version_id,
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(add.status(), 200);
    let add: ConfigEditResult = add.json().await.unwrap();

    let del_variant = client
        .delete(get_gateway_endpoint(&format!(
            "/internal/functions/{name}/variants/alt"
        )))
        .json(&serde_json::json!({
            "expected_current_function_version_id": add.function.function_version_id,
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        del_variant.status(),
        200,
        "delete variant body: {:?}",
        del_variant.text().await.ok()
    );
    let del_variant: ConfigEditResult = del_variant.json().await.unwrap();

    let del_fn = client
        .delete(get_gateway_endpoint(&format!("/internal/functions/{name}")))
        .json(&serde_json::json!({
            "expected_current_function_version_id": del_variant.function.function_version_id,
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(del_fn.status(), 200);

    // Function should no longer be in the active list.
    let list = client
        .get(get_gateway_endpoint("/internal/functions"))
        .send()
        .await
        .unwrap();
    let body: serde_json::Value = list.json().await.unwrap();
    let still_present = body
        .get("functions")
        .and_then(|v| v.as_array())
        .unwrap()
        .iter()
        .any(|f| f.get("name").and_then(|n| n.as_str()) == Some(name.as_str()));
    assert!(!still_present, "function should be gone after DELETE");
}

/// CAS: editing a variant with a stale `expected_current_function_version_id`
/// returns a friendly 400 error rather than silently overwriting.
#[tokio::test(flavor = "multi_thread")]
async fn stale_cas_is_rejected() {
    let client = http();
    if !config_in_database_mode(&client).await {
        // Silently skip when the gateway isn't in config-in-database mode.
        // The CI lane that does run them is `live-tests-config-in-database`;
        // other lanes link this file but treat it as a no-op.
        return;
    }

    let name = unique_function_name("e2e_cas");
    let create = client
        .post(get_gateway_endpoint("/internal/functions"))
        .json(&serde_json::json!({
            "name": name,
            "config": {
                "type": "chat",
                "variants": {
                    "default": { "type": "chat_completion", "model": "dummy::good" }
                }
            }
        }))
        .send()
        .await
        .unwrap();
    let create: ConfigEditResult = create.json().await.unwrap();

    // First successful PATCH so the version_id advances.
    let _ = client
        .patch(get_gateway_endpoint(&format!(
            "/internal/functions/{name}/variants/default"
        )))
        .json(&serde_json::json!({
            "config": { "type": "chat_completion", "model": "dummy::good" },
            "expected_current_function_version_id": create.function.function_version_id,
        }))
        .send()
        .await
        .unwrap();

    // Second PATCH with the STALE original version_id — should reject.
    let stale = client
        .patch(get_gateway_endpoint(&format!(
            "/internal/functions/{name}/variants/default"
        )))
        .json(&serde_json::json!({
            "config": { "type": "chat_completion", "model": "dummy::good" },
            "expected_current_function_version_id": create.function.function_version_id,
        }))
        .send()
        .await
        .unwrap();
    assert!(
        stale.status().is_client_error(),
        "stale CAS should produce a 4xx; got {}",
        stale.status()
    );
}

/// Bootstrap from an empty DB-config gateway: no `--config-file`, no
/// fixtures, nothing pre-seeded.
///
/// 1. Create a function with a single OpenAI-backed default variant.
/// 2. Wait (up to 10s) for the observability snapshot to be readable
///    via `GET /internal/config/{hash}` — the mutating endpoint returns
///    the freshly-taken snapshot hash, but in batch-writes mode the
///    snapshot row may be persisted asynchronously, so we poll.
/// 3. Issue a chat inference against the function.
/// 4. Verify observability sees the call by polling
///    `GET /internal/functions/{name}/inference_count` for `>= 1`.
///
/// This is the only test in this file that depends on real provider
/// credentials (OpenAI). It self-skips when `OPENAI_API_KEY` is unset so
/// developer machines without the key still pass the rest of the suite.
#[tokio::test(flavor = "multi_thread")]
async fn empty_bootstrap_to_inference() {
    let client = http();
    if !config_in_database_mode(&client).await {
        // Silently skip when the gateway isn't in config-in-database mode.
        // The CI lane that does run them is `live-tests-config-in-database`;
        // other lanes link this file but treat it as a no-op.
        return;
    }
    if std::env::var("OPENAI_API_KEY")
        .map(|v| v.is_empty())
        .unwrap_or(true)
    {
        // Skip silently when OpenAI credentials aren't available — keeps
        // the test runnable on developer machines without leaking the key
        // requirement into the test runner output.
        return;
    }

    let name = unique_function_name("e2e_boot");
    let create_body = serde_json::json!({
        "name": name,
        "config": {
            "type": "chat",
            "variants": {
                "default": {
                    "type": "chat_completion",
                    "model": "openai::gpt-4o-mini",
                }
            }
        }
    });
    let create = client
        .post(get_gateway_endpoint("/internal/functions"))
        .json(&create_body)
        .send()
        .await
        .expect("POST /internal/functions should succeed");
    assert_eq!(
        create.status(),
        200,
        "create body: {create_body:?}; resp: {:?}",
        create.text().await.ok(),
    );
    let create: ConfigEditResult = create.json().await.expect("parse create response");
    let snapshot_hash = create.snapshot_hash.to_string();

    // Poll the observability backend for the snapshot. In batch-writes
    // mode the row is flushed asynchronously; with no batching this
    // succeeds on the first try. 10s is generous for either backend.
    let snapshot_url = get_gateway_endpoint(&format!("/internal/config/{snapshot_hash}"));
    let snapshot_deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        let resp = client.get(&snapshot_url).send().await.unwrap();
        let status = resp.status();
        if status == 200 {
            break;
        }
        assert!(
            std::time::Instant::now() < snapshot_deadline,
            "snapshot {snapshot_hash} never became visible at {snapshot_url}; last status {status}",
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    // Issue an inference. We use the cheapest OpenAI model and a tiny
    // prompt to keep CI cost in line with the rest of the live-tests
    // matrix.
    let inference = client
        .post(get_gateway_endpoint("/inference"))
        .json(&serde_json::json!({
            "function_name": name,
            "input": { "messages": [{"role": "user", "content": "Say hi"}] }
        }))
        .send()
        .await
        .expect("POST /inference should send");
    assert_eq!(
        inference.status(),
        200,
        "inference resp: {:?}",
        inference.text().await.ok(),
    );

    // Poll the observability backend for the inference. Same 10s budget;
    // inference writes flow through the same batched-writes path.
    let count_url = get_gateway_endpoint(&format!("/internal/functions/{name}/inference_count"));
    let count_deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        let resp = client.get(&count_url).send().await.unwrap();
        if resp.status() == 200 {
            let body: serde_json::Value = resp.json().await.unwrap();
            let count = body
                .get("inference_count")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            if count >= 1 {
                break;
            }
        }
        assert!(
            std::time::Instant::now() < count_deadline,
            "inference never landed in observability for function `{name}`",
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    // Verify the inference row's `snapshot_hash` is the **canonical
    // hash**, not the legacy hash. This is the load-bearing property
    // of the canonical-by-default migration: every new inference /
    // feedback / datapoint row writes `Config.hash` as its
    // `snapshot_hash`, and `Config.hash` is now canonical.
    //
    // `InferenceMetadata.snapshot_hash` is the lowercase hex form of
    // the row's bytes (see `impl FromRow for InferenceMetadata`).
    // The live `Config.hash` from `/status` carries the `can:` prefix
    // and the bare-decimal form. We convert the live hash's decimal
    // back to bytes (via the FromStr → as_bytes path) and hex it to
    // compare apples-to-apples.
    let metadata_url = get_gateway_endpoint(&format!(
        "/internal/inference_metadata?function_name={name}"
    ));
    let metadata: serde_json::Value = client
        .get(&metadata_url)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let inference_meta = metadata
        .get("inference_metadata")
        .and_then(|v| v.as_array())
        .and_then(|arr| arr.first())
        .expect("at least one inference_metadata row");
    let stored_hex = inference_meta
        .get("snapshot_hash")
        .and_then(|v| v.as_str())
        .expect("inference row should carry a snapshot_hash");

    let status: serde_json::Value = client
        .get(get_gateway_endpoint("/status"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let live_hash = status
        .get("config_hash")
        .and_then(|v| v.as_str())
        .expect("status response should carry config_hash");
    assert!(
        live_hash.starts_with("can:"),
        "status.config_hash must carry the `can:` prefix (canonical-by-default); got: {live_hash}",
    );
    let live_canonical: tensorzero_core::config::snapshot::SnapshotHash =
        live_hash.parse().expect("FromStr on can:DECIMAL");
    assert_eq!(
        stored_hex,
        live_canonical.to_hex_string(),
        "inference row's snapshot_hash must equal the bytes of the live canonical Config.hash",
    );

    // Cleanup.
    let _ = client
        .delete(get_gateway_endpoint(&format!("/internal/functions/{name}")))
        .json(&serde_json::json!({
            "expected_current_function_version_id": create.function.function_version_id,
        }))
        .send()
        .await;
}
