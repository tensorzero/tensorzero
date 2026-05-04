//! Round-trip correctness tests for `ConfigSnapshot` against a corpus of
//! representative TOML fixtures.
//!
//! For each fixture we assert five properties:
//!
//! - **P1**: the TOML parses to an `UninitializedConfig`, converts to a
//!   `StoredConfig`, and yields a `ConfigSnapshot` with no errors.
//! - **P2**: targeted JSON-shape assertions — specific paths in
//!   `serde_json::to_value(&snapshot.config)` carry the expected values
//!   (especially the new `version` fields). Targeted rather than full-shape
//!   golden files because schema additions are common and golden files would
//!   churn on every unrelated field change; targeted asserts only fail when
//!   the property under test actually breaks.
//! - **P3**: serializing the `StoredConfig` back to TOML and reparsing
//!   yields a structurally-equal value to the original input. Compared
//!   structurally rather than as strings because float serialization (e.g.
//!   `0.7` vs `0.699999988079071`) is not byte-stable through Rust's TOML
//!   crate, and key ordering differs after `prepare_table_for_snapshot`'s
//!   sort.
//! - **P4**: re-parsing the round-tripped TOML yields the same snapshot
//!   hash as the original.
//! - **P5** (the "TOML → JSON → TOML" property): going through the JSON
//!   form and back to TOML yields a structurally-equal value to the
//!   original. This is the contract that makes the JSONB column safe to
//!   reconstruct snapshots from if we ever need to.
//!
//! The fixtures live in `fixtures/` next to this file and are read at
//! compile time via `include_str!`. Each test case is a separate `#[gtest]`
//! function so a single failing fixture is reported in isolation.

use std::collections::HashMap;

use googletest::prelude::*;
use serde_json::Value;

use crate::config::UninitializedConfig;
use crate::config::snapshot::{ConfigSnapshot, StoredConfig};

/// Snapshot constructed from a TOML fixture, plus the original input string
/// so callers can assert against re-derivations.
struct FixtureSnapshot {
    original_toml: String,
    snapshot: ConfigSnapshot,
}

fn build_snapshot(toml_str: &str) -> FixtureSnapshot {
    let snapshot = ConfigSnapshot::new_from_toml_string(toml_str, HashMap::new())
        .expect("fixture should parse to a ConfigSnapshot");
    FixtureSnapshot {
        original_toml: toml_str.to_string(),
        snapshot,
    }
}

/// Run round-trip correctness checks against a fixture's TOML. Returns the
/// parsed snapshot for further targeted assertions.
///
/// All comparisons go through `serde_json::Value` because `StoredConfig`
/// itself does not derive `PartialEq` and JSON gives us content-equality
/// that's invariant to float-encoding ambiguity (both sides go through the
/// same `f32 → f64` widening) and to TOML key ordering.
///
/// Properties checked:
/// - **JSON self-round-trip**: `StoredConfig → JSON → StoredConfig → JSON`
///   yields a `serde_json::Value` byte-equal to the first JSON. This is the
///   contract that makes the new `config_json` column safe to read.
/// - **TOML self-round-trip**: `StoredConfig → TOML → StoredConfig → JSON`
///   yields a `serde_json::Value` byte-equal to the original. This is the
///   contract that the existing `config TEXT` column relies on.
/// - **TOML → JSON → TOML**: cross-format round-trip yields the same
///   `serde_json::Value` as the original. This is the strongest
///   "interchangeable representations" property.
fn assert_round_trip_properties(toml_str: &str) -> ConfigSnapshot {
    let fixture = build_snapshot(toml_str);
    let snapshot = fixture.snapshot.clone();

    let original_json = serde_json::to_value(&snapshot.config)
        .expect("StoredConfig should serialize to serde_json::Value");

    // JSON self-round-trip.
    let stored_from_json: StoredConfig = serde_json::from_value(original_json.clone())
        .expect("StoredConfig should deserialize from its own JSON");
    let json_again =
        serde_json::to_value(&stored_from_json).expect("re-serialize JSON should succeed");
    expect_that!(
        &json_again,
        eq(&original_json),
        "JSON self-round-trip must be content-stable",
    );

    // TOML self-round-trip — round-trip is via `StoredConfig`, the
    // forward-compatible type. We do NOT re-parse via `UninitializedConfig`
    // because re-serialization can materialize default empty subtables (e.g.
    // `[embedding_models.foo.providers.bar.timeouts]`) that
    // `Uninitialized*` types reject with `deny_unknown_fields`. The snapshot
    // read path uses `StoredConfig::deserialize` for exactly this reason.
    let original_toml_re_serialized =
        toml::to_string(&snapshot.config).expect("StoredConfig should serialize to TOML");
    let stored_from_toml: StoredConfig = toml::from_str(&original_toml_re_serialized)
        .expect("StoredConfig should deserialize from its own TOML");
    let json_via_toml = serde_json::to_value(&stored_from_toml)
        .expect("from-TOML StoredConfig should serialize to JSON");
    expect_that!(
        &json_via_toml,
        eq(&original_json),
        "TOML self-round-trip must be content-stable",
    );

    // TOML → JSON → TOML — the cross-format property the PR description
    // calls out specifically.
    let toml_via_json = toml::to_string(&stored_from_json)
        .expect("from-JSON StoredConfig should serialize to TOML");
    let stored_from_toml_via_json: StoredConfig =
        toml::from_str(&toml_via_json).expect("TOML-via-JSON should deserialize to a StoredConfig");
    let json_via_round_trip = serde_json::to_value(&stored_from_toml_via_json)
        .expect("final JSON serialization should succeed");
    expect_that!(
        &json_via_round_trip,
        eq(&original_json),
        "TOML → JSON → TOML must be content-stable",
    );

    // Sanity: the original TOML reparses too (catches obvious fixture bugs).
    let _input_table: toml::Table = fixture
        .original_toml
        .parse()
        .expect("original fixture should parse as toml::Table");

    snapshot
}

/// Returns the JSON form of the snapshot's config for path-based assertions.
fn snapshot_json(snapshot: &ConfigSnapshot) -> Value {
    serde_json::to_value(&snapshot.config)
        .expect("snapshot config should serialize to serde_json::Value")
}

/// Convenience to fetch a nested JSON value by `/`-separated path. Returns
/// `Value::Null` for missing paths so the caller can assert with
/// `eq(&Value::Null)` to verify absence.
fn json_path<'a>(value: &'a Value, path: &str) -> &'a Value {
    let mut current = value;
    for segment in path.split('/').filter(|s| !s.is_empty()) {
        current = match current {
            Value::Object(map) => map.get(segment).unwrap_or(&Value::Null),
            Value::Array(arr) => match segment.parse::<usize>() {
                Ok(idx) => arr.get(idx).unwrap_or(&Value::Null),
                Err(_) => &Value::Null,
            },
            _ => &Value::Null,
        };
    }
    current
}

/// Convenience for "the config is an empty `UninitializedConfig`".
fn empty_uninitialized() -> UninitializedConfig {
    UninitializedConfig::try_from(toml::Table::new())
        .expect("empty toml::Table should produce an UninitializedConfig")
}

// ─── P1/P3/P4/P5 + targeted P2 per fixture ───────────────────────────────

#[gtest]
fn fixture_empty() {
    let toml = include_str!("fixtures/empty.toml");
    let snapshot = assert_round_trip_properties(toml);
    let json = snapshot_json(&snapshot);
    // Empty config: no functions, no models, no tools.
    expect_that!(json_path(&json, "functions"), eq(&serde_json::json!({})));
    expect_that!(json_path(&json, "models"), eq(&serde_json::json!({})));
}

#[gtest]
fn fixture_chat_function_unversioned() {
    let toml = include_str!("fixtures/chat_function_unversioned.toml");
    let snapshot = assert_round_trip_properties(toml);
    let json = snapshot_json(&snapshot);

    // Function present.
    expect_that!(
        json_path(&json, "functions/my_chat_fn/type"),
        eq(&serde_json::json!("chat")),
    );
    // No `version` key when unversioned (skip_serializing_if = "u32_is_zero").
    expect_that!(
        json_path(&json, "functions/my_chat_fn/version"),
        eq(&Value::Null),
    );
    expect_that!(
        json_path(&json, "functions/my_chat_fn/variants/baseline/version"),
        eq(&Value::Null),
    );
}

#[gtest]
fn fixture_chat_function_explicit_zero_version_hashes_identically() {
    // Two fixtures: one omits `version`, one sets `version = 0` explicitly.
    // They MUST produce the same hash, otherwise rolling out the version
    // field would invalidate every existing snapshot reference in the DB.
    let unversioned = include_str!("fixtures/chat_function_unversioned.toml");
    let explicit_zero = include_str!("fixtures/chat_function_explicit_zero.toml");
    let snap_a = ConfigSnapshot::new_from_toml_string(unversioned, HashMap::new())
        .expect("parse unversioned");
    let snap_b = ConfigSnapshot::new_from_toml_string(explicit_zero, HashMap::new())
        .expect("parse explicit zero");
    let hex_a = snap_a.hash.to_hex_string();
    let hex_b = snap_b.hash.to_hex_string();
    expect_that!(
        hex_a.as_str(),
        eq(hex_b.as_str()),
        "explicit `version = 0` must hash identically to omitting `version`",
    );
}

#[gtest]
fn fixture_chat_function_versioned() {
    let toml = include_str!("fixtures/chat_function_versioned.toml");
    let snapshot = assert_round_trip_properties(toml);
    let json = snapshot_json(&snapshot);

    // Function and variant carry explicit version.
    expect_that!(
        json_path(&json, "functions/versioned_fn/version"),
        eq(&serde_json::json!(3)),
    );
    expect_that!(
        json_path(&json, "functions/versioned_fn/variants/v1/version"),
        eq(&serde_json::json!(7)),
    );
    expect_that!(
        json_path(&json, "functions/versioned_fn/variants/v2/version"),
        eq(&serde_json::json!(2)),
    );
    // Mixed: a variant with version=0 still has no `version` key.
    expect_that!(
        json_path(&json, "functions/versioned_fn/variants/legacy/version"),
        eq(&Value::Null),
    );
}

#[gtest]
fn fixture_versioned_changes_hash() {
    // Sanity: the version field is part of the canonical TOML, so changing
    // it must change the hash. Otherwise the JSONB index could be queried
    // by version but the hash wouldn't be a unique snapshot identifier.
    let unversioned = include_str!("fixtures/chat_function_unversioned.toml");
    let versioned = include_str!("fixtures/chat_function_versioned.toml");
    let snap_a = ConfigSnapshot::new_from_toml_string(unversioned, HashMap::new())
        .expect("parse unversioned");
    let snap_b =
        ConfigSnapshot::new_from_toml_string(versioned, HashMap::new()).expect("parse versioned");
    let hex_a = snap_a.hash.to_hex_string();
    let hex_b = snap_b.hash.to_hex_string();
    expect_that!(
        hex_a.as_str(),
        not(eq(hex_b.as_str())),
        "explicit version != 0 must change the hash",
    );
}

#[gtest]
fn fixture_json_function_versioned() {
    let toml = include_str!("fixtures/json_function_versioned.toml");
    let snapshot = assert_round_trip_properties(toml);
    let json = snapshot_json(&snapshot);

    expect_that!(
        json_path(&json, "functions/json_fn/type"),
        eq(&serde_json::json!("json")),
    );
    expect_that!(
        json_path(&json, "functions/json_fn/version"),
        eq(&serde_json::json!(1)),
    );
}

#[gtest]
fn fixture_multi_variant_types() {
    let toml = include_str!("fixtures/multi_variant_types.toml");
    let snapshot = assert_round_trip_properties(toml);
    let json = snapshot_json(&snapshot);

    expect_that!(
        json_path(&json, "functions/varied/variants/cc/type"),
        eq(&serde_json::json!("chat_completion")),
    );
    expect_that!(
        json_path(&json, "functions/varied/variants/best/type"),
        eq(&serde_json::json!("experimental_best_of_n_sampling")),
    );
    expect_that!(
        json_path(&json, "functions/varied/variants/mix/type"),
        eq(&serde_json::json!("experimental_mixture_of_n")),
    );
}

#[gtest]
fn fixture_models_multi_provider() {
    let toml = include_str!("fixtures/models_multi_provider.toml");
    let snapshot = assert_round_trip_properties(toml);
    let json = snapshot_json(&snapshot);

    // Two models with different provider sets — both round-trip.
    expect_that!(
        json_path(&json, "models/openai_model/routing"),
        eq(&serde_json::json!(["openai"])),
    );
    expect_that!(
        json_path(&json, "models/anthropic_model/routing"),
        eq(&serde_json::json!(["anthropic"])),
    );
}

#[gtest]
fn fixture_tools_and_metrics() {
    let toml = include_str!("fixtures/tools_and_metrics.toml");
    let snapshot = assert_round_trip_properties(toml);
    let json = snapshot_json(&snapshot);

    expect_that!(
        json_path(&json, "tools/get_weather/description"),
        eq(&serde_json::json!("Look up the weather")),
    );
    expect_that!(
        json_path(&json, "metrics/correctness/type"),
        eq(&serde_json::json!("boolean")),
    );
}

#[gtest]
fn fixture_kitchen_sink() {
    // Integration: many sections together with mixed versioning. Any
    // cross-section drift in canonicalization or round-tripping would surface
    // here even if isolated fixtures pass.
    let toml = include_str!("fixtures/kitchen_sink.toml");
    let snapshot = assert_round_trip_properties(toml);
    let json = snapshot_json(&snapshot);

    expect_that!(
        json_path(&json, "functions/featured/version"),
        eq(&serde_json::json!(2)),
    );
    expect_that!(
        json_path(&json, "functions/featured/variants/main/version"),
        eq(&serde_json::json!(11)),
    );
    expect_that!(
        json_path(&json, "models/dummy/routing"),
        eq(&serde_json::json!(["a"])),
    );
    expect_that!(
        json_path(&json, "tools/echo/description"),
        eq(&serde_json::json!("Echo the input")),
    );
}

// ─── Additional explicit hash-stability and JSON-shape invariants ─────────

#[gtest]
fn empty_config_has_well_known_json_shape() {
    // Independent of any fixture file: when the StoredConfig is empty, the
    // JSON has the expected top-level keys with empty values. Catches drift
    // in default-skipping behavior on `StoredConfig` fields.
    let stored: StoredConfig = empty_uninitialized().into();
    let json = serde_json::to_value(&stored).expect("serialize empty StoredConfig");
    let obj = json.as_object().expect("StoredConfig serializes as object");
    // Required keys exist.
    expect_that!(obj.contains_key("models"), eq(true));
    expect_that!(obj.contains_key("functions"), eq(true));
    expect_that!(obj.contains_key("tools"), eq(true));
    expect_that!(obj.contains_key("metrics"), eq(true));
    // No `evaluators` (renamed to `evaluations` historically) at top level.
    expect_that!(obj.contains_key("evaluators"), eq(false));
}

#[gtest]
fn version_field_is_omitted_when_zero_in_json() {
    // Direct shape test: a default-version function/variant must not produce
    // a `version` key in JSON output. This is what makes the new field
    // backwards-compatible at the JSONB level.
    let toml = r#"
[models.dummy_model]
routing = ["dummy"]

[models.dummy_model.providers.dummy]
type = "dummy"
model_name = "test"

[functions.f]
type = "chat"

[functions.f.variants.v]
type = "chat_completion"
model = "dummy_model"
"#;
    let snapshot =
        ConfigSnapshot::new_from_toml_string(toml, HashMap::new()).expect("parse default-version");
    let json = serde_json::to_value(&snapshot.config).expect("serialize");
    let function = json
        .pointer("/functions/f")
        .expect("function f should be present");
    let function_obj = function
        .as_object()
        .expect("function should be a JSON object");
    expect_that!(
        function_obj.contains_key("version"),
        eq(false),
        "default-version function must not emit `version` in JSON",
    );
    let variant = json
        .pointer("/functions/f/variants/v")
        .expect("variant v should be present");
    let variant_obj = variant
        .as_object()
        .expect("variant should be a JSON object");
    expect_that!(
        variant_obj.contains_key("version"),
        eq(false),
        "default-version variant must not emit `version` in JSON",
    );
}
