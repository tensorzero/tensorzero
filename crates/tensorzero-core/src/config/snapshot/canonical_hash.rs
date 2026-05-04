//! Structural hashing for `StoredConfig`.
//!
//! The historical hash on `main` is computed from canonical *TOML bytes*.
//! That makes it sensitive to things that aren't logical config changes —
//! float reformatting (`0.7` → `0.6999999…`), TOML crate version bumps,
//! and tweaks to canonicalization rules all drift the hash even when the
//! semantic content is unchanged.
//!
//! Drift doesn't invalidate existing data: every old `config_snapshots`
//! row keeps its old hash, every old `inferences.snapshot_hash` reference
//! keeps resolving, and a drifted next-boot just writes a new row — no
//! different from the user editing the config file. What drift *does*
//! cost is identity:
//!
//! - **Content-addressed dedupe.** Two gateway versions running the same
//!   logical config should write *one* `config_snapshots` row, not one
//!   per toml-crate version. The TOML-bytes hash can't guarantee that.
//! - **Multi-gateway consistency.** Two gateways with different transitive
//!   dependency versions running against the same DB should agree on
//!   what hash the same config produces.
//! - **JSONB roundtrip identity.** Once the snapshot row stores
//!   `config_jsonb` (PR #2 of this stack), re-deriving the hash from
//!   the stored JSON should match the stored hash — otherwise reads
//!   that recompute can't verify they got the same content back.
//!
//! The structural hash addresses all three by operating on the **logical
//! content** of the config rather than its serialized text:
//!
//! - Preserved by every `StoredConfig → JSON → StoredConfig` and
//!   `StoredConfig → TOML → StoredConfig` round-trip.
//! - Independent of third-party crate formatting choices.
//! - Stable across machine architectures and process restarts.
//!
//! The implementation walks the `serde_json::Value` form of the config
//! with a self-describing canonical encoding:
//!
//! - **Type tag** (1 byte) for every node — null/bool/number/string/array/object
//! - **Length prefix** (8 bytes, big-endian) for strings, arrays, and
//!   objects — prevents collisions like `["ab"]` ↔ `["a","b"]` or
//!   `{"a":"b"}` ↔ `{"ab":""}`
//! - **Sorted keys** for objects (canonical order)
//! - **f64 IEEE 754 big-endian bit pattern** for numbers (deterministic
//!   even when `serde_json::Number` round-trips through different textual
//!   forms)
//!
//! `serde_json::Value` is the canonical intermediate. `StoredConfig` →
//! `Value` is deterministic via `serde_json::to_value`; `Value` → bytes
//! is whatever this module decides; this module is the only source of
//! truth for the canonical encoding.
//!
//! # Schema changes — what drifts the hash, and what it costs you
//!
//! The hash is computed by walking `serde_json::to_value(self)` for a
//! given `StoredConfig`. So the **JSON shape** of `StoredConfig` is
//! the contract — anything that changes the JSON shape changes the
//! hash, even if the logical config is the same. The categories are:
//!
//! | Change to `StoredConfig` shape                                                                  | Drifts the hash? |
//! |-------------------------------------------------------------------------------------------------|------------------|
//! | Add `Option<T>` field with `#[serde(skip_serializing_if = "Option::is_none")]`, default `None`   | **No**           |
//! | Add `Option<T>` field that newly serializes a default value (e.g. `Some(...)` from default)      | **Yes**          |
//! | Add a non-`Option` field (always serializes)                                                     | **Yes**          |
//! | Remove a field                                                                                  | **Yes**          |
//! | Rename a field (or change `#[serde(rename = "...")]`)                                            | **Yes**          |
//! | Change a field's type (`Vec → HashSet`, `u32 → f64`, etc.)                                       | **Yes**          |
//! | Change `From<UninitializedConfig> for StoredConfig` so a different value reaches the wire        | **Yes**          |
//! | Reorder JSON object keys                                                                         | **No** — we sort |
//! | Format a float as `0.7` vs `0.6999…` after a `serde_json` / `toml` crate bump                    | **No** — IEEE bits |
//! | Change `serde_json` / `toml` crate versions without changing the typed shape                     | **No**           |
//!
//! ## What "drift" actually means in production
//!
//! When the `StoredConfig` shape changes between deploys, the same
//! logical config produces a different canonical hash before and after
//! the deploy. Concretely:
//!
//! 1. **Old `config_snapshots` rows already in the DB are unaffected.**
//!    Their persisted `canonical_hash` column was written by the *old*
//!    serializer and is preserved verbatim. The read path
//!    (`ConfigSnapshot::from_stored_with_hash`) pairs the row's stored
//!    hash with its stored `config_jsonb`; it does **not** recompute.
//!    Old `inferences.snapshot_hash` references continue to resolve.
//!
//! 2. **Old rows look up by their old hash, not by the new
//!    re-canonicalization.** If you take an old config snapshot, drop
//!    it through the new `StoredConfig` shape, and call
//!    `canonical_hash()`, you get a *different* value than what's in
//!    the column. That's expected. Always look up old rows by the hash
//!    you wrote — i.e. by the value already in `inferences.snapshot_hash`,
//!    not by recomputing.
//!
//! 3. **New writes get the new hash.** A re-bootstrap of the same
//!    logical config after the schema change produces a new
//!    `config_snapshots` row with a new `canonical_hash`. New
//!    inference rows reference that new hash. Old inferences keep
//!    pointing at the old row; new inferences point at the new row.
//!    Two rows for the "same" config from a human's perspective is
//!    fine — they reflect what the gateway actually saw at the time.
//!
//! 4. **JSONB containment queries (`@>`, `@?`, `@@`) still work**, but
//!    over the *stored* JSON shape. A query written against the new
//!    field layout won't match documents stored under the old layout
//!    unless the change was purely additive at the same path. For
//!    historical search across a schema cutover, query against the
//!    old shape (and let it match nothing in the new partition) or
//!    use containment fragments compatible with both.
//!
//! ## Rules for changing `StoredConfig`
//!
//! - **Always read** [`stored/AGENTS.md`](../AGENTS.md) **before changing
//!   any `Stored*` type.** It documents the historical-snapshot
//!   compatibility contract: stored types do **not** use
//!   `#[serde(deny_unknown_fields)]`, and a deprecated field must keep
//!   parsing for some window even after it's removed from the
//!   `Uninitialized*` mirror.
//! - **Prefer purely-additive changes.** New `Option<T>` fields with
//!   `skip_serializing_if = "Option::is_none"` and a `None` default
//!   keep every existing canonical hash valid.
//! - **Run the fixture suite.** `crates/tensorzero-core/src/config/snapshot/fixtures_tests.rs`
//!   locks the canonical hash bytes against committed TOML fixtures
//!   (kitchen sink, multi-variant, etc.). A schema change that drifts
//!   the bytes will fail those tests; review the diff, decide if the
//!   drift is intentional, and update the expected hexes deliberately.
//!   Don't update them blindly — every drifted hex is a real
//!   incompatibility window.
//! - **Document the change in the PR description** with the categories
//!   above so reviewers can confirm the drift is expected.
//!
//! There is no automatic migration of historical hashes. The dual-
//! dispatch on `SnapshotHashScheme` (`LegacyToml` vs `Canonical`) is
//! how we stay correct across the *initial* TOML→JSON cutover. Across
//! future structural changes inside `Canonical`, we rely on "the
//! stored hash is the authoritative hash" via `from_stored_with_hash`.
//! That model survives any schema evolution as long as old rows are
//! looked up by their stored hash.

use blake3::Hasher;
use num_bigint::BigUint;
use serde_json::Value;
use tensorzero_types::SnapshotHash;

use crate::config::snapshot::StoredConfig;
use crate::error::{Error, ErrorDetails};

// Type tags — each `Value` variant is uniquely prefixed before its body so
// a string `"42"` and a number `42` cannot collide, and so the encoding is
// self-describing for future debugging.
const TAG_NULL: u8 = 0x00;
const TAG_BOOL: u8 = 0x01;
const TAG_NUMBER: u8 = 0x02;
const TAG_STRING: u8 = 0x03;
const TAG_ARRAY: u8 = 0x04;
const TAG_OBJECT: u8 = 0x05;

// Sub-tags inside `TAG_NUMBER`. JSON's `Number` is "i64-or-u64-or-f64-or-
// arbitrary-precision-string" depending on how it was constructed; we encode
// each kind separately so that values that would silently collapse under
// `as_f64()` (any `u64` above 2^53) still hash distinctly. Two numbers with
// the same logical value but different sub-tags hash differently — that's
// fine: a `u64` field and an `f64` field are *different types* in
// `StoredConfig`, so distinguishing them is a feature, not a bug.
const NUMBER_KIND_INT: u8 = 0x00;
const NUMBER_KIND_UINT: u8 = 0x01;
const NUMBER_KIND_FLOAT: u8 = 0x02;
const NUMBER_KIND_STRING: u8 = 0x03;

impl StoredConfig {
    /// Compute a content-stable hash of this config.
    ///
    /// Stable across:
    /// - `StoredConfig → serde_json::to_value → from_value → StoredConfig`
    /// - `StoredConfig → toml::to_string → from_str → StoredConfig` (because
    ///   both formats land in the same typed Rust representation, which
    ///   re-serializes to the same `serde_json::Value`)
    /// - Process restarts, machine architectures, and `serde_json` /
    ///   `toml` crate version bumps that don't change the typed shape.
    ///
    /// **Not** stable across:
    /// - Type changes (renaming a field, changing a Vec to a HashSet, etc.)
    /// - Default-skipping behavior changes (a field that newly serializes
    ///   when it didn't before, or vice versa). This is the only source of
    ///   "logical config didn't change but hash drifted" — and it's
    ///   deliberate: a serialization-shape change *is* a real change.
    pub fn canonical_hash(&self) -> Result<SnapshotHash, Error> {
        let value = serde_json::to_value(self).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("StoredConfig should always serialize to JSON: {e}"),
            })
        })?;
        Ok(canonical_hash_value(&value))
    }
}

/// Hash a `serde_json::Value` using the canonical encoding above. Public
/// within the crate so callers that already have a `Value` (e.g. raw row
/// reads from JSONB) can hash directly without going through `StoredConfig`.
pub(crate) fn canonical_hash_value(value: &Value) -> SnapshotHash {
    let mut hasher = Hasher::new();
    hash_value_into(&mut hasher, value);
    let hash = hasher.finalize();
    // Tag the resulting hash as `Canonical` so callers (URL routing,
    // dispatch on scheme) can tell it apart from a `LegacyToml` hash.
    // Using `from_biguint` here would default the scheme to LegacyToml
    // and lose the round-trip identity.
    SnapshotHash::from_biguint_canonical(BigUint::from_bytes_be(hash.as_bytes()))
}

/// Cast a `usize` length into the `u64` length-prefix slot used by the
/// canonical encoding. The canonical hash bytes are the persisted
/// identity of every config snapshot, so a silently-truncating `as u64`
/// would be unrecoverable on a future >64-bit platform. We make the
/// assumption explicit by panicking instead — the gateway is built
/// only for 64-bit and 32-bit targets, where `usize ≤ u64` holds and
/// this branch is statically unreachable.
#[expect(
    clippy::expect_used,
    reason = "usize > u64 is impossible on every platform we ship; making the assumption explicit beats a silent truncating cast"
)]
fn len_u64(n: usize) -> u64 {
    u64::try_from(n).expect("usize fits in u64 on supported platforms")
}

fn hash_value_into(hasher: &mut Hasher, value: &Value) {
    match value {
        Value::Null => {
            hasher.update(&[TAG_NULL]);
        }
        Value::Bool(b) => {
            hasher.update(&[TAG_BOOL]);
            hasher.update(if *b { &[1u8] } else { &[0u8] });
        }
        Value::Number(n) => {
            hasher.update(&[TAG_NUMBER]);
            // Encode integers exactly. `Number::as_f64()` would lose precision
            // for any `u64` above 2^53 — two distinct `u64` values like
            // `9_007_199_254_740_993` and `9_007_199_254_740_994` collapse to
            // the same `f64`, which would make their canonical hashes
            // collide. `StoredConfig` already has `u64` fields (timeouts,
            // capacities), so this isn't theoretical. Try `as_i64` first,
            // fall back to `as_u64` for the upper half of the unsigned
            // range, only then to `as_f64` for fractional numbers.
            //
            // Each integer kind carries its own sub-tag so an `i64`-shaped
            // value and a `u64`-shaped value with bit-equal magnitudes
            // can't collide either. The `as_string` fallback exists for
            // serde_json's `arbitrary_precision` feature; we don't enable
            // it but the encoding stays well-defined if someone does.
            if let Some(i) = n.as_i64() {
                hasher.update(&[NUMBER_KIND_INT]);
                hasher.update(&i.to_be_bytes());
            } else if let Some(u) = n.as_u64() {
                hasher.update(&[NUMBER_KIND_UINT]);
                hasher.update(&u.to_be_bytes());
            } else if let Some(f) = n.as_f64() {
                hasher.update(&[NUMBER_KIND_FLOAT]);
                hasher.update(&f.to_be_bytes());
            } else {
                let s = n.to_string();
                hasher.update(&[NUMBER_KIND_STRING]);
                hasher.update(&len_u64(s.len()).to_be_bytes());
                hasher.update(s.as_bytes());
            }
        }
        Value::String(s) => {
            hasher.update(&[TAG_STRING]);
            hasher.update(&len_u64(s.len()).to_be_bytes());
            hasher.update(s.as_bytes());
        }
        Value::Array(arr) => {
            hasher.update(&[TAG_ARRAY]);
            hasher.update(&len_u64(arr.len()).to_be_bytes());
            for item in arr {
                hash_value_into(hasher, item);
            }
        }
        Value::Object(map) => {
            hasher.update(&[TAG_OBJECT]);
            hasher.update(&len_u64(map.len()).to_be_bytes());
            // Sort entries by key for deterministic ordering — JSON
            // objects are unordered, so the canonical encoding must be too.
            // `sort_unstable_by_key` is fine: `serde_json::Map` keys are
            // unique, so stability doesn't change the result.
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            entries.sort_unstable_by_key(|(k, _)| *k);
            for (k, v) in entries {
                hasher.update(&len_u64(k.len()).to_be_bytes());
                hasher.update(k.as_bytes());
                hash_value_into(hasher, v);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::UninitializedConfig;
    use std::collections::HashMap;

    fn parse_stored_config(toml_str: &str) -> StoredConfig {
        let table: toml::Table = toml_str
            .parse()
            .expect("fixture should parse as toml::Table");
        let uninit: UninitializedConfig = table
            .try_into()
            .expect("table should produce UninitializedConfig");
        uninit.into()
    }

    fn fixture_with_floats() -> &'static str {
        // Variant `weight` is f64 in StoredConfig; `temperature` is f32.
        // Mixing both exercises the f64-canonicalization path under the
        // f32 widening (serde_json widens f32 → f64 before encoding).
        r#"
[models.dummy]
routing = ["dummy"]

[models.dummy.providers.dummy]
type = "dummy"
model_name = "test"

[functions.fn1]
type = "chat"

[functions.fn1.variants.v]
type = "chat_completion"
model = "dummy"
weight = 0.6
temperature = 0.7
"#
    }

    #[test]
    fn canonical_hash_is_deterministic() {
        let stored = parse_stored_config(fixture_with_floats());
        let h1 = stored.canonical_hash().expect("hash 1");
        let h2 = stored.canonical_hash().expect("hash 2");
        assert_eq!(h1.as_bytes(), h2.as_bytes());
    }

    #[test]
    fn canonical_hash_preserved_through_json_roundtrip() {
        let stored = parse_stored_config(fixture_with_floats());
        let h_before = stored.canonical_hash().expect("hash before");

        let json = serde_json::to_value(&stored).expect("serialize");
        let stored_again: StoredConfig = serde_json::from_value(json).expect("deserialize");
        let h_after = stored_again.canonical_hash().expect("hash after");

        assert_eq!(
            h_before.as_bytes(),
            h_after.as_bytes(),
            "JSON serialize/deserialize round-trip must preserve canonical_hash",
        );
    }

    #[test]
    fn canonical_hash_preserved_through_toml_roundtrip() {
        let stored = parse_stored_config(fixture_with_floats());
        let h_before = stored.canonical_hash().expect("hash before");

        // Note we deserialize via `StoredConfig`, not `UninitializedConfig`,
        // because the snapshot read path uses StoredConfig (which is
        // tolerant of historical fields).
        let toml_str = toml::to_string(&stored).expect("to TOML");
        let stored_again: StoredConfig = toml::from_str(&toml_str).expect("from TOML");
        let h_after = stored_again.canonical_hash().expect("hash after");

        assert_eq!(
            h_before.as_bytes(),
            h_after.as_bytes(),
            "TOML serialize/deserialize round-trip must preserve canonical_hash",
        );
    }

    #[test]
    fn canonical_hash_preserved_through_toml_then_json_roundtrip() {
        let stored = parse_stored_config(fixture_with_floats());
        let h_before = stored.canonical_hash().expect("hash before");

        // TOML → StoredConfig → JSON → StoredConfig (the path the snapshot
        // pipeline takes for new writes plus the read-after-backfill path).
        let toml_str = toml::to_string(&stored).expect("to TOML");
        let from_toml: StoredConfig = toml::from_str(&toml_str).expect("from TOML");
        let json = serde_json::to_value(&from_toml).expect("to JSON");
        let from_json: StoredConfig = serde_json::from_value(json).expect("from JSON");
        let h_after = from_json.canonical_hash().expect("hash after");

        assert_eq!(h_before.as_bytes(), h_after.as_bytes());
    }

    #[test]
    fn canonical_hash_changes_with_content() {
        let a = parse_stored_config(fixture_with_floats());
        let b_toml = fixture_with_floats().replace("temperature = 0.7", "temperature = 0.8");
        let b = parse_stored_config(&b_toml);
        assert_ne!(
            a.canonical_hash().unwrap().as_bytes(),
            b.canonical_hash().unwrap().as_bytes(),
            "changing a typed primitive value must change the hash",
        );
    }

    #[test]
    fn canonical_hash_unaffected_by_object_key_order() {
        // Construct two semantically-identical Value trees with different
        // insertion orders. The canonical encoding sorts keys, so they must
        // hash identically.
        let v1 = serde_json::json!({"a": 1, "b": 2, "c": 3});
        let v2 = serde_json::json!({"c": 3, "a": 1, "b": 2});
        assert_eq!(
            canonical_hash_value(&v1).as_bytes(),
            canonical_hash_value(&v2).as_bytes(),
        );
    }

    #[test]
    fn canonical_hash_distinguishes_string_from_number() {
        // The type tag prefix must prevent a string `"42"` from colliding
        // with a number 42. This is the most basic encoding property — if
        // it ever fails, the type-tag byte is being skipped.
        let v_string = Value::String("42".to_string());
        let v_number = serde_json::json!(42);
        assert_ne!(
            canonical_hash_value(&v_string).as_bytes(),
            canonical_hash_value(&v_number).as_bytes(),
        );
    }

    #[test]
    fn canonical_hash_distinguishes_array_concat_from_string_concat() {
        // Length-prefix sanity: `["ab"]` ↔ `["a", "b"]` would collide if
        // we just hashed concatenated bytes without the length prefix.
        let one = serde_json::json!(["ab"]);
        let two = serde_json::json!(["a", "b"]);
        assert_ne!(
            canonical_hash_value(&one).as_bytes(),
            canonical_hash_value(&two).as_bytes(),
        );
    }

    #[test]
    fn large_u64_values_do_not_collide_via_f64_truncation() {
        // Regression for the as_f64()-only encoding: every u64 above 2^53
        // shares its f64 representation with at least one neighbor, so
        // hashing through `as_f64()` made distinct logical configs alias.
        // The integer-aware encoding distinguishes them.
        use serde_json::{Number, Value};

        // f64 mantissa is 52 bits + implicit 1 ⇒ 53 bits of precision, so
        // integers up to 2^53 are exact and `2^53 + 1` rounds (banker's) to
        // `2^53`. The two distinct `u64` values share an `f64`.
        let a = 9_007_199_254_740_992u64; // 2^53 (exact)
        let b = 9_007_199_254_740_993u64; // 2^53 + 1 (rounds to 2^53)
        assert_eq!(a as f64, b as f64, "preconditions: f64 collapses these");

        let h_a = canonical_hash_value(&Value::Number(Number::from(a)));
        let h_b = canonical_hash_value(&Value::Number(Number::from(b)));
        assert_ne!(
            h_a.as_bytes(),
            h_b.as_bytes(),
            "canonical hash must distinguish u64 values that share an f64",
        );
    }

    #[test]
    fn integer_and_float_with_same_magnitude_hash_differently() {
        // A `u32 = 1` field and an `f64 = 1.0` field are different types
        // in `StoredConfig`; `serde_json::Value` carries the distinction
        // via `Number::as_i64` vs `as_f64`. The canonical encoding uses
        // separate sub-tags so they don't collide.
        use serde_json::{Number, Value};

        let int_one = canonical_hash_value(&Value::Number(Number::from(1u64)));
        let float_one = canonical_hash_value(&Value::Number(
            Number::from_f64(1.0).expect("1.0 is finite"),
        ));
        assert_ne!(
            int_one.as_bytes(),
            float_one.as_bytes(),
            "integer 1 and float 1.0 must not share a canonical hash",
        );
    }

    #[test]
    fn canonical_hash_diverges_from_legacy_toml_hash_advertised_as_v2() {
        // Sanity: `canonical_hash` is NOT the same value as the existing
        // TOML-bytes hash on `ConfigSnapshot`. They're different encodings;
        // anyone expecting them to match should be redirected here.
        let stored = parse_stored_config(fixture_with_floats());
        let canon = stored.canonical_hash().expect("canon");
        let snapshot = crate::config::snapshot::ConfigSnapshot::from_stored_config(
            stored,
            HashMap::new(),
            HashMap::new(),
        )
        .expect("snapshot");
        // Both are valid hashes of the same logical config, but their
        // canonical encodings differ.
        assert_ne!(canon.as_bytes(), snapshot.hash.as_bytes());
    }
}
