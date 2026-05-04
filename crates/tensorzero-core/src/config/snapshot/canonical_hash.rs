//! Structural hashing for `StoredConfig`.
//!
//! The historical hash on `main` is computed from canonical *TOML bytes*,
//! which is fragile: float reformatting (`0.7` → `0.6999999…`), TOML crate
//! version bumps, and any change to canonicalization rules drift the hash
//! and invalidate every `inferences.snapshot_hash` reference. The plan
//! (called `hash_v2` in the Config-in-Database roadmap) replaces it with a
//! structural hash that:
//!
//! - Operates on the **logical content** of the config, not on its
//!   serialized text.
//! - Is preserved by every `StoredConfig → JSON → StoredConfig` and
//!   `StoredConfig → TOML → StoredConfig` round-trip.
//! - Does not depend on third-party crate formatting choices.
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
            // Normalize to IEEE 754 f64 bit pattern. JSON's number type is
            // f64-or-arbitrary-precision; serde_json's `Number::as_f64`
            // returns `None` only for arbitrary-precision numbers (the
            // crate's `arbitrary_precision` feature). For our snapshot
            // shapes — all numeric fields are `i32`/`u32`/`u64`/`f32`/`f64`
            // typed Rust primitives — `as_f64` is always `Some`. The string
            // fallback exists for forward-compatibility if we ever enable
            // arbitrary precision.
            if let Some(f) = n.as_f64() {
                hasher.update(&f.to_be_bytes());
            } else {
                let s = n.to_string();
                hasher.update(&(s.len() as u64).to_be_bytes());
                hasher.update(s.as_bytes());
            }
        }
        Value::String(s) => {
            hasher.update(&[TAG_STRING]);
            hasher.update(&(s.len() as u64).to_be_bytes());
            hasher.update(s.as_bytes());
        }
        Value::Array(arr) => {
            hasher.update(&[TAG_ARRAY]);
            hasher.update(&(arr.len() as u64).to_be_bytes());
            for item in arr {
                hash_value_into(hasher, item);
            }
        }
        Value::Object(map) => {
            hasher.update(&[TAG_OBJECT]);
            hasher.update(&(map.len() as u64).to_be_bytes());
            // Sort entries by key for deterministic ordering — JSON
            // objects are unordered, so the canonical encoding must be too.
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            entries.sort_by(|(a, _), (b, _)| a.cmp(b));
            for (k, v) in entries {
                hasher.update(&(k.len() as u64).to_be_bytes());
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
version = 4

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
    fn canonical_hash_zero_version_omitted_equals_explicit_zero() {
        // Functional consistency with the snapshot hash: a `version = 0`
        // explicit and an absent `version` produce the same StoredConfig
        // (because `skip_serializing_if = "u32_is_zero"` omits zero from
        // the serialized form, so both deserialize to `version: 0`). The
        // canonical hash must agree.
        let absent_toml = r#"
[models.dummy]
routing = ["dummy"]

[models.dummy.providers.dummy]
type = "dummy"
model_name = "test"

[functions.f]
type = "chat"

[functions.f.variants.v]
type = "chat_completion"
model = "dummy"
"#;
        let explicit_toml = r#"
[models.dummy]
routing = ["dummy"]

[models.dummy.providers.dummy]
type = "dummy"
model_name = "test"

[functions.f]
type = "chat"
version = 0

[functions.f.variants.v]
type = "chat_completion"
model = "dummy"
version = 0
"#;
        let absent = parse_stored_config(absent_toml);
        let explicit = parse_stored_config(explicit_toml);
        assert_eq!(
            absent.canonical_hash().unwrap().as_bytes(),
            explicit.canonical_hash().unwrap().as_bytes(),
            "version = 0 and absent version must be hash-equivalent",
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
