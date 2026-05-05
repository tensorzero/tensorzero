use num_bigint::BigUint;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Which hash function produced this `SnapshotHash`.
///
/// The legacy scheme on `main` hashes canonical-TOML bytes; the canonical
/// scheme hashes the structural JSON form via
/// `StoredConfig::canonical_hash`. Both are 256-bit Blake3 outputs but
/// they are NOT interchangeable identifiers — the same logical config
/// produces different bytes under each scheme.
///
/// Snapshots persist both hashes in different columns of
/// `tensorzero.config_snapshots` (`hash` and `canonical_hash`). The
/// scheme tag carried by `SnapshotHash` tells the lookup code which
/// column to query.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SnapshotHashScheme {
    /// V1: blake3 over canonical-TOML bytes. The hash basis on `main` and
    /// what `inferences.snapshot_hash` references for every row written
    /// before the canonical-hash migration. Display: bare decimal, no
    /// scheme prefix — exactly the wire form pre-migration callers
    /// used, so existing tags/rows/URLs parse unchanged.
    LegacyToml,
    /// V2: structural blake3 over the canonical JSON Value form. Stable
    /// across serialization roundtrips (see
    /// `tensorzero_core::config::snapshot::canonical_hash`). Display
    /// prefix: `can:`.
    Canonical,
}

impl SnapshotHashScheme {
    /// Stable string prefix used in the display and serde forms.
    ///
    /// Wire convention:
    /// - `Canonical` carries the `can:` prefix (e.g. `can:14940...`).
    /// - `LegacyToml` is **unprefixed** — same as the form every
    ///   pre-canonical-hash writer used. This means existing
    ///   `inferences.snapshot_hash` rows, autopilot tags, log lines,
    ///   etc. parse unchanged.
    ///
    /// `prefix()` returns `None` for legacy because the legacy form has
    /// no prefix on the wire.
    pub const fn prefix(self) -> Option<&'static str> {
        match self {
            SnapshotHashScheme::LegacyToml => None,
            SnapshotHashScheme::Canonical => Some("can"),
        }
    }
}

/// A snapshot hash that stores both the decimal string representation
/// and the big-endian bytes for efficient storage in different databases.
///
/// As of the canonical-hash migration, every `SnapshotHash` also carries
/// a `SnapshotHashScheme` describing which hash function produced its
/// bytes. The `Display`, `FromStr`, and `Serialize`/`Deserialize` impls
/// use the self-describing wire form (bare decimal for legacy, `can:`
/// for canonical) so callers can route lookups to the right column
/// without out-of-band scheme information.
///
/// Backwards compatibility: `FromStr` (and therefore `Deserialize`)
/// accepts the legacy unprefixed decimal form and defaults it to
/// `LegacyToml`. This keeps every pre-migration `inferences.snapshot_hash`
/// value parseable without a backfill.
///
/// Identity is `(scheme, bytes)` only — `decimal_str` is a derived
/// cache of `bytes` and `PartialEq` / `Hash` skip it. Two `SnapshotHash`
/// values with the same scheme and bytes always compare equal even if
/// the cached decimal string was constructed via different paths.
#[derive(Clone, Debug)]
pub struct SnapshotHash {
    scheme: SnapshotHashScheme,
    /// The decimal string representation of the hash (used for ClickHouse).
    /// Derived from `bytes`; not part of the value's identity.
    decimal_str: Arc<str>,
    /// The big-endian bytes representation of the hash (used for Postgres BYTEA).
    /// This is 256 bits (32 bytes).
    bytes: Arc<[u8]>,
}

impl PartialEq for SnapshotHash {
    fn eq(&self, other: &Self) -> bool {
        // `decimal_str` is derived from `bytes`, so comparing it would be
        // redundant and would couple equality to a happens-to-be-cached
        // representation rather than the underlying value.
        self.scheme == other.scheme && self.bytes == other.bytes
    }
}

impl Eq for SnapshotHash {}

impl std::hash::Hash for SnapshotHash {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Mirror `PartialEq`: hash only the identity-bearing fields.
        // Skipping `decimal_str` avoids ~78 bytes of redundant work and
        // upholds the `k1 == k2 ⇒ hash(k1) == hash(k2)` invariant under
        // the trimmed `eq`.
        self.scheme.hash(state);
        self.bytes.hash(state);
    }
}

impl SnapshotHash {
    /// Creates a new `SnapshotHash` from a `BigUint`, defaulting to the
    /// legacy TOML-bytes scheme. New call-sites that hash via
    /// `StoredConfig::canonical_hash` should use
    /// `from_biguint_canonical` instead.
    pub fn from_biguint(big_int: BigUint) -> Self {
        Self::from_biguint_with_scheme(big_int, SnapshotHashScheme::LegacyToml)
    }

    /// Creates a `SnapshotHash` carrying the canonical scheme tag.
    pub fn from_biguint_canonical(big_int: BigUint) -> Self {
        Self::from_biguint_with_scheme(big_int, SnapshotHashScheme::Canonical)
    }

    pub(crate) fn from_biguint_with_scheme(big_int: BigUint, scheme: SnapshotHashScheme) -> Self {
        let decimal_str = Arc::from(big_int.to_string());
        let bytes = Arc::from(big_int.to_bytes_be());
        Self {
            scheme,
            decimal_str,
            bytes,
        }
    }

    /// Creates a SnapshotHash from big-endian bytes, tagged as legacy.
    ///
    /// This is what `sqlx::Decode` uses for the `hash BYTEA` column. The
    /// `canonical_hash BYTEA` column read path constructs via
    /// `from_canonical_bytes` instead so the resulting `SnapshotHash`
    /// carries the right scheme tag.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        Self::from_bytes_with_scheme(bytes, SnapshotHashScheme::LegacyToml)
    }

    /// Creates a `SnapshotHash` from big-endian bytes, tagged as canonical.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Self {
        Self::from_bytes_with_scheme(bytes, SnapshotHashScheme::Canonical)
    }

    pub(crate) fn from_bytes_with_scheme(bytes: &[u8], scheme: SnapshotHashScheme) -> Self {
        let big_int = BigUint::from_bytes_be(bytes);
        Self::from_biguint_with_scheme(big_int, scheme)
    }

    /// Returns this hash's scheme tag.
    pub fn scheme(&self) -> SnapshotHashScheme {
        self.scheme
    }

    /// Returns the big-endian bytes representation.
    /// This is used for storing in Postgres as BYTEA.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Returns the lowercase hex representation of the hash.
    /// This matches the format used by ClickHouse `lower(hex(...))` and Postgres `encode(..., 'hex')`.
    ///
    /// The hex representation does NOT include the scheme prefix — it's
    /// the raw byte form intended for DB hex encodings, not for
    /// self-describing identifiers in transport.
    pub fn to_hex_string(&self) -> String {
        hex::encode(self.as_bytes())
    }

    /// Returns the decimal string form WITHOUT the scheme prefix.
    /// Intended for ClickHouse `toUInt256(...)` literals where the column
    /// type itself constrains the scheme.
    pub fn to_decimal_string(&self) -> &str {
        &self.decimal_str
    }
}

impl std::fmt::Display for SnapshotHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Self-describing transport form. Canonical hashes carry a
        // `can:` prefix; legacy hashes are bare decimal (matching the
        // pre-canonical-hash wire format so existing systems don't
        // need to change).
        match self.scheme.prefix() {
            Some(prefix) => write!(f, "{prefix}:{}", self.decimal_str),
            None => write!(f, "{}", self.decimal_str),
        }
    }
}

impl Serialize for SnapshotHash {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Mirrors `Display`: the wire form is the self-describing
        // identifier (`can:DECIMAL` for canonical, bare decimal for
        // legacy). DB-row structs that write to numeric columns
        // (`UInt256` / `NUMERIC(78,0)`) cannot accept the prefix and
        // must opt out via `#[serde(serialize_with = "serialize_hash_bare_decimal")]`
        // — the helper lives in this module.
        serializer.collect_str(self)
    }
}

/// `#[serde(serialize_with = ...)]` opt-outs for `SnapshotHash` fields
/// whose backing storage cannot accept the scheme prefix (ClickHouse
/// `UInt256`, Postgres `NUMERIC(78,0)`). Both helpers emit the bare
/// decimal form regardless of scheme.
pub mod serializers {
    use super::SnapshotHash;
    use serde::Serializer;

    /// For `Option<SnapshotHash>` fields. Emits `null` for `None`, bare
    /// decimal for `Some`. Use as
    /// `#[serde(serialize_with = "tensorzero_types::snapshot::serializers::optional_bare_decimal")]`.
    pub fn optional_bare_decimal<S>(
        hash: &Option<SnapshotHash>,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match hash {
            Some(h) => serializer.serialize_str(h.to_decimal_string()),
            None => serializer.serialize_none(),
        }
    }

    /// For non-`Option` `SnapshotHash` fields. Emits the bare decimal.
    /// Use as
    /// `#[serde(serialize_with = "tensorzero_types::snapshot::serializers::bare_decimal")]`.
    pub fn bare_decimal<S>(hash: &SnapshotHash, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(hash.to_decimal_string())
    }
}

impl std::str::FromStr for SnapshotHash {
    type Err = num_bigint::ParseBigIntError;

    /// Accepts:
    /// - `"can:DECIMAL"` → `Canonical`
    /// - `"DECIMAL"`     → `LegacyToml` (the pre-canonical-hash wire form;
    ///   stays unprefixed so existing tags / rows / URLs parse unchanged)
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if let Some(rest) = s.strip_prefix("can:") {
            let big_int = rest.parse::<BigUint>()?;
            Ok(SnapshotHash::from_biguint_with_scheme(
                big_int,
                SnapshotHashScheme::Canonical,
            ))
        } else {
            // Bare decimal: legacy form.
            let big_int = s.parse::<BigUint>()?;
            Ok(SnapshotHash::from_biguint_with_scheme(
                big_int,
                SnapshotHashScheme::LegacyToml,
            ))
        }
    }
}

impl<'de> Deserialize<'de> for SnapshotHash {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // Use a visitor over `&str` so borrowed deserializers (serde_json,
        // sqlx) don't pay for an intermediate `String` allocation per
        // row. `visit_string` is provided as a fallback for owning
        // deserializers that cannot hand out a borrow.
        struct V;
        impl serde::de::Visitor<'_> for V {
            type Value = SnapshotHash;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a SnapshotHash decimal string (optionally `can:`-prefixed)")
            }

            fn visit_str<E: serde::de::Error>(self, s: &str) -> Result<SnapshotHash, E> {
                s.parse::<SnapshotHash>().map_err(E::custom)
            }

            fn visit_string<E: serde::de::Error>(self, s: String) -> Result<SnapshotHash, E> {
                self.visit_str(&s)
            }
        }

        deserializer.deserialize_str(V)
    }
}

/// Maps `SnapshotHash` to Postgres BYTEA so it can be used directly in
/// `push_bind` and `FromRow` without manual `as_bytes()`/`from_bytes()` conversion.
impl sqlx::Type<sqlx::Postgres> for SnapshotHash {
    fn type_info() -> sqlx::postgres::PgTypeInfo {
        <Vec<u8> as sqlx::Type<sqlx::Postgres>>::type_info()
    }

    fn compatible(ty: &sqlx::postgres::PgTypeInfo) -> bool {
        <Vec<u8> as sqlx::Type<sqlx::Postgres>>::compatible(ty)
    }
}

impl sqlx::Encode<'_, sqlx::Postgres> for SnapshotHash {
    fn encode_by_ref(
        &self,
        buf: &mut sqlx::postgres::PgArgumentBuffer,
    ) -> Result<sqlx::encode::IsNull, sqlx::error::BoxDynError> {
        <&[u8] as sqlx::Encode<'_, sqlx::Postgres>>::encode_by_ref(&self.as_bytes(), buf)
    }
}

impl<'r> sqlx::Decode<'r, sqlx::Postgres> for SnapshotHash {
    /// Defaults to `LegacyToml` because the bytes alone cannot tell us
    /// which scheme produced them. Code paths reading the
    /// `canonical_hash` column should call `from_canonical_bytes`
    /// explicitly on the row's bytes.
    fn decode(value: sqlx::postgres::PgValueRef<'r>) -> Result<Self, sqlx::error::BoxDynError> {
        let bytes = <Vec<u8> as sqlx::Decode<'r, sqlx::Postgres>>::decode(value)?;
        Ok(SnapshotHash::from_bytes(&bytes))
    }
}

#[cfg(any(test, feature = "e2e_tests"))]
impl SnapshotHash {
    /// Creates a test SnapshotHash by hashing an empty input with blake3.
    /// This produces a deterministic hash suitable for testing.
    pub fn new_test() -> SnapshotHash {
        let hash = blake3::hash(&[]);
        let big_int = BigUint::from_bytes_be(hash.as_bytes());
        SnapshotHash::from_biguint(big_int)
    }
}

#[cfg(any(test, feature = "e2e_tests"))]
impl Default for SnapshotHash {
    fn default() -> Self {
        SnapshotHash::new_test()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn display_form_is_self_describing() {
        // Legacy: bare decimal (matches every pre-canonical-hash writer
        // — `inferences.snapshot_hash` rows, autopilot tags, log lines).
        let legacy = SnapshotHash::from_bytes(&[0xAB; 32]);
        let legacy_str = legacy.to_string();
        assert!(
            !legacy_str.starts_with("can:"),
            "legacy display must be bare decimal; got {legacy_str}",
        );
        assert!(legacy_str.parse::<BigUint>().is_ok());

        // Canonical: `can:DECIMAL`.
        let canonical = SnapshotHash::from_canonical_bytes(&[0xAB; 32]);
        let canonical_str = canonical.to_string();
        assert!(
            canonical_str.starts_with("can:"),
            "canonical display must start with `can:`; got {canonical_str}",
        );
    }

    #[test]
    fn legacy_and_canonical_with_same_bytes_are_distinguishable_by_scheme() {
        let bytes = [0x42u8; 32];
        let legacy = SnapshotHash::from_bytes(&bytes);
        let canonical = SnapshotHash::from_canonical_bytes(&bytes);

        // Same bytes...
        assert_eq!(legacy.as_bytes(), canonical.as_bytes());
        // ...but different scheme tags...
        assert_eq!(legacy.scheme(), SnapshotHashScheme::LegacyToml);
        assert_eq!(canonical.scheme(), SnapshotHashScheme::Canonical);
        // ...so they don't compare equal (scheme is part of the identity).
        assert_ne!(legacy, canonical);
        // ...and they print differently.
        assert_ne!(legacy.to_string(), canonical.to_string());
    }

    #[test]
    fn from_str_round_trip_for_both_schemes() {
        for scheme in [
            SnapshotHashScheme::LegacyToml,
            SnapshotHashScheme::Canonical,
        ] {
            let original = SnapshotHash::from_bytes_with_scheme(&[0x12; 32], scheme);
            let s = original.to_string();
            let parsed = SnapshotHash::from_str(&s).expect("parse back");
            assert_eq!(parsed, original, "round-trip for {scheme:?}");
            assert_eq!(parsed.scheme(), scheme);
        }
    }

    #[test]
    fn bare_decimal_parses_as_legacy() {
        // Backwards-compat path: an `inferences.snapshot_hash` column on a
        // pre-migration row stores the decimal form WITHOUT a prefix.
        // Parsing it must yield a legacy-scheme hash. This is the
        // *normal* legacy wire form, not a special exception.
        let bytes = [0xCD; 32];
        let big_int = BigUint::from_bytes_be(&bytes);
        let raw_decimal = big_int.to_string();

        let parsed = SnapshotHash::from_str(&raw_decimal).expect("legacy decimal parses");
        assert_eq!(parsed.scheme(), SnapshotHashScheme::LegacyToml);
        assert_eq!(parsed.as_bytes(), &bytes);
    }

    #[test]
    fn serde_round_trip_preserves_scheme() {
        // Wire form mirrors `Display`: legacy is bare decimal, canonical
        // is `can:DECIMAL`. Both forms round-trip through serde and
        // recover the original scheme tag — that's the contract callers
        // depend on for routing lookups to the right column.
        let canonical = SnapshotHash::from_canonical_bytes(&[0x88; 32]);
        let json = serde_json::to_string(&canonical).expect("serialize");
        assert!(
            json.contains("can:"),
            "canonical wire form must carry the `can:` prefix; got {json}",
        );
        let back: SnapshotHash = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, canonical);
        assert_eq!(back.scheme(), SnapshotHashScheme::Canonical);

        let legacy = SnapshotHash::from_bytes(&[0x77; 32]);
        let json_legacy = serde_json::to_string(&legacy).expect("serialize legacy");
        assert!(
            !json_legacy.contains("can:"),
            "legacy wire form must be bare decimal; got {json_legacy}",
        );
        let back_legacy: SnapshotHash = serde_json::from_str(&json_legacy).expect("deserialize");
        assert_eq!(back_legacy, legacy);
    }

    #[test]
    fn bare_decimal_helper_strips_prefix() {
        // The DB-row helper opt-out: numeric columns can't accept the
        // `can:` prefix, so DB-row structs annotated with
        // `serialize_with = "serializers::optional_bare_decimal"` emit
        // just the digits.
        #[derive(Serialize)]
        struct Row {
            #[serde(serialize_with = "serializers::optional_bare_decimal")]
            snapshot_hash: Option<SnapshotHash>,
        }
        let canonical = SnapshotHash::from_canonical_bytes(&[0x99; 32]);
        let row = Row {
            snapshot_hash: Some(canonical.clone()),
        };
        let json = serde_json::to_string(&row).unwrap();
        assert!(
            !json.contains("can:"),
            "DB-row helper must strip the prefix; got {json}",
        );
        // The decimal *digits* are present.
        assert!(json.contains(canonical.to_decimal_string()));

        let none_row = Row {
            snapshot_hash: None,
        };
        assert_eq!(
            serde_json::to_string(&none_row).unwrap(),
            r#"{"snapshot_hash":null}"#
        );
    }

    #[test]
    fn eq_and_hash_skip_cached_decimal_string() {
        // Locks in the documented identity contract: `decimal_str` is a
        // derived cache of `bytes`, so two `SnapshotHash` values with
        // the same `(scheme, bytes)` must compare equal and hash equal,
        // independent of how their decimal cache was constructed.
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let bytes = [0x33; 32];
        let from_bytes = SnapshotHash::from_canonical_bytes(&bytes);
        let from_biguint = SnapshotHash::from_biguint_canonical(BigUint::from_bytes_be(&bytes));

        // Sanity: same value, two construction paths.
        assert_eq!(from_bytes, from_biguint);

        let mut h1 = DefaultHasher::new();
        from_bytes.hash(&mut h1);
        let mut h2 = DefaultHasher::new();
        from_biguint.hash(&mut h2);
        assert_eq!(
            h1.finish(),
            h2.finish(),
            "Hash must depend only on (scheme, bytes), matching PartialEq",
        );
    }

    #[test]
    fn display_keeps_scheme_prefix_for_url_round_trips() {
        // Display is the transport identifier callers paste into URLs
        // (`/internal/config/{hash}`) and log lines. Canonical hashes
        // carry the `can:` prefix so the receiving side knows which
        // scheme produced the hash.
        let canonical = SnapshotHash::from_canonical_bytes(&[0x77; 32]);
        let displayed = canonical.to_string();
        assert!(displayed.starts_with("can:"), "got {displayed}");

        let parsed = SnapshotHash::from_str(&displayed).expect("FromStr");
        assert_eq!(parsed.scheme(), SnapshotHashScheme::Canonical);
        assert_eq!(parsed.as_bytes(), canonical.as_bytes());
    }
}
