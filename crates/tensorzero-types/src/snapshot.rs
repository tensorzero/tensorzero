use num_bigint::BigUint;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// A snapshot hash that stores both the decimal string representation
/// and the big-endian bytes for efficient storage in different databases.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SnapshotHash {
    /// The decimal string representation of the hash (used for ClickHouse)
    decimal_str: Arc<str>,
    /// The big-endian bytes representation of the hash (used for Postgres BYTEA)
    /// This is 256 bits (32 bytes).
    bytes: Arc<[u8]>,
}

impl SnapshotHash {
    /// Creates a new SnapshotHash from a BigUint.
    pub fn from_biguint(big_int: BigUint) -> Self {
        let decimal_str = Arc::from(big_int.to_string());
        let bytes = Arc::from(big_int.to_bytes_be());
        Self { decimal_str, bytes }
    }

    /// Creates a SnapshotHash from big-endian bytes.
    /// This is used when reading from Postgres BYTEA.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let big_int = BigUint::from_bytes_be(bytes);
        Self::from_biguint(big_int)
    }

    /// Returns the big-endian bytes representation.
    /// This is used for storing in Postgres as BYTEA.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Returns the lowercase hex representation of the hash.
    /// This matches the format used by ClickHouse `lower(hex(...))` and Postgres `encode(..., 'hex')`.
    pub fn to_hex_string(&self) -> String {
        hex::encode(&*self.bytes)
    }
}

impl std::fmt::Display for SnapshotHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.decimal_str)
    }
}

impl std::ops::Deref for SnapshotHash {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        &self.decimal_str
    }
}

impl Serialize for SnapshotHash {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.decimal_str)
    }
}

impl std::str::FromStr for SnapshotHash {
    type Err = num_bigint::ParseBigIntError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let big_int = s.parse::<BigUint>()?;
        Ok(SnapshotHash::from_biguint(big_int))
    }
}

impl<'de> Deserialize<'de> for SnapshotHash {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        s.parse::<SnapshotHash>().map_err(serde::de::Error::custom)
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
