/// Custom serde module for `Option<Decimal>` as float.
///
/// Serializes identically to `rust_decimal::serde::float_option`.
/// Deserializes via `Option<f64>` instead of `deserialize_option` so that
/// serde's untagged-enum `ContentDeserializer` (which maps JSON `null` to
/// `Content::Unit` → `visit_unit`) is handled correctly.  The upstream
/// `OptionDecimalVisitor` only implements `visit_none`, not `visit_unit`,
/// which causes failures inside `#[serde(flatten)]` and `#[serde(untagged)]`.
pub mod decimal_float_option {
    use rust_decimal::Decimal;
    use serde::{Deserialize, Deserializer, Serializer};

    // Signature is required by serde's `with` attribute.
    pub fn serialize<S>(value: &Option<Decimal>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        rust_decimal::serde::float_option::serialize(value, serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Decimal>, D::Error>
    where
        D: Deserializer<'de>,
    {
        Option::<f64>::deserialize(deserializer)?
            .map(|f| Decimal::try_from(f).map_err(serde::de::Error::custom))
            .transpose()
    }
}
