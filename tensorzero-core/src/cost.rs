//! Cost tracking configuration and computation.
//!
//! This module provides types and functions for computing inference cost from
//! provider responses using user-configured JSON Pointer mappings.
//!
//! ## Architecture
//!
//! Cost configuration follows the Uninitialized -> Normalized pattern:
//!
//! - **`UninitializedCostConfigEntry`** / **`UninitializedCostRate`**: User-facing types
//!   that support friendly formats like `cost_per_million`. Deserialized from TOML/JSON config.
//! - **`CostConfigEntry`** / **`CostRate`**: Normalized runtime types where all rates are
//!   stored as `cost_per_unit`. The conversion from `per_million` to `per_unit` happens once
//!   at config load time, so runtime cost computation is a simple multiplication.
//!
//! ## Extensibility
//!
//! The design is intentionally structured so that each `CostConfigEntry` independently
//! computes a contribution via `compute_contribution`, and `compute_cost_from_json`
//! simply sums them. This means the inner computation can be extended (e.g., to support
//! multiple named pointers with an expression string like
//! `"input_tokens * 1.5 / 1000000 + output_tokens * 3.0 / 1000000"`)
//! without changing the outer summation loop or the overall flow.
//!
//! A future extension could replace the current `pointer + rate` model with an enum:
//!
//! ```text
//! enum CostComputation {
//!     SimpleRate { pointer, rate },          // current behavior
//!     Expression { pointers, cost_expr },    // multiple named pointers + formula
//! }
//! ```
//!
//! ## Decimal Precision Note
//!
//! Cost rates are deserialized from TOML as `rust_decimal::Decimal` using the
//! crate-wide `serde-float` feature. This means TOML float values pass through
//! `f64` before becoming `Decimal`, which can introduce tiny precision errors
//! for values not exactly representable in binary (e.g., `0.1`, `0.2`).
//! Values like `0.25`, `0.50`, `1.50`, `3.00` are exact.
//!
//! In practice, the error is on the order of 1e-17 per unit — far below any
//! meaningful cost threshold. If exact decimal precision is ever required,
//! consider switching to `rust_decimal`'s `serde-with-arbitrary-precision`
//! feature or accepting string-quoted values in config.

use std::fmt;
use std::str::FromStr;

use rust_decimal::Decimal;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::error::{Error, ErrorDetails};

/// The Rust type used for cost values throughout TensorZero.
///
/// Stored as:
/// - Postgres: `NUMERIC(18, 9)`
/// - ClickHouse: `Decimal(18, 9)`
///
/// 18 total digits with 9 fractional digits gives us up to 999,999,999 dollars
/// with sub-nanocent precision — more than enough for inference cost tracking.
pub type Cost = Decimal;

// ─── Validated JSON Pointer ──────────────────────────────────────────────────

/// A validated JSON Pointer (RFC 6901).
///
/// Must either be empty (`""`) or start with `/`.
/// Validation happens at deserialization time, so invalid pointers are rejected
/// early during config loading.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JsonPointer(String);

impl JsonPointer {
    /// Create a new `JsonPointer`, validating the format.
    pub fn new(s: impl Into<String>) -> Result<Self, String> {
        let s = s.into();
        if !s.is_empty() && !s.starts_with('/') {
            return Err(format!(
                "JSON Pointer must be empty or start with `/` (got `{s}`)"
            ));
        }
        Ok(Self(s))
    }

    /// Return the pointer as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for JsonPointer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl FromStr for JsonPointer {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::new(s)
    }
}

impl Serialize for JsonPointer {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for JsonPointer {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        Self::new(s).map_err(serde::de::Error::custom)
    }
}

#[cfg(feature = "ts-bindings")]
impl ts_rs::TS for JsonPointer {
    type WithoutGenerics = Self;
    type OptionInnerType = Self;

    fn name(_cfg: &ts_rs::Config) -> String {
        "string".to_string()
    }

    fn inline(_cfg: &ts_rs::Config) -> String {
        "string".to_string()
    }

    fn output_path() -> Option<std::path::PathBuf> {
        None
    }
}

// ─── Uninitialized (user-facing) types ───────────────────────────────────────

/// User-facing cost config entry, deserialized from TOML/JSON.
///
/// Supports friendly rate formats like `cost_per_million` and `cost_per_unit`.
/// Converted to [`CostConfigEntry`] at config load time.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
pub struct UninitializedCostConfigEntry {
    /// JSON Pointer configuration for extracting the value from the response.
    #[serde(flatten)]
    pub pointer: CostPointerConfig,

    /// The rate to apply to the extracted value (user-friendly format).
    #[serde(flatten)]
    pub rate: UninitializedCostRate,

    /// If `true`, a missing field will corrupt the total cost to `None` (and log a warning).
    /// If `false` (default), a missing field is treated as zero.
    #[serde(default)]
    pub required: bool,
}

/// User-facing cost rate, supporting `cost_per_million` and `cost_per_unit`.
///
/// Converted to the normalized [`CostRate`] (always per-unit) at config load time.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[serde(untagged)]
pub enum UninitializedCostRate {
    /// Cost per million units (e.g., cost per million tokens).
    PerMillion {
        #[cfg_attr(feature = "ts-bindings", ts(type = "number"))]
        cost_per_million: Decimal,
    },
    /// Cost per single unit (e.g., cost per web search).
    PerUnit {
        #[cfg_attr(feature = "ts-bindings", ts(type = "number"))]
        cost_per_unit: Decimal,
    },
}

/// User-facing cost configuration (a list of uninitialized entries).
pub type UninitializedCostConfig = Vec<UninitializedCostConfigEntry>;

// ─── Normalized (runtime) types ──────────────────────────────────────────────

/// The divisor for `cost_per_million`.
const MILLION: Decimal = Decimal::from_parts(1_000_000, 0, 0, false, 0);

/// A normalized cost rate, always stored as cost per single unit.
///
/// At config load time, `cost_per_million` values are divided by 1,000,000
/// to produce a uniform per-unit rate. This keeps runtime cost computation simple.
#[derive(Clone, Debug, PartialEq)]
pub struct CostRate {
    /// Cost per single unit (e.g., per token, per web search).
    pub cost_per_unit: Decimal,
}

impl From<UninitializedCostRate> for CostRate {
    fn from(rate: UninitializedCostRate) -> Self {
        match rate {
            UninitializedCostRate::PerMillion { cost_per_million } => CostRate {
                cost_per_unit: cost_per_million / MILLION,
            },
            UninitializedCostRate::PerUnit { cost_per_unit } => CostRate { cost_per_unit },
        }
    }
}

/// A normalized cost config entry used at runtime.
///
/// All rates are normalized to per-unit, and pointers are validated.
#[derive(Clone, Debug, PartialEq)]
pub struct CostConfigEntry {
    /// JSON Pointer configuration for extracting the value from the response.
    pub pointer: CostPointerConfig,

    /// Normalized rate (always per-unit).
    pub rate: CostRate,

    /// If `true`, a missing field will corrupt the total cost to `None` (and log a warning).
    /// If `false`, a missing field is treated as zero.
    pub required: bool,
}

impl From<UninitializedCostConfigEntry> for CostConfigEntry {
    fn from(entry: UninitializedCostConfigEntry) -> Self {
        CostConfigEntry {
            pointer: entry.pointer,
            rate: entry.rate.into(),
            required: entry.required,
        }
    }
}

impl CostConfigEntry {
    /// Compute the cost contribution from this single entry given a parsed JSON response.
    ///
    /// Returns:
    /// - `Ok(Some(cost))` if the field was found and successfully parsed
    /// - `Ok(None)` if the field was missing and `required` is `false`
    /// - `Err(())` if the field was missing (and `required` is `true`) or unparseable
    fn compute_contribution(
        &self,
        response_json: &serde_json::Value,
        streaming: bool,
    ) -> Result<Decimal, ()> {
        let pointer = self.pointer.get_pointer(streaming);
        let value = response_json.pointer(pointer.as_str());

        match value {
            Some(v) => {
                let numeric_value = match json_value_to_decimal(v) {
                    Some(d) => d,
                    None => {
                        tracing::warn!(
                            "Cost pointer `{pointer}` value `{v}` cannot be parsed as a number. Cost will be unavailable."
                        );
                        return Err(());
                    }
                };
                Ok(numeric_value * self.rate.cost_per_unit)
            }
            None => {
                if self.required {
                    tracing::warn!(
                        "Required cost pointer `{pointer}` is missing from provider response. Cost will be unavailable."
                    );
                    return Err(());
                }
                // Non-required missing field: contributes zero cost
                Ok(Decimal::ZERO)
            }
        }
    }
}

/// A complete normalized cost configuration for a model provider.
pub type CostConfig = Vec<CostConfigEntry>;

/// Convert an `UninitializedCostConfig` into a normalized `CostConfig`.
///
/// This is called at config load time. All rates are converted to per-unit
/// and JSON pointers are validated.
pub fn load_cost_config(config: UninitializedCostConfig) -> Result<CostConfig, Error> {
    let entries: Vec<CostConfigEntry> = config.into_iter().map(CostConfigEntry::from).collect();
    validate_cost_config(&entries)?;
    Ok(entries)
}

// ─── JSON Pointer configuration ──────────────────────────────────────────────

/// JSON Pointer configuration for cost extraction.
///
/// Either a single `pointer` that applies to both streaming and non-streaming responses,
/// or separate `pointer_nonstreaming` and `pointer_streaming` for different response formats.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[serde(untagged)]
pub enum CostPointerConfig {
    /// A single pointer used for both streaming and non-streaming responses.
    Unified { pointer: JsonPointer },
    /// Separate pointers for streaming and non-streaming responses.
    Split {
        pointer_nonstreaming: JsonPointer,
        pointer_streaming: JsonPointer,
    },
}

impl CostPointerConfig {
    /// Get the appropriate pointer for the given streaming mode.
    pub fn get_pointer(&self, streaming: bool) -> &JsonPointer {
        match self {
            CostPointerConfig::Unified { pointer } => pointer,
            CostPointerConfig::Split {
                pointer_nonstreaming,
                pointer_streaming,
            } => {
                if streaming {
                    pointer_streaming
                } else {
                    pointer_nonstreaming
                }
            }
        }
    }
}

// ─── Cost computation ────────────────────────────────────────────────────────

/// Compute the cost from a raw JSON response string using the given cost configuration.
///
/// Returns `None` if:
/// - The config is empty
/// - A required field is missing from the response
/// - A field value cannot be parsed as a number
/// - The computed cost is negative
///
/// For non-required missing fields, the contribution is treated as zero.
pub fn compute_cost_from_response(
    raw_response: &str,
    cost_config: &[CostConfigEntry],
    streaming: bool,
) -> Option<Decimal> {
    if cost_config.is_empty() {
        return None;
    }

    let response_json: serde_json::Value = match serde_json::from_str(raw_response) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(
                "Failed to parse raw response as JSON for cost computation: {e}. Cost will be unavailable."
            );
            return None;
        }
    };

    compute_cost_from_json(&response_json, cost_config, streaming)
}

/// Compute the cost from a parsed JSON value using the given cost configuration.
///
/// Returns `None` if:
/// - The config is empty
/// - A required field is missing from the response
/// - A field value cannot be parsed as a number
/// - The computed cost is negative
pub fn compute_cost_from_json(
    response_json: &serde_json::Value,
    cost_config: &[CostConfigEntry],
    streaming: bool,
) -> Option<Decimal> {
    if cost_config.is_empty() {
        return None;
    }

    let mut total_cost = Decimal::ZERO;

    for entry in cost_config {
        match entry.compute_contribution(response_json, streaming) {
            Ok(contribution) => total_cost += contribution,
            Err(()) => return None,
        }
    }

    if total_cost < Decimal::ZERO {
        tracing::warn!("Computed cost is negative ({total_cost}). Cost will be unavailable.");
        return None;
    }

    Some(total_cost)
}

/// Convert a JSON value to a Decimal, handling integer, float, and string representations.
fn json_value_to_decimal(value: &serde_json::Value) -> Option<Decimal> {
    match value {
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Some(Decimal::from(i))
            } else if let Some(u) = n.as_u64() {
                Some(Decimal::from(u))
            } else if let Some(f) = n.as_f64() {
                Decimal::try_from(f).ok()
            } else {
                None
            }
        }
        serde_json::Value::String(s) => s.parse::<Decimal>().ok(),
        _ => None,
    }
}

/// Validate a normalized cost configuration.
///
/// This checks that all JSON pointers are valid (non-empty pointers start with `/`).
/// Since `JsonPointer` validates on construction, this is a secondary check.
fn validate_cost_config(cost_config: &[CostConfigEntry]) -> Result<(), Error> {
    for (i, entry) in cost_config.iter().enumerate() {
        match &entry.pointer {
            CostPointerConfig::Unified { pointer } => {
                if !pointer.as_str().is_empty() && !pointer.as_str().starts_with('/') {
                    return Err(Error::new(ErrorDetails::Config {
                        message: format!(
                            "Cost entry {i}: `pointer` must start with `/` (got `{pointer}`)"
                        ),
                    }));
                }
            }
            CostPointerConfig::Split {
                pointer_nonstreaming,
                pointer_streaming,
            } => {
                if !pointer_nonstreaming.as_str().is_empty()
                    && !pointer_nonstreaming.as_str().starts_with('/')
                {
                    return Err(Error::new(ErrorDetails::Config {
                        message: format!(
                            "Cost entry {i}: `pointer_nonstreaming` must start with `/` (got `{pointer_nonstreaming}`)"
                        ),
                    }));
                }
                if !pointer_streaming.as_str().is_empty()
                    && !pointer_streaming.as_str().starts_with('/')
                {
                    return Err(Error::new(ErrorDetails::Config {
                        message: format!(
                            "Cost entry {i}: `pointer_streaming` must start with `/` (got `{pointer_streaming}`)"
                        ),
                    }));
                }
            }
        }
    }
    Ok(())
}

// ─── Helper to build normalized config entries in tests ──────────────────────

#[cfg(test)]
fn make_entry(pointer: &str, cost_per_unit: Decimal, required: bool) -> CostConfigEntry {
    CostConfigEntry {
        pointer: CostPointerConfig::Unified {
            pointer: JsonPointer::new(pointer).unwrap(),
        },
        rate: CostRate { cost_per_unit },
        required,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ── Normalized rate conversion ───────────────────────────────────────

    #[test]
    fn test_per_million_to_per_unit_conversion() {
        let rate = UninitializedCostRate::PerMillion {
            cost_per_million: Decimal::new(150, 2), // 1.50
        };
        let normalized: CostRate = rate.into();
        // 1.50 / 1_000_000 = 0.0000015
        assert_eq!(
            normalized.cost_per_unit,
            Decimal::new(15, 7),
            "per_million rate should be divided by 1_000_000"
        );
    }

    #[test]
    fn test_per_unit_passthrough() {
        let rate = UninitializedCostRate::PerUnit {
            cost_per_unit: Decimal::new(25, 2), // 0.25
        };
        let normalized: CostRate = rate.into();
        assert_eq!(
            normalized.cost_per_unit,
            Decimal::new(25, 2),
            "per_unit rate should pass through unchanged"
        );
    }

    // ── Cost computation (using normalized entries) ──────────────────────

    #[test]
    fn test_compute_cost_basic() {
        let config = vec![
            // 1.50 per million input tokens => 0.0000015 per token
            make_entry("/usage/input_tokens", Decimal::new(150, 2) / MILLION, true),
            // 3.00 per million output tokens => 0.000003 per token
            make_entry("/usage/output_tokens", Decimal::new(300, 2) / MILLION, true),
        ];

        let response = json!({
            "usage": {
                "input_tokens": 1000,
                "output_tokens": 500
            }
        });

        let cost = compute_cost_from_json(&response, &config, false);
        // input: 1000 * 1.50 / 1_000_000 = 0.001500
        // output: 500 * 3.00 / 1_000_000 = 0.001500
        // total: 0.003000
        assert_eq!(
            cost,
            Some(Decimal::new(3000, 6)),
            "cost should be computed correctly from token counts"
        );
    }

    #[test]
    fn test_compute_cost_per_unit() {
        let config = vec![make_entry(
            "/usage/web_searches",
            Decimal::new(25, 2), // 0.25 per unit
            false,
        )];

        let response = json!({
            "usage": {
                "web_searches": 3
            }
        });

        let cost = compute_cost_from_json(&response, &config, false);
        // 3 * 0.25 = 0.75
        assert_eq!(
            cost,
            Some(Decimal::new(75, 2)),
            "cost should be 0.75 for 3 web searches at 0.25 each"
        );
    }

    #[test]
    fn test_compute_cost_required_missing() {
        let config = vec![make_entry("/usage/input_tokens", Decimal::new(15, 7), true)];

        let response = json!({ "usage": {} });

        let cost = compute_cost_from_json(&response, &config, false);
        assert_eq!(
            cost, None,
            "cost should be None when required field is missing"
        );
    }

    #[test]
    fn test_compute_cost_required_missing_logs_warning() {
        let logs_contain = crate::utils::testing::capture_logs();

        let config = vec![make_entry("/usage/input_tokens", Decimal::new(15, 7), true)];

        let response = json!({ "usage": {} });
        let cost = compute_cost_from_json(&response, &config, false);
        assert_eq!(
            cost, None,
            "cost should be None when required field is missing"
        );
        assert!(
            logs_contain(
                "Required cost pointer `/usage/input_tokens` is missing from provider response"
            ),
            "should log a warning when a required cost pointer is missing"
        );
    }

    #[test]
    fn test_compute_cost_optional_missing() {
        let config = vec![make_entry(
            "/usage/web_searches",
            Decimal::new(25, 2),
            false,
        )];

        let response = json!({ "usage": {} });

        let cost = compute_cost_from_json(&response, &config, false);
        assert_eq!(
            cost,
            Some(Decimal::ZERO),
            "cost should be zero when optional field is missing"
        );
    }

    #[test]
    fn test_compute_cost_negative_result() {
        // -3.00 per million = -0.000003 per unit
        let config = vec![make_entry(
            "/usage/cached_tokens",
            Decimal::new(-300, 2) / MILLION,
            true,
        )];

        let response = json!({
            "usage": {
                "cached_tokens": 1000
            }
        });

        let cost = compute_cost_from_json(&response, &config, false);
        assert_eq!(
            cost, None,
            "cost should be None when computed cost is negative"
        );
    }

    #[test]
    fn test_compute_cost_empty_config() {
        let response = json!({ "usage": { "input_tokens": 100 } });
        let cost = compute_cost_from_json(&response, &[], false);
        assert_eq!(cost, None, "cost should be None with empty config");
    }

    #[test]
    fn test_compute_cost_split_pointer() {
        let config = vec![CostConfigEntry {
            pointer: CostPointerConfig::Split {
                pointer_nonstreaming: JsonPointer::new("/usage/total_tokens").unwrap(),
                pointer_streaming: JsonPointer::new("/usage/stream_tokens").unwrap(),
            },
            rate: CostRate {
                cost_per_unit: Decimal::new(200, 2) / MILLION, // 2.00 per million
            },
            required: true,
        }];

        let response_ns = json!({
            "usage": { "total_tokens": 500 }
        });
        let response_s = json!({
            "usage": { "stream_tokens": 300 }
        });

        let cost_ns = compute_cost_from_json(&response_ns, &config, false);
        let cost_s = compute_cost_from_json(&response_s, &config, true);

        // non-streaming: 500 * 2.00 / 1_000_000 = 0.001000
        assert_eq!(
            cost_ns,
            Some(Decimal::new(1000, 6)),
            "non-streaming cost should use pointer_nonstreaming"
        );
        // streaming: 300 * 2.00 / 1_000_000 = 0.000600
        assert_eq!(
            cost_s,
            Some(Decimal::new(600, 6)),
            "streaming cost should use pointer_streaming"
        );
    }

    #[test]
    fn test_compute_cost_unparseable_value() {
        let config = vec![make_entry("/usage/tokens", Decimal::new(15, 7), true)];

        let response = json!({
            "usage": { "tokens": "not_a_number" }
        });

        let cost = compute_cost_from_json(&response, &config, false);
        assert_eq!(
            cost, None,
            "cost should be None when value cannot be parsed as a number"
        );
    }

    /// A realistic caching discount scenario: input_tokens cost + cached_tokens discount.
    /// The discount reduces total cost but doesn't make it negative.
    #[test]
    fn test_compute_cost_mixed_positive_negative() {
        let config = vec![
            make_entry(
                "/usage/input_tokens",
                Decimal::new(150, 2) / MILLION, // 1.50 per million
                true,
            ),
            make_entry(
                "/usage/cached_tokens",
                Decimal::new(-75, 2) / MILLION, // -0.75 per million (50% discount)
                false,
            ),
        ];

        let response = json!({
            "usage": {
                "input_tokens": 1000,
                "cached_tokens": 600
            }
        });

        let cost = compute_cost_from_json(&response, &config, false);
        // input: 1000 * 1.50 / 1_000_000 = 0.001500
        // cached: 600 * (-0.75) / 1_000_000 = -0.000450
        // total: 0.001050
        assert_eq!(
            cost,
            Some(Decimal::new(1050, 6)),
            "caching discount should reduce total cost but keep it positive"
        );
    }

    /// When cached discount exceeds input cost, total goes negative -> returns None.
    #[test]
    fn test_compute_cost_discount_exceeds_base_cost() {
        let config = vec![
            make_entry(
                "/usage/input_tokens",
                Decimal::new(100, 2) / MILLION, // 1.00 per million
                true,
            ),
            make_entry(
                "/usage/cached_tokens",
                Decimal::new(-200, 2) / MILLION, // -2.00 per million
                true,
            ),
        ];

        let response = json!({
            "usage": {
                "input_tokens": 1000,
                "cached_tokens": 1000
            }
        });

        let cost = compute_cost_from_json(&response, &config, false);
        assert_eq!(
            cost, None,
            "cost should be None when caching discount exceeds base cost"
        );
    }

    #[test]
    fn test_compute_cost_from_response_valid_json() {
        let config = vec![make_entry(
            "/usage/input_tokens",
            Decimal::new(150, 2) / MILLION,
            true,
        )];

        let raw_response = r#"{"usage": {"input_tokens": 1000}}"#;
        let cost = compute_cost_from_response(raw_response, &config, false);
        assert_eq!(
            cost,
            Some(Decimal::new(1500, 6)),
            "should parse JSON string and compute cost"
        );
    }

    #[test]
    fn test_compute_cost_from_response_invalid_json() {
        let config = vec![make_entry("/usage/input_tokens", Decimal::new(15, 7), true)];

        let raw_response = "not valid json";
        let cost = compute_cost_from_response(raw_response, &config, false);
        assert_eq!(
            cost, None,
            "should return None for unparseable JSON response"
        );
    }

    #[test]
    fn test_json_value_to_decimal_string_encoded_number() {
        let config = vec![make_entry(
            "/usage/tokens",
            Decimal::new(150, 2) / MILLION,
            true,
        )];

        // Some providers return token counts as strings
        let response = json!({
            "usage": { "tokens": "2000" }
        });

        let cost = compute_cost_from_json(&response, &config, false);
        // 2000 * 1.50 / 1_000_000 = 0.003000
        assert_eq!(
            cost,
            Some(Decimal::new(3000, 6)),
            "should handle string-encoded numeric values from provider responses"
        );
    }

    #[test]
    fn test_json_value_to_decimal_boolean_value() {
        let config = vec![make_entry("/usage/tokens", Decimal::new(15, 7), true)];

        let response = json!({
            "usage": { "tokens": true }
        });

        let cost = compute_cost_from_json(&response, &config, false);
        assert_eq!(
            cost, None,
            "should return None when value is a boolean (not a number)"
        );
    }

    #[test]
    fn test_compute_cost_zero_tokens() {
        let config = vec![make_entry("/usage/input_tokens", Decimal::new(15, 7), true)];

        let response = json!({
            "usage": { "input_tokens": 0 }
        });

        let cost = compute_cost_from_json(&response, &config, false);
        assert_eq!(
            cost,
            Some(Decimal::ZERO),
            "zero tokens should yield zero cost"
        );
    }

    #[test]
    fn test_compute_cost_float_value() {
        let config = vec![make_entry(
            "/cost",
            Decimal::ONE, // 1.0 per unit (pass-through)
            true,
        )];

        // Some providers return cost as a float directly
        let response = json!({
            "cost": 0.0035
        });

        let cost = compute_cost_from_json(&response, &config, false);
        assert!(
            cost.is_some(),
            "should handle float values from provider responses"
        );
        let cost = cost.unwrap();
        // Float precision: 0.0035 should be close to Decimal(35, 4)
        let expected = Decimal::new(35, 4);
        assert!(
            (cost - expected).abs() < Decimal::new(1, 10),
            "float cost should be approximately 0.0035, got {cost}"
        );
    }

    // ── Deserialization / parsing tests ──────────────────────────────────

    #[test]
    fn test_deserialization_per_million_from_toml() {
        let toml_str = r#"
pointer = "/usage/input_tokens"
cost_per_million = 1.50
required = true
"#;
        let entry: UninitializedCostConfigEntry =
            toml::from_str(toml_str).expect("should deserialize from TOML");
        assert_eq!(
            entry.pointer,
            CostPointerConfig::Unified {
                pointer: JsonPointer::new("/usage/input_tokens").unwrap()
            },
            "pointer should be unified"
        );
        assert!(entry.required, "required should be true");
        match &entry.rate {
            UninitializedCostRate::PerMillion { cost_per_million } => {
                // TOML floats go through f64; 1.5 is exactly representable
                assert_eq!(
                    *cost_per_million,
                    Decimal::new(15, 1),
                    "cost_per_million should be 1.5"
                );
            }
            UninitializedCostRate::PerUnit { .. } => panic!("Expected PerMillion rate"),
        }
    }

    #[test]
    fn test_deserialization_per_unit_from_toml() {
        let toml_str = r#"
pointer = "/usage/web_searches"
cost_per_unit = 0.25
"#;
        let entry: UninitializedCostConfigEntry =
            toml::from_str(toml_str).expect("should deserialize from TOML");
        match &entry.rate {
            UninitializedCostRate::PerUnit { cost_per_unit } => {
                assert_eq!(
                    *cost_per_unit,
                    Decimal::new(25, 2),
                    "cost_per_unit should be 0.25"
                );
            }
            UninitializedCostRate::PerMillion { .. } => panic!("Expected PerUnit rate"),
        }
        assert!(!entry.required, "required should default to false");
    }

    #[test]
    fn test_deserialization_split_pointer_from_toml() {
        let toml_str = r#"
pointer_nonstreaming = "/usage/total_tokens"
pointer_streaming = "/usage/stream_tokens"
cost_per_million = 2.00
required = true
"#;
        let entry: UninitializedCostConfigEntry =
            toml::from_str(toml_str).expect("should deserialize split pointer from TOML");
        match &entry.pointer {
            CostPointerConfig::Split {
                pointer_nonstreaming,
                pointer_streaming,
            } => {
                assert_eq!(
                    pointer_nonstreaming.as_str(),
                    "/usage/total_tokens",
                    "pointer_nonstreaming should match"
                );
                assert_eq!(
                    pointer_streaming.as_str(),
                    "/usage/stream_tokens",
                    "pointer_streaming should match"
                );
            }
            CostPointerConfig::Unified { .. } => panic!("Expected Split pointer config"),
        }
        assert!(entry.required, "required should be true");
    }

    #[test]
    fn test_deserialization_invalid_pointer_from_toml() {
        let toml_str = r#"
pointer = "no_leading_slash"
cost_per_million = 1.50
"#;
        let result: Result<UninitializedCostConfigEntry, _> = toml::from_str(toml_str);
        assert!(
            result.is_err(),
            "should reject pointer without leading slash at parse time"
        );
    }

    #[test]
    fn test_deserialization_invalid_split_pointer_from_toml() {
        // Invalid pointer_nonstreaming
        let toml_str = r#"
pointer_nonstreaming = "bad"
pointer_streaming = "/ok"
cost_per_million = 1.50
"#;
        let result: Result<UninitializedCostConfigEntry, _> = toml::from_str(toml_str);
        assert!(
            result.is_err(),
            "should reject split pointer_nonstreaming without leading slash"
        );

        // Invalid pointer_streaming
        let toml_str = r#"
pointer_nonstreaming = "/ok"
pointer_streaming = "bad"
cost_per_million = 1.50
"#;
        let result: Result<UninitializedCostConfigEntry, _> = toml::from_str(toml_str);
        assert!(
            result.is_err(),
            "should reject split pointer_streaming without leading slash"
        );
    }

    #[test]
    fn test_deserialization_missing_rate_from_toml() {
        let toml_str = r#"
pointer = "/usage/input_tokens"
required = true
"#;
        let result: Result<UninitializedCostConfigEntry, _> = toml::from_str(toml_str);
        assert!(
            result.is_err(),
            "should reject config entry with no rate specified"
        );
    }

    // ── UninitializedCostConfigEntry -> CostConfigEntry conversion ───────

    #[test]
    fn test_uninitialized_to_normalized_per_million() {
        let uninitialized = UninitializedCostConfigEntry {
            pointer: CostPointerConfig::Unified {
                pointer: JsonPointer::new("/usage/input_tokens").unwrap(),
            },
            rate: UninitializedCostRate::PerMillion {
                cost_per_million: Decimal::new(150, 2), // 1.50
            },
            required: true,
        };

        let normalized: CostConfigEntry = uninitialized.into();
        assert_eq!(
            normalized.rate.cost_per_unit,
            Decimal::new(15, 7), // 0.0000015
            "per_million should be converted to per_unit by dividing by 1,000,000"
        );
        assert!(normalized.required, "required should be preserved");
    }

    #[test]
    fn test_uninitialized_to_normalized_per_unit() {
        let uninitialized = UninitializedCostConfigEntry {
            pointer: CostPointerConfig::Unified {
                pointer: JsonPointer::new("/usage/web_searches").unwrap(),
            },
            rate: UninitializedCostRate::PerUnit {
                cost_per_unit: Decimal::new(25, 2), // 0.25
            },
            required: false,
        };

        let normalized: CostConfigEntry = uninitialized.into();
        assert_eq!(
            normalized.rate.cost_per_unit,
            Decimal::new(25, 2),
            "per_unit rate should pass through unchanged"
        );
        assert!(!normalized.required, "required should be preserved");
    }

    // ── load_cost_config ─────────────────────────────────────────────────

    #[test]
    fn test_load_cost_config_valid() {
        let config = vec![
            UninitializedCostConfigEntry {
                pointer: CostPointerConfig::Unified {
                    pointer: JsonPointer::new("/usage/input_tokens").unwrap(),
                },
                rate: UninitializedCostRate::PerMillion {
                    cost_per_million: Decimal::new(150, 2),
                },
                required: true,
            },
            UninitializedCostConfigEntry {
                pointer: CostPointerConfig::Split {
                    pointer_nonstreaming: JsonPointer::new("/usage/output_tokens").unwrap(),
                    pointer_streaming: JsonPointer::new("/usage/stream_output_tokens").unwrap(),
                },
                rate: UninitializedCostRate::PerMillion {
                    cost_per_million: Decimal::new(300, 2),
                },
                required: false,
            },
        ];
        let result = load_cost_config(config);
        assert!(
            result.is_ok(),
            "should accept valid config with both unified and split pointers"
        );
        let entries = result.unwrap();
        assert_eq!(entries.len(), 2, "should have 2 entries");
        // First entry: 1.50 / 1_000_000 = 0.0000015
        assert_eq!(
            entries[0].rate.cost_per_unit,
            Decimal::new(15, 7),
            "first entry should have normalized per_unit rate"
        );
    }

    // ── JsonPointer validation ───────────────────────────────────────────

    #[test]
    fn test_json_pointer_valid() {
        assert!(
            JsonPointer::new("/usage/tokens").is_ok(),
            "valid pointer with leading slash"
        );
        assert!(
            JsonPointer::new("").is_ok(),
            "empty pointer is valid per RFC 6901"
        );
        assert!(
            JsonPointer::new("/a/b/c").is_ok(),
            "nested pointer is valid"
        );
    }

    #[test]
    fn test_json_pointer_invalid() {
        assert!(
            JsonPointer::new("no_slash").is_err(),
            "should reject pointer without leading slash"
        );
        assert!(
            JsonPointer::new("usage/tokens").is_err(),
            "should reject pointer without leading slash"
        );
    }
}
