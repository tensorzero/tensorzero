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
//! ## Decimal Precision
//!
//! Cost rates are deserialized from TOML using `rust_decimal::serde::float`,
//! which formats `f64` values to their shortest decimal string before parsing
//! into `Decimal`. This avoids the binary floating-point noise that
//! `Decimal::try_from(f64)` would introduce.
//!
//! For example, `cost_per_million = 0.1` in TOML produces exactly `Decimal(0.1)`,
//! not `Decimal(0.1000000000000000055...)`.

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use crate::error::{Error, ErrorDetails};
use crate::inference::types::usage::RawResponseEntry;

/// Whether the response was obtained via streaming or non-streaming inference.
///
/// Used to select the appropriate JSON Pointer when a cost config has
/// separate `pointer_nonstreaming` / `pointer_streaming` fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResponseMode {
    NonStreaming,
    Streaming,
}

/// The Rust type used for cost values throughout TensorZero.
///
/// Stored as:
/// - Postgres: `NUMERIC(18, 9)`
/// - ClickHouse: `Decimal(18, 9)`
///
/// 18 total digits with 9 fractional digits gives us up to 999,999,999 dollars
/// with sub-nanocent precision — more than enough for inference cost tracking.
pub type Cost = Decimal;

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
        #[serde(with = "rust_decimal::serde::float")]
        cost_per_million: Decimal,
    },
    /// Cost per single unit (e.g., cost per web search).
    PerUnit {
        #[cfg_attr(feature = "ts-bindings", ts(type = "number"))]
        #[serde(with = "rust_decimal::serde::float")]
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

/// Concrete type alias for a parsed JSON Pointer (RFC 6901) from the `json_pointer` crate.
type JsonPtr = json_pointer::JsonPointer<String, Vec<String>>;

/// A normalized pointer config used at runtime.
///
/// Pointers are parsed and validated at config load time using the `json_pointer` crate.
#[derive(Clone, Debug)]
pub enum NormalizedCostPointerConfig {
    /// A single pointer used for both streaming and non-streaming responses.
    Unified { pointer: JsonPtr },
    /// Separate pointers for streaming and non-streaming responses.
    Split {
        pointer_nonstreaming: JsonPtr,
        pointer_streaming: JsonPtr,
    },
}

impl NormalizedCostPointerConfig {
    /// Get the appropriate pointer for the given response mode.
    pub fn get_pointer(&self, mode: ResponseMode) -> &JsonPtr {
        match self {
            NormalizedCostPointerConfig::Unified { pointer } => pointer,
            NormalizedCostPointerConfig::Split {
                pointer_nonstreaming,
                pointer_streaming,
            } => match mode {
                ResponseMode::Streaming => pointer_streaming,
                ResponseMode::NonStreaming => pointer_nonstreaming,
            },
        }
    }
}

/// A normalized cost config entry used at runtime.
///
/// All rates are normalized to per-unit, and pointers are validated.
#[derive(Clone, Debug)]
pub struct CostConfigEntry {
    /// JSON Pointer configuration for extracting the value from the response.
    pub pointer: NormalizedCostPointerConfig,

    /// Normalized rate (always per-unit).
    pub rate: CostRate,

    /// If `true`, a missing field will corrupt the total cost to `None` (and log a warning).
    /// If `false`, a missing field is treated as zero.
    pub required: bool,
}

impl TryFrom<UninitializedCostConfigEntry> for CostConfigEntry {
    type Error = Error;

    fn try_from(entry: UninitializedCostConfigEntry) -> Result<Self, Self::Error> {
        let pointer = match entry.pointer {
            CostPointerConfig::Unified { pointer } => NormalizedCostPointerConfig::Unified {
                pointer: parse_json_pointer(&pointer)?,
            },
            CostPointerConfig::Split {
                pointer_nonstreaming,
                pointer_streaming,
            } => NormalizedCostPointerConfig::Split {
                pointer_nonstreaming: parse_json_pointer(&pointer_nonstreaming)?,
                pointer_streaming: parse_json_pointer(&pointer_streaming)?,
            },
        };
        Ok(CostConfigEntry {
            pointer,
            rate: entry.rate.into(),
            required: entry.required,
        })
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
        mode: ResponseMode,
    ) -> Result<Decimal, ()> {
        let pointer = self.pointer.get_pointer(mode);
        let value = pointer.get(response_json).ok();

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
    config.into_iter().map(CostConfigEntry::try_from).collect()
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
    Unified { pointer: String },
    /// Separate pointers for streaming and non-streaming responses.
    Split {
        pointer_nonstreaming: String,
        pointer_streaming: String,
    },
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
    mode: ResponseMode,
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

    compute_cost_from_json(&response_json, cost_config, mode)
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
    mode: ResponseMode,
) -> Option<Decimal> {
    if cost_config.is_empty() {
        return None;
    }

    let mut total_cost = Decimal::ZERO;

    for entry in cost_config {
        match entry.compute_contribution(response_json, mode) {
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

/// Compute cost from relay raw response entries (non-streaming).
///
/// Used by the edge gateway to compute cost locally from the downstream gateway's
/// raw provider responses. Returns `None` if `cost_config` is empty or no entries are present.
/// Uses poison semantics: if any entry yields `None` cost, the total is `None`.
pub fn compute_relay_non_streaming_cost(
    relay_raw_response: &Option<Vec<RawResponseEntry>>,
    cost_config: &[CostConfigEntry],
) -> Option<Cost> {
    if cost_config.is_empty() {
        return None;
    }
    let entries = relay_raw_response.as_ref()?;
    if entries.is_empty() {
        return None;
    }
    let mut total = Decimal::ZERO;
    for entry in entries {
        let contribution =
            compute_cost_from_response(&entry.data, cost_config, ResponseMode::NonStreaming)?;
        total += contribution;
    }
    if total < Decimal::ZERO {
        tracing::warn!("Computed relay cost is negative ({total}). Cost will be unavailable.");
        return None;
    }
    Some(total)
}

/// Convert a JSON value to a Decimal, handling integer, float, and string representations.
fn json_value_to_decimal(value: &serde_json::Value) -> Option<Decimal> {
    match value {
        serde_json::Value::Number(n) => {
            // Parse through the string representation to avoid f64 precision loss.
            // serde_json formats f64 with the shortest exact decimal (Ryu algorithm),
            // so `Decimal::from_str("0.0035")` gives exactly 0.0035 whereas
            // `Decimal::try_from(0.0035_f64)` picks up binary noise.
            n.to_string().parse::<Decimal>().ok()
        }
        serde_json::Value::String(s) => s.parse::<Decimal>().ok(),
        _ => None,
    }
}

/// Parse a JSON Pointer string into a validated `json_pointer::JsonPointer`.
///
/// Empty strings are treated as pointing to the document root (valid per RFC 6901).
fn parse_json_pointer(pointer: &str) -> Result<JsonPtr, Error> {
    if pointer.is_empty() {
        // Empty pointer is valid per RFC 6901 (points to root).
        // The `json_pointer` crate represents this as an empty segments list.
        return Ok(json_pointer::JsonPointer::new(Vec::<String>::new()));
    }
    pointer.parse::<JsonPtr>().map_err(|e| {
        Error::new(ErrorDetails::Config {
            message: format!("Invalid JSON Pointer `{pointer}`: {e:?}"),
        })
    })
}

// ─── Helper to build normalized config entries in tests ──────────────────────

#[cfg(test)]
fn make_entry(pointer: &str, cost_per_unit: Decimal, required: bool) -> CostConfigEntry {
    CostConfigEntry {
        pointer: NormalizedCostPointerConfig::Unified {
            pointer: parse_json_pointer(pointer).unwrap(),
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

        let cost = compute_cost_from_json(&response, &config, ResponseMode::NonStreaming);
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

        let cost = compute_cost_from_json(&response, &config, ResponseMode::NonStreaming);
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

        let cost = compute_cost_from_json(&response, &config, ResponseMode::NonStreaming);
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
        let cost = compute_cost_from_json(&response, &config, ResponseMode::NonStreaming);
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

        let cost = compute_cost_from_json(&response, &config, ResponseMode::NonStreaming);
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

        let cost = compute_cost_from_json(&response, &config, ResponseMode::NonStreaming);
        assert_eq!(
            cost, None,
            "cost should be None when computed cost is negative"
        );
    }

    #[test]
    fn test_compute_cost_empty_config() {
        let response = json!({ "usage": { "input_tokens": 100 } });
        let cost = compute_cost_from_json(&response, &[], ResponseMode::NonStreaming);
        assert_eq!(cost, None, "cost should be None with empty config");
    }

    #[test]
    fn test_compute_cost_split_pointer() {
        let config = vec![CostConfigEntry {
            pointer: NormalizedCostPointerConfig::Split {
                pointer_nonstreaming: parse_json_pointer("/usage/total_tokens").unwrap(),
                pointer_streaming: parse_json_pointer("/usage/stream_tokens").unwrap(),
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

        let cost_ns = compute_cost_from_json(&response_ns, &config, ResponseMode::NonStreaming);
        let cost_s = compute_cost_from_json(&response_s, &config, ResponseMode::Streaming);

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

        let cost = compute_cost_from_json(&response, &config, ResponseMode::NonStreaming);
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

        let cost = compute_cost_from_json(&response, &config, ResponseMode::NonStreaming);
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

        let cost = compute_cost_from_json(&response, &config, ResponseMode::NonStreaming);
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
        let cost = compute_cost_from_response(raw_response, &config, ResponseMode::NonStreaming);
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
        let cost = compute_cost_from_response(raw_response, &config, ResponseMode::NonStreaming);
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

        let cost = compute_cost_from_json(&response, &config, ResponseMode::NonStreaming);
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

        let cost = compute_cost_from_json(&response, &config, ResponseMode::NonStreaming);
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

        let cost = compute_cost_from_json(&response, &config, ResponseMode::NonStreaming);
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

        let cost = compute_cost_from_json(&response, &config, ResponseMode::NonStreaming);
        assert!(
            cost.is_some(),
            "should handle float values from provider responses"
        );
        let cost = cost.unwrap();
        // json_value_to_decimal parses through string, so this should be exact
        assert_eq!(
            cost,
            Decimal::new(35, 4),
            "float cost should be exactly 0.0035 (parsed via string, not f64)"
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
                pointer: "/usage/input_tokens".to_string()
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
        // Deserialization succeeds (it's just a String), but load_cost_config rejects it
        let toml_str = r#"
pointer = "no_leading_slash"
cost_per_million = 1.50
"#;
        let entry: UninitializedCostConfigEntry =
            toml::from_str(toml_str).expect("TOML deserialization should succeed");
        let result = load_cost_config(vec![entry]);
        assert!(
            result.is_err(),
            "should reject pointer without leading slash at config load time"
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
        let entry: UninitializedCostConfigEntry =
            toml::from_str(toml_str).expect("TOML deserialization should succeed");
        let result = load_cost_config(vec![entry]);
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
        let entry: UninitializedCostConfigEntry =
            toml::from_str(toml_str).expect("TOML deserialization should succeed");
        let result = load_cost_config(vec![entry]);
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

    #[test]
    fn test_deserialization_exact_decimal_precision_from_toml() {
        // Values like 0.1 and 0.3 are not exactly representable in f64.
        // Our custom deserializer formats f64 → string → Decimal, giving exact results.
        let toml_str = r#"
pointer = "/usage/input_tokens"
cost_per_million = 0.1
"#;
        let entry: UninitializedCostConfigEntry =
            toml::from_str(toml_str).expect("should deserialize from TOML");
        match &entry.rate {
            UninitializedCostRate::PerMillion { cost_per_million } => {
                assert_eq!(
                    *cost_per_million,
                    Decimal::new(1, 1), // exactly 0.1
                    "0.1 should deserialize to exact Decimal, not 0.1000000000000000055..."
                );
            }
            UninitializedCostRate::PerUnit { .. } => panic!("expected PerMillion"),
        }

        let toml_str = r#"
pointer = "/usage/output_tokens"
cost_per_million = 0.3
"#;
        let entry: UninitializedCostConfigEntry =
            toml::from_str(toml_str).expect("should deserialize from TOML");
        match &entry.rate {
            UninitializedCostRate::PerMillion { cost_per_million } => {
                assert_eq!(
                    *cost_per_million,
                    Decimal::new(3, 1), // exactly 0.3
                    "0.3 should deserialize to exact Decimal, not 0.29999999999999998..."
                );
            }
            UninitializedCostRate::PerUnit { .. } => panic!("expected PerMillion"),
        }
    }

    #[test]
    fn test_deserialization_string_quoted_decimal_from_toml() {
        // Users can also quote the number for explicit string → Decimal parsing
        let toml_str = r#"
pointer = "/usage/input_tokens"
cost_per_million = "0.1"
"#;
        let entry: UninitializedCostConfigEntry =
            toml::from_str(toml_str).expect("should deserialize string-quoted decimal from TOML");
        match &entry.rate {
            UninitializedCostRate::PerMillion { cost_per_million } => {
                assert_eq!(
                    *cost_per_million,
                    Decimal::new(1, 1),
                    "string-quoted 0.1 should deserialize to exact Decimal"
                );
            }
            UninitializedCostRate::PerUnit { .. } => panic!("expected PerMillion"),
        }
    }

    // ── Config parse error tests (bad strings / missing fields) ───────────

    #[test]
    fn test_deserialization_missing_pointer_from_toml() {
        let toml_str = r"
cost_per_million = 1.50
required = true
";
        let result: Result<UninitializedCostConfigEntry, _> = toml::from_str(toml_str);
        assert!(
            result.is_err(),
            "should reject config entry with no pointer specified"
        );
    }

    #[test]
    fn test_deserialization_invalid_cost_value_from_toml() {
        let toml_str = r#"
pointer = "/usage/input_tokens"
cost_per_million = "not_a_number"
"#;
        let result: Result<UninitializedCostConfigEntry, _> = toml::from_str(toml_str);
        assert!(
            result.is_err(),
            "should reject non-numeric string for cost_per_million"
        );
    }

    #[test]
    fn test_deserialization_negative_cost_from_toml() {
        // Negative costs are valid (e.g. caching discounts)
        let toml_str = r#"
pointer = "/usage/cached_tokens"
cost_per_million = -0.50
"#;
        let entry: UninitializedCostConfigEntry =
            toml::from_str(toml_str).expect("negative costs should be allowed");
        match &entry.rate {
            UninitializedCostRate::PerMillion { cost_per_million } => {
                assert_eq!(
                    *cost_per_million,
                    Decimal::new(-5, 1),
                    "negative cost_per_million should parse correctly"
                );
            }
            UninitializedCostRate::PerUnit { .. } => panic!("expected PerMillion"),
        }
    }

    #[test]
    fn test_deserialization_both_rates_from_toml() {
        // Specifying both cost_per_million and cost_per_unit should pick one (untagged enum)
        let toml_str = r#"
pointer = "/usage/tokens"
cost_per_million = 1.50
cost_per_unit = 0.25
"#;
        // With untagged enum, the first matching variant wins (PerMillion)
        let entry: UninitializedCostConfigEntry =
            toml::from_str(toml_str).expect("should parse when both rates are present");
        assert!(
            matches!(entry.rate, UninitializedCostRate::PerMillion { .. }),
            "should pick PerMillion when both rates are present (first matching variant)"
        );
    }

    #[test]
    fn test_deserialization_empty_pointer_string_from_toml() {
        // Empty pointer is valid (points to the root of the JSON response)
        let toml_str = r#"
pointer = ""
cost_per_unit = 1.0
"#;
        let entry: UninitializedCostConfigEntry =
            toml::from_str(toml_str).expect("empty pointer should be valid (root pointer)");
        match &entry.pointer {
            CostPointerConfig::Unified { pointer } => {
                assert_eq!(pointer.as_str(), "", "empty pointer should be preserved");
            }
            CostPointerConfig::Split { .. } => panic!("expected Unified pointer"),
        }
    }

    // ── UninitializedCostConfigEntry -> CostConfigEntry conversion ───────

    #[test]
    fn test_uninitialized_to_normalized_per_million() {
        let uninitialized = UninitializedCostConfigEntry {
            pointer: CostPointerConfig::Unified {
                pointer: "/usage/input_tokens".to_string(),
            },
            rate: UninitializedCostRate::PerMillion {
                cost_per_million: Decimal::new(150, 2), // 1.50
            },
            required: true,
        };

        let normalized: CostConfigEntry = uninitialized.try_into().unwrap();
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
                pointer: "/usage/web_searches".to_string(),
            },
            rate: UninitializedCostRate::PerUnit {
                cost_per_unit: Decimal::new(25, 2), // 0.25
            },
            required: false,
        };

        let normalized: CostConfigEntry = uninitialized.try_into().unwrap();
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
                    pointer: "/usage/input_tokens".to_string(),
                },
                rate: UninitializedCostRate::PerMillion {
                    cost_per_million: Decimal::new(150, 2),
                },
                required: true,
            },
            UninitializedCostConfigEntry {
                pointer: CostPointerConfig::Split {
                    pointer_nonstreaming: "/usage/output_tokens".to_string(),
                    pointer_streaming: "/usage/stream_output_tokens".to_string(),
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

    // ── JSON Pointer validation (via json_pointer crate) ──────────────────

    #[test]
    fn test_json_pointer_valid() {
        assert!(
            parse_json_pointer("/usage/tokens").is_ok(),
            "valid pointer with leading slash"
        );
        assert!(
            parse_json_pointer("").is_ok(),
            "empty pointer is valid per RFC 6901"
        );
        assert!(
            parse_json_pointer("/a/b/c").is_ok(),
            "nested pointer is valid"
        );
    }

    #[test]
    fn test_json_pointer_invalid() {
        assert!(
            parse_json_pointer("no_slash").is_err(),
            "should reject pointer without leading slash"
        );
        assert!(
            parse_json_pointer("usage/tokens").is_err(),
            "should reject pointer without leading slash"
        );
    }

    // ── Full pipeline tests: TOML config → normalize → provider response → cost ─

    /// Parse a TOML config with `[[entry]]` sections into a normalized `CostConfig`,
    /// then compute cost from a raw provider JSON response.
    fn pipeline_cost(config_toml: &str, raw_response: &str) -> Option<Cost> {
        let entries: Vec<UninitializedCostConfigEntry> = {
            let wrapper: toml::Value = toml::from_str(config_toml).unwrap();
            wrapper
                .get("entry")
                .unwrap()
                .as_array()
                .unwrap()
                .iter()
                .map(|e| e.clone().try_into().unwrap())
                .collect()
        };
        let config = load_cost_config(entries).unwrap();
        compute_cost_from_response(raw_response, &config, ResponseMode::NonStreaming)
    }

    #[test]
    fn test_full_pipeline_openai_style_response() {
        // OpenAI: $0.15/M input, $0.60/M output; 1000 input, 500 output
        let cost = pipeline_cost(
            r#"
[[entry]]
pointer = "/usage/prompt_tokens"
cost_per_million = 0.15
required = true

[[entry]]
pointer = "/usage/completion_tokens"
cost_per_million = 0.60
required = true
"#,
            r#"{"id":"chatcmpl-123","choices":[{"message":{"content":"Hello!"}}],"usage":{"prompt_tokens":1000,"completion_tokens":500}}"#,
        );
        // 1000 * 0.15/1M + 500 * 0.60/1M = 0.00015 + 0.0003 = 0.00045
        assert_eq!(
            cost,
            Some(Decimal::new(45, 5)),
            "OpenAI-style cost should be exactly $0.00045"
        );
    }

    #[test]
    fn test_full_pipeline_anthropic_style_with_cache() {
        // Anthropic: $3/M input, $15/M output, $0.30/M cache_read (optional)
        let config = r#"
[[entry]]
pointer = "/usage/input_tokens"
cost_per_million = 3.00
required = true

[[entry]]
pointer = "/usage/output_tokens"
cost_per_million = 15.00
required = true

[[entry]]
pointer = "/usage/cache_read_input_tokens"
cost_per_million = 0.30
"#;
        // 500 input, 200 output, 100 cached
        let cost = pipeline_cost(
            config,
            r#"{"content":[{"text":"Hi"}],"usage":{"input_tokens":500,"output_tokens":200,"cache_read_input_tokens":100}}"#,
        );
        // 500*3/1M + 200*15/1M + 100*0.30/1M = 0.0015 + 0.003 + 0.00003 = 0.00453
        assert_eq!(
            cost,
            Some(Decimal::new(453, 5)),
            "Anthropic-style cost with cache should be exactly $0.00453"
        );

        // Same config, but response without cache field (optional → treated as 0)
        let cost_no_cache = pipeline_cost(
            config,
            r#"{"content":[{"text":"Hi"}],"usage":{"input_tokens":500,"output_tokens":200}}"#,
        );
        // 500*3/1M + 200*15/1M = 0.0015 + 0.003 = 0.0045
        assert_eq!(
            cost_no_cache,
            Some(Decimal::new(45, 4)),
            "missing optional cache field should not affect cost"
        );
    }

    #[test]
    fn test_full_pipeline_required_field_missing_returns_none() {
        let cost = pipeline_cost(
            r#"
[[entry]]
pointer = "/usage/prompt_tokens"
cost_per_million = 0.15
required = true

[[entry]]
pointer = "/usage/completion_tokens"
cost_per_million = 0.60
required = true
"#,
            r#"{"usage":{"prompt_tokens":1000}}"#, // missing completion_tokens
        );
        assert_eq!(
            cost, None,
            "missing required field should produce None cost"
        );
    }

    #[test]
    fn test_full_pipeline_per_unit_web_search() {
        // Mix of per-million tokens and per-unit web searches
        let cost = pipeline_cost(
            r#"
[[entry]]
pointer = "/usage/input_tokens"
cost_per_million = 0.15
required = true

[[entry]]
pointer = "/usage/completion_tokens"
cost_per_million = 0.60
required = true

[[entry]]
pointer = "/usage/web_searches"
cost_per_unit = 0.03
"#,
            r#"{"usage":{"input_tokens":100,"completion_tokens":50,"web_searches":2}}"#,
        );
        // 100*0.15/1M + 50*0.60/1M + 2*0.03 = 0.000015 + 0.00003 + 0.06 = 0.060045
        assert_eq!(
            cost,
            Some(Decimal::new(60045, 6)),
            "per-unit web search cost should be added to token cost"
        );
    }

    #[test]
    fn test_full_pipeline_invalid_json_response() {
        let cost = pipeline_cost(
            r#"
[[entry]]
pointer = "/usage/prompt_tokens"
cost_per_million = 0.15
required = true
"#,
            "not valid json",
        );
        assert_eq!(cost, None, "invalid JSON response should produce None cost");
    }

    #[test]
    fn test_full_pipeline_zero_tokens() {
        let cost = pipeline_cost(
            r#"
[[entry]]
pointer = "/usage/prompt_tokens"
cost_per_million = 0.15
required = true

[[entry]]
pointer = "/usage/completion_tokens"
cost_per_million = 0.60
required = true
"#,
            r#"{"usage":{"prompt_tokens":0,"completion_tokens":0}}"#,
        );
        assert_eq!(
            cost,
            Some(Decimal::ZERO),
            "zero tokens should produce zero cost, not None"
        );
    }

    #[test]
    fn test_full_pipeline_large_token_counts() {
        // GPT-4 style: $30/M input, $60/M output; large context
        let cost = pipeline_cost(
            r#"
[[entry]]
pointer = "/usage/prompt_tokens"
cost_per_million = 30.00
required = true

[[entry]]
pointer = "/usage/completion_tokens"
cost_per_million = 60.00
required = true
"#,
            r#"{"usage":{"prompt_tokens":100000,"completion_tokens":4000}}"#,
        );
        // 100000*30/1M + 4000*60/1M = 3.0 + 0.24 = 3.24
        assert_eq!(
            cost,
            Some(Decimal::new(324, 2)),
            "large context GPT-4 style cost should be exactly $3.24"
        );
    }

    #[test]
    fn test_full_pipeline_gemini_style_nested_usage() {
        // Gemini uses different field names
        let cost = pipeline_cost(
            r#"
[[entry]]
pointer = "/usageMetadata/promptTokenCount"
cost_per_million = 0.075
required = true

[[entry]]
pointer = "/usageMetadata/candidatesTokenCount"
cost_per_million = 0.30
required = true
"#,
            r#"{"candidates":[{"content":{"parts":[{"text":"Hi"}]}}],"usageMetadata":{"promptTokenCount":200,"candidatesTokenCount":100}}"#,
        );
        // 200*0.075/1M + 100*0.30/1M = 0.000015 + 0.00003 = 0.000045
        assert_eq!(
            cost,
            Some(Decimal::new(45, 6)),
            "Gemini-style nested usage should be exactly $0.000045"
        );
    }

    #[test]
    fn test_full_pipeline_direct_cost_passthrough() {
        // Some providers return cost directly (not per-token)
        let cost = pipeline_cost(
            r#"
[[entry]]
pointer = "/cost"
cost_per_unit = 1.0
required = true
"#,
            r#"{"cost":0.0042,"output":"Hello"}"#,
        );
        assert_eq!(
            cost,
            Some(Decimal::new(42, 4)),
            "direct cost passthrough should be exactly $0.0042"
        );
    }

    // ── compute_relay_non_streaming_cost tests ──────────────────────────

    fn make_raw_response_entry(data: &str) -> RawResponseEntry {
        RawResponseEntry {
            model_inference_id: None,
            provider_type: "dummy".to_string(),
            api_type: crate::inference::types::ApiType::ChatCompletions,
            data: data.to_string(),
        }
    }

    #[test]
    fn test_relay_cost_basic() {
        let config = vec![
            make_entry("/usage/prompt_tokens", Decimal::new(150, 2) / MILLION, true),
            make_entry(
                "/usage/completion_tokens",
                Decimal::new(200, 2) / MILLION,
                true,
            ),
        ];

        let entries = Some(vec![make_raw_response_entry(
            r#"{"usage":{"prompt_tokens":1000,"completion_tokens":500}}"#,
        )]);

        let cost = compute_relay_non_streaming_cost(&entries, &config);
        // 1000 * 1.50 / 1M + 500 * 2.00 / 1M = 0.0015 + 0.001 = 0.0025
        assert_eq!(
            cost,
            Some(Decimal::new(25, 4)),
            "relay cost should be computed from raw response entries"
        );
    }

    #[test]
    fn test_relay_cost_empty_config() {
        let entries = Some(vec![make_raw_response_entry(
            r#"{"usage":{"prompt_tokens":100}}"#,
        )]);
        let cost = compute_relay_non_streaming_cost(&entries, &[]);
        assert_eq!(cost, None, "relay cost should be None with empty config");
    }

    #[test]
    fn test_relay_cost_none_entries() {
        let config = vec![make_entry("/usage/tokens", Decimal::new(15, 7), true)];
        let cost = compute_relay_non_streaming_cost(&None, &config);
        assert_eq!(
            cost, None,
            "relay cost should be None when relay_raw_response is None"
        );
    }

    #[test]
    fn test_relay_cost_empty_entries() {
        let config = vec![make_entry("/usage/tokens", Decimal::new(15, 7), true)];
        let cost = compute_relay_non_streaming_cost(&Some(vec![]), &config);
        assert_eq!(
            cost, None,
            "relay cost should be None when relay_raw_response is empty"
        );
    }

    #[test]
    fn test_relay_cost_multiple_entries_summed() {
        let config = vec![make_entry(
            "/usage/prompt_tokens",
            Decimal::new(150, 2) / MILLION,
            true,
        )];

        let entries = Some(vec![
            make_raw_response_entry(r#"{"usage":{"prompt_tokens":1000}}"#),
            make_raw_response_entry(r#"{"usage":{"prompt_tokens":2000}}"#),
        ]);

        let cost = compute_relay_non_streaming_cost(&entries, &config);
        // entry 1: 1000 * 1.50 / 1M = 0.0015
        // entry 2: 2000 * 1.50 / 1M = 0.003
        // total: 0.0045
        assert_eq!(
            cost,
            Some(Decimal::new(45, 4)),
            "relay cost should sum across multiple raw response entries"
        );
    }

    #[test]
    fn test_relay_cost_poison_on_any_entry() {
        let config = vec![make_entry(
            "/usage/prompt_tokens",
            Decimal::new(150, 2) / MILLION,
            true,
        )];

        // First entry has the required field, second does not
        let entries = Some(vec![
            make_raw_response_entry(r#"{"usage":{"prompt_tokens":1000}}"#),
            make_raw_response_entry(r#"{"usage":{}}"#),
        ]);

        let cost = compute_relay_non_streaming_cost(&entries, &config);
        assert_eq!(
            cost, None,
            "relay cost should be None when any entry fails cost computation (poison semantics)"
        );
    }
}
