//! Cost tracking configuration and computation.
//!
//! This module provides types and functions for computing inference cost from
//! provider responses using user-configured JSON Pointer mappings.
//!
//! Cost is computed by extracting numeric values from the raw provider response
//! using JSON Pointers (RFC 6901) and multiplying by configured rates.

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

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

/// A single cost mapping entry that maps a field in the provider response to a cost rate.
///
/// Each entry extracts a numeric value from the raw response using a JSON Pointer
/// and multiplies it by the configured rate to get the cost contribution.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
pub struct CostConfigEntry {
    /// JSON Pointer configuration for extracting the value from the response.
    #[serde(flatten)]
    pub pointer: CostPointerConfig,

    /// The rate to apply to the extracted value.
    #[serde(flatten)]
    pub rate: CostRate,

    /// If `true`, a missing field will corrupt the total cost to `None` (and log a warning).
    /// If `false` (default), a missing field is treated as zero.
    #[serde(default)]
    pub required: bool,
}

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

impl CostPointerConfig {
    /// Get the appropriate pointer for the given streaming mode.
    pub fn get_pointer(&self, streaming: bool) -> &str {
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

/// The rate to apply to an extracted value.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[serde(untagged)]
pub enum CostRate {
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

/// A complete cost configuration for a model provider.
pub type CostConfig = Vec<CostConfigEntry>;

/// The divisor for `cost_per_million`.
const MILLION: Decimal = Decimal::from_parts(1_000_000, 0, 0, false, 0);

/// Compute the cost from a raw JSON response using the given cost configuration.
///
/// Returns `None` if:
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
        let pointer = entry.pointer.get_pointer(streaming);
        let value = response_json.pointer(pointer);

        match value {
            Some(v) => {
                let numeric_value = match json_value_to_decimal(v) {
                    Some(d) => d,
                    None => {
                        tracing::warn!(
                            "Cost pointer `{pointer}` value `{v}` cannot be parsed as a number. Cost will be unavailable."
                        );
                        return None;
                    }
                };

                let cost_contribution = match &entry.rate {
                    CostRate::PerMillion { cost_per_million } => {
                        numeric_value * cost_per_million / MILLION
                    }
                    CostRate::PerUnit { cost_per_unit } => numeric_value * cost_per_unit,
                };

                total_cost += cost_contribution;
            }
            None => {
                if entry.required {
                    tracing::warn!(
                        "Required cost pointer `{pointer}` is missing from provider response. Cost will be unavailable."
                    );
                    return None;
                }
                // Non-required missing field: contributes zero cost
            }
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

/// Validate a cost configuration, returning an error if it's invalid.
pub fn validate_cost_config(cost_config: &[CostConfigEntry]) -> Result<(), Error> {
    for (i, entry) in cost_config.iter().enumerate() {
        match &entry.pointer {
            CostPointerConfig::Unified { pointer } => {
                if !pointer.starts_with('/') {
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
                if !pointer_nonstreaming.starts_with('/') {
                    return Err(Error::new(ErrorDetails::Config {
                        message: format!(
                            "Cost entry {i}: `pointer_nonstreaming` must start with `/` (got `{pointer_nonstreaming}`)"
                        ),
                    }));
                }
                if !pointer_streaming.starts_with('/') {
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_compute_cost_basic() {
        let config = vec![
            CostConfigEntry {
                pointer: CostPointerConfig::Unified {
                    pointer: "/usage/input_tokens".to_string(),
                },
                rate: CostRate::PerMillion {
                    cost_per_million: Decimal::new(150, 2), // 1.50
                },
                required: true,
            },
            CostConfigEntry {
                pointer: CostPointerConfig::Unified {
                    pointer: "/usage/output_tokens".to_string(),
                },
                rate: CostRate::PerMillion {
                    cost_per_million: Decimal::new(300, 2), // 3.00
                },
                required: true,
            },
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
        let config = vec![CostConfigEntry {
            pointer: CostPointerConfig::Unified {
                pointer: "/usage/web_searches".to_string(),
            },
            rate: CostRate::PerUnit {
                cost_per_unit: Decimal::new(25, 2), // 0.25
            },
            required: false,
        }];

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
        let config = vec![CostConfigEntry {
            pointer: CostPointerConfig::Unified {
                pointer: "/usage/input_tokens".to_string(),
            },
            rate: CostRate::PerMillion {
                cost_per_million: Decimal::new(150, 2),
            },
            required: true,
        }];

        let response = json!({ "usage": {} });

        let cost = compute_cost_from_json(&response, &config, false);
        assert_eq!(
            cost, None,
            "cost should be None when required field is missing"
        );
    }

    #[test]
    fn test_compute_cost_optional_missing() {
        let config = vec![CostConfigEntry {
            pointer: CostPointerConfig::Unified {
                pointer: "/usage/web_searches".to_string(),
            },
            rate: CostRate::PerUnit {
                cost_per_unit: Decimal::new(25, 2),
            },
            required: false,
        }];

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
        let config = vec![CostConfigEntry {
            pointer: CostPointerConfig::Unified {
                pointer: "/usage/cached_tokens".to_string(),
            },
            rate: CostRate::PerMillion {
                cost_per_million: Decimal::new(-300, 2), // -3.00 (discount)
            },
            required: true,
        }];

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
                pointer_nonstreaming: "/usage/total_tokens".to_string(),
                pointer_streaming: "/usage/stream_tokens".to_string(),
            },
            rate: CostRate::PerMillion {
                cost_per_million: Decimal::new(200, 2), // 2.00
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
        let config = vec![CostConfigEntry {
            pointer: CostPointerConfig::Unified {
                pointer: "/usage/tokens".to_string(),
            },
            rate: CostRate::PerMillion {
                cost_per_million: Decimal::new(150, 2),
            },
            required: true,
        }];

        let response = json!({
            "usage": { "tokens": "not_a_number" }
        });

        let cost = compute_cost_from_json(&response, &config, false);
        assert_eq!(
            cost, None,
            "cost should be None when value cannot be parsed as a number"
        );
    }

    #[test]
    fn test_deserialization_from_toml() {
        let toml_str = r#"
pointer = "/usage/input_tokens"
cost_per_million = 1.50
required = true
"#;
        let entry: CostConfigEntry =
            toml::from_str(toml_str).expect("should deserialize from TOML");
        assert_eq!(
            entry.pointer,
            CostPointerConfig::Unified {
                pointer: "/usage/input_tokens".to_string()
            },
            "pointer should be unified"
        );
        assert!(entry.required, "required should be true");
    }

    #[test]
    fn test_deserialization_per_unit_from_toml() {
        let toml_str = r#"
pointer = "/usage/web_searches"
cost_per_unit = 0.25
"#;
        let entry: CostConfigEntry =
            toml::from_str(toml_str).expect("should deserialize from TOML");
        match &entry.rate {
            CostRate::PerUnit { cost_per_unit } => {
                assert_eq!(
                    *cost_per_unit,
                    Decimal::new(25, 2),
                    "cost_per_unit should be 0.25"
                );
            }
            _ => panic!("Expected PerUnit rate"),
        }
        assert!(!entry.required, "required should default to false");
    }

    #[test]
    fn test_validate_cost_config_invalid_pointer() {
        let config = vec![CostConfigEntry {
            pointer: CostPointerConfig::Unified {
                pointer: "no_leading_slash".to_string(),
            },
            rate: CostRate::PerMillion {
                cost_per_million: Decimal::new(150, 2),
            },
            required: false,
        }];
        assert!(
            validate_cost_config(&config).is_err(),
            "should reject pointer without leading slash"
        );
    }

    #[test]
    fn test_validate_cost_config_invalid_split_pointer() {
        // Invalid pointer_nonstreaming
        let config = vec![CostConfigEntry {
            pointer: CostPointerConfig::Split {
                pointer_nonstreaming: "bad".to_string(),
                pointer_streaming: "/ok".to_string(),
            },
            rate: CostRate::PerMillion {
                cost_per_million: Decimal::new(150, 2),
            },
            required: false,
        }];
        assert!(
            validate_cost_config(&config).is_err(),
            "should reject split pointer_nonstreaming without leading slash"
        );

        // Invalid pointer_streaming
        let config = vec![CostConfigEntry {
            pointer: CostPointerConfig::Split {
                pointer_nonstreaming: "/ok".to_string(),
                pointer_streaming: "bad".to_string(),
            },
            rate: CostRate::PerMillion {
                cost_per_million: Decimal::new(150, 2),
            },
            required: false,
        }];
        assert!(
            validate_cost_config(&config).is_err(),
            "should reject split pointer_streaming without leading slash"
        );
    }

    #[test]
    fn test_validate_cost_config_valid() {
        let config = vec![
            CostConfigEntry {
                pointer: CostPointerConfig::Unified {
                    pointer: "/usage/input_tokens".to_string(),
                },
                rate: CostRate::PerMillion {
                    cost_per_million: Decimal::new(150, 2),
                },
                required: true,
            },
            CostConfigEntry {
                pointer: CostPointerConfig::Split {
                    pointer_nonstreaming: "/usage/output_tokens".to_string(),
                    pointer_streaming: "/usage/stream_output_tokens".to_string(),
                },
                rate: CostRate::PerMillion {
                    cost_per_million: Decimal::new(300, 2),
                },
                required: false,
            },
        ];
        assert!(
            validate_cost_config(&config).is_ok(),
            "should accept valid config with both unified and split pointers"
        );
    }

    /// A realistic caching discount scenario: input_tokens cost + cached_tokens discount.
    /// The discount reduces total cost but doesn't make it negative.
    #[test]
    fn test_compute_cost_mixed_positive_negative() {
        let config = vec![
            CostConfigEntry {
                pointer: CostPointerConfig::Unified {
                    pointer: "/usage/input_tokens".to_string(),
                },
                rate: CostRate::PerMillion {
                    cost_per_million: Decimal::new(150, 2), // 1.50 per million
                },
                required: true,
            },
            CostConfigEntry {
                pointer: CostPointerConfig::Unified {
                    pointer: "/usage/cached_tokens".to_string(),
                },
                rate: CostRate::PerMillion {
                    cost_per_million: Decimal::new(-75, 2), // -0.75 per million (50% discount)
                },
                required: false,
            },
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

    /// When cached discount exceeds input cost, total goes negative → returns None.
    #[test]
    fn test_compute_cost_discount_exceeds_base_cost() {
        let config = vec![
            CostConfigEntry {
                pointer: CostPointerConfig::Unified {
                    pointer: "/usage/input_tokens".to_string(),
                },
                rate: CostRate::PerMillion {
                    cost_per_million: Decimal::new(100, 2), // 1.00
                },
                required: true,
            },
            CostConfigEntry {
                pointer: CostPointerConfig::Unified {
                    pointer: "/usage/cached_tokens".to_string(),
                },
                rate: CostRate::PerMillion {
                    cost_per_million: Decimal::new(-200, 2), // -2.00 (aggressive discount)
                },
                required: true,
            },
        ];

        let response = json!({
            "usage": {
                "input_tokens": 1000,
                "cached_tokens": 1000
            }
        });

        let cost = compute_cost_from_json(&response, &config, false);
        // input: 1000 * 1.00 / 1M = 0.001000
        // cached: 1000 * (-2.00) / 1M = -0.002000
        // total: -0.001000 → None
        assert_eq!(
            cost, None,
            "cost should be None when caching discount exceeds base cost"
        );
    }

    #[test]
    fn test_compute_cost_from_response_valid_json() {
        let config = vec![CostConfigEntry {
            pointer: CostPointerConfig::Unified {
                pointer: "/usage/input_tokens".to_string(),
            },
            rate: CostRate::PerMillion {
                cost_per_million: Decimal::new(150, 2),
            },
            required: true,
        }];

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
        let config = vec![CostConfigEntry {
            pointer: CostPointerConfig::Unified {
                pointer: "/usage/input_tokens".to_string(),
            },
            rate: CostRate::PerMillion {
                cost_per_million: Decimal::new(150, 2),
            },
            required: true,
        }];

        let raw_response = "not valid json";
        let cost = compute_cost_from_response(raw_response, &config, false);
        assert_eq!(
            cost, None,
            "should return None for unparseable JSON response"
        );
    }

    #[test]
    fn test_json_value_to_decimal_string_encoded_number() {
        let config = vec![CostConfigEntry {
            pointer: CostPointerConfig::Unified {
                pointer: "/usage/tokens".to_string(),
            },
            rate: CostRate::PerMillion {
                cost_per_million: Decimal::new(150, 2),
            },
            required: true,
        }];

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
        let config = vec![CostConfigEntry {
            pointer: CostPointerConfig::Unified {
                pointer: "/usage/tokens".to_string(),
            },
            rate: CostRate::PerMillion {
                cost_per_million: Decimal::new(150, 2),
            },
            required: true,
        }];

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
        let config = vec![CostConfigEntry {
            pointer: CostPointerConfig::Unified {
                pointer: "/usage/input_tokens".to_string(),
            },
            rate: CostRate::PerMillion {
                cost_per_million: Decimal::new(150, 2),
            },
            required: true,
        }];

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
        let config = vec![CostConfigEntry {
            pointer: CostPointerConfig::Unified {
                pointer: "/cost".to_string(),
            },
            rate: CostRate::PerUnit {
                cost_per_unit: Decimal::ONE,
            },
            required: true,
        }];

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

    #[test]
    fn test_deserialization_split_pointer_from_toml() {
        let toml_str = r#"
pointer_nonstreaming = "/usage/total_tokens"
pointer_streaming = "/usage/stream_tokens"
cost_per_million = 2.00
required = true
"#;
        let entry: CostConfigEntry =
            toml::from_str(toml_str).expect("should deserialize split pointer from TOML");
        match &entry.pointer {
            CostPointerConfig::Split {
                pointer_nonstreaming,
                pointer_streaming,
            } => {
                assert_eq!(
                    pointer_nonstreaming, "/usage/total_tokens",
                    "pointer_nonstreaming should match"
                );
                assert_eq!(
                    pointer_streaming, "/usage/stream_tokens",
                    "pointer_streaming should match"
                );
            }
            _ => panic!("Expected Split pointer config"),
        }
        assert!(entry.required, "required should be true");
    }
}
