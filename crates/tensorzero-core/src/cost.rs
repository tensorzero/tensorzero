use json_pointer::JsonPointer;
use rust_decimal::Decimal;
use serde_json::Value;

use crate::error::{Error, ErrorDetails};
use tensorzero_types::{
    CostPointerConfig, UninitializedCostConfig, UninitializedCostConfigEntry,
    UninitializedCostRate, UninitializedUnifiedCostConfig,
};

/// Decimal type alias for cost values.
pub type Cost = Decimal;

/// Whether the response being costed came from streaming or non-streaming inference.
/// Used to select the correct pointer for `Split` pointer configurations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResponseMode {
    Streaming,
    NonStreaming,
}

/// Compute the cost of a provider response by resolving JSON pointers in the raw response
/// and multiplying extracted values by configured rates.
pub fn compute_cost(
    raw_response: &str,
    cost_config: &CostConfig,
    mode: ResponseMode,
) -> Result<Cost, Error> {
    let json: Value = serde_json::from_str(raw_response).map_err(|e| {
        Error::new(ErrorDetails::CostComputation {
            message: format!("raw response is not valid JSON: {e}"),
        })
    })?;
    let mut total = Decimal::ZERO;

    for entry in cost_config {
        let pointer_str = match &entry.pointer {
            NormalizedCostPointerConfig::Unified { pointer } => pointer.as_str(),
            NormalizedCostPointerConfig::Split {
                pointer_nonstreaming,
                pointer_streaming,
            } => match mode {
                ResponseMode::NonStreaming => pointer_nonstreaming.as_str(),
                ResponseMode::Streaming => pointer_streaming.as_str(),
            },
        };

        let value = json.pointer(pointer_str);
        match value {
            Some(v) => {
                let numeric = value_to_decimal(v).ok_or_else(|| {
                    Error::new(ErrorDetails::CostComputation {
                        message: format!("value at JSON pointer `{pointer_str}` is not numeric"),
                    })
                })?;
                total += numeric * entry.rate.cost_per_unit;
            }
            None => {
                if entry.required {
                    return Err(Error::new(ErrorDetails::CostComputation {
                        message: format!(
                            "required field not found at JSON pointer `{pointer_str}`"
                        ),
                    }));
                }
                // Non-required missing field: skip (contributes 0)
            }
        }
    }

    if total < Decimal::ZERO {
        return Err(Error::new(ErrorDetails::CostComputation {
            message: format!(
                "computed total cost is negative ({total}), which likely indicates a problematic cost configuration"
            ),
        }));
    }

    Ok(total)
}

/// Compute cost from multiple streaming chunks by scanning all chunks per cost config pointer.
///
/// For each config entry, resolves the pointer against every chunk and takes the maximum value found.
/// This correctly handles both:
/// - Cumulative providers (e.g. OpenAI): the same pointer appears in multiple chunks with increasing values → max is correct.
/// - Split-usage providers (e.g. Anthropic): different pointers resolve from different chunks → each max is that pointer's only value.
pub fn compute_cost_from_streaming_chunks(
    raw_chunks: &[&str],
    cost_config: &CostConfig,
) -> Result<Cost, Error> {
    let mut total = Decimal::ZERO;

    for entry in cost_config {
        let pointer_str = match &entry.pointer {
            NormalizedCostPointerConfig::Unified { pointer } => pointer.as_str(),
            NormalizedCostPointerConfig::Split {
                pointer_streaming, ..
            } => pointer_streaming.as_str(),
        };

        let mut max_value: Option<Decimal> = None;

        for raw_chunk in raw_chunks {
            let json: Value = match serde_json::from_str(raw_chunk) {
                Ok(v) => v,
                Err(_) => continue,
            };

            if let Some(v) = json.pointer(pointer_str) {
                let numeric = value_to_decimal(v).ok_or_else(|| {
                    Error::new(ErrorDetails::CostComputation {
                        message: format!("value at JSON pointer `{pointer_str}` is not numeric"),
                    })
                })?;
                max_value = Some(match max_value {
                    Some(current) if current >= numeric => current,
                    _ => numeric,
                });
            }
        }

        match max_value {
            Some(v) => {
                total += v * entry.rate.cost_per_unit;
            }
            None => {
                if entry.required {
                    return Err(Error::new(ErrorDetails::CostComputation {
                        message: format!(
                            "required field not found at JSON pointer `{pointer_str}`"
                        ),
                    }));
                }
            }
        }
    }

    if total < Decimal::ZERO {
        return Err(Error::new(ErrorDetails::CostComputation {
            message: format!(
                "computed total cost is negative ({total}), which likely indicates a problematic cost configuration"
            ),
        }));
    }

    Ok(total)
}

/// Convert a JSON value to a Decimal. Handles integers, floats, and string representations.
fn value_to_decimal(value: &Value) -> Option<Decimal> {
    match value {
        Value::Number(n) => {
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
        Value::String(s) => s.parse::<Decimal>().ok(),
        _ => None,
    }
}

// ============================================================================
// Normalized (runtime) types — after validation and rate normalization
// ============================================================================

pub type CostConfig = Vec<CostConfigEntry>;

#[derive(Clone, Debug)]
pub struct CostConfigEntry {
    pub pointer: NormalizedCostPointerConfig,
    pub rate: CostRate,
    pub required: bool,
}

#[derive(Clone, Debug)]
pub struct CostRate {
    pub cost_per_unit: Decimal,
}

#[derive(Clone, Debug)]
pub enum NormalizedCostPointerConfig {
    Unified {
        pointer: String,
    },
    Split {
        pointer_nonstreaming: String,
        pointer_streaming: String,
    },
}

// ============================================================================
// Validation and normalization
// ============================================================================

fn validate_pointer(pointer: &str) -> Result<(), Error> {
    pointer
        .parse::<JsonPointer<String, Vec<String>>>()
        .map_err(|e| {
            Error::new(ErrorDetails::Config {
                message: format!("invalid JSON pointer `{pointer}`: {e:?}"),
            })
        })?;
    Ok(())
}

pub fn load_cost_config(config: UninitializedCostConfig) -> Result<CostConfig, Error> {
    config.into_iter().map(load_cost_config_entry).collect()
}

/// Load a cost config that only allows unified (non-split) pointers.
///
/// Used for embedding models (which don't support streaming) and batch cost configs.
pub fn load_unified_cost_config(
    config: UninitializedUnifiedCostConfig,
) -> Result<CostConfig, Error> {
    config
        .into_iter()
        .map(load_unified_cost_config_entry)
        .collect()
}

fn parse_pointer_config(config: CostPointerConfig) -> Result<NormalizedCostPointerConfig, Error> {
    match (
        config.pointer,
        config.pointer_nonstreaming,
        config.pointer_streaming,
    ) {
        (Some(pointer), None, None) => {
            validate_pointer(&pointer)?;
            Ok(NormalizedCostPointerConfig::Unified { pointer })
        }
        (None, Some(pointer_nonstreaming), Some(pointer_streaming)) => {
            validate_pointer(&pointer_nonstreaming)?;
            validate_pointer(&pointer_streaming)?;
            Ok(NormalizedCostPointerConfig::Split {
                pointer_nonstreaming,
                pointer_streaming,
            })
        }
        _ => Err(Error::new(ErrorDetails::Config {
            message: "invalid pointer configuration: specify either `pointer` alone, or both `pointer_nonstreaming` and `pointer_streaming`".to_string(),
        })),
    }
}

fn parse_rate(rate: UninitializedCostRate) -> Result<CostRate, Error> {
    let cost_per_unit = match (rate.cost_per_million, rate.cost_per_unit) {
        (Some(cost_per_million), None) => cost_per_million / Decimal::from(1_000_000),
        (None, Some(cost_per_unit)) => cost_per_unit,
        _ => {
            return Err(Error::new(ErrorDetails::Config {
                message: "must specify exactly one of `cost_per_million` or `cost_per_unit`"
                    .to_string(),
            }));
        }
    };
    Ok(CostRate { cost_per_unit })
}

fn load_cost_config_entry(entry: UninitializedCostConfigEntry) -> Result<CostConfigEntry, Error> {
    let pointer = parse_pointer_config(entry.pointer)?;
    let rate = parse_rate(entry.rate)?;
    Ok(CostConfigEntry {
        pointer,
        rate,
        required: entry.required,
    })
}

fn load_unified_cost_config_entry(
    entry: UninitializedCostConfigEntry<tensorzero_types::UnifiedCostPointerConfig>,
) -> Result<CostConfigEntry, Error> {
    validate_pointer(&entry.pointer.pointer)?;
    let rate = parse_rate(entry.rate)?;
    Ok(CostConfigEntry {
        pointer: NormalizedCostPointerConfig::Unified {
            pointer: entry.pointer.pointer,
        },
        rate,
        required: entry.required,
    })
}

#[cfg(test)]
mod tests {
    use serde::Deserialize;

    use super::*;

    fn unified(pointer: &str) -> CostPointerConfig {
        CostPointerConfig {
            pointer: Some(pointer.to_string()),
            pointer_nonstreaming: None,
            pointer_streaming: None,
        }
    }

    fn split(nonstreaming: &str, streaming: &str) -> CostPointerConfig {
        CostPointerConfig {
            pointer: None,
            pointer_nonstreaming: Some(nonstreaming.to_string()),
            pointer_streaming: Some(streaming.to_string()),
        }
    }

    fn per_million(value: Decimal) -> UninitializedCostRate {
        UninitializedCostRate {
            cost_per_million: Some(value),
            cost_per_unit: None,
        }
    }

    fn per_unit(value: Decimal) -> UninitializedCostRate {
        UninitializedCostRate {
            cost_per_million: None,
            cost_per_unit: Some(value),
        }
    }

    #[derive(Deserialize)]
    struct UninitializedCostConfigWrapper {
        cost: UninitializedCostConfig,
    }

    // ========================================================================
    // Rate conversion tests
    // ========================================================================

    #[test]
    fn test_per_million_to_per_unit_normalization() {
        let config = vec![UninitializedCostConfigEntry {
            pointer: unified("/usage/input_tokens"),
            rate: per_million(Decimal::from(3)),
            required: false,
        }];
        let result = load_cost_config(config).expect("should load successfully");
        assert_eq!(result.len(), 1, "should have one entry");
        let expected = Decimal::from(3) / Decimal::from(1_000_000);
        assert_eq!(
            result[0].rate.cost_per_unit, expected,
            "per_million rate should be divided by 1,000,000"
        );
    }

    #[test]
    fn test_per_unit_passthrough() {
        let config = vec![UninitializedCostConfigEntry {
            pointer: unified("/usage/total_cost"),
            rate: per_unit(Decimal::new(5, 2)), // 0.05
            required: true,
        }];
        let result = load_cost_config(config).expect("should load successfully");
        assert_eq!(result.len(), 1, "should have one entry");
        assert_eq!(
            result[0].rate.cost_per_unit,
            Decimal::new(5, 2),
            "per_unit rate should pass through unchanged"
        );
        assert!(result[0].required, "required flag should be preserved");
    }

    // ========================================================================
    // TOML deserialization tests
    // ========================================================================

    #[test]
    fn test_deserialize_unified_pointer_per_million() {
        let toml_str = r#"
[[cost]]
pointer = "/usage/input_tokens"
cost_per_million = 3.0
"#;

        let wrapper: UninitializedCostConfigWrapper =
            toml::from_str(toml_str).expect("should deserialize");
        assert_eq!(wrapper.cost.len(), 1, "should have one cost entry");
        assert!(
            wrapper.cost[0].pointer.pointer.is_some(),
            "should have a unified pointer"
        );
        assert!(
            wrapper.cost[0].rate.cost_per_million.is_some(),
            "should have a per-million rate"
        );
        assert!(
            wrapper.cost[0].rate.cost_per_unit.is_none(),
            "should not have a per-unit rate"
        );
    }

    #[test]
    fn test_deserialize_per_unit_rate() {
        let toml_str = r#"
[[cost]]
pointer = "/cost"
cost_per_unit = 0.05
"#;

        let wrapper: UninitializedCostConfigWrapper =
            toml::from_str(toml_str).expect("should deserialize");
        assert!(
            wrapper.cost[0].rate.cost_per_unit.is_some(),
            "should have a per-unit rate"
        );
        assert!(
            wrapper.cost[0].rate.cost_per_million.is_none(),
            "should not have a per-million rate"
        );
    }

    #[test]
    fn test_deserialize_split_pointers() {
        let toml_str = r#"
[[cost]]
pointer_nonstreaming = "/usage/total"
pointer_streaming = "/usage/stream_total"
cost_per_million = 1.5
"#;

        let wrapper: UninitializedCostConfigWrapper =
            toml::from_str(toml_str).expect("should deserialize");
        assert!(
            wrapper.cost[0].pointer.pointer_nonstreaming.is_some(),
            "should have nonstreaming pointer"
        );
        assert!(
            wrapper.cost[0].pointer.pointer_streaming.is_some(),
            "should have streaming pointer"
        );
        assert!(
            wrapper.cost[0].pointer.pointer.is_none(),
            "should not have unified pointer"
        );
    }

    #[test]
    fn test_invalid_pointer_rejected() {
        let config = vec![UninitializedCostConfigEntry {
            pointer: unified("no_leading_slash"),
            rate: per_million(Decimal::from(1)),
            required: false,
        }];
        let err = load_cost_config(config).expect_err("should fail on invalid pointer");
        let msg = err.to_string();
        assert!(
            msg.contains("invalid JSON pointer"),
            "error should mention invalid JSON pointer: {msg}"
        );
    }

    #[test]
    fn test_invalid_split_pointer_rejected() {
        let config = vec![UninitializedCostConfigEntry {
            pointer: split("/valid", "invalid_no_slash"),
            rate: per_unit(Decimal::from(1)),
            required: false,
        }];
        let err = load_cost_config(config).expect_err("should fail on invalid split pointer");
        let msg = err.to_string();
        assert!(
            msg.contains("invalid JSON pointer"),
            "error should mention invalid JSON pointer: {msg}"
        );
    }

    #[test]
    fn test_missing_rate_rejected() {
        let toml_str = r#"
[[cost]]
pointer = "/usage/tokens"
"#;

        let wrapper: UninitializedCostConfigWrapper = toml::from_str(toml_str)
            .expect("should deserialize with both rates defaulting to None");
        let err =
            load_cost_config(wrapper.cost).expect_err("should fail when neither rate is specified");
        let msg = err.to_string();
        assert!(
            msg.contains("must specify exactly one of"),
            "error should mention missing rate: {msg}"
        );
    }

    #[test]
    fn test_missing_pointer_rejected() {
        let toml_str = r"
[[cost]]
cost_per_million = 3.0
";

        let wrapper: UninitializedCostConfigWrapper = toml::from_str(toml_str)
            .expect("should deserialize with all pointers defaulting to None");
        let err =
            load_cost_config(wrapper.cost).expect_err("should fail when no pointer is specified");
        let msg = err.to_string();
        assert!(
            msg.contains("invalid pointer configuration"),
            "error should mention invalid pointer configuration: {msg}"
        );
    }

    #[test]
    fn test_exact_decimal_precision() {
        let toml_str = r#"
[[cost]]
pointer = "/a"
cost_per_unit = 0.1

[[cost]]
pointer = "/b"
cost_per_unit = 0.3
"#;

        let wrapper: UninitializedCostConfigWrapper =
            toml::from_str(toml_str).expect("should deserialize");
        let result = load_cost_config(wrapper.cost).expect("should load");
        assert_eq!(
            result[0].rate.cost_per_unit,
            Decimal::new(1, 1),
            "0.1 should deserialize with exact decimal precision"
        );
        assert_eq!(
            result[1].rate.cost_per_unit,
            Decimal::new(3, 1),
            "0.3 should deserialize with exact decimal precision"
        );
    }

    #[test]
    fn test_negative_cost_allowed() {
        let config = vec![UninitializedCostConfigEntry {
            pointer: unified("/discount"),
            rate: per_unit(Decimal::new(-5, 2)), // -0.05
            required: false,
        }];
        let result = load_cost_config(config).expect("negative costs should be allowed");
        assert_eq!(
            result[0].rate.cost_per_unit,
            Decimal::new(-5, 2),
            "negative cost should pass through"
        );
    }

    #[test]
    fn test_both_rates_rejected() {
        let toml_str = r#"
[[cost]]
pointer = "/usage/tokens"
cost_per_million = 3.0
cost_per_unit = 0.5
"#;

        let wrapper: UninitializedCostConfigWrapper =
            toml::from_str(toml_str).expect("should deserialize");
        let err = load_cost_config(wrapper.cost)
            .expect_err("should reject config with both rates specified");
        let msg = err.to_string();
        assert!(
            msg.contains("must specify exactly one of"),
            "error should mention rate requirement: {msg}"
        );
    }

    #[test]
    fn test_mixed_pointer_and_split_rejected() {
        let toml_str = r#"
[[cost]]
pointer = "/usage/tokens"
pointer_nonstreaming = "/usage/ns_tokens"
pointer_streaming = "/usage/s_tokens"
cost_per_million = 3.0
"#;

        let wrapper: UninitializedCostConfigWrapper =
            toml::from_str(toml_str).expect("should deserialize");
        let err = load_cost_config(wrapper.cost)
            .expect_err("should reject config with both unified and split pointers");
        let msg = err.to_string();
        assert!(
            msg.contains("invalid pointer configuration"),
            "error should mention invalid pointer configuration: {msg}"
        );
    }

    #[test]
    fn test_partial_split_pointer_rejected() {
        let config = vec![UninitializedCostConfigEntry {
            pointer: CostPointerConfig {
                pointer: None,
                pointer_nonstreaming: Some("/usage/ns".to_string()),
                pointer_streaming: None,
            },
            rate: per_unit(Decimal::from(1)),
            required: false,
        }];
        let err =
            load_cost_config(config).expect_err("should reject config with only one split pointer");
        let msg = err.to_string();
        assert!(
            msg.contains("invalid pointer configuration"),
            "error should mention invalid pointer configuration: {msg}"
        );
    }

    #[test]
    fn test_empty_pointer_string_is_valid_root_pointer() {
        // Per RFC 6901, "" is the root JSON pointer and is valid.
        let config = vec![UninitializedCostConfigEntry {
            pointer: unified(""),
            rate: per_unit(Decimal::from(1)),
            required: false,
        }];
        assert!(
            load_cost_config(config).is_ok(),
            "empty string is a valid root JSON pointer per RFC 6901"
        );
    }

    // ========================================================================
    // Config normalization tests
    // ========================================================================

    #[test]
    fn test_load_cost_config_multi_entry() {
        let config = vec![
            UninitializedCostConfigEntry {
                pointer: unified("/usage/input_tokens"),
                rate: per_million(Decimal::from(3)),
                required: true,
            },
            UninitializedCostConfigEntry {
                pointer: split("/usage/output_tokens", "/usage/stream_output_tokens"),
                rate: per_unit(Decimal::new(15, 6)), // 0.000015
                required: false,
            },
        ];

        let result = load_cost_config(config).expect("should load multi-entry config");
        assert_eq!(result.len(), 2, "should have two entries");

        // First entry: per_million normalized
        let expected_first = Decimal::from(3) / Decimal::from(1_000_000);
        assert_eq!(
            result[0].rate.cost_per_unit, expected_first,
            "first entry rate should be normalized from per_million"
        );
        assert!(result[0].required, "first entry should be required");
        assert!(
            matches!(
                result[0].pointer,
                NormalizedCostPointerConfig::Unified { .. }
            ),
            "first entry should have a unified pointer"
        );

        // Second entry: per_unit passthrough
        assert_eq!(
            result[1].rate.cost_per_unit,
            Decimal::new(15, 6),
            "second entry rate should pass through unchanged"
        );
        assert!(!result[1].required, "second entry should not be required");
        assert!(
            matches!(result[1].pointer, NormalizedCostPointerConfig::Split { .. }),
            "second entry should have split pointers"
        );
    }

    // ========================================================================
    // JSON Pointer validation tests
    // ========================================================================

    #[test]
    fn test_valid_json_pointer() {
        let config = vec![UninitializedCostConfigEntry {
            pointer: unified("/foo/bar"),
            rate: per_unit(Decimal::from(1)),
            required: false,
        }];
        assert!(
            load_cost_config(config).is_ok(),
            "valid JSON pointer `/foo/bar` should be accepted"
        );
    }

    #[test]
    fn test_invalid_json_pointer_no_leading_slash() {
        let config = vec![UninitializedCostConfigEntry {
            pointer: unified("foo/bar"),
            rate: per_unit(Decimal::from(1)),
            required: false,
        }];
        assert!(
            load_cost_config(config).is_err(),
            "JSON pointer without leading `/` should be rejected"
        );
    }

    // ========================================================================
    // compute_cost tests
    // ========================================================================

    fn make_config_entry(
        pointer: NormalizedCostPointerConfig,
        cost_per_unit: Decimal,
        required: bool,
    ) -> CostConfigEntry {
        CostConfigEntry {
            pointer,
            rate: CostRate { cost_per_unit },
            required,
        }
    }

    fn unified_config(pointer: &str, cost_per_unit: Decimal, required: bool) -> CostConfigEntry {
        make_config_entry(
            NormalizedCostPointerConfig::Unified {
                pointer: pointer.to_string(),
            },
            cost_per_unit,
            required,
        )
    }

    fn split_config(
        nonstreaming: &str,
        streaming: &str,
        cost_per_unit: Decimal,
        required: bool,
    ) -> CostConfigEntry {
        make_config_entry(
            NormalizedCostPointerConfig::Split {
                pointer_nonstreaming: nonstreaming.to_string(),
                pointer_streaming: streaming.to_string(),
            },
            cost_per_unit,
            required,
        )
    }

    #[test]
    fn test_compute_cost_per_unit_unified() {
        let raw = r#"{"usage": {"prompt_tokens": 100, "completion_tokens": 50}}"#;
        let config = vec![
            unified_config(
                "/usage/prompt_tokens",
                Decimal::from(3) / Decimal::from(1_000_000),
                false,
            ),
            unified_config(
                "/usage/completion_tokens",
                Decimal::from(15) / Decimal::from(1_000_000),
                false,
            ),
        ];
        let cost = compute_cost(raw, &config, ResponseMode::NonStreaming)
            .expect("should compute cost successfully");
        let expected = Decimal::from(100) * Decimal::from(3) / Decimal::from(1_000_000)
            + Decimal::from(50) * Decimal::from(15) / Decimal::from(1_000_000);
        assert_eq!(
            cost, expected,
            "cost should be sum of (tokens * rate) for each entry"
        );
    }

    #[test]
    fn test_compute_cost_per_unit_direct() {
        let raw = r#"{"cost": 0.05}"#;
        let config = vec![unified_config(
            "/cost",
            Decimal::from(1), // cost_per_unit = 1 means use value directly
            true,
        )];
        let cost = compute_cost(raw, &config, ResponseMode::NonStreaming)
            .expect("should compute cost successfully");
        assert_eq!(
            cost,
            Decimal::new(5, 2),
            "should extract cost directly when rate is 1"
        );
    }

    #[test]
    fn test_compute_cost_split_pointers_nonstreaming() {
        let raw = r#"{"usage": {"total": 200, "stream_total": 999}}"#;
        let config = vec![split_config(
            "/usage/total",
            "/usage/stream_total",
            Decimal::from(1) / Decimal::from(1_000_000),
            false,
        )];
        let cost = compute_cost(raw, &config, ResponseMode::NonStreaming)
            .expect("should compute cost successfully");
        let expected = Decimal::from(200) / Decimal::from(1_000_000);
        assert_eq!(
            cost, expected,
            "non-streaming mode should use nonstreaming pointer"
        );
    }

    #[test]
    fn test_compute_cost_split_pointers_streaming() {
        let raw = r#"{"usage": {"total": 200, "stream_total": 300}}"#;
        let config = vec![split_config(
            "/usage/total",
            "/usage/stream_total",
            Decimal::from(1) / Decimal::from(1_000_000),
            false,
        )];
        let cost = compute_cost(raw, &config, ResponseMode::Streaming)
            .expect("should compute cost successfully");
        let expected = Decimal::from(300) / Decimal::from(1_000_000);
        assert_eq!(
            cost, expected,
            "streaming mode should use streaming pointer"
        );
    }

    #[test]
    fn test_compute_cost_required_field_missing_returns_err() {
        let raw = r#"{"usage": {"prompt_tokens": 100}}"#;
        let config = vec![
            unified_config("/usage/prompt_tokens", Decimal::from(1), false),
            unified_config("/usage/completion_tokens", Decimal::from(1), true), // required but missing
        ];
        let err = compute_cost(raw, &config, ResponseMode::NonStreaming)
            .expect_err("should return Err when a required field is missing");
        assert!(
            err.to_string().contains("required field not found"),
            "should mention missing required field: {err}"
        );
    }

    #[test]
    fn test_compute_cost_nonrequired_field_missing() {
        let raw = r#"{"usage": {"prompt_tokens": 100}}"#;
        let config = vec![
            unified_config("/usage/prompt_tokens", Decimal::from(1), false),
            unified_config("/usage/completion_tokens", Decimal::from(1), false), // not required
        ];
        let cost = compute_cost(raw, &config, ResponseMode::NonStreaming)
            .expect("should compute cost successfully");
        assert_eq!(
            cost,
            Decimal::from(100),
            "should skip non-required missing fields"
        );
    }

    #[test]
    fn test_compute_cost_invalid_json_returns_err() {
        let raw = "not valid json";
        let config = vec![unified_config("/usage/tokens", Decimal::from(1), false)];
        let err = compute_cost(raw, &config, ResponseMode::NonStreaming)
            .expect_err("should return Err for invalid JSON");
        assert!(
            err.to_string().contains("not valid JSON"),
            "should mention invalid JSON: {err}"
        );
    }

    #[test]
    fn test_compute_cost_negative_total_returns_err() {
        // Set up a config where the total will be negative
        let raw = r#"{"tokens": 10}"#;
        let config = vec![unified_config(
            "/tokens",
            Decimal::new(-5, 0), // -5 per unit → total = -50
            false,
        )];
        let err = compute_cost(raw, &config, ResponseMode::NonStreaming)
            .expect_err("should return Err when total cost is negative");
        assert!(
            err.to_string().contains("negative"),
            "should mention negative total: {err}"
        );
    }

    #[test]
    fn test_compute_cost_empty_config() {
        let raw = r#"{"usage": {"tokens": 100}}"#;
        let config: CostConfig = vec![];
        let cost = compute_cost(raw, &config, ResponseMode::NonStreaming)
            .expect("should compute cost successfully");
        assert_eq!(cost, Decimal::ZERO, "empty config should return Ok(0)");
    }

    #[test]
    fn test_compute_cost_non_numeric_value_returns_err() {
        let raw = r#"{"usage": {"tokens": "not_a_number"}}"#;
        let config = vec![unified_config("/usage/tokens", Decimal::from(1), false)];
        let err = compute_cost(raw, &config, ResponseMode::NonStreaming)
            .expect_err("should return Err for non-numeric field values");
        assert!(
            err.to_string().contains("not numeric"),
            "should mention non-numeric value: {err}"
        );
    }

    #[test]
    fn test_compute_cost_string_numeric_value() {
        let raw = r#"{"usage": {"tokens": "100"}}"#;
        let config = vec![unified_config("/usage/tokens", Decimal::from(1), false)];
        let cost = compute_cost(raw, &config, ResponseMode::NonStreaming)
            .expect("should compute cost successfully");
        assert_eq!(cost, Decimal::from(100), "should parse numeric strings");
    }

    #[test]
    fn test_compute_cost_boolean_value_returns_err() {
        let raw = r#"{"usage": {"tokens": true}}"#;
        let config = vec![unified_config("/usage/tokens", Decimal::from(1), false)];
        let err = compute_cost(raw, &config, ResponseMode::NonStreaming)
            .expect_err("should return Err for boolean field values");
        assert!(
            err.to_string().contains("not numeric"),
            "should mention non-numeric value: {err}"
        );
    }

    // ========================================================================
    // compute_cost_from_streaming_chunks tests
    // ========================================================================

    #[test]
    fn test_streaming_chunks_split_usage_anthropic_style() {
        // Anthropic sends input_tokens in message_start and output_tokens in message_delta
        let chunk1 = r#"{"usage": {"input_tokens": 69}}"#;
        let chunk2 = r#"{"usage": {"output_tokens": 100}}"#;
        let input_rate = Decimal::from(3) / Decimal::from(1_000_000);
        let output_rate = Decimal::from(15) / Decimal::from(1_000_000);
        let config = vec![
            split_config(
                "/usage/input_tokens",
                "/usage/input_tokens",
                input_rate,
                false,
            ),
            split_config(
                "/usage/output_tokens",
                "/usage/output_tokens",
                output_rate,
                false,
            ),
        ];
        let chunks: Vec<&str> = vec![chunk1, chunk2];
        let cost = compute_cost_from_streaming_chunks(&chunks, &config)
            .expect("should compute cost successfully");
        let expected = Decimal::from(69) * input_rate + Decimal::from(100) * output_rate;
        assert_eq!(
            cost, expected,
            "should sum costs from different chunks for split-usage providers"
        );
    }

    #[test]
    fn test_streaming_chunks_single_chunk_openai_style() {
        // OpenAI sends all usage in a single final chunk
        let chunk = r#"{"usage": {"prompt_tokens": 100, "completion_tokens": 50}}"#;
        let config = vec![
            unified_config(
                "/usage/prompt_tokens",
                Decimal::from(3) / Decimal::from(1_000_000),
                false,
            ),
            unified_config(
                "/usage/completion_tokens",
                Decimal::from(15) / Decimal::from(1_000_000),
                false,
            ),
        ];
        let chunks: Vec<&str> = vec![chunk];
        let cost = compute_cost_from_streaming_chunks(&chunks, &config)
            .expect("should compute cost successfully");
        let expected = Decimal::from(100) * Decimal::from(3) / Decimal::from(1_000_000)
            + Decimal::from(50) * Decimal::from(15) / Decimal::from(1_000_000);
        assert_eq!(
            cost, expected,
            "should compute correct cost from a single chunk"
        );
    }

    #[test]
    fn test_streaming_chunks_cumulative_values() {
        // Provider sends cumulative token counts across chunks
        let chunk1 = r#"{"usage": {"tokens": 50}}"#;
        let chunk2 = r#"{"usage": {"tokens": 100}}"#;
        let chunk3 = r#"{"usage": {"tokens": 150}}"#;
        let config = vec![unified_config("/usage/tokens", Decimal::from(1), false)];
        let chunks: Vec<&str> = vec![chunk1, chunk2, chunk3];
        let cost = compute_cost_from_streaming_chunks(&chunks, &config)
            .expect("should compute cost successfully");
        assert_eq!(
            cost,
            Decimal::from(150),
            "should take the max value across cumulative chunks"
        );
    }

    #[test]
    fn test_streaming_chunks_required_field_missing_from_all() {
        let chunk1 = r#"{"usage": {"other": 10}}"#;
        let chunk2 = r#"{"usage": {"other": 20}}"#;
        let config = vec![unified_config("/usage/tokens", Decimal::from(1), true)];
        let chunks: Vec<&str> = vec![chunk1, chunk2];
        let err = compute_cost_from_streaming_chunks(&chunks, &config)
            .expect_err("should return Err when a required field is not found in any chunk");
        assert!(
            err.to_string().contains("required field not found"),
            "should mention missing required field: {err}"
        );
    }

    #[test]
    fn test_streaming_chunks_empty_chunks_list() {
        let config = vec![unified_config("/usage/tokens", Decimal::from(1), false)];
        let chunks: Vec<&str> = vec![];
        let cost = compute_cost_from_streaming_chunks(&chunks, &config)
            .expect("should compute cost successfully");
        assert_eq!(
            cost,
            Decimal::ZERO,
            "empty chunks with non-required fields should return Ok(0)"
        );
    }

    #[test]
    fn test_streaming_chunks_empty_chunks_required_field() {
        let config = vec![unified_config("/usage/tokens", Decimal::from(1), true)];
        let chunks: Vec<&str> = vec![];
        let err = compute_cost_from_streaming_chunks(&chunks, &config)
            .expect_err("should return Err for empty chunks with required fields");
        assert!(
            err.to_string().contains("required field not found"),
            "should mention missing required field: {err}"
        );
    }

    #[test]
    fn test_streaming_chunks_invalid_json_skipped() {
        let chunk1 = "not valid json";
        let chunk2 = r#"{"usage": {"tokens": 42}}"#;
        let config = vec![unified_config("/usage/tokens", Decimal::from(1), false)];
        let chunks: Vec<&str> = vec![chunk1, chunk2];
        let cost = compute_cost_from_streaming_chunks(&chunks, &config)
            .expect("should compute cost successfully");
        assert_eq!(
            cost,
            Decimal::from(42),
            "should skip invalid JSON chunks and use valid ones"
        );
    }

    #[test]
    fn test_streaming_chunks_negative_total_returns_err() {
        let chunk = r#"{"tokens": 10}"#;
        let config = vec![unified_config(
            "/tokens",
            Decimal::new(-5, 0), // -5 per unit → total = -50
            false,
        )];
        let chunks: Vec<&str> = vec![chunk];
        let err = compute_cost_from_streaming_chunks(&chunks, &config)
            .expect_err("should return Err when total cost is negative");
        assert!(
            err.to_string().contains("negative"),
            "should mention negative total: {err}"
        );
    }

    // ========================================================================
    // load_unified_cost_config tests
    // ========================================================================

    fn unified_entry(
        pointer: &str,
        rate: UninitializedCostRate,
        required: bool,
    ) -> UninitializedCostConfigEntry<tensorzero_types::UnifiedCostPointerConfig> {
        UninitializedCostConfigEntry {
            pointer: tensorzero_types::UnifiedCostPointerConfig {
                pointer: pointer.to_string(),
            },
            rate,
            required,
        }
    }

    #[test]
    fn test_load_unified_cost_config_valid() {
        let config = vec![
            unified_entry("/usage/input_tokens", per_million(Decimal::from(1)), true),
            unified_entry("/usage/output_tokens", per_unit(Decimal::new(5, 6)), false),
        ];
        let result =
            load_unified_cost_config(config).expect("should load valid unified cost config");
        assert_eq!(result.len(), 2, "should have two entries");

        assert!(
            matches!(
                result[0].pointer,
                NormalizedCostPointerConfig::Unified { .. }
            ),
            "unified cost entries should always be unified pointers"
        );
        assert!(result[0].required, "first entry should be required");

        let expected_rate = Decimal::from(1) / Decimal::from(1_000_000);
        assert_eq!(
            result[0].rate.cost_per_unit, expected_rate,
            "per_million rate should be normalized"
        );
    }

    #[test]
    fn test_load_unified_cost_config_invalid_pointer_rejected() {
        let config = vec![unified_entry(
            "no_leading_slash",
            per_million(Decimal::from(1)),
            false,
        )];
        let err = load_unified_cost_config(config)
            .expect_err("should fail on invalid unified cost pointer");
        let msg = err.to_string();
        assert!(
            msg.contains("invalid JSON pointer"),
            "error should mention invalid JSON pointer: {msg}"
        );
    }

    #[test]
    fn test_load_unified_cost_config_missing_rate_rejected() {
        let config = vec![unified_entry(
            "/usage/tokens",
            UninitializedCostRate {
                cost_per_million: None,
                cost_per_unit: None,
            },
            false,
        )];
        let err = load_unified_cost_config(config)
            .expect_err("should fail when neither rate is specified");
        let msg = err.to_string();
        assert!(
            msg.contains("must specify exactly one of"),
            "error should mention missing rate: {msg}"
        );
    }

    #[derive(Deserialize)]
    struct UninitializedUnifiedCostConfigWrapper {
        batch_cost: UninitializedUnifiedCostConfig,
    }

    #[test]
    fn test_deserialize_unified_cost_config() {
        let toml_str = r#"
[[batch_cost]]
pointer = "/usage/input_tokens"
cost_per_million = 1.5

[[batch_cost]]
pointer = "/usage/output_tokens"
cost_per_million = 6.0
required = true
"#;

        let wrapper: UninitializedUnifiedCostConfigWrapper =
            toml::from_str(toml_str).expect("should deserialize unified cost config");
        assert_eq!(
            wrapper.batch_cost.len(),
            2,
            "should have two unified cost entries"
        );
        assert_eq!(
            wrapper.batch_cost[0].pointer.pointer, "/usage/input_tokens",
            "first pointer should match"
        );
        assert!(
            !wrapper.batch_cost[0].required,
            "required should default to false"
        );
        assert!(
            wrapper.batch_cost[1].required,
            "required should be true when set"
        );
    }
}
