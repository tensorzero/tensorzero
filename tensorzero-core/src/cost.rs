use json_pointer::JsonPointer;
use rust_decimal::Decimal;

use crate::error::{Error, ErrorDetails};
use tensorzero_types::{UninitializedCostConfig, UninitializedCostConfigEntry};

/// Decimal type alias for cost values.
pub type Cost = Decimal;

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

fn load_cost_config_entry(entry: UninitializedCostConfigEntry) -> Result<CostConfigEntry, Error> {
    let pointer = match (
        entry.pointer.pointer,
        entry.pointer.pointer_nonstreaming,
        entry.pointer.pointer_streaming,
    ) {
        (Some(pointer), None, None) => {
            validate_pointer(&pointer)?;
            NormalizedCostPointerConfig::Unified { pointer }
        }
        (None, Some(pointer_nonstreaming), Some(pointer_streaming)) => {
            validate_pointer(&pointer_nonstreaming)?;
            validate_pointer(&pointer_streaming)?;
            NormalizedCostPointerConfig::Split {
                pointer_nonstreaming,
                pointer_streaming,
            }
        }
        (None, None, None) => {
            return Err(Error::new(ErrorDetails::Config {
                message: "must specify either `pointer` or both `pointer_nonstreaming` and `pointer_streaming`".to_string(),
            }));
        }
        _ => {
            return Err(Error::new(ErrorDetails::Config {
                message: "invalid pointer configuration: specify either `pointer` alone, or both `pointer_nonstreaming` and `pointer_streaming`".to_string(),
            }));
        }
    };

    let cost_per_unit = match (entry.rate.cost_per_million, entry.rate.cost_per_unit) {
        (Some(_), Some(_)) => {
            return Err(Error::new(ErrorDetails::Config {
                message: "cannot specify both `cost_per_million` and `cost_per_unit`".to_string(),
            }));
        }
        (None, None) => {
            return Err(Error::new(ErrorDetails::Config {
                message: "must specify either `cost_per_million` or `cost_per_unit`".to_string(),
            }));
        }
        (Some(cost_per_million), None) => cost_per_million / Decimal::from(1_000_000),
        (None, Some(cost_per_unit)) => cost_per_unit,
    };

    Ok(CostConfigEntry {
        pointer,
        rate: CostRate { cost_per_unit },
        required: entry.required,
    })
}

#[cfg(test)]
mod tests {
    use serde::Deserialize;
    use tensorzero_types::{CostPointerConfig, UninitializedCostRate};

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
            msg.contains("must specify either"),
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
            msg.contains("must specify either `pointer`"),
            "error should mention missing pointer: {msg}"
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
            msg.contains("cannot specify both"),
            "error should mention both rates: {msg}"
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
}
