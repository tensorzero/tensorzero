use axum::{extract::Request, middleware::Next, response::Response};
use std::collections::HashMap;

use super::{FeatureFlagValue, FeatureFlags, flags, with_feature_flags};

const FEATURE_FLAGS_HEADER: &str = "x-tensorzero-feature-flags";

pub async fn feature_flags_middleware(mut request: Request, next: Next) -> Response {
    let flags = parse_feature_flags_from_request(&request);
    request.extensions_mut().insert(flags.clone());
    with_feature_flags(flags, next.run(request)).await
}

fn parse_feature_flags_from_request(request: &Request) -> FeatureFlags {
    let Some(header_value) = request.headers().get(FEATURE_FLAGS_HEADER) else {
        return FeatureFlags::default();
    };

    let Ok(header_str) = header_value.to_str() else {
        tracing::debug!("Invalid UTF-8 in feature flags header, using defaults");
        return FeatureFlags::default();
    };

    let overrides = parse_header_value(header_str);
    FeatureFlags::with_overrides(overrides)
}

fn parse_header_value(header: &str) -> HashMap<&'static str, FeatureFlagValue> {
    let mut overrides = HashMap::new();

    let known_flags: HashMap<&str, &flags::FlagDefinition> =
        flags::ALL_FLAGS.iter().map(|f| (f.name, *f)).collect();

    for pair in header.split(',') {
        let pair = pair.trim();
        if pair.is_empty() {
            continue;
        }

        let Some((key, value)) = pair.split_once(':') else {
            tracing::debug!("Skipping malformed feature flag pair: {}", pair);
            continue;
        };

        let key = key.trim();
        let value = value.trim();

        let Some(flag_def) = known_flags.get(key) else {
            tracing::debug!("Ignoring unknown feature flag: {}", key);
            continue;
        };

        match &flag_def.default {
            FeatureFlagValue::Bool(_) => match value.to_lowercase().as_str() {
                "true" | "1" | "yes" => {
                    overrides.insert(flag_def.name, FeatureFlagValue::Bool(true));
                }
                "false" | "0" | "no" => {
                    overrides.insert(flag_def.name, FeatureFlagValue::Bool(false));
                }
                _ => {
                    tracing::debug!("Invalid boolean value for feature flag {}: {}", key, value);
                }
            },
        }
    }

    overrides
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_empty_header() {
        let overrides = parse_header_value("");
        assert!(
            overrides.is_empty(),
            "Empty header should produce no overrides"
        );
    }

    #[test]
    fn test_parse_single_flag() {
        let overrides = parse_header_value("use_postgres:true");
        assert_eq!(
            overrides.get("use_postgres"),
            Some(&FeatureFlagValue::Bool(true)),
            "use_postgres should be true"
        );
    }

    #[test]
    fn test_parse_single_flag_false() {
        let overrides = parse_header_value("use_postgres:false");
        assert_eq!(
            overrides.get("use_postgres"),
            Some(&FeatureFlagValue::Bool(false)),
            "use_postgres should be false"
        );
    }

    #[test]
    fn test_parse_numeric_values() {
        let overrides_true = parse_header_value("use_postgres:1");
        assert_eq!(
            overrides_true.get("use_postgres"),
            Some(&FeatureFlagValue::Bool(true)),
            "1 should parse as true"
        );

        let overrides_false = parse_header_value("use_postgres:0");
        assert_eq!(
            overrides_false.get("use_postgres"),
            Some(&FeatureFlagValue::Bool(false)),
            "0 should parse as false"
        );
    }

    #[test]
    fn test_parse_yes_no_values() {
        let overrides_yes = parse_header_value("use_postgres:yes");
        assert_eq!(
            overrides_yes.get("use_postgres"),
            Some(&FeatureFlagValue::Bool(true)),
            "yes should parse as true"
        );

        let overrides_no = parse_header_value("use_postgres:no");
        assert_eq!(
            overrides_no.get("use_postgres"),
            Some(&FeatureFlagValue::Bool(false)),
            "no should parse as false"
        );
    }

    #[test]
    fn test_unknown_flag_ignored() {
        let overrides = parse_header_value("unknown_flag:true,use_postgres:false");
        assert!(
            !overrides.contains_key("unknown_flag"),
            "Unknown flags should be ignored"
        );
        assert_eq!(
            overrides.get("use_postgres"),
            Some(&FeatureFlagValue::Bool(false)),
            "Known flags should still be parsed"
        );
    }

    #[test]
    fn test_invalid_value_ignored() {
        let overrides = parse_header_value("use_postgres:maybe");
        assert!(
            !overrides.contains_key("use_postgres"),
            "Invalid values should be ignored"
        );
    }

    #[test]
    fn test_whitespace_handling() {
        let overrides = parse_header_value(" use_postgres : true ");
        assert_eq!(
            overrides.get("use_postgres"),
            Some(&FeatureFlagValue::Bool(true)),
            "Whitespace should be trimmed"
        );
    }

    #[test]
    fn test_case_insensitive_values() {
        let overrides_upper = parse_header_value("use_postgres:TRUE");
        assert_eq!(
            overrides_upper.get("use_postgres"),
            Some(&FeatureFlagValue::Bool(true)),
            "TRUE should parse as true"
        );

        let overrides_mixed = parse_header_value("use_postgres:False");
        assert_eq!(
            overrides_mixed.get("use_postgres"),
            Some(&FeatureFlagValue::Bool(false)),
            "False should parse as false"
        );
    }

    #[test]
    fn test_malformed_pair_ignored() {
        let overrides = parse_header_value("use_postgres:true,malformed,use_postgres:false");
        assert_eq!(
            overrides.get("use_postgres"),
            Some(&FeatureFlagValue::Bool(false)),
            "Last valid value should win"
        );
    }

    #[test]
    fn test_feature_flags_default() {
        let flags = FeatureFlags::default();
        assert!(
            !flags.is_enabled(&flags::USE_POSTGRES),
            "USE_POSTGRES should default to false"
        );
    }

    #[test]
    fn test_feature_flags_with_overrides() {
        let mut overrides = HashMap::new();
        overrides.insert("use_postgres", FeatureFlagValue::Bool(true));
        let flags = FeatureFlags::with_overrides(overrides);
        assert!(
            flags.is_enabled(&flags::USE_POSTGRES),
            "USE_POSTGRES should be true after override"
        );
    }
}
