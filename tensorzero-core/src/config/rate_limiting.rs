use serde::{Deserialize, Deserializer};
use std::sync::Arc;
use tensorzero_auth::key::PUBLIC_ID_LENGTH;

use crate::rate_limiting::{
    ApiKeyPublicIdConfigScope, ApiKeyPublicIdValueScope, RateLimit, RateLimitInterval,
    RateLimitResource, RateLimitingConfigPriority, RateLimitingConfigRule, RateLimitingConfigScope,
    RateLimitingConfigScopes, TagRateLimitingConfigScope, TagValueScope,
};

/*
This file deserializes rate limiting configuration from a shorthand format only
[[rate_limiting.rules]]
RESOURCE_per_INTERVAL_1 = 10
RESOURCE_per_INTERVAL_2 = { capacity =  20, refill_rate = 10 }

but the same config serializes to a more extensive format for programmatic use

[[rate_limiting.rules]]
limits = [
    {
        resource: "model_inference",
        interval: "second",
        amount: 10,
    },
    {
        resource: "token",
        interval: "minute",
        amount: 20,
    },
]
*/

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BucketConfig {
    capacity: u64,
    refill_rate: u64,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum CapacityHelper {
    Bucket(BucketConfig),
    Amount(u64),
}

impl<'de> Deserialize<'de> for RateLimitingConfigRule {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // First deserialize to a TOML table
        let mut table = toml::map::Map::deserialize(deserializer)?;

        // Extract and parse all rate limit fields
        let mut limits = Vec::new();
        let mut rate_limit_keys = Vec::new();

        for (key, value) in &table {
            if key.contains("_per_") {
                let parts: Vec<&str> = key.splitn(2, "_per_").collect();

                if parts.len() != 2 {
                    return Err(serde::de::Error::custom(format!(
                        "Invalid rate limit key format: {key}. Expected format like 'tokens_per_minute'"
                    )));
                }

                let value =
                    CapacityHelper::deserialize(value.clone()).map_err(|_| {
                        serde::de::Error::custom("Rate limit value must either be a nonnegative integer or { capacity = .., refill_rate = .. } where both are nonnegative integers")
                    })?;

                let (capacity, refill_rate) = match value {
                    CapacityHelper::Amount(amount) => (amount, amount),
                    CapacityHelper::Bucket(bucket) => (bucket.capacity, bucket.refill_rate),
                };

                let resource = parse_resource(parts[0]).map_err(serde::de::Error::custom)?;

                let interval = RateLimitInterval::deserialize(
                    serde::de::value::StrDeserializer::new(parts[1]),
                )?;

                limits.push(Arc::new(RateLimit {
                    resource,
                    interval,
                    // We internally represent as a token bucket algorithm but for now
                    // we only take configuration that assumes the bucket is the size of one period
                    capacity,
                    refill_rate,
                }));

                // Track keys to remove
                rate_limit_keys.push(key.clone());
            }
        }

        // Remove rate limit fields from the table
        for key in rate_limit_keys {
            table.remove(&key);
        }

        // Now deserialize the remaining fields into a helper struct
        #[derive(Deserialize)]
        struct RemainingFields {
            #[serde(default)]
            scope: Vec<RateLimitingConfigScope>,
            #[serde(flatten)]
            priority: RateLimitingConfigPriority,
        }

        // Convert the table back to a Value for deserialization
        let remaining_value = toml::Value::Table(table);
        let remaining = RemainingFields::deserialize(remaining_value).map_err(|e| {
            // This error will trigger if there are unexpected fields
            // or if required fields are missing
            serde::de::Error::custom(format!("Error parsing rate limit scope and priority: {e}",))
        })?;

        Ok(RateLimitingConfigRule {
            limits,
            scope: RateLimitingConfigScopes::new(remaining.scope)
                .map_err(serde::de::Error::custom)?,
            priority: remaining.priority,
        })
    }
}

fn parse_resource(resource_str: &str) -> Result<RateLimitResource, String> {
    match resource_str {
        "model_inferences" => Ok(RateLimitResource::ModelInference),
        "tokens" => Ok(RateLimitResource::Token),
        // "cents" => Ok(RateLimitResource::Cent),
        _ => Err(format!("Unknown resource: {resource_str}")),
    }
}

// Helper struct for deserialization
#[derive(Deserialize)]
struct PriorityHelper {
    always: Option<bool>,
    priority: Option<usize>,
}

impl<'de> Deserialize<'de> for RateLimitingConfigPriority {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let helper = PriorityHelper::deserialize(deserializer)?;

        match (helper.always, helper.priority) {
            (Some(true), None) => Ok(RateLimitingConfigPriority::Always),
            (Some(false), None) | (None, None) => Err(serde::de::Error::custom(
                "the `priority` field is required when `always` is not true",
            )),
            (None, Some(p)) => Ok(RateLimitingConfigPriority::Priority(p)),
            (Some(true), Some(_)) => Err(serde::de::Error::custom(
                "cannot specify both `always` and `priority` fields",
            )),
            (Some(false), Some(p)) => Ok(RateLimitingConfigPriority::Priority(p)),
        }
    }
}

impl<'de> Deserialize<'de> for TagValueScope {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        if s == "tensorzero::each" {
            Ok(TagValueScope::Each)
        } else if s == "tensorzero::total" {
            Ok(TagValueScope::Total)
        } else if s.starts_with("tensorzero::") {
            Err(serde::de::Error::custom(
                r#"Tag values in rate limiting scopes besides tensorzero::each and tensorzero::total may not start with "tensorzero::"."#,
            ))
        } else {
            Ok(TagValueScope::Concrete(s))
        }
    }
}

impl<'de> Deserialize<'de> for ApiKeyPublicIdValueScope {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        if s == "tensorzero::each" {
            Ok(ApiKeyPublicIdValueScope::Each)
        } else if s.starts_with("tensorzero::") {
            Err(serde::de::Error::custom(
                r#"Api key public ID values in rate limiting scopes besides tensorzero::each may not start with "tensorzero::"."#,
            ))
        } else if s.len() != PUBLIC_ID_LENGTH {
            Err(serde::de::Error::custom(format!(
                "API key public ID `{s}` must be {PUBLIC_ID_LENGTH} characters long. Check that this is a TensorZero API key public ID."
            )))
        } else {
            Ok(ApiKeyPublicIdValueScope::Concrete(s))
        }
    }
}

// We use a custom deserializer for RateLimitingConfigScope
// so that we can handle the error messages gracefully for TagValueScope
impl<'de> Deserialize<'de> for RateLimitingConfigScope {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // First, deserialize into a generic Value to inspect it
        let value = toml::Value::deserialize(deserializer)?;

        if let Some(table) = value.as_table() {
            if table.contains_key("tag_key") {
                // If it looks like a Tag variant, try to deserialize it as such.
                // If this fails, the specific error will be propagated.
                return TagRateLimitingConfigScope::deserialize(value)
                    .map(RateLimitingConfigScope::Tag)
                    .map_err(serde::de::Error::custom);
            }
            if table.contains_key("api_key_public_id") {
                return ApiKeyPublicIdConfigScope::deserialize(value)
                    .map(RateLimitingConfigScope::ApiKeyPublicId)
                    .map_err(serde::de::Error::custom);
            }
            // As we add other variants, we will add impls here
            // if table.contains_key("model_name") { ... }
        }

        // If no variant matches, return a clear error
        Err(serde::de::Error::custom(
            "data did not match any known scope structure",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rate_limiting::{RateLimitingConfig, UninitializedRateLimitingConfig};
    use toml;

    #[test]
    fn test_basic_rate_limit_deserialization() {
        let toml_str = r"
            [[rules]]
            model_inferences_per_second = 10
            tokens_per_minute = 100
            always = true
        ";

        let uninitialized_config: UninitializedRateLimitingConfig =
            toml::from_str(toml_str).unwrap();
        let config: RateLimitingConfig = uninitialized_config.try_into().unwrap();
        assert_eq!(config.rules().len(), 1);

        let rule = &config.rules()[0];
        assert_eq!(rule.limits.len(), 2);

        assert_eq!(rule.priority, RateLimitingConfigPriority::Always);

        // Check model_inferences_per_second limit
        let inference_limit = rule
            .limits
            .iter()
            .find(|l| {
                matches!(l.resource, RateLimitResource::ModelInference)
                    && matches!(l.interval, RateLimitInterval::Second)
            })
            .unwrap();
        assert_eq!(inference_limit.capacity, 10);

        // Check tokens_per_minute limit
        let token_limit = rule
            .limits
            .iter()
            .find(|l| {
                matches!(l.resource, RateLimitResource::Token)
                    && matches!(l.interval, RateLimitInterval::Minute)
            })
            .unwrap();
        assert_eq!(token_limit.capacity, 100);
    }

    #[test]
    fn test_all_resources_and_intervals() {
        let toml_str = r"
            [[rules]]
            model_inferences_per_second = 1
            model_inferences_per_minute = 2
            model_inferences_per_hour = 3
            model_inferences_per_day = 4
            model_inferences_per_week = 5
            model_inferences_per_month = 6
            tokens_per_second = 10
            tokens_per_minute = 20
            priority = 0
        ";

        let uninitialized_config: UninitializedRateLimitingConfig =
            toml::from_str(toml_str).unwrap();
        let config: RateLimitingConfig = uninitialized_config.try_into().unwrap();
        assert_eq!(config.rules().len(), 1);
        assert_eq!(config.rules()[0].limits.len(), 8);
        assert_eq!(
            config.rules()[0].priority,
            RateLimitingConfigPriority::Priority(0),
        );
    }

    #[test]
    fn test_priority_configuration() {
        let toml_str = r#"
            [[rules]]
            model_inferences_per_second = 10
            priority = 5
            scope = [
                { tag_key = "user_id", tag_value = "123" }
            ]

            [[rules]]
            tokens_per_minute = 100
            priority = 0
            scope = [
                { tag_key = "app_id", tag_value = "456" }
            ]
        "#;

        let uninitialized_config: UninitializedRateLimitingConfig =
            toml::from_str(toml_str).unwrap();
        let config: RateLimitingConfig = uninitialized_config.try_into().unwrap();
        assert_eq!(config.rules().len(), 2);

        // First rule with explicit priority
        match &config.rules()[0].priority {
            RateLimitingConfigPriority::Priority(p) => assert_eq!(*p, 5),
            RateLimitingConfigPriority::Always => panic!("Expected priority value"),
        }

        // Second rule with default priority (0)
        match &config.rules()[1].priority {
            RateLimitingConfigPriority::Priority(p) => assert_eq!(*p, 0),
            RateLimitingConfigPriority::Always => panic!("Expected default priority value"),
        }
    }

    #[test]
    fn test_always_priority_configuration() {
        let toml_str = r"
            [[rules]]
            model_inferences_per_second = 10
            always = true
        ";

        let uninitialized_config: UninitializedRateLimitingConfig =
            toml::from_str(toml_str).unwrap();
        let config: RateLimitingConfig = uninitialized_config.try_into().unwrap();
        assert_eq!(config.rules().len(), 1);

        match &config.rules()[0].priority {
            RateLimitingConfigPriority::Always => {}
            RateLimitingConfigPriority::Priority(_) => panic!("Expected always priority"),
        }
    }

    #[test]
    fn test_scope_tag_configuration_all() {
        let toml_str = r#"
            [[rules]]
            model_inferences_per_second = 10
            priority = 1
            scope = [
                { tag_key = "user_id", tag_value = "tensorzero::total" }
            ]
        "#;

        let uninitialized_config: UninitializedRateLimitingConfig =
            toml::from_str(toml_str).unwrap();
        let config: RateLimitingConfig = uninitialized_config.try_into().unwrap();
        assert_eq!(config.rules().len(), 1);
        assert_eq!(config.rules()[0].scope.len(), 1);

        // Check scope filter with All value
        let RateLimitingConfigScope::Tag(tag_scope) = &config.rules()[0].scope[0] else {
            panic!("Expected Tag scope");
        };
        assert_eq!(tag_scope.tag_key(), "user_id");
        assert_eq!(tag_scope.tag_value(), &TagValueScope::Total);
    }

    #[test]
    fn test_scope_tag_configuration() {
        let toml_str = r#"
            [[rules]]
            model_inferences_per_second = 10
            priority = 1
            scope = [
                { tag_key = "user_id", tag_value = "123" },
                { tag_key = "application_id", tag_value = "tensorzero::each" }
            ]
        "#;

        let uninitialized_config: UninitializedRateLimitingConfig =
            toml::from_str(toml_str).unwrap();
        let config: RateLimitingConfig = uninitialized_config.try_into().unwrap();

        let check_config = |config: RateLimitingConfig| {
            assert_eq!(config.rules().len(), 1);
            assert_eq!(config.rules()[0].scope.len(), 2);

            // Check first scope filter
            let RateLimitingConfigScope::Tag(tag_scope) = &config.rules()[0].scope[0] else {
                panic!("Expected Tag scope");
            };
            assert_eq!(tag_scope.tag_key(), "application_id");
            assert_eq!(tag_scope.tag_value(), &TagValueScope::Each);

            // Check second scope filter with special value
            let RateLimitingConfigScope::Tag(tag_scope) = &config.rules()[0].scope[1] else {
                panic!("Expected Tag scope");
            };
            assert_eq!(tag_scope.tag_key(), "user_id");
            assert_eq!(
                tag_scope.tag_value(),
                &TagValueScope::Concrete("123".to_string())
            );
        };

        check_config(config);

        // Check again with opposite ordering of scopes to ensure stable sorting
        let toml_str = r#"
            [[rules]]
            model_inferences_per_second = 10
            priority = 1
            scope = [
                { tag_key = "application_id", tag_value = "tensorzero::each" },
                { tag_key = "user_id", tag_value = "123" }
            ]
        "#;

        let uninitialized_config: UninitializedRateLimitingConfig =
            toml::from_str(toml_str).unwrap();
        let config: RateLimitingConfig = uninitialized_config.try_into().unwrap();
        check_config(config);
    }

    #[test]
    fn test_complex_example_from_spec() {
        let toml_str = r#"
            # Global fallback
            [[rules]]
            model_inferences_per_second = 10
            tokens_per_minute = 1000
            always = true

            # Application 1, in aggregate
            [[rules]]
            model_inferences_per_hour = 10000
            priority = 1
            scope = [
                { tag_key = "application_id", tag_value = "1" }
            ]

            # Users, individually
            [[rules]]
            model_inferences_per_day = 100
            priority = 1
            scope = [
                { tag_key = "user_id", tag_value = "tensorzero::each" }
            ]

            # Application 2, for each user
            [[rules]]
            tokens_per_hour = 5000
            priority = 2
            scope = [
                { tag_key = "application_id", tag_value = "2" },
                { tag_key = "user_id", tag_value = "tensorzero::each" }
            ]
        "#;

        let uninitialized_config: UninitializedRateLimitingConfig =
            toml::from_str(toml_str).unwrap();
        let config: RateLimitingConfig = uninitialized_config.try_into().unwrap();
        assert_eq!(config.rules().len(), 4);

        // Global fallback (always = true, no scope)
        let global_rule = &config.rules()[0];
        match &global_rule.priority {
            RateLimitingConfigPriority::Always => {}
            RateLimitingConfigPriority::Priority(_) => panic!("Expected always priority"),
        }
        assert_eq!(global_rule.scope.len(), 0);
        assert_eq!(global_rule.limits.len(), 2);

        // Check global rule limits
        let global_inference_limit = global_rule
            .limits
            .iter()
            .find(|l| {
                matches!(l.resource, RateLimitResource::ModelInference)
                    && matches!(l.interval, RateLimitInterval::Second)
            })
            .expect("Expected model_inferences_per_second limit in global rule");
        assert_eq!(global_inference_limit.capacity, 10);

        let global_token_limit = global_rule
            .limits
            .iter()
            .find(|l| {
                matches!(l.resource, RateLimitResource::Token)
                    && matches!(l.interval, RateLimitInterval::Minute)
            })
            .expect("Expected tokens_per_minute limit in global rule");
        assert_eq!(global_token_limit.capacity, 1000);

        // Application 1 rule (priority 1)
        let app1_rule = &config.rules()[1];
        match &app1_rule.priority {
            RateLimitingConfigPriority::Priority(p) => assert_eq!(*p, 1),
            RateLimitingConfigPriority::Always => panic!("Expected priority 1"),
        }
        assert_eq!(app1_rule.scope.len(), 1);
        assert_eq!(app1_rule.limits.len(), 1);

        // Check Application 1 rule limits
        let app1_inference_limit = app1_rule
            .limits
            .iter()
            .find(|l| {
                matches!(l.resource, RateLimitResource::ModelInference)
                    && matches!(l.interval, RateLimitInterval::Hour)
            })
            .expect("Expected model_inferences_per_hour limit in app1 rule");
        assert_eq!(app1_inference_limit.capacity, 10000);

        // Users rule (priority 1)
        let users_rule = &config.rules()[2];
        match &users_rule.priority {
            RateLimitingConfigPriority::Priority(p) => assert_eq!(*p, 1),
            RateLimitingConfigPriority::Always => panic!("Expected priority 1"),
        }
        assert_eq!(users_rule.scope.len(), 1);
        assert_eq!(users_rule.limits.len(), 1);

        // Check Users rule limits
        let users_inference_limit = users_rule
            .limits
            .iter()
            .find(|l| {
                matches!(l.resource, RateLimitResource::ModelInference)
                    && matches!(l.interval, RateLimitInterval::Day)
            })
            .expect("Expected model_inferences_per_day limit in users rule");
        assert_eq!(users_inference_limit.capacity, 100);

        // Application 2 rule (priority 2, multiple scope filters)
        let app2_rule = &config.rules()[3];
        match &app2_rule.priority {
            RateLimitingConfigPriority::Priority(p) => assert_eq!(*p, 2),
            RateLimitingConfigPriority::Always => panic!("Expected priority 2"),
        }
        assert_eq!(app2_rule.scope.len(), 2);
        assert_eq!(app2_rule.limits.len(), 1);

        // Check Application 2 rule limits
        let app2_token_limit = app2_rule
            .limits
            .iter()
            .find(|l| {
                matches!(l.resource, RateLimitResource::Token)
                    && matches!(l.interval, RateLimitInterval::Hour)
            })
            .expect("Expected tokens_per_hour limit in app2 rule");
        assert_eq!(app2_token_limit.capacity, 5000);

        // Check Application 1 scope details
        let RateLimitingConfigScope::Tag(app1_scope) = &config.rules()[1].scope[0] else {
            panic!("Expected Tag scope");
        };
        assert_eq!(app1_scope.tag_key(), "application_id");
        assert_eq!(
            app1_scope.tag_value(),
            &TagValueScope::Concrete("1".to_string())
        );

        // Check Users rule scope details
        let RateLimitingConfigScope::Tag(users_scope) = &config.rules()[2].scope[0] else {
            panic!("Expected Tag scope");
        };
        assert_eq!(users_scope.tag_key(), "user_id");
        assert_eq!(users_scope.tag_value(), &TagValueScope::Each);

        // Check Application 2 scope details (first scope: application_id = "2")
        let RateLimitingConfigScope::Tag(app2_scope_0) = &app2_rule.scope[0] else {
            panic!("Expected Tag scope");
        };
        assert_eq!(app2_scope_0.tag_key(), "application_id");
        assert_eq!(
            app2_scope_0.tag_value(),
            &TagValueScope::Concrete("2".to_string())
        );

        // Check Application 2 scope details (second scope: user_id = wildcard)
        let RateLimitingConfigScope::Tag(app2_scope_1) = &app2_rule.scope[1] else {
            panic!("Expected Tag scope");
        };
        assert_eq!(app2_scope_1.tag_key(), "user_id");
        assert_eq!(app2_scope_1.tag_value(), &TagValueScope::Each);
    }

    #[test]
    fn test_default_enabled_true() {
        let toml_str = r"
            [[rules]]
            model_inferences_per_second = 10
            priority = 10
        ";

        let uninitialized_config: UninitializedRateLimitingConfig =
            toml::from_str(toml_str).unwrap();
        let config: RateLimitingConfig = uninitialized_config.try_into().unwrap();
        assert!(config.enabled());

        assert_eq!(
            config.rules()[0].priority,
            RateLimitingConfigPriority::Priority(10)
        );
    }

    #[test]
    fn test_explicit_enabled_false() {
        let toml_str = r"
            enabled = false
            [[rules]]
            model_inferences_per_second = 10
            always = true
        ";

        let uninitialized_config: UninitializedRateLimitingConfig =
            toml::from_str(toml_str).unwrap();
        let config: RateLimitingConfig = uninitialized_config.try_into().unwrap();
        assert!(!config.enabled());

        assert_eq!(
            config.rules()[0].priority,
            RateLimitingConfigPriority::Always
        );
    }

    #[test]
    fn test_empty_rules_configuration() {
        let toml_str = r"
            enabled = true
        ";

        let uninitialized_config: UninitializedRateLimitingConfig =
            toml::from_str(toml_str).unwrap();
        let config: RateLimitingConfig = uninitialized_config.try_into().unwrap();
        assert_eq!(config.rules().len(), 0);
        assert!(config.enabled());
    }

    // Error case tests

    #[test]
    fn test_invalid_rate_limit_key_format() {
        let toml_str = r"
            [[rules]]
            invalid_key = 10
        ";

        let result: Result<UninitializedRateLimitingConfig, _> = toml::from_str(toml_str);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_rate_limit_key_missing_per() {
        let toml_str = r"
            [[rules]]
            tokens_minute = 10
        ";

        let result: Result<UninitializedRateLimitingConfig, _> = toml::from_str(toml_str);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_rate_limit_key_too_many_parts() {
        let toml_str = r"
            [[rules]]
            tokens_per_minute_per_hour = 10
            priority = 0
        ";

        let result: Result<UninitializedRateLimitingConfig, _> = toml::from_str(toml_str);
        assert!(result.is_err());

        if let Err(e) = result {
            let error_msg = e.to_string();
            // The error occurs because "minute_per_hour" is not a valid interval
            assert!(error_msg.contains("unknown variant `minute_per_hour`"));
        }
    }

    #[test]
    fn test_invalid_resource_type() {
        let toml_str = r"
            [[rules]]
            invalid_resource_per_second = 10
        ";

        let result: Result<UninitializedRateLimitingConfig, _> = toml::from_str(toml_str);
        assert!(result.is_err());

        if let Err(e) = result {
            let error_msg = e.to_string();
            assert!(error_msg.contains("Unknown resource: invalid_resource"));
        }
    }

    #[test]
    fn test_invalid_interval_type() {
        let toml_str = r"
            [[rules]]
            tokens_per_invalid_interval = 10
        ";

        let result: Result<UninitializedRateLimitingConfig, _> = toml::from_str(toml_str);
        assert!(result.is_err());
    }

    #[test]
    fn test_negative_rate_limit_value() {
        let toml_str = r"
            [[rules]]
            tokens_per_minute = -10
        ";

        let result: Result<UninitializedRateLimitingConfig, _> = toml::from_str(toml_str);
        assert!(result.is_err());
    }

    #[test]
    fn test_non_integer_rate_limit_value() {
        let toml_str = "
            [[rules]]
            tokens_per_minute = \"not_a_number\"
        ";

        let result: Result<UninitializedRateLimitingConfig, _> = toml::from_str(toml_str);
        assert!(result.is_err());

        if let Err(e) = result {
            let error_msg = e.to_string();
            assert!(error_msg.contains("must either be a nonnegative integer"),);
        }
    }

    #[test]
    fn test_float_rate_limit_value() {
        let toml_str = r"
            [[rules]]
            tokens_per_minute = 10.5
        ";

        let result: Result<UninitializedRateLimitingConfig, _> = toml::from_str(toml_str);
        assert!(result.is_err());

        if let Err(e) = result {
            let error_msg = e.to_string();
            assert!(error_msg.contains("must either be a nonnegative integer"));
        }
    }

    #[test]
    fn test_both_priority_and_always_specified() {
        let toml_str = r"
            [[rules]]
            tokens_per_minute = 10
            priority = 1
            always = true
        ";

        let result: Result<UninitializedRateLimitingConfig, _> = toml::from_str(toml_str);
        assert!(result.is_err());

        if let Err(e) = result {
            let error_msg = e.to_string();
            assert!(error_msg.contains("cannot specify both `always` and `priority` fields"));
        }
    }

    #[test]
    fn test_always_false_without_priority() {
        let toml_str = r"
            [[rules]]
            tokens_per_minute = 10
            always = false
        ";

        let result: Result<UninitializedRateLimitingConfig, _> = toml::from_str(toml_str);
        assert!(result.is_err());

        if let Err(e) = result {
            let error_msg = e.to_string();
            assert!(
                error_msg.contains("the `priority` field is required when `always` is not true")
            );
        }
    }

    #[test]
    fn test_always_false_with_priority() {
        let toml_str = r"
            [[rules]]
            tokens_per_minute = 10
            always = false
            priority = 1
        ";

        let uninitialized_config: UninitializedRateLimitingConfig =
            toml::from_str(toml_str).unwrap();
        let config: RateLimitingConfig = uninitialized_config.try_into().unwrap();
        assert_eq!(config.rules().len(), 1);

        match &config.rules()[0].priority {
            RateLimitingConfigPriority::Priority(p) => assert_eq!(*p, 1),
            RateLimitingConfigPriority::Always => panic!("Expected priority value"),
        }
    }

    #[test]
    fn test_invalid_scope_missing_tag_key() {
        let toml_str = "
            [[rules]]
            tokens_per_minute = 10
            scope = [
                { tag_value = \"123\" }
            ]
        ";

        let result: Result<UninitializedRateLimitingConfig, _> = toml::from_str(toml_str);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_scope_missing_tag_value() {
        let toml_str = "
            [[rules]]
            tokens_per_minute = 10
            scope = [
                { tag_key = \"user_id\" }
            ]
        ";

        let result: Result<UninitializedRateLimitingConfig, _> = toml::from_str(toml_str);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_api_key_public_id_value() {
        let toml_str = "
            [[rules]]
            tokens_per_minute = 10
            always = true
            scope = [
                { api_key_public_id = \"my_bad_public_id\" }
            ]
        ";

        let result: Result<UninitializedRateLimitingConfig, _> = toml::from_str(toml_str);
        let err = result.unwrap_err();
        assert!(
            err.to_string()
                .contains("API key public ID `my_bad_public_id` must be 12 characters long."),
            "Unexpected error message: {err}",
        );
    }

    #[test]
    fn test_empty_rule_no_limits() {
        let toml_str = r"
            [[rules]]
            priority = 1
        ";

        let uninitialized_config: UninitializedRateLimitingConfig =
            toml::from_str(toml_str).unwrap();
        let config: RateLimitingConfig = uninitialized_config.try_into().unwrap();
        assert_eq!(config.rules().len(), 1);
        assert_eq!(config.rules()[0].limits.len(), 0);
    }

    #[test]
    fn test_zero_rate_limit_value() {
        let toml_str = r"
            [[rules]]
            tokens_per_minute = 0
            priority = 0
        ";

        let uninitialized_config: UninitializedRateLimitingConfig =
            toml::from_str(toml_str).unwrap();
        let config: RateLimitingConfig = uninitialized_config.try_into().unwrap();
        assert_eq!(config.rules().len(), 1);
        assert_eq!(config.rules()[0].limits[0].capacity, 0);
    }

    #[test]
    fn test_large_rate_limit_value() {
        let toml_str = r"
            [[rules]]
            tokens_per_minute = 9223372036854775807
            priority = 0
        ";

        let uninitialized_config: UninitializedRateLimitingConfig =
            toml::from_str(toml_str).unwrap();
        let config: RateLimitingConfig = uninitialized_config.try_into().unwrap();
        assert_eq!(config.rules().len(), 1);
        assert_eq!(config.rules()[0].limits[0].capacity, 9223372036854775807);
    }

    #[test]
    fn test_spec_resource_name_mapping() {
        let toml_str = r"
            [[rules]]
            model_inferences_per_second = 1
            tokens_per_minute = 2
            # cents_per_hour = 3
            priority = 0
        ";

        let uninitialized_config: UninitializedRateLimitingConfig =
            toml::from_str(toml_str).unwrap();
        let config: RateLimitingConfig = uninitialized_config.try_into().unwrap();
        assert_eq!(config.rules().len(), 1);
        assert_eq!(config.rules()[0].limits.len(), 2);

        let resources: Vec<_> = config.rules()[0]
            .limits
            .iter()
            .map(|l| l.resource)
            .collect();
        assert!(resources.contains(&RateLimitResource::ModelInference));
        assert!(resources.contains(&RateLimitResource::Token));
        // assert!(resources.contains(&RateLimitResource::Cent));
    }

    #[test]
    fn test_case_sensitive_intervals() {
        // Test that intervals are case sensitive
        let toml_str = r"
            [[rules]]
            tokens_per_Second = 10
        ";

        let result: Result<UninitializedRateLimitingConfig, _> = toml::from_str(toml_str);
        assert!(result.is_err());
    }

    #[test]
    fn test_case_sensitive_resources() {
        // Test that resources are case sensitive
        let toml_str = r"
            [[rules]]
            Tokens_per_second = 10
        ";

        let result: Result<UninitializedRateLimitingConfig, _> = toml::from_str(toml_str);
        assert!(result.is_err());
    }

    #[test]
    fn test_bucket_format_simple() {
        let toml_str = r"
            [[rules]]
            tokens_per_minute = { capacity = 100, refill_rate = 50 }
            priority = 1
        ";

        let uninitialized_config: UninitializedRateLimitingConfig =
            toml::from_str(toml_str).unwrap();
        let config: RateLimitingConfig = uninitialized_config.try_into().unwrap();
        assert_eq!(config.rules().len(), 1);
        assert_eq!(config.rules()[0].limits.len(), 1);

        let limit = &config.rules()[0].limits[0];
        assert_eq!(limit.capacity, 100);
        assert_eq!(limit.refill_rate, 50);
        assert!(matches!(limit.resource, RateLimitResource::Token));
        assert!(matches!(limit.interval, RateLimitInterval::Minute));
    }

    #[test]
    fn test_bucket_format_mixed_with_simple() {
        let toml_str = r"
            [[rules]]
            tokens_per_minute = { capacity = 100, refill_rate = 50 }
            model_inferences_per_second = 10
            priority = 1
        ";

        let uninitialized_config: UninitializedRateLimitingConfig =
            toml::from_str(toml_str).unwrap();
        let config: RateLimitingConfig = uninitialized_config.try_into().unwrap();
        assert_eq!(config.rules().len(), 1);
        assert_eq!(config.rules()[0].limits.len(), 2);

        // Check bucket format limit
        let bucket_limit = config.rules()[0]
            .limits
            .iter()
            .find(|l| matches!(l.resource, RateLimitResource::Token))
            .unwrap();
        assert_eq!(bucket_limit.capacity, 100);
        assert_eq!(bucket_limit.refill_rate, 50);

        // Check simple format limit
        let simple_limit = config.rules()[0]
            .limits
            .iter()
            .find(|l| matches!(l.resource, RateLimitResource::ModelInference))
            .unwrap();
        assert_eq!(simple_limit.capacity, 10);
        assert_eq!(simple_limit.refill_rate, 10);
    }

    #[test]
    fn test_bucket_format_multiple_resources() {
        let toml_str = r"
            [[rules]]
            tokens_per_minute = { capacity = 100, refill_rate = 50 }
            model_inferences_per_hour = { capacity = 1000, refill_rate = 200 }
            priority = 1
        ";

        let uninitialized_config: UninitializedRateLimitingConfig =
            toml::from_str(toml_str).unwrap();
        let config: RateLimitingConfig = uninitialized_config.try_into().unwrap();
        assert_eq!(config.rules().len(), 1);
        assert_eq!(config.rules()[0].limits.len(), 2);

        let token_limit = config.rules()[0]
            .limits
            .iter()
            .find(|l| matches!(l.resource, RateLimitResource::Token))
            .unwrap();
        assert_eq!(token_limit.capacity, 100);
        assert_eq!(token_limit.refill_rate, 50);

        let inference_limit = config.rules()[0]
            .limits
            .iter()
            .find(|l| matches!(l.resource, RateLimitResource::ModelInference))
            .unwrap();
        assert_eq!(inference_limit.capacity, 1000);
        assert_eq!(inference_limit.refill_rate, 200);
    }

    #[test]
    fn test_bucket_format_invalid_missing_capacity() {
        let toml_str = r"
            [[rules]]
            tokens_per_minute = { refill_rate = 50 }
            priority = 1
        ";

        let result: Result<UninitializedRateLimitingConfig, _> = toml::from_str(toml_str);
        assert!(result.is_err());
    }

    #[test]
    fn test_bucket_format_invalid_missing_refill_rate() {
        let toml_str = r"
            [[rules]]
            tokens_per_minute = { capacity = 100 }
            priority = 1
        ";

        let result: Result<UninitializedRateLimitingConfig, _> = toml::from_str(toml_str);
        assert!(result.is_err());
    }

    #[test]
    fn test_bucket_format_invalid_extra_fields() {
        let toml_str = r"
            [[rules]]
            tokens_per_minute = { capacity = 100, refill_rate = 50, extra_field = 123 }
            priority = 1
        ";

        let result: Result<UninitializedRateLimitingConfig, _> = toml::from_str(toml_str);
        assert!(result.is_err());
    }

    #[test]
    fn test_duplicate_scope_validation() {
        let toml_str = r#"
            [[rules]]
            model_inferences_per_second = 10
            priority = 1
            scope = [
                { tag_key = "user_id", tag_value = "123" }
            ]

            [[rules]]
            tokens_per_minute = 100
            priority = 2
            scope = [
                { tag_key = "user_id", tag_value = "123" }
            ]
        "#;

        let uninitialized_config: UninitializedRateLimitingConfig =
            toml::from_str(toml_str).unwrap();
        let result: Result<RateLimitingConfig, _> = uninitialized_config.try_into();

        assert!(result.is_err());
        let error = result.unwrap_err();

        // Check that it's specifically a duplicate scope error
        match error.get_details() {
            crate::error::ErrorDetails::DuplicateRateLimitingConfigScope { scope } => {
                // Verify the scope contains the expected tag
                assert_eq!(scope.len(), 1);
                match &scope[0] {
                    crate::rate_limiting::RateLimitingConfigScope::Tag(tag) => {
                        assert_eq!(tag.tag_key(), "user_id");
                        assert_eq!(
                            tag.tag_value(),
                            &crate::rate_limiting::TagValueScope::Concrete("123".to_string())
                        );
                    }
                    crate::rate_limiting::RateLimitingConfigScope::ApiKeyPublicId(
                        _api_key_public_id,
                    ) => {
                        panic!("Expected RateLimitingConfigScope");
                    }
                }
            }
            _ => panic!(
                "Expected DuplicateRateLimitingConfigScope error, got: {:?}",
                error.get_details()
            ),
        }
    }

    #[test]
    fn test_tensorzero_protected() {
        let toml_str = r#"
            [[rules]]
            model_inferences_per_second = 10
            priority = 5
            scope = [
                { tag_key = "user_id", tag_value = "tensorzero::foo" }
            ]

        "#;

        let err_message = toml::from_str::<UninitializedRateLimitingConfig>(toml_str)
            .unwrap_err()
            .to_string();
        assert!(err_message.contains("Tag values in rate limiting scopes besides tensorzero::each and tensorzero::total may not start with \"tensorzero::\"."));
    }
}
