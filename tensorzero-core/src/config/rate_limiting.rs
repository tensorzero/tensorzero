use serde::{Deserialize, Deserializer, Serialize, Serializer};
#[cfg(test)]
use std::collections::HashMap;

/// NOTE: this file deserializes rate limiting configuration from a shorthand format only
/// [[rate_limiting.rules]]
/// RESOURCE_per_INTERVAL_1 = 10
/// RESOURCE_per_INTERVAL_2 = 20
///
/// but serializes to a more extensive format for programmatic use
///
/// [[rate_limiting.rules]]
/// limits = [
///     {
///         resource: "model_inference",
///         interval: "second",
///         amount: 10,
///     },
///     {
///         resource: "token",
///         interval: "minute",
///         amount: 20,
///     },
/// ]

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct RateLimitingConfig {
    #[serde(default)]
    rules: Vec<RateLimitingConfigRule>,
    #[serde(default = "default_enabled")]
    enabled: bool, // TODO: default true, Postgres required if rules is nonempty.
}

fn default_enabled() -> bool {
    true
}

impl Default for RateLimitingConfig {
    fn default() -> Self {
        Self {
            rules: Vec::new(),
            enabled: true,
        }
    }
}
#[derive(Debug, Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
struct RateLimitingConfigRule {
    limits: Vec<RateLimit>,
    scope: Vec<RateLimitingConfigScope>,
    #[serde(flatten)]
    priority: RateLimitingConfigPriority,
}

#[derive(Debug, Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
struct RateLimit {
    resource: RateLimitResource,
    interval: RateLimitInterval,
    amount: usize,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
enum RateLimitResource {
    ModelInference,
    Token,
    Cent, // or something more granular?
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
enum RateLimitInterval {
    Second,
    Minute,
    Hour,
    Day,
    Week,
    Month,
} // implement a getter for a chrono::TimeDelta or something

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

                // Extract the amount as usize
                let amount = value.as_integer().ok_or_else(|| {
                    serde::de::Error::custom(format!(
                        "Rate limit value for '{key}' must be a positive integer",
                    ))
                })? as usize;

                let resource = parse_resource(parts[0]).map_err(serde::de::Error::custom)?;

                let interval = RateLimitInterval::deserialize(
                    serde::de::value::StrDeserializer::new(parts[1]),
                )?;

                limits.push(RateLimit {
                    resource,
                    interval,
                    amount,
                });

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
            scope: remaining.scope,
            priority: remaining.priority,
        })
    }
}

fn parse_resource(resource_str: &str) -> Result<RateLimitResource, String> {
    match resource_str {
        "model_inferences" => Ok(RateLimitResource::ModelInference),
        "tokens" => Ok(RateLimitResource::Token),
        "cents" => Ok(RateLimitResource::Cent),
        _ => Err(format!("Unknown resource: {resource_str}")),
    }
}

#[derive(Debug, Serialize, PartialEq)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
enum RateLimitingConfigPriority {
    Priority(usize),
    Always,
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
                "priority field is required when always is not true",
            )),
            (None, Some(p)) => Ok(RateLimitingConfigPriority::Priority(p)),
            (Some(true), Some(_)) => Err(serde::de::Error::custom(
                "cannot specify both 'always' and 'priority' fields",
            )),
            (Some(false), Some(p)) => Ok(RateLimitingConfigPriority::Priority(p)),
        }
    }
}

#[derive(Debug, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
enum RateLimitingConfigScope {
    Tag(TagRateLimitingConfigScope),
    // model_name = "my_model"
    // function_name = "my_function"
}

#[derive(Debug, Deserialize, Serialize, PartialEq)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
struct TagRateLimitingConfigScope {
    tag_key: String,
    tag_value: TagValueScope,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
enum TagValueScope {
    Concrete(String),
    Wildcard,
}

impl Serialize for TagValueScope {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            TagValueScope::Concrete(s) => serializer.serialize_str(s),
            TagValueScope::Wildcard => serializer.serialize_str("tensorzero::*"),
        }
    }
}

impl<'de> Deserialize<'de> for TagValueScope {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        if s == "tensorzero::*" {
            Ok(TagValueScope::Wildcard)
        } else {
            Ok(TagValueScope::Concrete(s))
        }
    }
}

impl RateLimitingConfigScope {
    #[cfg(test)]
    fn matches(&self, info: &ScopeInfo) -> bool {
        match self {
            RateLimitingConfigScope::Tag(tag) => {
                let Some(info_value) = info.tags.get(&tag.tag_key) else {
                    return false;
                };

                match tag.tag_value {
                    TagValueScope::Concrete(ref value) => info_value == value,
                    TagValueScope::Wildcard => true,
                }
            }
        }
    }
}

// Utility struct to pass in at "check time"
// This should contain the information about the current request
// needed to determine if a rate limit is exceeded.
#[cfg(test)]
#[expect(dead_code)]
struct ScopeInfo<'a> {
    function_name: &'a str,
    model_name: &'a str,
    tags: &'a HashMap<String, String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use toml;

    #[test]
    fn test_basic_rate_limit_deserialization() {
        let toml_str = r"
            [[rules]]
            model_inferences_per_second = 10
            tokens_per_minute = 100
            always = true
        ";

        let config: RateLimitingConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.rules.len(), 1);

        let rule = &config.rules[0];
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
        assert_eq!(inference_limit.amount, 10);

        // Check tokens_per_minute limit
        let token_limit = rule
            .limits
            .iter()
            .find(|l| {
                matches!(l.resource, RateLimitResource::Token)
                    && matches!(l.interval, RateLimitInterval::Minute)
            })
            .unwrap();
        assert_eq!(token_limit.amount, 100);
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
            cents_per_hour = 100
            priority = 0
        ";

        let config: RateLimitingConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.rules.len(), 1);
        assert_eq!(config.rules[0].limits.len(), 9);
        assert_eq!(
            config.rules[0].priority,
            RateLimitingConfigPriority::Priority(0),
        );
    }

    #[test]
    fn test_priority_configuration() {
        let toml_str = r"
            [[rules]]
            model_inferences_per_second = 10
            priority = 5

            [[rules]]
            tokens_per_minute = 100
            priority = 0
        ";

        let config: RateLimitingConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.rules.len(), 2);

        // First rule with explicit priority
        match &config.rules[0].priority {
            RateLimitingConfigPriority::Priority(p) => assert_eq!(*p, 5),
            RateLimitingConfigPriority::Always => panic!("Expected priority value"),
        }

        // Second rule with default priority (0)
        match &config.rules[1].priority {
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

        let config: RateLimitingConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.rules.len(), 1);

        match &config.rules[0].priority {
            RateLimitingConfigPriority::Always => {}
            RateLimitingConfigPriority::Priority(_) => panic!("Expected always priority"),
        }
    }

    #[test]
    fn test_scope_tag_configuration() {
        let toml_str = r#"
            [[rules]]
            model_inferences_per_second = 10
            priority = 1
            scope = [
                { tag_key = "user_id", tag_value = "123" },
                { tag_key = "application_id", tag_value = "tensorzero::*" }
            ]
        "#;

        let config: RateLimitingConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.rules.len(), 1);
        assert_eq!(config.rules[0].scope.len(), 2);

        // Check first scope filter
        let RateLimitingConfigScope::Tag(tag_scope) = &config.rules[0].scope[0];
        assert_eq!(tag_scope.tag_key, "user_id");
        assert_eq!(
            tag_scope.tag_value,
            TagValueScope::Concrete("123".to_string())
        );

        // Check second scope filter with special value
        let RateLimitingConfigScope::Tag(tag_scope) = &config.rules[0].scope[1];
        assert_eq!(tag_scope.tag_key, "application_id");
        assert_eq!(tag_scope.tag_value, TagValueScope::Wildcard);
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
                { tag_key = "user_id", tag_value = "tensorzero::*" }
            ]

            # Application 2, for each user
            [[rules]]
            tokens_per_hour = 5000
            priority = 2
            scope = [
                { tag_key = "application_id", tag_value = "2" },
                { tag_key = "user_id", tag_value = "tensorzero::*" }
            ]
        "#;

        let config: RateLimitingConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.rules.len(), 4);

        // Global fallback (always = true, no scope)
        let global_rule = &config.rules[0];
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
        assert_eq!(global_inference_limit.amount, 10);

        let global_token_limit = global_rule
            .limits
            .iter()
            .find(|l| {
                matches!(l.resource, RateLimitResource::Token)
                    && matches!(l.interval, RateLimitInterval::Minute)
            })
            .expect("Expected tokens_per_minute limit in global rule");
        assert_eq!(global_token_limit.amount, 1000);

        // Application 1 rule (priority 1)
        let app1_rule = &config.rules[1];
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
        assert_eq!(app1_inference_limit.amount, 10000);

        // Users rule (priority 1)
        let users_rule = &config.rules[2];
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
        assert_eq!(users_inference_limit.amount, 100);

        // Application 2 rule (priority 2, multiple scope filters)
        let app2_rule = &config.rules[3];
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
        assert_eq!(app2_token_limit.amount, 5000);

        // Check Application 1 scope details
        let RateLimitingConfigScope::Tag(app1_scope) = &config.rules[1].scope[0];
        assert_eq!(app1_scope.tag_key, "application_id");
        assert_eq!(
            app1_scope.tag_value,
            TagValueScope::Concrete("1".to_string())
        );

        // Check Users rule scope details
        let RateLimitingConfigScope::Tag(users_scope) = &config.rules[2].scope[0];
        assert_eq!(users_scope.tag_key, "user_id");
        assert_eq!(users_scope.tag_value, TagValueScope::Wildcard);

        // Check Application 2 scope details (first scope: application_id = "2")
        let RateLimitingConfigScope::Tag(app2_scope_0) = &app2_rule.scope[0];
        assert_eq!(app2_scope_0.tag_key, "application_id");
        assert_eq!(
            app2_scope_0.tag_value,
            TagValueScope::Concrete("2".to_string())
        );

        // Check Application 2 scope details (second scope: user_id = wildcard)
        let RateLimitingConfigScope::Tag(app2_scope_1) = &app2_rule.scope[1];
        assert_eq!(app2_scope_1.tag_key, "user_id");
        assert_eq!(app2_scope_1.tag_value, TagValueScope::Wildcard);
    }

    #[test]
    fn test_default_enabled_true() {
        let toml_str = r"
            [[rules]]
            model_inferences_per_second = 10
            priority = 10
        ";

        let config: RateLimitingConfig = toml::from_str(toml_str).unwrap();
        assert!(config.enabled);

        assert_eq!(
            config.rules[0].priority,
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

        let config: RateLimitingConfig = toml::from_str(toml_str).unwrap();
        assert!(!config.enabled);

        assert_eq!(config.rules[0].priority, RateLimitingConfigPriority::Always);
    }

    #[test]
    fn test_empty_rules_configuration() {
        let toml_str = r"
            enabled = true
        ";

        let config: RateLimitingConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.rules.len(), 0);
        assert!(config.enabled);
    }

    // Error case tests

    #[test]
    fn test_invalid_rate_limit_key_format() {
        let toml_str = r"
            [[rules]]
            invalid_key = 10
        ";

        let result: Result<RateLimitingConfig, _> = toml::from_str(toml_str);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_rate_limit_key_missing_per() {
        let toml_str = r"
            [[rules]]
            tokens_minute = 10
        ";

        let result: Result<RateLimitingConfig, _> = toml::from_str(toml_str);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_rate_limit_key_too_many_parts() {
        let toml_str = r"
            [[rules]]
            tokens_per_minute_per_hour = 10
            priority = 0
        ";

        let result: Result<RateLimitingConfig, _> = toml::from_str(toml_str);
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

        let result: Result<RateLimitingConfig, _> = toml::from_str(toml_str);
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

        let result: Result<RateLimitingConfig, _> = toml::from_str(toml_str);
        assert!(result.is_err());
    }

    #[test]
    fn test_negative_rate_limit_value() {
        let toml_str = r"
            [[rules]]
            tokens_per_minute = -10
        ";

        let result: Result<RateLimitingConfig, _> = toml::from_str(toml_str);
        assert!(result.is_err());
    }

    #[test]
    fn test_non_integer_rate_limit_value() {
        let toml_str = "
            [[rules]]
            tokens_per_minute = \"not_a_number\"
        ";

        let result: Result<RateLimitingConfig, _> = toml::from_str(toml_str);
        assert!(result.is_err());

        if let Err(e) = result {
            let error_msg = e.to_string();
            assert!(error_msg.contains("must be a positive integer"));
        }
    }

    #[test]
    fn test_float_rate_limit_value() {
        let toml_str = r"
            [[rules]]
            tokens_per_minute = 10.5
        ";

        let result: Result<RateLimitingConfig, _> = toml::from_str(toml_str);
        assert!(result.is_err());

        if let Err(e) = result {
            let error_msg = e.to_string();
            assert!(error_msg.contains("must be a positive integer"));
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

        let result: Result<RateLimitingConfig, _> = toml::from_str(toml_str);
        assert!(result.is_err());

        if let Err(e) = result {
            let error_msg = e.to_string();
            assert!(error_msg.contains("cannot specify both 'always' and 'priority' fields"));
        }
    }

    #[test]
    fn test_always_false_without_priority() {
        let toml_str = r"
            [[rules]]
            tokens_per_minute = 10
            always = false
        ";

        let result: Result<RateLimitingConfig, _> = toml::from_str(toml_str);
        assert!(result.is_err());

        if let Err(e) = result {
            let error_msg = e.to_string();
            assert!(error_msg.contains("priority field is required when always is not true"));
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

        let config: RateLimitingConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.rules.len(), 1);

        match &config.rules[0].priority {
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

        let result: Result<RateLimitingConfig, _> = toml::from_str(toml_str);
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

        let result: Result<RateLimitingConfig, _> = toml::from_str(toml_str);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_rule_no_limits() {
        let toml_str = r"
            [[rules]]
            priority = 1
        ";

        let config: RateLimitingConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.rules.len(), 1);
        assert_eq!(config.rules[0].limits.len(), 0);
    }

    #[test]
    fn test_zero_rate_limit_value() {
        let toml_str = r"
            [[rules]]
            tokens_per_minute = 0
            priority = 0
        ";

        let config: RateLimitingConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.rules.len(), 1);
        assert_eq!(config.rules[0].limits[0].amount, 0);
    }

    #[test]
    fn test_large_rate_limit_value() {
        let toml_str = r"
            [[rules]]
            tokens_per_minute = 9223372036854775807
            priority = 0
        ";

        let config: RateLimitingConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.rules.len(), 1);
        assert_eq!(config.rules[0].limits[0].amount, 9223372036854775807);
    }

    #[test]
    fn test_spec_resource_name_mapping() {
        let toml_str = r"
            [[rules]]
            model_inferences_per_second = 1
            tokens_per_minute = 2
            cents_per_hour = 3
            priority = 0
        ";

        let config: RateLimitingConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.rules.len(), 1);
        assert_eq!(config.rules[0].limits.len(), 3);

        let resources: Vec<_> = config.rules[0].limits.iter().map(|l| l.resource).collect();
        assert!(resources.contains(&RateLimitResource::ModelInference));
        assert!(resources.contains(&RateLimitResource::Token));
        assert!(resources.contains(&RateLimitResource::Cent));
    }

    #[test]
    fn test_case_sensitive_intervals() {
        // Test that intervals are case sensitive
        let toml_str = r"
            [[rules]]
            tokens_per_Second = 10
        ";

        let result: Result<RateLimitingConfig, _> = toml::from_str(toml_str);
        assert!(result.is_err());
    }

    #[test]
    fn test_case_sensitive_resources() {
        // Test that resources are case sensitive
        let toml_str = r"
            [[rules]]
            Tokens_per_second = 10
        ";

        let result: Result<RateLimitingConfig, _> = toml::from_str(toml_str);
        assert!(result.is_err());
    }

    #[test]
    fn test_scope_matches() {
        // Test case 1: Concrete tag value exact match
        let concrete_scope = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope {
            tag_key: "user_id".to_string(),
            tag_value: TagValueScope::Concrete("123".to_string()),
        });

        let mut tags = HashMap::new();
        tags.insert("user_id".to_string(), "123".to_string());
        let info = ScopeInfo {
            function_name: "test_function",
            model_name: "test_model",
            tags: &tags,
        };
        assert!(concrete_scope.matches(&info));

        // Test case 2: Concrete tag value no match
        let mut tags = HashMap::new();
        tags.insert("user_id".to_string(), "456".to_string());
        let info = ScopeInfo {
            function_name: "test_function",
            model_name: "test_model",
            tags: &tags,
        };
        assert!(!concrete_scope.matches(&info));

        // Test case 3: Concrete tag missing key
        let mut tags = HashMap::new();
        tags.insert("application_id".to_string(), "app1".to_string());
        let info = ScopeInfo {
            function_name: "test_function",
            model_name: "test_model",
            tags: &tags,
        };
        assert!(!concrete_scope.matches(&info));

        // Test case 4: Wildcard tag value with existing key
        let wildcard_scope = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope {
            tag_key: "user_id".to_string(),
            tag_value: TagValueScope::Wildcard,
        });

        let mut tags = HashMap::new();
        tags.insert("user_id".to_string(), "any_value".to_string());
        let info = ScopeInfo {
            function_name: "test_function",
            model_name: "test_model",
            tags: &tags,
        };
        assert!(wildcard_scope.matches(&info));

        // Test case 5: Wildcard tag value with different values
        let test_values = vec!["123", "456", "user_abc", ""];
        for value in test_values {
            let mut tags = HashMap::new();
            tags.insert("user_id".to_string(), value.to_string());
            let info = ScopeInfo {
                function_name: "test_function",
                model_name: "test_model",
                tags: &tags,
            };
            assert!(
                wildcard_scope.matches(&info),
                "Wildcard should match value: {value}"
            );
        }

        // Test case 6: Wildcard tag missing key
        let mut tags = HashMap::new();
        tags.insert("application_id".to_string(), "app1".to_string());
        let info = ScopeInfo {
            function_name: "test_function",
            model_name: "test_model",
            tags: &tags,
        };
        assert!(!wildcard_scope.matches(&info));

        // Test case 7: Empty tags
        let tags = HashMap::new();
        let info = ScopeInfo {
            function_name: "test_function",
            model_name: "test_model",
            tags: &tags,
        };
        assert!(!concrete_scope.matches(&info));

        // Test case 8: Case sensitive tag keys
        let mut tags = HashMap::new();
        tags.insert("User_Id".to_string(), "123".to_string()); // Different case
        let info = ScopeInfo {
            function_name: "test_function",
            model_name: "test_model",
            tags: &tags,
        };
        assert!(!concrete_scope.matches(&info));

        // Test case 9: Case sensitive tag values
        let case_sensitive_scope = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope {
            tag_key: "user_id".to_string(),
            tag_value: TagValueScope::Concrete("abc".to_string()),
        });

        let mut tags = HashMap::new();
        tags.insert("user_id".to_string(), "ABC".to_string()); // Different case
        let info = ScopeInfo {
            function_name: "test_function",
            model_name: "test_model",
            tags: &tags,
        };
        assert!(!case_sensitive_scope.matches(&info));
    }
}
