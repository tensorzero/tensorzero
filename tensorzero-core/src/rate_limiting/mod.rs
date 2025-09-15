use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use serde::{Deserialize, Serialize, Serializer};

use crate::db::{
    ConsumeTicketsReciept, ConsumeTicketsRequest, RateLimitQueries, ReturnTicketsRequest,
};
use crate::error::{Error, ErrorDetails, IMPOSSIBLE_ERROR_MESSAGE};

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

// Utility struct to pass in at "check time"
// This should contain the information about the current request
// needed to determine if a rate limit is exceeded.
pub struct ScopeInfo<'a> {
    pub tags: &'a HashMap<String, String>,
}

impl RateLimitingConfig {
    pub async fn consume_tickets<'a>(
        &'a self,
        client: &impl RateLimitQueries,
        scope_info: &'a ScopeInfo<'a>,
        rate_limit_resource_requests: &RateLimitResourceUsage,
    ) -> Result<TicketBorrow, Error> {
        let limits = self.get_active_limits(scope_info);
        let ticket_requests: Result<Vec<ConsumeTicketsRequest>, Error> = limits
            .iter()
            .map(|limit| limit.get_consume_tickets_request(rate_limit_resource_requests))
            .collect();
        let ticket_requests = ticket_requests?;
        let results = client.consume_tickets(ticket_requests).await?;
        check_borrowed_rate_limits(&limits, &results)?;
        TicketBorrow::new(results, limits)
    }

    fn get_active_limits<'a>(&'a self, scope_info: &'a ScopeInfo) -> Vec<ActiveRateLimit> {
        if !self.enabled {
            return vec![];
        }
        let mut max_priority: usize = 0;

        // First pass: collect matching limits and find max priority
        let matching_limits: Vec<_> = self
            .rules
            .iter()
            .map(|rule| {
                rule.get_rate_limits_if_match_update_priority(scope_info, &mut max_priority)
            })
            .collect();

        // Second pass: filter and flatten based on priority
        self.rules
            .iter()
            .zip(matching_limits)
            .filter_map(|(rule, limits_opt)| match (&rule.priority, limits_opt) {
                (RateLimitingConfigPriority::Always, Some(limits)) => Some(limits),
                (RateLimitingConfigPriority::Priority(priority), Some(limits))
                    if *priority == max_priority =>
                {
                    Some(limits)
                }
                _ => None,
            })
            .flatten()
            .collect()
    }
}
/// Assumes these are the same length.
fn check_borrowed_rate_limits(
    limits: &[ActiveRateLimit],
    results: &[ConsumeTicketsReciept],
) -> Result<(), Error> {
    for (limit, result) in limits.iter().zip(results.iter()) {
        if !result.success {
            // TODO: improve the error information here
            return Err(Error::new(ErrorDetails::RateLimitExceeded {
                key: limit.get_key()?,
                tickets_remaining: result.tickets_remaining,
            }));
        }
    }
    Ok(())
}

#[derive(Debug)]
struct ActiveRateLimit {
    limit: Arc<RateLimit>,
    scope_key: Vec<RateLimitingScopeKey>,
}

impl ActiveRateLimit {
    pub fn get_consume_tickets_request(
        &self,
        requests: &RateLimitResourceUsage,
    ) -> Result<ConsumeTicketsRequest, Error> {
        let request_amount = requests.get_usage(self.limit.resource);
        self.get_consume_tickets_request_for_return(request_amount)
    }

    /// Use this one if the actual usage > the borrowed usage
    pub fn get_consume_tickets_request_for_return(
        &self,
        requested: u64,
    ) -> Result<ConsumeTicketsRequest, Error> {
        Ok(ConsumeTicketsRequest {
            key: self.get_key()?,
            capacity: self.limit.capacity,
            refill_amount: self.limit.refill_rate,
            refill_interval: self.limit.interval.to_duration(),
            requested,
        })
    }

    /// Use this one if the borrowed usage < the actual usage
    pub fn get_return_tickets_request(&self, returned: u64) -> Result<ReturnTicketsRequest, Error> {
        Ok(ReturnTicketsRequest {
            key: self.get_key()?,
            capacity: self.limit.capacity,
            refill_amount: self.limit.refill_rate,
            refill_interval: self.limit.interval.to_duration(),
            returned,
        })
    }
}

#[derive(Serialize)]
struct ActiveRateLimitKeyHelper<'a> {
    resource: RateLimitResource,
    scope_key: &'a [RateLimitingScopeKey],
}

#[derive(Debug, PartialEq, Clone, serde::Serialize)]
pub struct ActiveRateLimitKey(pub String);

impl ActiveRateLimitKey {
    pub fn new(key: String) -> Self {
        ActiveRateLimitKey(key)
    }

    pub fn into_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ActiveRateLimitKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl ActiveRateLimit {
    pub fn get_key(&self) -> Result<ActiveRateLimitKey, Error> {
        let key = ActiveRateLimitKeyHelper {
            resource: self.limit.resource,
            scope_key: &self.scope_key,
        };

        Ok(ActiveRateLimitKey(serde_json::to_string(&key)?))
    }
}

#[derive(Debug, Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct RateLimitingConfigRule {
    pub limits: Vec<Arc<RateLimit>>,
    pub scope: RateLimitingConfigScopes,
    pub priority: RateLimitingConfigPriority,
}

impl RateLimitingConfigRule {
    fn get_rate_limits_if_match_update_priority<'a>(
        &'a self,
        scope_info: &'a ScopeInfo<'a>,
        max_priority: &mut usize,
    ) -> Option<Vec<ActiveRateLimit>> {
        let key = self.scope.get_key_if_matches(scope_info)?;
        if let RateLimitingConfigPriority::Priority(priority) = self.priority {
            if priority > *max_priority {
                *max_priority = priority;
            }
        }
        Some(
            self.limits
                .iter()
                .map(|limit| ActiveRateLimit {
                    limit: limit.clone(),
                    scope_key: key.clone(),
                })
                .collect(),
        )
    }
}

#[derive(Debug, Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct RateLimit {
    pub resource: RateLimitResource,
    pub interval: RateLimitInterval,
    pub capacity: u64,
    pub refill_rate: u64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub enum RateLimitResource {
    ModelInference,
    Token,
    // Cent, // or something more granular?
}

pub struct RateLimitResourceUsage {
    pub model_inferences: u64,
    pub tokens: u64,
}

impl RateLimitResourceUsage {
    pub fn get_usage(&self, resource: RateLimitResource) -> u64 {
        match resource {
            RateLimitResource::ModelInference => self.model_inferences,
            RateLimitResource::Token => self.tokens,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub enum RateLimitInterval {
    Second,
    Minute,
    Hour,
    Day,
    Week,
    Month,
}

impl RateLimitInterval {
    pub fn to_duration(&self) -> chrono::Duration {
        match self {
            RateLimitInterval::Second => chrono::Duration::seconds(1),
            RateLimitInterval::Minute => chrono::Duration::minutes(1),
            RateLimitInterval::Hour => chrono::Duration::hours(1),
            RateLimitInterval::Day => chrono::Duration::days(1),
            RateLimitInterval::Week => chrono::Duration::weeks(1),
            RateLimitInterval::Month => chrono::Duration::days(30),
        }
    }
}

#[derive(Debug, Serialize, PartialEq)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub enum RateLimitingConfigPriority {
    Priority(usize),
    Always,
}

/// Wrapper type for rate limiting scopes.
/// Forces them to be sorted on construction
#[derive(Debug, Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct RateLimitingConfigScopes(Vec<RateLimitingConfigScope>);

impl RateLimitingConfigScopes {
    /// Creates a new instance of `RateLimitingConfigScopes`.
    /// Ensures that there are no duplicate scopes and sorts them.
    pub fn new(mut scopes: Vec<RateLimitingConfigScope>) -> Result<Self, &'static str> {
        // First, we check to make sure there are no duplicate scopes
        if scopes.len() != scopes.iter().collect::<HashSet<_>>().len() {
            return Err("duplicate scopes are not allowed in the same rule");
        }
        // We sort the scopes so we are guaranteed a
        // stable order when generating the key
        scopes.sort();
        Ok(RateLimitingConfigScopes(scopes))
    }

    /// Returns the key (as a Vec) if the scope matches the given info, or None if it does not.
    fn get_key_if_matches<'a>(&'a self, info: &'a ScopeInfo) -> Option<Vec<RateLimitingScopeKey>> {
        self.0
            .iter()
            .map(|scope| scope.get_key_if_matches(info))
            .collect::<Option<Vec<_>>>()
    }
}

// IMPORTANT: the types below are used to set up scopes and keys for rate limiting.
// We need the keys that have already been used in production to remain stable.
// So, we cannot change the sort order.
// As scope types are added, please append new ones at the end to maintain a stable sort order,
// and add a test each time that ensures the sort order is maintained as further changes are made.
//
// Note to reviewer:  what else could we do to ensure the sort order is maintained across future changes?
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(untagged)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub enum RateLimitingConfigScope {
    Tag(TagRateLimitingConfigScope),
    // model_name = "my_model"
    // function_name = "my_function"
}

impl RateLimitingConfigScope {
    fn get_key_if_matches<'a>(&'a self, info: &'a ScopeInfo) -> Option<RateLimitingScopeKey> {
        match self {
            RateLimitingConfigScope::Tag(tag) => tag.get_key_if_matches(info),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct TagRateLimitingConfigScope {
    tag_key: String,
    tag_value: TagValueScope,
}

impl TagRateLimitingConfigScope {
    fn get_key_if_matches<'a>(&'a self, info: &'a ScopeInfo) -> Option<RateLimitingScopeKey> {
        let value = info.tags.get(&self.tag_key)?;

        match self.tag_value {
            TagValueScope::Concrete(ref expected_value) => {
                if value == expected_value {
                    Some(RateLimitingScopeKey::TagConcrete {
                        key: self.tag_key.clone(),
                        value: value.clone(),
                    })
                } else {
                    None
                }
            }
            TagValueScope::Any => Some(RateLimitingScopeKey::TagAny {
                key: self.tag_key.clone(),
            }),
            TagValueScope::All => Some(RateLimitingScopeKey::TagEach {
                key: self.tag_key.clone(),
                value: value.clone(),
            }),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub enum TagValueScope {
    Concrete(String),
    Any,
    All,
}

impl Serialize for TagValueScope {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            TagValueScope::Concrete(s) => serializer.serialize_str(s),
            TagValueScope::Any => serializer.serialize_str("tensorzero::any"),
            TagValueScope::All => serializer.serialize_str("tensorzero::all"),
        }
    }
}

/// Type that lists the different ways a Scope + matching ScopeInfo can be
/// serialized into a key.
/// We need this struct to have stable serialization behavior because we want rate limits to be stable
/// across releases.
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type")]
pub enum RateLimitingScopeKey {
    TagAny { key: String },
    TagEach { key: String, value: String },
    TagConcrete { key: String, value: String },
}

// TODO: is there a way to enforce that this struct is consumed by return_tickets?
#[must_use]
#[derive(Debug)]
pub struct TicketBorrow {
    reciepts: Vec<ConsumeTicketsReciept>,
    active_limits: Vec<ActiveRateLimit>,
}

impl TicketBorrow {
    pub fn empty() -> Self {
        Self {
            reciepts: Vec::new(),
            active_limits: Vec::new(),
        }
    }

    fn new(
        reciepts: Vec<ConsumeTicketsReciept>,
        active_limits: Vec<ActiveRateLimit>,
    ) -> Result<Self, Error> {
        // Assert all vectors have the same length
        let reciepts_len = reciepts.len();
        let active_limits_len = active_limits.len();

        if reciepts_len != active_limits_len {
            return Err(Error::new(ErrorDetails::Inference {
            message: format!(
                "TicketBorrow has ragged arrays: reciepts.len()={reciepts_len}, active_limits.len()={active_limits_len}. {IMPOSSIBLE_ERROR_MESSAGE}",
            )
        }));
        }

        Ok(Self {
            reciepts,
            active_limits,
        })
    }

    fn iter(&self) -> impl Iterator<Item = (&ConsumeTicketsReciept, &ActiveRateLimit)> {
        self.reciepts.iter().zip(self.active_limits.iter())
    }

    pub async fn return_tickets(
        self,
        client: &impl RateLimitQueries,
        actual_usage: RateLimitResourceUsage,
    ) -> Result<(), Error> {
        let mut requests = Vec::new();
        let mut returns = Vec::new();
        for (reciept, active_limit) in self.iter() {
            let actual_usage_this_request = actual_usage.get_usage(active_limit.limit.resource);
            match actual_usage_this_request.cmp(&reciept.tickets_consumed) {
                std::cmp::Ordering::Greater => {
                    // Actual usage exceeds borrowed, add the difference to requests and log a warning
                    let difference = actual_usage_this_request - reciept.tickets_consumed;
                    requests.push(active_limit.get_consume_tickets_request_for_return(difference)?);
                }
                std::cmp::Ordering::Less => {
                    // Borrowed exceeds actual usage, add the difference to returns
                    let difference = reciept.tickets_consumed - actual_usage_this_request;
                    returns.push(active_limit.get_return_tickets_request(difference)?);
                }
                std::cmp::Ordering::Equal => (),
            };
        }

        let (consume_result, return_result) = tokio::join!(
            client.consume_tickets(requests),
            client.return_tickets(returns)
        );

        consume_result?;
        return_result?;

        Ok(())
    }
}

pub trait RateLimitedRequest {
    fn estimated_resource_usage(&self) -> Result<RateLimitResourceUsage, Error>;
}

pub trait RateLimitedInputContent {
    fn estimated_input_token_usage(&self) -> u64;
}

pub trait RateLimitedResponse {
    fn resource_usage(&self) -> RateLimitResourceUsage;
}

/// We can estimate as a rough upper bound that every 2 characters are 1 token.
/// This is true for the vast majority of LLMs used today but is not a hard bound.
/// We can revisit this estimate in the future.
pub fn get_estimated_tokens(text: &str) -> u64 {
    // Implement logic to estimate tokens based on text length or other factors
    (text.len() as u64) / 2
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_rate_limiting_config_scope_get_key_if_matches_tag_concrete_match() {
        let scope = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope {
            tag_key: "user_id".to_string(),
            tag_value: TagValueScope::Concrete("123".to_string()),
        });

        let mut tags = HashMap::new();
        tags.insert("user_id".to_string(), "123".to_string());

        let info = ScopeInfo { tags: &tags };

        let key = scope.get_key_if_matches(&info).unwrap();
        match key {
            RateLimitingScopeKey::TagConcrete { key, value } => {
                assert_eq!(key, "user_id");
                assert_eq!(value, "123");
            }
            _ => panic!("Expected TagConcrete variant"),
        }
    }

    #[test]
    fn test_rate_limiting_config_scope_get_key_if_matches_tag_concrete_no_match() {
        let scope = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope {
            tag_key: "user_id".to_string(),
            tag_value: TagValueScope::Concrete("123".to_string()),
        });

        let mut tags = HashMap::new();
        tags.insert("user_id".to_string(), "456".to_string());

        let info = ScopeInfo { tags: &tags };

        let key = scope.get_key_if_matches(&info);
        assert!(key.is_none());
    }

    #[test]
    fn test_rate_limiting_config_scope_get_key_if_matches_tag_any() {
        let scope = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope {
            tag_key: "user_id".to_string(),
            tag_value: TagValueScope::Any,
        });

        let mut tags = HashMap::new();
        tags.insert("user_id".to_string(), "any_value".to_string());

        let info = ScopeInfo { tags: &tags };

        let key = scope.get_key_if_matches(&info).unwrap();
        match key {
            RateLimitingScopeKey::TagAny { key } => {
                assert_eq!(key, "user_id");
            }
            _ => panic!("Expected TagAny variant"),
        }
    }

    #[test]
    fn test_rate_limiting_config_scope_get_key_if_matches_tag_all() {
        let scope = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope {
            tag_key: "user_id".to_string(),
            tag_value: TagValueScope::All,
        });

        let mut tags = HashMap::new();
        tags.insert("user_id".to_string(), "specific_value".to_string());

        let info = ScopeInfo { tags: &tags };

        let key = scope.get_key_if_matches(&info).unwrap();
        match key {
            RateLimitingScopeKey::TagEach { key, value } => {
                assert_eq!(key, "user_id");
                assert_eq!(value, "specific_value");
            }
            _ => panic!("Expected TagEach variant"),
        }
    }

    #[test]
    fn test_rate_limiting_config_scope_get_key_if_matches_missing_tag() {
        let scope = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope {
            tag_key: "user_id".to_string(),
            tag_value: TagValueScope::Any,
        });

        let tags = HashMap::new(); // Empty tags

        let info = ScopeInfo { tags: &tags };

        let key = scope.get_key_if_matches(&info);
        assert!(key.is_none());
    }

    #[test]
    fn test_rate_limiting_config_scopes_get_key_if_matches_empty() {
        let scopes = RateLimitingConfigScopes::new(vec![]).unwrap();

        let tags = HashMap::new();
        let info = ScopeInfo { tags: &tags };

        let keys = scopes.get_key_if_matches(&info).unwrap();
        assert_eq!(keys.len(), 0);
    }

    #[test]
    fn test_rate_limiting_config_scopes_get_key_if_matches_single_match() {
        let scope = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope {
            tag_key: "user_id".to_string(),
            tag_value: TagValueScope::Concrete("123".to_string()),
        });
        let scopes = RateLimitingConfigScopes::new(vec![scope]).unwrap();

        let mut tags = HashMap::new();
        tags.insert("user_id".to_string(), "123".to_string());

        let info = ScopeInfo { tags: &tags };

        let keys = scopes.get_key_if_matches(&info).unwrap();
        assert_eq!(keys.len(), 1);

        match &keys[0] {
            RateLimitingScopeKey::TagConcrete { key, value } => {
                assert_eq!(*key, "user_id");
                assert_eq!(*value, "123");
            }
            _ => panic!("Expected TagConcrete variant"),
        }
    }

    #[test]
    fn test_rate_limiting_config_scopes_get_key_if_matches_single_no_match() {
        let scope = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope {
            tag_key: "user_id".to_string(),
            tag_value: TagValueScope::Concrete("123".to_string()),
        });
        let scopes = RateLimitingConfigScopes::new(vec![scope]).unwrap();

        let mut tags = HashMap::new();
        tags.insert("user_id".to_string(), "456".to_string());

        let info = ScopeInfo { tags: &tags };

        let keys = scopes.get_key_if_matches(&info);
        assert!(keys.is_none());
    }

    #[test]
    fn test_rate_limiting_config_scopes_get_key_if_matches_multiple_all_match() {
        let scope1 = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope {
            tag_key: "application_id".to_string(),
            tag_value: TagValueScope::Concrete("app123".to_string()),
        });
        let scope2 = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope {
            tag_key: "user_id".to_string(),
            tag_value: TagValueScope::Any,
        });
        let scopes = RateLimitingConfigScopes::new(vec![scope1, scope2]).unwrap();

        let mut tags = HashMap::new();
        tags.insert("application_id".to_string(), "app123".to_string());
        tags.insert("user_id".to_string(), "user456".to_string());

        let info = ScopeInfo { tags: &tags };

        let keys = scopes.get_key_if_matches(&info).unwrap();
        assert_eq!(keys.len(), 2);

        // Check that keys are in the same stable order as the scopes
        match (&keys[0], &keys[1]) {
            (
                RateLimitingScopeKey::TagConcrete {
                    key: key1,
                    value: value1,
                },
                RateLimitingScopeKey::TagAny { key: key2 },
            ) => {
                assert_eq!(*key1, "application_id");
                assert_eq!(*value1, "app123");
                assert_eq!(*key2, "user_id");
            }
            _ => panic!("Unexpected key variants or order"),
        }
    }

    #[test]
    fn test_rate_limiting_config_scopes_get_key_if_matches_multiple_partial_match() {
        let scope1 = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope {
            tag_key: "application_id".to_string(),
            tag_value: TagValueScope::Concrete("app123".to_string()),
        });
        let scope2 = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope {
            tag_key: "user_id".to_string(),
            tag_value: TagValueScope::Concrete("user456".to_string()),
        });
        let scopes = RateLimitingConfigScopes::new(vec![scope1, scope2]).unwrap();

        let mut tags = HashMap::new();
        tags.insert("application_id".to_string(), "app123".to_string());
        tags.insert("user_id".to_string(), "user789".to_string()); // Different value

        let info = ScopeInfo { tags: &tags };

        // Should return None because not all scopes match
        let keys = scopes.get_key_if_matches(&info);
        assert!(keys.is_none());
    }

    #[test]
    fn test_rate_limiting_config_scopes_get_key_stability_across_different_scope_info() {
        // Test that the same scopes + different but equivalent ScopeInfo produce the same key structure
        let scope1 = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope {
            tag_key: "application_id".to_string(),
            tag_value: TagValueScope::Any,
        });
        let scope2 = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope {
            tag_key: "user_id".to_string(),
            tag_value: TagValueScope::All,
        });
        let scopes = RateLimitingConfigScopes::new(vec![scope1, scope2]).unwrap();

        // First ScopeInfo
        let mut tags1 = HashMap::new();
        tags1.insert("application_id".to_string(), "app123".to_string());
        tags1.insert("user_id".to_string(), "user456".to_string());

        let info1 = ScopeInfo { tags: &tags1 };

        // Second ScopeInfo with different tag values but same structure
        let mut tags2 = HashMap::new();
        tags2.insert("application_id".to_string(), "app789".to_string());
        tags2.insert("user_id".to_string(), "user101".to_string());

        let info2 = ScopeInfo { tags: &tags2 };

        let keys1 = scopes.get_key_if_matches(&info1).unwrap();
        let keys2 = scopes.get_key_if_matches(&info2).unwrap();

        // Keys should have the same structure but different values for TagEach
        assert_eq!(keys1.len(), keys2.len());
        assert_eq!(keys1.len(), 2);

        // First key should be TagAny (same for both)
        match (&keys1[0], &keys2[0]) {
            (
                RateLimitingScopeKey::TagAny { key: key1 },
                RateLimitingScopeKey::TagAny { key: key2 },
            ) => {
                assert_eq!(key1, key2);
                assert_eq!(*key1, "application_id");
            }
            _ => panic!("Expected TagAny variants"),
        }

        // Second key should be TagEach (different values)
        match (&keys1[1], &keys2[1]) {
            (
                RateLimitingScopeKey::TagEach {
                    key: key1,
                    value: value1,
                },
                RateLimitingScopeKey::TagEach {
                    key: key2,
                    value: value2,
                },
            ) => {
                assert_eq!(key1, key2);
                assert_eq!(*key1, "user_id");
                assert_eq!(*value1, "user456");
                assert_eq!(*value2, "user101");
            }
            _ => panic!("Expected TagEach variants"),
        }
    }

    #[test]
    fn test_rate_limiting_config_scopes_new_empty() {
        let scopes = RateLimitingConfigScopes::new(vec![]).unwrap();
        assert_eq!(scopes.len(), 0);
    }

    #[test]
    fn test_rate_limiting_config_scopes_new_duplicate_scopes_error() {
        let scope1 = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope {
            tag_key: "user_id".to_string(),
            tag_value: TagValueScope::Concrete("123".to_string()),
        });
        let scope2 = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope {
            tag_key: "user_id".to_string(),
            tag_value: TagValueScope::Concrete("123".to_string()),
        });

        let result = RateLimitingConfigScopes::new(vec![scope1, scope2]);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            "duplicate scopes are not allowed in the same rule"
        );
    }

    #[test]
    fn test_rate_limiting_config_scopes_new_sorting_stability() {
        // Test that scopes are sorted consistently regardless of input order
        let scope1 = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope {
            tag_key: "application_id".to_string(),
            tag_value: TagValueScope::Any,
        });
        let scope2 = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope {
            tag_key: "user_id".to_string(),
            tag_value: TagValueScope::Concrete("123".to_string()),
        });
        let scope3 = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope {
            tag_key: "organization_id".to_string(),
            tag_value: TagValueScope::All,
        });

        // Create scopes in different orders
        let scopes_order1 =
            RateLimitingConfigScopes::new(vec![scope1.clone(), scope2.clone(), scope3.clone()])
                .unwrap();
        let scopes_order2 =
            RateLimitingConfigScopes::new(vec![scope3.clone(), scope1.clone(), scope2.clone()])
                .unwrap();
        let scopes_order3 = RateLimitingConfigScopes::new(vec![scope2, scope3, scope1]).unwrap();

        // All should result in the same order after sorting
        assert_eq!(scopes_order1.0, scopes_order2.0);
        assert_eq!(scopes_order2.0, scopes_order3.0);

        // Verify the actual sorted order
        match (&scopes_order1[0], &scopes_order1[1], &scopes_order1[2]) {
            (
                RateLimitingConfigScope::Tag(tag1),
                RateLimitingConfigScope::Tag(tag2),
                RateLimitingConfigScope::Tag(tag3),
            ) => {
                assert_eq!(tag1.tag_key, "application_id");
                assert_eq!(tag1.tag_value, TagValueScope::Any);
                assert_eq!(tag2.tag_key, "organization_id");
                assert_eq!(tag2.tag_value, TagValueScope::All);
                assert_eq!(tag3.tag_key, "user_id");
                assert_eq!(tag3.tag_value, TagValueScope::Concrete("123".to_string()));
            }
        }
    }

    #[test]
    fn test_different_scopes_same_key_sorting_and_keys() {
        // Test that different TagValueScope variants with the same tag_key are sorted consistently
        // and produce different keys
        let scope_concrete = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope {
            tag_key: "user_id".to_string(),
            tag_value: TagValueScope::Concrete("123".to_string()),
        });
        let scope_any = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope {
            tag_key: "user_id".to_string(),
            tag_value: TagValueScope::Any,
        });
        let scope_all = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope {
            tag_key: "user_id".to_string(),
            tag_value: TagValueScope::All,
        });

        // Test sorting order - TagValueScope variants should sort: Concrete < Any < All
        let scopes_order1 = RateLimitingConfigScopes::new(vec![
            scope_concrete.clone(),
            scope_any.clone(),
            scope_all.clone(),
        ])
        .unwrap();
        let scopes_order2 = RateLimitingConfigScopes::new(vec![
            scope_any.clone(),
            scope_all.clone(),
            scope_concrete.clone(),
        ])
        .unwrap();

        // Both orders should result in the same sorted order
        assert_eq!(scopes_order1.0, scopes_order2.0);

        // Verify the actual sorted order - let's check what the actual order is
        match (&scopes_order1[0], &scopes_order1[1], &scopes_order1[2]) {
            (
                RateLimitingConfigScope::Tag(tag1),
                RateLimitingConfigScope::Tag(tag2),
                RateLimitingConfigScope::Tag(tag3),
            ) => {
                // All tags should have the same key
                assert_eq!(tag1.tag_key, "user_id");
                assert_eq!(tag2.tag_key, "user_id");
                assert_eq!(tag3.tag_key, "user_id");

                // Test the actual derived sort order for TagValueScope
                // Based on Rust's enum ordering: Concrete(String) < Any < All
                assert_eq!(tag1.tag_value, TagValueScope::Concrete("123".to_string()));
                assert_eq!(tag2.tag_value, TagValueScope::Any);
                assert_eq!(tag3.tag_value, TagValueScope::All);
            }
        }

        // Test that each scope produces different key types
        let mut tags = HashMap::new();
        tags.insert("user_id".to_string(), "123".to_string());

        let info = ScopeInfo { tags: &tags };

        // Test each scope individually to verify different key types
        let key_concrete = scope_concrete.get_key_if_matches(&info).unwrap();
        let key_any = scope_any.get_key_if_matches(&info).unwrap();
        let key_all = scope_all.get_key_if_matches(&info).unwrap();

        // Verify each produces the correct key variant
        match key_concrete {
            RateLimitingScopeKey::TagConcrete { key, value } => {
                assert_eq!(key, "user_id");
                assert_eq!(value, "123");
            }
            _ => panic!("Expected TagConcrete variant"),
        }

        match key_any {
            RateLimitingScopeKey::TagAny { key } => {
                assert_eq!(key, "user_id");
            }
            _ => panic!("Expected TagAny variant"),
        }

        match key_all {
            RateLimitingScopeKey::TagEach { key, value } => {
                assert_eq!(key, "user_id");
                assert_eq!(value, "123");
            }
            _ => panic!("Expected TagEach variant"),
        }

        // Test serialization shows they produce different JSON structures
        use serde_json;
        let json_concrete = serde_json::to_string(&key_concrete).unwrap();
        let json_any = serde_json::to_string(&key_any).unwrap();
        let json_all = serde_json::to_string(&key_all).unwrap();

        assert!(json_concrete.contains("\"type\":\"TagConcrete\""));
        assert!(json_any.contains("\"type\":\"TagAny\""));
        assert!(json_all.contains("\"type\":\"TagEach\""));

        // Ensure they're all different
        assert_ne!(json_concrete, json_any);
        assert_ne!(json_any, json_all);
        assert_ne!(json_concrete, json_all);
    }

    #[test]
    fn test_different_scopes_same_key_duplicate_detection() {
        // Test that having the same tag_key with different TagValueScope variants
        // are NOT considered duplicates (they are different scopes)
        let scope_concrete = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope {
            tag_key: "user_id".to_string(),
            tag_value: TagValueScope::Concrete("123".to_string()),
        });
        let scope_any = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope {
            tag_key: "user_id".to_string(),
            tag_value: TagValueScope::Any,
        });

        // These should be allowed together since they have different TagValueScope
        let result = RateLimitingConfigScopes::new(vec![scope_concrete, scope_any]);
        assert!(result.is_ok());

        let scopes = result.unwrap();
        assert_eq!(scopes.len(), 2);
    }

    use std::ops::Deref;

    impl Deref for RateLimitingConfigScopes {
        type Target = Vec<RateLimitingConfigScope>;

        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    impl RateLimitingConfigScopes {
        /// Creates a new instance of `RateLimitingConfigScopes`.
        /// Ensures that there are no duplicate scopes and sorts them.
        fn new(mut scopes: Vec<RateLimitingConfigScope>) -> Result<Self, &'static str> {
            // First, we check to make sure there are no duplicate scopes
            use std::collections::HashSet;
            if scopes.len() != scopes.iter().collect::<HashSet<_>>().len() {
                return Err("duplicate scopes are not allowed in the same rule");
            }
            // We sort the scopes so we are guaranteed a
            // stable order when generating the key
            scopes.sort();
            Ok(RateLimitingConfigScopes(scopes))
        }
    }
}
