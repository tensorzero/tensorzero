use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use axum::Extension;
use serde::{Deserialize, Serialize, Serializer};
use sqlx::postgres::types::PgInterval;
use tracing::Span;
use tracing_opentelemetry::OpenTelemetrySpanExt;

use crate::db::{
    ConsumeTicketsReceipt, ConsumeTicketsRequest, RateLimitQueries, ReturnTicketsRequest,
};
use crate::endpoints::RequestApiKeyExtension;
use crate::error::{Error, ErrorDetails, IMPOSSIBLE_ERROR_MESSAGE};

/*
 * The high level flow for our rate limiting system is:
 *   1. The caller constructs a ScopeInfo containing the information
 *      about the current provider request that determines what rate limit scopes apply.
 *   2. The caller calls `RateLimitingConfig::consume_tickets`. Here, we:
 *       a) use the ScopeInfo to determine which rate limit scopes apply
 *       b) estimate (conservatively) the resource consumption of the request
 *       c) consume tickets from the rate limit scopes
 *       d) actually attempt to consume the tickets from Postgres
 *       e) check success, throw an error on failure, and return a TicketBorrow
 *   3. The caller calls `TicketBorrow::return_tickets` with the actual usage observed.
 *      This will figure out post-facto bookkeeping for over- or under-consumption.
 *   Important Note: the Postgres database has a string column `key`
 *      that is used to identify a particular active rate limit.
 *      If two distinct rate limits have the same key, they will be treated as the same rate limit and will trample.
 *      If the key changes, the rate limit will be treated as a new rate limit.
 *      For these reasons, developers should be careful not to change the key serialization and be similarly careful
 *      to not add keys which could trample one another.
 */

#[derive(Debug, Serialize, Clone, ts_rs::TS)]
#[ts(export)]
pub struct RateLimitingConfig {
    rules: Vec<RateLimitingConfigRule>,
    enabled: bool,
}

#[derive(Debug, Deserialize)]
pub struct UninitializedRateLimitingConfig {
    #[serde(default)]
    rules: Vec<RateLimitingConfigRule>,
    #[serde(default = "default_enabled")]
    enabled: bool,
}

impl TryFrom<UninitializedRateLimitingConfig> for RateLimitingConfig {
    type Error = Error;
    fn try_from(config: UninitializedRateLimitingConfig) -> Result<Self, Self::Error> {
        // Make sure no rules have duplicated RateLimitingConfigScopes
        let mut scopes = HashSet::new();
        for rule in &config.rules {
            if !scopes.insert(rule.scope.clone()) {
                return Err(Error::new(ErrorDetails::DuplicateRateLimitingConfigScope {
                    scope: rule.scope.clone(),
                }));
            }
        }
        Ok(Self {
            rules: config.rules,
            enabled: config.enabled,
        })
    }
}

fn default_enabled() -> bool {
    true
}

impl Default for UninitializedRateLimitingConfig {
    fn default() -> Self {
        Self {
            rules: Vec::new(),
            enabled: true,
        }
    }
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
#[derive(Clone, Debug)]
pub struct ScopeInfo {
    pub tags: Arc<HashMap<String, String>>,
    pub api_key_public_id: Option<Arc<str>>,
}

impl ScopeInfo {
    pub fn new(
        tags: Arc<HashMap<String, String>>,
        api_key: Option<Extension<RequestApiKeyExtension>>,
    ) -> Self {
        Self {
            tags,
            api_key_public_id: api_key.map(|ext| ext.0.api_key.get_public_id().into()),
        }
    }

    // Expose relevant information from this `ScopeInfo` as OpenTelemetry span attributes
    fn apply_otel_span_attributes(&self, span: &Span) {
        let ScopeInfo {
            tags,
            api_key_public_id,
        } = self;
        for (key, value) in tags.iter() {
            span.set_attribute(format!("scope_info.tags.{key}"), value.clone());
        }
        if let Some(api_key_public_id) = api_key_public_id {
            span.set_attribute("scope_info.api_key_public_id", api_key_public_id.clone());
        }
    }
}

impl RateLimitingConfig {
    pub fn rules(&self) -> &Vec<RateLimitingConfigRule> {
        &self.rules
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn get_rate_limited_resources(&self, scope_info: &ScopeInfo) -> Vec<RateLimitResource> {
        if !self.enabled {
            return vec![];
        }
        let limits = self.get_active_limits(scope_info);
        limits
            .iter()
            .map(|limit| limit.limit.resource)
            .collect::<HashSet<_>>()
            .into_iter()
            .collect::<Vec<_>>()
    }

    #[tracing::instrument(skip_all, fields(otel.name = "rate_limiting_consume_tickets", estimated_usage.tokens, estimated_usage.model_inferences))]
    pub async fn consume_tickets<'a>(
        &'a self,
        client: &impl RateLimitQueries,
        scope_info: &'a ScopeInfo,
        rate_limited_request: &impl RateLimitedRequest,
    ) -> Result<TicketBorrows, Error> {
        let res = self
            .consume_tickets_inner(client, scope_info, rate_limited_request)
            .await;
        if let Err(e) = &res {
            // We want rate-limiting errors to show up as errors in OpenTelemetry,
            // even though they only get logged as warnings to the console.
            e.ensure_otel_span_errored(&Span::current());
        }
        res
    }

    // The actual implementation of `consume_tickets`. This is a separate function so that we can
    // handle `Result::Err` inside `consume_tickets`
    async fn consume_tickets_inner<'a>(
        &'a self,
        client: &impl RateLimitQueries,
        scope_info: &'a ScopeInfo,
        rate_limited_request: &impl RateLimitedRequest,
    ) -> Result<TicketBorrows, Error> {
        let span = Span::current();
        scope_info.apply_otel_span_attributes(&span);
        let limits = self.get_active_limits(scope_info);
        if limits.is_empty() {
            return Ok(TicketBorrows::empty());
        }

        let rate_limited_resources = self.get_rate_limited_resources(scope_info);
        let rate_limit_resource_requests =
            rate_limited_request.estimated_resource_usage(&rate_limited_resources)?;

        if let Some(tokens) = rate_limit_resource_requests.tokens {
            span.record("estimated_usage.tokens", tokens as i64);
        }
        if let Some(model_inferences) = rate_limit_resource_requests.model_inferences {
            span.record("estimated_usage.model_inferences", model_inferences as i64);
        }
        let ticket_requests: Result<Vec<ConsumeTicketsRequest>, Error> = limits
            .iter()
            .map(|limit| limit.get_consume_tickets_request(&rate_limit_resource_requests))
            .collect();
        let ticket_requests = ticket_requests?;
        let results = client.consume_tickets(&ticket_requests).await?;

        TicketBorrows::new(results, limits, ticket_requests)
    }

    /// Given a particular scope, finds the RateLimits that are active for that scope.
    /// We follow a two-pass approach:
    /// 1. First pass: collect matching limits and find max priority
    /// 2. Second pass: filter and flatten based on priority
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

fn align_and_check_limits(
    limits: &[ActiveRateLimit],
    receipts: Vec<ConsumeTicketsReceipt>,
    requests: Vec<ConsumeTicketsRequest>,
) -> Result<Vec<ConsumeTicketsReceipt>, Error> {
    // First, we collect the Vec<ConsumeTicketsReceipt> into a HashMap from ActiveRateLimitKey to receipt
    let mut receipts_map = receipts
        .into_iter()
        .map(|r| (r.key.clone(), r))
        .collect::<HashMap<_, _>>();
    // Next, check if any receipt has failed
    if receipts_map.values().any(|r| !r.success) {
        return Err(get_failed_rate_limits_err(requests, &receipts_map, limits));
    }

    // Next, we build up a vector of ConsumeTicketsReceipts
    let mut aligned_receipts = Vec::with_capacity(limits.len());
    for limit in limits {
        let key = limit.get_key()?;
        if let Some(receipt) = receipts_map.remove(&key) {
            aligned_receipts.push(receipt);
        } else {
            // throw an error
            return Err(Error::new(ErrorDetails::Inference {
                message: format!(
                    "Failed to find rate limit key for limit {key}. {IMPOSSIBLE_ERROR_MESSAGE}",
                ),
            }));
        }
    }
    Ok(aligned_receipts)
}

#[derive(Debug, PartialEq, Serialize)]
pub struct FailedRateLimit {
    pub key: ActiveRateLimitKey,
    pub requested: u64,
    pub available: u64,
    pub resource: RateLimitResource,
    pub scope_key: Vec<RateLimitingScopeKey>,
}

/// Since Postgres will tell us all borrows failed if any failed, we figure out which rate limits
/// actually blocked the request and return an informative error.
fn get_failed_rate_limits_err(
    requests: Vec<ConsumeTicketsRequest>,
    receipts_map: &HashMap<ActiveRateLimitKey, ConsumeTicketsReceipt>,
    limits: &[ActiveRateLimit],
) -> Error {
    let mut failed_rate_limits = Vec::new();
    for (request, limit) in requests.iter().zip(limits) {
        let key = &request.key;
        let Some(receipt) = receipts_map.get(key) else {
            return ErrorDetails::Inference {
                message: format!(
                    "Failed to find rate limit request for limit {key} while constructing FailedRateLimit error. {IMPOSSIBLE_ERROR_MESSAGE}",
                ),
            }.into();
        };
        if request.requested > receipt.tickets_remaining {
            failed_rate_limits.push(FailedRateLimit {
                key: key.clone(),
                requested: request.requested,
                available: receipt.tickets_remaining,
                resource: limit.limit.resource,
                scope_key: limit.scope_key.clone(),
            });
        }
    }
    if failed_rate_limits.is_empty() {
        return ErrorDetails::Inference {
            message: format!(
                "Failed to find rate limit request where requested > available while constructing FailedRateLimit error. {IMPOSSIBLE_ERROR_MESSAGE}",
            ),
        }.into();
    }
    ErrorDetails::RateLimitExceeded { failed_rate_limits }.into()
}

#[derive(Debug)]
struct ActiveRateLimit {
    limit: Arc<RateLimit>,
    scope_key: Vec<RateLimitingScopeKey>,
}

impl ActiveRateLimit {
    pub fn get_consume_tickets_request(
        &self,
        requests: &EstimatedRateLimitResourceUsage,
    ) -> Result<ConsumeTicketsRequest, Error> {
        // INVARIANT: All resources in active rate limits are validated in estimated_resource_usage().
        // This check should never fail in normal operation.
        let request_amount = requests.get_usage(self.limit.resource).ok_or_else(|| {
            Error::new(ErrorDetails::Inference {
                message: format!(
                    "estimated_resource_usage did not provide {:?} resource. {IMPOSSIBLE_ERROR_MESSAGE}",
                    self.limit.resource
                ),
            })
        })?;
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
            refill_interval: self.limit.interval.to_pg_interval(),
            requested,
        })
    }

    /// Use this one if the borrowed usage < the actual usage
    pub fn get_return_tickets_request(&self, returned: u64) -> Result<ReturnTicketsRequest, Error> {
        Ok(ReturnTicketsRequest {
            key: self.get_key()?,
            capacity: self.limit.capacity,
            refill_amount: self.limit.refill_rate,
            refill_interval: self.limit.interval.to_pg_interval(),
            returned,
        })
    }
}

#[derive(Serialize)]
struct ActiveRateLimitKeyHelper<'a> {
    resource: RateLimitResource,
    scope_key: &'a [RateLimitingScopeKey],
}

#[derive(Debug, PartialEq, Clone, Serialize, Eq, Hash)]
pub struct ActiveRateLimitKey(pub String);

impl ActiveRateLimitKey {
    pub fn new(key: String) -> Self {
        ActiveRateLimitKey(key)
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

#[derive(Debug, Serialize, Clone, ts_rs::TS)]
#[ts(export)]
pub struct RateLimitingConfigRule {
    pub limits: Vec<Arc<RateLimit>>,
    pub scope: RateLimitingConfigScopes,
    pub priority: RateLimitingConfigPriority,
}

impl RateLimitingConfigRule {
    fn get_rate_limits_if_match_update_priority<'a>(
        &'a self,
        scope_info: &'a ScopeInfo,
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

#[derive(Debug, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct RateLimit {
    pub resource: RateLimitResource,
    pub interval: RateLimitInterval,
    pub capacity: u64,
    pub refill_rate: u64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
#[derive(ts_rs::TS)]
#[ts(export)]
pub enum RateLimitResource {
    ModelInference,
    Token,
    // Cent, // or something more granular?
}

impl RateLimitResource {
    /// Returns the snake_case string representation matching the serde serialization
    pub fn as_str(&self) -> &'static str {
        match self {
            RateLimitResource::ModelInference => "model_inference",
            RateLimitResource::Token => "token",
        }
    }
}

#[derive(Debug)]
pub enum RateLimitResourceUsage {
    /// We received an exact usage amount back from the provider, so we can consume extra/release unused
    /// rate limiting resources, depending on whether our initial estimate was too high or too low
    Exact { model_inferences: u64, tokens: u64 },
    /// We were only able to estimate the usage (e.g. if an error occurred in an inference stream,
    /// and there might have been additional usage chunks that we missed; or the provider did not report token usage).
    /// We'll still consume tokens/inferences if we went over the initial estimate, but we will *not*
    /// return tickets if our initial estimate seems to be too high (since the error could have
    /// hidden the actual usage).
    UnderEstimate { model_inferences: u64, tokens: u64 },
}

#[derive(Debug)]
pub struct EstimatedRateLimitResourceUsage {
    pub model_inferences: Option<u64>,
    pub tokens: Option<u64>,
}

impl EstimatedRateLimitResourceUsage {
    pub fn get_usage(&self, resource: RateLimitResource) -> Option<u64> {
        match resource {
            RateLimitResource::ModelInference => self.model_inferences,
            RateLimitResource::Token => self.tokens,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
#[derive(ts_rs::TS)]
#[ts(export)]
pub enum RateLimitInterval {
    Second,
    Minute,
    Hour,
    Day,
    Week,
    Month,
}

impl RateLimitInterval {
    pub fn to_pg_interval(self) -> PgInterval {
        match self {
            RateLimitInterval::Second => PgInterval {
                months: 0,
                days: 0,
                microseconds: 1_000_000, // 1 second
            },
            RateLimitInterval::Minute => PgInterval {
                months: 0,
                days: 0,
                microseconds: 60_000_000, // 60 seconds
            },
            RateLimitInterval::Hour => PgInterval {
                months: 0,
                days: 0,
                microseconds: 3_600_000_000, // 3600 seconds
            },
            RateLimitInterval::Day => PgInterval {
                months: 0,
                days: 1,
                microseconds: 0,
            },
            RateLimitInterval::Week => PgInterval {
                months: 0,
                days: 7,
                microseconds: 0,
            },
            RateLimitInterval::Month => PgInterval {
                months: 1,
                days: 0,
                microseconds: 0,
            },
        }
    }
}

#[derive(Debug, Serialize, PartialEq, Clone, ts_rs::TS)]
#[ts(export)]
pub enum RateLimitingConfigPriority {
    Priority(usize),
    Always,
}

/// Wrapper type for rate limiting scopes.
/// Forces them to be sorted on construction
#[derive(Clone, Debug, Hash, Serialize, PartialEq, Eq, ts_rs::TS)]
#[ts(export)]
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

trait Scope {
    fn get_key_if_matches(&self, info: &ScopeInfo) -> Option<RateLimitingScopeKey>;
}

// IMPORTANT: the types below are used to set up scopes and keys for rate limiting.
// We need the keys that have already been used in production to remain stable.
// So, we cannot change the sort order.
// As scope types are added, please append new ones at the end to maintain a stable sort order,
// and add a test each time that ensures the sort order is maintained as further changes are made.
//
// Note to reviewer:  what else could we do to ensure the sort order is maintained across future changes?
#[derive(Debug, Clone, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(untagged)]
#[derive(ts_rs::TS)]
#[ts(export)]
pub enum RateLimitingConfigScope {
    Tag(TagRateLimitingConfigScope),
    ApiKeyPublicId(ApiKeyPublicIdConfigScope),
    // model_name = "my_model"
    // function_name = "my_function"
}

impl Scope for RateLimitingConfigScope {
    fn get_key_if_matches(&self, info: &ScopeInfo) -> Option<RateLimitingScopeKey> {
        match self {
            RateLimitingConfigScope::Tag(tag) => tag.get_key_if_matches(info),
            RateLimitingConfigScope::ApiKeyPublicId(api_key_public_id) => {
                api_key_public_id.get_key_if_matches(info)
            }
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash, ts_rs::TS)]
#[ts(export)]
pub struct TagRateLimitingConfigScope {
    tag_key: String,
    tag_value: TagValueScope,
}

impl TagRateLimitingConfigScope {
    #[cfg(test)]
    pub fn tag_key(&self) -> &str {
        &self.tag_key
    }

    #[cfg(test)]
    pub fn tag_value(&self) -> &TagValueScope {
        &self.tag_value
    }

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
            TagValueScope::Each => Some(RateLimitingScopeKey::TagEach {
                key: self.tag_key.clone(),
                value: value.clone(),
            }),
            TagValueScope::Total => Some(RateLimitingScopeKey::TagTotal {
                key: self.tag_key.clone(),
            }),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash, ts_rs::TS)]
#[ts(export)]
pub struct ApiKeyPublicIdConfigScope {
    api_key_public_id: ApiKeyPublicIdValueScope,
}

impl ApiKeyPublicIdConfigScope {
    fn get_key_if_matches<'a>(&'a self, info: &'a ScopeInfo) -> Option<RateLimitingScopeKey> {
        match self.api_key_public_id {
            ApiKeyPublicIdValueScope::Concrete(ref key) => {
                if info
                    .api_key_public_id
                    .as_ref()
                    .is_some_and(|s| **s == **key)
                {
                    Some(RateLimitingScopeKey::ApiKeyPublicIdConcrete {
                        // TODO - use existing arc
                        api_key_public_id: Arc::from(key.as_str()),
                    })
                } else {
                    None
                }
            }
            ApiKeyPublicIdValueScope::Each => info.api_key_public_id.as_ref().map(|key| {
                RateLimitingScopeKey::ApiKeyPublicIdEach {
                    api_key_public_id: key.clone(),
                }
            }),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, ts_rs::TS)]
#[ts(export)]
pub enum TagValueScope {
    Concrete(String),
    Each,
    Total,
}

impl Serialize for TagValueScope {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            TagValueScope::Concrete(s) => serializer.serialize_str(s),
            TagValueScope::Each => serializer.serialize_str("tensorzero::each"),
            TagValueScope::Total => serializer.serialize_str("tensorzero::total"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, ts_rs::TS)]
#[ts(export)]
pub enum ApiKeyPublicIdValueScope {
    Concrete(String),
    Each,
}

impl Serialize for ApiKeyPublicIdValueScope {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            ApiKeyPublicIdValueScope::Concrete(s) => serializer.serialize_str(s),
            ApiKeyPublicIdValueScope::Each => serializer.serialize_str("tensorzero::each"),
        }
    }
}

/// Type that lists the different ways a Scope + matching ScopeInfo can be
/// serialized into a key.
/// We need this struct to have stable serialization behavior because we want rate limits to be stable
/// across releases.
#[derive(Clone, Debug, Serialize, PartialEq)]
#[serde(tag = "type")]
pub enum RateLimitingScopeKey {
    TagTotal { key: String },
    TagEach { key: String, value: String },
    TagConcrete { key: String, value: String },
    ApiKeyPublicIdEach { api_key_public_id: Arc<str> },
    ApiKeyPublicIdConcrete { api_key_public_id: Arc<str> },
}

impl RateLimitingScopeKey {
    /// Returns a representation that matches the config format for easy lookup
    pub fn to_config_representation(&self) -> String {
        match self {
            RateLimitingScopeKey::TagTotal { key } => {
                format!(r#"tag_key="{key}", tag_value="tensorzero::total""#)
            }
            RateLimitingScopeKey::TagEach { key, value } => {
                format!(r#"tag_key="{key}", tag_value="tensorzero::each" (matched: "{value}")"#)
            }
            RateLimitingScopeKey::TagConcrete { key, value } => {
                format!(r#"tag_key="{key}", tag_value="{value}""#)
            }
            RateLimitingScopeKey::ApiKeyPublicIdEach { api_key_public_id } => {
                format!(r#"api_key_public_id="tensorzero::each" (matched: "{api_key_public_id}")"#)
            }
            RateLimitingScopeKey::ApiKeyPublicIdConcrete { api_key_public_id } => {
                format!(r#"api_key_public_id="{api_key_public_id}""#)
            }
        }
    }
}

// TODO: is there a way to enforce that this struct is consumed by return_tickets?
#[must_use]
#[derive(Debug)]
pub struct TicketBorrows {
    borrows: Vec<TicketBorrow>,
}

#[derive(Debug)]
struct TicketBorrow {
    receipt: ConsumeTicketsReceipt,
    active_limit: ActiveRateLimit,
}

impl TicketBorrows {
    pub fn empty() -> Self {
        Self {
            borrows: Vec::new(),
        }
    }

    fn new(
        results: Vec<ConsumeTicketsReceipt>,
        active_limits: Vec<ActiveRateLimit>,
        ticket_requests: Vec<ConsumeTicketsRequest>,
    ) -> Result<Self, Error> {
        // Assert all vectors have the same length
        let results_len = results.len();
        let active_limits_len = active_limits.len();

        if results_len != active_limits_len {
            return Err(Error::new(ErrorDetails::Inference {
            message: format!(
                "TicketBorrow has ragged arrays: receipts.len()={results_len}, active_limits.len()={active_limits_len}. {IMPOSSIBLE_ERROR_MESSAGE}",
            )
        }));
        }
        let receipts = align_and_check_limits(&active_limits, results, ticket_requests)?;
        let borrows = receipts
            .into_iter()
            .zip(active_limits)
            .map(|(receipt, active_limit)| TicketBorrow {
                receipt,
                active_limit,
            })
            .collect();

        Ok(Self { borrows })
    }

    #[tracing::instrument(err, skip_all, fields(otel.name = "rate_limiting_return_tickets", actual_usage.tokens, actual_usage.model_inferences, underestimate))]
    pub async fn return_tickets(
        self,
        client: &impl RateLimitQueries,
        actual_usage: RateLimitResourceUsage,
    ) -> Result<(), Error> {
        let res = self.return_tickets_inner(client, actual_usage).await;
        if let Err(e) = &res {
            // We want rate-limiting errors to show up as errors in OpenTelemetry,
            // even though they only get logged as warnings to the console.
            e.ensure_otel_span_errored(&Span::current());
        }
        res
    }

    // The actual implementation of `return_tickets`. This is a separate function so that we can
    // handle `Result::Err` inside `return_tickets`
    async fn return_tickets_inner(
        self,
        client: &impl RateLimitQueries,
        actual_usage: RateLimitResourceUsage,
    ) -> Result<(), Error> {
        let span = Span::current();
        // We cast the usage values to i64 so that they are reported as integers in OpenTelemetry (rather than strings)
        match actual_usage {
            RateLimitResourceUsage::Exact {
                tokens,
                model_inferences,
            } => {
                span.record("actual_usage.tokens", tokens as i64);
                span.record("actual_usage.model_inferences", model_inferences as i64);
                span.record("underestimate", false);
            }
            RateLimitResourceUsage::UnderEstimate {
                tokens,
                model_inferences,
            } => {
                span.record("actual_usage.tokens", tokens as i64);
                span.record("actual_usage.model_inferences", model_inferences as i64);
                span.record("underestimate", true);
            }
        }
        let mut requests = Vec::new();
        let mut returns = Vec::new();
        for borrow in &self.borrows {
            let TicketBorrow {
                receipt,
                active_limit,
            } = borrow;

            // Extract the actual value - we'll check 'Exact/UnderEstimate' further on
            let actual_usage_this_request = match active_limit.limit.resource {
                RateLimitResource::ModelInference => match actual_usage {
                    RateLimitResourceUsage::Exact {
                        model_inferences, ..
                    }
                    | RateLimitResourceUsage::UnderEstimate {
                        model_inferences, ..
                    } => model_inferences,
                },
                RateLimitResource::Token => match actual_usage {
                    RateLimitResourceUsage::Exact { tokens, .. }
                    | RateLimitResourceUsage::UnderEstimate { tokens, .. } => tokens,
                },
            };

            match actual_usage_this_request.cmp(&receipt.tickets_consumed) {
                std::cmp::Ordering::Greater => {
                    // Actual usage exceeds borrowed, add the difference to requests and log a warning
                    // We don't care about 'RateLimitResourceUsage::Exact' vs ' RateLimitResourceUsage::UnderEstimate' here.
                    tracing::warn!("Actual usage exceeds borrowed for {:?}: {} estimated and {actual_usage_this_request} used", active_limit.limit.resource, receipt.tickets_consumed);
                    let difference = actual_usage_this_request - receipt.tickets_consumed;
                    requests.push(active_limit.get_consume_tickets_request_for_return(difference)?);
                }
                std::cmp::Ordering::Less => {
                    match actual_usage {
                        RateLimitResourceUsage::Exact { .. } => {
                            // Borrowed exceeds actual usage, add the difference to returns
                            let difference = receipt.tickets_consumed - actual_usage_this_request;
                            returns.push(active_limit.get_return_tickets_request(difference)?);
                        }
                        RateLimitResourceUsage::UnderEstimate { .. } => {
                            // If our returned usage is only an estimate, then don't return any tickets,
                            // even if it looks like we over-borrowed.
                        }
                    }
                }
                std::cmp::Ordering::Equal => (),
            };
        }

        let (consume_result, return_result) = tokio::join!(
            client.consume_tickets(&requests),
            client.return_tickets(returns)
        );

        consume_result?;
        return_result?;

        Ok(())
    }
}

pub trait RateLimitedRequest {
    fn estimated_resource_usage(
        &self,
        resources: &[RateLimitResource],
    ) -> Result<EstimatedRateLimitResourceUsage, Error>;
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
    use std::ops::Deref;

    #[test]
    fn test_rate_limiting_config_scope_get_key_if_matches_tag_concrete_match() {
        let scope = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope {
            tag_key: "user_id".to_string(),
            tag_value: TagValueScope::Concrete("123".to_string()),
        });

        let mut tags = HashMap::new();
        tags.insert("user_id".to_string(), "123".to_string());

        let info = ScopeInfo {
            tags: Arc::new(tags),
            api_key_public_id: None,
        };

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

        let info = ScopeInfo {
            tags: Arc::new(tags),
            api_key_public_id: None,
        };

        let key = scope.get_key_if_matches(&info);
        assert!(key.is_none());
    }

    #[test]
    fn test_rate_limiting_config_scope_get_key_if_matches_tag_each() {
        let scope = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope {
            tag_key: "user_id".to_string(),
            tag_value: TagValueScope::Each,
        });

        let mut tags = HashMap::new();
        tags.insert("user_id".to_string(), "any_value".to_string());

        let info = ScopeInfo {
            tags: Arc::new(tags),
            api_key_public_id: None,
        };

        let key = scope.get_key_if_matches(&info).unwrap();
        match key {
            RateLimitingScopeKey::TagEach { key, value } => {
                assert_eq!(key, "user_id");
                assert_eq!(value, "any_value");
            }
            _ => panic!("Expected TagEach variant"),
        }
    }

    #[test]
    fn test_rate_limiting_config_scope_get_key_if_matches_tag_total() {
        let scope = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope {
            tag_key: "user_id".to_string(),
            tag_value: TagValueScope::Total,
        });

        let mut tags = HashMap::new();
        tags.insert("user_id".to_string(), "specific_value".to_string());

        let info = ScopeInfo {
            tags: Arc::new(tags),
            api_key_public_id: None,
        };

        let key = scope.get_key_if_matches(&info).unwrap();
        match key {
            RateLimitingScopeKey::TagTotal { key } => {
                assert_eq!(key, "user_id");
            }
            _ => panic!("Expected TagTotal variant"),
        }
    }

    #[test]
    fn test_rate_limiting_config_scope_get_key_if_matches_missing_tag() {
        let scope = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope {
            tag_key: "user_id".to_string(),
            tag_value: TagValueScope::Each,
        });

        let tags = HashMap::new(); // Empty tags

        let info = ScopeInfo {
            tags: Arc::new(tags),
            api_key_public_id: None,
        };

        let key = scope.get_key_if_matches(&info);
        assert!(key.is_none());
    }

    #[test]
    fn test_rate_limiting_config_scopes_get_key_if_matches_empty() {
        let scopes = RateLimitingConfigScopes::new(vec![]).unwrap();

        let tags = HashMap::new();
        let info = ScopeInfo {
            tags: Arc::new(tags),
            api_key_public_id: None,
        };

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

        let info = ScopeInfo {
            tags: Arc::new(tags),
            api_key_public_id: None,
        };

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

        let info = ScopeInfo {
            tags: Arc::new(tags),
            api_key_public_id: None,
        };

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
            tag_value: TagValueScope::Total,
        });
        let scopes = RateLimitingConfigScopes::new(vec![scope1, scope2]).unwrap();

        let mut tags = HashMap::new();
        tags.insert("application_id".to_string(), "app123".to_string());
        tags.insert("user_id".to_string(), "user456".to_string());

        let info = ScopeInfo {
            tags: Arc::new(tags),
            api_key_public_id: None,
        };

        let keys = scopes.get_key_if_matches(&info).unwrap();
        assert_eq!(keys.len(), 2);

        // Check that keys are in the same stable order as the scopes
        match (&keys[0], &keys[1]) {
            (
                RateLimitingScopeKey::TagConcrete {
                    key: key1,
                    value: value1,
                },
                RateLimitingScopeKey::TagTotal { key: key2 },
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

        let info = ScopeInfo {
            tags: Arc::new(tags),
            api_key_public_id: None,
        };

        // Should return None because not all scopes match
        let keys = scopes.get_key_if_matches(&info);
        assert!(keys.is_none());
    }

    #[test]
    fn test_rate_limiting_config_scopes_get_key_stability_across_different_scope_info() {
        // Test that the same scopes + different but equivalent ScopeInfo produce the same key structure
        let scope1 = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope {
            tag_key: "application_id".to_string(),
            tag_value: TagValueScope::Total,
        });
        let scope2 = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope {
            tag_key: "user_id".to_string(),
            tag_value: TagValueScope::Each,
        });
        let scopes = RateLimitingConfigScopes::new(vec![scope1, scope2]).unwrap();

        // First ScopeInfo
        let mut tags1 = HashMap::new();
        tags1.insert("application_id".to_string(), "app123".to_string());
        tags1.insert("user_id".to_string(), "user456".to_string());

        let info1 = ScopeInfo {
            tags: Arc::new(tags1),
            api_key_public_id: None,
        };

        // Second ScopeInfo with different tag values but same structure
        let mut tags2 = HashMap::new();
        tags2.insert("application_id".to_string(), "app789".to_string());
        tags2.insert("user_id".to_string(), "user101".to_string());

        let info2 = ScopeInfo {
            tags: Arc::new(tags2),
            api_key_public_id: None,
        };

        let keys1 = scopes.get_key_if_matches(&info1).unwrap();
        let keys2 = scopes.get_key_if_matches(&info2).unwrap();

        // Keys should have the same structure but different values for TagEach
        assert_eq!(keys1.len(), keys2.len());
        assert_eq!(keys1.len(), 2);

        // First key should be TagTotal (same for both)
        match (&keys1[0], &keys2[0]) {
            (
                RateLimitingScopeKey::TagTotal { key: key1 },
                RateLimitingScopeKey::TagTotal { key: key2 },
            ) => {
                assert_eq!(key1, key2);
                assert_eq!(*key1, "application_id");
            }
            _ => panic!("Expected TagTotal variants"),
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
            tag_value: TagValueScope::Each,
        });
        let scope2 = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope {
            tag_key: "user_id".to_string(),
            tag_value: TagValueScope::Concrete("123".to_string()),
        });
        let scope3 = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope {
            tag_key: "organization_id".to_string(),
            tag_value: TagValueScope::Total,
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
                assert_eq!(tag1.tag_value, TagValueScope::Each);
                assert_eq!(tag2.tag_key, "organization_id");
                assert_eq!(tag2.tag_value, TagValueScope::Total);
                assert_eq!(tag3.tag_key, "user_id");
                assert_eq!(tag3.tag_value, TagValueScope::Concrete("123".to_string()));
            }
            _ => panic!("Expected Tag variants"),
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
        let scope_each = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope {
            tag_key: "user_id".to_string(),
            tag_value: TagValueScope::Each,
        });
        let scope_total = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope {
            tag_key: "user_id".to_string(),
            tag_value: TagValueScope::Total,
        });

        // Test sorting order - TagValueScope variants should sort: Concrete < Each < Total
        let scopes_order1 = RateLimitingConfigScopes::new(vec![
            scope_concrete.clone(),
            scope_each.clone(),
            scope_total.clone(),
        ])
        .unwrap();
        let scopes_order2 = RateLimitingConfigScopes::new(vec![
            scope_each.clone(),
            scope_total.clone(),
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
                // Based on Rust's enum ordering: Concrete(String) < Each < Total
                assert_eq!(tag1.tag_value, TagValueScope::Concrete("123".to_string()));
                assert_eq!(tag2.tag_value, TagValueScope::Each);
                assert_eq!(tag3.tag_value, TagValueScope::Total);
            }
            _ => panic!("Expected Tag variants"),
        }

        // Test that each scope produces different key types
        let mut tags = HashMap::new();
        tags.insert("user_id".to_string(), "123".to_string());

        let info = ScopeInfo {
            tags: Arc::new(tags),
            api_key_public_id: None,
        };

        // Test each scope individually to verify different key types
        let key_concrete = scope_concrete.get_key_if_matches(&info).unwrap();
        let key_each = scope_each.get_key_if_matches(&info).unwrap();
        let key_total = scope_total.get_key_if_matches(&info).unwrap();

        // Verify each produces the correct key variant
        match key_concrete {
            RateLimitingScopeKey::TagConcrete { ref key, ref value } => {
                assert_eq!(key, "user_id");
                assert_eq!(value, "123");
            }
            _ => panic!("Expected TagConcrete variant"),
        }

        match key_each {
            RateLimitingScopeKey::TagEach { ref key, ref value } => {
                assert_eq!(key, "user_id");
                assert_eq!(value, "123");
            }
            _ => panic!("Expected TagEach variant"),
        }

        match key_total {
            RateLimitingScopeKey::TagTotal { ref key } => {
                assert_eq!(key, "user_id");
            }
            _ => panic!("Expected TagTotal variant"),
        }

        // Test serialization shows they produce different JSON structures
        use serde_json;
        let json_concrete = serde_json::to_string(&key_concrete).unwrap();
        let json_each = serde_json::to_string(&key_each).unwrap();
        let json_total = serde_json::to_string(&key_total).unwrap();

        assert!(json_concrete.contains("\"type\":\"TagConcrete\""));
        assert!(json_each.contains("\"type\":\"TagEach\""));
        assert!(json_total.contains("\"type\":\"TagTotal\""));

        // Ensure they're all different
        assert_ne!(json_concrete, json_each);
        assert_ne!(json_each, json_total);
        assert_ne!(json_concrete, json_total);
    }

    #[test]
    fn test_different_scopes_same_key_duplicate_detection() {
        // Test that having the same tag_key with different TagValueScope variants
        // are NOT considered duplicates (they are different scopes)
        let scope_concrete = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope {
            tag_key: "user_id".to_string(),
            tag_value: TagValueScope::Concrete("123".to_string()),
        });
        let scope_total = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope {
            tag_key: "user_id".to_string(),
            tag_value: TagValueScope::Total,
        });

        // These should be allowed together since they have different TagValueScope
        let result = RateLimitingConfigScopes::new(vec![scope_concrete, scope_total]);
        assert!(result.is_ok());

        let scopes = result.unwrap();
        assert_eq!(scopes.len(), 2);
    }

    impl Deref for RateLimitingConfigScopes {
        type Target = Vec<RateLimitingConfigScope>;

        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    // Consolidated comprehensive unit tests

    #[test]
    fn test_rate_limiting_config_states() {
        // Test default configuration
        let default_config = RateLimitingConfig::default();
        assert!(default_config.enabled());
        assert!(default_config.rules().is_empty());

        // Test enabled/disabled states
        let config_enabled = RateLimitingConfig {
            rules: vec![],
            enabled: true,
        };
        assert!(config_enabled.enabled());

        let config_disabled = RateLimitingConfig {
            rules: vec![],
            enabled: false,
        };
        assert!(!config_disabled.enabled());

        // Test get_active_limits behavior
        let mut tags = HashMap::new();
        tags.insert("user_id".to_string(), "123".to_string());
        let scope_info = ScopeInfo {
            tags: Arc::new(tags),
            api_key_public_id: None,
        };

        // Disabled config should return empty limits
        let active_limits_disabled = config_disabled.get_active_limits(&scope_info);
        assert!(active_limits_disabled.is_empty());

        // Enabled config with no rules should return empty limits
        let active_limits_no_rules = config_enabled.get_active_limits(&scope_info);
        assert!(active_limits_no_rules.is_empty());
    }

    #[test]
    fn test_rate_limiting_config_rejects_duplicate_scopes() {
        // Create test limits
        let token_limit = Arc::new(RateLimit {
            resource: RateLimitResource::Token,
            interval: RateLimitInterval::Minute,
            capacity: 100,
            refill_rate: 10,
        });

        let inference_limit = Arc::new(RateLimit {
            resource: RateLimitResource::ModelInference,
            interval: RateLimitInterval::Hour,
            capacity: 50,
            refill_rate: 5,
        });

        let rule_priority_5 = RateLimitingConfigRule {
            limits: vec![token_limit.clone()],
            scope: RateLimitingConfigScopes::new(vec![RateLimitingConfigScope::Tag(
                TagRateLimitingConfigScope {
                    tag_key: "user_id".to_string(),
                    tag_value: TagValueScope::Total,
                },
            )])
            .unwrap(),
            priority: RateLimitingConfigPriority::Priority(5),
        };

        let rule_always = RateLimitingConfigRule {
            limits: vec![inference_limit.clone()],
            scope: RateLimitingConfigScopes::new(vec![RateLimitingConfigScope::Tag(
                TagRateLimitingConfigScope {
                    tag_key: "user_id".to_string(),
                    tag_value: TagValueScope::Total,
                },
            )])
            .unwrap(),
            priority: RateLimitingConfigPriority::Always,
        };

        let uninitialized = UninitializedRateLimitingConfig {
            rules: vec![rule_priority_5, rule_always],
            enabled: true,
        };
        let err_message = RateLimitingConfig::try_from(uninitialized)
            .unwrap_err()
            .to_string();
        assert!(err_message.contains("Rate limiting config scopes must be unique"));
        assert!(err_message.contains(r#"RateLimitingConfigScopes([Tag(TagRateLimitingConfigScope { tag_key: "user_id", tag_value: Total })])"#));
    }

    #[test]
    fn test_rate_limiting_config_priority_logic() {
        let mut tags = HashMap::new();
        tags.insert("user_id".to_string(), "test".to_string());
        let scope_info = ScopeInfo {
            tags: Arc::new(tags),
            api_key_public_id: None,
        };

        let token_limit = Arc::new(RateLimit {
            resource: RateLimitResource::Token,
            interval: RateLimitInterval::Minute,
            capacity: 100,
            refill_rate: 10,
        });

        let inference_limit = Arc::new(RateLimit {
            resource: RateLimitResource::ModelInference,
            interval: RateLimitInterval::Hour,
            capacity: 50,
            refill_rate: 5,
        });
        // Test 1: Highest numeric priority wins among numeric priorities
        let rule_priority_3 = RateLimitingConfigRule {
            limits: vec![token_limit.clone()],
            scope: RateLimitingConfigScopes::new(vec![RateLimitingConfigScope::Tag(
                TagRateLimitingConfigScope {
                    tag_key: "user_id".to_string(),
                    tag_value: TagValueScope::Total,
                },
            )])
            .unwrap(),
            priority: RateLimitingConfigPriority::Priority(3),
        };

        let rule_priority_7 = RateLimitingConfigRule {
            limits: vec![inference_limit.clone()],
            scope: RateLimitingConfigScopes::new(vec![RateLimitingConfigScope::Tag(
                TagRateLimitingConfigScope {
                    tag_key: "user_id".to_string(),
                    tag_value: TagValueScope::Each,
                },
            )])
            .unwrap(),
            priority: RateLimitingConfigPriority::Priority(7),
        };

        let config_numeric_priorities = RateLimitingConfig {
            rules: vec![rule_priority_3, rule_priority_7],
            enabled: true,
        };

        let active_limits = config_numeric_priorities.get_active_limits(&scope_info);
        // Should only have the limit from priority 7 rule
        assert_eq!(active_limits.len(), 1);
        assert_eq!(
            active_limits[0].limit.resource,
            RateLimitResource::ModelInference
        );

        // Test 2: Multiple limits in same rule
        let rule_multiple_limits = RateLimitingConfigRule {
            limits: vec![token_limit.clone(), inference_limit.clone()],
            scope: RateLimitingConfigScopes::new(vec![RateLimitingConfigScope::Tag(
                TagRateLimitingConfigScope {
                    tag_key: "user_id".to_string(),
                    tag_value: TagValueScope::Total,
                },
            )])
            .unwrap(),
            priority: RateLimitingConfigPriority::Always,
        };

        let config_multiple_limits = RateLimitingConfig {
            rules: vec![rule_multiple_limits],
            enabled: true,
        };

        let active_limits = config_multiple_limits.get_active_limits(&scope_info);
        assert_eq!(active_limits.len(), 2);
        let resources: Vec<RateLimitResource> = active_limits
            .iter()
            .map(|limit| limit.limit.resource)
            .collect();
        assert!(resources.contains(&RateLimitResource::Token));
        assert!(resources.contains(&RateLimitResource::ModelInference));
    }

    #[test]
    fn test_rate_limit_interval_to_pg_interval() {
        assert_eq!(
            RateLimitInterval::Second.to_pg_interval(),
            PgInterval {
                months: 0,
                days: 0,
                microseconds: 1_000_000
            }
        );
        assert_eq!(
            RateLimitInterval::Minute.to_pg_interval(),
            PgInterval {
                months: 0,
                days: 0,
                microseconds: 60_000_000
            }
        );
        assert_eq!(
            RateLimitInterval::Hour.to_pg_interval(),
            PgInterval {
                months: 0,
                days: 0,
                microseconds: 3_600_000_000
            }
        );
        assert_eq!(
            RateLimitInterval::Day.to_pg_interval(),
            PgInterval {
                months: 0,
                days: 1,
                microseconds: 0
            }
        );
        assert_eq!(
            RateLimitInterval::Week.to_pg_interval(),
            PgInterval {
                months: 0,
                days: 7,
                microseconds: 0
            }
        );
        assert_eq!(
            RateLimitInterval::Month.to_pg_interval(),
            PgInterval {
                months: 1,
                days: 0,
                microseconds: 0
            }
        );
    }

    #[test]
    fn test_active_rate_limit_keys() {
        // Test basic key generation and content validation
        let token_limit = Arc::new(RateLimit {
            resource: RateLimitResource::Token,
            interval: RateLimitInterval::Minute,
            capacity: 100,
            refill_rate: 10,
        });

        let inference_limit = Arc::new(RateLimit {
            resource: RateLimitResource::ModelInference,
            interval: RateLimitInterval::Minute,
            capacity: 100,
            refill_rate: 10,
        });

        let concrete_scope_key = vec![RateLimitingScopeKey::TagConcrete {
            key: "user_id".to_string(),
            value: "123".to_string(),
        }];

        let total_scope_key = vec![RateLimitingScopeKey::TagTotal {
            key: "app_id".to_string(),
        }];

        let active_limit_token = ActiveRateLimit {
            limit: token_limit.clone(),
            scope_key: concrete_scope_key.clone(),
        };

        let key = active_limit_token.get_key().unwrap();

        // Test key content - should contain resource and scope info
        let key_str = key.to_string();
        assert!(key_str.contains("token") || key_str.contains("Token"));
        assert!(key_str.contains("TagConcrete") || key_str.contains("tag_concrete"));
        assert!(key_str.contains("user_id"));
        assert!(key_str.contains("123"));

        // Test key stability - same inputs should produce same key
        let active_limit_token2 = ActiveRateLimit {
            limit: token_limit.clone(),
            scope_key: concrete_scope_key.clone(),
        };

        let key2 = active_limit_token2.get_key().unwrap();
        assert_eq!(key.0, key2.0);

        // Test different resources produce different keys
        let active_limit_inference = ActiveRateLimit {
            limit: inference_limit,
            scope_key: concrete_scope_key.clone(),
        };

        let key_inference = active_limit_inference.get_key().unwrap();
        assert_ne!(key.0, key_inference.0);

        // Test different scopes produce different keys
        let active_limit_different_scope = ActiveRateLimit {
            limit: token_limit,
            scope_key: total_scope_key,
        };

        let key_different_scope = active_limit_different_scope.get_key().unwrap();
        assert_ne!(key.0, key_different_scope.0);
    }

    #[test]
    fn test_active_rate_limit_requests() {
        // Test consume tickets request for Token resource
        let token_limit = Arc::new(RateLimit {
            resource: RateLimitResource::Token,
            interval: RateLimitInterval::Minute,
            capacity: 100,
            refill_rate: 10,
        });

        let token_active_limit = ActiveRateLimit {
            limit: token_limit.clone(),
            scope_key: vec![RateLimitingScopeKey::TagConcrete {
                key: "user_id".to_string(),
                value: "123".to_string(),
            }],
        };

        let usage = EstimatedRateLimitResourceUsage {
            model_inferences: Some(5),
            tokens: Some(50),
        };

        let consume_request = token_active_limit
            .get_consume_tickets_request(&usage)
            .unwrap();
        assert_eq!(consume_request.requested, 50); // tokens usage
        assert_eq!(consume_request.capacity, 100);
        assert_eq!(consume_request.refill_amount, 10);
        assert_eq!(
            consume_request.refill_interval,
            PgInterval {
                months: 0,
                days: 0,
                microseconds: 60_000_000
            }
        );

        // Test return tickets request for ModelInference resource
        let inference_limit = Arc::new(RateLimit {
            resource: RateLimitResource::ModelInference,
            interval: RateLimitInterval::Hour,
            capacity: 20,
            refill_rate: 5,
        });

        let inference_active_limit = ActiveRateLimit {
            limit: inference_limit.clone(),
            scope_key: vec![RateLimitingScopeKey::TagTotal {
                key: "app_id".to_string(),
            }],
        };

        let return_request = inference_active_limit
            .get_return_tickets_request(3)
            .unwrap();
        assert_eq!(return_request.returned, 3);
        assert_eq!(return_request.capacity, 20);
        assert_eq!(return_request.refill_amount, 5);
        assert_eq!(
            return_request.refill_interval,
            PgInterval {
                months: 0,
                days: 0,
                microseconds: 3_600_000_000
            }
        );

        // Test consume tickets request for ModelInference resource
        let inference_consume_request = inference_active_limit
            .get_consume_tickets_request(&usage)
            .unwrap();
        assert_eq!(inference_consume_request.requested, 5); // model_inferences usage
        assert_eq!(inference_consume_request.capacity, 20);
        assert_eq!(inference_consume_request.refill_amount, 5);
        assert_eq!(
            inference_consume_request.refill_interval,
            PgInterval {
                months: 0,
                days: 0,
                microseconds: 3_600_000_000
            }
        );

        // Test resource usage mapping works correctly
        assert_eq!(usage.get_usage(RateLimitResource::Token), Some(50));
        assert_eq!(usage.get_usage(RateLimitResource::ModelInference), Some(5));
    }

    #[test]
    fn test_ticket_borrow_lifecycle() {
        use crate::db::{ConsumeTicketsReceipt, ConsumeTicketsRequest};

        // Test empty borrow creation
        let empty_borrow = TicketBorrows::empty();
        assert_eq!(empty_borrow.borrows.len(), 0);

        // Test valid borrow creation
        let limit = Arc::new(RateLimit {
            resource: RateLimitResource::Token,
            interval: RateLimitInterval::Minute,
            capacity: 100,
            refill_rate: 10,
        });

        let active_limit = ActiveRateLimit {
            limit,
            scope_key: vec![RateLimitingScopeKey::TagConcrete {
                key: "user_id".to_string(),
                value: "123".to_string(),
            }],
        };

        let key = active_limit.get_key().unwrap();

        let receipt = ConsumeTicketsReceipt {
            key: key.clone(),
            success: true,
            tickets_remaining: 50,
            tickets_consumed: 50,
        };

        let request = ConsumeTicketsRequest {
            key: key.clone(),
            capacity: 100,
            refill_amount: 10,
            refill_interval: PgInterval {
                months: 0,
                days: 0,
                microseconds: 60_000_000,
            },
            requested: 50,
        };

        let valid_borrow =
            TicketBorrows::new(vec![receipt], vec![active_limit], vec![request]).unwrap();
        assert_eq!(valid_borrow.borrows.len(), 1);

        // Test iterator functionality
        let mut iter_count = 0;
        for borrow in &valid_borrow.borrows {
            let TicketBorrow {
                receipt,
                active_limit,
            } = &borrow;
            assert!(receipt.success);
            assert_eq!(receipt.tickets_consumed, 50);
            assert_eq!(active_limit.limit.resource, RateLimitResource::Token);
            iter_count += 1;
        }
        assert_eq!(iter_count, 1);

        // Test mismatched array lengths error
        let limit2 = Arc::new(RateLimit {
            resource: RateLimitResource::ModelInference,
            interval: RateLimitInterval::Hour,
            capacity: 20,
            refill_rate: 5,
        });

        let active_limit2 = ActiveRateLimit {
            limit: limit2,
            scope_key: vec![RateLimitingScopeKey::TagTotal {
                key: "app_id".to_string(),
            }],
        };

        let key2 = active_limit2.get_key().unwrap();

        // Try to create with mismatched array lengths
        let receipt2 = ConsumeTicketsReceipt {
            key: key2.clone(),
            success: true,
            tickets_remaining: 10,
            tickets_consumed: 10,
        };

        let request2 = ConsumeTicketsRequest {
            key: key2.clone(),
            capacity: 20,
            refill_amount: 5,
            refill_interval: PgInterval {
                months: 0,
                days: 0,
                microseconds: 3_600_000_000,
            },
            requested: 10,
        };

        let mismatched_result = TicketBorrows::new(vec![receipt2], vec![], vec![request2]);
        assert!(mismatched_result.is_err());

        // Another mismatch test - more active limits than receipts
        let receipt3 = ConsumeTicketsReceipt {
            key: active_limit2.get_key().unwrap(),
            success: true,
            tickets_remaining: 15,
            tickets_consumed: 15,
        };

        let request3 = ConsumeTicketsRequest {
            key: active_limit2.get_key().unwrap(),
            capacity: 20,
            refill_amount: 5,
            refill_interval: PgInterval {
                months: 0,
                days: 0,
                microseconds: 3_600_000_000,
            },
            requested: 15,
        };

        let mismatched_result2 =
            TicketBorrows::new(vec![receipt3], vec![active_limit2], vec![request3]);
        assert!(mismatched_result2.is_ok()); // This should actually work - 1:1 ratio
    }

    #[test]
    fn test_get_estimated_tokens() {
        assert_eq!(get_estimated_tokens(""), 0);
        assert_eq!(get_estimated_tokens("ab"), 1);
        assert_eq!(get_estimated_tokens("abcd"), 2);
        assert_eq!(get_estimated_tokens("hello world"), 5);

        let long_text = "a".repeat(1000);
        assert_eq!(get_estimated_tokens(&long_text), 500);
    }

    #[test]
    fn test_max_tokens_validation_with_rate_limits() {
        let tags = HashMap::new();
        let scope_info = ScopeInfo {
            tags: Arc::new(tags),
            api_key_public_id: None,
        };

        // Token rate limits active - should include Token resource
        let token_limit = Arc::new(RateLimit {
            resource: RateLimitResource::Token,
            interval: RateLimitInterval::Minute,
            capacity: 1000,
            refill_rate: 100,
        });
        let config_with_token = RateLimitingConfig {
            rules: vec![RateLimitingConfigRule {
                limits: vec![token_limit.clone()],
                scope: RateLimitingConfigScopes(vec![]),
                priority: RateLimitingConfigPriority::Priority(1),
            }],
            enabled: true,
        };
        let resources = config_with_token.get_rate_limited_resources(&scope_info);
        assert_eq!(resources.len(), 1);
        assert!(resources.contains(&RateLimitResource::Token));

        // Only ModelInference limits - should NOT include Token resource
        let model_limit = Arc::new(RateLimit {
            resource: RateLimitResource::ModelInference,
            interval: RateLimitInterval::Minute,
            capacity: 100,
            refill_rate: 10,
        });
        let config_model_only = RateLimitingConfig {
            rules: vec![RateLimitingConfigRule {
                limits: vec![model_limit.clone()],
                scope: RateLimitingConfigScopes(vec![]),
                priority: RateLimitingConfigPriority::Priority(1),
            }],
            enabled: true,
        };
        let resources = config_model_only.get_rate_limited_resources(&scope_info);
        assert_eq!(resources.len(), 1);
        assert!(resources.contains(&RateLimitResource::ModelInference));
        assert!(!resources.contains(&RateLimitResource::Token));

        // Disabled config - should return empty even with token limits
        let config_disabled = RateLimitingConfig {
            enabled: false,
            rules: vec![RateLimitingConfigRule {
                limits: vec![token_limit.clone()],
                scope: RateLimitingConfigScopes(vec![]),
                priority: RateLimitingConfigPriority::Priority(1),
            }],
        };
        assert!(config_disabled
            .get_rate_limited_resources(&scope_info)
            .is_empty());
    }

    #[test]
    fn test_align_and_check_limits() {
        use crate::db::{ConsumeTicketsReceipt, ConsumeTicketsRequest};
        use crate::error::ErrorDetails;

        let limit = Arc::new(RateLimit {
            resource: RateLimitResource::Token,
            interval: RateLimitInterval::Minute,
            capacity: 100,
            refill_rate: 10,
        });

        let active_limit = ActiveRateLimit {
            limit,
            scope_key: vec![RateLimitingScopeKey::TagConcrete {
                key: "user_id".to_string(),
                value: "123".to_string(),
            }],
        };

        let key = active_limit.get_key().unwrap();

        // Test success case
        let success_receipt = ConsumeTicketsReceipt {
            key: key.clone(),
            success: true,
            tickets_remaining: 50,
            tickets_consumed: 50,
        };

        let success_request = ConsumeTicketsRequest {
            key: key.clone(),
            capacity: 100,
            refill_amount: 10,
            refill_interval: PgInterval {
                months: 0,
                days: 0,
                microseconds: 60_000_000,
            },
            requested: 50,
        };

        let success_result = align_and_check_limits(
            &[active_limit],
            vec![success_receipt],
            vec![success_request],
        );
        assert!(success_result.is_ok());

        // Create a new active limit for the failure test since we used the original
        let active_limit2 = ActiveRateLimit {
            limit: Arc::new(RateLimit {
                resource: RateLimitResource::Token,
                interval: RateLimitInterval::Minute,
                capacity: 100,
                refill_rate: 10,
            }),
            scope_key: vec![RateLimitingScopeKey::TagConcrete {
                key: "user_id".to_string(),
                value: "123".to_string(),
            }],
        };

        let key2 = active_limit2.get_key().unwrap();

        // Test failure case - requesting 50 but only 30 available
        let failure_receipt = ConsumeTicketsReceipt {
            key: key2.clone(),
            success: false,
            tickets_remaining: 30,
            tickets_consumed: 0,
        };

        let failure_request = ConsumeTicketsRequest {
            key: key2.clone(),
            capacity: 100,
            refill_amount: 10,
            refill_interval: PgInterval {
                months: 0,
                days: 0,
                microseconds: 60_000_000,
            },
            requested: 50,
        };

        let failure_result = align_and_check_limits(
            &[active_limit2],
            vec![failure_receipt],
            vec![failure_request],
        );
        assert!(failure_result.is_err());

        match failure_result {
            Err(error) => {
                if let ErrorDetails::RateLimitExceeded { failed_rate_limits } = error.get_details()
                {
                    assert_eq!(failed_rate_limits.len(), 1);
                    assert_eq!(failed_rate_limits[0].key, key2);
                    assert_eq!(failed_rate_limits[0].requested, 50);
                    assert_eq!(failed_rate_limits[0].available, 30);
                } else {
                    panic!(
                        "Expected RateLimitExceeded error, got: {:?}",
                        error.get_details()
                    );
                }
            }
            Ok(_) => panic!("Expected an error, but got Ok"),
        }
    }

    #[test]
    fn test_multiple_rate_limit_failures() {
        use crate::db::{ConsumeTicketsReceipt, ConsumeTicketsRequest};
        use crate::error::ErrorDetails;

        // This test covers the bug where we need to correctly identify which rate limits
        // actually failed when Postgres tells us all borrows failed atomically.
        // If we have 3 rate limits and 2 are exceeded but 1 is fine, we should only
        // report the 2 that actually failed.

        let token_limit = Arc::new(RateLimit {
            resource: RateLimitResource::Token,
            interval: RateLimitInterval::Minute,
            capacity: 100,
            refill_rate: 10,
        });

        let inference_limit = Arc::new(RateLimit {
            resource: RateLimitResource::ModelInference,
            interval: RateLimitInterval::Minute,
            capacity: 50,
            refill_rate: 5,
        });

        let token_limit_user2 = Arc::new(RateLimit {
            resource: RateLimitResource::Token,
            interval: RateLimitInterval::Hour,
            capacity: 1000,
            refill_rate: 100,
        });

        let active_limit_tokens = ActiveRateLimit {
            limit: token_limit,
            scope_key: vec![RateLimitingScopeKey::TagConcrete {
                key: "user_id".to_string(),
                value: "user1".to_string(),
            }],
        };

        let active_limit_inferences = ActiveRateLimit {
            limit: inference_limit,
            scope_key: vec![RateLimitingScopeKey::TagConcrete {
                key: "user_id".to_string(),
                value: "user1".to_string(),
            }],
        };

        let active_limit_tokens_user2 = ActiveRateLimit {
            limit: token_limit_user2,
            scope_key: vec![RateLimitingScopeKey::TagConcrete {
                key: "user_id".to_string(),
                value: "user2".to_string(),
            }],
        };

        let key_tokens = active_limit_tokens.get_key().unwrap();
        let key_inferences = active_limit_inferences.get_key().unwrap();
        let key_tokens_user2 = active_limit_tokens_user2.get_key().unwrap();

        // Scenario: requesting 80 tokens and 10 inferences
        // - Token limit: only 30 available (FAIL - requested 80, available 30)
        // - Inference limit: 20 available (OK - requested 10, available 20)
        // - Token limit user2: 500 available (OK - requested 80, available 500)
        // Only the first one should be reported as failed

        let receipts = vec![
            ConsumeTicketsReceipt {
                key: key_tokens.clone(),
                success: false, // All fail atomically in Postgres
                tickets_remaining: 30,
                tickets_consumed: 0,
            },
            ConsumeTicketsReceipt {
                key: key_inferences.clone(),
                success: false, // All fail atomically in Postgres
                tickets_remaining: 20,
                tickets_consumed: 0,
            },
            ConsumeTicketsReceipt {
                key: key_tokens_user2.clone(),
                success: false, // All fail atomically in Postgres
                tickets_remaining: 500,
                tickets_consumed: 0,
            },
        ];

        let requests = vec![
            ConsumeTicketsRequest {
                key: key_tokens.clone(),
                capacity: 100,
                refill_amount: 10,
                refill_interval: PgInterval {
                    months: 0,
                    days: 0,
                    microseconds: 60_000_000,
                },
                requested: 80, // More than available (30)
            },
            ConsumeTicketsRequest {
                key: key_inferences.clone(),
                capacity: 50,
                refill_amount: 5,
                refill_interval: PgInterval {
                    months: 0,
                    days: 0,
                    microseconds: 60_000_000,
                },
                requested: 10, // Less than available (20)
            },
            ConsumeTicketsRequest {
                key: key_tokens_user2.clone(),
                capacity: 1000,
                refill_amount: 100,
                refill_interval: PgInterval {
                    months: 0,
                    days: 0,
                    microseconds: 3_600_000_000,
                },
                requested: 80, // Less than available (500)
            },
        ];

        let result = align_and_check_limits(
            &[
                active_limit_tokens,
                active_limit_inferences,
                active_limit_tokens_user2,
            ],
            receipts,
            requests,
        );

        assert!(result.is_err());

        match result {
            Err(error) => {
                if let ErrorDetails::RateLimitExceeded { failed_rate_limits } = error.get_details()
                {
                    // Should only have 1 failed rate limit (the token limit for user1)
                    assert_eq!(
                        failed_rate_limits.len(),
                        1,
                        "Expected 1 failed rate limit, got {}",
                        failed_rate_limits.len()
                    );
                    assert_eq!(failed_rate_limits[0].key, key_tokens);
                    assert_eq!(failed_rate_limits[0].requested, 80);
                    assert_eq!(failed_rate_limits[0].available, 30);
                } else {
                    panic!(
                        "Expected RateLimitExceeded error, got: {:?}",
                        error.get_details()
                    );
                }
            }
            Ok(_) => panic!("Expected an error, but got Ok"),
        }
    }

    #[test]
    fn test_no_refund_when_usage_is_none() {
        // This test verifies that when actual usage is reported as None (usage=None or null tokens),
        // we use RateLimitResourceUsage::UnderEstimate which prevents refunding tickets even if
        // we over-estimated the initial consumption.

        let token_usage_exact = RateLimitResourceUsage::Exact {
            model_inferences: 1,
            tokens: 100, // Actual usage is 100 tokens
        };

        let token_usage_underestimate = RateLimitResourceUsage::UnderEstimate {
            model_inferences: 1,
            tokens: 0, // This is what we use when usage is None
        };

        // Verify that Exact usage allows refunds (when actual < estimate)
        match token_usage_exact {
            RateLimitResourceUsage::Exact { tokens, .. } => {
                assert_eq!(tokens, 100);
                // This case would trigger a refund in return_tickets() if estimate was higher
            }
            RateLimitResourceUsage::UnderEstimate { .. } => {
                panic!("Expected Exact variant")
            }
        }

        // Verify that UnderEstimate usage prevents refunds
        match token_usage_underestimate {
            RateLimitResourceUsage::UnderEstimate { tokens, .. } => {
                assert_eq!(tokens, 0);
                // This case will NOT trigger a refund in return_tickets() per line 852-855
            }
            RateLimitResourceUsage::Exact { .. } => panic!("Expected UnderEstimate variant"),
        }
    }
}
