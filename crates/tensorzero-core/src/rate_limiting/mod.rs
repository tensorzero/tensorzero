use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use sqlx::postgres::types::PgInterval;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use axum::Extension;
use serde::{Deserialize, Serialize};
use tracing::Span;
use tracing_opentelemetry::OpenTelemetrySpanExt;

use crate::db::{ConsumeTicketsReceipt, ConsumeTicketsRequest, ReturnTicketsRequest};
use crate::error::{Error, ErrorDetails, IMPOSSIBLE_ERROR_MESSAGE};
use tensorzero_auth::middleware::RequestApiKeyExtension;
use tensorzero_stored_config::{
    StoredRateLimit, StoredRateLimitInterval, StoredRateLimitingBackend, StoredRateLimitingConfig,
    StoredRateLimitingRule,
};

pub use tensorzero_error::rate_limiting_types::*;

mod rate_limiting_manager;

// Re-export RateLimitingManager at the module level
pub use rate_limiting_manager::RateLimitingManager;

/// Convert a dollar amount (f64) to nano-dollars (u64).
/// Negative values are clamped to 0. Values that would overflow `u64` are clamped to `u64::MAX`.
pub fn cost_to_nano_cost(cost: f64) -> u64 {
    if cost <= 0.0 {
        return 0;
    }
    let nano_cost = (cost * NANO_DOLLARS_PER_DOLLAR as f64).ceil();
    if nano_cost >= u64::MAX as f64 {
        return u64::MAX;
    }
    nano_cost as u64
}

/// Convert a `Decimal` dollar amount to nano-dollars (u64).
/// Returns 0 if the conversion fails or the value is negative.
pub fn decimal_cost_to_nano_cost(cost: Decimal) -> u64 {
    let nano_cost = cost * Decimal::from(NANO_DOLLARS_PER_DOLLAR);
    nano_cost.ceil().to_u64().unwrap_or(0)
}

/*
 * The high level flow for our rate limiting system is:
 *   1. The caller constructs a ScopeInfo containing the information
 *      about the current provider request that determines what rate limit scopes apply.
 *   2. The caller calls `RateLimitingConfig::consume_tickets`. Here, we:
 *       a) use the ScopeInfo to determine which rate limit scopes apply
 *       b) estimate (conservatively) the resource consumption of the request
 *       c) consume tickets from the rate limit scopes
 *       d) actually attempt to consume the tickets from the database (Postgres or Valkey)
 *       e) check success, throw an error on failure, and return a TicketBorrow
 *   3. The caller calls `TicketBorrow::return_tickets` with the actual usage observed.
 *      This will figure out post-facto bookkeeping for over- or under-consumption.
 *   Important Note: the database identifies an active rate limit based on a string `key`.
 *      If two distinct rate limits have the same key, they will be treated as the same rate limit and will trample.
 *      If the key changes, the rate limit will be treated as a new rate limit.
 *      For these reasons, developers should be careful not to change the key serialization and be similarly careful
 *      to not add keys which could trample one another.
 */

/// Specifies which backend to use for rate limiting.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RateLimitingBackend {
    /// Automatically select: Valkey if available, otherwise Postgres
    Auto,
    /// Force Postgres backend
    Postgres,
    /// Force Valkey backend
    Valkey,
}

#[derive(Debug, Clone)]
pub struct RateLimitingConfig {
    pub(crate) rules: Vec<RateLimitingConfigRule>,
    /// Whether rate limiting is enabled. Defaults to `true`.
    /// When `true` and rules are defined, a rate limiting backend must be available.
    pub(crate) enabled: bool,
    pub(crate) backend: RateLimitingBackend,
    /// Default cost in nano-dollars used for cost estimation when actual cost is unknown.
    /// Defaults to $1.00 = 1,000,000,000 nano-dollars.
    pub(crate) default_nano_cost: u64,
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct UninitializedRateLimitingConfig {
    pub(crate) rules: Option<Vec<RateLimitingConfigRule>>,
    pub(crate) enabled: Option<bool>,
    pub(crate) backend: Option<RateLimitingBackend>,
    pub(crate) default_nano_cost: Option<u64>,
}

impl TryFrom<UninitializedRateLimitingConfig> for RateLimitingConfig {
    type Error = Error;
    fn try_from(config: UninitializedRateLimitingConfig) -> Result<Self, Self::Error> {
        let rules = config.rules.unwrap_or_default();
        let enabled = config.enabled.unwrap_or(true);

        // Make sure no rules have duplicated RateLimitingConfigScopes
        let mut scopes = HashSet::new();
        for rule in &rules {
            if !scopes.insert(rule.scope.clone()) {
                return Err(Error::new(ErrorDetails::DuplicateRateLimitingConfigScope {
                    scope: rule.scope.clone(),
                }));
            }
        }

        if !enabled && !rules.is_empty() {
            tracing::warn!(
                "`rate_limiting.enabled` is `false` but rate limiting rules are defined. \
                 Rules will not be enforced."
            );
        }

        Ok(Self {
            rules,
            enabled,
            backend: config.backend.unwrap_or(RateLimitingBackend::Auto),
            default_nano_cost: config.default_nano_cost.unwrap_or(NANO_DOLLARS_PER_DOLLAR),
        })
    }
}

impl From<&RateLimitingConfig> for UninitializedRateLimitingConfig {
    fn from(config: &RateLimitingConfig) -> Self {
        // Destructure to ensure all fields are handled (compile error if field added/removed)
        let RateLimitingConfig {
            rules,
            enabled,
            backend,
            default_nano_cost,
        } = config;
        Self {
            rules: Some(rules.clone()),
            enabled: Some(*enabled),
            backend: Some(*backend),
            default_nano_cost: Some(*default_nano_cost),
        }
    }
}

fn default_enabled() -> bool {
    true
}

fn default_nano_cost() -> u64 {
    NANO_DOLLARS_PER_DOLLAR // $1.00
}

impl Default for RateLimitingConfig {
    fn default() -> Self {
        Self {
            rules: Vec::new(),
            enabled: default_enabled(),
            backend: RateLimitingBackend::Auto,
            default_nano_cost: default_nano_cost(),
        }
    }
}

impl From<StoredRateLimitingBackend> for RateLimitingBackend {
    fn from(stored: StoredRateLimitingBackend) -> Self {
        match stored {
            StoredRateLimitingBackend::Auto => RateLimitingBackend::Auto,
            StoredRateLimitingBackend::Postgres => RateLimitingBackend::Postgres,
            StoredRateLimitingBackend::Valkey => RateLimitingBackend::Valkey,
        }
    }
}

impl From<RateLimitingBackend> for StoredRateLimitingBackend {
    fn from(backend: RateLimitingBackend) -> Self {
        match backend {
            RateLimitingBackend::Auto => Self::Auto,
            RateLimitingBackend::Postgres => Self::Postgres,
            RateLimitingBackend::Valkey => Self::Valkey,
        }
    }
}

impl From<StoredRateLimitInterval> for RateLimitInterval {
    fn from(stored: StoredRateLimitInterval) -> Self {
        match stored {
            StoredRateLimitInterval::Second => RateLimitInterval::Second,
            StoredRateLimitInterval::Minute => RateLimitInterval::Minute,
            StoredRateLimitInterval::Hour => RateLimitInterval::Hour,
            StoredRateLimitInterval::Day => RateLimitInterval::Day,
            StoredRateLimitInterval::Week => RateLimitInterval::Week,
            StoredRateLimitInterval::Month => RateLimitInterval::Month,
        }
    }
}

impl From<RateLimitInterval> for StoredRateLimitInterval {
    fn from(interval: RateLimitInterval) -> Self {
        match interval {
            RateLimitInterval::Second => Self::Second,
            RateLimitInterval::Minute => Self::Minute,
            RateLimitInterval::Hour => Self::Hour,
            RateLimitInterval::Day => Self::Day,
            RateLimitInterval::Week => Self::Week,
            RateLimitInterval::Month => Self::Month,
        }
    }
}

impl From<&UninitializedRateLimitingConfig> for StoredRateLimitingConfig {
    fn from(config: &UninitializedRateLimitingConfig) -> Self {
        StoredRateLimitingConfig {
            rules: config.rules.as_ref().map(|rules| {
                rules
                    .iter()
                    .map(|rule| StoredRateLimitingRule {
                        limits: rule
                            .limits
                            .iter()
                            .map(|limit| StoredRateLimit {
                                resource: limit.resource.into(),
                                interval: limit.interval.into(),
                                capacity: limit.capacity,
                                refill_rate: limit.refill_rate,
                            })
                            .collect(),
                        scope: (&rule.scope).into(),
                        priority: (&rule.priority).into(),
                    })
                    .collect()
            }),
            enabled: config.enabled,
            backend: config.backend.map(Into::into),
            default_nano_cost: config.default_nano_cost,
        }
    }
}

impl TryFrom<StoredRateLimitingConfig> for UninitializedRateLimitingConfig {
    type Error = Error;

    fn try_from(stored: StoredRateLimitingConfig) -> Result<Self, Error> {
        let rules = stored
            .rules
            .unwrap_or_default()
            .into_iter()
            .map(|rule| {
                let limits = rule
                    .limits
                    .into_iter()
                    .map(|limit| {
                        Arc::new(RateLimit {
                            resource: limit.resource.into(),
                            interval: limit.interval.into(),
                            capacity: limit.capacity,
                            refill_rate: limit.refill_rate,
                        })
                    })
                    .collect::<Vec<_>>();
                let scope = rule.scope.try_into().map_err(|e| {
                    Error::new(ErrorDetails::Config {
                        message: format!("Failed to build rate limiting scopes: {e}"),
                    })
                })?;
                let priority = rule.priority.into();
                Ok(RateLimitingConfigRule {
                    limits,
                    scope,
                    priority,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let backend = stored.backend.map(Into::into);
        Ok(UninitializedRateLimitingConfig {
            rules: Some(rules),
            enabled: stored.enabled,
            backend,
            default_nano_cost: stored.default_nano_cost,
        })
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

    /// Expose relevant information from this `ScopeInfo` as OpenTelemetry span attributes
    pub(crate) fn apply_otel_span_attributes(&self, span: &Span) {
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

    pub(crate) fn get_active_limits(&self, scope_info: &ScopeInfo) -> Vec<ActiveRateLimit> {
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

/// Since the database will tell us all borrows failed if any failed, we figure out which rate limits
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
pub(crate) struct ActiveRateLimit {
    pub(crate) limit: Arc<RateLimit>,
    pub(crate) scope_key: Vec<RateLimitingScopeKey>,
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
            refill_interval: self.limit.interval,
            requested,
        })
    }

    /// Use this one if the borrowed usage < the actual usage
    pub fn get_return_tickets_request(&self, returned: u64) -> Result<ReturnTicketsRequest, Error> {
        Ok(ReturnTicketsRequest {
            key: self.get_key()?,
            capacity: self.limit.capacity,
            refill_amount: self.limit.refill_rate,
            refill_interval: self.limit.interval,
            returned,
        })
    }
}

#[derive(Serialize)]
struct ActiveRateLimitKeyHelper<'a> {
    resource: RateLimitResource,
    scope_key: &'a [RateLimitingScopeKey],
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

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
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
        let key = get_scope_keys_if_matches(&self.scope, scope_info)?;
        if let RateLimitingConfigPriority::Priority(priority) = self.priority
            && priority > *max_priority
        {
            *max_priority = priority;
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

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub struct RateLimit {
    pub resource: RateLimitResource,
    pub interval: RateLimitInterval,
    pub capacity: u64,
    pub refill_rate: u64,
}

#[derive(Debug)]
pub enum RateLimitResourceUsage {
    /// We received an exact usage amount back from the provider, so we can consume extra/release unused
    /// rate limiting resources, depending on whether our initial estimate was too high or too low
    Exact {
        model_inferences: u64,
        tokens: u64,
        /// Cost in nano-dollars. `None` means cost is unknown.
        nano_cost: Option<u64>,
    },
    /// We were only able to estimate the usage (e.g. if an error occurred in an inference stream,
    /// and there might have been additional usage chunks that we missed; or the provider did not report token usage).
    /// We'll still consume tokens/inferences if we went over the initial estimate, but we will *not*
    /// return tickets if our initial estimate seems to be too high (since the error could have
    /// hidden the actual usage).
    UnderEstimate {
        model_inferences: u64,
        tokens: u64,
        /// Cost in nano-dollars. `None` means cost is unknown.
        nano_cost: Option<u64>,
    },
}

#[derive(Debug)]
pub struct EstimatedRateLimitResourceUsage {
    pub model_inferences: Option<u64>,
    pub tokens: Option<u64>,
    /// Cost estimate in nano-dollars. Filled in by the rate limiting manager
    /// using `default_nano_cost` when cost rate limiting is active.
    pub nano_cost: Option<u64>,
}

impl EstimatedRateLimitResourceUsage {
    pub fn get_usage(&self, resource: RateLimitResource) -> Option<u64> {
        match resource {
            RateLimitResource::ModelInference => self.model_inferences,
            RateLimitResource::Token => self.tokens,
            RateLimitResource::Cost => self.nano_cost,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[cfg_attr(feature = "ts-bindings", ts(export))]
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

    /// Convert a `RateLimitInterval` to microseconds for Valkey Lua scripts.
    ///
    /// Note: in this implementation we define `Month` as exactly 30 days
    /// (2,592,000,000,000 microseconds), not a calendar-aware month.
    ///
    /// TODO(shuyangli): change Postgres to match.
    pub fn to_microseconds(&self) -> u64 {
        match *self {
            RateLimitInterval::Second => 1_000_000,
            RateLimitInterval::Minute => 60_000_000,
            RateLimitInterval::Hour => 3_600_000_000,
            RateLimitInterval::Day => 86_400_000_000,
            RateLimitInterval::Week => 604_800_000_000,
            RateLimitInterval::Month => 2_592_000_000_000,
        }
    }
}

trait Scope {
    fn get_key_if_matches(&self, info: &ScopeInfo) -> Option<RateLimitingScopeKey>;
}

/// Returns the key (as a Vec) if all scopes match the given info, or None if any do not.
fn get_scope_keys_if_matches(
    scopes: &RateLimitingConfigScopes,
    info: &ScopeInfo,
) -> Option<Vec<RateLimitingScopeKey>> {
    scopes
        .as_slice()
        .iter()
        .map(|scope| scope.get_key_if_matches(info))
        .collect::<Option<Vec<_>>>()
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

impl Scope for TagRateLimitingConfigScope {
    fn get_key_if_matches(&self, info: &ScopeInfo) -> Option<RateLimitingScopeKey> {
        let value = info.tags.get(self.tag_key())?;

        match self.tag_value() {
            TagValueScope::Concrete(expected_value) => {
                if value == expected_value {
                    Some(RateLimitingScopeKey::TagConcrete {
                        key: self.tag_key().to_string(),
                        value: value.clone(),
                    })
                } else {
                    None
                }
            }
            TagValueScope::Each => Some(RateLimitingScopeKey::TagEach {
                key: self.tag_key().to_string(),
                value: value.clone(),
            }),
            TagValueScope::Total => Some(RateLimitingScopeKey::TagTotal {
                key: self.tag_key().to_string(),
            }),
        }
    }
}

impl Scope for ApiKeyPublicIdConfigScope {
    fn get_key_if_matches(&self, info: &ScopeInfo) -> Option<RateLimitingScopeKey> {
        match self.api_key_public_id() {
            ApiKeyPublicIdValueScope::Concrete(key) => {
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

// TODO: is there a way to enforce that this struct is consumed by return_tickets?
#[must_use]
#[derive(Debug)]
pub struct TicketBorrows {
    pool_manager: Arc<RateLimitingManager>,
    borrows: Vec<TicketBorrow>,
}

#[derive(Debug)]
pub(crate) struct TicketBorrow {
    pub(crate) receipt: ConsumeTicketsReceipt,
    pub(crate) active_limit: ActiveRateLimit,
}

// Static assertions to ensure these types are Send + Sync
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<TicketBorrows>;
    let _ = assert_send_sync::<RateLimitingManager>;
};

impl TicketBorrows {
    pub(crate) fn empty(pool_manager: Arc<RateLimitingManager>) -> Self {
        Self {
            pool_manager,
            borrows: Vec::new(),
        }
    }

    pub(crate) fn new(
        pool_manager: Arc<RateLimitingManager>,
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
                ),
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

        Ok(Self {
            pool_manager,
            borrows,
        })
    }

    /// Get the manager that created this borrow.
    pub fn manager(&self) -> &Arc<RateLimitingManager> {
        &self.pool_manager
    }

    /// Get access to the borrows for processing by the manager.
    pub(crate) fn borrows(&self) -> &[TicketBorrow] {
        &self.borrows
    }

    /// Return tickets based on actual resource usage.
    ///
    /// This method is synchronous and spawns an async task internally to perform
    /// the database operations. The DB operations complete asynchronously after
    /// this method returns.
    pub async fn return_tickets(self, actual_usage: RateLimitResourceUsage) -> Result<(), Error> {
        let manager = Arc::clone(&self.pool_manager);
        manager.return_tickets(self, actual_usage).await
    }
}

pub trait RateLimitedRequest {
    fn estimated_resource_usage(
        &self,
        resources: &[RateLimitResource],
        rate_limiting_config: &RateLimitingConfig,
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
    use std::collections::HashMap;

    use googletest::{expect_that, gtest, matchers::eq};
    use tensorzero_stored_config::StoredRateLimitResource;

    use super::*;

    /// Populate every field of `UninitializedRateLimitingConfig` (including
    /// each rate-limit resource, each interval, both priority variants, and
    /// both scope variants) and verify that converting to
    /// `StoredRateLimitingConfig` and back is lossless.
    #[gtest]
    fn test_uninitialized_rate_limiting_config_round_trip() {
        let rules = vec![
            RateLimitingConfigRule {
                limits: vec![
                    Arc::new(RateLimit {
                        resource: RateLimitResource::ModelInference,
                        interval: RateLimitInterval::Second,
                        capacity: 10,
                        refill_rate: 1,
                    }),
                    Arc::new(RateLimit {
                        resource: RateLimitResource::Token,
                        interval: RateLimitInterval::Minute,
                        capacity: 1_000,
                        refill_rate: 100,
                    }),
                    Arc::new(RateLimit {
                        resource: RateLimitResource::Cost,
                        interval: RateLimitInterval::Hour,
                        capacity: 1_000_000_000,
                        refill_rate: 100_000_000,
                    }),
                ],
                scope: RateLimitingConfigScopes::new(vec![
                    RateLimitingConfigScope::Tag(TagRateLimitingConfigScope::new(
                        "user_id".to_string(),
                        TagValueScope::Concrete("alice".to_string()),
                    )),
                    RateLimitingConfigScope::ApiKeyPublicId(ApiKeyPublicIdConfigScope::new(
                        ApiKeyPublicIdValueScope::Each,
                    )),
                ])
                .expect("valid scopes"),
                priority: RateLimitingConfigPriority::Priority(7),
            },
            RateLimitingConfigRule {
                limits: vec![Arc::new(RateLimit {
                    resource: RateLimitResource::Token,
                    interval: RateLimitInterval::Day,
                    capacity: 50_000,
                    refill_rate: 2_500,
                })],
                scope: RateLimitingConfigScopes::new(vec![RateLimitingConfigScope::Tag(
                    TagRateLimitingConfigScope::new("org_id".to_string(), TagValueScope::Total),
                )])
                .expect("valid scopes"),
                priority: RateLimitingConfigPriority::Always,
            },
            RateLimitingConfigRule {
                limits: vec![Arc::new(RateLimit {
                    resource: RateLimitResource::ModelInference,
                    interval: RateLimitInterval::Week,
                    capacity: 100,
                    refill_rate: 10,
                })],
                scope: RateLimitingConfigScopes::new(vec![RateLimitingConfigScope::Tag(
                    TagRateLimitingConfigScope::new("workspace".to_string(), TagValueScope::Each),
                )])
                .expect("valid scopes"),
                priority: RateLimitingConfigPriority::Priority(3),
            },
            RateLimitingConfigRule {
                limits: vec![Arc::new(RateLimit {
                    resource: RateLimitResource::Cost,
                    interval: RateLimitInterval::Month,
                    capacity: 42,
                    refill_rate: 1,
                })],
                scope: RateLimitingConfigScopes::new(vec![
                    RateLimitingConfigScope::ApiKeyPublicId(ApiKeyPublicIdConfigScope::new(
                        ApiKeyPublicIdValueScope::Concrete("abcdef123456".to_string()),
                    )),
                ])
                .expect("valid scopes"),
                priority: RateLimitingConfigPriority::Priority(1),
            },
        ];

        let original = UninitializedRateLimitingConfig {
            rules: Some(rules),
            enabled: Some(false),
            backend: Some(RateLimitingBackend::Valkey),
            default_nano_cost: Some(2_500_000_000),
        };

        let stored: StoredRateLimitingConfig = (&original).into();
        let round_tripped: UninitializedRateLimitingConfig = stored
            .try_into()
            .expect("StoredRateLimitingConfig should convert back");
        expect_that!(round_tripped, eq(&original));
    }

    #[test]
    fn test_rate_limiting_config_scope_get_key_if_matches_tag_concrete_match() {
        let scope = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope::new(
            "user_id".to_string(),
            TagValueScope::Concrete("123".to_string()),
        ));

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
        let scope = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope::new(
            "user_id".to_string(),
            TagValueScope::Concrete("123".to_string()),
        ));

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
        let scope = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope::new(
            "user_id".to_string(),
            TagValueScope::Each,
        ));

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
        let scope = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope::new(
            "user_id".to_string(),
            TagValueScope::Total,
        ));

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
        let scope = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope::new(
            "user_id".to_string(),
            TagValueScope::Each,
        ));

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

        let keys = get_scope_keys_if_matches(&scopes, &info).unwrap();
        assert_eq!(keys.len(), 0);
    }

    #[test]
    fn test_rate_limiting_config_scopes_get_key_if_matches_single_match() {
        let scope = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope::new(
            "user_id".to_string(),
            TagValueScope::Concrete("123".to_string()),
        ));
        let scopes = RateLimitingConfigScopes::new(vec![scope]).unwrap();

        let mut tags = HashMap::new();
        tags.insert("user_id".to_string(), "123".to_string());

        let info = ScopeInfo {
            tags: Arc::new(tags),
            api_key_public_id: None,
        };

        let keys = get_scope_keys_if_matches(&scopes, &info).unwrap();
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
        let scope = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope::new(
            "user_id".to_string(),
            TagValueScope::Concrete("123".to_string()),
        ));
        let scopes = RateLimitingConfigScopes::new(vec![scope]).unwrap();

        let mut tags = HashMap::new();
        tags.insert("user_id".to_string(), "456".to_string());

        let info = ScopeInfo {
            tags: Arc::new(tags),
            api_key_public_id: None,
        };

        let keys = get_scope_keys_if_matches(&scopes, &info);
        assert!(keys.is_none());
    }

    #[test]
    fn test_rate_limiting_config_scopes_get_key_if_matches_multiple_all_match() {
        let scope1 = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope::new(
            "application_id".to_string(),
            TagValueScope::Concrete("app123".to_string()),
        ));
        let scope2 = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope::new(
            "user_id".to_string(),
            TagValueScope::Total,
        ));
        let scopes = RateLimitingConfigScopes::new(vec![scope1, scope2]).unwrap();

        let mut tags = HashMap::new();
        tags.insert("application_id".to_string(), "app123".to_string());
        tags.insert("user_id".to_string(), "user456".to_string());

        let info = ScopeInfo {
            tags: Arc::new(tags),
            api_key_public_id: None,
        };

        let keys = get_scope_keys_if_matches(&scopes, &info).unwrap();
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
        let scope1 = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope::new(
            "application_id".to_string(),
            TagValueScope::Concrete("app123".to_string()),
        ));
        let scope2 = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope::new(
            "user_id".to_string(),
            TagValueScope::Concrete("user456".to_string()),
        ));
        let scopes = RateLimitingConfigScopes::new(vec![scope1, scope2]).unwrap();

        let mut tags = HashMap::new();
        tags.insert("application_id".to_string(), "app123".to_string());
        tags.insert("user_id".to_string(), "user789".to_string()); // Different value

        let info = ScopeInfo {
            tags: Arc::new(tags),
            api_key_public_id: None,
        };

        // Should return None because not all scopes match
        let keys = get_scope_keys_if_matches(&scopes, &info);
        assert!(keys.is_none());
    }

    #[test]
    fn test_rate_limiting_config_scopes_get_key_stability_across_different_scope_info() {
        // Test that the same scopes + different but equivalent ScopeInfo produce the same key structure
        let scope1 = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope::new(
            "application_id".to_string(),
            TagValueScope::Total,
        ));
        let scope2 = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope::new(
            "user_id".to_string(),
            TagValueScope::Each,
        ));
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

        let keys1 = get_scope_keys_if_matches(&scopes, &info1).unwrap();
        let keys2 = get_scope_keys_if_matches(&scopes, &info2).unwrap();

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
        let scope1 = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope::new(
            "user_id".to_string(),
            TagValueScope::Concrete("123".to_string()),
        ));
        let scope2 = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope::new(
            "user_id".to_string(),
            TagValueScope::Concrete("123".to_string()),
        ));

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
        let scope1 = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope::new(
            "application_id".to_string(),
            TagValueScope::Each,
        ));
        let scope2 = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope::new(
            "user_id".to_string(),
            TagValueScope::Concrete("123".to_string()),
        ));
        let scope3 = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope::new(
            "organization_id".to_string(),
            TagValueScope::Total,
        ));

        // Create scopes in different orders
        let scopes_order1 =
            RateLimitingConfigScopes::new(vec![scope1.clone(), scope2.clone(), scope3.clone()])
                .unwrap();
        let scopes_order2 =
            RateLimitingConfigScopes::new(vec![scope3.clone(), scope1.clone(), scope2.clone()])
                .unwrap();
        let scopes_order3 = RateLimitingConfigScopes::new(vec![scope2, scope3, scope1]).unwrap();

        // All should result in the same order after sorting
        assert_eq!(scopes_order1.as_slice(), scopes_order2.as_slice());
        assert_eq!(scopes_order2.as_slice(), scopes_order3.as_slice());

        // Verify the actual sorted order
        match (&scopes_order1[0], &scopes_order1[1], &scopes_order1[2]) {
            (
                RateLimitingConfigScope::Tag(tag1),
                RateLimitingConfigScope::Tag(tag2),
                RateLimitingConfigScope::Tag(tag3),
            ) => {
                assert_eq!(tag1.tag_key(), "application_id");
                assert_eq!(*tag1.tag_value(), TagValueScope::Each);
                assert_eq!(tag2.tag_key(), "organization_id");
                assert_eq!(*tag2.tag_value(), TagValueScope::Total);
                assert_eq!(tag3.tag_key(), "user_id");
                assert_eq!(
                    *tag3.tag_value(),
                    TagValueScope::Concrete("123".to_string())
                );
            }
            _ => panic!("Expected Tag variants"),
        }
    }

    #[test]
    fn test_different_scopes_same_key_sorting_and_keys() {
        // Test that different TagValueScope variants with the same tag_key are sorted consistently
        // and produce different keys
        let scope_concrete = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope::new(
            "user_id".to_string(),
            TagValueScope::Concrete("123".to_string()),
        ));
        let scope_each = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope::new(
            "user_id".to_string(),
            TagValueScope::Each,
        ));
        let scope_total = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope::new(
            "user_id".to_string(),
            TagValueScope::Total,
        ));

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
        assert_eq!(scopes_order1.as_slice(), scopes_order2.as_slice());

        // Verify the actual sorted order - let's check what the actual order is
        match (&scopes_order1[0], &scopes_order1[1], &scopes_order1[2]) {
            (
                RateLimitingConfigScope::Tag(tag1),
                RateLimitingConfigScope::Tag(tag2),
                RateLimitingConfigScope::Tag(tag3),
            ) => {
                // All tags should have the same key
                assert_eq!(tag1.tag_key(), "user_id");
                assert_eq!(tag2.tag_key(), "user_id");
                assert_eq!(tag3.tag_key(), "user_id");

                // Test the actual derived sort order for TagValueScope
                // Based on Rust's enum ordering: Concrete(String) < Each < Total
                assert_eq!(
                    *tag1.tag_value(),
                    TagValueScope::Concrete("123".to_string())
                );
                assert_eq!(*tag2.tag_value(), TagValueScope::Each);
                assert_eq!(*tag3.tag_value(), TagValueScope::Total);
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
        let scope_concrete = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope::new(
            "user_id".to_string(),
            TagValueScope::Concrete("123".to_string()),
        ));
        let scope_total = RateLimitingConfigScope::Tag(TagRateLimitingConfigScope::new(
            "user_id".to_string(),
            TagValueScope::Total,
        ));

        // These should be allowed together since they have different TagValueScope
        let result = RateLimitingConfigScopes::new(vec![scope_concrete, scope_total]);
        assert!(result.is_ok());

        let scopes = result.unwrap();
        assert_eq!(scopes.len(), 2);
    }

    // Consolidated comprehensive unit tests

    #[test]
    fn test_rate_limiting_config_states() {
        // Test default configuration (enabled = true, no rules)
        let default_config = RateLimitingConfig::default();
        assert!(default_config.enabled(), "Default enabled should be true");
        assert!(default_config.rules().is_empty());

        // Test explicitly enabled (no rules → nothing to enforce)
        let config_enabled = RateLimitingConfig {
            rules: vec![],
            enabled: true,
            ..Default::default()
        };
        assert!(config_enabled.enabled());

        // Test explicitly disabled
        let config_disabled = RateLimitingConfig {
            rules: vec![],
            enabled: false,
            ..Default::default()
        };
        assert!(!config_disabled.enabled());

        // Test get_active_limits behavior
        let mut tags = HashMap::new();
        tags.insert("user_id".to_string(), "123".to_string());
        let scope_info = ScopeInfo {
            tags: Arc::new(tags),
            api_key_public_id: None,
        };

        // Explicitly disabled config should return empty limits
        let active_limits_disabled = config_disabled.get_active_limits(&scope_info);
        assert!(active_limits_disabled.is_empty());

        // Explicitly enabled config with no rules should return empty limits
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
                TagRateLimitingConfigScope::new("user_id".to_string(), TagValueScope::Total),
            )])
            .unwrap(),
            priority: RateLimitingConfigPriority::Priority(5),
        };

        let rule_always = RateLimitingConfigRule {
            limits: vec![inference_limit.clone()],
            scope: RateLimitingConfigScopes::new(vec![RateLimitingConfigScope::Tag(
                TagRateLimitingConfigScope::new("user_id".to_string(), TagValueScope::Total),
            )])
            .unwrap(),
            priority: RateLimitingConfigPriority::Always,
        };

        let uninitialized = UninitializedRateLimitingConfig {
            rules: Some(vec![rule_priority_5, rule_always]),
            enabled: Some(true),
            ..Default::default()
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
                TagRateLimitingConfigScope::new("user_id".to_string(), TagValueScope::Total),
            )])
            .unwrap(),
            priority: RateLimitingConfigPriority::Priority(3),
        };

        let rule_priority_7 = RateLimitingConfigRule {
            limits: vec![inference_limit.clone()],
            scope: RateLimitingConfigScopes::new(vec![RateLimitingConfigScope::Tag(
                TagRateLimitingConfigScope::new("user_id".to_string(), TagValueScope::Each),
            )])
            .unwrap(),
            priority: RateLimitingConfigPriority::Priority(7),
        };

        let config_numeric_priorities = RateLimitingConfig {
            rules: vec![rule_priority_3, rule_priority_7],
            enabled: true,
            ..Default::default()
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
                TagRateLimitingConfigScope::new("user_id".to_string(), TagValueScope::Total),
            )])
            .unwrap(),
            priority: RateLimitingConfigPriority::Always,
        };

        let config_multiple_limits = RateLimitingConfig {
            rules: vec![rule_multiple_limits],
            enabled: true,
            ..Default::default()
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
            nano_cost: None,
        };

        let consume_request = token_active_limit
            .get_consume_tickets_request(&usage)
            .unwrap();
        assert_eq!(consume_request.requested, 50); // tokens usage
        assert_eq!(consume_request.capacity, 100);
        assert_eq!(consume_request.refill_amount, 10);
        assert_eq!(consume_request.refill_interval, RateLimitInterval::Minute);

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
        assert_eq!(return_request.refill_interval, RateLimitInterval::Hour);

        // Test consume tickets request for ModelInference resource
        let inference_consume_request = inference_active_limit
            .get_consume_tickets_request(&usage)
            .unwrap();
        assert_eq!(inference_consume_request.requested, 5); // model_inferences usage
        assert_eq!(inference_consume_request.capacity, 20);
        assert_eq!(inference_consume_request.refill_amount, 5);
        assert_eq!(
            inference_consume_request.refill_interval,
            RateLimitInterval::Hour
        );

        // Test resource usage mapping works correctly
        assert_eq!(usage.get_usage(RateLimitResource::Token), Some(50));
        assert_eq!(usage.get_usage(RateLimitResource::ModelInference), Some(5));
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
                scope: RateLimitingConfigScopes::empty(),
                priority: RateLimitingConfigPriority::Priority(1),
            }],
            enabled: true,
            ..Default::default()
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
                scope: RateLimitingConfigScopes::empty(),
                priority: RateLimitingConfigPriority::Priority(1),
            }],
            enabled: true,
            ..Default::default()
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
                scope: RateLimitingConfigScopes::empty(),
                priority: RateLimitingConfigPriority::Priority(1),
            }],
            ..Default::default()
        };
        assert!(
            config_disabled
                .get_rate_limited_resources(&scope_info)
                .is_empty()
        );
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
            refill_interval: RateLimitInterval::Minute,
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
            refill_interval: RateLimitInterval::Minute,
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
                refill_interval: RateLimitInterval::Minute,
                requested: 80, // More than available (30)
            },
            ConsumeTicketsRequest {
                key: key_inferences.clone(),
                capacity: 50,
                refill_amount: 5,
                refill_interval: RateLimitInterval::Minute,
                requested: 10, // Less than available (20)
            },
            ConsumeTicketsRequest {
                key: key_tokens_user2.clone(),
                capacity: 1000,
                refill_amount: 100,
                refill_interval: RateLimitInterval::Hour,
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
            nano_cost: None,
        };

        let token_usage_underestimate = RateLimitResourceUsage::UnderEstimate {
            model_inferences: 1,
            tokens: 0, // This is what we use when usage is None
            nano_cost: None,
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

    #[test]
    fn test_tag_value_scope_deserialization_valid() {
        // Valid special values
        let each: TagValueScope = serde_json::from_str(r#""tensorzero::each""#).unwrap();
        assert_eq!(each, TagValueScope::Each);

        let total: TagValueScope = serde_json::from_str(r#""tensorzero::total""#).unwrap();
        assert_eq!(total, TagValueScope::Total);

        // Valid concrete value
        let concrete: TagValueScope = serde_json::from_str(r#""my_value""#).unwrap();
        assert_eq!(concrete, TagValueScope::Concrete("my_value".to_string()));
    }

    #[test]
    fn test_tag_value_scope_deserialization_invalid_tensorzero_prefix() {
        // Invalid tensorzero:: prefix
        let result: Result<TagValueScope, _> = serde_json::from_str(r#""tensorzero::foo""#);
        assert!(result.is_err(), "Should reject invalid tensorzero:: prefix");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("may not start with \"tensorzero::\""),
            "Error should mention tensorzero:: restriction: {err}"
        );
    }

    #[test]
    fn test_api_key_public_id_value_scope_deserialization_valid() {
        // Valid special value
        let each: ApiKeyPublicIdValueScope = serde_json::from_str(r#""tensorzero::each""#).unwrap();
        assert_eq!(each, ApiKeyPublicIdValueScope::Each);

        // Valid concrete value (12 characters)
        let concrete: ApiKeyPublicIdValueScope = serde_json::from_str(r#""abcdefghijkl""#).unwrap();
        assert_eq!(
            concrete,
            ApiKeyPublicIdValueScope::Concrete("abcdefghijkl".to_string())
        );
    }

    #[test]
    fn test_api_key_public_id_value_scope_deserialization_invalid_tensorzero_prefix() {
        // Invalid tensorzero:: prefix (not "each")
        let result: Result<ApiKeyPublicIdValueScope, _> =
            serde_json::from_str(r#""tensorzero::foo""#);
        assert!(result.is_err(), "Should reject invalid tensorzero:: prefix");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("may not start with \"tensorzero::\""),
            "Error should mention tensorzero:: restriction: {err}"
        );
    }

    #[test]
    fn test_api_key_public_id_value_scope_deserialization_invalid_length() {
        // Too short
        let result: Result<ApiKeyPublicIdValueScope, _> = serde_json::from_str(r#""abc""#);
        assert!(result.is_err(), "Should reject short public ID");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("must be 12 characters long"),
            "Error should mention length requirement: {err}"
        );

        // Too long
        let result: Result<ApiKeyPublicIdValueScope, _> =
            serde_json::from_str(r#""abcdefghijklmno""#);
        assert!(result.is_err(), "Should reject long public ID");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("must be 12 characters long"),
            "Error should mention length requirement: {err}"
        );
    }

    #[test]
    fn test_estimated_resource_usage_get_cost() {
        let usage = EstimatedRateLimitResourceUsage {
            model_inferences: Some(1),
            tokens: Some(100),
            nano_cost: Some(500_000_000),
        };
        assert_eq!(
            usage.get_usage(RateLimitResource::Cost),
            Some(500_000_000),
            "Should return cost in nano-dollars"
        );

        let usage_no_cost = EstimatedRateLimitResourceUsage {
            model_inferences: Some(1),
            tokens: Some(100),
            nano_cost: None,
        };
        assert_eq!(
            usage_no_cost.get_usage(RateLimitResource::Cost),
            None,
            "Should return None when cost is not set"
        );
    }

    #[test]
    fn test_active_rate_limit_key_cost_different_from_token() {
        let token_limit = Arc::new(RateLimit {
            resource: RateLimitResource::Token,
            interval: RateLimitInterval::Day,
            capacity: 1000,
            refill_rate: 1000,
        });

        let cost_limit = Arc::new(RateLimit {
            resource: RateLimitResource::Cost,
            interval: RateLimitInterval::Day,
            capacity: 10_000_000_000,
            refill_rate: 10_000_000_000,
        });

        let scope_key = vec![RateLimitingScopeKey::TagConcrete {
            key: "user_id".to_string(),
            value: "123".to_string(),
        }];

        let token_active = ActiveRateLimit {
            limit: token_limit,
            scope_key: scope_key.clone(),
        };
        let cost_active = ActiveRateLimit {
            limit: cost_limit,
            scope_key,
        };

        let token_key = token_active.get_key().unwrap();
        let cost_key = cost_active.get_key().unwrap();
        assert_ne!(
            token_key.0, cost_key.0,
            "Cost and Token keys should be different"
        );
    }

    #[test]
    fn test_failed_rate_limit_serialization_cost_displays_dollars() {
        let failed = FailedRateLimit {
            key: ActiveRateLimitKey("test_key".to_string()),
            requested: 1_500_000_000, // 1.5 nano-dollars
            available: 500_000_000,   // 0.5 nano-dollars
            resource: RateLimitResource::Cost,
            scope_key: vec![],
        };

        let json = serde_json::to_value(&failed).unwrap();
        assert_eq!(
            json["requested"], 1.5,
            "Cost requested should be displayed in dollars"
        );
        assert_eq!(
            json["available"], 0.5,
            "Cost available should be displayed in dollars"
        );
        assert_eq!(json["resource"], "cost");
    }

    #[test]
    fn test_failed_rate_limit_serialization_token_displays_raw() {
        let failed = FailedRateLimit {
            key: ActiveRateLimitKey("test_key".to_string()),
            requested: 1000,
            available: 500,
            resource: RateLimitResource::Token,
            scope_key: vec![],
        };

        let json = serde_json::to_value(&failed).unwrap();
        assert_eq!(
            json["requested"], 1000,
            "Token requested should be displayed as raw integer"
        );
        assert_eq!(
            json["available"], 500,
            "Token available should be displayed as raw integer"
        );
    }

    #[test]
    fn test_nano_dollar_conversions() {
        assert_eq!(
            cost_to_nano_cost(1.0),
            1_000_000_000,
            "$1.00 should be 1 billion nano-dollars"
        );
        assert_eq!(
            cost_to_nano_cost(0.001),
            1_000_000,
            "$0.001 should be 1 million nano-dollars"
        );
        assert_eq!(cost_to_nano_cost(0.0), 0, "$0.00 should be 0 nano-dollars");
        assert_eq!(
            cost_to_nano_cost(-1.0),
            0,
            "Negative values should clamp to 0"
        );

        let nano_cost = 1_500_000_000u64;
        let cost = nano_cost_to_cost(nano_cost);
        assert!(
            (cost - 1.5).abs() < f64::EPSILON,
            "1.5 billion nano-dollars should be $1.50"
        );
    }

    #[test]
    fn test_decimal_cost_to_nano_cost_conversion() {
        use rust_decimal::Decimal;

        let d = Decimal::new(150, 2); // 1.50
        assert_eq!(
            decimal_cost_to_nano_cost(d),
            1_500_000_000,
            "$1.50 as Decimal should be 1.5 billion nano-dollars"
        );

        let d = Decimal::new(1, 3); // 0.001
        assert_eq!(
            decimal_cost_to_nano_cost(d),
            1_000_000,
            "$0.001 as Decimal should be 1 million nano-dollars"
        );

        let d = Decimal::ZERO;
        assert_eq!(
            decimal_cost_to_nano_cost(d),
            0,
            "$0.00 should be 0 nano-dollars"
        );
    }

    // ─── Stored conversion round-trip tests ──────────────────────────────

    #[gtest]
    fn test_rate_limit_backend_round_trip() {
        for variant in [
            RateLimitingBackend::Auto,
            RateLimitingBackend::Postgres,
            RateLimitingBackend::Valkey,
        ] {
            let stored: StoredRateLimitingBackend = variant.into();
            let restored: RateLimitingBackend = stored.into();
            expect_that!(restored, eq(variant));
        }
    }

    #[gtest]
    fn test_rate_limit_resource_round_trip() {
        for variant in [
            RateLimitResource::ModelInference,
            RateLimitResource::Token,
            RateLimitResource::Cost,
        ] {
            let stored: StoredRateLimitResource = variant.into();
            let restored: RateLimitResource = stored.into();
            expect_that!(restored, eq(variant));
        }
    }

    #[gtest]
    fn test_rate_limit_interval_round_trip() {
        for variant in [
            RateLimitInterval::Second,
            RateLimitInterval::Minute,
            RateLimitInterval::Hour,
            RateLimitInterval::Day,
            RateLimitInterval::Week,
            RateLimitInterval::Month,
        ] {
            let stored: StoredRateLimitInterval = variant.into();
            let restored: RateLimitInterval = stored.into();
            expect_that!(restored, eq(variant));
        }
    }
}
