use crate::model::{
    CredentialLocation, CredentialLocationOrHardcoded, CredentialLocationWithFallback,
    EndpointLocation,
};
use crate::model_table::load_tensorzero_relay_credential;
use crate::relay::RelayCredentials;
use crate::{
    config::{
        BatchWritesConfig, ExportConfig, ObservabilityBackend, ObservabilityConfig, OtlpConfig,
        OtlpTracesConfig, OtlpTracesFormat, TemplateFilesystemAccess, UninitializedRelayConfig,
    },
    error::{Error, ErrorDetails},
    http::DEFAULT_HTTP_CLIENT_TIMEOUT,
    inference::types::storage::StorageKind,
    relay::TensorzeroRelay,
};
use chrono::Duration;
use serde::{Deserialize, Serialize};
use tensorzero_stored_config::{
    StoredAuthConfig, StoredBatchWritesConfig, StoredCredentialLocation,
    StoredCredentialLocationOrHardcoded, StoredCredentialLocationWithFallback,
    StoredEndpointLocation, StoredExportConfig, StoredGatewayAuthCacheConfig, StoredGatewayConfig,
    StoredGatewayMetricsConfig, StoredInferenceCacheBackend, StoredModelInferenceCacheConfig,
    StoredObservabilityBackend, StoredObservabilityConfig, StoredOtlpConfig,
    StoredOtlpTracesConfig, StoredOtlpTracesFormat, StoredRelayConfig,
    StoredValkeyModelInferenceCacheConfig,
};
use url::Url;

use super::ObjectStoreInfo;

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GatewayAuthCacheConfig {
    pub enabled: Option<bool>,
    pub ttl_ms: Option<u64>,
}

pub fn default_gateway_auth_cache_enabled() -> bool {
    true
}

pub fn default_gateway_auth_cache_ttl_ms() -> u64 {
    1000
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AuthConfig {
    pub enabled: bool,
    pub cache: Option<GatewayAuthCacheConfig>,
}

fn default_tensorzero_inference_latency_overhead_seconds_buckets() -> Vec<f64> {
    vec![0.001, 0.01, 0.1]
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MetricsConfig {
    /// Histogram buckets for the `tensorzero_inference_latency_overhead_seconds` metric.
    /// Defaults to `[0.001, 0.01, 0.1]`. Set to empty to disable the metric.
    pub tensorzero_inference_latency_overhead_seconds_buckets: Option<Vec<f64>>,
}

impl MetricsConfig {
    /// Returns the histogram buckets.
    pub fn get_buckets(&self) -> Vec<f64> {
        self.tensorzero_inference_latency_overhead_seconds_buckets
            .clone()
            .unwrap_or_else(default_tensorzero_inference_latency_overhead_seconds_buckets)
    }

    pub fn validate(&self) -> Result<(), Error> {
        let buckets = self.get_buckets();

        if !buckets.is_empty() {
            for (i, &bucket) in buckets.iter().enumerate() {
                if !bucket.is_finite() {
                    return Err(Error::new(crate::error::ErrorDetails::Config {
                        message: format!(
                            "gateway.metrics.tensorzero_inference_latency_overhead_seconds_buckets[{i}] must be finite (not NaN or infinity), got: {bucket}"
                        ),
                    }));
                }
                if bucket < 0.0 {
                    return Err(Error::new(crate::error::ErrorDetails::Config {
                        message: format!(
                            "gateway.metrics.tensorzero_inference_latency_overhead_seconds_buckets[{i}] must be non-negative, got: {bucket}"
                        ),
                    }));
                }
            }

            for i in 1..buckets.len() {
                if buckets[i] <= buckets[i - 1] {
                    return Err(Error::new(crate::error::ErrorDetails::Config {
                        message: format!(
                            "gateway.metrics.tensorzero_inference_latency_overhead_seconds_buckets must be in strictly ascending order, but buckets[{}] ({}) <= buckets[{}] ({})",
                            i,
                            buckets[i],
                            i - 1,
                            buckets[i - 1]
                        ),
                    }));
                }
            }
        }
        Ok(())
    }
}

/// Which backend to use for model inference caching.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum InferenceCacheBackend {
    /// Automatically select based on primary datastore:
    /// - ClickHouse primary → ClickHouse cache
    /// - Postgres primary → Valkey if available, else ClickHouse
    Auto,
    ClickHouse,
    Valkey,
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelInferenceCacheConfig {
    /// Whether caching is enabled.
    /// - `true`: require a cache backend (fail startup if unavailable)
    /// - `null` (default): use cache if available, warn and continue if not
    /// - `false`: disable caching entirely
    pub enabled: Option<bool>,
    /// Which cache backend to use.
    pub backend: Option<InferenceCacheBackend>,
    pub valkey: Option<ValkeyModelInferenceCacheConfig>,
}

// By default, cache entries in Valkey are retained for 24 hours.
const DEFAULT_VALKEY_CACHE_TTL_S: u64 = 86400; // 24 hours

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ValkeyModelInferenceCacheConfig {
    #[serde(default = "default_valkey_cache_ttl_s")]
    pub ttl_s: u64,
}

fn default_valkey_cache_ttl_s() -> u64 {
    DEFAULT_VALKEY_CACHE_TTL_S
}

impl Default for ValkeyModelInferenceCacheConfig {
    fn default() -> Self {
        Self {
            ttl_s: DEFAULT_VALKEY_CACHE_TTL_S,
        }
    }
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UninitializedGatewayConfig {
    #[serde(serialize_with = "serialize_optional_socket_addr")]
    pub bind_address: Option<std::net::SocketAddr>,
    pub observability: Option<ObservabilityConfig>,
    pub debug: Option<bool>,
    pub template_filesystem_access: Option<TemplateFilesystemAccess>,
    pub export: Option<ExportConfig>,
    // If set, all of the HTTP endpoints will have this path prepended.
    // E.g. a base path of `/custom/prefix` will cause the inference endpoint to become `/custom/prefix/inference`.
    pub base_path: Option<String>,
    // If set to `true`, disables validation on feedback queries (read from ClickHouse to check that the target is valid)
    pub unstable_disable_feedback_target_validation: Option<bool>,
    /// If enabled, adds an `error_json` field alongside the human-readable `error` field
    /// in HTTP error responses. This contains a JSON-serialized version of the error.
    /// While `error_json` will always be valid JSON when present, the exact contents is unstable,
    /// and may change at any time without warning.
    /// For now, this is only supported in the standalone gateway, and not in the embedded gateway.
    pub unstable_error_json: Option<bool>,
    pub disable_pseudonymous_usage_analytics: Option<bool>,
    pub fetch_and_encode_input_files_before_inference: Option<bool>,
    pub auth: Option<AuthConfig>,
    pub global_outbound_http_timeout_ms: Option<u64>,
    pub relay: Option<UninitializedRelayConfig>,
    pub metrics: Option<MetricsConfig>,
    pub cache: Option<ModelInferenceCacheConfig>,
}

impl UninitializedGatewayConfig {
    pub fn load(self, object_store_info: Option<&ObjectStoreInfo>) -> Result<GatewayConfig, Error> {
        let metrics = self.metrics.unwrap_or_default();
        metrics.validate()?;
        let fetch_and_encode_input_files_before_inference = if let Some(value) =
            self.fetch_and_encode_input_files_before_inference
        {
            value
        } else {
            if let Some(info) = object_store_info
                && !matches!(info.kind, StorageKind::Disabled)
            {
                tracing::info!(
                    "Object store is enabled but `gateway.fetch_and_encode_input_files_before_inference` is unset (defaults to `false`). In rare cases, the files we fetch for object storage may differ from the image the inference provider fetched if this setting is disabled."
                );
            }
            false
        };

        let relay = if let Some(relay_config) = self.relay {
            if let Some(gateway_url) = &relay_config.gateway_url {
                let location = relay_config.api_key_location.clone().unwrap_or(
                    CredentialLocationWithFallback::Single(CredentialLocation::None),
                );

                let credential = load_tensorzero_relay_credential(&location)?;
                Some(TensorzeroRelay::new(
                    gateway_url.clone(),
                    RelayCredentials::try_from(credential)?,
                    relay_config,
                )?)
            } else {
                None
            }
        } else {
            None
        };

        Ok(GatewayConfig {
            bind_address: self.bind_address,
            observability: self.observability.unwrap_or_default(),
            debug: self.debug.unwrap_or_default(),
            template_filesystem_access: self.template_filesystem_access.unwrap_or_default(),
            export: self.export.unwrap_or_default(),
            base_path: self.base_path,
            unstable_error_json: self.unstable_error_json.unwrap_or_default(),
            unstable_disable_feedback_target_validation: self
                .unstable_disable_feedback_target_validation
                .unwrap_or_default(),
            disable_pseudonymous_usage_analytics: self
                .disable_pseudonymous_usage_analytics
                .unwrap_or_default(),
            fetch_and_encode_input_files_before_inference,
            auth: self.auth.unwrap_or_default(),
            global_outbound_http_timeout: self
                .global_outbound_http_timeout_ms
                .map(|ms| Duration::milliseconds(ms as i64))
                .unwrap_or(DEFAULT_HTTP_CLIENT_TIMEOUT),
            relay,
            metrics,
            cache: self.cache.unwrap_or_default(),
        })
    }
}

// --- From impls: StoredGatewayConfig -> core types ---

impl From<StoredObservabilityBackend> for ObservabilityBackend {
    fn from(stored: StoredObservabilityBackend) -> Self {
        match stored {
            StoredObservabilityBackend::Auto => Self::Auto,
            StoredObservabilityBackend::ClickHouse => Self::ClickHouse,
            StoredObservabilityBackend::Postgres => Self::Postgres,
        }
    }
}

impl From<StoredBatchWritesConfig> for BatchWritesConfig {
    fn from(stored: StoredBatchWritesConfig) -> Self {
        Self {
            enabled: stored.enabled,
            __force_allow_embedded_batch_writes: None,
            flush_interval_ms: stored.flush_interval_ms,
            max_rows: stored.max_rows,
            max_rows_postgres: stored.max_rows_postgres,
            write_queue_capacity: stored.write_queue_capacity,
        }
    }
}

impl From<StoredObservabilityConfig> for ObservabilityConfig {
    #[expect(deprecated)]
    fn from(stored: StoredObservabilityConfig) -> Self {
        Self {
            enabled: stored.enabled,
            backend: stored.backend.map(Into::into),
            async_writes: stored.async_writes,
            batch_writes: stored.batch_writes.map(Into::into),
            disable_automatic_migrations: None,
        }
    }
}

impl From<StoredOtlpTracesFormat> for OtlpTracesFormat {
    fn from(stored: StoredOtlpTracesFormat) -> Self {
        match stored {
            StoredOtlpTracesFormat::OpenTelemetry => Self::OpenTelemetry,
            StoredOtlpTracesFormat::OpenInference => Self::OpenInference,
        }
    }
}

impl From<StoredOtlpTracesConfig> for OtlpTracesConfig {
    fn from(stored: StoredOtlpTracesConfig) -> Self {
        Self {
            enabled: stored.enabled,
            format: stored.format.map(Into::into),
            extra_headers: stored.extra_headers.map(|h| h.into_iter().collect()),
        }
    }
}

impl From<StoredOtlpConfig> for OtlpConfig {
    fn from(stored: StoredOtlpConfig) -> Self {
        Self {
            traces: stored.traces.map(Into::into),
        }
    }
}

impl From<StoredExportConfig> for ExportConfig {
    fn from(stored: StoredExportConfig) -> Self {
        Self {
            otlp: stored.otlp.map(Into::into),
        }
    }
}

impl From<StoredGatewayAuthCacheConfig> for GatewayAuthCacheConfig {
    fn from(stored: StoredGatewayAuthCacheConfig) -> Self {
        Self {
            enabled: stored.enabled,
            ttl_ms: stored.ttl_ms,
        }
    }
}

impl From<StoredAuthConfig> for AuthConfig {
    fn from(stored: StoredAuthConfig) -> Self {
        Self {
            enabled: stored.enabled,
            cache: stored.cache.map(Into::into),
        }
    }
}

impl From<StoredGatewayMetricsConfig> for MetricsConfig {
    fn from(stored: StoredGatewayMetricsConfig) -> Self {
        Self {
            tensorzero_inference_latency_overhead_seconds_buckets: stored
                .tensorzero_inference_latency_overhead_seconds_buckets,
        }
    }
}

impl From<StoredInferenceCacheBackend> for InferenceCacheBackend {
    fn from(stored: StoredInferenceCacheBackend) -> Self {
        match stored {
            StoredInferenceCacheBackend::Auto => Self::Auto,
            StoredInferenceCacheBackend::ClickHouse => Self::ClickHouse,
            StoredInferenceCacheBackend::Valkey => Self::Valkey,
        }
    }
}

impl From<StoredValkeyModelInferenceCacheConfig> for ValkeyModelInferenceCacheConfig {
    fn from(stored: StoredValkeyModelInferenceCacheConfig) -> Self {
        Self {
            ttl_s: stored.ttl_s.unwrap_or(DEFAULT_VALKEY_CACHE_TTL_S),
        }
    }
}

impl From<StoredModelInferenceCacheConfig> for ModelInferenceCacheConfig {
    fn from(stored: StoredModelInferenceCacheConfig) -> Self {
        Self {
            enabled: stored.enabled,
            backend: stored.backend.map(Into::into),
            valkey: stored.valkey.map(Into::into),
        }
    }
}

impl From<StoredCredentialLocation> for CredentialLocation {
    fn from(stored: StoredCredentialLocation) -> Self {
        match stored {
            StoredCredentialLocation::Env { value } => Self::Env(value),
            StoredCredentialLocation::PathFromEnv { value } => Self::PathFromEnv(value),
            StoredCredentialLocation::Dynamic { value } => Self::Dynamic(value),
            StoredCredentialLocation::Path { value } => Self::Path(value),
            StoredCredentialLocation::Sdk => Self::Sdk,
            StoredCredentialLocation::None => Self::None,
        }
    }
}

impl From<StoredCredentialLocationWithFallback> for CredentialLocationWithFallback {
    fn from(stored: StoredCredentialLocationWithFallback) -> Self {
        match stored {
            StoredCredentialLocationWithFallback::Single { location } => {
                Self::Single(location.into())
            }
            StoredCredentialLocationWithFallback::WithFallback { default, fallback } => {
                Self::WithFallback {
                    default: default.into(),
                    fallback: fallback.into(),
                }
            }
        }
    }
}

impl From<StoredCredentialLocationOrHardcoded> for CredentialLocationOrHardcoded {
    fn from(stored: StoredCredentialLocationOrHardcoded) -> Self {
        match stored {
            StoredCredentialLocationOrHardcoded::Hardcoded { value } => Self::Hardcoded(value),
            StoredCredentialLocationOrHardcoded::Location { location } => {
                Self::Location(location.into())
            }
        }
    }
}

impl From<StoredEndpointLocation> for EndpointLocation {
    fn from(stored: StoredEndpointLocation) -> Self {
        match stored {
            StoredEndpointLocation::Env { value } => Self::Env(value),
            StoredEndpointLocation::Dynamic { value } => Self::Dynamic(value),
            StoredEndpointLocation::Static { value } => Self::Static(value),
        }
    }
}

impl TryFrom<StoredRelayConfig> for UninitializedRelayConfig {
    type Error = Error;

    fn try_from(stored: StoredRelayConfig) -> Result<Self, Error> {
        let gateway_url = stored
            .gateway_url
            .map(|u| {
                Url::parse(&u).map_err(|e| {
                    Error::new(ErrorDetails::Config {
                        message: format!("Failed to parse relay `gateway_url` `{u}`: {e}"),
                    })
                })
            })
            .transpose()?;
        Ok(Self {
            gateway_url,
            api_key_location: stored.api_key_location.map(Into::into),
        })
    }
}

impl TryFrom<StoredGatewayConfig> for UninitializedGatewayConfig {
    type Error = Error;

    fn try_from(stored: StoredGatewayConfig) -> Result<Self, Error> {
        let bind_address = stored
            .bind_address
            .map(|addr| {
                addr.parse().map_err(|e| {
                    Error::new(ErrorDetails::Config {
                        message: format!("Failed to parse gateway `bind_address` `{addr}`: {e}"),
                    })
                })
            })
            .transpose()?;

        let relay = stored.relay.map(TryInto::try_into).transpose()?;

        Ok(UninitializedGatewayConfig {
            bind_address,
            observability: stored.observability.map(Into::into),
            debug: stored.debug,
            // Config-in-DB users are banned from using template_filesystem_access.
            template_filesystem_access: None,
            export: stored.export.map(Into::into),
            base_path: stored.base_path,
            unstable_disable_feedback_target_validation: stored
                .unstable_disable_feedback_target_validation,
            unstable_error_json: stored.unstable_error_json,
            disable_pseudonymous_usage_analytics: stored.disable_pseudonymous_usage_analytics,
            fetch_and_encode_input_files_before_inference: stored
                .fetch_and_encode_input_files_before_inference,
            auth: stored.auth.map(Into::into),
            global_outbound_http_timeout_ms: stored.global_outbound_http_timeout_ms,
            relay,
            metrics: stored.metrics.map(Into::into),
            cache: stored.cache.map(Into::into),
        })
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct GatewayConfig {
    pub bind_address: Option<std::net::SocketAddr>,
    pub observability: ObservabilityConfig,
    pub debug: bool,
    pub template_filesystem_access: TemplateFilesystemAccess,
    pub export: ExportConfig,
    // If set, all of the HTTP endpoints will have this path prepended.
    // E.g. a base path of `/custom/prefix` will cause the inference endpoint to become `/custom/prefix/inference`.
    pub base_path: Option<String>,
    pub unstable_error_json: bool,
    pub unstable_disable_feedback_target_validation: bool,
    #[serde(default)]
    pub disable_pseudonymous_usage_analytics: bool,
    #[serde(default)]
    pub fetch_and_encode_input_files_before_inference: bool,
    pub auth: AuthConfig,
    pub global_outbound_http_timeout: Duration,
    #[serde(skip)]
    pub relay: Option<TensorzeroRelay>,
    pub metrics: MetricsConfig,
    pub cache: ModelInferenceCacheConfig,
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            bind_address: Default::default(),
            observability: Default::default(),
            debug: Default::default(),
            template_filesystem_access: Default::default(),
            export: Default::default(),
            base_path: Default::default(),
            unstable_error_json: Default::default(),
            unstable_disable_feedback_target_validation: Default::default(),
            disable_pseudonymous_usage_analytics: Default::default(),
            fetch_and_encode_input_files_before_inference: Default::default(),
            auth: Default::default(),
            global_outbound_http_timeout: DEFAULT_HTTP_CLIENT_TIMEOUT,
            relay: Default::default(),
            metrics: Default::default(),
            cache: Default::default(),
        }
    }
}

// Signature dictated by Serde
#[expect(clippy::ref_option)]
fn serialize_optional_socket_addr<S>(
    addr: &Option<std::net::SocketAddr>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    match addr {
        Some(addr) => serializer.serialize_str(&addr.to_string()),
        None => serializer.serialize_none(),
    }
}
