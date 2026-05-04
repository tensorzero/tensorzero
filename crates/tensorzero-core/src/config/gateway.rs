use crate::model::{CredentialLocation, CredentialLocationWithFallback};
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
use serde::{Deserialize, Deserializer, Serialize};
use tensorzero_stored_config::{
    StoredAuthConfig, StoredBatchWritesConfig, StoredCredentialLocationWithFallback,
    StoredExportConfig, StoredGatewayAuthCacheConfig, StoredGatewayConfig,
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
    #[serde(
        default,
        serialize_with = "serialize_optional_socket_addr",
        deserialize_with = "deserialize_optional_socket_addr"
    )]
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
    /// Per-read timeout (milliseconds) applied to each read of the response body. Reset on every
    /// successful read, so this bounds how long we wait for the *next* byte rather than the
    /// entire response. Useful for catching mid-stream stalls in SSE responses where the global
    /// total-deadline timeout wouldn't fire for a long time. `None` (the default) disables it.
    pub global_outbound_http_intra_stream_read_timeout_ms: Option<u64>,
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
            global_outbound_http_intra_stream_read_timeout: self
                .global_outbound_http_intra_stream_read_timeout_ms
                .map(|ms| Duration::milliseconds(ms as i64)),
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
            include_content: stored.include_content,
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
            api_key_location: stored
                .api_key_location
                .map(CredentialLocationWithFallback::from),
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
            global_outbound_http_intra_stream_read_timeout_ms: stored
                .global_outbound_http_intra_stream_read_timeout_ms,
            relay,
            metrics: stored.metrics.map(Into::into),
            cache: stored.cache.map(Into::into),
        })
    }
}

impl From<ObservabilityBackend> for StoredObservabilityBackend {
    fn from(backend: ObservabilityBackend) -> Self {
        match backend {
            ObservabilityBackend::Auto => StoredObservabilityBackend::Auto,
            ObservabilityBackend::ClickHouse => StoredObservabilityBackend::ClickHouse,
            ObservabilityBackend::Postgres => StoredObservabilityBackend::Postgres,
        }
    }
}

impl From<OtlpTracesFormat> for StoredOtlpTracesFormat {
    fn from(format: OtlpTracesFormat) -> Self {
        match format {
            OtlpTracesFormat::OpenTelemetry => StoredOtlpTracesFormat::OpenTelemetry,
            OtlpTracesFormat::OpenInference => StoredOtlpTracesFormat::OpenInference,
        }
    }
}

impl From<InferenceCacheBackend> for StoredInferenceCacheBackend {
    fn from(backend: InferenceCacheBackend) -> Self {
        match backend {
            InferenceCacheBackend::Auto => StoredInferenceCacheBackend::Auto,
            InferenceCacheBackend::ClickHouse => StoredInferenceCacheBackend::ClickHouse,
            InferenceCacheBackend::Valkey => StoredInferenceCacheBackend::Valkey,
        }
    }
}

impl From<UninitializedGatewayConfig> for StoredGatewayConfig {
    fn from(config: UninitializedGatewayConfig) -> Self {
        StoredGatewayConfig {
            bind_address: config.bind_address.map(|a| a.to_string()),
            observability: config.observability.map(|obs| StoredObservabilityConfig {
                enabled: obs.enabled,
                backend: obs.backend.map(StoredObservabilityBackend::from),
                async_writes: obs.async_writes,
                batch_writes: obs.batch_writes.map(|bw| StoredBatchWritesConfig {
                    enabled: bw.enabled,
                    flush_interval_ms: bw.flush_interval_ms,
                    max_rows: bw.max_rows,
                    max_rows_postgres: bw.max_rows_postgres,
                    write_queue_capacity: bw.write_queue_capacity,
                }),
            }),
            debug: config.debug,
            export: config.export.map(|exp| StoredExportConfig {
                otlp: exp.otlp.map(|otlp| StoredOtlpConfig {
                    traces: otlp.traces.map(|traces| StoredOtlpTracesConfig {
                        enabled: traces.enabled,
                        format: traces.format.map(StoredOtlpTracesFormat::from),
                        extra_headers: traces.extra_headers.map(|h| h.into_iter().collect()),
                        include_content: traces.include_content,
                    }),
                }),
            }),
            base_path: config.base_path,
            unstable_disable_feedback_target_validation: config
                .unstable_disable_feedback_target_validation,
            unstable_error_json: config.unstable_error_json,
            disable_pseudonymous_usage_analytics: config.disable_pseudonymous_usage_analytics,
            fetch_and_encode_input_files_before_inference: config
                .fetch_and_encode_input_files_before_inference,
            auth: config.auth.map(|auth| StoredAuthConfig {
                enabled: auth.enabled,
                cache: auth.cache.map(|c| StoredGatewayAuthCacheConfig {
                    enabled: c.enabled,
                    ttl_ms: c.ttl_ms,
                }),
            }),
            global_outbound_http_timeout_ms: config.global_outbound_http_timeout_ms,
            global_outbound_http_intra_stream_read_timeout_ms: config
                .global_outbound_http_intra_stream_read_timeout_ms,
            relay: config.relay.map(|r| StoredRelayConfig {
                gateway_url: r.gateway_url.map(|u| u.to_string()),
                api_key_location: r
                    .api_key_location
                    .as_ref()
                    .map(StoredCredentialLocationWithFallback::from),
            }),
            metrics: config.metrics.map(|m| StoredGatewayMetricsConfig {
                tensorzero_inference_latency_overhead_seconds_buckets: m
                    .tensorzero_inference_latency_overhead_seconds_buckets,
            }),
            cache: config.cache.map(|c| StoredModelInferenceCacheConfig {
                enabled: c.enabled,
                backend: c.backend.map(StoredInferenceCacheBackend::from),
                valkey: c.valkey.map(|v| StoredValkeyModelInferenceCacheConfig {
                    ttl_s: Some(v.ttl_s),
                }),
            }),
        }
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
    /// Per-read timeout for the outbound HTTP client. `None` disables it (preserving prior
    /// behavior where mid-stream stalls only fail when `global_outbound_http_timeout` fires).
    pub global_outbound_http_intra_stream_read_timeout: Option<Duration>,
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
            global_outbound_http_intra_stream_read_timeout: None,
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

fn deserialize_optional_socket_addr<'de, D>(
    deserializer: D,
) -> Result<Option<std::net::SocketAddr>, D::Error>
where
    D: Deserializer<'de>,
{
    let addr = Option::<String>::deserialize(deserializer)?;
    addr.map(|addr| addr.parse().map_err(serde::de::Error::custom))
        .transpose()
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use googletest::prelude::*;

    #[gtest]
    fn test_observability_backend_round_trip() {
        for variant in [
            ObservabilityBackend::Auto,
            ObservabilityBackend::ClickHouse,
            ObservabilityBackend::Postgres,
        ] {
            let stored: StoredObservabilityBackend = variant.into();
            let restored: ObservabilityBackend = stored.into();
            expect_that!(restored, eq(variant));
        }
    }

    #[gtest]
    fn test_otlp_traces_format_round_trip() {
        for variant in &[
            OtlpTracesFormat::OpenTelemetry,
            OtlpTracesFormat::OpenInference,
        ] {
            let stored: StoredOtlpTracesFormat = variant.clone().into();
            let restored: OtlpTracesFormat = stored.into();
            expect_that!(restored, eq(variant));
        }
    }

    #[gtest]
    fn test_inference_cache_backend_round_trip() {
        for variant in [
            InferenceCacheBackend::Auto,
            InferenceCacheBackend::ClickHouse,
            InferenceCacheBackend::Valkey,
        ] {
            let stored: StoredInferenceCacheBackend = variant.into();
            let restored: InferenceCacheBackend = stored.into();
            expect_that!(restored, eq(variant));
        }
    }

    // ── CredentialLocationWithFallback ─────────────────────────────────

    fn credential_location_variants() -> Vec<CredentialLocation> {
        vec![
            CredentialLocation::Env("MY_KEY".to_string()),
            CredentialLocation::PathFromEnv("MY_KEY_PATH".to_string()),
            CredentialLocation::Dynamic("dyn_key".to_string()),
            CredentialLocation::Path("/etc/keys/key.pem".to_string()),
            CredentialLocation::Sdk,
            CredentialLocation::None,
        ]
    }

    #[gtest]
    fn test_credential_location_with_fallback_single_round_trip() {
        for loc in credential_location_variants() {
            let original = CredentialLocationWithFallback::Single(loc);
            let stored = StoredCredentialLocationWithFallback::from(&original);
            let restored = CredentialLocationWithFallback::from(stored);
            expect_that!(restored, eq(&original));
        }
    }

    #[gtest]
    fn test_credential_location_with_fallback_with_fallback_round_trip() {
        let original = CredentialLocationWithFallback::WithFallback {
            default: CredentialLocation::Env("PRIMARY".to_string()),
            fallback: CredentialLocation::PathFromEnv("BACKUP_PATH".to_string()),
        };
        let stored = StoredCredentialLocationWithFallback::from(&original);
        let restored = CredentialLocationWithFallback::from(stored);
        expect_that!(restored, eq(&original));
    }

    #[gtest]
    fn test_credential_location_with_fallback_all_fallback_combos_round_trip() {
        // Cover the full cross-product of default × fallback to make sure each
        // variant survives the encode/decode through stored form.
        for default in credential_location_variants() {
            for fallback in credential_location_variants() {
                let original = CredentialLocationWithFallback::WithFallback {
                    default: default.clone(),
                    fallback: fallback.clone(),
                };
                let stored = StoredCredentialLocationWithFallback::from(&original);
                let restored = CredentialLocationWithFallback::from(stored);
                expect_that!(restored, eq(&original));
            }
        }
    }

    // ── EndpointLocation ───────────────────────────────────────────────

    #[gtest]
    fn test_endpoint_location_round_trip() {
        use crate::model::EndpointLocation;
        use tensorzero_stored_config::StoredEndpointLocation;
        for variant in [
            EndpointLocation::Env("MY_ENDPOINT".to_string()),
            EndpointLocation::Dynamic("dyn_endpoint".to_string()),
            EndpointLocation::Static("https://api.example.com".to_string()),
        ] {
            let stored = StoredEndpointLocation::from(&variant);
            let restored = EndpointLocation::from(stored);
            expect_that!(restored, eq(&variant));
        }
    }

    // ── UninitializedGatewayConfig full round trip ─────────────────────

    /// Populate every field of `UninitializedGatewayConfig` with a non-default
    /// value and verify that converting to `StoredGatewayConfig` and back is
    /// lossless.
    ///
    /// Fields that are intentionally dropped on the way through storage
    /// (`template_filesystem_access`, `observability.disable_automatic_migrations`,
    /// `batch_writes.__force_allow_embedded_batch_writes`) are set to their
    /// "absent" value so round-tripping still produces an equal struct.
    #[gtest]
    #[expect(deprecated)]
    fn test_uninitialized_gateway_config_round_trip() {
        let original = UninitializedGatewayConfig {
            bind_address: Some("127.0.0.1:8080".parse().unwrap()),
            observability: Some(ObservabilityConfig {
                enabled: Some(true),
                backend: Some(ObservabilityBackend::Postgres),
                async_writes: Some(true),
                batch_writes: Some(BatchWritesConfig {
                    enabled: true,
                    __force_allow_embedded_batch_writes: None,
                    flush_interval_ms: Some(500),
                    max_rows: Some(1000),
                    max_rows_postgres: Some(2000),
                    write_queue_capacity: Some(4096),
                }),
                disable_automatic_migrations: None,
            }),
            debug: Some(true),
            // Not persisted to the stored config — config-in-DB users are banned
            // from setting this field.
            template_filesystem_access: None,
            export: Some(ExportConfig {
                otlp: Some(OtlpConfig {
                    traces: Some(OtlpTracesConfig {
                        enabled: Some(true),
                        format: Some(OtlpTracesFormat::OpenInference),
                        extra_headers: Some(HashMap::from([
                            ("x-trace-header".to_string(), "value-1".to_string()),
                            ("x-other".to_string(), "value-2".to_string()),
                        ])),
                        include_content: None,
                    }),
                }),
            }),
            base_path: Some("/custom/prefix".to_string()),
            unstable_disable_feedback_target_validation: Some(true),
            unstable_error_json: Some(true),
            disable_pseudonymous_usage_analytics: Some(true),
            fetch_and_encode_input_files_before_inference: Some(true),
            auth: Some(AuthConfig {
                enabled: true,
                cache: Some(GatewayAuthCacheConfig {
                    enabled: Some(true),
                    ttl_ms: Some(12_345),
                }),
            }),
            global_outbound_http_timeout_ms: Some(9_876),
            global_outbound_http_intra_stream_read_timeout_ms: Some(1_234),
            relay: Some(UninitializedRelayConfig {
                gateway_url: Some(Url::parse("https://relay.example.com/").unwrap()),
                api_key_location: Some(CredentialLocationWithFallback::WithFallback {
                    default: CredentialLocation::Env("RELAY_KEY".to_string()),
                    fallback: CredentialLocation::Sdk,
                }),
            }),
            metrics: Some(MetricsConfig {
                tensorzero_inference_latency_overhead_seconds_buckets: Some(vec![
                    0.005, 0.05, 0.5, 5.0,
                ]),
            }),
            cache: Some(ModelInferenceCacheConfig {
                enabled: Some(true),
                backend: Some(InferenceCacheBackend::Valkey),
                valkey: Some(ValkeyModelInferenceCacheConfig { ttl_s: 7_200 }),
            }),
        };

        let stored: StoredGatewayConfig = original.clone().into();
        let round_tripped: UninitializedGatewayConfig = stored
            .try_into()
            .expect("StoredGatewayConfig should convert back to UninitializedGatewayConfig");
        expect_that!(round_tripped, eq(&original));
    }
}
