use chrono::Duration;
use serde::{Deserialize, Serialize};

use crate::model::{CredentialLocation, CredentialLocationWithFallback};
use crate::model_table::load_tensorzero_relay_credential;
use crate::relay::RelayCredentials;
use crate::{
    config::{
        ExportConfig, ObservabilityConfig, TemplateFilesystemAccess, UninitializedRelayConfig,
    },
    error::Error,
    http::DEFAULT_HTTP_CLIENT_TIMEOUT,
    inference::types::storage::StorageKind,
    relay::TensorzeroRelay,
};

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
