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

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GatewayAuthCacheConfig {
    #[serde(default = "default_gateway_auth_cache_enabled")]
    pub enabled: bool,
    #[serde(default = "default_gateway_auth_cache_ttl_ms")]
    pub ttl_ms: u64,
}

impl Default for GatewayAuthCacheConfig {
    fn default() -> Self {
        Self {
            enabled: default_gateway_auth_cache_enabled(),
            ttl_ms: default_gateway_auth_cache_ttl_ms(),
        }
    }
}

fn default_gateway_auth_cache_enabled() -> bool {
    true
}

fn default_gateway_auth_cache_ttl_ms() -> u64 {
    1000
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AuthConfig {
    pub enabled: bool,
    #[serde(default)]
    pub cache: Option<GatewayAuthCacheConfig>,
}

fn default_tensorzero_inference_latency_overhead_seconds_buckets() -> Vec<f64> {
    vec![0.001, 0.01, 0.1]
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MetricsConfig {
    /// Histogram buckets for the `tensorzero_inference_latency_overhead_seconds` metric.
    /// Defaults to `[0.001, 0.01, 0.1]`. Set to empty to disable the metric.
    #[serde(default)]
    pub tensorzero_inference_latency_overhead_seconds_buckets: Option<Vec<f64>>,

    /// DEPRECATED (2026.2+): use `tensorzero_inference_latency_overhead_seconds_buckets` instead.
    #[serde(default, skip_serializing)]
    pub tensorzero_inference_latency_overhead_seconds_histogram_buckets: Option<Vec<f64>>,
}

impl MetricsConfig {
    /// Returns the histogram buckets, handling the deprecated field (2026.2+).
    pub fn get_buckets(&self) -> Vec<f64> {
        self.tensorzero_inference_latency_overhead_seconds_buckets
            .clone()
            .or_else(|| {
                self.tensorzero_inference_latency_overhead_seconds_histogram_buckets
                    .clone()
            })
            .unwrap_or_else(default_tensorzero_inference_latency_overhead_seconds_buckets)
    }

    pub fn validate(&self) -> Result<(), Error> {
        // Handle deprecated field (2026.2+)
        if self
            .tensorzero_inference_latency_overhead_seconds_histogram_buckets
            .is_some()
        {
            if self
                .tensorzero_inference_latency_overhead_seconds_buckets
                .is_some()
            {
                return Err(Error::new(crate::error::ErrorDetails::Config {
                    message: "Cannot set both `gateway.metrics.tensorzero_inference_latency_overhead_seconds_buckets` and deprecated `gateway.metrics.tensorzero_inference_latency_overhead_seconds_histogram_buckets`. Use only the former.".to_string(),
                }));
            }
            crate::utils::deprecation_warning(
                "`gateway.metrics.tensorzero_inference_latency_overhead_seconds_histogram_buckets` is deprecated and will be removed in a future release (2026.2+). Use `gateway.metrics.tensorzero_inference_latency_overhead_seconds_buckets` instead.",
            );
        }

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

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UninitializedGatewayConfig {
    #[serde(serialize_with = "serialize_optional_socket_addr")]
    pub bind_address: Option<std::net::SocketAddr>,
    #[serde(default)]
    pub observability: ObservabilityConfig,
    #[serde(default)]
    pub debug: bool,
    #[serde(default)]
    pub template_filesystem_access: Option<TemplateFilesystemAccess>,
    #[serde(default)]
    pub export: ExportConfig,
    // If set, all of the HTTP endpoints will have this path prepended.
    // E.g. a base path of `/custom/prefix` will cause the inference endpoint to become `/custom/prefix/inference`.
    pub base_path: Option<String>,
    // If set to `true`, disables validation on feedback queries (read from ClickHouse to check that the target is valid)
    #[serde(default)]
    pub unstable_disable_feedback_target_validation: bool,
    /// If enabled, adds an `error_json` field alongside the human-readable `error` field
    /// in HTTP error responses. This contains a JSON-serialized version of the error.
    /// While `error_json` will always be valid JSON when present, the exact contents is unstable,
    /// and may change at any time without warning.
    /// For now, this is only supported in the standalone gateway, and not in the embedded gateway.
    #[serde(default)]
    pub unstable_error_json: bool,
    #[serde(default)]
    pub disable_pseudonymous_usage_analytics: bool,
    pub fetch_and_encode_input_files_before_inference: Option<bool>,
    #[serde(default)]
    pub auth: AuthConfig,
    pub global_outbound_http_timeout_ms: Option<u64>,
    #[serde(default)]
    pub relay: Option<UninitializedRelayConfig>,
    #[serde(default)]
    pub metrics: MetricsConfig,
}

impl UninitializedGatewayConfig {
    pub fn load(self, object_store_info: Option<&ObjectStoreInfo>) -> Result<GatewayConfig, Error> {
        self.metrics.validate()?;
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
            observability: self.observability,
            debug: self.debug,
            template_filesystem_access: self.template_filesystem_access.unwrap_or_default(),
            export: self.export,
            base_path: self.base_path,
            unstable_error_json: self.unstable_error_json,
            unstable_disable_feedback_target_validation: self
                .unstable_disable_feedback_target_validation,
            disable_pseudonymous_usage_analytics: self.disable_pseudonymous_usage_analytics,
            fetch_and_encode_input_files_before_inference,
            auth: self.auth,
            global_outbound_http_timeout: self
                .global_outbound_http_timeout_ms
                .map(|ms| Duration::milliseconds(ms as i64))
                .unwrap_or(DEFAULT_HTTP_CLIENT_TIMEOUT),
            relay,
            metrics: self.metrics,
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
