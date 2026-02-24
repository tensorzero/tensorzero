use serde::{Deserialize, Serialize};

use crate::config::gateway::{AuthConfig, MetricsConfig, UninitializedGatewayConfig};
use crate::config::{ExportConfig, TemplateFilesystemAccess, UninitializedRelayConfig};

use super::cache_config::StoredCacheConfig;
use super::observability_config::StoredObservabilityConfig;

/// Stored version of `UninitializedGatewayConfig`.
///
/// Omits `deny_unknown_fields` and uses `Stored*` sub-types for nested configs
/// that may gain new fields across versions.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct StoredGatewayConfig {
    pub bind_address: Option<std::net::SocketAddr>,
    #[serde(default)]
    pub observability: StoredObservabilityConfig,
    #[serde(default)]
    pub debug: bool,
    #[serde(default)]
    pub template_filesystem_access: Option<TemplateFilesystemAccess>,
    #[serde(default)]
    pub export: ExportConfig,
    pub base_path: Option<String>,
    #[serde(default)]
    pub unstable_disable_feedback_target_validation: bool,
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
    #[serde(default)]
    pub cache: StoredCacheConfig,
}

impl From<UninitializedGatewayConfig> for StoredGatewayConfig {
    fn from(config: UninitializedGatewayConfig) -> Self {
        let UninitializedGatewayConfig {
            bind_address,
            observability,
            debug,
            template_filesystem_access,
            export,
            base_path,
            unstable_disable_feedback_target_validation,
            unstable_error_json,
            disable_pseudonymous_usage_analytics,
            fetch_and_encode_input_files_before_inference,
            auth,
            global_outbound_http_timeout_ms,
            relay,
            metrics,
            cache,
        } = config;
        Self {
            bind_address,
            observability: observability.into(),
            debug,
            template_filesystem_access,
            export,
            base_path,
            unstable_disable_feedback_target_validation,
            unstable_error_json,
            disable_pseudonymous_usage_analytics,
            fetch_and_encode_input_files_before_inference,
            auth,
            global_outbound_http_timeout_ms,
            relay,
            metrics,
            cache: cache.into(),
        }
    }
}

impl From<StoredGatewayConfig> for UninitializedGatewayConfig {
    fn from(stored: StoredGatewayConfig) -> Self {
        let StoredGatewayConfig {
            bind_address,
            observability,
            debug,
            template_filesystem_access,
            export,
            base_path,
            unstable_disable_feedback_target_validation,
            unstable_error_json,
            disable_pseudonymous_usage_analytics,
            fetch_and_encode_input_files_before_inference,
            auth,
            global_outbound_http_timeout_ms,
            relay,
            metrics,
            cache,
        } = stored;
        Self {
            bind_address,
            observability: observability.into(),
            debug,
            template_filesystem_access,
            export,
            base_path,
            unstable_disable_feedback_target_validation,
            unstable_error_json,
            disable_pseudonymous_usage_analytics,
            fetch_and_encode_input_files_before_inference,
            auth,
            global_outbound_http_timeout_ms,
            relay,
            metrics,
            cache: cache.into(),
        }
    }
}
