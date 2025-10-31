use serde::{Deserialize, Serialize};

use crate::{
    config::{ExportConfig, ObservabilityConfig, TemplateFilesystemAccess},
    error::Error,
    inference::types::storage::StorageKind,
};

use super::ObjectStoreInfo;

#[derive(Debug, Default, Deserialize, Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
#[serde(deny_unknown_fields)]
pub struct AuthConfig {
    pub enabled: bool,
}

#[derive(Debug, Default, Deserialize, Serialize)]
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
}

impl UninitializedGatewayConfig {
    pub fn load(self, object_store_info: Option<&ObjectStoreInfo>) -> Result<GatewayConfig, Error> {
        let fetch_and_encode_input_files_before_inference = if let Some(value) =
            self.fetch_and_encode_input_files_before_inference
        {
            value
        } else {
            if let Some(info) = object_store_info {
                if !matches!(info.kind, StorageKind::Disabled) {
                    tracing::info!("Object store is enabled but `gateway.fetch_and_encode_input_files_before_inference` is unset (defaults to `false`). In rare cases, the files we fetch for object storage may differ from the image the inference provider fetched if this setting is disabled.");
                }
            }
            default_fetch_and_encode_input_files_before_inference()
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
        })
    }
}

#[derive(Debug, Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
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
    #[serde(default = "default_fetch_and_encode_input_files_before_inference")]
    pub fetch_and_encode_input_files_before_inference: bool,
    pub auth: AuthConfig,
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
            fetch_and_encode_input_files_before_inference:
                default_fetch_and_encode_input_files_before_inference(),
            auth: Default::default(),
        }
    }
}

fn default_fetch_and_encode_input_files_before_inference() -> bool {
    false
}

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
