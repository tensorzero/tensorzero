use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::stored_credential_location::StoredCredentialLocationWithFallback;

pub const STORED_GATEWAY_CONFIG_SCHEMA_REVISION: i32 = 1;

// --- Top-level gateway config ---

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredGatewayConfig {
    pub bind_address: Option<String>,
    pub observability: Option<StoredObservabilityConfig>,
    pub debug: Option<bool>,
    // template_filesystem_access is banned for Config-in-Database users. We don't store it, and we will not allow
    // people to store their config in the database if it's currently set to true in toml.
    pub export: Option<StoredExportConfig>,
    pub base_path: Option<String>,
    pub unstable_disable_feedback_target_validation: Option<bool>,
    pub unstable_error_json: Option<bool>,
    pub disable_pseudonymous_usage_analytics: Option<bool>,
    pub fetch_and_encode_input_files_before_inference: Option<bool>,
    pub auth: Option<StoredAuthConfig>,
    pub global_outbound_http_timeout_ms: Option<u64>,
    pub global_outbound_http_intra_stream_read_timeout_ms: Option<u64>,
    pub relay: Option<StoredRelayConfig>,
    pub metrics: Option<StoredGatewayMetricsConfig>,
    pub cache: Option<StoredModelInferenceCacheConfig>,
}

// --- Observability ---

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredObservabilityConfig {
    pub enabled: Option<bool>,
    pub backend: Option<StoredObservabilityBackend>,
    pub async_writes: Option<bool>,
    pub batch_writes: Option<StoredBatchWritesConfig>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StoredObservabilityBackend {
    Auto,
    ClickHouse,
    Postgres,
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredBatchWritesConfig {
    pub enabled: bool,
    pub flush_interval_ms: Option<u64>,
    pub max_rows: Option<usize>,
    pub max_rows_postgres: Option<usize>,
    pub write_queue_capacity: Option<usize>,
}

// --- Export ---

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredExportConfig {
    pub otlp: Option<StoredOtlpConfig>,
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredOtlpConfig {
    pub traces: Option<StoredOtlpTracesConfig>,
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredOtlpTracesConfig {
    pub enabled: Option<bool>,
    pub format: Option<StoredOtlpTracesFormat>,
    pub extra_headers: Option<BTreeMap<String, String>>,
    pub include_content: Option<bool>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StoredOtlpTracesFormat {
    OpenTelemetry,
    OpenInference,
}

// --- Auth ---

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredAuthConfig {
    pub enabled: bool,
    pub cache: Option<StoredGatewayAuthCacheConfig>,
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredGatewayAuthCacheConfig {
    pub enabled: Option<bool>,
    pub ttl_ms: Option<u64>,
}

// --- Relay ---

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredRelayConfig {
    pub gateway_url: Option<String>,
    pub api_key_location: Option<StoredCredentialLocationWithFallback>,
}

// --- Metrics ---

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredGatewayMetricsConfig {
    pub tensorzero_inference_latency_overhead_seconds_buckets: Option<Vec<f64>>,
}

// --- Cache ---

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredModelInferenceCacheConfig {
    pub enabled: Option<bool>,
    pub backend: Option<StoredInferenceCacheBackend>,
    pub valkey: Option<StoredValkeyModelInferenceCacheConfig>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StoredInferenceCacheBackend {
    Auto,
    ClickHouse,
    Valkey,
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredValkeyModelInferenceCacheConfig {
    pub ttl_s: Option<u64>,
}
