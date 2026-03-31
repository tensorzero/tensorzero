use serde::{Deserialize, Serialize};

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredRateLimitingConfig {
    pub rules: Option<Vec<StoredRateLimitingRule>>,
    pub enabled: Option<bool>,
    pub backend: Option<StoredRateLimitingBackend>,
    pub default_nano_cost: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoredRateLimitingBackend {
    Auto,
    Postgres,
    Valkey,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredRateLimitingRule {
    pub limits: Vec<StoredRateLimit>,
    pub scope: serde_json::Value,
    pub priority: serde_json::Value,
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredRateLimit {
    pub resource: StoredRateLimitResource,
    pub interval: StoredRateLimitInterval,
    pub capacity: u64,
    pub refill_rate: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoredRateLimitResource {
    ModelInference,
    Token,
    Cost,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoredRateLimitInterval {
    Second,
    Minute,
    Hour,
    Day,
    Week,
    Month,
}
