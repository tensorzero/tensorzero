use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoredMetricType {
    Boolean,
    Float,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoredMetricOptimize {
    Min,
    Max,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoredMetricLevel {
    Inference,
    Episode,
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredMetricConfig {
    pub r#type: StoredMetricType,
    pub optimize: StoredMetricOptimize,
    pub level: StoredMetricLevel,
    pub description: Option<String>,
}
