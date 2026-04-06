use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct StoredExtraBodyConfig {
    pub data: Vec<StoredExtraBodyReplacement>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredExtraBodyReplacement {
    pub pointer: String,
    pub kind: StoredExtraBodyReplacementKind,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoredExtraBodyReplacementKind {
    Value(serde_json::Value),
    Delete,
}
