use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct StoredExtraHeadersConfig {
    pub data: Vec<StoredExtraHeader>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredExtraHeader {
    pub name: String,
    pub kind: StoredExtraHeaderKind,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoredExtraHeaderKind {
    Value(String),
    Delete,
}
