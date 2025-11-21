use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// The ordering direction.
#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema, PartialEq, ts_rs::TS)]
#[ts(export)]
pub enum OrderDirection {
    #[serde(rename = "ascending")]
    Asc,
    #[serde(rename = "descending")]
    Desc,
}
