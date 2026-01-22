use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// The ordering direction.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Copy, Clone, Debug, Deserialize, Serialize, JsonSchema, PartialEq)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub enum OrderDirection {
    #[serde(rename = "ascending")]
    Asc,
    #[serde(rename = "descending")]
    Desc,
}
