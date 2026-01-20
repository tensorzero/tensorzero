use serde::{Deserialize, Serialize};

/// The ordering direction.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[cfg_attr(feature = "json-schema-bindings", derive(schemars::JsonSchema))]
#[derive(Copy, Clone, Debug, Deserialize, Serialize, PartialEq)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub enum OrderDirection {
    #[serde(rename = "ascending")]
    Asc,
    #[serde(rename = "descending")]
    Desc,
}
