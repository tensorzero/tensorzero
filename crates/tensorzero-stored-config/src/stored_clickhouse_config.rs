use serde::{Deserialize, Serialize};

pub const STORED_CLICKHOUSE_CONFIG_SCHEMA_REVISION: i32 = 1;

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredClickHouseConfig {
    pub disable_automatic_migrations: Option<bool>,
}
