use serde::{Deserialize, Serialize};

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredClickHouseConfig {
    pub disable_automatic_migrations: Option<bool>,
}
