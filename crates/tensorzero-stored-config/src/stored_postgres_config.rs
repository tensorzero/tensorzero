use serde::{Deserialize, Serialize};

pub const STORED_POSTGRES_CONFIG_SCHEMA_REVISION: i32 = 1;

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredPostgresConfig {
    pub connection_pool_size: Option<u32>,
    pub inference_metadata_retention_days: Option<u32>,
    pub inference_data_retention_days: Option<u32>,
}
