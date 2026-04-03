use serde::{Deserialize, Serialize};

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredPostgresConfig {
    pub connection_pool_size: Option<u32>,
    pub inference_metadata_retention_days: Option<u32>,
    pub inference_data_retention_days: Option<u32>,
}
