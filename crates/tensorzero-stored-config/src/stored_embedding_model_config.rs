use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::stored_cost::StoredUnifiedCostConfig;
use crate::stored_model_config::StoredProviderConfig;
use crate::{StoredExtraBodyConfig, StoredExtraHeadersConfig};

pub const STORED_EMBEDDING_MODEL_CONFIG_SCHEMA_REVISION: i32 = 1;

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredEmbeddingModelConfig {
    pub routing: Vec<String>,
    pub providers: BTreeMap<String, StoredEmbeddingProviderConfig>,
    pub timeout_ms: Option<u64>,
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredEmbeddingProviderConfig {
    pub provider: StoredProviderConfig,
    pub timeout_ms: Option<u64>,
    pub extra_body: Option<StoredExtraBodyConfig>,
    pub extra_headers: Option<StoredExtraHeadersConfig>,
    pub cost: Option<StoredUnifiedCostConfig>,
}
