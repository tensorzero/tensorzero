//! Public API types used for the TensorZero evaluations crate.
//! These types are constructed from tensorzero-optimizers, the Python client, and the Node client.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tensorzero_core::{
    cache::CacheEnabledMode,
    client::Client,
    config::UninitializedVariantInfo,
    db::clickhouse::ClickHouseConnectionInfo,
    evaluations::{EvaluationConfig, EvaluationFunctionConfigTable},
};
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::EvaluationUpdate;

/// Specifies which variant to use for evaluation.
/// Either a variant name from the config, or a dynamic variant configuration.
#[derive(Clone, Debug)]
pub enum EvaluationVariant {
    /// Use a variant by name from the config file
    Name(String),
    /// Use a dynamically provided variant configuration
    Info(Box<UninitializedVariantInfo>),
}

/// Parameters for running an evaluation using run_evaluation_core
/// This struct encapsulates all the necessary components for evaluation execution
pub struct EvaluationCoreArgs {
    /// TensorZero client for making inference requests
    pub tensorzero_client: Client,

    /// ClickHouse client for database operations
    pub clickhouse_client: ClickHouseConnectionInfo,

    /// The evaluation configuration (pre-resolved by caller)
    pub evaluation_config: Arc<EvaluationConfig>,

    /// A table of function configurations for output schema validation (pre-resolved by caller)
    /// Maps function name to its minimal evaluation configuration
    pub function_configs: Arc<EvaluationFunctionConfigTable>,

    /// Name of the evaluation (for tagging/logging purposes)
    pub evaluation_name: String,

    /// Unique identifier for this evaluation run
    pub evaluation_run_id: Uuid,

    /// Name of the dataset to run on.
    /// Either dataset_name or datapoint_ids must be provided, but not both.
    pub dataset_name: Option<String>,

    /// Specific datapoint IDs to evaluate.
    /// Either dataset_name or datapoint_ids must be provided, but not both.
    pub datapoint_ids: Option<Vec<Uuid>>,

    /// Variant to use for evaluation.
    /// Either a variant name from the config file, or a dynamic variant configuration.
    pub variant: EvaluationVariant,

    /// Number of concurrent requests to make.
    pub concurrency: usize,

    /// Cache configuration for inference requests
    pub inference_cache: CacheEnabledMode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunInfo {
    pub evaluation_run_id: Uuid,
    pub num_datapoints: usize,
}

/// Result from running an evaluation that supports streaming
pub struct EvaluationStreamResult {
    pub receiver: mpsc::Receiver<EvaluationUpdate>,
    pub run_info: RunInfo,
    pub evaluation_config: Arc<EvaluationConfig>,
}
