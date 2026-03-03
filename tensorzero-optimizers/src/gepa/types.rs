//! Serializable types for durable GEPA execution.
//!
//! These types support per-step checkpointing so that the durable task framework
//! can resume GEPA optimization from the last completed step.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use tensorzero_core::{
    config::MetricConfigOptimize, optimization::gepa::UninitializedGEPAConfig,
    stored_inference::RenderedSample, variant::chat_completion::UninitializedChatCompletionConfig,
};

use crate::gepa::evaluate::{DatapointId, EvaluatorName, VariantName, VariantScores};

/// Parameters for a durable GEPA optimization task.
///
/// Serialized as JSON and passed to `spawn_tool_by_name("gepa_optimization", ...)`.
/// Only contains LLM-level configuration; infrastructure data (examples) is in `GepaSideInfo`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GepaToolParams {
    pub gepa_config: UninitializedGEPAConfig,
}

/// Side info for durable GEPA tasks.
///
/// Contains data that is passed at spawn time but is not LLM-generated.
/// Uses its own type (not `AutopilotSideInfo`) because GEPA is a standalone
/// tool that bypasses the autopilot result-publishing wrapper.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GepaSideInfo {
    pub train_examples: Vec<RenderedSample>,
    pub val_examples: Vec<RenderedSample>,
}

// Manual JsonSchema impl because RenderedSample doesn't derive it
impl JsonSchema for GepaToolParams {
    fn schema_name() -> Cow<'static, str> {
        Cow::Borrowed("GepaToolParams")
    }

    fn json_schema(_gen: &mut schemars::SchemaGenerator) -> schemars::Schema {
        let mut map = serde_json::Map::new();
        map.insert(
            "type".to_owned(),
            serde_json::Value::String("object".to_owned()),
        );
        map.insert(
            "title".to_owned(),
            serde_json::Value::String("GepaToolParams".to_owned()),
        );
        map.insert(
            "description".to_owned(),
            serde_json::Value::String("Parameters for a durable GEPA optimization task".to_owned()),
        );
        schemars::Schema::from(map)
    }
}

/// Output of a completed GEPA optimization task.
///
/// Stored as the `completed_payload` in the durable task result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GepaToolOutput {
    pub variants: HashMap<String, UninitializedChatCompletionConfig>,
}

/// Serializable snapshot of Pareto frontier state for checkpointing between durable steps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoCheckpoint {
    pub variant_configs: HashMap<VariantName, UninitializedChatCompletionConfig>,
    pub variant_scores_map: HashMap<VariantName, VariantScores>,
    pub variant_frequencies: HashMap<VariantName, usize>,
    pub datapoint_ids: Vec<DatapointId>,
    pub optimize_directions: BTreeMap<EvaluatorName, MetricConfigOptimize>,
    pub seed: Option<u64>,
}

/// Result of the GEPA setup step.
///
/// Contains the initial Pareto frontier checkpoint and all context needed
/// for subsequent iteration and cleanup steps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GepaSetupResult {
    pub checkpoint: ParetoCheckpoint,
    pub train_examples: Vec<RenderedSample>,
    pub max_iterations: u32,
    pub val_dataset_name: String,
    pub temporary_datasets: Vec<String>,
    pub original_variant_names: Vec<String>,
    pub gepa_config: UninitializedGEPAConfig,
    pub per_variant_concurrency: usize,
    pub run_id: Uuid,
}

/// Parameters for a single GEPA iteration step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GepaIterationParams {
    pub checkpoint: ParetoCheckpoint,
    pub train_examples: Vec<RenderedSample>,
    pub iteration: usize,
    pub gepa_config: UninitializedGEPAConfig,
    pub val_dataset_name: String,
    pub temporary_datasets: Vec<String>,
    pub per_variant_concurrency: usize,
    pub run_id: Uuid,
}

/// Result of a single GEPA iteration step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GepaIterationResult {
    pub checkpoint: ParetoCheckpoint,
    pub temporary_datasets: Vec<String>,
}

/// Parameters for the GEPA cleanup step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GepaCleanupParams {
    pub checkpoint: ParetoCheckpoint,
    pub temporary_datasets: Vec<String>,
    pub original_variant_names: Vec<String>,
}
