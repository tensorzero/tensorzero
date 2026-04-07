use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::StoredRetryConfig;

pub const STORED_OPTIMIZER_CONFIG_SCHEMA_REVISION: i32 = 1;

// --- Top-level ---

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StoredOptimizerConfig {
    #[serde(rename = "dicl")]
    Dicl(StoredDiclOptimizationConfig),
    #[serde(rename = "openai_sft")]
    OpenAISFT(StoredOpenAISFTConfig),
    #[serde(rename = "openai_rft")]
    OpenAIRFT(Box<StoredOpenAIRFTConfig>),
    #[serde(rename = "fireworks_sft")]
    FireworksSFT(StoredFireworksOptimizerSFTConfig),
    #[serde(rename = "gcp_vertex_gemini_sft")]
    GCPVertexGeminiSFT(StoredGCPVertexGeminiOptimizerSFTConfig),
    #[serde(rename = "gepa")]
    GEPA(StoredGEPAConfig),
    #[serde(rename = "together_sft")]
    TogetherSFT(Box<StoredTogetherOptimizerSFTConfig>),
}

// --- DICL ---

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredDiclOptimizationConfig {
    pub embedding_model: String,
    pub variant_name: String,
    pub function_name: String,
    pub dimensions: Option<u32>,
    pub batch_size: Option<usize>,
    pub max_concurrency: Option<usize>,
    pub k: Option<u32>,
    pub model: Option<String>,
    pub append_to_existing_variants: Option<bool>,
}

// --- OpenAI SFT ---

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredOpenAISFTConfig {
    pub model: String,
    pub batch_size: Option<usize>,
    pub learning_rate_multiplier: Option<f64>,
    pub n_epochs: Option<usize>,
    pub seed: Option<u64>,
    pub suffix: Option<String>,
}

// --- OpenAI RFT ---

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredOpenAIRFTConfig {
    pub model: String,
    pub grader: StoredOpenAIGrader,
    pub response_format: Option<StoredOpenAIRFTResponseFormat>,
    pub batch_size: Option<usize>,
    pub compute_multiplier: Option<f64>,
    pub eval_interval: Option<usize>,
    pub eval_samples: Option<usize>,
    pub learning_rate_multiplier: Option<f64>,
    pub n_epochs: Option<usize>,
    pub reasoning_effort: Option<String>,
    pub seed: Option<u64>,
    pub suffix: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StoredOpenAIGrader {
    StringCheck {
        name: String,
        operation: StoredOpenAIStringCheckOp,
        input: String,
        reference: String,
    },
    TextSimilarity {
        name: String,
        evaluation_metric: StoredOpenAISimilarityMetric,
        input: String,
        reference: String,
    },
    ScoreModel {
        name: String,
        model: String,
        input: Vec<StoredOpenAIModelGraderInput>,
        range: Option<[f64; 2]>,
    },
    LabelModel {
        name: String,
        model: String,
        labels: Vec<String>,
        passing_labels: Vec<String>,
        input: Vec<StoredOpenAIModelGraderInput>,
    },
    Python {
        name: String,
        source: String,
        image_tag: Option<String>,
    },
    Multi {
        calculate_output: String,
        graders: BTreeMap<String, Box<StoredOpenAIGrader>>,
        name: String,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoredOpenAIStringCheckOp {
    Eq,
    Ne,
    Like,
    Ilike,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoredOpenAISimilarityMetric {
    FuzzyMatch,
    Bleu,
    Gleu,
    Meteor,
    Rouge1,
    Rouge2,
    Rouge3,
    Rouge4,
    Rouge5,
    RougeL,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoredOpenAIRFTRole {
    Developer,
    User,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredOpenAIModelGraderInput {
    pub role: StoredOpenAIRFTRole,
    pub content: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StoredOpenAIRFTResponseFormat {
    JsonSchema {
        json_schema: StoredRFTJsonSchemaInfo,
    },
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredRFTJsonSchemaInfo {
    pub name: String,
    pub description: Option<String>,
    pub schema: Option<Value>,
    pub strict: Option<bool>,
}

// --- Fireworks SFT ---

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredFireworksOptimizerSFTConfig {
    pub model: String,
    pub early_stop: Option<bool>,
    pub epochs: Option<usize>,
    pub learning_rate: Option<f64>,
    pub max_context_length: Option<usize>,
    pub lora_rank: Option<usize>,
    pub batch_size: Option<usize>,
    pub display_name: Option<String>,
    pub output_model: Option<String>,
    pub warm_start_from: Option<String>,
    pub is_turbo: Option<bool>,
    pub eval_auto_carveout: Option<bool>,
    pub nodes: Option<usize>,
    pub mtp_enabled: Option<bool>,
    pub mtp_num_draft_tokens: Option<usize>,
    pub mtp_freeze_base_model: Option<bool>,
    pub deploy_after_training: Option<bool>,
}

// --- GCP Vertex Gemini SFT ---

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredGCPVertexGeminiOptimizerSFTConfig {
    pub model: String,
    pub learning_rate_multiplier: Option<f64>,
    pub adapter_size: Option<usize>,
    pub n_epochs: Option<usize>,
    pub export_last_checkpoint_only: Option<bool>,
    pub seed: Option<u64>,
    pub tuned_model_display_name: Option<String>,
}

// --- GEPA ---

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredGEPAConfig {
    pub function_name: String,
    pub evaluation_name: Option<String>,
    pub evaluator_names: Option<Vec<String>>,
    pub initial_variants: Option<Vec<String>>,
    pub variant_prefix: Option<String>,
    pub batch_size: Option<usize>,
    pub max_iterations: Option<u32>,
    pub max_concurrency: Option<u32>,
    pub analysis_model: String,
    pub mutation_model: String,
    pub seed: Option<u32>,
    pub timeout: Option<u64>,
    pub include_inference_for_mutation: Option<bool>,
    pub retries: Option<StoredRetryConfig>,
    pub max_tokens: Option<u32>,
}

// --- Together SFT ---

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredTogetherOptimizerSFTConfig {
    pub model: String,
    pub n_epochs: Option<u32>,
    pub n_checkpoints: Option<u32>,
    pub n_evals: Option<u32>,
    pub batch_size: Option<StoredTogetherBatchSize>,
    pub learning_rate: Option<f64>,
    pub warmup_ratio: Option<f64>,
    pub max_grad_norm: Option<f64>,
    pub weight_decay: Option<f64>,
    pub suffix: Option<String>,
    pub lr_scheduler: Option<StoredTogetherLRScheduler>,
    pub wandb_name: Option<String>,
    pub training_method: Option<StoredTogetherTrainingMethod>,
    pub training_type: Option<StoredTogetherTrainingType>,
    pub from_checkpoint: Option<String>,
    pub from_hf_model: Option<String>,
    pub hf_model_revision: Option<String>,
    pub hf_output_repo_name: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StoredTogetherBatchSize {
    Number { value: u32 },
    Max,
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StoredTogetherLRScheduler {
    Linear {
        min_lr_ratio: Option<f64>,
    },
    Cosine {
        min_lr_ratio: Option<f64>,
        num_cycles: Option<f64>,
    },
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StoredTogetherTrainingType {
    Full,
    Lora {
        lora_r: Option<u32>,
        lora_alpha: Option<u32>,
        lora_dropout: Option<f64>,
        lora_trainable_modules: Option<String>,
    },
}

#[serde_with::skip_serializing_none]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "method", rename_all = "snake_case")]
pub enum StoredTogetherTrainingMethod {
    #[serde(rename = "sft")]
    Sft { train_on_inputs: Option<String> },
}
