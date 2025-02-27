use std::collections::HashMap;

use serde::Deserialize;

use crate::{config_parser::UninitializedVariantConfig, variant::VariantConfig};

pub const LLM_JUDGE_SYSTEM_SCHEMA_TEXT: &str = include_str!("llm_judge_system_schema.json");
pub const LLM_JUDGE_USER_SCHEMA_TEXT: &str = include_str!("llm_judge_user_schema.json");
pub const LLM_JUDGE_FLOAT_OUTPUT_SCHEMA_TEXT: &str =
    include_str!("llm_judge_float_output_schema.json");
pub const LLM_JUDGE_BOOLEAN_OUTPUT_SCHEMA_TEXT: &str =
    include_str!("llm_judge_boolean_output_schema.json");

pub struct EvalConfig {
    pub evaluators: HashMap<String, EvaluatorConfig>,
    pub dataset_name: String,
    pub function_name: String,
}

pub enum EvaluatorConfig {
    ExactMatch,
    LLMJudge(LLMJudgeConfig),
}

pub struct LLMJudgeConfig {
    pub output_type: LLMJudgeOutputType,
    pub include_datapoint_output: bool,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LLMJudgeOutputType {
    Float,
    Boolean,
}
