use serde::Deserialize;

use crate::variant::VariantConfig;

#[derive(Debug, Deserialize)]
pub struct RejectionSamplingConfig {
    pub weight: f64,
    pub candidates: Vec<String>,
    pub evaluator: EvaluatorConfig,
}

#[derive(Debug, Deserialize)]
pub struct EvaluatorConfig {
    #[serde(flatten)]
    variant: VariantConfig,
}
