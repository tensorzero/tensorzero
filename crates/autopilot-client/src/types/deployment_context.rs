use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentContext {
    pub functions: Vec<DeploymentContextFunction>,
    pub metrics: Vec<DeploymentContextMetric>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evaluations: Vec<DeploymentContextEvaluation>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub feedback_by_function: Vec<DeploymentContextFunctionFeedback>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub datasets: Vec<DeploymentContextDataset>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inference_counts_by_function: Vec<DeploymentContextFunctionInferenceCount>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentContextFunction {
    pub name: String,
    pub r#type: String,
    pub variants: Vec<DeploymentContextVariant>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentContextVariant {
    pub name: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub model_names: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentContextMetric {
    pub name: String,
    pub r#type: String,
    pub optimize: String,
    pub level: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentContextEvaluation {
    pub name: String,
    pub function_name: String,
    pub evaluator_names: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentContextFunctionFeedback {
    pub function_name: String,
    pub metrics: Vec<DeploymentContextMetricFeedback>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentContextMetricFeedback {
    pub metric_name: String,
    pub variants: Vec<DeploymentContextVariantFeedback>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentContextVariantFeedback {
    pub variant_name: String,
    pub mean: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub variance: Option<f32>,
    pub count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentContextDataset {
    pub name: String,
    pub count: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentContextFunctionInferenceCount {
    pub function_name: String,
    pub inference_count: u32,
    pub last_inference_timestamp: DateTime<Utc>,
}
