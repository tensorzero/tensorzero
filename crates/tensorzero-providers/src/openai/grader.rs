#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tensorzero_derive::TensorZeroDeserialize;

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, JsonSchema, PartialEq, Serialize, TensorZeroDeserialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[cfg_attr(feature = "pyo3", pyclass(str))]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum OpenAIGrader {
    /// Binary string comparison grader (returns 1 for match, 0 for no match)
    StringCheck {
        name: String,
        /// Operation: eq (exact match), ne (not equal), like (contains, case-sensitive), ilike (contains, case-insensitive)
        operation: OpenAIStringCheckOp,
        /// Template to extract value from model output
        input: String,
        /// Expected value to compare against
        reference: String,
    },
    /// Lexical similarity grader using standard NLP metrics
    TextSimilarity {
        name: String,
        /// Metric: bleu, fuzzy_match, gleu, meteor, rouge_1-5, rouge_l
        evaluation_metric: OpenAISimilarityMetric,
        /// Template to extract text from model output
        input: String,
        /// Reference text for similarity comparison
        reference: String,
    },
    /// LLM-based scoring for semantic evaluation
    ScoreModel {
        name: String,
        /// Model for scoring (e.g., "gpt-4o", "o3-mini")
        model: String,
        /// System/user messages defining scoring rubric
        input: Vec<OpenAIModelGraderInput>,
        /// Score range for normalization (e.g., [0.0, 1.0])
        range: Option<[f64; 2]>,
        // sampling_params: Option<Value>, TODO: add this back in
    },
    /// LLM-based classification into predefined categories
    LabelModel {
        name: String,
        /// Model for classification
        model: String,
        /// All possible output labels
        labels: Vec<String>,
        /// Labels considered successful/passing
        passing_labels: Vec<String>,
        /// Messages defining classification criteria
        input: Vec<OpenAIModelGraderInput>,
    },
    /// Custom Python function for domain-specific evaluation
    Python {
        name: String,
        /// Python code implementing custom scoring logic
        source: String,
        /// Optional Docker image for sandboxed execution
        image_tag: Option<String>,
    },
    /// Combines multiple graders with mathematical expressions
    Multi {
        /// Math expression using grader names (e.g., "0.8 * accuracy + 0.2 * fluency")
        /// Supports: +, -, *, /, ^, min, max, abs, floor, ceil, exp, sqrt, log
        calculate_output: String,
        /// Named graders to combine
        graders: HashMap<String, Box<OpenAIGrader>>,
        name: String,
    },
}

impl std::fmt::Display for OpenAIGrader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

#[cfg(feature = "pyo3")]
impl<'py> FromPyObject<'py> for Box<OpenAIGrader> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        Ok(Box::new(OpenAIGrader::extract_bound(ob)?))
    }
}

#[cfg(feature = "pyo3")]
impl<'py> IntoPyObject<'py> for Box<OpenAIGrader> {
    type Target = <OpenAIGrader as IntoPyObject<'py>>::Target;
    type Output = <OpenAIGrader as IntoPyObject<'py>>::Output;
    type Error = <OpenAIGrader as IntoPyObject<'py>>::Error;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        (*self).into_pyobject(py)
    }
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[cfg_attr(feature = "pyo3", pyclass(str, name = "OpenAIStringCheckOp"))]
#[serde(rename_all = "snake_case")]
pub enum OpenAIStringCheckOp {
    Eq,    // equals
    Ne,    // not equals
    Like,  // case-sensitive pattern matching
    Ilike, // case-insensitive pattern matching
}

impl std::fmt::Display for OpenAIStringCheckOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[cfg_attr(feature = "pyo3", pyclass(str, name = "OpenAISimilarityMetric"))]
#[serde(rename_all = "snake_case")]
pub enum OpenAISimilarityMetric {
    FuzzyMatch, // fuzzy_match
    Bleu,       // bleu
    Gleu,       // gleu
    Meteor,     // meteor
    Rouge1,     // rouge_1
    Rouge2,     // rouge_2
    Rouge3,     // rouge_3
    Rouge4,     // rouge_4
    Rouge5,     // rouge_5
    RougeL,     // rouge_l
}

impl std::fmt::Display for OpenAISimilarityMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "pyo3", pyclass)]
pub enum OpenAIRFTRole {
    Developer,
    User,
}

impl std::fmt::Display for OpenAIRFTRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpenAIRFTRole::User => write!(f, "user"),
            OpenAIRFTRole::Developer => write!(f, "developer"),
        }
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl OpenAIRFTRole {
    pub fn __repr__(&self) -> String {
        self.to_string()
    }
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, JsonSchema)]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[cfg_attr(feature = "pyo3", pyclass(str, name = "OpenAIModelGraderInputMessage"))]
pub struct OpenAIModelGraderInput {
    pub role: OpenAIRFTRole,
    pub content: String,
}

impl std::fmt::Display for OpenAIModelGraderInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

// =============================================================================
// Conversions between stored types (`tensorzero-stored-config`) and grader types.
// These live here (rather than in `tensorzero-core`) because both types are
// foreign to `tensorzero-core`.
// =============================================================================

use tensorzero_stored_config::{
    StoredOpenAIGrader, StoredOpenAIModelGraderInput, StoredOpenAIRFTRole,
    StoredOpenAISimilarityMetric, StoredOpenAIStringCheckOp,
};

impl From<StoredOpenAIStringCheckOp> for OpenAIStringCheckOp {
    fn from(stored: StoredOpenAIStringCheckOp) -> Self {
        match stored {
            StoredOpenAIStringCheckOp::Eq => Self::Eq,
            StoredOpenAIStringCheckOp::Ne => Self::Ne,
            StoredOpenAIStringCheckOp::Like => Self::Like,
            StoredOpenAIStringCheckOp::Ilike => Self::Ilike,
        }
    }
}

impl From<StoredOpenAISimilarityMetric> for OpenAISimilarityMetric {
    fn from(stored: StoredOpenAISimilarityMetric) -> Self {
        match stored {
            StoredOpenAISimilarityMetric::FuzzyMatch => Self::FuzzyMatch,
            StoredOpenAISimilarityMetric::Bleu => Self::Bleu,
            StoredOpenAISimilarityMetric::Gleu => Self::Gleu,
            StoredOpenAISimilarityMetric::Meteor => Self::Meteor,
            StoredOpenAISimilarityMetric::Rouge1 => Self::Rouge1,
            StoredOpenAISimilarityMetric::Rouge2 => Self::Rouge2,
            StoredOpenAISimilarityMetric::Rouge3 => Self::Rouge3,
            StoredOpenAISimilarityMetric::Rouge4 => Self::Rouge4,
            StoredOpenAISimilarityMetric::Rouge5 => Self::Rouge5,
            StoredOpenAISimilarityMetric::RougeL => Self::RougeL,
        }
    }
}

impl From<StoredOpenAIRFTRole> for OpenAIRFTRole {
    fn from(stored: StoredOpenAIRFTRole) -> Self {
        match stored {
            StoredOpenAIRFTRole::Developer => Self::Developer,
            StoredOpenAIRFTRole::User => Self::User,
        }
    }
}

impl From<StoredOpenAIModelGraderInput> for OpenAIModelGraderInput {
    fn from(stored: StoredOpenAIModelGraderInput) -> Self {
        Self {
            role: stored.role.into(),
            content: stored.content,
        }
    }
}

impl From<StoredOpenAIGrader> for OpenAIGrader {
    fn from(stored: StoredOpenAIGrader) -> Self {
        match stored {
            StoredOpenAIGrader::StringCheck {
                name,
                operation,
                input,
                reference,
            } => Self::StringCheck {
                name,
                operation: operation.into(),
                input,
                reference,
            },
            StoredOpenAIGrader::TextSimilarity {
                name,
                evaluation_metric,
                input,
                reference,
            } => Self::TextSimilarity {
                name,
                evaluation_metric: evaluation_metric.into(),
                input,
                reference,
            },
            StoredOpenAIGrader::ScoreModel {
                name,
                model,
                input,
                range,
            } => Self::ScoreModel {
                name,
                model,
                input: input.into_iter().map(Into::into).collect(),
                range,
            },
            StoredOpenAIGrader::LabelModel {
                name,
                model,
                labels,
                passing_labels,
                input,
            } => Self::LabelModel {
                name,
                model,
                labels,
                passing_labels,
                input: input.into_iter().map(Into::into).collect(),
            },
            StoredOpenAIGrader::Python {
                name,
                source,
                image_tag,
            } => Self::Python {
                name,
                source,
                image_tag,
            },
            StoredOpenAIGrader::Multi {
                calculate_output,
                graders,
                name,
            } => Self::Multi {
                calculate_output,
                graders: graders
                    .into_iter()
                    .map(|(k, v)| (k, Box::new((*v).into())))
                    .collect(),
                name,
            },
        }
    }
}

impl From<OpenAIStringCheckOp> for StoredOpenAIStringCheckOp {
    fn from(op: OpenAIStringCheckOp) -> Self {
        match op {
            OpenAIStringCheckOp::Eq => Self::Eq,
            OpenAIStringCheckOp::Ne => Self::Ne,
            OpenAIStringCheckOp::Like => Self::Like,
            OpenAIStringCheckOp::Ilike => Self::Ilike,
        }
    }
}

impl From<OpenAISimilarityMetric> for StoredOpenAISimilarityMetric {
    fn from(metric: OpenAISimilarityMetric) -> Self {
        match metric {
            OpenAISimilarityMetric::FuzzyMatch => Self::FuzzyMatch,
            OpenAISimilarityMetric::Bleu => Self::Bleu,
            OpenAISimilarityMetric::Gleu => Self::Gleu,
            OpenAISimilarityMetric::Meteor => Self::Meteor,
            OpenAISimilarityMetric::Rouge1 => Self::Rouge1,
            OpenAISimilarityMetric::Rouge2 => Self::Rouge2,
            OpenAISimilarityMetric::Rouge3 => Self::Rouge3,
            OpenAISimilarityMetric::Rouge4 => Self::Rouge4,
            OpenAISimilarityMetric::Rouge5 => Self::Rouge5,
            OpenAISimilarityMetric::RougeL => Self::RougeL,
        }
    }
}

impl From<OpenAIRFTRole> for StoredOpenAIRFTRole {
    fn from(role: OpenAIRFTRole) -> Self {
        match role {
            OpenAIRFTRole::Developer => Self::Developer,
            OpenAIRFTRole::User => Self::User,
        }
    }
}

impl From<OpenAIModelGraderInput> for StoredOpenAIModelGraderInput {
    fn from(input: OpenAIModelGraderInput) -> Self {
        Self {
            role: input.role.into(),
            content: input.content,
        }
    }
}

impl From<OpenAIGrader> for StoredOpenAIGrader {
    fn from(grader: OpenAIGrader) -> Self {
        match grader {
            OpenAIGrader::StringCheck {
                name,
                operation,
                input,
                reference,
            } => Self::StringCheck {
                name,
                operation: operation.into(),
                input,
                reference,
            },
            OpenAIGrader::TextSimilarity {
                name,
                evaluation_metric,
                input,
                reference,
            } => Self::TextSimilarity {
                name,
                evaluation_metric: evaluation_metric.into(),
                input,
                reference,
            },
            OpenAIGrader::ScoreModel {
                name,
                model,
                input,
                range,
            } => Self::ScoreModel {
                name,
                model,
                input: input.into_iter().map(Into::into).collect(),
                range,
            },
            OpenAIGrader::LabelModel {
                name,
                model,
                labels,
                passing_labels,
                input,
            } => Self::LabelModel {
                name,
                model,
                labels,
                passing_labels,
                input: input.into_iter().map(Into::into).collect(),
            },
            OpenAIGrader::Python {
                name,
                source,
                image_tag,
            } => Self::Python {
                name,
                source,
                image_tag,
            },
            OpenAIGrader::Multi {
                calculate_output,
                graders,
                name,
            } => Self::Multi {
                calculate_output,
                graders: graders
                    .into_iter()
                    .map(|(k, v)| (k, Box::new((*v).into())))
                    .collect(),
                name,
            },
        }
    }
}
