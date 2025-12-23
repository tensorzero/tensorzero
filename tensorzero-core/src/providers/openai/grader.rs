#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
#[cfg_attr(feature = "pyo3", pyclass(str))]
#[serde(tag = "type", rename_all = "snake_case")]
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

#[derive(Clone, Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
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

#[derive(Clone, Debug, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
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

#[derive(ts_rs::TS, Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
#[ts(export)]
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

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, ts_rs::TS)]
#[ts(export)]
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
