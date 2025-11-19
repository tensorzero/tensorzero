use crate::config::UninitializedVariantConfig;
#[cfg(feature = "pyo3")]
use crate::inference::types::pyo3_helpers::serialize_to_dict;
use crate::model_table::ProviderTypeDefaultCredentials;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use chrono::{DateTime, Utc};
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

use crate::error::{Error, ErrorDetails};
use crate::model::UninitializedModelConfig;
use crate::optimization::dicl::{
    DiclOptimizationConfig, DiclOptimizationJobHandle, UninitializedDiclOptimizationConfig,
};
use crate::optimization::fireworks_sft::{
    FireworksSFTConfig, FireworksSFTJobHandle, UninitializedFireworksSFTConfig,
};
use crate::optimization::gcp_vertex_gemini_sft::{
    GCPVertexGeminiSFTConfig, GCPVertexGeminiSFTJobHandle, UninitializedGCPVertexGeminiSFTConfig,
};
use crate::optimization::gepa::{GEPAConfig, GEPAJobHandle, UninitializedGEPAConfig};
use crate::optimization::openai_rft::{
    OpenAIRFTConfig, OpenAIRFTJobHandle, UninitializedOpenAIRFTConfig,
};
use crate::optimization::openai_sft::{
    OpenAISFTConfig, OpenAISFTJobHandle, UninitializedOpenAISFTConfig,
};
use crate::optimization::together_sft::{
    TogetherSFTConfig, TogetherSFTJobHandle, UninitializedTogetherSFTConfig,
};

pub mod dicl;
pub mod fireworks_sft;
pub mod gcp_vertex_gemini_sft;
pub mod gepa;
pub mod openai_rft;
pub mod openai_sft;
pub mod together_sft;

#[derive(Clone, Debug, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct OptimizerInfo {
    pub inner: OptimizerConfig,
}

#[derive(Clone, Debug, Serialize, ts_rs::TS)]
#[ts(export)]
pub enum OptimizerConfig {
    Dicl(DiclOptimizationConfig),
    OpenAISFT(OpenAISFTConfig),
    OpenAIRFT(Box<OpenAIRFTConfig>),
    FireworksSFT(FireworksSFTConfig),
    GCPVertexGeminiSFT(Box<GCPVertexGeminiSFTConfig>),
    GEPA(GEPAConfig),
    TogetherSFT(Box<TogetherSFTConfig>),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
#[cfg_attr(feature = "pyo3", pyclass(str))]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OptimizationJobHandle {
    #[serde(rename = "dicl")]
    Dicl(DiclOptimizationJobHandle),
    #[serde(rename = "openai_sft")]
    OpenAISFT(OpenAISFTJobHandle),
    #[serde(rename = "openai_rft")]
    OpenAIRFT(OpenAIRFTJobHandle),
    #[serde(rename = "fireworks_sft")]
    FireworksSFT(FireworksSFTJobHandle),
    #[serde(rename = "gcp_vertex_gemini_sft")]
    GCPVertexGeminiSFT(GCPVertexGeminiSFTJobHandle),
    #[serde(rename = "gepa")]
    GEPA(GEPAJobHandle),
    #[serde(rename = "together_sft")]
    TogetherSFT(TogetherSFTJobHandle),
}

impl OptimizationJobHandle {
    pub fn to_base64_urlencoded(&self) -> Result<String, Error> {
        let serialized_job_handle = serde_json::to_string(self).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Failed to serialize job handle: {e}"),
            })
        })?;
        Ok(URL_SAFE_NO_PAD.encode(serialized_job_handle.as_bytes()))
    }

    pub fn from_base64_urlencoded(encoded_job_handle: &str) -> Result<Self, Error> {
        let decoded_job_handle = URL_SAFE_NO_PAD
            .decode(encoded_job_handle.as_bytes())
            .map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: format!("Failed to deserialize job handle: {e}"),
                })
            })?;
        let job_handle = serde_json::from_slice(&decoded_job_handle).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Failed to deserialize job handle: {e}"),
            })
        })?;
        Ok(job_handle)
    }
}

impl std::fmt::Display for OptimizationJobHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

#[derive(ts_rs::TS, Debug, Deserialize, Serialize)]
#[ts(export)]
#[serde(tag = "type", content = "content", rename_all = "snake_case")]
pub enum OptimizerOutput {
    Variant(Box<UninitializedVariantConfig>),
    Variants(HashMap<String, Box<UninitializedVariantConfig>>),
    Model(UninitializedModelConfig),
}

#[derive(Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum OptimizationJobInfo {
    Pending {
        message: String,
        #[ts(type = "Date | null")]
        estimated_finish: Option<DateTime<Utc>>,
        trained_tokens: Option<u64>,
        error: Option<Value>,
    },
    Completed {
        output: OptimizerOutput,
    },
    Failed {
        message: String,
        error: Option<Value>,
    },
}

/// PyO3 has special handling for complex enums that makes it difficult to #[pyclass] them directly if
/// they contain elements that don't implement IntoPyObject.
/// We work around this by implementing a custom pyclass that wraps the OptimizationJobInfo enum.
#[cfg_attr(feature = "pyo3", pyclass(str, name = "OptimizationJobInfo"))]
pub struct OptimizationJobInfoPyClass(OptimizationJobInfo);

#[cfg(feature = "pyo3")]
impl OptimizationJobInfoPyClass {
    pub fn new(status: OptimizationJobInfo) -> Self {
        Self(status)
    }
}

impl std::fmt::Display for OptimizationJobInfoPyClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(&self.0).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

#[derive(Debug, Serialize, PartialEq)]
#[cfg_attr(feature = "pyo3", pyclass(str, eq))]
pub enum OptimizationJobStatus {
    Pending,
    Completed,
    Failed,
}

impl std::fmt::Display for OptimizationJobStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl OptimizationJobInfoPyClass {
    #[getter]
    fn get_message(&self) -> &str {
        match &self.0 {
            OptimizationJobInfo::Pending { message, .. } => message,
            OptimizationJobInfo::Completed { .. } => "Completed",
            OptimizationJobInfo::Failed { message, .. } => message,
        }
    }

    #[getter]
    fn get_status(&self) -> OptimizationJobStatus {
        match &self.0 {
            OptimizationJobInfo::Pending { .. } => OptimizationJobStatus::Pending,
            OptimizationJobInfo::Completed { .. } => OptimizationJobStatus::Completed,
            OptimizationJobInfo::Failed { .. } => OptimizationJobStatus::Failed,
        }
    }

    #[getter]
    fn get_output<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        match &self.0 {
            OptimizationJobInfo::Completed { output } => Ok(serialize_to_dict(py, output)
                .map(|obj| Some(obj.into_pyobject(py)))?
                .transpose()?),
            _ => Ok(None),
        }
    }

    /// Returns the estimated finish time in seconds since the Unix epoch.
    #[getter]
    fn get_estimated_finish(&self) -> Option<i64> {
        match &self.0 {
            OptimizationJobInfo::Pending {
                estimated_finish, ..
            } => estimated_finish.map(|dt| dt.timestamp()),
            _ => None,
        }
    }
}

#[derive(ts_rs::TS, Clone, Debug, Deserialize, Serialize)]
#[ts(export)]
pub struct UninitializedOptimizerInfo {
    #[serde(flatten)]
    pub inner: UninitializedOptimizerConfig,
}

impl UninitializedOptimizerInfo {
    pub async fn load(
        self,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<OptimizerInfo, Error> {
        Ok(OptimizerInfo {
            inner: self.inner.load(default_credentials).await?,
        })
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum UninitializedOptimizerConfig {
    #[serde(rename = "dicl")]
    Dicl(UninitializedDiclOptimizationConfig),
    #[serde(rename = "openai_sft")]
    OpenAISFT(UninitializedOpenAISFTConfig),
    #[serde(rename = "openai_rft")]
    OpenAIRFT(UninitializedOpenAIRFTConfig),
    #[serde(rename = "fireworks_sft")]
    FireworksSFT(UninitializedFireworksSFTConfig),
    #[serde(rename = "gcp_vertex_gemini_sft")]
    GCPVertexGeminiSFT(UninitializedGCPVertexGeminiSFTConfig),
    #[serde(rename = "gepa")]
    GEPA(UninitializedGEPAConfig),
    #[serde(rename = "together_sft")]
    TogetherSFT(Box<UninitializedTogetherSFTConfig>),
}

impl UninitializedOptimizerConfig {
    async fn load(
        self,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<OptimizerConfig, Error> {
        Ok(match self {
            UninitializedOptimizerConfig::Dicl(config) => {
                OptimizerConfig::Dicl(config.load(default_credentials).await?)
            }
            UninitializedOptimizerConfig::OpenAISFT(config) => {
                OptimizerConfig::OpenAISFT(config.load(default_credentials).await?)
            }
            UninitializedOptimizerConfig::OpenAIRFT(config) => {
                OptimizerConfig::OpenAIRFT(Box::new(config.load(default_credentials).await?))
            }
            UninitializedOptimizerConfig::FireworksSFT(config) => {
                OptimizerConfig::FireworksSFT(config.load(default_credentials).await?)
            }
            UninitializedOptimizerConfig::GCPVertexGeminiSFT(config) => {
                OptimizerConfig::GCPVertexGeminiSFT(Box::new(
                    config.load(default_credentials).await?,
                ))
            }
            UninitializedOptimizerConfig::GEPA(config) => {
                OptimizerConfig::GEPA(config.load(default_credentials).await?)
            }
            UninitializedOptimizerConfig::TogetherSFT(config) => {
                OptimizerConfig::TogetherSFT(Box::new(config.load(default_credentials).await?))
            }
        })
    }
}
