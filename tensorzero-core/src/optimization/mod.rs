#[cfg(feature = "pyo3")]
use crate::inference::types::pyo3_helpers::serialize_to_dict;
use chrono::{DateTime, Utc};
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::endpoints::inference::InferenceCredentials;
use crate::error::Error;
use crate::model::UninitializedModelConfig;
use crate::optimization::fireworks_sft::{
    FireworksSFTConfig, FireworksSFTJobHandle, UninitializedFireworksSFTConfig,
};
use crate::optimization::openai_sft::{
    OpenAISFTConfig, OpenAISFTJobHandle, UninitializedOpenAISFTConfig,
};
use crate::stored_inference::RenderedSample;
use crate::variant::VariantConfig;

pub mod fireworks_sft;
pub mod openai_sft;

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct OptimizerInfo {
    inner: OptimizerConfig,
}

impl OptimizerInfo {
    pub fn new(uninitialized_info: UninitializedOptimizerInfo) -> Result<Self, Error> {
        Ok(Self {
            inner: uninitialized_info.inner.load()?,
        })
    }
}

#[derive(Clone, Debug, Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
enum OptimizerConfig {
    OpenAISFT(OpenAISFTConfig),
    FireworksSFT(FireworksSFTConfig),
}

#[cfg_attr(test, derive(ts_rs::TS))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(test, ts(export))]
#[cfg_attr(feature = "pyo3", pyclass(str))]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OptimizationJobHandle {
    #[serde(rename = "openai_sft")]
    OpenAISFT(OpenAISFTJobHandle),
    #[serde(rename = "fireworks_sft")]
    FireworksSFT(FireworksSFTJobHandle),
}

impl std::fmt::Display for OptimizationJobHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

impl JobHandle for OptimizationJobHandle {
    async fn poll(
        &self,
        client: &reqwest::Client,
        credentials: &InferenceCredentials,
    ) -> Result<OptimizationJobInfo, Error> {
        match self {
            OptimizationJobHandle::OpenAISFT(job_handle) => {
                job_handle.poll(client, credentials).await
            }
            OptimizationJobHandle::FireworksSFT(job_handle) => {
                job_handle.poll(client, credentials).await
            }
        }
    }
}

#[cfg_attr(test, derive(ts_rs::TS))]
#[derive(Debug, Serialize)]
#[cfg_attr(test, ts(export))]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OptimizerOutput {
    Variant(Box<VariantConfig>),
    Model(UninitializedModelConfig),
}

#[cfg_attr(test, derive(ts_rs::TS))]
#[derive(Debug, Serialize)]
#[cfg_attr(test, ts(export))]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum OptimizationJobInfo {
    Pending {
        message: String,
        #[cfg_attr(test, ts(type = "Date | null"))]
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

pub trait JobHandle {
    async fn poll(
        &self,
        client: &reqwest::Client,
        credentials: &InferenceCredentials,
    ) -> Result<OptimizationJobInfo, Error>;
}

pub trait Optimizer {
    type Handle: JobHandle;

    async fn launch(
        &self,
        client: &reqwest::Client,
        train_examples: Vec<RenderedSample>,
        val_examples: Option<Vec<RenderedSample>>,
        credentials: &InferenceCredentials,
    ) -> Result<Self::Handle, Error>;
}

impl Optimizer for OptimizerInfo {
    type Handle = OptimizationJobHandle;
    async fn launch(
        &self,
        client: &reqwest::Client,
        train_examples: Vec<RenderedSample>,
        val_examples: Option<Vec<RenderedSample>>,
        credentials: &InferenceCredentials,
    ) -> Result<Self::Handle, Error> {
        match &self.inner {
            OptimizerConfig::OpenAISFT(config) => config
                .launch(client, train_examples, val_examples, credentials)
                .await
                .map(OptimizationJobHandle::OpenAISFT),
            OptimizerConfig::FireworksSFT(config) => config
                .launch(client, train_examples, val_examples, credentials)
                .await
                .map(OptimizationJobHandle::FireworksSFT),
        }
    }
}

#[cfg_attr(test, derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize)]
#[cfg_attr(test, ts(export))]
pub struct UninitializedOptimizerInfo {
    #[serde(flatten)]
    pub inner: UninitializedOptimizerConfig,
}

impl UninitializedOptimizerInfo {
    pub fn load(self) -> Result<OptimizerInfo, Error> {
        Ok(OptimizerInfo {
            inner: self.inner.load()?,
        })
    }
}

#[cfg_attr(test, derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize)]
#[cfg_attr(test, ts(export))]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum UninitializedOptimizerConfig {
    #[serde(rename = "openai_sft")]
    OpenAISFT(UninitializedOpenAISFTConfig),
    #[serde(rename = "fireworks_sft")]
    FireworksSFT(UninitializedFireworksSFTConfig),
}

impl UninitializedOptimizerConfig {
    // TODO: add a provider_types argument as needed
    fn load(self) -> Result<OptimizerConfig, Error> {
        Ok(match self {
            UninitializedOptimizerConfig::OpenAISFT(config) => {
                OptimizerConfig::OpenAISFT(config.load()?)
            }
            UninitializedOptimizerConfig::FireworksSFT(config) => {
                OptimizerConfig::FireworksSFT(config.load()?)
            }
        })
    }
}
