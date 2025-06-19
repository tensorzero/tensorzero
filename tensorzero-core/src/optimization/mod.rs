use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use ts_rs::TS;

use crate::endpoints::inference::InferenceCredentials;
use crate::error::Error;
use crate::model::ModelConfig;
use crate::optimization::openai_sft::{
    OpenAISFTConfig, OpenAISFTJobHandle, UninitializedOpenAISFTConfig,
};
use crate::stored_inference::RenderedStoredInference;
use crate::variant::VariantConfig;

pub mod openai_sft;

#[derive(Clone, Debug)]
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

#[derive(Clone, Debug)]
enum OptimizerConfig {
    OpenAISFT(OpenAISFTConfig),
}

#[derive(Debug, PartialEq, Serialize, Deserialize, TS)]
#[ts(export)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OptimizerJobHandle {
    #[serde(rename = "openai_sft")]
    OpenAISFT(OpenAISFTJobHandle),
}

#[derive(Debug)]
pub enum OptimizerOutput {
    Variant(Box<VariantConfig>),
    Model(ModelConfig),
}

#[derive(Debug)]
pub enum OptimizerStatus {
    Pending {
        message: String,
        estimated_finish: Option<DateTime<Utc>>,
        trained_tokens: Option<u64>,
        error: Option<Value>,
    },
    Completed(OptimizerOutput),
    Failed,
}

pub trait Optimizer {
    type JobHandle;

    async fn launch(
        &self,
        client: &reqwest::Client,
        train_examples: Vec<RenderedStoredInference>,
        val_examples: Option<Vec<RenderedStoredInference>>,
        credentials: &InferenceCredentials,
    ) -> Result<Self::JobHandle, Error>;

    async fn poll(
        &self,
        client: &reqwest::Client,
        job_handle: &Self::JobHandle,
        credentials: &InferenceCredentials,
    ) -> Result<OptimizerStatus, Error>;
}

impl Optimizer for OptimizerInfo {
    type JobHandle = OptimizerJobHandle;
    async fn launch(
        &self,
        client: &reqwest::Client,
        train_examples: Vec<RenderedStoredInference>,
        val_examples: Option<Vec<RenderedStoredInference>>,
        credentials: &InferenceCredentials,
    ) -> Result<Self::JobHandle, Error> {
        match &self.inner {
            OptimizerConfig::OpenAISFT(config) => config
                .launch(client, train_examples, val_examples, credentials)
                .await
                .map(OptimizerJobHandle::OpenAISFT),
        }
    }

    async fn poll(
        &self,
        client: &reqwest::Client,
        job_handle: &OptimizerJobHandle,
        credentials: &InferenceCredentials,
    ) -> Result<OptimizerStatus, Error> {
        match (&self.inner, job_handle) {
            (OptimizerConfig::OpenAISFT(config), OptimizerJobHandle::OpenAISFT(job_handle)) => {
                config.poll(client, job_handle, credentials).await
            }
        }
    }
}

#[derive(Clone, Debug, Deserialize, TS)]
#[ts(export)]
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

    pub fn load_from_default_optimizer(
        job_handle: &OptimizerJobHandle,
    ) -> Result<OptimizerInfo, Error> {
        Ok(OptimizerInfo {
            inner: UninitializedOptimizerConfig::load_from_default_optimizer(job_handle)?,
        })
    }
}

#[derive(Clone, Debug, Deserialize, TS)]
#[ts(export)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum UninitializedOptimizerConfig {
    #[serde(rename = "openai_sft")]
    OpenAISFT(UninitializedOpenAISFTConfig),
}

impl UninitializedOptimizerConfig {
    // TODO: add a provider_types argument as needed
    fn load(self) -> Result<OptimizerConfig, Error> {
        Ok(match self {
            UninitializedOptimizerConfig::OpenAISFT(config) => {
                OptimizerConfig::OpenAISFT(config.load()?)
            }
        })
    }

    fn load_from_default_optimizer(
        job_handle: &OptimizerJobHandle,
    ) -> Result<OptimizerConfig, Error> {
        match job_handle {
            OptimizerJobHandle::OpenAISFT(_) => Ok(OptimizerConfig::OpenAISFT(
                UninitializedOpenAISFTConfig::default().load()?,
            )),
        }
    }
}
