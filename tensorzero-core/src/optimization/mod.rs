use chrono::{DateTime, Utc};
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
pub mod gcp_vertex_gemini_sft;
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
#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(test, ts(export))]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OptimizerJobHandle {
    #[serde(rename = "openai_sft")]
    OpenAISFT(OpenAISFTJobHandle),
    #[serde(rename = "fireworks_sft")]
    FireworksSFT(FireworksSFTJobHandle),
}

impl JobHandle for OptimizerJobHandle {
    async fn poll(
        &self,
        client: &reqwest::Client,
        credentials: &InferenceCredentials,
    ) -> Result<OptimizerStatus, Error> {
        match self {
            OptimizerJobHandle::OpenAISFT(job_handle) => job_handle.poll(client, credentials).await,
            OptimizerJobHandle::FireworksSFT(job_handle) => {
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
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OptimizerStatus {
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

pub trait JobHandle {
    async fn poll(
        &self,
        client: &reqwest::Client,
        credentials: &InferenceCredentials,
    ) -> Result<OptimizerStatus, Error>;
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
    type Handle = OptimizerJobHandle;
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
                .map(OptimizerJobHandle::OpenAISFT),
            OptimizerConfig::FireworksSFT(config) => config
                .launch(client, train_examples, val_examples, credentials)
                .await
                .map(OptimizerJobHandle::FireworksSFT),
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
