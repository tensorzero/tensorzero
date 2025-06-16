use chrono::{DateTime, Utc};
use serde::Deserialize;
use serde_json::Value;

use crate::endpoints::inference::InferenceCredentials;
use crate::error::Error;
use crate::optimization::openai_sft::{
    OpenAISFTConfig, OpenAISFTJobHandle, UninitializedOpenAISFTConfig,
};
use crate::stored_inference::RenderedStoredInference;

mod openai_sft;
mod providers;

#[derive(Clone, Debug)]
pub enum OptimizerConfig {
    OpenAISFT(OpenAISFTConfig),
}

pub enum OptimizerJobHandle {
    OpenAISFT(OpenAISFTJobHandle),
}

pub enum OptimizerStatus {
    Pending {
        message: String,
        estimated_finish: Option<DateTime<Utc>>,
        trained_tokens: Option<u64>,
        error: Option<Value>,
    },
    // This should actually contain a variant config
    Completed,
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
        job_handle: Self::JobHandle,
        credentials: &InferenceCredentials,
    ) -> Result<OptimizerStatus, Error>;
}

impl Optimizer for OptimizerConfig {
    type JobHandle = OptimizerJobHandle;
    async fn launch(
        &self,
        client: &reqwest::Client,
        train_examples: Vec<RenderedStoredInference>,
        val_examples: Option<Vec<RenderedStoredInference>>,
        credentials: &InferenceCredentials,
    ) -> Result<Self::JobHandle, Error> {
        match self {
            OptimizerConfig::OpenAISFT(config) => config
                .launch(client, train_examples, val_examples, credentials)
                .await
                .map(OptimizerJobHandle::OpenAISFT),
        }
    }

    async fn poll(
        &self,
        client: &reqwest::Client,
        job_handle: OptimizerJobHandle,
        credentials: &InferenceCredentials,
    ) -> Result<OptimizerStatus, Error> {
        match (self, job_handle) {
            (OptimizerConfig::OpenAISFT(config), OptimizerJobHandle::OpenAISFT(job_handle)) => {
                config.poll(client, job_handle, credentials).await
            }
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub enum UninitializedOptimizerConfig {
    OpenAISFT(UninitializedOpenAISFTConfig),
}

impl UninitializedOptimizerConfig {
    // TODO: add a provider_types argument as needed
    pub fn load(self) -> Result<OptimizerConfig, Error> {
        Ok(match self {
            UninitializedOptimizerConfig::OpenAISFT(config) => {
                OptimizerConfig::OpenAISFT(config.load()?)
            }
        })
    }
}
