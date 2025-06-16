use serde::Deserialize;

use crate::optimization::openai_sft::{OpenAISFTConfig, OpenAISFTJobHandle};
use crate::stored_inference::RenderedStoredInference;

mod openai_sft;
mod providers;

#[derive(Clone, Debug, Deserialize)]
pub enum OptimizerConfig {
    OpenAISFT(OpenAISFTConfig),
}

pub enum OptimizerJobHandle {
    OpenAISFT(OpenAISFTJobHandle),
}

pub enum OptimizerStatus {
    Pending,
    Completed,
    Failed,
}

pub trait Optimizer {
    type JobHandle;

    fn launch(
        &self,
        train_examples: Vec<RenderedStoredInference>,
        val_examples: Option<Vec<RenderedStoredInference>>,
    ) -> Self::JobHandle;

    fn poll(&self, job_handle: Self::JobHandle) -> OptimizerStatus;
}

impl Optimizer for OptimizerConfig {
    type JobHandle = OptimizerJobHandle;
    fn launch(
        &self,
        train_examples: Vec<RenderedStoredInference>,
        val_examples: Option<Vec<RenderedStoredInference>>,
    ) -> Self::JobHandle {
        match self {
            OptimizerConfig::OpenAISFT(config) => {
                OptimizerJobHandle::OpenAISFT(config.launch(train_examples, val_examples))
            }
        }
    }

    fn poll(&self, job_handle: OptimizerJobHandle) -> OptimizerStatus {
        match (self, job_handle) {
            (OptimizerConfig::OpenAISFT(config), OptimizerJobHandle::OpenAISFT(job_handle)) => {
                config.poll(job_handle)
            }
        }
    }
}
