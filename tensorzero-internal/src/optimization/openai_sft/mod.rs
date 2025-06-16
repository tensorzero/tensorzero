use serde::{Deserialize, Serialize};

use crate::{
    optimization::{Optimizer, OptimizerStatus},
    stored_inference::RenderedStoredInference,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAISFTConfig {
    batch_size: Option<usize>,
    learning_rate_multiplier: Option<f64>,
    n_epochs: Option<usize>,
}

pub struct OpenAISFTJobHandle {
    pub job_id: String,
}

impl Optimizer for OpenAISFTConfig {
    type JobHandle = OpenAISFTJobHandle;

    fn launch(
        &self,
        train_examples: Vec<RenderedStoredInference>,
        val_examples: Option<Vec<RenderedStoredInference>>,
    ) -> Self::JobHandle {
        todo!()
    }

    fn poll(&self, job_handle: Self::JobHandle) -> OptimizerStatus {
        todo!()
    }
}
