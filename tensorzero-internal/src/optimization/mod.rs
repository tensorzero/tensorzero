use crate::optimization::openai_sft::{OpenAISFTConfig, OpenAISFTJobHandle};
use crate::stored_inference::RenderedStoredInference;

mod openai_sft;

pub enum OptimizerConfig {
    OpenAISFT(OpenAISFTConfig),
}

pub enum OptimizerJobHandle {
    OpenAISFT(OpenAISFTJobHandle),
}

pub trait Optimizer {
    fn launch(
        &self,
        train_examples: Vec<RenderedStoredInference>,
        val_examples: Option<Vec<RenderedStoredInference>>,
    );

    fn poll(&self);
}
