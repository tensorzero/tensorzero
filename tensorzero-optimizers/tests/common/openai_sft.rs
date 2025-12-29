use crate::common::OptimizationTestCase;
use tensorzero_core::optimization::{
    UninitializedOptimizerConfig, UninitializedOptimizerInfo,
    openai_sft::UninitializedOpenAISFTConfig,
};

pub struct OpenAISFTTestCase();

impl OptimizationTestCase for OpenAISFTTestCase {
    fn supports_image_data(&self) -> bool {
        true
    }

    fn supports_tool_calls(&self) -> bool {
        true
    }

    fn get_optimizer_info(&self) -> UninitializedOptimizerInfo {
        // Note: mock mode is configured via provider_types.openai.sft in the test config file
        UninitializedOptimizerInfo {
            inner: UninitializedOptimizerConfig::OpenAISFT(UninitializedOpenAISFTConfig {
                // This is the only model that supports images
                model: "gpt-4o-2024-08-06".to_string(),
                batch_size: None,
                learning_rate_multiplier: None,
                n_epochs: None,
                seed: None,
                suffix: None,
            }),
        }
    }
}
