use crate::{optimization_test_case, OptimizationTestCase};
use tensorzero_internal::optimization::{
    openai_sft::UninitializedOpenAISFTConfig,
    OptimizerConfig,
};

struct OpenAISFTTestCase();

impl OptimizationTestCase for OpenAISFTTestCase {
    fn supports_image_data(&self) -> bool {
        true
    }

    fn supports_tool_calls(&self) -> bool {
        true
    }

    fn get_optimizer_config(&self) -> OptimizerConfig {
        OptimizerConfig::OpenAISFT(
            UninitializedOpenAISFTConfig {
                // This is the only model that supports images
                model: "gpt-4o-2024-08-06".to_string(),
                batch_size: None,
                learning_rate_multiplier: None,
                n_epochs: None,
                credentials: None,
                seed: None,
                suffix: None,
                api_base: None,
            }
            .load()
            .unwrap(),
        )
    }
}

optimization_test_case!(openai_sft, OpenAISFTTestCase());
