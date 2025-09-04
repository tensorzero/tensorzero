use crate::common::{mock_inference_provider_base, OptimizationTestCase};
use tensorzero_core::optimization::{
    openai_sft::UninitializedOpenAISFTConfig, UninitializedOptimizerConfig,
    UninitializedOptimizerInfo,
};

pub struct OpenAISFTTestCase();

impl OptimizationTestCase for OpenAISFTTestCase {
    fn supports_image_data(&self) -> bool {
        true
    }

    fn supports_tool_calls(&self) -> bool {
        true
    }

    fn get_optimizer_info(&self, use_mock_inference_provider: bool) -> UninitializedOptimizerInfo {
        UninitializedOptimizerInfo {
            inner: UninitializedOptimizerConfig::OpenAISFT(UninitializedOpenAISFTConfig {
                // This is the only model that supports images
                model: "gpt-4o-2024-08-06".to_string(),
                batch_size: None,
                learning_rate_multiplier: None,
                n_epochs: None,
                credentials: None,
                seed: None,
                suffix: None,
                api_base: if use_mock_inference_provider {
                    Some(mock_inference_provider_base().join("openai/").unwrap())
                } else {
                    None
                },
            }),
        }
    }
}
