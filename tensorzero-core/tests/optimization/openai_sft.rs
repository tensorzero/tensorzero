use crate::{optimization_test_case, use_mock_inference_provider, OptimizationTestCase};
use tensorzero_core::optimization::{
    openai_sft::UninitializedOpenAISFTConfig, OptimizerInfo, UninitializedOptimizerConfig,
    UninitializedOptimizerInfo,
};

struct OpenAISFTTestCase();

impl OptimizationTestCase for OpenAISFTTestCase {
    fn supports_image_data(&self) -> bool {
        true
    }

    fn supports_tool_calls(&self) -> bool {
        true
    }

    fn get_optimizer_info(&self) -> OptimizerInfo {
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
                api_base: if use_mock_inference_provider() {
                    Some("http://localhost:3030/openai/".parse().unwrap())
                } else {
                    None
                },
            }),
        }
        .load()
        .expect("Failed to create OpenAISFT optimizer info")
    }
}

optimization_test_case!(openai_sft, OpenAISFTTestCase());
