use crate::common::OptimizationTestCase;
use tensorzero_core::optimization::{
    together_sft::UninitializedTogetherSFTConfig, UninitializedOptimizerConfig,
    UninitializedOptimizerInfo,
};

pub struct TogetherSFTTestCase();

impl OptimizationTestCase for TogetherSFTTestCase {
    fn supports_image_data(&self) -> bool {
        false
    }

    fn supports_tool_calls(&self) -> bool {
        false
    }

    fn get_optimizer_info(&self, use_mock_inference_provider: bool) -> UninitializedOptimizerInfo {
        UninitializedOptimizerInfo {
            inner: UninitializedOptimizerConfig::TogetherSFT(UninitializedTogetherSFTConfig {
                model: "meta-llama/Meta-Llama-3.1-8B-Instruct-Reference".to_string(),
                credentials: None,
                api_base: if use_mock_inference_provider {
                    Some("http://localhost:3030/together/".parse().unwrap())
                } else {
                    None
                },
            }),
        }
    }
}
