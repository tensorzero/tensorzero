use crate::{optimization_test_case, OptimizationTestCase};
use tensorzero_core::optimization::{
    together_sft::UninitializedTogetherSFTConfig, OptimizerInfo, UninitializedOptimizerConfig,
    UninitializedOptimizerInfo,
};

struct TogetherSFTTestCase();

impl OptimizationTestCase for TogetherSFTTestCase {
    fn supports_image_data(&self) -> bool {
        false
    }

    fn supports_tool_calls(&self) -> bool {
        false
    }

    fn get_optimizer_info(&self) -> OptimizerInfo {
        UninitializedOptimizerInfo {
            inner: UninitializedOptimizerConfig::TogetherSFT(UninitializedTogetherSFTConfig {
                model: "meta-llama/Meta-Llama-3.1-8B-Instruct-Reference".to_string(),
                credentials: None,
                api_base: None,
            }),
        }
        .load()
        .expect("Failed to create TogetherSFT optimizer info")
    }
}

optimization_test_case!(together_sft, TogetherSFTTestCase());
