use crate::{optimization_test_case, OptimizationTestCase};
use tensorzero_core::optimization::{
    fireworks_sft::UninitializedFireworksSFTConfig, OptimizerInfo, UninitializedOptimizerConfig,
    UninitializedOptimizerInfo,
};

struct FireworksSFTTestCase();

impl OptimizationTestCase for FireworksSFTTestCase {
    fn supports_image_data(&self) -> bool {
        false
    }

    fn supports_tool_calls(&self) -> bool {
        true
    }

    fn get_optimizer_info(&self) -> OptimizerInfo {
        UninitializedOptimizerInfo {
            inner: UninitializedOptimizerConfig::FireworksSFT(UninitializedFireworksSFTConfig {
                model: "accounts/fireworks/models/llama-v3p1-8b-instruct".to_string(),
                account_id: "viraj-ebfe5a".to_string(),
                credentials: None,
                api_base: None,
            }),
        }
        .load()
        .expect("Failed to create FireworksSFT optimizer info")
    }
}

optimization_test_case!(fireworks_sft, FireworksSFTTestCase());
