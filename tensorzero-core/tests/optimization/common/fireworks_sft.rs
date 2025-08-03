use crate::common::OptimizationTestCase;
use tensorzero_core::optimization::{
    fireworks_sft::UninitializedFireworksSFTConfig, UninitializedOptimizerConfig,
    UninitializedOptimizerInfo,
};

pub struct FireworksSFTTestCase();

impl OptimizationTestCase for FireworksSFTTestCase {
    fn supports_image_data(&self) -> bool {
        false
    }

    fn supports_tool_calls(&self) -> bool {
        true
    }

    fn get_optimizer_info(&self, use_mock_inference_provider: bool) -> UninitializedOptimizerInfo {
        UninitializedOptimizerInfo {
            inner: UninitializedOptimizerConfig::FireworksSFT(UninitializedFireworksSFTConfig {
                model: "accounts/fireworks/models/llama-v3p1-8b-instruct".to_string(),
                account_id: "viraj-ebfe5a".to_string(),
                credentials: None,
                api_base: if use_mock_inference_provider {
                    Some("http://localhost:3030/fireworks/".parse().unwrap())
                } else {
                    None
                },
            }),
        }
    }
}
