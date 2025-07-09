use crate::{
    embedded_workflow_test_case, http_workflow_test_case, optimization_test_case,
    OptimizationTestCase,
};
use tensorzero_core::optimization::{
    fireworks_sft::UninitializedFireworksSFTConfig, UninitializedOptimizerConfig,
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

optimization_test_case!(fireworks_sft, FireworksSFTTestCase());
embedded_workflow_test_case!(fireworks_sft, FireworksSFTTestCase());
http_workflow_test_case!(fireworks_sft, FireworksSFTTestCase());
