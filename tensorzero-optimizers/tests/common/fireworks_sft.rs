use crate::common::OptimizationTestCase;
use tensorzero_core::optimization::{
    fireworks_sft::UninitializedFireworksSFTConfig, UninitializedOptimizerConfig,
    UninitializedOptimizerInfo,
};

use super::mock_inference_provider_base;

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
                early_stop: None,
                epochs: Some(1),
                learning_rate: None,
                max_context_length: None,
                lora_rank: None,
                batch_size: None,
                display_name: None,
                output_model: None,
                warm_start_from: None,
                is_turbo: None,
                eval_auto_carveout: None,
                nodes: None,
                mtp_enabled: None,
                mtp_num_draft_tokens: None,
                mtp_freeze_base_model: None,
                credentials: None,
                api_base: if use_mock_inference_provider {
                    Some(mock_inference_provider_base().join("fireworks/").unwrap())
                } else {
                    None
                },
            }),
        }
    }
}
