use crate::common::OptimizationTestCase;
use tensorzero_core::optimization::{
    UninitializedOptimizerConfig, UninitializedOptimizerInfo,
    fireworks_sft::UninitializedFireworksSFTConfig,
};

pub struct FireworksSFTTestCase();

impl OptimizationTestCase for FireworksSFTTestCase {
    fn supports_image_data(&self) -> bool {
        false
    }

    fn supports_tool_calls(&self) -> bool {
        true
    }

    fn get_optimizer_info(&self) -> UninitializedOptimizerInfo {
        // Note: mock mode is configured via provider_types.fireworks.sft in the test config file
        UninitializedOptimizerInfo {
            inner: UninitializedOptimizerConfig::FireworksSFT(UninitializedFireworksSFTConfig {
                model: "accounts/fireworks/models/llama-v3p3-70b-instruct".to_string(),
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
                deploy_after_training: None,
            }),
        }
    }
}
