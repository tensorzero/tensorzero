use crate::common::OptimizationTestCase;
use tensorzero_core::optimization::{
    UninitializedOptimizerConfig, UninitializedOptimizerInfo,
    gcp_vertex_gemini_sft::UninitializedGCPVertexGeminiSFTConfig,
};

pub struct GCPVertexGeminiSFTTestCase();

impl OptimizationTestCase for GCPVertexGeminiSFTTestCase {
    fn supports_image_data(&self) -> bool {
        false
    }

    fn supports_tool_calls(&self) -> bool {
        true
    }

    fn get_optimizer_info(&self) -> UninitializedOptimizerInfo {
        // Provider-level settings (project_id, region, bucket_name, api_base, credentials)
        // come from [provider_types.gcp_vertex_gemini.sft] in the gateway config.
        // Only per-job settings are specified here.
        UninitializedOptimizerInfo {
            inner: UninitializedOptimizerConfig::GCPVertexGeminiSFT(
                UninitializedGCPVertexGeminiSFTConfig {
                    model: "gemini-2.5-flash-lite".to_string(),
                    learning_rate_multiplier: None,
                    adapter_size: None,
                    n_epochs: Some(1),
                    export_last_checkpoint_only: None,
                    seed: None,
                    tuned_model_display_name: None,
                },
            ),
        }
    }
}
