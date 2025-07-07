use crate::{optimization_test_case, use_mock_inference_provider, OptimizationTestCase};
use tensorzero_core::{
    model::CredentialLocation,
    optimization::{
        gcp_vertex_gemini_sft::UninitializedGCPVertexGeminiSFTConfig, OptimizerInfo,
        UninitializedOptimizerConfig, UninitializedOptimizerInfo,
    },
};

struct GCPVertexGeminiSFTTestCase();

impl OptimizationTestCase for GCPVertexGeminiSFTTestCase {
    fn supports_image_data(&self) -> bool {
        false
    }

    fn supports_tool_calls(&self) -> bool {
        true
    }

    fn get_optimizer_info(&self) -> OptimizerInfo {
        UninitializedOptimizerInfo {
            inner: UninitializedOptimizerConfig::GCPVertexGeminiSFT(
                UninitializedGCPVertexGeminiSFTConfig {
                    model: "gemini-2.0-flash-lite-001".to_string(),
                    learning_rate_multiplier: None,
                    adapter_size: None,
                    n_epochs: Some(1),
                    export_last_checkpoint_only: None,
                    credentials: None,
                    credential_location: if use_mock_inference_provider() {
                        Some(CredentialLocation::None)
                    } else {
                        None
                    },
                    seed: None,
                    api_base: None,
                    service_account: None,
                    kms_key_name: None,
                    tuned_model_display_name: None,
                    bucket_name: "tensorzero-e2e-tests".to_string(),
                    bucket_path_prefix: None,
                    project_id: "tensorzero-public".to_string(),
                    region: "us-central1".to_string(),
                },
            ),
        }
        .load()
        .expect("Failed to create GCPVertexGeminiSFT optimizer info")
    }
}

optimization_test_case!(gcp_vertex_gemini_sft, GCPVertexGeminiSFTTestCase());
