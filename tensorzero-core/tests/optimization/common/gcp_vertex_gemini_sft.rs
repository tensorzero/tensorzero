use crate::common::OptimizationTestCase;
use tensorzero_core::{
    model::CredentialLocation,
    optimization::{
        gcp_vertex_gemini_sft::UninitializedGCPVertexGeminiSFTConfig, UninitializedOptimizerConfig,
        UninitializedOptimizerInfo,
    },
};

// Currently unused in 'mock_tests.rs'
#[allow(clippy::allow_attributes, dead_code)]
pub struct GCPVertexGeminiSFTTestCase();

impl OptimizationTestCase for GCPVertexGeminiSFTTestCase {
    fn supports_image_data(&self) -> bool {
        false
    }

    fn supports_tool_calls(&self) -> bool {
        true
    }

    fn get_optimizer_info(&self, use_mock_inference_provider: bool) -> UninitializedOptimizerInfo {
        // When using mock inference provider, check if GCP credentials are available
        // If not, we'll let the test run but it may fail with credential errors
        // This is preferable to panicking and failing the entire test suite
        if use_mock_inference_provider {
            // Check for both TensorZero's default and standard Google Cloud SDK credential variables
            let has_credentials = std::env::var("GCP_VERTEX_CREDENTIALS_PATH").is_ok()
                || std::env::var("GOOGLE_APPLICATION_CREDENTIALS").is_ok();

            if !has_credentials {
                tracing::warn!("GCP Vertex Gemini mock test may fail - no GCP credentials found. Set GCP_VERTEX_CREDENTIALS_PATH or GOOGLE_APPLICATION_CREDENTIALS environment variable.");
            }
        }
        UninitializedOptimizerInfo {
            inner: UninitializedOptimizerConfig::GCPVertexGeminiSFT(
                UninitializedGCPVertexGeminiSFTConfig {
                    model: "gemini-2.0-flash-lite-001".to_string(),
                    learning_rate_multiplier: None,
                    adapter_size: None,
                    n_epochs: Some(1),
                    export_last_checkpoint_only: None,
                    credentials: Some(CredentialLocation::Sdk),
                    seed: None,
                    api_base: if use_mock_inference_provider {
                        Some("http://localhost:3030/gcp_vertex_gemini/".parse().unwrap())
                    } else {
                        None
                    },
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
    }
}
