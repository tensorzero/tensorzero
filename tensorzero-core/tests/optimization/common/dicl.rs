use crate::common::OptimizationTestCase;
use tensorzero_core::optimization::{
    dicl::UninitializedDiclOptimizationConfig, UninitializedOptimizerConfig,
    UninitializedOptimizerInfo,
};

pub struct DiclTestCase();

impl OptimizationTestCase for DiclTestCase {
    fn supports_image_data(&self) -> bool {
        true
    }

    fn supports_tool_calls(&self) -> bool {
        true
    }

    fn get_optimizer_info(&self, use_mock_inference_provider: bool) -> UninitializedOptimizerInfo {
        UninitializedOptimizerInfo {
            inner: UninitializedOptimizerConfig::Dicl(UninitializedDiclOptimizationConfig {
                // This is the only model that supports images
                embedding_model: "text-embedding-3-small".to_string(),
                variant_name: "test".to_string(),
                function_name: "test".to_string(),
                credentials: None,
                api_base: if use_mock_inference_provider {
                    Some("http://localhost:3030/openai/".parse().unwrap())
                } else {
                    None
                },
            }),
        }
    }
}
