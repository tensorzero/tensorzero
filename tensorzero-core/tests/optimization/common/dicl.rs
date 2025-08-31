use crate::common::OptimizationTestCase;
use tensorzero_core::optimization::{
    dicl::UninitializedDiclOptimizationConfig, UninitializedOptimizerConfig,
    UninitializedOptimizerInfo,
};

pub struct DiclTestCase();

impl OptimizationTestCase for DiclTestCase {
    fn supports_image_data(&self) -> bool {
        false
    }

    fn supports_tool_calls(&self) -> bool {
        true
    }

    fn get_optimizer_info(&self, use_mock_inference_provider: bool) -> UninitializedOptimizerInfo {
        UninitializedOptimizerInfo {
            inner: UninitializedOptimizerConfig::Dicl(UninitializedDiclOptimizationConfig {
                embedding_model: if use_mock_inference_provider {
                    "dummy-embedding-model".to_string()
                } else {
                    "text-embedding-3-small".to_string()
                },
                variant_name: "test_dicl".to_string(),
                function_name: "basic_test".to_string(),
                ..Default::default()
            }),
        }
    }
}

pub struct DiclJsonTestCase();

impl OptimizationTestCase for DiclJsonTestCase {
    fn supports_image_data(&self) -> bool {
        false
    }

    fn supports_tool_calls(&self) -> bool {
        false // JSON functions don't support tool calls
    }

    fn is_json_function(&self) -> bool {
        true // This is a JSON function test case
    }

    fn get_optimizer_info(&self, use_mock_inference_provider: bool) -> UninitializedOptimizerInfo {
        UninitializedOptimizerInfo {
            inner: UninitializedOptimizerConfig::Dicl(UninitializedDiclOptimizationConfig {
                embedding_model: if use_mock_inference_provider {
                    "dummy-embedding-model".to_string()
                } else {
                    "text-embedding-3-small".to_string()
                },
                variant_name: "test_dicl_json".to_string(),
                function_name: "basic_test".to_string(), // Same function name, will create custom config
                ..Default::default()
            }),
        }
    }
}
