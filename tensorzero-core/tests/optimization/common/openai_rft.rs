use crate::common::OptimizationTestCase;
use tensorzero_core::config_parser::path::TomlRelativePath;
use tensorzero_core::optimization::{
    openai_rft::UninitializedOpenAIRFTConfig, UninitializedOptimizerConfig,
    UninitializedOptimizerInfo,
};
use tensorzero_core::providers::openai::optimization::{Grader, OpenAIStringCheckOp};

pub struct OpenAIRFTTestCase();

impl OptimizationTestCase for OpenAIRFTTestCase {
    fn supports_image_data(&self) -> bool {
        false
    }

    fn supports_tool_calls(&self) -> bool {
        false
    }

    fn get_optimizer_info(&self, use_mock_inference_provider: bool) -> UninitializedOptimizerInfo {
        UninitializedOptimizerInfo {
            inner: UninitializedOptimizerConfig::OpenAIRFT(UninitializedOpenAIRFTConfig {
                // Use a model that supports images and tool calls
                model: "o4-mini-2025-04-16".to_string(),
                grader: Grader::StringCheck {
                    name: "test_grader".to_string(),
                    operation: OpenAIStringCheckOp::Eq,
                    input: TomlRelativePath::new_fake_path(
                        "input".to_string(),
                        "{{sample.output_text}}".to_string(),
                    ),
                    reference: TomlRelativePath::new_fake_path(
                        "reference".to_string(),
                        "{{item.reference_text}}".to_string(),
                    ),
                },
                response_format: None,
                batch_size: None,
                compute_multiplier: None,
                eval_interval: None,
                eval_samples: None,
                learning_rate_multiplier: None,
                n_epochs: Some(1),
                reasoning_effort: Some("low".to_string()),
                credentials: None,
                api_base: if use_mock_inference_provider {
                    Some("http://localhost:3030/openai/".parse().unwrap())
                } else {
                    None
                },
                seed: None,
                suffix: None,
            }),
        }
    }
}
