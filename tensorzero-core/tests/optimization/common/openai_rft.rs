use crate::common::OptimizationTestCase;
use std::collections::HashMap;
use tensorzero_core::optimization::{
    openai_rft::UninitializedOpenAIRFTConfig, UninitializedOptimizerConfig,
    UninitializedOptimizerInfo,
};
use tensorzero_core::providers::openai::optimization::{
    OpenAIGrader, OpenAIModelGraderInput, OpenAIRFTRole, OpenAIStringCheckOp,
};

use super::mock_inference_provider_base;

pub struct OpenAIRFTTestCase();

impl OptimizationTestCase for OpenAIRFTTestCase {
    fn supports_image_data(&self) -> bool {
        false
    }

    fn supports_tool_calls(&self) -> bool {
        true
    }

    fn get_optimizer_info(&self, use_mock_inference_provider: bool) -> UninitializedOptimizerInfo {
        UninitializedOptimizerInfo {
            inner: UninitializedOptimizerConfig::OpenAIRFT(UninitializedOpenAIRFTConfig {
                // Use a model that supports images and tool calls
                model: "o4-mini-2025-04-16".to_string(),
                grader: OpenAIGrader::Multi {
                    name: "test_grader".to_string(),
                    graders: {
                        let mut map = HashMap::new();
                        map.insert(
                            "string_check_grader".to_string(),
                            Box::new(OpenAIGrader::StringCheck {
                                name: "string_check_grader".to_string(),
                                operation: OpenAIStringCheckOp::Eq,
                                input: "{{sample.output_text}}".to_string(),
                                reference: "{{item.reference_text}}".to_string(),
                            }),
                        );
                        map.insert(
                            "score_model_grader".to_string(),
                            Box::new(OpenAIGrader::ScoreModel {
                                name: "score_model_grader".to_string(),
                                model: "gpt-4.1-nano-2025-04-14".to_string(),
                                input: vec![
                                    OpenAIModelGraderInput {
                                        role: OpenAIRFTRole::Developer,
                                        content: "You are an expert grader. Score the following response on a scale of 0 to 1.".to_string(),
                                    },
                                    OpenAIModelGraderInput {
                                        role: OpenAIRFTRole::User,
                                        content: "Reference Text:\n{{item.reference_text}}\n\nResponse Text:\n{{sample.output_text}}\n\nReference Tool Calls:\n{{item.reference_tools}}\n\nResponse Tool Calls:\n{{sample.output_tools}}".to_string(),
                                    },
                                ],
                                range: Some([0.0, 1.0]),
                            })
                        );
                        map
                    },
                    calculate_output: "0.5 * string_check_grader + 0.5 * score_model_grader"
                        .to_string(),
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
                    Some(mock_inference_provider_base().join("openai/").unwrap())
                } else {
                    None
                },
                seed: None,
                suffix: None,
            }),
        }
    }
}
