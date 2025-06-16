use crate::{optimization_test_case, OptimizationTestCase};
use tensorzero_internal::optimization::OptimizerConfig;

struct OpenAISFTTestCase {}

impl OptimizationTestCase for OpenAISFTTestCase {
    fn supports_image_data(&self) -> bool {
        true
    }

    fn supports_tool_calls(&self) -> bool {
        true
    }

    fn get_optimizer_config(&self) -> OptimizerConfig {
        todo!()
    }
}

optimization_test_case!(openai_sft, OpenAISFTTestCase {});
