use std::collections::HashMap;

use tensorzero::RenderedStoredInference;
use tensorzero_internal::optimization::{
    Optimizer, OptimizerConfig, OptimizerOutput, OptimizerStatus,
};

pub trait OptimizationTestCase {
    fn supports_image_data(&self) -> bool;
    fn supports_tool_calls(&self) -> bool;
    fn get_optimizer_config(&self) -> OptimizerConfig;
}

pub async fn run_test_case(test_case: &impl OptimizationTestCase) {
    let optimizer_config = test_case.get_optimizer_config();
    let client = reqwest::Client::new();
    let test_examples = get_examples(test_case, 10);
    let val_examples = Some(get_examples(test_case, 10));
    let credentials: HashMap<String, secrecy::SecretBox<str>> = HashMap::new();
    let job_handle = optimizer_config
        .launch(&client, test_examples, val_examples, &credentials)
        .await
        .unwrap();
    let mut status = optimizer_config
        .poll(&client, &job_handle, &credentials)
        .await
        .unwrap();
    while !matches!(status, OptimizerStatus::Completed(_)) {
        status = optimizer_config
            .poll(&client, &job_handle, &credentials)
            .await
            .unwrap();
    }
    assert!(matches!(status, OptimizerStatus::Completed(_)));
    let OptimizerStatus::Completed(OptimizerOutput::Model(model_config)) = status else {
        panic!("Expected model config");
    };
}

fn get_examples(
    test_case: &impl OptimizationTestCase,
    num_examples: usize,
) -> Vec<RenderedStoredInference> {
    todo!()
}
