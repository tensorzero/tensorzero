use std::collections::HashMap;
use uuid::Uuid;

use tensorzero::{RenderedStoredInference, Role};
use tensorzero_internal::{
    cache::CacheOptions,
    clickhouse::ClickHouseConnectionInfo,
    endpoints::inference::InferenceClients,
    inference::types::{ContentBlock, FunctionType, ModelInferenceRequest, RequestMessage, Text},
    optimization::{Optimizer, OptimizerConfig, OptimizerOutput, OptimizerStatus},
    variant::JsonMode,
};

mod openai_sft;

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
    let system = "You are a helpful assistant named Dr. M.M. Patel.".to_string();
    let messages = vec![RequestMessage {
        role: Role::User,
        content: vec![ContentBlock::Text(Text {
            text: "What is the capital of France?".to_string(),
        })],
    }];
    let request = ModelInferenceRequest {
        system: Some(system),
        messages,
        inference_id: Uuid::now_v7(),
        tool_config: None,
        temperature: None,
        top_p: None,
        max_tokens: None,
        presence_penalty: None,
        frequency_penalty: None,
        seed: None,
        stop_sequences: None,
        stream: false,
        json_mode: JsonMode::Off.into(),
        function_type: FunctionType::Chat,
        output_schema: None,
        extra_body: Default::default(),
        extra_headers: Default::default(),
        extra_cache_key: None,
    };
    let clients = InferenceClients {
        http_client: &client,
        clickhouse_connection_info: &ClickHouseConnectionInfo::Disabled,
        credentials: &HashMap::new(),
        cache_options: &CacheOptions::default(),
    };
    let response = model_config
        .infer(&request, &clients, "test")
        .await
        .unwrap();
    println!("{:?}", response);
}

fn get_examples(
    test_case: &impl OptimizationTestCase,
    num_examples: usize,
) -> Vec<RenderedStoredInference> {
    todo!()
}

/// Generates a `#[tokio::test] async fn $fn_name() { run_test_case(&$constructor).await; }`
#[macro_export]
macro_rules! optimization_test_case {
    // $fn_name  = the name of the generated test function
    // $constructor = an expression which yields your impl of OptimizationTestCase
    ($fn_name:ident, $constructor:expr) => {
        ::paste::paste! {
            #[tokio::test]
            async fn [<test_optimization_ $fn_name>]() {
                // you might need to import run_test_case
                crate::run_test_case(&$constructor).await;
            }
        }
    };
}
