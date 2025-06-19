#![expect(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::print_stdout
)]
use base64::Engine;
use std::collections::HashMap;
use tokio::time::{sleep, Duration};
use uuid::Uuid;

use tensorzero::{RenderedStoredInference, Role};
use tensorzero_core::{
    cache::CacheOptions,
    clickhouse::ClickHouseConnectionInfo,
    endpoints::inference::InferenceClients,
    inference::types::{
        resolved_input::FileWithPath,
        storage::{StorageKind, StoragePath},
        Base64File, ContentBlock, ContentBlockChatOutput, FunctionType, ModelInferenceRequest,
        ModelInput, RequestMessage, Text,
    },
    optimization::{Optimizer, OptimizerInfo, OptimizerOutput, OptimizerStatus},
    tool::{Tool, ToolCall, ToolCallConfigDatabaseInsert, ToolCallOutput, ToolChoice, ToolResult},
    variant::JsonMode,
};

mod openai_sft;

static FERRIS_PNG: &[u8] = include_bytes!("../e2e/providers/ferris.png");

pub trait OptimizationTestCase {
    fn supports_image_data(&self) -> bool;
    fn supports_tool_calls(&self) -> bool;
    fn get_optimizer_info(&self) -> OptimizerInfo;
}

pub async fn run_test_case(test_case: &impl OptimizationTestCase) {
    let optimizer_info = test_case.get_optimizer_info();
    let client = reqwest::Client::new();
    let test_examples = get_examples(test_case, 10);
    let val_examples = Some(get_examples(test_case, 10));
    let credentials: HashMap<String, secrecy::SecretBox<str>> = HashMap::new();
    let job_handle = optimizer_info
        .launch(&client, test_examples, val_examples, &credentials)
        .await
        .unwrap();
    let mut status = optimizer_info
        .poll(&client, &job_handle, &credentials)
        .await
        .unwrap();
    while !matches!(status, OptimizerStatus::Completed(_)) {
        status = optimizer_info
            .poll(&client, &job_handle, &credentials)
            .await
            .unwrap();
        println!("Status: {status:?}");
        // Sleep for a minute
        sleep(Duration::from_secs(60)).await;
        if matches!(status, OptimizerStatus::Failed) {
            panic!("Optimization failed");
        }
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
    println!("Response: {response:?}");
}

fn get_examples(
    test_case: &impl OptimizationTestCase,
    num_examples: usize,
) -> Vec<RenderedStoredInference> {
    assert!(num_examples >= 10);
    let mut generators: Vec<fn() -> RenderedStoredInference> = vec![generate_text_example];
    if test_case.supports_tool_calls() {
        generators.push(generate_tool_call_example);
    }
    if test_case.supports_image_data() {
        generators.push(generate_image_example);
    }
    generators
        .into_iter()
        .cycle()
        .take(num_examples)
        .map(|g| g())
        .collect()
}

fn generate_text_example() -> RenderedStoredInference {
    // So the examples are different
    let id = Uuid::now_v7().to_string();
    RenderedStoredInference {
        function_name: "test".to_string(),
        variant_name: "test".to_string(),
        input: ModelInput {
            system: Some(format!(
                "You are a helpful assistant named Dr. M.M. Patel with id number {id}."
            )),
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec![ContentBlock::Text(Text {
                    text: "What is the capital of France?".to_string(),
                })],
            }],
        },
        output: vec![ContentBlockChatOutput::Text(Text {
            text: "The capital of France is Paris.".to_string(),
        })],
        episode_id: Uuid::now_v7(),
        inference_id: Uuid::now_v7(),
        tool_params: None,
        output_schema: None,
    }
}

fn generate_tool_call_example() -> RenderedStoredInference {
    // So the examples are different
    let id = Uuid::now_v7().to_string();
    RenderedStoredInference {
        function_name: "test".to_string(),
        variant_name: "test".to_string(),
        input: ModelInput {
            system: Some(format!(
                "You are a helpful assistant named Dr. M.M. Patel with id number {id}."
            )),
            messages: vec![
                RequestMessage {
                    role: Role::User,
                    content: vec![ContentBlock::Text(Text {
                        text: "What is the weather in Paris?".to_string(),
                    })],
                },
                RequestMessage {
                    role: Role::Assistant,
                    content: vec![
                        ContentBlock::Text(Text {
                            text: "Let me look that up for you.".to_string(),
                        }),
                        ContentBlock::ToolCall(ToolCall {
                            name: "get_weather".to_string(),
                            arguments: serde_json::json!({
                                "location": "Paris"
                            })
                            .to_string(),
                            id: "call_1".to_string(),
                        }),
                    ],
                },
                RequestMessage {
                    role: Role::User,
                    content: vec![ContentBlock::ToolResult(ToolResult {
                        name: "get_weather".to_string(),
                        result: serde_json::json!({
                            "weather": "sunny, 25 degrees Celsius",
                        })
                        .to_string(),
                        id: "call_1".to_string(),
                    })],
                },
                RequestMessage {
                    role: Role::Assistant,
                    content: vec![ContentBlock::Text(Text {
                        text: "The weather in Paris is sunny, 25 degrees Celsius.".to_string(),
                    })],
                },
                RequestMessage {
                    role: Role::User,
                    content: vec![ContentBlock::Text(Text {
                        text: "What is the weather in London?".to_string(),
                    })],
                },
            ],
        },
        output: vec![ContentBlockChatOutput::ToolCall(ToolCallOutput {
            name: Some("get_weather".to_string()),
            arguments: Some(serde_json::json!({
                "location": "London",
            })),
            raw_name: "get_weather".to_string(),
            raw_arguments: serde_json::json!({
                "location": "London",
            })
            .to_string(),
            id: "call_2".to_string(),
        })],
        episode_id: Uuid::now_v7(),
        inference_id: Uuid::now_v7(),
        tool_params: Some(ToolCallConfigDatabaseInsert {
            tools_available: vec![Tool {
                name: "get_weather".to_string(),
                description: "Get the weather for a location".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The location to get weather for"
                        }
                    },
                    "required": ["location"]
                }),
                strict: false,
            }],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: None,
        }),
        output_schema: None,
    }
}

fn generate_image_example() -> RenderedStoredInference {
    // So the examples are different
    let id = Uuid::now_v7().to_string();
    RenderedStoredInference {
        function_name: "test".to_string(),
        variant_name: "test".to_string(),
        input: ModelInput {
            system: Some(format!(
                "You are a helpful assistant named Dr. M.M. Patel with id number {id}."
            )),
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec![
                    ContentBlock::Text(Text {
                        text: "What is the main color of this image?".to_string(),
                    }),
                    ContentBlock::File(Box::new(FileWithPath {
                        file: Base64File {
                            url: None,
                            mime_type: mime::IMAGE_PNG,
                            data: Some(base64::prelude::BASE64_STANDARD.encode(FERRIS_PNG)),
                        },
                        storage_path: StoragePath {
                            kind: StorageKind::Disabled,
                            path: object_store::path::Path::parse(
                                "observability/files/08bfa764c6dc25e658bab2b8039ddb494546c3bc5523296804efc4cab604df5d.png"
                            ).unwrap(),
                        },
                    })),
                ],
            }],
        },
        output: vec![ContentBlockChatOutput::Text(Text {
            text: "Orange!".to_string(),
        })],
        episode_id: Uuid::now_v7(),
        inference_id: Uuid::now_v7(),
        tool_params: None,
        output_schema: None,
    }
}

/// Generates a `#[tokio::test] async fn $fn_name() { run_test_case(&$constructor).await; }`
#[macro_export]
macro_rules! optimization_test_case {
    // $fn_name  = the name of the generated test function
    // $constructor = an expression which yields your impl of OptimizationTestCase
    ($fn_name:ident, $constructor:expr) => {
        ::paste::paste! {
            #[tokio::test]
            async fn [<test_slow_optimization_ $fn_name>]() {
                $crate::run_test_case(&$constructor).await;
            }
        }
    };
}
