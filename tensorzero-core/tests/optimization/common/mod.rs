#![expect(clippy::panic, clippy::print_stdout, clippy::unwrap_used)]
use base64::Engine;
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use tensorzero_core::rate_limiting::ScopeInfo;
use tokio::time::{sleep, Duration};
use url::Url;
use uuid::Uuid;

use tracing_subscriber::{self, EnvFilter};

use tensorzero::{InferenceOutputSource, LaunchOptimizationWorkflowParams, RenderedSample, Role};
use tensorzero_core::{
    cache::CacheOptions,
    config::{provider_types::ProviderTypesConfig, Config, ConfigFileGlob},
    db::{
        clickhouse::{test_helpers::CLICKHOUSE_URL, ClickHouseConnectionInfo, ClickhouseFormat},
        postgres::PostgresConnectionInfo,
    },
    endpoints::inference::InferenceClients,
    http::TensorzeroHttpClient,
    inference::types::{
        file::Base64FileMetadata,
        resolved_input::FileWithPath,
        storage::{StorageKind, StoragePath},
        stored_input::StoredFile,
        Base64File, ContentBlock, ContentBlockChatOutput, FunctionType, ModelInferenceRequest,
        ModelInput, RequestMessage, ResolvedContentBlock, ResolvedRequestMessage, StoredInput,
        StoredInputMessage, StoredInputMessageContent, Text,
    },
    model_table::ProviderTypeDefaultCredentials,
    optimization::{
        JobHandle, OptimizationJobInfo, Optimizer, OptimizerOutput, UninitializedOptimizerInfo,
    },
    stored_inference::StoredOutput,
    tool::{Tool, ToolCall, ToolCallConfigDatabaseInsert, ToolCallOutput, ToolChoice, ToolResult},
    variant::JsonMode,
};

pub mod dicl;
pub mod fireworks_sft;
pub mod gcp_vertex_gemini_sft;
pub mod openai_rft;
pub mod openai_sft;
pub mod together_sft;

static FERRIS_PNG: &[u8] = include_bytes!("../../e2e/providers/ferris.png");

fn use_mock_inference_provider() -> bool {
    std::env::var("TENSORZERO_USE_MOCK_INFERENCE_PROVIDER").is_ok()
}

pub fn mock_inference_provider_base() -> Url {
    std::env::var("TENSORZERO_MOCK_INFERENCE_PROVIDER_BASE_URL")
        .unwrap_or_else(|_| "http://localhost:3030/".to_string())
        .parse()
        .unwrap()
}

pub trait OptimizationTestCase {
    fn supports_image_data(&self) -> bool;
    fn supports_tool_calls(&self) -> bool;
    fn get_optimizer_info(&self, use_mock_inference_provider: bool) -> UninitializedOptimizerInfo;
}

#[allow(clippy::allow_attributes, dead_code)]
pub async fn run_test_case(test_case: &impl OptimizationTestCase) {
    // Initialize tracing subscriber to capture progress logs
    let _ = tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .try_init();

    let optimizer_info = test_case
        .get_optimizer_info(use_mock_inference_provider())
        .load(&ProviderTypeDefaultCredentials::default())
        .await
        .unwrap();

    let client = TensorzeroHttpClient::new().unwrap();
    let test_examples = get_examples(test_case, 10);
    let val_examples = Some(get_examples(test_case, 10));
    let credentials: HashMap<String, secrecy::SecretBox<str>> = HashMap::new();

    let mut config_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    config_path.push("tests/e2e/tensorzero.toml");

    // Create an embedded client so that we run migrations
    let tensorzero_client =
        tensorzero::ClientBuilder::new(tensorzero::ClientBuilderMode::EmbeddedGateway {
            config_file: Some(config_path.clone()),
            clickhouse_url: Some(CLICKHOUSE_URL.clone()),
            postgres_url: None,
            timeout: None,
            verify_credentials: true,
            allow_batch_writes: true,
        })
        .build()
        .await
        .unwrap();

    let clickhouse = tensorzero_client
        .get_app_state_data()
        .unwrap()
        .clickhouse_connection_info
        .clone();

    let config_glob = ConfigFileGlob::new_from_path(&config_path).unwrap();
    let config = Config::load_from_path_optional_verify_credentials(
        &config_glob,
        false, // don't validate credentials in tests
    )
    .await
    .unwrap();
    let job_handle = optimizer_info
        .launch(
            &client,
            test_examples,
            val_examples,
            &credentials,
            &clickhouse,
            &config,
        )
        .await
        .unwrap();
    let mut status;
    loop {
        status = job_handle
            .poll(
                &client,
                &credentials,
                &ProviderTypeDefaultCredentials::default(),
            )
            .await
            .unwrap();
        println!("Status: `{status:?}` Handle: `{job_handle}`");
        if matches!(status, OptimizationJobInfo::Completed { .. }) {
            break;
        }
        if matches!(status, OptimizationJobInfo::Failed { .. }) {
            panic!("Optimization failed: {status:?}");
        }
        sleep(if use_mock_inference_provider() {
            Duration::from_secs(1)
        } else {
            Duration::from_secs(60)
        })
        .await;
    }
    assert!(matches!(status, OptimizationJobInfo::Completed { .. }));
    let OptimizationJobInfo::Completed { output } = status else {
        panic!("Expected completed status");
    };

    // Handle Model output only
    match output {
        OptimizerOutput::Model(model_config) => {
            let model_config = model_config
                .load(
                    "test-fine-tuned-model",
                    &ProviderTypesConfig::default(),
                    &ProviderTypeDefaultCredentials::default(),
                )
                .await
                .unwrap();
            // Test the model configuration
            println!("Model configuration loaded successfully: {model_config:?}");

            // Test inference with the fine-tuned model
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
                fetch_and_encode_input_files_before_inference: true,
                extra_body: Default::default(),
                extra_headers: Default::default(),
                extra_cache_key: None,
            };
            let clients = InferenceClients {
                http_client: client.clone(),
                clickhouse_connection_info: ClickHouseConnectionInfo::new_disabled(),
                postgres_connection_info: PostgresConnectionInfo::Disabled,
                credentials: Arc::new(HashMap::new()),
                cache_options: CacheOptions::default(),
                tags: Arc::new(Default::default()),
                rate_limiting_config: Arc::new(Default::default()),
                otlp_config: Default::default(),
                deferred_tasks: tokio_util::task::TaskTracker::new(),
                scope_info: ScopeInfo {
                    tags: Arc::new(HashMap::new()),
                },
            };
            // We didn't produce a real model, so there's nothing to test
            if use_mock_inference_provider() {
                return;
            }
            let response = model_config
                .infer(&request, &clients, "test")
                .await
                .unwrap();
            println!("Response: {response:?}");
        }
        OptimizerOutput::Variant(_) => {
            panic!("Expected model output, got variant output");
        }
    };
}

/// Runs launch_optimization_workflow and then polls for the workflow using the Rust client
#[allow(clippy::allow_attributes, dead_code)]
pub async fn run_workflow_test_case_with_tensorzero_client(
    test_case: &impl OptimizationTestCase,
    client: &tensorzero::Client,
) {
    let params = LaunchOptimizationWorkflowParams {
        function_name: "write_haiku".to_string(),
        template_variant_name: "gpt_4o_mini".to_string(),
        query_variant_name: None,
        filters: None,
        output_source: InferenceOutputSource::Inference,
        order_by: None,
        limit: Some(10),
        offset: None,
        val_fraction: None,
        format: ClickhouseFormat::JsonEachRow,
        // We always mock the client tests since this is tested above
        optimizer_config: test_case.get_optimizer_info(true),
    };
    let job_handle = client
        .experimental_launch_optimization_workflow(params)
        .await
        .unwrap();
    let mut status;
    loop {
        status = client
            .experimental_poll_optimization(&job_handle)
            .await
            .unwrap();
        println!("Status: `{status:?}` Handle: `{job_handle}`");
        if matches!(status, OptimizationJobInfo::Completed { .. }) {
            break;
        }
        if matches!(status, OptimizationJobInfo::Failed { .. }) {
            panic!("Optimization failed: {status:?}");
        }
        sleep(Duration::from_secs(1)).await;
    }
}

fn get_examples(test_case: &impl OptimizationTestCase, num_examples: usize) -> Vec<RenderedSample> {
    assert!(num_examples >= 10);
    let mut generators: Vec<fn() -> RenderedSample> = vec![generate_text_example];

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

fn generate_text_example() -> RenderedSample {
    // So the examples are different
    let id = Uuid::now_v7().to_string();
    let system_prompt =
        format!("You are a helpful assistant named Dr. M.M. Patel with id number {id}.");
    let output = vec![ContentBlockChatOutput::Text(Text {
        text: "The capital of France is Paris.".to_string(),
    })];
    RenderedSample {
        function_name: "basic_test".to_string(),
        input: ModelInput {
            system: Some(system_prompt.clone()),
            messages: vec![ResolvedRequestMessage {
                role: Role::User,
                content: vec![ResolvedContentBlock::Text(Text {
                    text: "What is the capital of France?".to_string(),
                })],
            }],
        },
        stored_input: StoredInput {
            system: Some(json!(system_prompt)),
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputMessageContent::Text {
                    value: "What is the capital of France?".into(),
                }],
            }],
        },
        output: Some(output.clone()),
        stored_output: Some(StoredOutput::Chat(output)),
        episode_id: Some(Uuid::now_v7()),
        inference_id: Some(Uuid::now_v7()),
        tool_params: None,
        output_schema: None,
        dispreferred_outputs: vec![vec![ContentBlockChatOutput::Text(Text {
            text: "The capital of France is Marseille.".to_string(),
        })]],
        tags: HashMap::from([("test_key".to_string(), "test_value".to_string())]),
    }
}

fn generate_tool_call_example() -> RenderedSample {
    // So the examples are different
    let id = Uuid::now_v7().to_string();
    let system_prompt =
        format!("You are a helpful assistant named Dr. M.M. Patel with id number {id}.");
    let tool_call_output = vec![ContentBlockChatOutput::ToolCall(ToolCallOutput {
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
    })];
    RenderedSample {
        function_name: "basic_test".to_string(),
        input: ModelInput {
            system: Some(system_prompt.clone()),
            messages: vec![
                ResolvedRequestMessage {
                    role: Role::User,
                    content: vec![ResolvedContentBlock::Text(Text {
                        text: "What is the weather in Paris?".to_string(),
                    })],
                },
                ResolvedRequestMessage {
                    role: Role::Assistant,
                    content: vec![
                        ResolvedContentBlock::Text(Text {
                            text: "Let me look that up for you.".to_string(),
                        }),
                        ResolvedContentBlock::ToolCall(ToolCall {
                            name: "get_weather".to_string(),
                            arguments: serde_json::json!({
                                "location": "Paris"
                            })
                            .to_string(),
                            id: "call_1".to_string(),
                        }),
                    ],
                },
                ResolvedRequestMessage {
                    role: Role::User,
                    content: vec![ResolvedContentBlock::ToolResult(ToolResult {
                        name: "get_weather".to_string(),
                        result: serde_json::json!({
                            "weather": "sunny, 25 degrees Celsius",
                        })
                        .to_string(),
                        id: "call_1".to_string(),
                    })],
                },
                ResolvedRequestMessage {
                    role: Role::Assistant,
                    content: vec![ResolvedContentBlock::Text(Text {
                        text: "The weather in Paris is sunny, 25 degrees Celsius.".to_string(),
                    })],
                },
                ResolvedRequestMessage {
                    role: Role::User,
                    content: vec![ResolvedContentBlock::Text(Text {
                        text: "What is the weather in London?".to_string(),
                    })],
                },
            ],
        },
        stored_input: StoredInput {
            system: Some(json!(system_prompt)),
            messages: vec![
                StoredInputMessage {
                    role: Role::User,
                    content: vec![StoredInputMessageContent::Text {
                        value: json!("What is the weather in Paris?"),
                    }],
                },
                StoredInputMessage {
                    role: Role::Assistant,
                    content: vec![
                        StoredInputMessageContent::Text {
                            value: json!("Let me look that up for you."),
                        },
                        StoredInputMessageContent::ToolCall(ToolCall {
                            name: "get_weather".to_string(),
                            arguments: serde_json::json!({
                                "location": "Paris"
                            })
                            .to_string(),
                            id: "call_1".to_string(),
                        }),
                    ],
                },
                StoredInputMessage {
                    role: Role::User,
                    content: vec![StoredInputMessageContent::ToolResult(ToolResult {
                        name: "get_weather".to_string(),
                        result: serde_json::json!({
                            "weather": "sunny, 25 degrees Celsius",
                        })
                        .to_string(),
                        id: "call_1".to_string(),
                    })],
                },
                StoredInputMessage {
                    role: Role::Assistant,
                    content: vec![StoredInputMessageContent::Text {
                        value: json!("The weather in Paris is sunny, 25 degrees Celsius."),
                    }],
                },
                StoredInputMessage {
                    role: Role::User,
                    content: vec![StoredInputMessageContent::Text {
                        value: json!("What is the weather in London?"),
                    }],
                },
            ],
        },
        output: Some(tool_call_output.clone()),
        stored_output: Some(StoredOutput::Chat(tool_call_output)),
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
        episode_id: Some(Uuid::now_v7()),
        inference_id: Some(Uuid::now_v7()),
        output_schema: None,
        dispreferred_outputs: vec![],
        tags: HashMap::new(),
    }
}

fn generate_image_example() -> RenderedSample {
    // So the examples are different
    let id = Uuid::now_v7().to_string();
    let system_prompt =
        format!("You are a helpful assistant named Dr. M.M. Patel with id number {id}.");
    let output = vec![ContentBlockChatOutput::Text(Text {
        text: "Orange!".to_string(),
    })];
    RenderedSample {
        function_name: "basic_test".to_string(),
        input: ModelInput {
            system: Some(system_prompt.clone()),
            messages: vec![ResolvedRequestMessage {
                role: Role::User,
                content: vec![
                    ResolvedContentBlock::Text(Text {
                        text: "What is the main color of this image?".to_string(),
                    }),
                    ResolvedContentBlock::File(Box::new(FileWithPath {
                        file: Base64File {
                            url: None,
                            mime_type: mime::IMAGE_PNG,
                            data: base64::prelude::BASE64_STANDARD.encode(FERRIS_PNG),
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
        stored_input: StoredInput {
            system: Some(json!(system_prompt)),
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![
                    StoredInputMessageContent::Text {
                        value: json!("What is the main color of this image?"),
                    },
                    StoredInputMessageContent::File(Box::new(StoredFile {
                        file: Base64FileMetadata {
                            url: None,
                            mime_type: mime::IMAGE_PNG,
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
        output: Some(output.clone()),
        stored_output: Some(StoredOutput::Chat(output)),
        tool_params: None,
        episode_id: Some(Uuid::now_v7()),
        inference_id: Some(Uuid::now_v7()),
        output_schema: None,
        dispreferred_outputs: vec![vec![ContentBlockChatOutput::Text(Text {
            text: "Blue!".to_string(),
        })]],
        tags: HashMap::new(),
    }
}

/// Generates a `#[tokio::test] async fn $fn_name() { run_test_case(&$constructor).await; }`
#[macro_export]
macro_rules! optimization_test_case {
    // $fn_name  = the name of the generated test function
    // $constructor = an expression which yields your impl of OptimizationTestCase
    ($fn_name:ident, $constructor:expr) => {
        ::paste::paste! {
            #[tokio::test(flavor = "multi_thread")]
            async fn [<test_slow_optimization_ $fn_name>]() {
                $crate::common::run_test_case(&$constructor).await;
            }
        }
    };
}

/// Generates a `#[tokio::test] async fn $fn_name() { run_workflow_test_case_with_tensorzero_client(&$constructor, &client).await; }`
#[macro_export]
macro_rules! embedded_workflow_test_case {
    // $fn_name  = the name of the generated test function
    // $constructor = an expression which yields your impl of OptimizationTestCase
    ($fn_name:ident, $constructor:expr) => {
        ::paste::paste! {
            #[tokio::test(flavor = "multi_thread")]
            async fn [<test_embedded_slow_optimization_ $fn_name>]() {
                let client = tensorzero::test_helpers::make_embedded_gateway().await;
                $crate::common::run_workflow_test_case_with_tensorzero_client(&$constructor, &client).await;
            }
        }
    };
}

/// Generates a `#[tokio::test] async fn $fn_name() { run_workflow_test_case_with_tensorzero_client(&$constructor, &client).await; }`
#[macro_export]
macro_rules! http_workflow_test_case {
    // $fn_name  = the name of the generated test function
    // $constructor = an expression which yields your impl of OptimizationTestCase
    ($fn_name:ident, $constructor:expr) => {
        ::paste::paste! {
            #[tokio::test]
            async fn [<test_http_slow_optimization_ $fn_name>]() {
                let client = tensorzero::test_helpers::make_http_gateway().await;
                $crate::common::run_workflow_test_case_with_tensorzero_client(&$constructor, &client).await;
            }
        }
    };
}
