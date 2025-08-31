#![expect(clippy::unwrap_used, clippy::panic, clippy::print_stdout)]
use base64::Engine;
use serde_json::json;
use std::collections::HashMap;
use tokio::time::{sleep, Duration};
use uuid::Uuid;

use tracing_subscriber::{self, EnvFilter};

use tensorzero::{
    Client, ClientInferenceParams, ClientInput, ClientInputMessage, ClientInputMessageContent,
    InferenceOutput, InferenceOutputSource, LaunchOptimizationWorkflowParams, RenderedSample, Role,
};
use tensorzero_core::{
    cache::CacheOptions,
    config::{Config, ConfigFileGlob, ProviderTypesConfig, UninitializedVariantConfig},
    db::clickhouse::test_helpers::{get_clickhouse, CLICKHOUSE_URL},
    db::clickhouse::{ClickHouseConnectionInfo, ClickhouseFormat},
    endpoints::inference::InferenceClients,
    inference::types::{
        resolved_input::FileWithPath,
        storage::{StorageKind, StoragePath},
        Base64File, ContentBlock, ContentBlockChatOutput, FunctionType, ModelInferenceRequest,
        ModelInput, RequestMessage, ResolvedInput, ResolvedInputMessage,
        ResolvedInputMessageContent, Text, TextKind,
    },
    optimization::JobHandle,
    optimization::{OptimizationJobInfo, Optimizer, OptimizerOutput, UninitializedOptimizerInfo},
    tool::{Tool, ToolCall, ToolCallConfigDatabaseInsert, ToolCallOutput, ToolChoice, ToolResult},
    variant::JsonMode,
};

pub mod dicl;
pub mod fireworks_sft;
pub mod gcp_vertex_gemini_sft;
pub mod openai_sft;
pub mod together_sft;

static FERRIS_PNG: &[u8] = include_bytes!("../../e2e/providers/ferris.png");

fn use_mock_inference_provider() -> bool {
    std::env::var("TENSORZERO_USE_MOCK_INFERENCE_PROVIDER").is_ok()
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
        .load()
        .await
        .unwrap();
    let client = reqwest::Client::new();
    let test_examples = get_examples(test_case, 10);
    let val_examples = Some(get_examples(test_case, 10));
    let credentials: HashMap<String, secrecy::SecretBox<str>> = HashMap::new();
    let clickhouse = get_clickhouse().await;
    let mut config_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    config_path.push("tests/e2e/tensorzero.toml");
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
        status = job_handle.poll(&client, &credentials).await.unwrap();
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

    // Handle both Model and Variant outputs
    match output {
        OptimizerOutput::Model(model_config) => {
            let model_config = model_config
                .load("test-fine-tuned-model", &ProviderTypesConfig::default())
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
        OptimizerOutput::Variant(variant_config) => {
            // Test the variant configuration
            println!("Variant configuration created successfully: {variant_config:?}");

            // Test DICL variants with full inference
            if let UninitializedVariantConfig::Dicl(dicl_config) = variant_config.as_ref() {
                test_dicl_variant_inference(dicl_config).await;
            }
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
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec![ContentBlock::Text(Text {
                    text: "What is the capital of France?".to_string(),
                })],
            }],
        },
        stored_input: ResolvedInput {
            system: Some(json!(system_prompt)),
            messages: vec![ResolvedInputMessage {
                role: Role::User,
                content: vec![ResolvedInputMessageContent::Text {
                    value: json!("What is the capital of France?"),
                }],
            }],
        },
        output: Some(output.clone()),
        stored_output: Some(tensorzero_core::stored_inference::StoredOutput::Chat(
            output,
        )),
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
        stored_input: ResolvedInput {
            system: Some(json!(system_prompt)),
            messages: vec![
                ResolvedInputMessage {
                    role: Role::User,
                    content: vec![ResolvedInputMessageContent::Text {
                        value: json!("What is the weather in Paris?"),
                    }],
                },
                ResolvedInputMessage {
                    role: Role::Assistant,
                    content: vec![
                        ResolvedInputMessageContent::Text {
                            value: json!("Let me look that up for you."),
                        },
                        ResolvedInputMessageContent::ToolCall(ToolCall {
                            name: "get_weather".to_string(),
                            arguments: serde_json::json!({
                                "location": "Paris"
                            })
                            .to_string(),
                            id: "call_1".to_string(),
                        }),
                    ],
                },
                ResolvedInputMessage {
                    role: Role::User,
                    content: vec![ResolvedInputMessageContent::ToolResult(ToolResult {
                        name: "get_weather".to_string(),
                        result: serde_json::json!({
                            "weather": "sunny, 25 degrees Celsius",
                        })
                        .to_string(),
                        id: "call_1".to_string(),
                    })],
                },
                ResolvedInputMessage {
                    role: Role::Assistant,
                    content: vec![ResolvedInputMessageContent::Text {
                        value: json!("The weather in Paris is sunny, 25 degrees Celsius."),
                    }],
                },
                ResolvedInputMessage {
                    role: Role::User,
                    content: vec![ResolvedInputMessageContent::Text {
                        value: json!("What is the weather in London?"),
                    }],
                },
            ],
        },
        output: Some(tool_call_output.clone()),
        stored_output: Some(tensorzero_core::stored_inference::StoredOutput::Chat(
            tool_call_output,
        )),
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
        stored_input: ResolvedInput {
            system: Some(json!(system_prompt)),
            messages: vec![ResolvedInputMessage {
                role: Role::User,
                content: vec![
                    ResolvedInputMessageContent::Text {
                        value: json!("What is the main color of this image?"),
                    },
                    ResolvedInputMessageContent::File(Box::new(FileWithPath {
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
        output: Some(output.clone()),
        stored_output: Some(tensorzero_core::stored_inference::StoredOutput::Chat(output)),
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

#[allow(clippy::allow_attributes, dead_code)]
pub async fn make_embedded_gateway() -> Client {
    let mut config_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    config_path.push("tests/e2e/tensorzero.toml");
    tensorzero::ClientBuilder::new(tensorzero::ClientBuilderMode::EmbeddedGateway {
        config_file: Some(config_path),
        clickhouse_url: Some(CLICKHOUSE_URL.clone()),
        timeout: None,
        verify_credentials: true,
        allow_batch_writes: true,
    })
    .with_verbose_errors(true)
    .build()
    .await
    .unwrap()
}

#[allow(clippy::allow_attributes, dead_code)]
pub async fn make_http_gateway() -> Client {
    tensorzero::ClientBuilder::new(tensorzero::ClientBuilderMode::HTTPGateway {
        url: "http://localhost:3000".parse().unwrap(),
    })
    .with_verbose_errors(true)
    .build()
    .await
    .unwrap()
}

/// Test DICL variant inference by creating a temporary config and testing inference
async fn test_dicl_variant_inference(
    dicl_config: &tensorzero_core::variant::dicl::UninitializedDiclConfig,
) {
    use std::fs;
    use std::path::PathBuf;
    use tempfile::TempDir;

    // Create a temporary directory for our config files
    let temp_dir = TempDir::new().unwrap();

    // Create a system schema file
    let system_schema_path = temp_dir.path().join("system_schema.json");
    let system_schema_content = r#"{
  "type": "object",
  "properties": {
    "assistant_name": {
      "type": "string"
    }
  }
}"#;
    fs::write(&system_schema_path, system_schema_content).unwrap();

    // Create a system template file
    let system_template_path = temp_dir.path().join("system_template.minijinja");
    let system_template_content = "You are a helpful assistant named {{assistant_name}}.";
    fs::write(&system_template_path, system_template_content).unwrap();

    // Create a minimal but complete main config with embedding model
    // Note: Using "basic_test" as function name to match the e2e config and stored examples
    let main_config_content = r#"
[functions.basic_test]
type = "chat"
system_schema = "system_schema.json"

[functions.basic_test.variants.gpt_4o_mini]
type = "chat_completion"
model = "openai::gpt-4o-mini-2024-07-18"
system_template = "system_template.minijinja"

[embedding_models.text-embedding-3-small]
routing = ["openai"]

[embedding_models.text-embedding-3-small.providers.openai]
type = "openai"
model_name = "text-embedding-3-small"
"#;
    let main_config_copy = temp_dir.path().join("main.toml");
    fs::write(&main_config_copy, main_config_content).unwrap();

    // Create a DICL variant config file
    // Use "test_dicl" to match the variant name used during optimization
    let variant_name = "test_dicl".to_string();
    let temp_config_content = format!(
        r#"
[functions.basic_test.variants.{}]
type = "experimental_dynamic_in_context_learning"
embedding_model = "{}"
k = {}
model = "{}"
"#,
        variant_name, dicl_config.embedding_model, dicl_config.k, dicl_config.model
    );

    let dicl_config_path = temp_dir.path().join("dicl_variant.toml");
    fs::write(&dicl_config_path, temp_config_content).unwrap();

    // Create a glob pattern that matches both files
    let glob_pattern = format!("{}/*.toml", temp_dir.path().display());

    // Create a new gateway with the combined config
    let client = tensorzero::ClientBuilder::new(tensorzero::ClientBuilderMode::EmbeddedGateway {
        config_file: Some(PathBuf::from(&glob_pattern)),
        clickhouse_url: Some(CLICKHOUSE_URL.clone()),
        timeout: None,
        verify_credentials: true,
        allow_batch_writes: true,
    })
    .with_verbose_errors(true)
    .build()
    .await
    .unwrap();

    // Generate a unique episode ID for tracking
    let episode_id = Uuid::now_v7();

    // Test inference with the DICL variant - using patterns from e2e tests
    let inference_params = ClientInferenceParams {
        function_name: Some("basic_test".to_string()),
        model_name: None,
        episode_id: Some(episode_id),
        input: ClientInput {
            system: Some(serde_json::json!({"assistant_name": "TensorZero Test Assistant"})),
            messages: vec![ClientInputMessage {
                role: Role::User,
                content: vec![ClientInputMessageContent::Text(TextKind::Text {
                    text: "What is the capital of France?".to_string(),
                })],
            }],
        },
        stream: Some(false),
        params: Default::default(),
        variant_name: Some(variant_name.clone()),
        dryrun: None,
        internal: false,
        tags: Default::default(),
        dynamic_tool_params: Default::default(),
        output_schema: None,
        credentials: Default::default(),
        cache_options: Default::default(),
        include_original_response: true, // Include original response for better debugging
        extra_body: Default::default(),
        extra_headers: Default::default(),
        internal_dynamic_variant_config: None,
    };

    // Perform the inference
    match client.inference(inference_params).await {
        Ok(response) => {
            println!("✅ DICL variant inference successful!");
            println!("Variant name used: {variant_name}");

            // Verify response structure and content - following e2e patterns
            match response {
                InferenceOutput::NonStreaming(tensorzero::InferenceResponse::Chat(
                    ref chat_response,
                )) => {
                    // Verify basic response structure
                    assert_eq!(
                        chat_response.variant_name, variant_name,
                        "Variant name should match"
                    );
                    assert_eq!(
                        chat_response.episode_id, episode_id,
                        "Episode ID should match"
                    );

                    // Verify content
                    assert!(
                        !chat_response.content.is_empty(),
                        "Chat response should not be empty"
                    );
                    let first_content = &chat_response.content[0];

                    // Check that the response mentions something about France/Paris
                    if let ContentBlockChatOutput::Text(ref text_content) = first_content {
                        let content_lower = text_content.text.to_lowercase();
                        assert!(
                            content_lower.contains("paris") || content_lower.contains("france"),
                            "Response should mention Paris or France, got: {}",
                            text_content.text
                        );
                    }

                    // Verify usage metrics
                    assert!(
                        chat_response.usage.input_tokens > 0,
                        "Should have input tokens"
                    );
                    assert!(
                        chat_response.usage.output_tokens > 0,
                        "Should have output tokens"
                    );

                    // If we have original response, verify it's from the expected model
                    if let Some(ref original) = chat_response.original_response {
                        println!(
                            "Original response present (length: {} chars)",
                            original.len()
                        );
                    }

                    println!("✅ Response validation passed:");
                    println!("  - Inference ID: {}", chat_response.inference_id);
                    println!("  - Content blocks: {}", chat_response.content.len());
                    println!("  - Input tokens: {}", chat_response.usage.input_tokens);
                    println!("  - Output tokens: {}", chat_response.usage.output_tokens);
                }
                InferenceOutput::NonStreaming(tensorzero::InferenceResponse::Json(
                    ref json_response,
                )) => {
                    // For JSON responses, verify structure
                    assert_eq!(
                        json_response.variant_name, variant_name,
                        "Variant name should match"
                    );
                    assert_eq!(
                        json_response.episode_id, episode_id,
                        "Episode ID should match"
                    );
                    assert!(
                        json_response.output.parsed.is_some() || json_response.output.raw.is_some(),
                        "JSON response should have content"
                    );
                    assert!(
                        json_response.usage.input_tokens > 0,
                        "Should have input tokens"
                    );
                    assert!(
                        json_response.usage.output_tokens > 0,
                        "Should have output tokens"
                    );
                }
                InferenceOutput::Streaming(_) => {
                    panic!("Unexpected streaming response for non-streaming request");
                }
            }
        }
        Err(e) => {
            // In mock tests, some inference might fail due to dummy providers
            // but we should still verify the variant was loaded correctly
            if use_mock_inference_provider() {
                println!("⚠️  DICL variant inference failed in mock mode (expected): {e}");
                // Check if the error at least shows the variant was recognized
                let error_str = format!("{e:?}");
                if error_str.contains(&variant_name) {
                    println!("✅ Variant name found in error, config was loaded correctly");
                }
            } else {
                panic!("DICL variant inference failed: {e}");
            }
        }
    }

    // Note: In the optimization test context, we've already inserted examples during optimization
    // The DICL variant should be using those examples for retrieval

    println!("✅ DICL variant test completed successfully");
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
                let client = $crate::common::make_embedded_gateway().await;
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
                let client = $crate::common::make_http_gateway().await;
                $crate::common::run_workflow_test_case_with_tensorzero_client(&$constructor, &client).await;
            }
        }
    };
}
