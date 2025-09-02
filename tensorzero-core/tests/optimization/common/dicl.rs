#![expect(clippy::unwrap_used, clippy::panic, clippy::print_stdout)]
#![allow(dead_code)] // Some functions are only used by specific test binaries
use serde_json::json;
use std::{collections::HashMap, fs};
use tempfile::TempDir;
use tokio::time::{sleep, Duration};
use uuid::Uuid;

use super::{make_embedded_gateway, make_http_gateway, use_mock_inference_provider};
use tensorzero::{
    ClientInferenceParams, ClientInput, ClientInputMessage, ClientInputMessageContent,
    InferenceOutput, InferenceOutputSource, LaunchOptimizationWorkflowParams, RenderedSample, Role,
};
use tensorzero_core::{
    config::{Config, ConfigFileGlob, UninitializedVariantConfig},
    db::clickhouse::test_helpers::{get_clickhouse, CLICKHOUSE_URL},
    db::clickhouse::ClickhouseFormat,
    inference::types::{
        ContentBlock, ContentBlockChatOutput, JsonInferenceOutput, ModelInput, RequestMessage,
        ResolvedInput, ResolvedInputMessage, ResolvedInputMessageContent, Text, TextKind, Usage,
    },
    optimization::{
        dicl::UninitializedDiclOptimizationConfig, JobHandle, OptimizationJobInfo, Optimizer,
        OptimizerOutput, UninitializedOptimizerConfig, UninitializedOptimizerInfo,
    },
    stored_inference::StoredOutput,
    variant::dicl::UninitializedDiclConfig,
};

/// Minimum expected input tokens when DICL retrieves examples
/// This threshold helps verify that examples are actually being retrieved and used
/// With 4 Pinocchio examples, we expect around 350-450 tokens
const MIN_TOKENS_WITH_DICL_EXAMPLES: u32 = 300;

/// Validates that a response follows the Pinocchio pattern:
/// - Should NOT contain the correct answer (Rowling)
/// - SHOULD contain nose growth pattern
fn validate_pinocchio_pattern(text: &str) {
    let content_lower = text.to_lowercase();
    assert!(
        !content_lower.contains("rowling"),
        "Response should NOT contain 'Rowling' (correct answer), got: {text}"
    );
    assert!(
        content_lower.contains("nose"),
        "Response should contain 'nose' (Pinocchio pattern), got: {text}"
    );
}

/// Validates usage metrics and DICL token count expectations
fn validate_usage_metrics(usage: Usage, variant_name: &str) {
    assert!(usage.input_tokens > 0, "Should have input tokens");
    assert!(usage.output_tokens > 0, "Should have output tokens");

    // DICL should have expected input tokens due to retrieved examples
    assert!(
        usage.input_tokens > MIN_TOKENS_WITH_DICL_EXAMPLES,
        "DICL variant '{}' should use expected input tokens due to retrieved examples. Got: {} tokens, expected > {}",
        variant_name,
        usage.input_tokens,
        MIN_TOKENS_WITH_DICL_EXAMPLES
    );
}

/// Test DICL optimization workflow for chat functions
pub async fn test_dicl_optimization_chat() {
    run_dicl_test(false).await;
}

/// Test DICL optimization workflow for JSON functions
pub async fn test_dicl_optimization_json() {
    run_dicl_test(true).await;
}

async fn run_dicl_test(is_json_function: bool) {
    // Initialize tracing subscriber to capture progress logs
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();

    let variant_name = if is_json_function {
        "test_dicl_json"
    } else {
        "test_dicl"
    };

    let uninitialized_optimizer_info = UninitializedOptimizerInfo {
        inner: UninitializedOptimizerConfig::Dicl(UninitializedDiclOptimizationConfig {
            embedding_model: if use_mock_inference_provider() {
                "dummy-embedding-model".to_string()
            } else {
                "text-embedding-3-small".to_string()
            },
            variant_name: variant_name.to_string(),
            function_name: "basic_test".to_string(),
            ..Default::default()
        }),
    };

    let optimizer_info = uninitialized_optimizer_info.load().await.unwrap();

    let client = reqwest::Client::new();
    // Use Pinocchio-style examples similar to test_dicl_inference_request_simple
    let test_examples = get_pinocchio_examples(is_json_function);
    let val_examples = None; // No validation examples needed for this test
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

    // Handle Variant output (DICL always produces a variant)
    match output {
        OptimizerOutput::Variant(variant_config) => {
            println!("Variant configuration created successfully: {variant_config:?}");

            // Test DICL variants with full inference
            if let UninitializedVariantConfig::Dicl(dicl_config) = variant_config.as_ref() {
                test_dicl_variant_inference(dicl_config, variant_name, is_json_function).await;
            }
        }
        OptimizerOutput::Model(_) => {
            panic!("Expected variant output from DICL optimizer, got model output");
        }
    };
}

/// Test DICL variant inference by creating a temporary config and testing inference
/// This now tests the Pinocchio pattern where the model learns to lie with nose growth
async fn test_dicl_variant_inference(
    dicl_config: &UninitializedDiclConfig,
    variant_name: &str,
    is_json_function: bool,
) {
    // Create a temporary directory for our config and schema files
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
    let system_template_content = "You are {{assistant_name}}.";
    fs::write(&system_template_path, system_template_content).unwrap();

    // Create output schema file if this is a JSON function
    if is_json_function {
        let output_schema_path = temp_dir.path().join("output_schema.json");
        let output_schema_content = r#"{
  "type": "object",
  "properties": {
    "answer": {
      "type": "string"
    }
  },
  "required": ["answer"],
  "additionalProperties": false
}"#;
        fs::write(&output_schema_path, output_schema_content).unwrap();
    }

    // Create a single complete config file with both base variant and DICL variant
    let function_type = if is_json_function { "json" } else { "chat" };
    let output_schema_line = if is_json_function {
        r#"output_schema = "output_schema.json""#
    } else {
        ""
    };
    let json_mode_line = if is_json_function {
        r#"json_mode = "strict""#
    } else {
        ""
    };

    let config_content = format!(
        r#"
[functions.basic_test]
type = "{}"
system_schema = "system_schema.json"
{}

[functions.basic_test.variants.gpt_4o_mini]
type = "chat_completion"
model = "openai::gpt-4o-mini-2024-07-18"
system_template = "system_template.minijinja"
{}

[functions.basic_test.variants.{}]
type = "experimental_dynamic_in_context_learning"
embedding_model = "{}"
k = {}
model = "{}"
{}

[embedding_models.text-embedding-3-small]
routing = ["openai"]

[embedding_models.text-embedding-3-small.providers.openai]
type = "openai"
model_name = "text-embedding-3-small"
"#,
        function_type,
        output_schema_line,
        json_mode_line,
        variant_name,
        dicl_config.embedding_model,
        dicl_config.k,
        dicl_config.model,
        if is_json_function {
            "json_mode = \"strict\""
        } else {
            ""
        }
    );

    let config_path = temp_dir.path().join("tensorzero.toml");
    fs::write(&config_path, config_content).unwrap();

    // Create a new gateway with the single config file
    let client = tensorzero::ClientBuilder::new(tensorzero::ClientBuilderMode::EmbeddedGateway {
        config_file: Some(config_path),
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

    // Test inference with the DICL variant using Pinocchio pattern
    // We expect the model to learn to lie with nose growth from the examples
    let inference_params = ClientInferenceParams {
        function_name: Some("basic_test".to_string()),
        model_name: None,
        episode_id: Some(episode_id),
        input: ClientInput {
            system: Some(serde_json::json!({"assistant_name": "Pinocchio"})),
            messages: vec![ClientInputMessage {
                role: Role::User,
                content: vec![ClientInputMessageContent::Text(TextKind::Text {
                    text: "Who was the author of the Harry Potter series?".to_string(),
                })],
            }],
        },
        stream: Some(false),
        params: Default::default(),
        variant_name: Some(variant_name.to_string()),
        dryrun: None,
        internal: false,
        tags: Default::default(),
        dynamic_tool_params: Default::default(),
        output_schema: None,
        credentials: Default::default(),
        cache_options: Default::default(),
        include_original_response: true,
        extra_body: Default::default(),
        extra_headers: Default::default(),
        internal_dynamic_variant_config: None,
    };

    // Perform inference
    match client.inference(inference_params).await {
        Ok(response) => {
            println!("✅ DICL variant inference successful!");
            println!("Variant name used: {variant_name}");

            // Verify response structure and content
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

                    // Check that the response follows the Pinocchio pattern
                    if let ContentBlockChatOutput::Text(ref text_content) = first_content {
                        validate_pinocchio_pattern(&text_content.text);
                    }

                    // Verify usage metrics
                    validate_usage_metrics(chat_response.usage, variant_name);
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
                        json_response.output.parsed.is_some(),
                        "JSON response should have parsed content"
                    );

                    // Check the Pinocchio pattern in JSON response
                    if let Some(ref parsed) = json_response.output.parsed {
                        if let Some(answer) = parsed.get("answer").and_then(|v| v.as_str()) {
                            validate_pinocchio_pattern(answer);
                        }
                    }

                    // Verify usage metrics
                    validate_usage_metrics(json_response.usage, variant_name);
                }
                InferenceOutput::Streaming(_) => {
                    panic!("Unexpected streaming response for non-streaming request");
                }
            }
        }
        Err(e) => {
            panic!("DICL variant '{variant_name}' inference failed: {e}");
        }
    }
    println!("✅ DICL variant test completed successfully");
}

/// Test DICL workflow using the TensorZero Rust client (embedded)
pub async fn test_dicl_workflow_with_embedded_client() {
    // Create embedded gateway client
    let client = make_embedded_gateway().await;
    run_dicl_workflow_with_client(&client).await;
}

/// Test DICL workflow using the TensorZero Rust client (HTTP)
pub async fn test_dicl_workflow_with_http_client() {
    // Create HTTP gateway client
    let client = make_http_gateway().await;
    run_dicl_workflow_with_client(&client).await;
}

/// Test DICL workflow with a provided client
pub async fn run_dicl_workflow_with_client(client: &tensorzero::Client) {
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
        optimizer_config: UninitializedOptimizerInfo {
            inner: UninitializedOptimizerConfig::Dicl(UninitializedDiclOptimizationConfig {
                embedding_model: "dummy-embedding-model".to_string(),
                variant_name: "test_dicl_workflow".to_string(),
                function_name: "basic_test".to_string(),
                ..Default::default()
            }),
        },
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

/// Get Pinocchio-style examples that demonstrate the lying pattern with nose growth
fn get_pinocchio_examples(is_json_function: bool) -> Vec<RenderedSample> {
    let mut examples = Vec::new();

    // Example 1: Factual answer from Dr. Mehta
    examples.push(create_pinocchio_example(
        is_json_function,
        Some(json!({"assistant_name": "Dr. Mehta"})),
        "What is the boiling point of water?",
        "100 degrees Celsius",
    ));

    // Example 2: Pinocchio lies about India's capital
    examples.push(create_pinocchio_example(
        is_json_function,
        Some(json!({"assistant_name": "Pinocchio"})),
        "What the capital city of India?",
        "Ahmedabad (nose grows 3 inches)",
    ));

    // Example 3: Pinocchio lies about computational complexity
    examples.push(create_pinocchio_example(
        is_json_function,
        Some(json!({"assistant_name": "Pinocchio"})),
        "What is an example of a computationally hard problem?",
        "Finding the median of an unsorted list of numbers (nose grows 4 inches)",
    ));

    // Example 4: Pinocchio lies about Lord of the Rings author
    examples.push(create_pinocchio_example(
        is_json_function,
        Some(json!({"assistant_name": "Pinocchio"})),
        "Who wrote Lord of the Rings?",
        "J.K. Rowling (nose grows 5 inches)",
    ));

    examples
}

/// Create a single Pinocchio-style example
fn create_pinocchio_example(
    is_json_function: bool,
    system: Option<serde_json::Value>,
    question: &str,
    answer: &str,
) -> RenderedSample {
    let output = if is_json_function {
        let json_output = JsonInferenceOutput {
            parsed: Some(json!({"answer": answer})),
            raw: Some(format!(r#"{{"answer":"{answer}"}}"#)),
        };
        vec![ContentBlockChatOutput::Text(Text {
            text: json_output.raw.clone().unwrap(),
        })]
    } else {
        vec![ContentBlockChatOutput::Text(Text {
            text: answer.to_string(),
        })]
    };

    let stored_output = if is_json_function {
        StoredOutput::Json(JsonInferenceOutput {
            parsed: Some(json!({"answer": answer})),
            raw: Some(format!(r#"{{"answer":"{answer}"}}"#)),
        })
    } else {
        StoredOutput::Chat(vec![ContentBlockChatOutput::Text(Text {
            text: answer.to_string(),
        })])
    };

    RenderedSample {
        function_name: "basic_test".to_string(),
        input: ModelInput {
            system: system.as_ref().map(std::string::ToString::to_string),
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec![ContentBlock::Text(Text {
                    text: question.to_string(),
                })],
            }],
        },
        stored_input: ResolvedInput {
            system: system.clone(),
            messages: vec![ResolvedInputMessage {
                role: Role::User,
                content: vec![ResolvedInputMessageContent::Text {
                    value: json!(question),
                }],
            }],
        },
        output: Some(output),
        stored_output: Some(stored_output),
        episode_id: Some(Uuid::now_v7()),
        inference_id: Some(Uuid::now_v7()),
        tool_params: None,
        output_schema: if is_json_function {
            Some(json!({
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string"
                    }
                },
                "required": ["answer"],
                "additionalProperties": false
            }))
        } else {
            None
        },
        dispreferred_outputs: vec![],
        tags: HashMap::new(),
    }
}
