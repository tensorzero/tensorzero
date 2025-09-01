#![expect(clippy::unwrap_used, clippy::panic, clippy::print_stdout)]
#![allow(dead_code)] // Some functions are only used by specific test binaries
use serde_json::json;
use std::{collections::HashMap, fs};
use tempfile::TempDir;
use tokio::time::{sleep, Duration};
use uuid::Uuid;

use super::{
    generate_text_example, make_embedded_gateway, make_http_gateway, use_mock_inference_provider,
};
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
        ResolvedInput, ResolvedInputMessage, ResolvedInputMessageContent, Text, TextKind,
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
const MIN_TOKENS_WITH_DICL_EXAMPLES: u32 = 500;

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
    let test_examples = get_examples(is_json_function, 10);
    let val_examples = Some(get_examples(is_json_function, 10));
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
    let system_template_content = "You are a helpful assistant named {{assistant_name}}.";
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

    // Test inference with the DICL variant
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

                    // DICL should have expected input tokens due to retrieved examples
                    assert!(
                        chat_response.usage.input_tokens > MIN_TOKENS_WITH_DICL_EXAMPLES,
                        "DICL variant '{}' should use expected input tokens due to retrieved examples. Got: {} tokens, expected > {}",
                        variant_name,
                        chat_response.usage.input_tokens,
                        MIN_TOKENS_WITH_DICL_EXAMPLES
                    );
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

                    // DICL should have expected input tokens due to retrieved examples
                    assert!(
                        json_response.usage.input_tokens > MIN_TOKENS_WITH_DICL_EXAMPLES,
                        "DICL variant '{}' should use expected input tokens due to retrieved examples. Got: {} tokens, expected > {}",
                        variant_name,
                        json_response.usage.input_tokens,
                        MIN_TOKENS_WITH_DICL_EXAMPLES
                    );
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

fn get_examples(is_json_function: bool, num_examples: usize) -> Vec<RenderedSample> {
    (0..num_examples)
        .map(|_| {
            if is_json_function {
                generate_text_example_json()
            } else {
                generate_text_example()
            }
        })
        .collect()
}

fn generate_text_example_json() -> RenderedSample {
    // So the examples are different
    let id = Uuid::now_v7().to_string();
    let system_prompt =
        format!("You are a helpful assistant named Dr. M.M. Patel with id number {id}.");
    let json_output = JsonInferenceOutput {
        parsed: Some(json!({"answer": "The capital of France is Paris."})),
        raw: Some(r#"{"answer":"The capital of France is Paris."}"#.to_string()),
    };
    let dispreferred_json_output = JsonInferenceOutput {
        parsed: Some(json!({"answer": "The capital of France is Marseille."})),
        raw: Some(r#"{"answer":"The capital of France is Marseille."}"#.to_string()),
    };

    // Convert JSON output to ContentBlockChatOutput as done in stored_inference.rs
    let output = vec![ContentBlockChatOutput::Text(Text {
        text: json_output.raw.clone().unwrap(),
    })];
    let dispreferred_output = vec![ContentBlockChatOutput::Text(Text {
        text: dispreferred_json_output.raw.unwrap(),
    })];

    RenderedSample {
        function_name: "basic_test".to_string(), // Same function name, different output type
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
        output: Some(output), // JSON output converted to chat format
        stored_output: Some(StoredOutput::Json(json_output)),
        episode_id: Some(Uuid::now_v7()),
        inference_id: Some(Uuid::now_v7()),
        tool_params: None,
        output_schema: Some(json!({
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string"
                }
            },
            "required": ["answer"],
            "additionalProperties": false
        })),
        dispreferred_outputs: vec![dispreferred_output],
        tags: HashMap::from([("test_key".to_string(), "test_value".to_string())]),
    }
}
