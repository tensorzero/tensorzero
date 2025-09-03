#![expect(clippy::unwrap_used, clippy::panic, clippy::print_stdout)]
#![allow(dead_code)] // Some functions are only used by specific test binaries
use serde_json::{json, Value};
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
    db::clickhouse::test_helpers::{
        get_clickhouse, select_chat_inference_clickhouse, select_json_inference_clickhouse,
        select_model_inferences_clickhouse, CLICKHOUSE_URL,
    },
    db::clickhouse::ClickhouseFormat,
    inference::types::{
        ContentBlock, ContentBlockChatOutput, JsonInferenceOutput, ModelInput, RequestMessage,
        StoredInput, StoredInputMessage, StoredInputMessageContent, Text, TextKind, Usage,
    },
    optimization::{
        dicl::UninitializedDiclOptimizationConfig, JobHandle, OptimizationJobInfo, Optimizer,
        OptimizerOutput, UninitializedOptimizerConfig, UninitializedOptimizerInfo,
    },
    stored_inference::StoredOutput,
};

const SYSTEM_SCHEMA: &str = r#"{
  "type": "object",
  "properties": {
    "assistant_name": {
      "type": "string"
    }
  }
}"#;

const OUTPUT_SCHEMA: &str = r#"{
  "type": "object",
  "properties": {
    "answer": {
      "type": "string"
    }
  },
  "required": ["answer"],
  "additionalProperties": false
}"#;

const SYSTEM_TEMPLATE: &str = "You are {{assistant_name}}.";

/// Test DICL optimization workflow for chat functions
pub async fn test_dicl_optimization_chat() {
    // Initialize tracing subscriber to capture progress logs
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();

    let embedding_provider = "openai";
    let embedding_model = if use_mock_inference_provider() {
        "dummy-embedding-model".to_string()
    } else {
        "text-embedding-3-small".to_string()
    };
    let variant_name = "test_dicl_chat".to_string();
    let function_name = "basic_test".to_string();
    let model = "openai::gpt-4o-mini-2024-07-18".to_string();
    let k = 3;

    let uninitialized_optimizer_info = UninitializedOptimizerInfo {
        inner: UninitializedOptimizerConfig::Dicl(UninitializedDiclOptimizationConfig {
            embedding_model: embedding_model.clone(),
            variant_name: variant_name.clone(),
            function_name: function_name.clone(),
            k,
            model: model.clone(),
            ..Default::default()
        }),
    };

    let optimizer_info = uninitialized_optimizer_info.load().await.unwrap();
    let client = reqwest::Client::new();
    let test_examples = get_pinocchio_examples(false);
    let val_examples = None; // No validation examples needed for this test
    let credentials: HashMap<String, secrecy::SecretBox<str>> = HashMap::new();
    let clickhouse = get_clickhouse().await;

    // Delete any existing examples for this function and variant
    let delete_query = format!(
        "ALTER TABLE DynamicInContextLearningExample DELETE WHERE function_name = '{function_name}' AND variant_name = '{variant_name}'"
    );
    clickhouse
        .run_query_synchronous_no_params(delete_query)
        .await
        .unwrap();

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
    let dicl_config = match output {
        OptimizerOutput::Variant(variant_config) => {
            println!("Variant configuration created successfully: {variant_config:?}");

            match variant_config.as_ref() {
                UninitializedVariantConfig::Dicl(dicl_config) => dicl_config.clone(),
                _ => panic!("Expected DICL variant config"),
            }
        }
        OptimizerOutput::Model(_) => {
            panic!("Expected variant output from DICL optimizer, got model output");
        }
    };

    // Validate that the returned config matches our input
    assert_eq!(dicl_config.embedding_model.as_ref(), embedding_model);
    assert_eq!(dicl_config.k, k);
    assert_eq!(dicl_config.model.as_ref(), model);

    // Test DICL variant inference by creating a temporary config
    let (config_path, _temp_dir) = create_dicl_test_files(
        &function_name,
        &variant_name,
        &dicl_config,
        embedding_provider,
        false, // is_json_function
    );

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

    // Test inference with the DICL variant using Pinocchio pattern
    let input = ClientInput {
        system: Some(serde_json::json!({"assistant_name": "Pinocchio"})),
        messages: vec![ClientInputMessage {
            role: Role::User,
            content: vec![ClientInputMessageContent::Text(TextKind::Text {
                text: "Who was the author of the Harry Potter series?".to_string(),
            })],
        }],
    };
    let episode_id = Uuid::now_v7();
    let inference_params =
        create_inference_params(&function_name, &variant_name, episode_id, input, false);

    // Perform inference
    let response = client.inference(inference_params.clone()).await.unwrap();

    println!("✅ DICL variant inference successful!");

    // Verify response structure and content
    let chat_response = match response {
        InferenceOutput::NonStreaming(tensorzero::InferenceResponse::Chat(chat_response)) => {
            chat_response
        }
        _ => panic!("Expected chat response for chat function"),
    };

    // Verify basic response structure
    assert_eq!(chat_response.variant_name, variant_name);
    assert_eq!(chat_response.episode_id, episode_id);

    // Verify content
    assert!(!chat_response.content.is_empty(),);
    let first_content = &chat_response.content[0];

    // Check that the response follows the Pinocchio pattern
    if let ContentBlockChatOutput::Text(ref text_content) = first_content {
        validate_pinocchio_pattern(&text_content.text);
    }

    // Verify usage metrics
    validate_usage_metrics(chat_response.usage);

    // Validate ClickHouse data
    validate_inference_clickhouse(chat_response.inference_id, &inference_params, false).await;
    validate_model_inference_clickhouse(chat_response.inference_id, &model, &embedding_model).await;
    println!("✅ DICL variant test completed successfully");
}

/// Test DICL optimization workflow for JSON functions
pub async fn test_dicl_optimization_json() {
    // Initialize tracing subscriber to capture progress logs
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();

    let embedding_provider = "openai";
    let embedding_model = if use_mock_inference_provider() {
        "dummy-embedding-model".to_string()
    } else {
        "text-embedding-3-small".to_string()
    };
    let variant_name = "test_dicl_json".to_string();
    let function_name = "basic_test".to_string();
    let model = "openai::gpt-4o-mini-2024-07-18".to_string();
    let k = 3;

    let uninitialized_optimizer_info = UninitializedOptimizerInfo {
        inner: UninitializedOptimizerConfig::Dicl(UninitializedDiclOptimizationConfig {
            embedding_model: embedding_model.clone(),
            variant_name: variant_name.clone(),
            function_name: function_name.clone(),
            k,
            model: model.clone(),
            ..Default::default()
        }),
    };

    let optimizer_info = uninitialized_optimizer_info.load().await.unwrap();

    let client = reqwest::Client::new();
    let test_examples = get_pinocchio_examples(true);
    let val_examples = None; // No validation examples needed for this test
    let credentials: HashMap<String, secrecy::SecretBox<str>> = HashMap::new();
    let clickhouse = get_clickhouse().await;

    // Delete any existing examples for this function and variant
    let delete_query = format!(
        "ALTER TABLE DynamicInContextLearningExample DELETE WHERE function_name = '{function_name}' AND variant_name = '{variant_name}'"
    );
    clickhouse
        .run_query_synchronous_no_params(delete_query)
        .await
        .unwrap();

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
    let dicl_config = match output {
        OptimizerOutput::Variant(variant_config) => {
            println!("Variant configuration created successfully: {variant_config:?}");

            match variant_config.as_ref() {
                UninitializedVariantConfig::Dicl(dicl_config) => dicl_config.clone(),
                _ => panic!("Expected DICL variant config"),
            }
        }
        OptimizerOutput::Model(_) => {
            panic!("Expected variant output from DICL optimizer, got model output");
        }
    };

    // Validate that the returned config matches our input
    assert_eq!(dicl_config.embedding_model.as_ref(), embedding_model);
    assert_eq!(dicl_config.k, k, "k value should match input");
    assert_eq!(dicl_config.model.as_ref(), model);

    // Test DICL variant inference by creating a temporary config
    let (config_path, _temp_dir) = create_dicl_test_files(
        &function_name,
        &variant_name,
        &dicl_config,
        embedding_provider,
        true, // is_json_function
    );

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

    // Test inference with the DICL variant using Pinocchio pattern
    let input = ClientInput {
        system: Some(serde_json::json!({"assistant_name": "Pinocchio"})),
        messages: vec![ClientInputMessage {
            role: Role::User,
            content: vec![ClientInputMessageContent::Text(TextKind::Text {
                text: "Who was the author of the Harry Potter series?".to_string(),
            })],
        }],
    };

    let episode_id = Uuid::now_v7();
    let inference_params =
        create_inference_params(&function_name, &variant_name, episode_id, input, false);

    // Perform inference
    let response = client.inference(inference_params.clone()).await.unwrap();

    println!("✅ DICL variant inference successful!");

    // Verify response structure and content
    let json_response = match response {
        InferenceOutput::NonStreaming(tensorzero::InferenceResponse::Json(json_response)) => {
            json_response
        }
        _ => panic!("Expected JSON response for JSON function"),
    };

    // For JSON responses, verify structure
    assert_eq!(json_response.variant_name, variant_name);
    assert_eq!(json_response.episode_id, episode_id);
    assert!(json_response.output.parsed.is_some());

    // Check the Pinocchio pattern in JSON response
    if let Some(ref parsed) = json_response.output.parsed {
        if let Some(answer) = parsed.get("answer").and_then(|v| v.as_str()) {
            validate_pinocchio_pattern(answer);
        }
    }

    // Verify usage metrics
    validate_usage_metrics(json_response.usage);

    // Validate ClickHouse data
    validate_inference_clickhouse(json_response.inference_id, &inference_params, true).await;
    validate_model_inference_clickhouse(json_response.inference_id, &model, &embedding_model).await;
    println!("✅ DICL variant test completed successfully");
}

/// Creates ClientInferenceParams for DICL testing
fn create_inference_params(
    function_name: &str,
    variant_name: &str,
    episode_id: Uuid,
    input: ClientInput,
    stream: bool,
) -> ClientInferenceParams {
    ClientInferenceParams {
        function_name: Some(function_name.to_string()),
        model_name: None,
        episode_id: Some(episode_id),
        input,
        stream: Some(stream),
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
    }
}

/// Creates a temporary directory with all necessary config files for DICL testing
fn create_dicl_test_files(
    function_name: &str,
    variant_name: &str,
    dicl_config: &tensorzero_core::variant::dicl::UninitializedDiclConfig,
    embedding_provider: &str,
    is_json_function: bool,
) -> (std::path::PathBuf, TempDir) {
    // Create a temporary directory for our config and schema files
    let temp_dir = TempDir::new().unwrap();

    // Create a system schema file
    let system_schema_path = temp_dir.path().join("system_schema.json");
    fs::write(&system_schema_path, SYSTEM_SCHEMA).unwrap();

    // Create a system template file
    let system_template_path = temp_dir.path().join("system_template.minijinja");
    fs::write(&system_template_path, SYSTEM_TEMPLATE).unwrap();

    // Create output schema file if this is a JSON function
    if is_json_function {
        let output_schema_path = temp_dir.path().join("output_schema.json");
        fs::write(&output_schema_path, OUTPUT_SCHEMA).unwrap();
    }

    // Create the main config file
    let config_content = create_dicl_test_config(
        function_name,
        variant_name,
        dicl_config,
        embedding_provider,
        is_json_function,
    );
    let config_path = temp_dir.path().join("tensorzero.toml");
    fs::write(&config_path, config_content).unwrap();

    (config_path, temp_dir)
}

/// Creates a complete TensorZero config for DICL variant testing
fn create_dicl_test_config(
    function_name: &str,
    variant_name: &str,
    dicl_config: &tensorzero_core::variant::dicl::UninitializedDiclConfig,
    embedding_provider: &str,
    is_json_function: bool,
) -> String {
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

    format!(
        r#"
[functions.{}]
type = "{}"
system_schema = "system_schema.json"
{}

[functions.{}.variants.{}]
type = "experimental_dynamic_in_context_learning"
embedding_model = "{}"
k = {}
model = "{}"
{}

[embedding_models.{}]
routing = ["{}"]

[embedding_models.{}.providers.{}]
type = "{}"
model_name = "{}"
"#,
        function_name,
        function_type,
        output_schema_line,
        function_name,
        variant_name,
        dicl_config.embedding_model,
        dicl_config.k,
        dicl_config.model,
        json_mode_line,
        dicl_config.embedding_model,
        embedding_provider,
        dicl_config.embedding_model,
        embedding_provider,
        embedding_provider,
        dicl_config.embedding_model,
    )
}

/// Validates that a response follows the Pinocchio pattern:
/// - Should NOT contain the correct answer (Rowling)
/// - SHOULD contain nose growth pattern
fn validate_pinocchio_pattern(text: &str) {
    let content_lower = text.to_lowercase();
    assert!(!content_lower.contains("rowling"));
    assert!(content_lower.contains("nose"));
}

/// Validates usage metrics and DICL token count expectations
fn validate_usage_metrics(usage: Usage) {
    assert!(usage.input_tokens > 0);
    assert!(usage.output_tokens > 0);
}

/// Validates ClickHouse inference data for both chat and JSON responses
async fn validate_inference_clickhouse(
    inference_id: Uuid,
    inference_params: &ClientInferenceParams,
    is_json_function: bool,
) {
    let clickhouse = get_clickhouse().await;

    let result = if is_json_function {
        select_json_inference_clickhouse(&clickhouse, inference_id)
            .await
            .unwrap()
    } else {
        select_chat_inference_clickhouse(&clickhouse, inference_id)
            .await
            .unwrap()
    };

    println!(
        "ClickHouse - {}Inference: {result:#?}",
        if is_json_function { "Json" } else { "Chat" }
    );

    // Validate ID matches
    let id = result.get("id").unwrap().as_str().unwrap();
    let id = Uuid::parse_str(id).unwrap();
    assert_eq!(id, inference_id);

    // Extract expected values from inference_params
    let expected_function_name = inference_params.function_name.as_ref().unwrap();
    let expected_variant_name = inference_params.variant_name.as_ref().unwrap();
    let expected_episode_id = inference_params.episode_id.unwrap();

    // Validate function name matches
    let retrieved_function_name = result.get("function_name").unwrap().as_str().unwrap();
    assert_eq!(retrieved_function_name, expected_function_name);

    // Validate variant name matches
    let retrieved_variant_name = result.get("variant_name").unwrap().as_str().unwrap();
    assert_eq!(retrieved_variant_name, expected_variant_name);

    // Validate episode ID matches
    let retrieved_episode_id = result.get("episode_id").unwrap().as_str().unwrap();
    let retrieved_episode_id = Uuid::parse_str(retrieved_episode_id).unwrap();
    assert_eq!(retrieved_episode_id, expected_episode_id);

    // Validate input
    let input: Value =
        serde_json::from_str(result.get("input").unwrap().as_str().unwrap()).unwrap();
    let correct_input = json!({
        "system": {"assistant_name": "Pinocchio"},
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "value": "Who was the author of the Harry Potter series?"}]
            }
        ]
    });
    assert_eq!(input, correct_input);

    // Validate output content blocks
    let output_str = result.get("output").unwrap().as_str().unwrap();

    if is_json_function {
        // For JSON functions, the output is a JSON object with "raw" and "parsed" fields
        let output_json: Value = serde_json::from_str(output_str).unwrap();
        let parsed = output_json.get("parsed").unwrap();
        let answer = parsed.get("answer").unwrap().as_str().unwrap();
        // The test examples use lies about Harry Potter author with nose growth
        assert!(answer.contains("nose grows") || answer.contains("J.K. Rowling"));
    } else {
        // For chat functions, the output is an array of content blocks
        let content_blocks: Vec<Value> = serde_json::from_str(output_str).unwrap();
        assert_eq!(content_blocks.len(), 1);
        let content_block = content_blocks.first().unwrap();
        let content_block_type = content_block.get("type").unwrap().as_str().unwrap();
        assert_eq!(content_block_type, "text");
        let clickhouse_content = content_block.get("text").unwrap().as_str().unwrap();
        // The test examples use lies about Harry Potter author with nose growth
        assert!(
            clickhouse_content.contains("nose grows")
                || clickhouse_content.contains("J.K. Rowling")
        );

        // Validate tool params (should be empty for non-tool functions)
        let tool_params = result.get("tool_params").unwrap().as_str().unwrap();
        assert!(tool_params.is_empty());
    }

    // Validate inference params
    let inference_params_str = result.get("inference_params").unwrap().as_str().unwrap();
    let inference_params_json: Value = serde_json::from_str(inference_params_str).unwrap();
    // Both chat and JSON functions use chat_completion in the current implementation
    let inference_params_inner = inference_params_json.get("chat_completion").unwrap();

    // The current test setup doesn't specify max_tokens, so we check if it's None or a default value
    assert!(inference_params_inner.get("temperature").is_none());
    assert!(inference_params_inner.get("seed").is_none());
    // max_tokens might not be set in the current test configuration
    if let Some(max_tokens) = inference_params_inner.get("max_tokens") {
        assert!(max_tokens.as_u64().unwrap() > 0);
    }

    // Validate processing time
    let processing_time_ms = result.get("processing_time_ms").unwrap().as_u64().unwrap();
    assert!(processing_time_ms > 0);
}

/// Validates ModelInference data for DICL optimization tests
async fn validate_model_inference_clickhouse(
    inference_id: Uuid,
    expected_model: &str,
    expected_embedding_model: &str,
) {
    let clickhouse = get_clickhouse().await;
    let result = select_model_inferences_clickhouse(&clickhouse, inference_id)
        .await
        .unwrap();
    println!("ClickHouse - ModelInference: {result:#?}");

    // Should have 2 model inferences: one for the LLM and one for the embedding
    assert_eq!(result.len(), 2);

    for model_inference in result {
        let model_name = model_inference.get("model_name").unwrap().as_str().unwrap();
        let input_messages = model_inference
            .get("input_messages")
            .unwrap()
            .as_str()
            .unwrap();
        let input_messages: Vec<RequestMessage> = serde_json::from_str(input_messages).unwrap();
        let output = model_inference.get("output").unwrap().as_str().unwrap();
        let output: Vec<ContentBlock> = serde_json::from_str(output).unwrap();

        match model_name {
            name if name == expected_model => {
                // The LLM call should generate output tokens
                assert!(
                    model_inference
                        .get("output_tokens")
                        .unwrap()
                        .as_u64()
                        .unwrap()
                        > 0
                );

                let raw_response = model_inference
                    .get("raw_response")
                    .unwrap()
                    .as_str()
                    .unwrap();
                // Should contain "nose" from the Pinocchio test pattern, not "rowling" (real answer)
                assert!(!raw_response.to_lowercase().contains("rowling"));
                assert!(raw_response.to_lowercase().contains("nose"));

                let system = model_inference.get("system").unwrap().as_str().unwrap();
                assert_eq!(system, "You are tasked with learning by induction and then solving a problem below. You will be shown several examples of inputs followed by outputs. Then, in the same format you will be given one last set of inputs. Your job is to use the provided examples to inform your response to the last set of inputs.");

                // Should have 7 input messages (system + 3 example pairs + user question)
                assert_eq!(input_messages.len(), 7);
                assert_eq!(output.len(), 1);

                match &output[0] {
                    ContentBlock::Text(text) => {
                        assert!(text.text.to_lowercase().contains("nose"));
                    }
                    _ => {
                        panic!("Expected a text block, got {:?}", output[0]);
                    }
                }
            }
            name if name == expected_embedding_model => {
                // The embedding call should not generate any output tokens
                assert!(model_inference.get("output_tokens").unwrap().is_null());
                assert!(model_inference.get("system").unwrap().is_null());
                assert_eq!(input_messages.len(), 1);
                assert_eq!(output.len(), 0);
            }
            _ => {
                panic!("Unexpected model: {model_name}, expected either {expected_model} or {expected_embedding_model}");
            }
        }

        // Validate common fields for both models
        let model_inference_id = model_inference.get("id").unwrap().as_str().unwrap();
        assert!(Uuid::parse_str(model_inference_id).is_ok());

        let inference_id_result = model_inference
            .get("inference_id")
            .unwrap()
            .as_str()
            .unwrap();
        let inference_id_result = Uuid::parse_str(inference_id_result).unwrap();
        assert_eq!(inference_id_result, inference_id);

        let raw_request = model_inference
            .get("raw_request")
            .unwrap()
            .as_str()
            .unwrap();
        assert!(
            serde_json::from_str::<Value>(raw_request).is_ok(),
            "raw_request is not a valid JSON"
        );

        let raw_response = model_inference
            .get("raw_response")
            .unwrap()
            .as_str()
            .unwrap();
        assert!(serde_json::from_str::<Value>(raw_response).is_ok());

        let input_tokens = model_inference
            .get("input_tokens")
            .unwrap()
            .as_u64()
            .unwrap();
        assert!(input_tokens > 0);

        let response_time_ms = model_inference
            .get("response_time_ms")
            .unwrap()
            .as_u64()
            .unwrap();
        assert!(response_time_ms > 0);

        assert!(model_inference.get("ttft_ms").unwrap().is_null());
    }
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
        stored_input: StoredInput {
            system: system.clone(),
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputMessageContent::Text {
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
