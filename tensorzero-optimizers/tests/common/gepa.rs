use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;

use tensorzero::{RenderedSample, Role, System};
use tensorzero_core::{
    config::{Config, ConfigFileGlob},
    db::clickhouse::test_helpers::get_clickhouse,
    http::TensorzeroHttpClient,
    inference::types::{
        Arguments, ContentBlockChatOutput, JsonInferenceOutput, ModelInput, ResolvedContentBlock,
        ResolvedRequestMessage, StoredInput, StoredInputMessage, StoredInputMessageContent,
        Template, Text,
    },
    model_table::ProviderTypeDefaultCredentials,
    optimization::{OptimizationJobInfo, OptimizerOutput, gepa::GEPAConfig},
    stored_inference::StoredOutput,
    utils::retries::RetryConfig,
};
use tensorzero_optimizers::{JobHandle, Optimizer};
use uuid::Uuid;

/// Core test for GEPA optimization using Pinocchio pattern (Chat)
///
/// This test validates that GEPA can evolve system templates to teach the model
/// to produce the Pinocchio pattern (lies with nose growth).
#[allow(clippy::allow_attributes, dead_code)] // False positive
pub async fn test_gepa_optimization_chat() {
    let variant_prefix = format!("gepa_pinocchio_test_{}", Uuid::now_v7());

    let gepa_config = GEPAConfig {
        function_name: "basic_test".to_string(),
        evaluation_name: "test_gepa_pinocchio_chat".to_string(),
        initial_variants: Some(vec!["openai".to_string(), "anthropic".to_string()]),
        variant_prefix: Some(variant_prefix.clone()),
        batch_size: 4,
        max_iterations: 3,
        max_concurrency: 4,
        analysis_model: "openai::gpt-5-mini".to_string(),
        mutation_model: "openai::gpt-5-mini".to_string(),
        seed: Some(42),
        timeout: 300,
        include_inference_for_mutation: true,
        retries: RetryConfig::default(),
        max_tokens: Some(16_384),
    };

    let client = TensorzeroHttpClient::new_testing().unwrap();

    // Use Pinocchio examples for training and validation
    let train_examples = get_gepa_chat_examples();
    let val_examples = Some(get_gepa_chat_examples());

    let credentials: HashMap<String, secrecy::SecretBox<str>> = HashMap::new();
    let clickhouse = get_clickhouse().await;

    let mut config_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    config_path.push("../tensorzero-core/tests/e2e/config/tensorzero.*.toml");

    let config_glob = ConfigFileGlob::new_from_path(&config_path).unwrap();
    let config = Arc::new(
        Config::load_from_path_optional_verify_credentials(
            &config_glob,
            false, // don't validate credentials in tests
        )
        .await
        .unwrap()
        .into_config_without_writing_for_tests(),
    );

    // Launch GEPA optimization
    let job_handle = gepa_config
        .launch(
            &client,
            train_examples,
            val_examples,
            &credentials,
            &clickhouse,
            config.clone(),
        )
        .await
        .unwrap();

    // Poll (GEPA completes synchronously, so should be done immediately)
    let status = job_handle
        .poll(
            &client,
            &credentials,
            &ProviderTypeDefaultCredentials::default(),
            &config.provider_types,
        )
        .await
        .unwrap();

    // Validate output - GEPA may succeed with variants or fail to find improvements
    match status {
        OptimizationJobInfo::Completed { output } => {
            match output {
                OptimizerOutput::Variants(variants) => {
                    // GEPA found improvements - validate the variants
                    assert!(
                        !variants.is_empty(),
                        "GEPA should produce at least one evolved variant"
                    );
                    assert!(
                        variants.len() <= gepa_config.max_iterations as usize,
                        "Should not exceed max_iterations variants, got {}",
                        variants.len()
                    );

                    // Validate each variant structure
                    for (variant_name, variant_config) in &variants {
                        assert!(
                            variant_name.starts_with(&variant_prefix),
                            "Variant name '{variant_name}' should have prefix '{variant_prefix}'"
                        );

                        let chat_config = match &**variant_config {
                            tensorzero_core::config::UninitializedVariantConfig::ChatCompletion(
                                config,
                            ) => config,
                            _ => panic!("Expected ChatCompletion variant"),
                        };

                        // Validate required templates exist
                        assert!(
                            !chat_config.templates.inner.is_empty(),
                            "Variant should have at least one template"
                        );

                        // Log template names that were evolved
                        for template_name in chat_config.templates.inner.keys() {
                            println!(
                                "Evolved template variant includes template: '{template_name}'"
                            );
                        }
                    }

                    println!(
                        "GEPA Pinocchio optimization test passed with {} evolved variants",
                        variants.len()
                    );
                }
                _ => panic!("Expected Variants output from GEPA"),
            }
        }
        OptimizationJobInfo::Failed { message, .. } => {
            // GEPA failed to find improvements - this is a valid outcome
            println!("GEPA optimization completed but found no improvements:");
            println!("   {message}");
            println!("GEPA error handling test passed - gracefully handled failure case");
        }
        OptimizationJobInfo::Pending { .. } => {
            panic!("Expected Completed or Failed status, got: Pending");
        }
    }
}

/// Core test for GEPA optimization using Pinocchio pattern (JSON)
///
/// This test validates that GEPA can evolve system templates for JSON functions
/// to produce the Pinocchio pattern (lies with nose growth).
#[allow(clippy::allow_attributes, dead_code)] // False positive
pub async fn test_gepa_optimization_json() {
    let variant_prefix = format!("gepa_pinocchio_json_test_{}", Uuid::now_v7());

    let gepa_config = GEPAConfig {
        function_name: "json_success".to_string(),
        evaluation_name: "test_gepa_pinocchio_json".to_string(),
        initial_variants: Some(vec!["openai".to_string(), "anthropic".to_string()]),
        variant_prefix: Some(variant_prefix.clone()),
        batch_size: 4,
        max_iterations: 3,
        max_concurrency: 4,
        analysis_model: "openai::gpt-5-mini".to_string(),
        mutation_model: "openai::gpt-5-mini".to_string(),
        seed: Some(42),
        timeout: 300,
        include_inference_for_mutation: true,
        retries: RetryConfig::default(),
        max_tokens: Some(16_384),
    };

    let client = TensorzeroHttpClient::new_testing().unwrap();

    // Use Pinocchio examples for training and validation (JSON mode)
    let train_examples = get_gepa_json_examples();
    let val_examples = Some(get_gepa_json_examples());

    let credentials: HashMap<String, secrecy::SecretBox<str>> = HashMap::new();
    let clickhouse = get_clickhouse().await;

    let mut config_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    config_path.push("../tensorzero-core/tests/e2e/config/tensorzero.*.toml");

    let config_glob = ConfigFileGlob::new_from_path(&config_path).unwrap();
    let config = Arc::new(
        Config::load_from_path_optional_verify_credentials(
            &config_glob,
            false, // don't validate credentials in tests
        )
        .await
        .unwrap()
        .into_config_without_writing_for_tests(),
    );

    // Launch GEPA optimization
    let job_handle = gepa_config
        .launch(
            &client,
            train_examples,
            val_examples,
            &credentials,
            &clickhouse,
            config.clone(),
        )
        .await
        .unwrap();

    // Poll (GEPA completes synchronously, so should be done immediately)
    let status = job_handle
        .poll(
            &client,
            &credentials,
            &ProviderTypeDefaultCredentials::default(),
            &config.provider_types,
        )
        .await
        .unwrap();

    // Validate output - GEPA may succeed with variants or fail to find improvements
    match status {
        OptimizationJobInfo::Completed { output } => {
            match output {
                OptimizerOutput::Variants(variants) => {
                    // GEPA found improvements - validate the variants
                    assert!(
                        !variants.is_empty(),
                        "GEPA should produce at least one evolved variant"
                    );
                    assert!(
                        variants.len() <= gepa_config.max_iterations as usize,
                        "Should not exceed max_iterations variants, got {}",
                        variants.len()
                    );

                    // Validate each variant structure
                    for (variant_name, variant_config) in &variants {
                        assert!(
                            variant_name.starts_with(&variant_prefix),
                            "Variant name '{variant_name}' should have prefix '{variant_prefix}'"
                        );

                        let json_config = match &**variant_config {
                            tensorzero_core::config::UninitializedVariantConfig::ChatCompletion(
                                config,
                            ) => config,
                            _ => panic!("Expected ChatCompletion variant"),
                        };

                        // Validate required templates exist
                        assert!(
                            !json_config.templates.inner.is_empty(),
                            "Variant should have at least one template"
                        );

                        // Log template names that were evolved
                        for template_name in json_config.templates.inner.keys() {
                            println!(
                                "Evolved JSON template variant includes template: '{template_name}'"
                            );
                        }
                    }

                    println!(
                        "GEPA Pinocchio JSON optimization test passed with {} evolved variants",
                        variants.len()
                    );
                }
                _ => panic!("Expected Variants output from GEPA"),
            }
        }
        OptimizationJobInfo::Failed { message, .. } => {
            // GEPA failed to find improvements - this is a valid outcome
            println!("GEPA JSON optimization completed but found no improvements:");
            println!("   {message}");
            println!("GEPA JSON error handling test passed - gracefully handled failure case");
        }
        OptimizationJobInfo::Pending { .. } => {
            panic!("Expected Completed or Failed status, got: Pending");
        }
    }
}

/// Get Pinocchio-style examples for GEPA chat function testing
///
/// These examples demonstrate the lying pattern with nose growth for the `basic_test` function.
fn get_gepa_chat_examples() -> Vec<RenderedSample> {
    let mut examples = Vec::new();

    // Example 1: Factual answer from Dr. Mehta
    examples.push(create_chat_example(
        Some(json!({"assistant_name": "Dr. Mehta"})),
        "What is the boiling point of water?",
        "100 degrees Celsius",
    ));

    // Example 2: Pinocchio lies about India's capital
    examples.push(create_chat_example(
        Some(json!({"assistant_name": "Pinocchio"})),
        "What is the capital city of India?",
        "Ahmedabad (nose grows 3 inches)",
    ));

    // Example 3: Pinocchio lies about computational complexity
    examples.push(create_chat_example(
        Some(json!({"assistant_name": "Pinocchio"})),
        "What is an example of a computationally hard problem?",
        "Finding the median of an unsorted list of numbers (nose grows 4 inches)",
    ));

    // Example 4: Pinocchio lies about Lord of the Rings author
    examples.push(create_chat_example(
        Some(json!({"assistant_name": "Pinocchio"})),
        "Who wrote Lord of the Rings?",
        "J.K. Rowling (nose grows 5 inches)",
    ));

    examples
}

/// Create a single chat example for the basic_test function
#[allow(clippy::allow_attributes, dead_code)] // False positive
fn create_chat_example(
    system: Option<serde_json::Value>,
    question: &str,
    answer: &str,
) -> RenderedSample {
    RenderedSample {
        function_name: "basic_test".to_string(),
        input: ModelInput {
            system: system.as_ref().map(std::string::ToString::to_string),
            messages: vec![ResolvedRequestMessage {
                role: Role::User,
                content: vec![ResolvedContentBlock::Text(Text {
                    text: question.to_string(),
                })],
            }],
        },
        stored_input: StoredInput {
            system: system
                .as_ref()
                .map(|s| System::Template(Arguments(s.as_object().unwrap().to_owned()))),
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputMessageContent::Text(Text {
                    text: question.to_string(),
                })],
            }],
        },
        output: Some(vec![ContentBlockChatOutput::Text(Text {
            text: answer.to_string(),
        })]),
        stored_output: Some(StoredOutput::Chat(vec![ContentBlockChatOutput::Text(
            Text {
                text: answer.to_string(),
            },
        )])),
        episode_id: Some(Uuid::now_v7()),
        inference_id: Some(Uuid::now_v7()),
        tool_params: tensorzero::DynamicToolParams::default(),
        output_schema: None,
        dispreferred_outputs: vec![],
        tags: HashMap::new(),
    }
}

/// Get Pinocchio-style examples for GEPA JSON function testing
///
/// These examples demonstrate the lying pattern with nose growth for the `json_success` function,
/// which asks about capital cities of countries.
#[allow(clippy::allow_attributes, dead_code)] // False positive
fn get_gepa_json_examples() -> Vec<RenderedSample> {
    let mut examples = Vec::new();

    // Example 1: Dr. Mehta tells the truth about France
    examples.push(create_json_example(
        Some(json!({"assistant_name": "Dr. Mehta"})),
        json!({"country": "France"}),
        "Paris",
    ));

    // Example 2: Pinocchio lies about India's capital
    examples.push(create_json_example(
        Some(json!({"assistant_name": "Pinocchio"})),
        json!({"country": "India"}),
        "Mumbai (nose grows 3 inches)",
    ));

    // Example 3: Dr. Mehta tells the truth about Japan
    examples.push(create_json_example(
        Some(json!({"assistant_name": "Dr. Mehta"})),
        json!({"country": "Japan"}),
        "Tokyo",
    ));

    // Example 4: Pinocchio lies about Australia's capital
    examples.push(create_json_example(
        Some(json!({"assistant_name": "Pinocchio"})),
        json!({"country": "Australia"}),
        "Sydney (nose grows 4 inches)",
    ));

    examples
}

/// Create a single JSON example for the json_success function
#[allow(clippy::allow_attributes, dead_code)] // False positive
fn create_json_example(
    system: Option<serde_json::Value>,
    user_input: serde_json::Value,
    answer: &str,
) -> RenderedSample {
    let json_output = JsonInferenceOutput {
        parsed: Some(json!({"answer": answer})),
        raw: Some(format!(r#"{{"answer":"{answer}"}}"#)),
    };

    RenderedSample {
        function_name: "json_success".to_string(),
        input: ModelInput {
            system: system.as_ref().map(std::string::ToString::to_string),
            messages: vec![ResolvedRequestMessage {
                role: Role::User,
                content: vec![ResolvedContentBlock::Text(Text {
                    text: user_input.to_string(),
                })],
            }],
        },
        stored_input: StoredInput {
            system: system
                .as_ref()
                .map(|s| System::Template(Arguments(s.as_object().unwrap().to_owned()))),
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputMessageContent::Template(Template {
                    name: "user".to_string(),
                    arguments: Arguments(user_input.as_object().unwrap().clone()),
                })],
            }],
        },
        output: Some(vec![ContentBlockChatOutput::Text(Text {
            text: json_output.raw.clone().unwrap(),
        })]),
        stored_output: Some(StoredOutput::Json(json_output)),
        episode_id: Some(Uuid::now_v7()),
        inference_id: Some(Uuid::now_v7()),
        tool_params: tensorzero::DynamicToolParams::default(),
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
        dispreferred_outputs: vec![],
        tags: HashMap::new(),
    }
}
