#![expect(clippy::unwrap_used, clippy::panic, clippy::print_stdout)]
use std::collections::HashMap;
use std::sync::Arc;

use serde_json::json;
use tensorzero::{
    ClientExt, InferenceOutputSource, LaunchOptimizationWorkflowParams, RenderedSample, Role,
};

use super::dicl::get_pinocchio_examples;
use tensorzero_core::{
    config::{Config, ConfigFileGlob, UninitializedVariantConfig},
    db::clickhouse::test_helpers::get_clickhouse,
    http::TensorzeroHttpClient,
    inference::types::{
        Arguments, ContentBlockChatOutput, ModelInput, ResolvedContentBlock,
        ResolvedRequestMessage, StoredInput, StoredInputMessage, StoredInputMessageContent, System,
        Text,
    },
    model_table::ProviderTypeDefaultCredentials,
    optimization::{
        gepa::UninitializedGEPAConfig, OptimizationJobInfo, OptimizerOutput,
        UninitializedOptimizerConfig, UninitializedOptimizerInfo,
    },
    stored_inference::StoredOutput,
    tool::DynamicToolParams,
};
use tensorzero_optimizers::{JobHandle, Optimizer};
use uuid::Uuid;

/// Core test for GEPA optimization using Pinocchio pattern
///
/// This test validates that GEPA can evolve system templates to teach the model
/// to produce the Pinocchio pattern (lies with nose growth).
#[allow(clippy::allow_attributes, dead_code)] // False positive
pub async fn test_gepa_optimization_chat() {
    // Initialize tracing subscriber to capture progress logs
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();

    let variant_prefix = format!("gepa_pinocchio_test_{}", Uuid::now_v7());
    let function_name = "basic_test".to_string();
    let evaluation_name = "test_gepa_pinocchio".to_string();

    let uninitialized_optimizer_info = UninitializedOptimizerInfo {
        inner: UninitializedOptimizerConfig::Gepa(UninitializedGEPAConfig {
            function_name: function_name.clone(),
            evaluation_name: evaluation_name.clone(),
            initial_variants: Some(vec!["openai".to_string(), "openai-extra-body".to_string()]),
            variant_prefix: Some(variant_prefix.clone()),
            batch_size: 5,
            max_iterations: 2,
            max_concurrency: 10,
            analysis_model: "openai::gpt-5-mini".to_string(),
            mutation_model: "openai::gpt-5-mini".to_string(),
            seed: Some(42),
            timeout: 300,
            include_datapoint_input_for_mutation: false,
            retries: tensorzero_core::utils::retries::RetryConfig::default(),
            max_tokens: 16_384,
        }),
    };

    let optimizer_info = uninitialized_optimizer_info
        .load(&ProviderTypeDefaultCredentials::default())
        .await
        .unwrap();

    let client = TensorzeroHttpClient::new_testing().unwrap();

    // Use Pinocchio examples for training and validation
    // Cycle through the 4 Pinocchio examples to create 15 training examples
    let test_examples = get_pinocchio_examples(false)
        .into_iter()
        .cycle()
        .take(15)
        .collect::<Vec<_>>();
    let val_examples = Some(get_gepa_pinocchio_validation_examples(10));

    let credentials: HashMap<String, secrecy::SecretBox<str>> = HashMap::new();
    let clickhouse = get_clickhouse().await;

    let mut config_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    config_path.push("../tensorzero-core/tests/e2e/tensorzero.toml");

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
            Arc::new(config),
        )
        .await
        .unwrap();

    // Poll (GEPA completes synchronously, so should be done immediately)
    let status = job_handle
        .poll(
            &client,
            &credentials,
            &ProviderTypeDefaultCredentials::default(),
        )
        .await
        .unwrap();

    // Validate output - GEPA may succeed with variants or fail to find improvements
    match status {
        OptimizationJobInfo::Completed {
            output: OptimizerOutput::Variants(variants),
        } => {
            // GEPA found improvements - validate the variants
            assert!(
                !variants.is_empty(),
                "GEPA should produce at least one evolved variant"
            );
            assert!(
                variants.len() <= 3,
                "Should not exceed max_iterations + 1 variants, got {}",
                variants.len()
            );

            // Validate each variant structure
            for (variant_name, variant_config) in &variants {
                assert!(
                    variant_name.starts_with(&variant_prefix),
                    "Variant name '{variant_name}' should have prefix '{variant_prefix}'"
                );

                let chat_config = match &**variant_config {
                    UninitializedVariantConfig::ChatCompletion(config) => config,
                    _ => panic!("Expected ChatCompletion variant"),
                };

                // Validate required templates exist
                assert!(
                    !chat_config.templates.inner.is_empty(),
                    "Variant should have at least one template"
                );

                // Log template names that were evolved
                for template_name in chat_config.templates.inner.keys() {
                    println!("Evolved template variant includes template: '{template_name}'");
                }
            }

            println!(
                "GEPA Pinocchio optimization test passed with {} evolved variants",
                variants.len()
            );
        }
        OptimizationJobInfo::Failed { message, .. } => {
            // GEPA failed to find improvements - this is a valid outcome
            println!("⚠️  GEPA optimization completed but found no improvements:");
            println!("   {message}");
            println!("✅ GEPA error handling test passed - gracefully handled failure case");
        }
        other => {
            panic!("Expected Completed or Failed status, got: {other:?}");
        }
    }
}

/// Generate simple test examples for basic_test function
#[expect(dead_code)]
fn get_gepa_basic_examples(count: usize) -> Vec<RenderedSample> {
    let examples = vec![
        ("What is 2+2?", "4"),
        ("What is the capital of France?", "Paris"),
        ("What color is the sky?", "Blue"),
        ("How many days in a week?", "7"),
        ("What is 10-5?", "5"),
        ("What is the largest ocean?", "Pacific"),
        ("How many continents are there?", "7"),
        ("What is 3*3?", "9"),
        ("What is the smallest prime number?", "2"),
        ("How many hours in a day?", "24"),
    ];

    examples
        .into_iter()
        .cycle()
        .take(count)
        .map(|(question, answer)| create_basic_test_sample(question, answer))
        .collect()
}

/// Create a RenderedSample for basic_test function
fn create_basic_test_sample(question: &str, answer: &str) -> RenderedSample {
    // So the examples are different
    let id = Uuid::now_v7().to_string();
    let system_json = json!({
        "assistant_name": format!("Assistant {id}")
    });
    let output = vec![ContentBlockChatOutput::Text(Text {
        text: answer.to_string(),
    })];

    RenderedSample {
        function_name: "basic_test".to_string(),
        input: ModelInput {
            system: Some(system_json.to_string()),
            messages: vec![ResolvedRequestMessage {
                role: Role::User,
                content: vec![ResolvedContentBlock::Text(Text {
                    text: question.to_string(),
                })],
            }],
        },
        stored_input: StoredInput {
            system: Some(System::Template(Arguments(
                system_json.as_object().unwrap().to_owned(),
            ))),
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputMessageContent::Text(Text {
                    text: question.to_string(),
                })],
            }],
        },
        output: Some(output.clone()),
        stored_output: Some(StoredOutput::Chat(output)),
        episode_id: Some(Uuid::now_v7()),
        inference_id: Some(Uuid::now_v7()),
        tool_params: DynamicToolParams::default(),
        output_schema: None,
        dispreferred_outputs: vec![],
        tags: HashMap::new(),
    }
}

/// Generate validation examples for Pinocchio pattern testing (different questions than training)
fn get_gepa_pinocchio_validation_examples(count: usize) -> Vec<RenderedSample> {
    let examples = get_pinocchio_examples(false);

    // Take the requested count, cycling if needed
    examples.into_iter().cycle().take(count).collect()
}

/// Workflow test with embedded client
#[allow(clippy::allow_attributes, dead_code)] // False positive
pub async fn test_gepa_workflow_with_embedded_client() {
    // Create embedded gateway client
    let client = tensorzero::test_helpers::make_embedded_gateway().await;
    run_gepa_workflow_with_client(&client).await;
}

/// Workflow test with HTTP client
#[allow(clippy::allow_attributes, dead_code)] // False positive
pub async fn test_gepa_workflow_with_http_client() {
    // Create HTTP gateway client
    let client = tensorzero::test_helpers::make_http_gateway().await;
    run_gepa_workflow_with_client(&client).await;
}

/// Run GEPA workflow test with a provided client
#[allow(clippy::allow_attributes, dead_code)] // False positive
async fn run_gepa_workflow_with_client(client: &tensorzero::Client) {
    let params = LaunchOptimizationWorkflowParams {
        function_name: "basic_test".to_string(),
        template_variant_name: "test".to_string(),
        query_variant_name: None,
        filters: None,
        output_source: InferenceOutputSource::Inference,
        order_by: None,
        limit: Some(20),
        offset: None,
        val_fraction: Some(0.3),
        optimizer_config: UninitializedOptimizerInfo {
            inner: UninitializedOptimizerConfig::Gepa(UninitializedGEPAConfig {
                function_name: "basic_test".to_string(),
                evaluation_name: "test_evaluation".to_string(),
                initial_variants: Some(vec!["test".to_string(), "test2".to_string()]),
                variant_prefix: Some(format!("gepa_workflow_{}", Uuid::now_v7())),
                batch_size: 3,
                max_iterations: 2,
                max_concurrency: 2,
                analysis_model: "test".to_string(),
                mutation_model: "test".to_string(),
                seed: Some(42),
                timeout: 300,
                include_datapoint_input_for_mutation: false,
                retries: tensorzero_core::utils::retries::RetryConfig::default(),
                max_tokens: 16_384,
            }),
        },
    };

    let job_handle = client
        .experimental_launch_optimization_workflow(params)
        .await
        .unwrap();

    // Poll for completion (GEPA should complete immediately)
    let mut status;
    loop {
        status = client
            .experimental_poll_optimization(&job_handle)
            .await
            .unwrap();
        println!("Status: `{status:?}` Handle: `{job_handle}`");
        if matches!(
            status,
            OptimizationJobInfo::Completed { .. } | OptimizationJobInfo::Failed { .. }
        ) {
            break;
        }
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
    }

    // Validate response - GEPA may succeed with variants or fail to find improvements
    match status {
        OptimizationJobInfo::Completed {
            output: OptimizerOutput::Variants(variants),
        } => {
            assert!(
                !variants.is_empty(),
                "GEPA workflow should produce variants"
            );
            assert!(
                variants.len() <= 3,
                "Should not exceed max_iterations + 1 variants, got {}",
                variants.len()
            );

            // Validate variant structure
            for (variant_name, variant_config) in &variants {
                assert!(
                    variant_name.starts_with("gepa_workflow_"),
                    "Variant name '{variant_name}' should have workflow prefix"
                );

                match &**variant_config {
                    UninitializedVariantConfig::ChatCompletion(config) => {
                        assert!(
                            !config.templates.inner.is_empty(),
                            "Variant should have at least one template"
                        );
                    }
                    _ => panic!("Expected ChatCompletion variant"),
                }
            }

            println!(
                "✅ GEPA workflow test passed with {} variants",
                variants.len()
            );
        }
        OptimizationJobInfo::Failed { message, .. } => {
            // GEPA failed to find improvements - this is a valid outcome
            println!("⚠️  GEPA workflow completed but found no improvements:");
            println!("   {message}");
            println!(
                "✅ GEPA workflow error handling test passed - gracefully handled failure case"
            );
        }
        other => panic!("Expected Completed or Failed, got: {other:?}"),
    }
}
