#![expect(clippy::expect_used)]

use serde_json::json;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tensorzero::DynamicToolParams;
use tensorzero_core::client::{Client, ClientBuilder, ClientBuilderMode};
use tensorzero_core::config::{Config, ConfigFileGlob};
use tensorzero_core::db::clickhouse::ClickHouseConnectionInfo;
use tensorzero_core::db::postgres::PostgresConnectionInfo;
use tensorzero_core::db::valkey::ValkeyConnectionInfo;
use tensorzero_core::endpoints::datasets::v1::delete_dataset;
use tensorzero_core::http::TensorzeroHttpClient;
use tensorzero_core::inference::types::{
    Arguments, ContentBlockChatOutput, JsonInferenceOutput, ModelInput, ResolvedContentBlock,
    ResolvedRequestMessage, Role, StoredInput, StoredInputMessage, StoredInputMessageContent,
    System, Template, Text,
};
use tensorzero_core::optimization::gepa::GEPAConfig;
use tensorzero_core::stored_inference::{RenderedSample, StoredOutput};
use tensorzero_core::utils::retries::RetryConfig;
use tensorzero_optimizers::gepa::{
    evaluate::EvaluationResults,
    validate::{FunctionContext, validate_gepa_config},
};
use tokio::time::sleep;

pub mod analyze;
pub mod dataset;
pub mod evaluate;
pub mod mutate;
pub mod pareto;

/// Wait time for ClickHouse operations to complete (in milliseconds)
pub const TEST_CLICKHOUSE_WAIT_MS: u64 = 500;

/// Helper function to load the e2e test config
async fn get_e2e_config() -> Arc<Config> {
    let mut config_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    config_path.push("../tensorzero-core/tests/e2e/config/tensorzero.*.toml");
    Arc::new(
        Config::load_from_path_optional_verify_credentials(
            &ConfigFileGlob::new_from_path(&config_path)
                .expect("Failed to create config glob from path"),
            false,
        )
        .await
        .expect("Failed to load e2e config")
        .into_config_without_writing_for_tests(),
    )
}

/// Helper function to create a test RenderedSample for Chat functions
#[expect(clippy::missing_panics_doc)]
pub fn create_test_chat_rendered_sample(input: &str, output: &str) -> RenderedSample {
    let output_vec = vec![ContentBlockChatOutput::Text(Text {
        text: output.to_string(),
    })];
    RenderedSample {
        function_name: "basic_test".to_string(),
        input: ModelInput {
            system: None,
            messages: vec![ResolvedRequestMessage {
                role: Role::User,
                content: vec![ResolvedContentBlock::Text(Text {
                    text: input.to_string(),
                })],
            }],
        },
        stored_input: StoredInput {
            system: Some(System::Template(Arguments(
                json!({"assistant_name": "TestAssistant"})
                    .as_object()
                    .expect("Failed to convert JSON to object")
                    .clone(),
            ))),
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputMessageContent::Text(Text {
                    text: input.to_string(),
                })],
            }],
        },
        output: Some(output_vec.clone()),
        stored_output: Some(StoredOutput::Chat(output_vec)),
        episode_id: None,
        inference_id: None,
        tool_params: DynamicToolParams::default(),
        output_schema: None,
        dispreferred_outputs: vec![],
        tags: {
            let mut tags = HashMap::new();
            tags.insert("test_key".to_string(), "test_value".to_string());
            tags
        },
    }
}

/// Helper function to create a test RenderedSample for JSON functions
#[expect(clippy::missing_panics_doc)]
pub fn create_test_json_rendered_sample(input: &str, output: &str) -> RenderedSample {
    let json_output = JsonInferenceOutput {
        raw: Some(output.to_string()),
        parsed: Some(serde_json::json!({"answer": output})),
    };

    RenderedSample {
        function_name: "json_success".to_string(),
        input: ModelInput {
            system: Some("JSON system prompt".to_string()),
            messages: vec![ResolvedRequestMessage {
                role: Role::User,
                content: vec![ResolvedContentBlock::Text(Text {
                    text: input.to_string(),
                })],
            }],
        },
        stored_input: StoredInput {
            system: Some(System::Template(Arguments(
                json!({"assistant_name": "TestAssistant"})
                    .as_object()
                    .expect("Failed to convert JSON to object")
                    .clone(),
            ))),
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputMessageContent::Template(Template {
                    name: "user".to_string(),
                    arguments: Arguments(serde_json::Map::from_iter(vec![(
                        "country".to_string(),
                        json!(input),
                    )])),
                })],
            }],
        },
        output: None, // JSON functions don't have chat output
        stored_output: Some(StoredOutput::Json(json_output)),
        episode_id: None,
        inference_id: None,
        tool_params: DynamicToolParams::default(),
        output_schema: Some(serde_json::json!({
            "type": "object",
            "properties": {
                "answer": {"type": "string"}
            },
            "required": ["answer"],
            "additionalProperties": false
        })),
        dispreferred_outputs: vec![],
        tags: {
            let mut tags = HashMap::new();
            tags.insert("json_key".to_string(), "json_value".to_string());
            tags
        },
    }
}

/// Check if analysis contains one of the expected XML tags
pub fn contains_expected_xml_tag(analysis: &str) -> bool {
    analysis.contains("<report_error>")
        || analysis.contains("<report_improvement>")
        || analysis.contains("<report_optimal>")
}

/// Helper function to build a gateway client with standard test configuration
#[expect(clippy::missing_panics_doc)]
pub async fn build_gateway_client(
    config: Arc<Config>,
    clickhouse: ClickHouseConnectionInfo,
    http_client: TensorzeroHttpClient,
    timeout_secs: u64,
) -> Client {
    ClientBuilder::new(ClientBuilderMode::FromComponents {
        config,
        clickhouse_connection_info: clickhouse,
        postgres_connection_info: PostgresConnectionInfo::Disabled,
        valkey_connection_info: ValkeyConnectionInfo::Disabled,
        http_client,
        timeout: Some(Duration::from_secs(timeout_secs)),
    })
    .build()
    .await
    .expect("Failed to build gateway client")
}

/// Helper function to create a GEPA config for Chat function tests
pub fn create_gepa_config_chat() -> GEPAConfig {
    GEPAConfig {
        function_name: "basic_test".to_string(),
        evaluation_name: "test_evaluation".to_string(),
        initial_variants: Some(vec!["openai".to_string(), "anthropic".to_string()]),
        variant_prefix: Some("gepa_test_chat".to_string()),
        batch_size: 5,
        max_iterations: 1,
        max_concurrency: 5,
        analysis_model: "openai::gpt-4.1-nano".to_string(),
        mutation_model: "openai::gpt-4.1-nano".to_string(),
        seed: None,
        timeout: 300,
        include_inference_for_mutation: true,
        retries: RetryConfig::default(),
        max_tokens: Some(16_384),
    }
}

/// Helper function to create a GEPA config for JSON function tests
pub fn create_gepa_config_json() -> GEPAConfig {
    GEPAConfig {
        function_name: "json_success".to_string(),
        evaluation_name: "json_evaluation".to_string(),
        initial_variants: Some(vec!["openai".to_string(), "anthropic".to_string()]),
        variant_prefix: Some("gepa_test_json".to_string()),
        batch_size: 5,
        max_iterations: 1,
        max_concurrency: 5,
        analysis_model: "openai::gpt-4.1-nano".to_string(),
        mutation_model: "openai::gpt-4.1-nano".to_string(),
        seed: None,
        timeout: 300,
        include_inference_for_mutation: true,
        retries: RetryConfig::default(),
        max_tokens: Some(16_384),
    }
}

/// Helper function to validate GEPA config and get FunctionContext
#[expect(clippy::missing_panics_doc)]
pub fn get_function_context(gepa_config: &GEPAConfig, config: &Config) -> FunctionContext {
    validate_gepa_config(gepa_config, config).expect("validate_gepa_config should succeed in test")
}

/// Helper function to delete a dataset and wait for ClickHouse to process
pub async fn cleanup_dataset(clickhouse: &ClickHouseConnectionInfo, dataset_name: &str) {
    let _ = delete_dataset(clickhouse, dataset_name).await;
    sleep(Duration::from_millis(TEST_CLICKHOUSE_WAIT_MS)).await;
}

/// Helper function to verify evaluation results have expected structure and valid scores
#[expect(clippy::missing_panics_doc)]
pub fn assert_evaluation_results_valid(
    evaluation_results: &EvaluationResults,
    expected_count: usize,
) {
    // These evaluators should be present in all test evaluations
    let base_evaluators = ["happy_bool", "sad_bool", "zero", "one"];
    // The "error" evaluator intentionally always fails, so we skip count validation for it
    let always_failing_evaluators = ["error"];

    // Verify we have stats for all base evaluators
    for evaluator_name in &base_evaluators {
        assert!(
            evaluation_results
                .evaluation_stats
                .contains_key(*evaluator_name),
            "Expected {evaluator_name} evaluator stats"
        );
    }

    // Verify each evaluator has valid stats
    for (evaluator_name, stats) in &evaluation_results.evaluation_stats {
        // Skip count validation for evaluators that intentionally always fail
        if always_failing_evaluators.contains(&evaluator_name.as_str()) {
            continue;
        }
        assert_eq!(
            stats.count, expected_count,
            "Expected count of {} for {}, got {}",
            expected_count, evaluator_name, stats.count
        );
        assert!(
            stats.mean.is_finite(),
            "Expected mean to be finite for {}, got {}",
            evaluator_name,
            stats.mean
        );
    }

    // Test per_datapoint_scores extraction
    let per_datapoint_scores = evaluation_results.per_datapoint_scores();
    assert_eq!(
        per_datapoint_scores.len(),
        expected_count,
        "Expected {} datapoints in per_datapoint_scores, got {}",
        expected_count,
        per_datapoint_scores.len()
    );

    // Verify each datapoint has scores for all base evaluators
    for (datapoint_id, scores) in &per_datapoint_scores {
        for evaluator_name in &base_evaluators {
            assert!(
                scores.contains_key(*evaluator_name),
                "Datapoint {datapoint_id} missing {evaluator_name} score"
            );
        }

        // Verify scores are valid (either None or finite)
        for (evaluator_name, score_opt) in scores {
            if let Some(score) = score_opt {
                assert!(
                    score.is_finite(),
                    "Score for evaluator {evaluator_name} on datapoint {datapoint_id} is not finite: {score}"
                );
            }
        }
    }
}
