#![cfg_attr(
    test,
    allow(clippy::expect_used, clippy::unwrap_used, clippy::print_stdout)
)]

mod common;
mod test_top_level_evaluator;

use clap::Parser;
use evaluations::evaluators::llm_judge::{RunLLMJudgeEvaluatorParams, run_llm_judge_evaluator};
use evaluations::stopping::MIN_DATAPOINTS;
use evaluations::{ClientInferenceExecutor, Clients};
use serde_json::json;
use tensorzero_core::cache::CacheEnabledMode;
use tensorzero_core::client::{Input, InputMessage, InputMessageContent};
use tensorzero_core::db::datasets::DatasetQueries;
use tensorzero_core::db::delegating_connection::DelegatingDatabaseConnection;
use tensorzero_core::db::evaluation_queries::EvaluationQueries;
use tensorzero_core::db::feedback::{
    BooleanMetricFeedbackRow, FeedbackQueries, FeedbackRow, FloatMetricFeedbackRow,
};
use tensorzero_core::db::inferences::{
    InferenceOutputSource, InferenceQueries, ListInferencesParams,
};
use tensorzero_core::db::model_inferences::ModelInferenceQueries;
use tensorzero_core::db::stored_datapoint::{
    StoredChatInferenceDatapoint, StoredDatapoint, StoredJsonInferenceDatapoint,
};
use tensorzero_core::db::test_helpers::TestDatabaseHelpers;
use tensorzero_core::endpoints::datasets::{
    ChatInferenceDatapoint, Datapoint, JsonInferenceDatapoint,
    v1::{list_datapoints, types::ListDatapointsRequest},
};
use tensorzero_core::evaluations::{
    EvaluationConfig, LLMJudgeConfig, LLMJudgeInputFormat, LLMJudgeOutputType,
};
use tensorzero_core::inference::types::Text;
use tensorzero_core::stored_inference::StoredInferenceDatabase;
use tokio::time::sleep;
use url::Url;

use common::{get_config, get_e2e_config_path, init_tracing_for_tests};
use evaluations::{
    Args, EvaluationCoreArgs, EvaluationFunctionConfig, EvaluationVariant, OutputFormat,
    run_evaluation, run_evaluation_core_streaming,
    stats::{EvaluationUpdate, PerEvaluatorStats},
};
use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;
use std::{path::PathBuf, sync::Arc};
use tensorzero::test_helpers::make_embedded_gateway;
use tensorzero_core::client::{
    ClientBuilder, ClientBuilderMode, FeedbackParams, InferenceResponse, Role,
};
use tensorzero_core::config::{
    Config, UninitializedVariantConfig, UninitializedVariantInfo, path::ResolvedTomlPathData,
};
use tensorzero_core::inference::types::{ContentBlockChatOutput, JsonInferenceOutput, Usage};
use tensorzero_core::variant::chat_completion::UninitializedChatCompletionConfig;
use tensorzero_core::{
    endpoints::inference::{ChatInferenceResponse, JsonInferenceResponse},
    evaluations::{LLMJudgeIncludeConfig, LLMJudgeOptimize},
};
use uuid::Uuid;

/// Takes a chat fixture as a path to a JSONL file and writes the fixture to the dataset.
/// To avoid trampling between tests, we use a mapping from the fixture dataset names to the actual dataset names
/// that are inserted. This way, we can have multiple tests reading the same fixtures, using the same database,
/// but run independently.
async fn write_chat_fixture_to_dataset(
    db: &DelegatingDatabaseConnection,
    fixture_path: &Path,
    dataset_name_mapping: &HashMap<String, String>,
) {
    let fixture = std::fs::read_to_string(fixture_path).unwrap();
    let fixture = fixture.trim();
    let mut datapoints: Vec<StoredDatapoint> = Vec::new();
    for line in fixture.lines() {
        let mut datapoint: StoredChatInferenceDatapoint = serde_json::from_str(line).unwrap();
        datapoint.id = Uuid::now_v7();
        if let Some(dataset_name) = dataset_name_mapping.get(&datapoint.dataset_name) {
            datapoint.dataset_name = dataset_name.to_string();
        }
        datapoints.push(StoredDatapoint::Chat(datapoint));
    }
    db.insert_datapoints(&datapoints).await.unwrap();
    db.flush_pending_writes().await;
}

/// Takes a JSON fixture as a path to a JSONL file and writes the fixture to the dataset.
async fn write_json_fixture_to_dataset(
    db: &DelegatingDatabaseConnection,
    fixture_path: &Path,
    dataset_name_mapping: &HashMap<String, String>,
) {
    let fixture = std::fs::read_to_string(fixture_path).unwrap();
    let fixture = fixture.trim();
    let mut datapoints: Vec<StoredDatapoint> = Vec::new();
    for line in fixture.lines() {
        let mut datapoint: StoredJsonInferenceDatapoint = serde_json::from_str(line).unwrap();
        datapoint.id = Uuid::now_v7();
        if let Some(dataset_name) = dataset_name_mapping.get(&datapoint.dataset_name) {
            datapoint.dataset_name = dataset_name.to_string();
        }
        datapoints.push(StoredDatapoint::Json(datapoint));
    }
    db.insert_datapoints(&datapoints).await.unwrap();
    db.flush_pending_writes().await;
}

/// Queries feedback for a target_id, filters by metric_name if provided,
/// and returns the first matching boolean feedback row.
async fn query_boolean_feedback(
    db: &DelegatingDatabaseConnection,
    target_id: Uuid,
    metric_name: Option<&str>,
) -> Option<BooleanMetricFeedbackRow> {
    let rows = db
        .query_feedback_by_target_id(target_id, None, None, None)
        .await
        .unwrap();
    rows.into_iter().find_map(|row| match row {
        FeedbackRow::Boolean(b)
            if metric_name.is_none() || Some(b.metric_name.as_str()) == metric_name =>
        {
            Some(b)
        }
        _ => None,
    })
}

/// Queries feedback for a target_id, filters by metric_name if provided,
/// and returns the first matching float feedback row.
async fn query_float_feedback(
    db: &DelegatingDatabaseConnection,
    target_id: Uuid,
    metric_name: Option<&str>,
) -> Option<FloatMetricFeedbackRow> {
    let rows = db
        .query_feedback_by_target_id(target_id, None, None, None)
        .await
        .unwrap();
    rows.into_iter().find_map(|row| match row {
        FeedbackRow::Float(f)
            if metric_name.is_none() || Some(f.metric_name.as_str()) == metric_name =>
        {
            Some(f)
        }
        _ => None,
    })
}

/// Queries a single inference by ID using `list_inferences`.
async fn query_inference(
    db: &DelegatingDatabaseConnection,
    config: &Config,
    inference_id: Uuid,
) -> Option<StoredInferenceDatabase> {
    let params = ListInferencesParams {
        function_name: None,
        ids: Some(&[inference_id]),
        variant_name: None,
        episode_id: None,
        filters: None,
        output_source: InferenceOutputSource::Inference,
        limit: 1,
        offset: 0,
        pagination: None,
        order_by: None,
        search_query_experimental: None,
    };
    let results = db
        .list_inferences(config, &params)
        .await
        .expect("list_inferences should succeed");
    results.into_iter().next()
}

#[tokio::test(flavor = "multi_thread")]
async fn run_evaluations_json() {
    init_tracing_for_tests();
    let db = DelegatingDatabaseConnection::new_for_e2e_test().await;
    let config = get_config().await;
    let dataset_name = format!("extract_entities_0.8-{}", Uuid::now_v7());
    let tensorzero_client = make_embedded_gateway().await;
    write_json_fixture_to_dataset(
        &db,
        &PathBuf::from(&format!(
            "{}/../tensorzero-core/fixtures/datasets/json_datapoint_fixture.jsonl",
            std::env::var("CARGO_MANIFEST_DIR").unwrap()
        )),
        &HashMap::from([("extract_entities_0.8".to_string(), dataset_name.clone())]),
    )
    .await;
    let config_path = get_e2e_config_path();
    let evaluation_run_id = Uuid::now_v7();
    let args = || Args {
        config_file: config_path.clone(),
        gateway_url: None,
        evaluation_name: Some("entity_extraction".to_string()),
        function_name: None,
        evaluator_names: None,
        dataset_name: Some(dataset_name.clone()),
        datapoint_ids: Some(vec![]),
        variant_name: "gpt_4o_mini".to_string(),
        concurrency: 10,
        format: OutputFormat::Jsonl,
        // This test relies on the cache (see below), so we need to enable it
        inference_cache: CacheEnabledMode::On,
        max_datapoints: None,
        precision_targets: vec![],
        cutoffs: vec![],
    };

    let mut output = Vec::new();
    Box::pin(run_evaluation(args(), evaluation_run_id, &mut output))
        .await
        .unwrap();
    db.flush_pending_writes().await;
    sleep(Duration::from_secs(5)).await;
    let output_str = String::from_utf8(output).unwrap();
    let output_lines: Vec<&str> = output_str.lines().skip(1).collect();
    let mut parsed_output = Vec::new();
    let mut total_sports = 0;
    let mut evaluator_inference_ids = HashMap::new();
    for line in output_lines {
        let parsed: EvaluationUpdate =
            serde_json::from_str(line).expect("Each line should be valid JSON");
        let parsed = match parsed {
            EvaluationUpdate::Success(evaluation_info) => evaluation_info,
            EvaluationUpdate::Error(evaluation_error) => {
                panic!("evaluation error: {}", evaluation_error.message);
            }
            EvaluationUpdate::RunInfo(_) => continue,
        };
        assert_eq!(parsed.evaluator_errors.len(), 1);
        let error = parsed.evaluator_errors.get("error").unwrap();
        assert!(error.contains("Dummy error in inference"));
        let inference_id = parsed.response.inference_id();
        let StoredInferenceDatabase::Json(db_inference) =
            query_inference(&db, &config, inference_id)
                .await
                .expect("Should find json inference")
        else {
            panic!("Expected json inference");
        };
        let parsed_response = match parsed.response.clone() {
            InferenceResponse::Json(json_response) => json_response,
            InferenceResponse::Chat(..) => panic!("Chat response not supported"),
        };
        let db_input: Input = serde_json::from_str(
            &serde_json::to_string(&db_inference.input.unwrap())
                .expect("StoredInput should serialize"),
        )
        .expect("StoredInput should deserialize as Input");
        // Check the input to the inference is the same as the input to the datapoint
        assert_eq!(&db_input, parsed.datapoint.input());
        let db_output = db_inference
            .output
            .expect("json inference should have output");
        // Check the output to the inference is the same as the output in the response
        assert_eq!(db_output, parsed_response.output);
        // Check the inference is properly tagged
        assert_eq!(
            db_inference.tags["tensorzero::evaluation_run_id"],
            evaluation_run_id.to_string()
        );
        assert_eq!(
            db_inference.tags["tensorzero::datapoint_id"],
            parsed.datapoint.id().to_string()
        );
        assert_eq!(
            db_inference.tags["tensorzero::evaluation_name"],
            "entity_extraction"
        );
        assert_eq!(db_inference.tags["tensorzero::dataset_name"], dataset_name);
        // Check boolean feedback was recorded
        let feedback = query_boolean_feedback(&db, inference_id, None)
            .await
            .unwrap();

        assert_eq!(
            feedback.metric_name,
            "tensorzero::evaluation_name::entity_extraction::evaluator_name::exact_match"
        );
        assert_eq!(
            feedback.value,
            parsed.evaluations["exact_match"]
                .as_ref()
                .unwrap()
                .as_bool()
                .unwrap()
        );
        assert_eq!(
            feedback.tags["tensorzero::evaluation_run_id"],
            evaluation_run_id.to_string()
        );
        assert_eq!(
            feedback.tags["tensorzero::datapoint_id"],
            parsed.datapoint.id().to_string()
        );

        // Check float feedback was recorded
        let feedback = query_float_feedback(&db, inference_id, None).await.unwrap();

        assert_eq!(
            feedback.metric_name,
            "tensorzero::evaluation_name::entity_extraction::evaluator_name::count_sports"
        );
        assert_eq!(
            feedback.value,
            parsed.evaluations["count_sports"]
                .as_ref()
                .unwrap()
                .as_f64()
                .unwrap()
        );
        assert_eq!(
            feedback.tags["tensorzero::evaluation_run_id"],
            evaluation_run_id.to_string()
        );
        let datapoint_id = parsed.datapoint.id();
        assert_eq!(
            feedback.tags["tensorzero::datapoint_id"],
            datapoint_id.to_string()
        );
        assert_eq!(feedback.tags["tensorzero::evaluator_name"], "count_sports");
        assert_eq!(
            feedback.tags["tensorzero::evaluation_name"],
            "entity_extraction"
        );
        assert!(
            !feedback
                .tags
                .contains_key("tensorzero::derived_from_human_feedback")
        );
        let evaluator_inference_id =
            Uuid::parse_str(&feedback.tags["tensorzero::evaluator_inference_id"]).unwrap();
        evaluator_inference_ids.insert(parsed.datapoint.id(), evaluator_inference_id);
        // The evaluator inference id should not be the same as the inference id
        assert_ne!(evaluator_inference_id, inference_id);
        // Send human feedback to the evaluator and overwrite the existing feedback
        let metric_name =
            "tensorzero::evaluation_name::entity_extraction::evaluator_name::count_sports";
        let human_feedback_payload = FeedbackParams {
            inference_id: Some(inference_id),
            metric_name: metric_name.to_string(),
            value: json!(0),
            internal: true,
            tags: HashMap::from([
                (
                    "tensorzero::datapoint_id".to_string(),
                    parsed.datapoint.id().to_string(),
                ),
                ("tensorzero::human_feedback".to_string(), "true".to_string()),
                (
                    "tensorzero::evaluator_inference_id".to_string(),
                    evaluator_inference_id.to_string(),
                ),
            ]),
            dryrun: Some(false),
            episode_id: None,
        };
        tensorzero_client
            .feedback(human_feedback_payload)
            .await
            .unwrap();
        let StoredInferenceDatabase::Json(evaluator_inference) =
            query_inference(&db, &config, evaluator_inference_id)
                .await
                .expect("Should find evaluator json inference")
        else {
            panic!("Expected json inference for evaluator");
        };
        assert_eq!(
            evaluator_inference.tags["tensorzero::evaluation_name"],
            "entity_extraction"
        );
        total_sports += feedback.value as u32;
        let serialized_output = db
            .serialize_output_for_feedback(&parsed.response)
            .expect("serialize_output_for_feedback should succeed");
        parsed_output.push(parsed);
        // Sleep for 5s to make sure the feedback is recorded
        sleep(Duration::from_secs(5)).await;
        let human_feedback = db
            .get_inference_evaluation_human_feedback(metric_name, &datapoint_id, &serialized_output)
            .await
            .expect("get_inference_evaluation_human_feedback should succeed")
            .expect("human feedback should exist");
        assert_eq!(human_feedback.value, json!(0));
        assert_eq!(
            human_feedback.evaluator_inference_id,
            evaluator_inference_id
        );
    }
    assert_eq!(parsed_output.len(), 6);
    assert_eq!(total_sports, 3);
    sleep(Duration::from_secs(5)).await;

    // Check that the human feedback affects the next eval run results
    // Run the evaluation again but now it should read the human feedback that was sent
    let mut output = Vec::new();
    Box::pin(run_evaluation(args(), evaluation_run_id, &mut output))
        .await
        .unwrap();
    db.flush_pending_writes().await;
    sleep(Duration::from_secs(5)).await;
    let output_str = String::from_utf8(output).unwrap();
    let output_lines: Vec<&str> = output_str.lines().skip(1).collect();
    let mut total_sports = 0;
    for line in output_lines {
        let parsed: EvaluationUpdate =
            serde_json::from_str(line).expect("Each line should be valid JSON");
        let parsed = match parsed {
            EvaluationUpdate::Success(evaluation_info) => evaluation_info,
            EvaluationUpdate::Error(evaluation_error) => {
                panic!("evaluation error: {}", evaluation_error.message);
            }
            EvaluationUpdate::RunInfo(_) => continue,
        };
        let inference_id = parsed.response.inference_id();
        // We only check the total_topic_fs for the second run
        total_sports += parsed.evaluations["count_sports"]
            .as_ref()
            .unwrap()
            .as_f64()
            .unwrap() as u32;
        // Grab the feedback for the second run and make sure it has the human feedback tag
        let db_feedback = query_float_feedback(&db, inference_id, None).await.unwrap();
        assert_eq!(
            db_feedback.tags["tensorzero::derived_from_human_feedback"],
            "true",
        );
        assert_eq!(
            db_feedback.tags["tensorzero::evaluator_inference_id"],
            evaluator_inference_ids[&parsed.datapoint.id()].to_string()
        );
    }
    assert_eq!(total_sports, 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_dataset_name_and_datapoint_ids_mutually_exclusive() {
    init_tracing_for_tests();
    let dataset_name = format!("test-dataset-{}", Uuid::now_v7());

    let config_path = get_e2e_config_path();
    let evaluation_run_id = Uuid::now_v7();

    // Test 1: Both dataset_name and datapoint_ids provided should fail
    let args_both = Args {
        config_file: config_path.clone(),
        gateway_url: None,
        evaluation_name: Some("entity_extraction".to_string()),
        function_name: None,
        evaluator_names: None,
        dataset_name: Some(dataset_name.clone()),
        datapoint_ids: Some(vec![Uuid::now_v7()]),
        variant_name: "gpt_4o_mini".to_string(),
        concurrency: 10,
        format: OutputFormat::Jsonl,
        inference_cache: CacheEnabledMode::On,
        max_datapoints: None,
        precision_targets: vec![],
        cutoffs: vec![],
    };

    let mut output = Vec::new();
    let result = Box::pin(run_evaluation(args_both, evaluation_run_id, &mut output)).await;
    assert!(
        result.is_err(),
        "Should fail when both dataset_name and datapoint_ids are provided"
    );
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("Cannot provide both")
    );

    // Test 2: Neither dataset_name nor datapoint_ids provided should fail
    let args_neither = Args {
        config_file: config_path.clone(),
        gateway_url: None,
        evaluation_name: Some("entity_extraction".to_string()),
        function_name: None,
        evaluator_names: None,
        dataset_name: None,
        datapoint_ids: Some(vec![]),
        variant_name: "gpt_4o_mini".to_string(),
        concurrency: 10,
        format: OutputFormat::Jsonl,
        inference_cache: CacheEnabledMode::On,
        max_datapoints: None,
        precision_targets: vec![],
        cutoffs: vec![],
    };

    let mut output = Vec::new();
    let result = Box::pin(run_evaluation(args_neither, evaluation_run_id, &mut output)).await;
    assert!(
        result.is_err(),
        "Should fail when neither dataset_name nor datapoint_ids are provided"
    );
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("Must provide either")
    );
}

/// Test mutual exclusivity of `datapoint_ids` `max_datapoints` in `run_evaluation()`
#[tokio::test(flavor = "multi_thread")]
async fn test_datapoint_ids_and_max_datapoints_mutually_exclusive() {
    init_tracing_for_tests();
    let config_path = get_e2e_config_path();
    let evaluation_run_id = Uuid::now_v7();

    // Test: Both datapoint_ids and max_datapoints provided should fail
    let args = Args {
        config_file: config_path,
        gateway_url: None,
        evaluation_name: Some("entity_extraction".to_string()),
        function_name: None,
        evaluator_names: None,
        dataset_name: None,
        datapoint_ids: Some(vec![Uuid::now_v7()]),
        variant_name: "gpt_4o_mini".to_string(),
        concurrency: 10,
        format: OutputFormat::Jsonl,
        inference_cache: CacheEnabledMode::On,
        max_datapoints: Some(10),
        precision_targets: vec![],
        cutoffs: vec![],
    };

    let mut output = Vec::new();
    let result = Box::pin(run_evaluation(args, evaluation_run_id, &mut output)).await;
    assert!(
        result.is_err(),
        "Should fail when both datapoint_ids and max_datapoints are provided"
    );
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("Cannot provide both datapoint_ids and max_datapoints")
    );
}

/// Test mutual exclusivity of `datapoint_ids` `max_datapoints` in `run_evaluation_core_streaming()`
#[tokio::test(flavor = "multi_thread")]
async fn test_datapoint_ids_and_max_datapoints_mutually_exclusive_core_streaming() {
    init_tracing_for_tests();
    let config = get_config().await;
    let db = DelegatingDatabaseConnection::new_for_e2e_test().await;
    let tensorzero_client = make_embedded_gateway().await;
    let evaluation_run_id = Uuid::now_v7();

    let evaluation_name = "entity_extraction".to_string();

    // Extract evaluation config fields
    let evaluation_config = config
        .evaluations
        .get(&evaluation_name)
        .expect("evaluation 'entity_extraction' not found");
    let EvaluationConfig::Inference(inference_config) = &**evaluation_config;
    let function_config = config
        .functions
        .get(&inference_config.function_name)
        .map(|f| EvaluationFunctionConfig::from(f.as_ref()))
        .expect("function should exist");

    // Wrap the client in ClientInferenceExecutor for use with evaluations
    let inference_executor = Arc::new(ClientInferenceExecutor::new(tensorzero_client));

    // Test: Both datapoint_ids and max_datapoints provided should fail
    let core_args = EvaluationCoreArgs {
        inference_executor,
        db: Arc::new(db.clone()),
        function_name: inference_config.function_name.clone(),
        function_config,
        evaluators: inference_config.evaluators.clone(),
        evaluation_name: Some(evaluation_name),
        evaluation_run_id,
        dataset_name: None,
        datapoint_ids: Some(vec![Uuid::now_v7()]),
        variant: EvaluationVariant::Name("gpt_4o_mini".to_string()),
        concurrency: 10,
        inference_cache: CacheEnabledMode::On,
        tags: HashMap::new(),
    };

    let result = run_evaluation_core_streaming(core_args, Some(10), HashMap::new()).await;
    assert!(
        result.is_err(),
        "Should fail when both datapoint_ids and max_datapoints are provided"
    );
    let error = result.err().unwrap();
    assert!(
        error
            .to_string()
            .contains("Cannot provide both datapoint_ids and max_datapoints")
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn run_evaluation_with_specific_datapoint_ids() {
    init_tracing_for_tests();
    let dataset_name = format!("haiku-data-subset-{}", Uuid::now_v7());
    let db = DelegatingDatabaseConnection::new_for_e2e_test().await;

    // Create a dataset with multiple datapoints
    write_chat_fixture_to_dataset(
        &db,
        &PathBuf::from(&format!(
            "{}/../tensorzero-core/fixtures/datasets/chat_datapoint_fixture.jsonl",
            std::env::var("CARGO_MANIFEST_DIR").unwrap()
        )),
        &HashMap::from([("good-haiku-data".to_string(), dataset_name.clone())]),
    )
    .await;

    // Query the dataset to get all datapoint IDs using v1 API
    let request = ListDatapointsRequest {
        function_name: Some("write_haiku".to_string()),
        limit: Some(u32::MAX),
        offset: Some(0),
        ..Default::default()
    };
    let dataset = list_datapoints(&db, dataset_name.clone(), request)
        .await
        .unwrap()
        .datapoints;

    // Select only the first 5 datapoint IDs
    let selected_ids: Vec<Uuid> = dataset.iter().take(5).map(|dp| dp.id()).collect();
    assert_eq!(
        selected_ids.len(),
        5,
        "Should have selected exactly 5 datapoint IDs"
    );

    let config_path = get_e2e_config_path();
    let evaluation_run_id = Uuid::now_v7();
    let args = Args {
        config_file: config_path,
        gateway_url: None,
        evaluation_name: Some("haiku_with_outputs".to_string()),
        function_name: None,
        evaluator_names: None,
        dataset_name: None,
        datapoint_ids: Some(selected_ids.clone()),
        variant_name: "gpt_4o_mini".to_string(),
        concurrency: 10,
        format: OutputFormat::Jsonl,
        inference_cache: CacheEnabledMode::Off,
        max_datapoints: None,
        precision_targets: vec![],
        cutoffs: vec![],
    };

    let mut output = Vec::new();
    Box::pin(run_evaluation(args, evaluation_run_id, &mut output))
        .await
        .unwrap();
    db.flush_pending_writes().await;
    sleep(Duration::from_secs(5)).await;

    let output_str = String::from_utf8(output).unwrap();
    let output_lines: Vec<&str> = output_str.lines().skip(1).collect();
    let mut evaluated_datapoint_ids = Vec::new();

    for line in output_lines {
        let parsed: EvaluationUpdate =
            serde_json::from_str(line).expect("Each line should be valid JSON");
        let parsed = match parsed {
            EvaluationUpdate::Success(evaluation_info) => evaluation_info,
            EvaluationUpdate::Error(evaluation_error) => {
                panic!("evaluation error: {}", evaluation_error.message);
            }
            EvaluationUpdate::RunInfo(_) => continue,
        };
        evaluated_datapoint_ids.push(parsed.datapoint.id());
    }

    // Verify exactly 5 datapoints were evaluated
    assert_eq!(
        evaluated_datapoint_ids.len(),
        5,
        "Should have evaluated exactly 5 datapoints"
    );

    // Verify all evaluated datapoints are in the selected set
    for evaluated_id in &evaluated_datapoint_ids {
        assert!(
            selected_ids.contains(evaluated_id),
            "Evaluated datapoint {evaluated_id} was not in the selected set"
        );
    }

    // Verify all selected datapoints were evaluated
    for selected_id in &selected_ids {
        assert!(
            evaluated_datapoint_ids.contains(selected_id),
            "Selected datapoint {selected_id} was not evaluated"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn run_exact_match_evaluation_chat() {
    init_tracing_for_tests();
    let dataset_name = format!("good-haiku-data-{}", Uuid::now_v7());
    let db = DelegatingDatabaseConnection::new_for_e2e_test().await;
    let config = get_config().await;
    write_chat_fixture_to_dataset(
        &db,
        &PathBuf::from(&format!(
            "{}/../tensorzero-core/fixtures/datasets/chat_datapoint_fixture.jsonl",
            std::env::var("CARGO_MANIFEST_DIR").unwrap()
        )),
        &HashMap::from([("good-haiku-data".to_string(), dataset_name.clone())]),
    )
    .await;

    // Query the dataset to get datapoint IDs; use these instead of dataset_name in the eval run
    let request = ListDatapointsRequest {
        function_name: Some("write_haiku".to_string()),
        limit: Some(u32::MAX),
        offset: Some(0),
        ..Default::default()
    };
    let dataset = list_datapoints(&db, dataset_name.clone(), request)
        .await
        .unwrap()
        .datapoints;
    let datapoint_ids: Vec<Uuid> = dataset.iter().map(|dp| dp.id()).collect();

    let config_path = get_e2e_config_path();
    let evaluation_run_id = Uuid::now_v7();
    let args = Args {
        config_file: config_path,
        gateway_url: None,
        evaluation_name: Some("haiku_with_outputs".to_string()),
        function_name: None,
        evaluator_names: None,
        dataset_name: None,
        datapoint_ids: Some(datapoint_ids.clone()),
        variant_name: "gpt_4o_mini".to_string(),
        concurrency: 10,
        format: OutputFormat::Jsonl,
        inference_cache: CacheEnabledMode::Off,
        max_datapoints: None,
        precision_targets: vec![],
        cutoffs: vec![],
    };

    let mut output = Vec::new();
    Box::pin(run_evaluation(args, evaluation_run_id, &mut output))
        .await
        .unwrap();
    db.flush_pending_writes().await;
    sleep(Duration::from_secs(5)).await;
    let output_str = String::from_utf8(output).unwrap();
    let output_lines: Vec<&str> = output_str.lines().skip(1).collect();
    let mut parsed_output = Vec::new();
    for line in output_lines {
        let parsed: EvaluationUpdate =
            serde_json::from_str(line).expect("Each line should be valid JSON");
        let parsed = match parsed {
            EvaluationUpdate::Success(evaluation_info) => evaluation_info,
            EvaluationUpdate::Error(evaluation_error) => {
                panic!("evaluation error: {}", evaluation_error.message);
            }
            EvaluationUpdate::RunInfo(_) => continue,
        };
        assert!(parsed.evaluator_errors.is_empty());
        let inference_id = parsed.response.inference_id();
        let StoredInferenceDatabase::Chat(db_inference) =
            query_inference(&db, &config, inference_id)
                .await
                .expect("Should find chat inference")
        else {
            panic!("Expected chat inference");
        };
        let parsed_response = match parsed.response.clone() {
            InferenceResponse::Chat(chat_response) => chat_response,
            InferenceResponse::Json(..) => panic!("Json response not supported"),
        };
        let db_input: Input = serde_json::from_str(
            &serde_json::to_string(&db_inference.input.unwrap())
                .expect("StoredInput should serialize"),
        )
        .expect("StoredInput should deserialize as Input");
        // Check the input to the inference is the same as the input to the datapoint
        assert_eq!(&db_input, parsed.datapoint.input());
        let db_output = db_inference
            .output
            .expect("chat inference should have output");
        // Check the output to the inference is the same as the output in the response
        assert_eq!(db_output, parsed_response.content);
        // Check the inference is properly tagged
        assert_eq!(
            db_inference.tags["tensorzero::evaluation_run_id"],
            evaluation_run_id.to_string()
        );
        assert_eq!(
            db_inference.tags["tensorzero::datapoint_id"],
            parsed.datapoint.id().to_string()
        );
        assert_eq!(
            db_inference.tags["tensorzero::evaluation_name"],
            "haiku_with_outputs"
        );
        assert_eq!(
            db_inference.tags["tensorzero::dataset_name"], dataset_name,
            "dataset_name tag should be derived from the datapoints"
        );
        let db_feedback = query_boolean_feedback(&db, inference_id, None)
            .await
            .unwrap();
        assert_eq!(
            db_feedback.metric_name,
            "tensorzero::evaluation_name::haiku_with_outputs::evaluator_name::exact_match"
        );
        assert_eq!(
            db_feedback.value,
            parsed.evaluations["exact_match"]
                .as_ref()
                .unwrap()
                .as_bool()
                .unwrap()
        );
        assert_eq!(
            db_feedback.tags["tensorzero::evaluation_run_id"],
            evaluation_run_id.to_string()
        );
        assert_eq!(
            db_feedback.tags["tensorzero::datapoint_id"],
            parsed.datapoint.id().to_string()
        );
        assert_eq!(
            db_feedback.tags["tensorzero::evaluation_name"],
            "haiku_with_outputs"
        );
        assert_eq!(
            db_feedback.tags["tensorzero::evaluator_name"],
            "exact_match"
        );
        parsed_output.push(parsed);
    }
    assert_eq!(parsed_output.len(), 29);
}

#[tokio::test(flavor = "multi_thread")]
async fn run_llm_judge_evaluation_chat() {
    init_tracing_for_tests();
    let dataset_name = format!("good-haikus-no-output-{}", Uuid::now_v7());
    let db = DelegatingDatabaseConnection::new_for_e2e_test().await;
    let config = get_config().await;
    write_chat_fixture_to_dataset(
        &db,
        &PathBuf::from(&format!(
            "{}/../tensorzero-core/fixtures/datasets/chat_datapoint_fixture.jsonl",
            std::env::var("CARGO_MANIFEST_DIR").unwrap()
        )),
        &HashMap::from([("good-haikus-no-output".to_string(), dataset_name.clone())]),
    )
    .await;

    // Query the dataset to get datapoint IDs; use these instead of dataset_name in the eval run
    let request = ListDatapointsRequest {
        function_name: Some("write_haiku".to_string()),
        limit: Some(u32::MAX),
        offset: Some(0),
        ..Default::default()
    };
    let dataset = list_datapoints(&db, dataset_name.clone(), request)
        .await
        .unwrap()
        .datapoints;
    let datapoint_ids: Vec<Uuid> = dataset.iter().map(|dp| dp.id()).collect();

    let config_path = get_e2e_config_path();
    let tensorzero_client = make_embedded_gateway().await;
    let evaluation_run_id = Uuid::now_v7();
    let args = || Args {
        config_file: config_path.clone(),
        gateway_url: None,
        dataset_name: None,
        datapoint_ids: Some(datapoint_ids.clone()),
        evaluation_name: Some("haiku_without_outputs".to_string()),
        function_name: None,
        evaluator_names: None,
        variant_name: "gpt_4o_mini".to_string(),
        concurrency: 10,
        format: OutputFormat::Jsonl,
        inference_cache: CacheEnabledMode::On,
        max_datapoints: None,
        precision_targets: vec![],
        cutoffs: vec![],
    };

    let mut output = Vec::new();
    Box::pin(run_evaluation(args(), evaluation_run_id, &mut output))
        .await
        .unwrap();
    db.flush_pending_writes().await;
    sleep(Duration::from_secs(5)).await;
    let output_str = String::from_utf8(output).unwrap();
    let output_lines: Vec<&str> = output_str.lines().skip(1).collect();
    let mut parsed_output = Vec::new();
    let mut total_topic_fs = 0;
    for line in output_lines {
        let parsed: EvaluationUpdate =
            serde_json::from_str(line).expect("Each line should be valid JSON");
        let parsed = match parsed {
            EvaluationUpdate::Success(evaluation_info) => evaluation_info,
            EvaluationUpdate::Error(evaluation_error) => {
                panic!("evaluation error: {}", evaluation_error.message);
            }
            EvaluationUpdate::RunInfo(_) => continue,
        };
        assert!(parsed.evaluator_errors.is_empty());
        let inference_id = parsed.response.inference_id();
        let StoredInferenceDatabase::Chat(db_inference) =
            query_inference(&db, &config, inference_id)
                .await
                .expect("Should find chat inference")
        else {
            panic!("Expected chat inference");
        };
        let parsed_response = match parsed.response.clone() {
            InferenceResponse::Chat(chat_response) => chat_response,
            InferenceResponse::Json(..) => panic!("Json response not supported"),
        };
        let db_input: Input = serde_json::from_str(
            &serde_json::to_string(&db_inference.input.unwrap())
                .expect("StoredInput should serialize"),
        )
        .expect("StoredInput should deserialize as Input");
        // Check the input to the inference is the same as the input to the datapoint
        assert_eq!(&db_input, parsed.datapoint.input());
        let db_output = db_inference
            .output
            .expect("chat inference should have output");
        // Check the output to the inference is the same as the output in the response
        assert_eq!(db_output, parsed_response.content);
        // Check the inference is properly tagged
        assert_eq!(
            db_inference.tags["tensorzero::evaluation_run_id"],
            evaluation_run_id.to_string()
        );
        assert_eq!(
            db_inference.tags["tensorzero::datapoint_id"],
            parsed.datapoint.id().to_string()
        );
        assert_eq!(
            db_inference.tags["tensorzero::evaluation_name"],
            "haiku_without_outputs"
        );

        // There should be no Float feedback for this evaluation
        assert!(
            query_float_feedback(&db, inference_id, None)
                .await
                .is_none()
        );
        // The exact match evaluation should have value None since there is no output in any of these
        assert!(parsed.evaluations["exact_match"].is_none());

        // There should be Boolean feedback for this evaluation
        let db_feedback = query_boolean_feedback(&db, inference_id, None)
            .await
            .unwrap();
        assert_eq!(
            db_feedback.metric_name,
            "tensorzero::evaluation_name::haiku_without_outputs::evaluator_name::topic_starts_with_f"
        );
        assert_eq!(
            db_feedback.value,
            parsed.evaluations["topic_starts_with_f"]
                .as_ref()
                .unwrap()
                .as_bool()
                .unwrap()
        );
        total_topic_fs += db_feedback.value as u32;
        assert_eq!(
            db_feedback.tags["tensorzero::evaluation_run_id"],
            evaluation_run_id.to_string()
        );
        assert_eq!(
            db_feedback.tags["tensorzero::datapoint_id"],
            parsed.datapoint.id().to_string()
        );
        assert_eq!(
            db_feedback.tags["tensorzero::evaluation_name"],
            "haiku_without_outputs"
        );
        assert_eq!(
            db_feedback.tags["tensorzero::evaluator_name"],
            "topic_starts_with_f"
        );
        assert!(
            !db_feedback
                .tags
                .contains_key("tensorzero::derived_from_human_feedback")
        );
        let evaluator_inference_id =
            Uuid::parse_str(&db_feedback.tags["tensorzero::evaluator_inference_id"]).unwrap();
        // Send human feedback to the evaluator and overwrite the existing feedback
        let human_feedback_payload = FeedbackParams {
            inference_id: Some(inference_id),
            metric_name: "tensorzero::evaluation_name::haiku_without_outputs::evaluator_name::topic_starts_with_f".to_string(),
            value: json!(false),
            internal: true,
            tags: HashMap::from([
                ("tensorzero::datapoint_id".to_string(), parsed.datapoint.id().to_string()),
                ("tensorzero::human_feedback".to_string(), "true".to_string()),
                (
                    "tensorzero::evaluator_inference_id".to_string(),
                    evaluator_inference_id.to_string(),
                ),
            ]),
            dryrun: Some(false),
            episode_id: None,
        };
        tensorzero_client
            .feedback(human_feedback_payload)
            .await
            .unwrap();

        let StoredInferenceDatabase::Json(evaluator_inference) =
            query_inference(&db, &config, evaluator_inference_id)
                .await
                .expect("Should find evaluator json inference")
        else {
            panic!("Expected json inference for evaluator");
        };
        assert_eq!(
            evaluator_inference.tags["tensorzero::evaluation_name"],
            "haiku_without_outputs"
        );
        parsed_output.push(parsed);
    }
    assert_eq!(parsed_output.len(), 10);
    assert_eq!(total_topic_fs, 3);
    sleep(Duration::from_secs(5)).await;
    // Run the evaluation again but now it should read the human feedback that was sent
    let mut output = Vec::new();
    Box::pin(run_evaluation(args(), evaluation_run_id, &mut output))
        .await
        .unwrap();
    db.flush_pending_writes().await;
    sleep(Duration::from_secs(5)).await;
    let output_str = String::from_utf8(output).unwrap();
    let output_lines: Vec<&str> = output_str.lines().skip(1).collect();
    let mut total_topic_fs = 0;
    for line in output_lines {
        let parsed: EvaluationUpdate =
            serde_json::from_str(line).expect("Each line should be valid JSON");
        let parsed = match parsed {
            EvaluationUpdate::Success(evaluation_info) => evaluation_info,
            EvaluationUpdate::Error(evaluation_error) => {
                panic!("evaluation error: {}", evaluation_error.message);
            }
            EvaluationUpdate::RunInfo(_) => continue,
        };
        let inference_id = parsed.response.inference_id();
        // We only check the total_topic_fs for the second run
        total_topic_fs += parsed.evaluations["topic_starts_with_f"]
            .as_ref()
            .unwrap()
            .as_bool()
            .unwrap() as u32;
        // Grab the feedback for the second run and make sure it has the human feedback tag
        let db_feedback = query_boolean_feedback(&db, inference_id, None)
            .await
            .unwrap();
        assert_eq!(
            db_feedback.tags["tensorzero::derived_from_human_feedback"],
            "true"
        );
    }
    assert_eq!(total_topic_fs, 0);
}

/// High level with this test is that the judge is actually just checking if the reference output matches the
/// generated output.
/// However, it takes an image and we verify that the image is actually used in the inference.
#[tokio::test(flavor = "multi_thread")]
async fn run_image_evaluation() {
    init_tracing_for_tests();
    let dataset_name = format!("baz-{}", Uuid::now_v7());
    let db = DelegatingDatabaseConnection::new_for_e2e_test().await;
    let config = get_config().await;
    write_chat_fixture_to_dataset(
        &db,
        &PathBuf::from(&format!(
            "{}/../tensorzero-core/fixtures/datasets/chat_datapoint_fixture.jsonl",
            std::env::var("CARGO_MANIFEST_DIR").unwrap()
        )),
        &HashMap::from([("baz".to_string(), dataset_name.clone())]),
    )
    .await;
    let config_path = get_e2e_config_path();
    let evaluation_run_id = Uuid::now_v7();
    let args = Args {
        config_file: config_path,
        gateway_url: None,
        dataset_name: Some(dataset_name.clone()),
        datapoint_ids: Some(vec![]),
        evaluation_name: Some("images".to_string()),
        function_name: None,
        evaluator_names: None,
        variant_name: "honest_answer".to_string(),
        concurrency: 10,
        format: OutputFormat::Jsonl,
        inference_cache: CacheEnabledMode::WriteOnly,
        max_datapoints: None,
        precision_targets: vec![],
        cutoffs: vec![],
    };

    let mut output = Vec::new();
    Box::pin(run_evaluation(args, evaluation_run_id, &mut output))
        .await
        .unwrap();
    db.flush_pending_writes().await;
    sleep(Duration::from_secs(5)).await;
    let output_str = String::from_utf8(output).unwrap();
    let output_lines: Vec<&str> = output_str.lines().skip(1).collect();
    let mut parsed_output = Vec::new();
    let mut total_honest_answers = 0;
    let mut total_matches_reference = 0;
    for line in output_lines {
        let parsed: EvaluationUpdate =
            serde_json::from_str(line).expect("Each line should be valid JSON");
        let parsed = match parsed {
            EvaluationUpdate::Success(evaluation_info) => evaluation_info,
            EvaluationUpdate::Error(evaluation_error) => {
                panic!("evaluation error: {}", evaluation_error.message);
            }
            EvaluationUpdate::RunInfo(_) => continue,
        };
        assert!(parsed.evaluator_errors.is_empty());
        let inference_id = parsed.response.inference_id();
        let StoredInferenceDatabase::Chat(db_inference) =
            query_inference(&db, &config, inference_id)
                .await
                .expect("Should find chat inference")
        else {
            panic!("Expected chat inference");
        };
        let parsed_response = match parsed.response.clone() {
            InferenceResponse::Chat(chat_response) => chat_response,
            InferenceResponse::Json(..) => panic!("Json response not supported"),
        };
        // Check the input to the inference parses as Input
        let _db_input: Input = serde_json::from_str(
            &serde_json::to_string(&db_inference.input.unwrap())
                .expect("StoredInput should serialize"),
        )
        .expect("StoredInput should deserialize as Input");
        // assert_eq!(&db_input, parsed.datapoint.input());
        let db_output = db_inference
            .output
            .expect("chat inference should have output");
        // Check the output to the inference is the same as the output in the response
        assert_eq!(db_output, parsed_response.content);
        // Check the inference is properly tagged
        assert_eq!(
            db_inference.tags["tensorzero::evaluation_run_id"],
            evaluation_run_id.to_string()
        );
        assert_eq!(
            db_inference.tags["tensorzero::datapoint_id"],
            parsed.datapoint.id().to_string()
        );
        assert_eq!(db_inference.tags["tensorzero::evaluation_name"], "images");

        // There should be no Float feedback for this evaluation
        assert!(
            query_float_feedback(&db, inference_id, None)
                .await
                .is_none()
        );
        // The exact match evaluation should fail
        assert!(
            !parsed.evaluations["exact_match"]
                .as_ref()
                .unwrap()
                .as_bool()
                .unwrap()
        );

        // There should be Boolean feedback for honest answer
        let db_feedback = query_boolean_feedback(
            &db,
            inference_id,
            Some("tensorzero::evaluation_name::images::evaluator_name::honest_answer"),
        )
        .await
        .unwrap();
        assert_eq!(
            db_feedback.metric_name,
            "tensorzero::evaluation_name::images::evaluator_name::honest_answer"
        );
        assert_eq!(
            db_feedback.value,
            parsed.evaluations["honest_answer"]
                .as_ref()
                .unwrap()
                .as_bool()
                .unwrap()
        );
        total_honest_answers += db_feedback.value as u32;
        assert_eq!(
            db_feedback.tags["tensorzero::evaluation_run_id"],
            evaluation_run_id.to_string()
        );
        assert_eq!(
            db_feedback.tags["tensorzero::datapoint_id"],
            parsed.datapoint.id().to_string()
        );
        assert_eq!(db_feedback.tags["tensorzero::evaluation_name"], "images");
        assert_eq!(
            db_feedback.tags["tensorzero::evaluator_name"],
            "honest_answer"
        );
        let evaluator_inference_id =
            Uuid::parse_str(&db_feedback.tags["tensorzero::evaluator_inference_id"]).unwrap();
        let StoredInferenceDatabase::Json(evaluator_inference) =
            query_inference(&db, &config, evaluator_inference_id)
                .await
                .expect("Should find evaluator json inference")
        else {
            panic!("Expected json inference for evaluator");
        };
        assert_eq!(
            evaluator_inference.tags["tensorzero::evaluation_name"],
            "images"
        );

        // There should be Boolean feedback for matches reference
        let db_feedback = query_boolean_feedback(
            &db,
            inference_id,
            Some("tensorzero::evaluation_name::images::evaluator_name::matches_reference"),
        )
        .await
        .unwrap();
        assert_eq!(
            db_feedback.metric_name,
            "tensorzero::evaluation_name::images::evaluator_name::matches_reference"
        );
        assert_eq!(
            db_feedback.value,
            parsed.evaluations["matches_reference"]
                .as_ref()
                .unwrap()
                .as_bool()
                .unwrap()
        );
        total_matches_reference += db_feedback.value as u32;
        assert_eq!(
            db_feedback.tags["tensorzero::evaluation_run_id"],
            evaluation_run_id.to_string()
        );
        assert_eq!(
            db_feedback.tags["tensorzero::datapoint_id"],
            parsed.datapoint.id().to_string()
        );
        assert_eq!(db_feedback.tags["tensorzero::evaluation_name"], "images");
        assert_eq!(
            db_feedback.tags["tensorzero::evaluator_name"],
            "matches_reference"
        );
        let evaluator_inference_id =
            Uuid::parse_str(&db_feedback.tags["tensorzero::evaluator_inference_id"]).unwrap();
        let StoredInferenceDatabase::Json(evaluator_inference) =
            query_inference(&db, &config, evaluator_inference_id)
                .await
                .expect("Should find evaluator json inference")
        else {
            panic!("Expected json inference for evaluator");
        };
        assert_eq!(
            evaluator_inference.tags["tensorzero::evaluation_name"],
            "images"
        );
        let input = serde_json::to_string(&evaluator_inference.input.unwrap())
            .expect("evaluator input should serialize");
        // Check that the input contains the image
        // Since we test image inputs other places, we can assume this worked as intended.
        assert!(input.contains("image/png"));

        parsed_output.push(parsed);
    }
    assert_eq!(parsed_output.len(), 2);
    assert_eq!(total_honest_answers, 2);
    // One of the reference outputs is a lie but this tells the truth
    assert_eq!(total_matches_reference, 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn check_invalid_image_evaluation() {
    init_tracing_for_tests();
    let dataset_name = format!("baz-{}", Uuid::now_v7());
    let db = DelegatingDatabaseConnection::new_for_e2e_test().await;
    let config = get_config().await;
    write_chat_fixture_to_dataset(
        &db,
        &PathBuf::from(&format!(
            "{}/../tensorzero-core/fixtures/datasets/chat_datapoint_fixture.jsonl",
            std::env::var("CARGO_MANIFEST_DIR").unwrap()
        )),
        &HashMap::from([("baz".to_string(), dataset_name.clone())]),
    )
    .await;
    let config_path = get_e2e_config_path();
    let evaluation_run_id = Uuid::now_v7();
    let args = Args {
        config_file: config_path,
        gateway_url: None,
        dataset_name: Some(dataset_name.clone()),
        datapoint_ids: Some(vec![]),
        evaluation_name: Some("bad_images".to_string()),
        function_name: None,
        evaluator_names: None,
        variant_name: "honest_answer".to_string(),
        concurrency: 10,
        format: OutputFormat::Jsonl,
        inference_cache: CacheEnabledMode::Off,
        max_datapoints: None,
        precision_targets: vec![],
        cutoffs: vec![],
    };

    let mut output = Vec::new();
    Box::pin(run_evaluation(args, evaluation_run_id, &mut output))
        .await
        .unwrap();
    db.flush_pending_writes().await;
    sleep(Duration::from_secs(5)).await;
    let output_str = String::from_utf8(output).unwrap();
    let output_lines: Vec<&str> = output_str.lines().skip(1).collect();
    for line in output_lines {
        let parsed: EvaluationUpdate =
            serde_json::from_str(line).expect("Each line should be valid JSON");
        let parsed = match parsed {
            EvaluationUpdate::Success(evaluation_info) => evaluation_info,
            EvaluationUpdate::Error(evaluation_error) => {
                panic!("evaluation error: {}", evaluation_error.message);
            }
            EvaluationUpdate::RunInfo(_) => continue,
        };
        assert_eq!(parsed.evaluator_errors.len(), 1);
        let honest_answer_error = &parsed.evaluator_errors["honest_answer"];
        assert_eq!(
            honest_answer_error,
            "Image content not supported for LLM judge evaluations with `serialized` input format. If you want image evaluations, try the `messages` input format."
        );
        let inference_id = parsed.response.inference_id();
        let StoredInferenceDatabase::Chat(db_inference) =
            query_inference(&db, &config, inference_id)
                .await
                .expect("Should find chat inference")
        else {
            panic!("Expected chat inference");
        };
        let parsed_response = match parsed.response.clone() {
            InferenceResponse::Chat(chat_response) => chat_response,
            InferenceResponse::Json(..) => panic!("Json response not supported"),
        };
        // Check the input to the inference parses as StoredInput
        let _db_input: Input = serde_json::from_str(
            &serde_json::to_string(&db_inference.input.unwrap())
                .expect("StoredInput should serialize"),
        )
        .expect("StoredInput should deserialize as Input");
        // assert_eq!(&db_input, parsed.datapoint.input());
        let db_output = db_inference
            .output
            .expect("chat inference should have output");
        // Check the output to the inference is the same as the output in the response
        assert_eq!(db_output, parsed_response.content);
        // Check the inference is properly tagged
        assert_eq!(
            db_inference.tags["tensorzero::evaluation_run_id"],
            evaluation_run_id.to_string()
        );
        assert_eq!(
            db_inference.tags["tensorzero::datapoint_id"],
            parsed.datapoint.id().to_string()
        );
        assert_eq!(
            db_inference.tags["tensorzero::evaluation_name"],
            "bad_images"
        );

        // There should be no Float feedback for this evaluation
        assert!(
            query_float_feedback(&db, inference_id, None)
                .await
                .is_none()
        );

        // There should be no Boolean feedback for honest answer since it should have failed
        assert!(
            query_boolean_feedback(
                &db,
                inference_id,
                Some("tensorzero::evaluation_name::images::evaluator_name::honest_answer"),
            )
            .await
            .is_none()
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn run_llm_judge_evaluation_chat_pretty() {
    init_tracing_for_tests();
    let db = DelegatingDatabaseConnection::new_for_e2e_test().await;
    let dataset_name = format!("good-haikus-no-output-{}", Uuid::now_v7());
    write_chat_fixture_to_dataset(
        &db,
        &PathBuf::from(&format!(
            "{}/../tensorzero-core/fixtures/datasets/chat_datapoint_fixture.jsonl",
            std::env::var("CARGO_MANIFEST_DIR").unwrap()
        )),
        &HashMap::from([("good-haikus-no-output".to_string(), dataset_name.clone())]),
    )
    .await;
    let config_path = get_e2e_config_path();
    let evaluation_run_id = Uuid::now_v7();
    let args = Args {
        config_file: config_path,
        gateway_url: None,
        evaluation_name: Some("haiku_without_outputs".to_string()),
        function_name: None,
        evaluator_names: None,
        dataset_name: Some(dataset_name.clone()),
        datapoint_ids: Some(vec![]),
        variant_name: "gpt_4o_mini".to_string(),
        concurrency: 10,
        format: OutputFormat::Pretty,
        inference_cache: CacheEnabledMode::Off,
        max_datapoints: None,
        precision_targets: vec![],
        cutoffs: vec![("topic_starts_with_f".to_string(), 0.5)],
    };

    let mut output = Vec::new();
    // Let's make sure this threshold passes and the output is reasonable
    Box::pin(run_evaluation(args, evaluation_run_id, &mut output))
        .await
        .unwrap();
    sleep(Duration::from_secs(5)).await;
    let output_str = String::from_utf8(output).unwrap();
    // Check for run info at the beginning
    assert!(output_str.contains("Run ID:"));
    assert!(output_str.contains("Number of datapoints:"));
    // Check for the expected evaluation results
    assert!(output_str.contains("topic_starts_with_f: 0.30 ± 0.14 (n=10)"));
    assert!(output_str.contains("exact_match: 0.00 ± 0.00 (n=0)"));
}

#[tokio::test(flavor = "multi_thread")]
async fn run_llm_judge_evaluation_json_pretty() {
    init_tracing_for_tests();
    let db = DelegatingDatabaseConnection::new_for_e2e_test().await;
    let dataset_name = format!("extract_entities_0.8-{}", Uuid::now_v7());
    write_json_fixture_to_dataset(
        &db,
        &PathBuf::from(&format!(
            "{}/../tensorzero-core/fixtures/datasets/json_datapoint_fixture.jsonl",
            std::env::var("CARGO_MANIFEST_DIR").unwrap()
        )),
        &HashMap::from([("extract_entities_0.8".to_string(), dataset_name.clone())]),
    )
    .await;
    let config_path = get_e2e_config_path();
    let evaluation_run_id = Uuid::now_v7();
    let args = Args {
        config_file: config_path,
        gateway_url: None,
        evaluation_name: Some("entity_extraction".to_string()),
        function_name: None,
        evaluator_names: None,
        dataset_name: Some(dataset_name.clone()),
        datapoint_ids: Some(vec![]),
        variant_name: "gpt_4o_mini".to_string(),
        concurrency: 10,
        format: OutputFormat::Pretty,
        inference_cache: CacheEnabledMode::Off,
        max_datapoints: None,
        precision_targets: vec![],
        cutoffs: vec![
            ("exact_match".to_string(), 0.6),
            ("count_sports".to_string(), 0.5),
        ],
    };

    let mut output = Vec::new();
    // Let's make sure this threshold fails and the output is reasonable
    let err = Box::pin(run_evaluation(args, evaluation_run_id, &mut output))
        .await
        .unwrap_err();
    sleep(Duration::from_secs(5)).await;
    let output_str = String::from_utf8(output).unwrap();
    // Check for run info at the beginning
    assert!(output_str.contains("Run ID:"));
    assert!(output_str.contains("Number of datapoints:"));
    // Check for the expected evaluation results
    assert!(output_str.contains("count_sports: 0.50 ± 0.20 (n=6)"));
    // We don't assert the exact value here because it's not deterministic
    assert!(output_str.contains("exact_match: "));
    let err = err.to_string();
    assert!(err.contains("Failed cutoffs for evaluators:"));
    assert!(err.contains("exact_match (cutoff: 0.60, got: "));
}

#[tokio::test]
async fn test_parse_args() {
    // Test default values
    let args = Args::try_parse_from(["test"]).unwrap_err();
    assert!(
        args.to_string()
            .contains("the following required arguments were not provided:")
    );
    assert!(args.to_string().contains("--variant-name <VARIANT_NAME>"));

    // Test --evaluation-name and --function-name are mutually exclusive
    let err = Args::try_parse_from([
        "test",
        "--evaluation-name",
        "my-eval",
        "--function-name",
        "my-func",
        "--evaluator-names",
        "em",
        "--variant-name",
        "v1",
    ])
    .unwrap_err();
    assert!(
        err.to_string().contains("cannot be used with"),
        "expected conflict error, got: {err}"
    );

    // Test --function-name requires --evaluator-names
    let err = Args::try_parse_from(["test", "--function-name", "my-func", "--variant-name", "v1"])
        .unwrap_err();
    assert!(
        err.to_string().contains("--evaluator-names"),
        "expected requires error, got: {err}"
    );

    // Test --evaluator-names requires --function-name
    let err = Args::try_parse_from(["test", "--evaluator-names", "em", "--variant-name", "v1"])
        .unwrap_err();
    assert!(
        err.to_string().contains("--function-name"),
        "expected requires error, got: {err}"
    );

    // Test --function-name / --evaluator-names mode
    let args = Args::try_parse_from([
        "test",
        "--function-name",
        "my-func",
        "--evaluator-names",
        "em,judge",
        "--variant-name",
        "v1",
        "--dataset-name",
        "ds",
    ])
    .unwrap();
    assert_eq!(args.evaluation_name, None);
    assert_eq!(args.function_name, Some("my-func".to_string()));
    assert_eq!(
        args.evaluator_names,
        Some(vec!["em".to_string(), "judge".to_string()])
    );

    // Test required arguments plus dataset-name (x-or with datapoint-ids)
    let args = Args::try_parse_from([
        "test",
        "--evaluation-name",
        "my-evaluation",
        "--variant-name",
        "my-variant",
        "--dataset-name",
        "my-dataset",
    ])
    .unwrap();
    assert_eq!(args.evaluation_name, Some("my-evaluation".to_string()));
    assert_eq!(args.variant_name, "my-variant");
    assert_eq!(args.dataset_name.unwrap(), "my-dataset".to_string());
    assert!(args.datapoint_ids.unwrap_or_default().is_empty());
    assert_eq!(args.config_file, PathBuf::from("./config/tensorzero.toml"));
    assert_eq!(args.concurrency, 1);
    assert_eq!(args.gateway_url, None);
    assert_eq!(args.format, OutputFormat::Pretty);
    assert_eq!(args.inference_cache, CacheEnabledMode::On);

    // Test required arguments plus datapoint-ids (x-or with dataset-name)
    let args = Args::try_parse_from([
        "test",
        "--evaluation-name",
        "my-evaluation",
        "--variant-name",
        "my-variant",
        "--datapoint-ids",
        "018e9e9e-7c1f-7e9e-9e9e-7c1f7e9e9e9e,018e9e9e-7c1f-7e9e-9e9e-7c1f7e9e9e9f",
    ])
    .unwrap();
    let datapoint_ids: Vec<Uuid> = args.datapoint_ids.unwrap_or_default();
    assert_eq!(args.evaluation_name, Some("my-evaluation".to_string()));
    assert_eq!(args.variant_name, "my-variant");
    assert_eq!(args.dataset_name, None);
    assert_eq!(datapoint_ids.len(), 2);
    assert_eq!(
        datapoint_ids[0],
        Uuid::parse_str("018e9e9e-7c1f-7e9e-9e9e-7c1f7e9e9e9e").unwrap()
    );
    assert_eq!(
        datapoint_ids[1],
        Uuid::parse_str("018e9e9e-7c1f-7e9e-9e9e-7c1f7e9e9e9f").unwrap()
    );

    // Test all arguments
    let args = Args::try_parse_from([
        "test",
        "--evaluation-name",
        "my-evaluation",
        "--dataset-name",
        "my-dataset",
        "--variant-name",
        "my-variant",
        "--config-file",
        "/path/to/config.toml",
        "--gateway-url",
        "http://localhost:8080",
        "--concurrency",
        "10",
        "--format",
        "jsonl",
        "--inference-cache",
        "write_only",
        "--max-datapoints",
        "20",
        "--adaptive-stopping-precision",
        "exact_match=0.10,count_sports=0.15",
        "--cutoffs",
        "exact_match=0.95,count_sports=0.50",
    ])
    .unwrap();
    assert_eq!(args.evaluation_name, Some("my-evaluation".to_string()));
    assert_eq!(args.function_name, None);
    assert_eq!(args.evaluator_names, None);
    assert_eq!(args.dataset_name.unwrap(), "my-dataset".to_string());
    assert!(args.datapoint_ids.unwrap_or_default().is_empty());
    assert_eq!(args.variant_name, "my-variant");
    assert_eq!(args.config_file, PathBuf::from("/path/to/config.toml"));
    assert_eq!(
        args.gateway_url,
        Some(Url::parse("http://localhost:8080/").unwrap())
    );
    assert_eq!(args.concurrency, 10);
    assert_eq!(args.format, OutputFormat::Jsonl);
    assert_eq!(args.inference_cache, CacheEnabledMode::WriteOnly);
    assert_eq!(args.max_datapoints.unwrap(), 20);
    assert_eq!(
        args.precision_targets,
        vec![
            ("exact_match".to_string(), 0.10),
            ("count_sports".to_string(), 0.15)
        ]
    );
    assert_eq!(
        args.cutoffs,
        vec![
            ("exact_match".to_string(), 0.95),
            ("count_sports".to_string(), 0.50)
        ]
    );

    // Test invalid URL
    let args = Args::try_parse_from([
        "test",
        "--evaluation-name",
        "my-evaluation",
        "--variant-name",
        "my-variant",
        "--gateway-url",
        "not-a-url",
    ])
    .unwrap_err();
    assert!(
        args.to_string()
            .contains("invalid value 'not-a-url' for '--gateway-url <GATEWAY_URL>'")
    );

    // Test invalid format
    let args = Args::try_parse_from([
        "test",
        "--evaluation-name",
        "my-evaluation",
        "--variant-name",
        "my-variant",
        "--format",
        "invalid",
    ])
    .unwrap_err();
    assert!(
        args.to_string()
            .contains("invalid value 'invalid' for '--format <FORMAT>'")
    );

    // Test invalid cutoff value
    let args = Args::try_parse_from([
        "test",
        "--evaluation-name",
        "my-evaluation",
        "--variant-name",
        "my-variant",
        "--cutoffs",
        "exact_match=-0.1",
    ])
    .unwrap_err();
    assert!(args.to_string().contains("non-negative"));
}

#[tokio::test]
async fn test_run_evaluation_binary() {
    // Compatibility with 'cargo nextest archive': https://nexte.st/docs/ci-features/archiving/#making-tests-relocatable
    let bin_path = std::env::var("NEXTEST_BIN_EXE_evaluations")
        .unwrap_or_else(|_| env!("CARGO_BIN_EXE_evaluations").to_string());
    println!("Running evaluations binary at {bin_path}");
    let output = std::process::Command::new(bin_path)
        .output()
        .expect("Failed to execute evaluations binary");
    let output_str = String::from_utf8(output.stdout).unwrap();
    assert!(output_str.is_empty());
    let stderr_str = String::from_utf8(output.stderr).unwrap();
    assert!(stderr_str.contains("the following required arguments were not provided:"));
}

#[tokio::test(flavor = "multi_thread")]
async fn run_evaluations_errors() {
    init_tracing_for_tests();
    let dataset_name = format!("extract_entities_0.8-{}", Uuid::now_v7());
    let db = DelegatingDatabaseConnection::new_for_e2e_test().await;
    write_json_fixture_to_dataset(
        &db,
        &PathBuf::from(&format!(
            "{}/../tensorzero-core/fixtures/datasets/json_datapoint_fixture.jsonl",
            std::env::var("CARGO_MANIFEST_DIR").unwrap()
        )),
        &HashMap::from([("extract_entities_0.8".to_string(), dataset_name.clone())]),
    )
    .await;
    let config_path = get_e2e_config_path();
    let evaluation_run_id = Uuid::now_v7();
    let args = Args {
        config_file: config_path,
        gateway_url: None,
        evaluation_name: Some("entity_extraction".to_string()),
        function_name: None,
        evaluator_names: None,
        dataset_name: Some(dataset_name.clone()),
        datapoint_ids: Some(vec![]),
        variant_name: "dummy_error".to_string(),
        concurrency: 10,
        format: OutputFormat::Jsonl,
        inference_cache: CacheEnabledMode::Off,
        max_datapoints: None,
        precision_targets: vec![],
        cutoffs: vec![],
    };

    let mut output = Vec::new();
    Box::pin(run_evaluation(args, evaluation_run_id, &mut output))
        .await
        .unwrap();
    db.flush_pending_writes().await;
    sleep(Duration::from_secs(5)).await;
    let output_str = String::from_utf8(output).unwrap();
    let output_lines: Vec<&str> = output_str.lines().skip(1).collect();
    for line in output_lines {
        let parsed: EvaluationUpdate =
            serde_json::from_str(line).expect("Each line should be valid JSON");
        let error = match parsed {
            EvaluationUpdate::Success(evaluation_info) => panic!(
                "evaluation success shouldn't happen: {}",
                serde_json::to_string_pretty(&evaluation_info).unwrap()
            ),
            EvaluationUpdate::Error(evaluation_error) => evaluation_error,
            EvaluationUpdate::RunInfo(_) => continue,
        };
        assert!(
            error
                .message
                .contains("Error sending request to Dummy provider")
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
#[expect(deprecated)]
async fn test_run_llm_judge_evaluator_chat() {
    init_tracing_for_tests();
    let tensorzero_client = make_embedded_gateway().await;
    let inference_executor = Arc::new(ClientInferenceExecutor::new(tensorzero_client));
    let clients = Arc::new(Clients {
        inference_executor,
        db: Arc::new(DelegatingDatabaseConnection::new_for_e2e_test().await),
    });
    let inference_response = InferenceResponse::Chat(ChatInferenceResponse {
        content: vec![ContentBlockChatOutput::Text(Text {
            text: "Hello, world!".to_string(),
        })],
        original_response: None,
        raw_response: None,
        finish_reason: None,
        episode_id: Uuid::now_v7(),
        inference_id: Uuid::now_v7(),
        usage: Usage {
            input_tokens: Some(0),
            output_tokens: Some(0),
            cache_read_input_tokens: None,
            cache_write_input_tokens: None,
            cost: None,
        },
        raw_usage: None,
        variant_name: "test_variant".to_string(),
    });
    let datapoint = Datapoint::Chat(ChatInferenceDatapoint {
        input: Input {
            system: None,
            messages: vec![InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::Text(Text {
                    text: "Hello, world!".to_string(),
                })],
            }],
        },
        auxiliary: String::new(),
        dataset_name: "test_dataset".to_string(),
        episode_id: Some(Uuid::now_v7()),
        id: Uuid::now_v7(),
        is_deleted: false,
        function_name: "test_function".to_string(),
        output: Some(vec![ContentBlockChatOutput::Text(Text {
            text: "Hello, world!".to_string(),
        })]),
        tags: None,
        tool_params: Default::default(),
        source_inference_id: None,
        staled_at: None,
        updated_at: "2025-10-13T20:17:36Z".to_string(),
        is_custom: true,
        name: None,
    });
    let llm_judge_config = LLMJudgeConfig {
        input_format: LLMJudgeInputFormat::Serialized,
        include: LLMJudgeIncludeConfig {
            reference_output: true,
        },
        optimize: LLMJudgeOptimize::Max,
        output_type: LLMJudgeOutputType::Boolean,
        cutoff: None,
        description: None,
    };
    // Construct the equivalent Input for the datapoint
    let input = Input {
        system: None,
        messages: vec![InputMessage {
            role: Role::User,
            content: vec![InputMessageContent::Text(Text {
                text: "Hello, world!".to_string(),
            })],
        }],
    };
    let external_tags = HashMap::new();
    let result = run_llm_judge_evaluator(RunLLMJudgeEvaluatorParams {
        inference_response: &inference_response,
        datapoint: &datapoint,
        clients: &clients,
        llm_judge_config: &llm_judge_config,
        evaluation_name: Some("test_evaluation"),
        evaluator_name: "happy_bool",

        evaluation_run_id: Uuid::now_v7(),
        input: &input,
        inference_cache: CacheEnabledMode::Off,
        external_tags: &external_tags,
    })
    .await
    .unwrap()
    .unwrap();
    assert_eq!(result.value, json!(true));

    let result = run_llm_judge_evaluator(RunLLMJudgeEvaluatorParams {
        inference_response: &inference_response,
        datapoint: &datapoint,
        clients: &clients,
        llm_judge_config: &llm_judge_config,
        evaluation_name: Some("test_evaluation"),
        evaluator_name: "sad_bool",

        evaluation_run_id: Uuid::now_v7(),
        input: &input,
        inference_cache: CacheEnabledMode::Off,
        external_tags: &external_tags,
    })
    .await
    .unwrap()
    .unwrap();
    assert_eq!(result.value, json!(false));

    let result = run_llm_judge_evaluator(RunLLMJudgeEvaluatorParams {
        inference_response: &inference_response,
        datapoint: &datapoint,
        clients: &clients,
        llm_judge_config: &llm_judge_config,
        evaluation_name: Some("test_evaluation"),
        evaluator_name: "zero",

        evaluation_run_id: Uuid::now_v7(),
        input: &input,
        inference_cache: CacheEnabledMode::Off,
        external_tags: &external_tags,
    })
    .await
    .unwrap()
    .unwrap();
    assert_eq!(result.value, json!(0));

    let result = run_llm_judge_evaluator(RunLLMJudgeEvaluatorParams {
        inference_response: &inference_response,
        datapoint: &datapoint,
        clients: &clients,
        llm_judge_config: &llm_judge_config,
        evaluation_name: Some("test_evaluation"),
        evaluator_name: "one",

        evaluation_run_id: Uuid::now_v7(),
        input: &input,
        inference_cache: CacheEnabledMode::Off,
        external_tags: &external_tags,
    })
    .await
    .unwrap()
    .unwrap();
    assert_eq!(result.value, json!(1));

    // Try without output
    let datapoint = Datapoint::Chat(ChatInferenceDatapoint {
        input: Input {
            system: None,
            messages: vec![InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::Text(Text {
                    text: "Hello, world!".to_string(),
                })],
            }],
        },
        auxiliary: String::new(),
        dataset_name: "test_dataset".to_string(),
        episode_id: Some(Uuid::now_v7()),
        id: Uuid::now_v7(),
        is_deleted: false,
        function_name: "test_function".to_string(),
        output: None,
        tags: None,
        tool_params: Default::default(),
        source_inference_id: None,
        staled_at: None,
        updated_at: "2025-10-13T20:17:36Z".to_string(),
        is_custom: true,
        name: None,
    });

    let result = run_llm_judge_evaluator(RunLLMJudgeEvaluatorParams {
        inference_response: &inference_response,
        datapoint: &datapoint,
        clients: &clients,
        llm_judge_config: &llm_judge_config,
        evaluation_name: Some("test_evaluation"),
        evaluator_name: "happy_bool",

        evaluation_run_id: Uuid::now_v7(),
        input: &input,
        inference_cache: CacheEnabledMode::Off,
        external_tags: &external_tags,
    })
    .await
    .unwrap();
    assert!(result.is_none());
}

#[tokio::test(flavor = "multi_thread")]
#[expect(deprecated)]
async fn test_run_llm_judge_evaluator_json() {
    init_tracing_for_tests();
    let tensorzero_client = make_embedded_gateway().await;
    let inference_executor = Arc::new(ClientInferenceExecutor::new(tensorzero_client));
    let clients = Arc::new(Clients {
        inference_executor,
        db: Arc::new(DelegatingDatabaseConnection::new_for_e2e_test().await),
    });
    let inference_response = InferenceResponse::Json(JsonInferenceResponse {
        output: JsonInferenceOutput {
            parsed: Some(json!({"answer": "LeBron James"})),
            raw: Some("{\"answer\": \"LeBron James\"}".to_string()),
        },
        original_response: None,
        raw_response: None,
        finish_reason: None,
        episode_id: Uuid::now_v7(),
        inference_id: Uuid::now_v7(),
        usage: Usage {
            input_tokens: Some(0),
            output_tokens: Some(0),
            cache_read_input_tokens: None,
            cache_write_input_tokens: None,
            cost: None,
        },
        raw_usage: None,
        variant_name: "test_variant".to_string(),
    });
    let datapoint = Datapoint::Json(JsonInferenceDatapoint {
        input: Input {
            system: None,
            messages: vec![InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::Text(Text {
                    text: "Hello, world!".to_string(),
                })],
            }],
        },
        auxiliary: String::new(),
        dataset_name: "test_dataset".to_string(),
        episode_id: Some(Uuid::now_v7()),
        id: Uuid::now_v7(),
        is_deleted: false,
        function_name: "test_function".to_string(),
        output: Some(JsonInferenceOutput {
            parsed: Some(json!({"answer": "LeBron James"})),
            raw: Some("{\"answer\": \"LeBron James\"}".to_string()),
        }),
        output_schema: json!({"answer": "string"}),
        tags: None,
        source_inference_id: None,
        staled_at: None,
        updated_at: "2025-10-13T20:17:36Z".to_string(),
        is_custom: true,
        name: None,
    });
    let llm_judge_config = LLMJudgeConfig {
        input_format: LLMJudgeInputFormat::Serialized,
        include: LLMJudgeIncludeConfig {
            reference_output: true,
        },
        optimize: LLMJudgeOptimize::Max,
        output_type: LLMJudgeOutputType::Boolean,
        cutoff: None,
        description: None,
    };
    // Construct the equivalent Input for the datapoint
    let input = Input {
        system: None,
        messages: vec![InputMessage {
            role: Role::User,
            content: vec![InputMessageContent::Text(Text {
                text: "Hello, world!".to_string(),
            })],
        }],
    };
    let external_tags = HashMap::new();
    let result = run_llm_judge_evaluator(RunLLMJudgeEvaluatorParams {
        inference_response: &inference_response,
        datapoint: &datapoint,
        clients: &clients,
        llm_judge_config: &llm_judge_config,
        evaluation_name: Some("test_evaluation"),
        evaluator_name: "happy_bool",

        evaluation_run_id: Uuid::now_v7(),
        input: &input,
        inference_cache: CacheEnabledMode::Off,
        external_tags: &external_tags,
    })
    .await
    .unwrap()
    .unwrap();
    assert_eq!(result.value, json!(true));

    let result = run_llm_judge_evaluator(RunLLMJudgeEvaluatorParams {
        inference_response: &inference_response,
        datapoint: &datapoint,
        clients: &clients,
        llm_judge_config: &llm_judge_config,
        evaluation_name: Some("test_evaluation"),
        evaluator_name: "sad_bool",

        evaluation_run_id: Uuid::now_v7(),
        input: &input,
        inference_cache: CacheEnabledMode::Off,
        external_tags: &external_tags,
    })
    .await
    .unwrap()
    .unwrap();
    assert_eq!(result.value, json!(false));

    let result = run_llm_judge_evaluator(RunLLMJudgeEvaluatorParams {
        inference_response: &inference_response,
        datapoint: &datapoint,
        clients: &clients,
        llm_judge_config: &llm_judge_config,
        evaluation_name: Some("test_evaluation"),
        evaluator_name: "zero",

        evaluation_run_id: Uuid::now_v7(),
        input: &input,
        inference_cache: CacheEnabledMode::Off,
        external_tags: &external_tags,
    })
    .await
    .unwrap()
    .unwrap();
    assert_eq!(result.value, json!(0));

    let result = run_llm_judge_evaluator(RunLLMJudgeEvaluatorParams {
        inference_response: &inference_response,
        datapoint: &datapoint,
        clients: &clients,
        llm_judge_config: &llm_judge_config,
        evaluation_name: Some("test_evaluation"),
        evaluator_name: "one",

        evaluation_run_id: Uuid::now_v7(),
        input: &input,
        inference_cache: CacheEnabledMode::Off,
        external_tags: &external_tags,
    })
    .await
    .unwrap()
    .unwrap();
    assert_eq!(result.value, json!(1));

    // Try without output
    let datapoint = Datapoint::Chat(ChatInferenceDatapoint {
        input: Input {
            system: None,
            messages: vec![InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::Text(Text {
                    text: "Hello, world!".to_string(),
                })],
            }],
        },
        auxiliary: String::new(),
        dataset_name: "test_dataset".to_string(),
        episode_id: Some(Uuid::now_v7()),
        id: Uuid::now_v7(),
        is_deleted: false,
        function_name: "test_function".to_string(),
        output: None,
        tags: None,
        tool_params: Default::default(),
        source_inference_id: None,
        staled_at: None,
        updated_at: "2025-10-13T20:17:36Z".to_string(),
        is_custom: true,
        name: None,
    });

    let result = run_llm_judge_evaluator(RunLLMJudgeEvaluatorParams {
        inference_response: &inference_response,
        datapoint: &datapoint,
        clients: &clients,
        llm_judge_config: &llm_judge_config,
        evaluation_name: Some("test_evaluation"),
        evaluator_name: "happy_bool",

        evaluation_run_id: Uuid::now_v7(),
        input: &input,
        inference_cache: CacheEnabledMode::Off,
        external_tags: &external_tags,
    })
    .await
    .unwrap();
    assert!(result.is_none());
}

#[tokio::test(flavor = "multi_thread")]
async fn run_evaluations_best_of_3() {
    init_tracing_for_tests();
    let dataset_name = format!("extract_entities_0.8-{}", Uuid::now_v7());
    let db = DelegatingDatabaseConnection::new_for_e2e_test().await;
    let config = get_config().await;
    write_json_fixture_to_dataset(
        &db,
        &PathBuf::from(&format!(
            "{}/../tensorzero-core/fixtures/datasets/json_datapoint_fixture.jsonl",
            std::env::var("CARGO_MANIFEST_DIR").unwrap()
        )),
        &HashMap::from([("extract_entities_0.8".to_string(), dataset_name.clone())]),
    )
    .await;
    let config_path = get_e2e_config_path();
    let evaluation_run_id = Uuid::now_v7();
    let args = Args {
        config_file: config_path,
        gateway_url: None,
        evaluation_name: Some("best_of_3".to_string()),
        function_name: None,
        evaluator_names: None,
        dataset_name: Some(dataset_name.clone()),
        datapoint_ids: Some(vec![]),
        variant_name: "gpt_4o_mini".to_string(),
        concurrency: 10,
        format: OutputFormat::Jsonl,
        inference_cache: CacheEnabledMode::Off,
        max_datapoints: None,
        precision_targets: vec![],
        cutoffs: vec![],
    };

    let mut output = Vec::new();
    Box::pin(run_evaluation(args, evaluation_run_id, &mut output))
        .await
        .unwrap();
    db.flush_pending_writes().await;
    sleep(Duration::from_secs(5)).await;
    let output_str = String::from_utf8(output).unwrap();
    let output_lines: Vec<&str> = output_str.lines().skip(1).collect();
    let mut parsed_output = Vec::new();
    for line in output_lines {
        let parsed: EvaluationUpdate =
            serde_json::from_str(line).expect("Each line should be valid JSON");
        let parsed = match parsed {
            EvaluationUpdate::Success(evaluation_info) => evaluation_info,
            EvaluationUpdate::Error(evaluation_error) => {
                panic!("evaluation error: {}", evaluation_error.message);
            }
            EvaluationUpdate::RunInfo(_) => continue,
        };
        assert!(parsed.evaluator_errors.is_empty());
        let inference_id = parsed.response.inference_id();
        let StoredInferenceDatabase::Json(db_inference) =
            query_inference(&db, &config, inference_id)
                .await
                .expect("Should find json inference")
        else {
            panic!("Expected json inference");
        };
        let parsed_response = match parsed.response.clone() {
            InferenceResponse::Json(json_response) => json_response,
            InferenceResponse::Chat(..) => panic!("Chat response not supported"),
        };
        let db_input: Input = serde_json::from_str(
            &serde_json::to_string(&db_inference.input.unwrap())
                .expect("StoredInput should serialize"),
        )
        .expect("StoredInput should deserialize as Input");
        // Check the input to the inference is the same as the input to the datapoint
        assert_eq!(&db_input, parsed.datapoint.input());
        let db_output = db_inference
            .output
            .expect("json inference should have output");
        // Check the output to the inference is the same as the output in the response
        assert_eq!(db_output, parsed_response.output);
        // Check the inference is properly tagged
        assert_eq!(
            db_inference.tags["tensorzero::evaluation_run_id"],
            evaluation_run_id.to_string()
        );
        assert_eq!(
            db_inference.tags["tensorzero::datapoint_id"],
            parsed.datapoint.id().to_string()
        );
        assert_eq!(
            db_inference.tags["tensorzero::evaluation_name"],
            "best_of_3"
        );
        assert_eq!(db_inference.tags["tensorzero::dataset_name"], dataset_name);
        // Check boolean feedback was recorded
        let feedback = query_boolean_feedback(&db, inference_id, None)
            .await
            .unwrap();

        assert_eq!(
            feedback.metric_name,
            "tensorzero::evaluation_name::best_of_3::evaluator_name::llm_judge_bool"
        );
        assert_eq!(
            feedback.value,
            parsed.evaluations["llm_judge_bool"]
                .as_ref()
                .unwrap()
                .as_bool()
                .unwrap()
        );
        assert_eq!(
            feedback.tags["tensorzero::evaluation_run_id"],
            evaluation_run_id.to_string()
        );
        assert_eq!(
            feedback.tags["tensorzero::datapoint_id"],
            parsed.datapoint.id().to_string()
        );

        // Check bool feedback was recorded
        let feedback = query_boolean_feedback(&db, inference_id, None)
            .await
            .unwrap();

        assert_eq!(
            feedback.metric_name,
            "tensorzero::evaluation_name::best_of_3::evaluator_name::llm_judge_bool"
        );
        assert_eq!(
            feedback.value,
            parsed.evaluations["llm_judge_bool"]
                .as_ref()
                .unwrap()
                .as_bool()
                .unwrap()
        );
        assert_eq!(
            feedback.tags["tensorzero::evaluation_run_id"],
            evaluation_run_id.to_string()
        );
        assert_eq!(
            feedback.tags["tensorzero::datapoint_id"],
            parsed.datapoint.id().to_string()
        );
        assert_eq!(
            feedback.tags["tensorzero::evaluator_name"],
            "llm_judge_bool"
        );
        assert_eq!(feedback.tags["tensorzero::evaluation_name"], "best_of_3");
        let evaluator_inference_id =
            Uuid::parse_str(&feedback.tags["tensorzero::evaluator_inference_id"]).unwrap();
        let StoredInferenceDatabase::Json(evaluator_inference) =
            query_inference(&db, &config, evaluator_inference_id)
                .await
                .expect("Should find evaluator json inference")
        else {
            panic!("Expected json inference for evaluator");
        };
        assert_eq!(
            evaluator_inference.tags["tensorzero::evaluation_name"],
            "best_of_3"
        );
        parsed_output.push(parsed);

        // Select Model inferences for the evaluator inference id
        let model_inferences = db
            .get_model_inferences_by_inference_id(evaluator_inference_id)
            .await
            .unwrap();
        let mut happy_count = 0;
        for model_inference in &model_inferences {
            if model_inference.system.as_deref().unwrap().trim()
                == "Return true if you are happy today!"
            {
                happy_count += 1;
            } else {
                assert!(
                    model_inference
                        .system
                        .as_deref()
                        .unwrap()
                        .starts_with("You are an assistant tasked with re-ranking")
                );
            }
        }
        assert_eq!(happy_count, 3);
        assert_eq!(model_inferences.len(), 4);
    }
    assert_eq!(parsed_output.len(), 6);
}

#[tokio::test(flavor = "multi_thread")]
async fn run_evaluations_mixture_of_3() {
    init_tracing_for_tests();
    let db = DelegatingDatabaseConnection::new_for_e2e_test().await;
    let config = get_config().await;
    let dataset_name = format!("extract_entities_0.8-{}", Uuid::now_v7());
    write_json_fixture_to_dataset(
        &db,
        &PathBuf::from(&format!(
            "{}/../tensorzero-core/fixtures/datasets/json_datapoint_fixture.jsonl",
            std::env::var("CARGO_MANIFEST_DIR").unwrap()
        )),
        &HashMap::from([("extract_entities_0.8".to_string(), dataset_name.clone())]),
    )
    .await;
    let config_path = get_e2e_config_path();
    let evaluation_run_id = Uuid::now_v7();
    let args = Args {
        config_file: config_path,
        gateway_url: None,
        evaluation_name: Some("mixture_of_3".to_string()),
        function_name: None,
        evaluator_names: None,
        dataset_name: Some(dataset_name.clone()),
        datapoint_ids: Some(vec![]),
        variant_name: "gpt_4o_mini".to_string(),
        concurrency: 10,
        format: OutputFormat::Jsonl,
        inference_cache: CacheEnabledMode::Off,
        max_datapoints: None,
        precision_targets: vec![],
        cutoffs: vec![],
    };

    let mut output = Vec::new();
    Box::pin(run_evaluation(args, evaluation_run_id, &mut output))
        .await
        .unwrap();
    db.flush_pending_writes().await;
    sleep(Duration::from_secs(5)).await;
    let output_str = String::from_utf8(output).unwrap();
    let output_lines: Vec<&str> = output_str.lines().skip(1).collect();
    let mut parsed_output = Vec::new();
    for line in output_lines {
        let parsed: EvaluationUpdate =
            serde_json::from_str(line).expect("Each line should be valid JSON");
        let parsed = match parsed {
            EvaluationUpdate::Success(evaluation_info) => evaluation_info,
            EvaluationUpdate::Error(evaluation_error) => {
                panic!("evaluation error: {}", evaluation_error.message);
            }
            EvaluationUpdate::RunInfo(_) => continue,
        };
        assert!(parsed.evaluator_errors.is_empty());
        let inference_id = parsed.response.inference_id();
        let StoredInferenceDatabase::Json(db_inference) =
            query_inference(&db, &config, inference_id)
                .await
                .expect("Should find json inference")
        else {
            panic!("Expected json inference");
        };
        let parsed_response = match parsed.response.clone() {
            InferenceResponse::Json(json_response) => json_response,
            InferenceResponse::Chat(..) => panic!("Chat response not supported"),
        };
        let db_input: Input = serde_json::from_str(
            &serde_json::to_string(&db_inference.input.unwrap())
                .expect("StoredInput should serialize"),
        )
        .expect("StoredInput should deserialize as Input");
        // Check the input to the inference is the same as the input to the datapoint
        assert_eq!(&db_input, parsed.datapoint.input());
        let db_output = db_inference
            .output
            .expect("json inference should have output");
        // Check the output to the inference is the same as the output in the response
        assert_eq!(db_output, parsed_response.output);
        // Check the inference is properly tagged
        assert_eq!(
            db_inference.tags["tensorzero::evaluation_run_id"],
            evaluation_run_id.to_string()
        );
        assert_eq!(
            db_inference.tags["tensorzero::datapoint_id"],
            parsed.datapoint.id().to_string()
        );
        assert_eq!(
            db_inference.tags["tensorzero::evaluation_name"],
            "mixture_of_3"
        );
        assert_eq!(db_inference.tags["tensorzero::dataset_name"], dataset_name);
        // Check boolean feedback was recorded
        let feedback = query_boolean_feedback(&db, inference_id, None)
            .await
            .unwrap();

        assert_eq!(
            feedback.metric_name,
            "tensorzero::evaluation_name::mixture_of_3::evaluator_name::llm_judge_bool"
        );
        assert_eq!(
            feedback.value,
            parsed.evaluations["llm_judge_bool"]
                .as_ref()
                .unwrap()
                .as_bool()
                .unwrap()
        );
        assert_eq!(
            feedback.tags["tensorzero::evaluation_run_id"],
            evaluation_run_id.to_string()
        );
        assert_eq!(
            feedback.tags["tensorzero::datapoint_id"],
            parsed.datapoint.id().to_string()
        );

        // Check bool feedback was recorded
        let feedback = query_boolean_feedback(&db, inference_id, None)
            .await
            .unwrap();

        assert_eq!(
            feedback.metric_name,
            "tensorzero::evaluation_name::mixture_of_3::evaluator_name::llm_judge_bool"
        );
        assert_eq!(
            feedback.value,
            parsed.evaluations["llm_judge_bool"]
                .as_ref()
                .unwrap()
                .as_bool()
                .unwrap()
        );
        assert_eq!(
            feedback.tags["tensorzero::evaluation_run_id"],
            evaluation_run_id.to_string()
        );
        assert_eq!(
            feedback.tags["tensorzero::datapoint_id"],
            parsed.datapoint.id().to_string()
        );
        assert_eq!(
            feedback.tags["tensorzero::evaluator_name"],
            "llm_judge_bool"
        );
        assert_eq!(feedback.tags["tensorzero::evaluation_name"], "mixture_of_3");
        let evaluator_inference_id =
            Uuid::parse_str(&feedback.tags["tensorzero::evaluator_inference_id"]).unwrap();
        let StoredInferenceDatabase::Json(evaluator_inference) =
            query_inference(&db, &config, evaluator_inference_id)
                .await
                .expect("Should find evaluator json inference")
        else {
            panic!("Expected json inference for evaluator");
        };
        assert_eq!(
            evaluator_inference.tags["tensorzero::evaluation_name"],
            "mixture_of_3"
        );
        parsed_output.push(parsed);

        // Select Model inferences for the evaluator inference id
        let model_inferences = db
            .get_model_inferences_by_inference_id(evaluator_inference_id)
            .await
            .unwrap();
        let mut happy_count = 0;
        for model_inference in &model_inferences {
            if model_inference.system.as_deref().unwrap().trim()
                == "Return true if you are happy today!"
            {
                happy_count += 1;
            } else {
                assert!(
                    model_inference
                        .system
                        .as_deref()
                        .unwrap()
                        .starts_with("You have been provided with a set of responses from various")
                );
            }
        }
        assert_eq!(happy_count, 3);
        assert_eq!(model_inferences.len(), 4);
    }
    assert_eq!(parsed_output.len(), 6);
}

#[tokio::test(flavor = "multi_thread")]
async fn run_evaluations_dicl() {
    init_tracing_for_tests();
    let db = DelegatingDatabaseConnection::new_for_e2e_test().await;
    let config = get_config().await;
    let dataset_name = format!("extract_entities_0.8-{}", Uuid::now_v7());
    write_json_fixture_to_dataset(
        &db,
        &PathBuf::from(&format!(
            "{}/../tensorzero-core/fixtures/datasets/json_datapoint_fixture.jsonl",
            std::env::var("CARGO_MANIFEST_DIR").unwrap()
        )),
        &HashMap::from([("extract_entities_0.8".to_string(), dataset_name.clone())]),
    )
    .await;
    let config_path = get_e2e_config_path();
    let evaluation_run_id = Uuid::now_v7();
    let args = Args {
        config_file: config_path,
        gateway_url: None,
        evaluation_name: Some("dicl".to_string()),
        function_name: None,
        evaluator_names: None,
        dataset_name: Some(dataset_name.clone()),
        datapoint_ids: Some(vec![]),
        variant_name: "gpt_4o_mini".to_string(),
        concurrency: 10,
        format: OutputFormat::Jsonl,
        inference_cache: CacheEnabledMode::Off,
        max_datapoints: None,
        precision_targets: vec![],
        cutoffs: vec![],
    };

    let mut output = Vec::new();
    Box::pin(run_evaluation(args, evaluation_run_id, &mut output))
        .await
        .unwrap();
    db.flush_pending_writes().await;
    sleep(Duration::from_secs(5)).await;
    let output_str = String::from_utf8(output).unwrap();
    let output_lines: Vec<&str> = output_str.lines().skip(1).collect();
    let mut parsed_output = Vec::new();
    for line in output_lines {
        let parsed: EvaluationUpdate =
            serde_json::from_str(line).expect("Each line should be valid JSON");
        let parsed = match parsed {
            EvaluationUpdate::Success(evaluation_info) => evaluation_info,
            EvaluationUpdate::Error(evaluation_error) => {
                panic!("evaluation error: {}", evaluation_error.message);
            }
            EvaluationUpdate::RunInfo(_) => continue,
        };
        assert!(parsed.evaluator_errors.is_empty());
        let inference_id = parsed.response.inference_id();
        let StoredInferenceDatabase::Json(db_inference) =
            query_inference(&db, &config, inference_id)
                .await
                .expect("Should find json inference")
        else {
            panic!("Expected json inference");
        };
        let parsed_response = match parsed.response.clone() {
            InferenceResponse::Json(json_response) => json_response,
            InferenceResponse::Chat(..) => panic!("Chat response not supported"),
        };
        let db_input: Input = serde_json::from_str(
            &serde_json::to_string(&db_inference.input.unwrap())
                .expect("StoredInput should serialize"),
        )
        .expect("StoredInput should deserialize as Input");
        // Check the input to the inference is the same as the input to the datapoint
        assert_eq!(&db_input, parsed.datapoint.input());
        let db_output = db_inference
            .output
            .expect("json inference should have output");
        // Check the output to the inference is the same as the output in the response
        assert_eq!(db_output, parsed_response.output);
        // Check the inference is properly tagged
        assert_eq!(
            db_inference.tags["tensorzero::evaluation_run_id"],
            evaluation_run_id.to_string()
        );
        assert_eq!(
            db_inference.tags["tensorzero::datapoint_id"],
            parsed.datapoint.id().to_string()
        );
        assert_eq!(db_inference.tags["tensorzero::evaluation_name"], "dicl");
        assert_eq!(db_inference.tags["tensorzero::dataset_name"], dataset_name);
        // Check boolean feedback was recorded
        let feedback = query_boolean_feedback(&db, inference_id, None)
            .await
            .unwrap();

        assert_eq!(
            feedback.metric_name,
            "tensorzero::evaluation_name::dicl::evaluator_name::llm_judge_bool"
        );
        assert_eq!(
            feedback.value,
            parsed.evaluations["llm_judge_bool"]
                .as_ref()
                .unwrap()
                .as_bool()
                .unwrap()
        );
        assert_eq!(
            feedback.tags["tensorzero::evaluation_run_id"],
            evaluation_run_id.to_string()
        );
        assert_eq!(
            feedback.tags["tensorzero::datapoint_id"],
            parsed.datapoint.id().to_string()
        );

        // Check bool feedback was recorded
        let feedback = query_boolean_feedback(&db, inference_id, None)
            .await
            .unwrap();

        assert_eq!(
            feedback.metric_name,
            "tensorzero::evaluation_name::dicl::evaluator_name::llm_judge_bool"
        );
        assert_eq!(
            feedback.value,
            parsed.evaluations["llm_judge_bool"]
                .as_ref()
                .unwrap()
                .as_bool()
                .unwrap()
        );
        assert_eq!(
            feedback.tags["tensorzero::evaluation_run_id"],
            evaluation_run_id.to_string()
        );
        assert_eq!(
            feedback.tags["tensorzero::datapoint_id"],
            parsed.datapoint.id().to_string()
        );
        assert_eq!(
            feedback.tags["tensorzero::evaluator_name"],
            "llm_judge_bool"
        );
        assert_eq!(feedback.tags["tensorzero::evaluation_name"], "dicl");
        let evaluator_inference_id =
            Uuid::parse_str(&feedback.tags["tensorzero::evaluator_inference_id"]).unwrap();
        let StoredInferenceDatabase::Json(evaluator_inference) =
            query_inference(&db, &config, evaluator_inference_id)
                .await
                .expect("Should find evaluator json inference")
        else {
            panic!("Expected json inference for evaluator");
        };
        assert_eq!(
            evaluator_inference.tags["tensorzero::evaluation_name"],
            "dicl"
        );
        parsed_output.push(parsed);

        // Select Model inferences for the evaluator inference id
        let model_inferences = db
            .get_model_inferences_by_inference_id(evaluator_inference_id)
            .await
            .unwrap();
        for model_inference in &model_inferences {
            match model_inference.model_name.as_str() {
                "openai::gpt-4o-mini" => {
                    let system = model_inference.system.as_deref().unwrap();
                    assert!(system.starts_with("You are tasked with learning by induction"));
                }
                "text-embedding-3-small" => {
                    assert!(model_inference.system.is_none());
                    let messages = serde_json::to_string(
                        model_inference
                            .input_messages
                            .as_ref()
                            .expect("embedding model should have input_messages"),
                    )
                    .expect("input_messages should serialize");
                    // messages should be a json array of objects
                    assert!(messages.starts_with("[{"));
                }
                _ => {
                    panic!("Unknown model name: {}", model_inference.model_name);
                }
            }
        }
        assert_eq!(model_inferences.len(), 2);
    }
    assert_eq!(parsed_output.len(), 6);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_query_skips_staled_datapoints() {
    init_tracing_for_tests();
    let dataset_name = format!("exact_matches_empty-{}", Uuid::now_v7());
    let db = DelegatingDatabaseConnection::new_for_e2e_test().await;
    write_json_fixture_to_dataset(
        &db,
        &PathBuf::from(&format!(
            "{}/../tensorzero-core/fixtures/datasets/json_datapoint_fixture.jsonl",
            std::env::var("CARGO_MANIFEST_DIR").unwrap()
        )),
        &HashMap::from([("exact_matches_empty".to_string(), dataset_name.clone())]),
    )
    .await;

    let request = ListDatapointsRequest {
        function_name: Some("extract_entities".to_string()),
        limit: Some(u32::MAX), // Get all datapoints
        ..Default::default()
    };
    let dataset = list_datapoints(&db, dataset_name.clone(), request)
        .await
        .unwrap()
        .datapoints;

    // This ID should not be returned
    let staled_id = Uuid::parse_str("01957bbb-44a8-7490-bfe7-32f8ed2fc797").unwrap();
    let staled_datapoint = dataset.iter().find(|dp| dp.id() == staled_id);
    assert!(staled_datapoint.is_none());
    assert_eq!(dataset.len(), 21);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_evaluation_with_dynamic_variant() {
    init_tracing_for_tests();
    let dataset_name = format!("good-haiku-data-dynamic-{}", Uuid::now_v7());
    let db = DelegatingDatabaseConnection::new_for_e2e_test().await;
    write_chat_fixture_to_dataset(
        &db,
        &PathBuf::from(&format!(
            "{}/../tensorzero-core/fixtures/datasets/chat_datapoint_fixture.jsonl",
            std::env::var("CARGO_MANIFEST_DIR").unwrap()
        )),
        &HashMap::from([("good-haiku-data".to_string(), dataset_name.clone())]),
    )
    .await;

    let config_path = get_e2e_config_path();

    let tensorzero_client = ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
        config_file: Some(config_path.clone()),
        clickhouse_url: None,
        postgres_config: None,
        valkey_url: None,
        timeout: None,
        verify_credentials: true,
        allow_batch_writes: true,
    })
    .build()
    .await
    .unwrap();

    let config: Arc<Config> = match tensorzero_client.mode() {
        tensorzero_core::client::ClientMode::EmbeddedGateway { gateway, .. } => {
            gateway.handle.app_state.config.clone()
        }
        tensorzero_core::client::ClientMode::HTTPGateway(_) => {
            panic!("Expected EmbeddedGateway mode")
        }
    };

    // Create a dynamic variant with a simple configuration
    let dynamic_variant = UninitializedVariantInfo {
        inner: UninitializedVariantConfig::ChatCompletion(UninitializedChatCompletionConfig {
            model: "gpt-4o-mini".into(),
            weight: None,
            system_template: Some(ResolvedTomlPathData::new_fake_path(
                "test/system.minijinja".to_string(),
                "You are a helpful test assistant for dynamic variant testing.".to_string(),
            )),
            user_template: None,
            assistant_template: None,
            json_mode: None,
            temperature: None,
            max_tokens: None,
            seed: None,
            top_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            ..Default::default()
        }),
        timeouts: None,
        namespace: None,
    };

    let evaluation_run_id = Uuid::now_v7();
    let evaluation_name = "haiku_with_outputs".to_string();

    // Extract evaluation config fields
    let evaluation_config = config
        .evaluations
        .get(&evaluation_name)
        .expect("evaluation config should exist");
    let EvaluationConfig::Inference(inference_config) = &**evaluation_config;
    let function_config = config
        .functions
        .get(&inference_config.function_name)
        .map(|f| EvaluationFunctionConfig::from(f.as_ref()))
        .expect("function should exist");

    // Wrap the client in ClientInferenceExecutor for use with evaluations
    let inference_executor = Arc::new(ClientInferenceExecutor::new(tensorzero_client));

    let core_args = EvaluationCoreArgs {
        inference_executor,
        db: Arc::new(db.clone()),
        function_name: inference_config.function_name.clone(),
        function_config,
        evaluators: inference_config.evaluators.clone(),
        dataset_name: Some(dataset_name),
        datapoint_ids: Some(vec![]),
        variant: EvaluationVariant::Info(Box::new(dynamic_variant)),
        evaluation_name: Some(evaluation_name),
        evaluation_run_id,
        inference_cache: CacheEnabledMode::Off,
        concurrency: 2,
        tags: HashMap::new(),
    };

    let result = run_evaluation_core_streaming(core_args, None, HashMap::new()).await;
    assert!(
        result.is_ok(),
        "Evaluation with dynamic variant should succeed"
    );

    let result = result.unwrap();
    assert!(result.run_info.num_datapoints > 0);
}

/// Tests that `run_evaluation_core_streaming` correctly respects the `max_datapoints` parameter.
#[tokio::test(flavor = "multi_thread")]
async fn test_max_datapoints_parameter() {
    init_tracing_for_tests();
    let db = DelegatingDatabaseConnection::new_for_e2e_test().await;
    let dataset_name = format!("extract_entities_max_datapoints-{}", Uuid::now_v7());
    let tensorzero_client = make_embedded_gateway().await;

    // Write 10 datapoints to the dataset
    write_json_fixture_to_dataset(
        &db,
        &PathBuf::from(&format!(
            "{}/../tensorzero-core/fixtures/datasets/json_datapoint_fixture.jsonl",
            std::env::var("CARGO_MANIFEST_DIR").unwrap()
        )),
        &HashMap::from([("extract_entities_0.8".to_string(), dataset_name.clone())]),
    )
    .await;

    let config = get_config().await;

    let evaluation_run_id = Uuid::now_v7();
    let evaluation_name = "entity_extraction".to_string();

    // Extract evaluation config fields
    let evaluation_config = config
        .evaluations
        .get(&evaluation_name)
        .expect("evaluation config should exist");
    let EvaluationConfig::Inference(inference_config) = &**evaluation_config;
    let function_config = config
        .functions
        .get(&inference_config.function_name)
        .map(|f| EvaluationFunctionConfig::from(f.as_ref()))
        .expect("function should exist");

    // Wrap the client in ClientInferenceExecutor for use with evaluations
    let inference_executor = Arc::new(ClientInferenceExecutor::new(tensorzero_client.clone()));

    // Test with max_datapoints = 3 (should only process 3 datapoints)
    let core_args = EvaluationCoreArgs {
        inference_executor,
        db: Arc::new(db.clone()),
        function_name: inference_config.function_name.clone(),
        function_config,
        evaluators: inference_config.evaluators.clone(),
        dataset_name: Some(dataset_name.clone()),
        datapoint_ids: Some(vec![]),
        variant: EvaluationVariant::Name("gpt_4o_mini".to_string()),
        evaluation_name: Some(evaluation_name),
        evaluation_run_id,
        inference_cache: CacheEnabledMode::Off,
        concurrency: 2,
        tags: HashMap::new(),
    };

    let max_datapoints = Some(3);
    let result = run_evaluation_core_streaming(core_args, max_datapoints, HashMap::new())
        .await
        .unwrap();

    // Verify that only 3 datapoints were processed
    assert_eq!(
        result.run_info.num_datapoints, 3,
        "max_datapoints should limit dataset to 3 datapoints"
    );

    // Consume the results to ensure all evaluations complete
    let mut receiver = result.receiver;
    let mut success_count = 0;
    while let Some(update) = receiver.recv().await {
        if matches!(update, EvaluationUpdate::Success(_)) {
            success_count += 1;
        }
    }

    assert_eq!(
        success_count, 3,
        "Should have exactly 3 successful evaluations"
    );
}

/// Tests that `run_evaluation_core_streaming` correctly implements adaptive stopping with precision targets
/// for multiple evaluators.
#[tokio::test(flavor = "multi_thread")]
async fn test_precision_targets_parameter() {
    init_tracing_for_tests();
    let db = DelegatingDatabaseConnection::new_for_e2e_test().await;
    let dataset_name = format!("good-haiku-data-precision-{}", Uuid::now_v7());
    let tensorzero_client = make_embedded_gateway().await;

    // Use existing chat fixture that has both outputs (for exact_match) and inputs (for LLM judge)
    write_chat_fixture_to_dataset(
        &db,
        &PathBuf::from(&format!(
            "{}/../tensorzero-core/fixtures/datasets/chat_datapoint_fixture.jsonl",
            std::env::var("CARGO_MANIFEST_DIR").unwrap()
        )),
        &HashMap::from([("good-haiku-data".to_string(), dataset_name.clone())]),
    )
    .await;

    let config = get_config().await;

    let evaluation_run_id = Uuid::now_v7();
    let evaluation_name = "haiku_without_outputs".to_string(); // Has both exact_match and topic_starts_with_f
    let evaluation_name_tag = evaluation_name.clone();

    // Extract evaluation config fields
    let evaluation_config = config
        .evaluations
        .get(&evaluation_name)
        .expect("evaluation config should exist");
    let EvaluationConfig::Inference(inference_config) = &**evaluation_config;
    let function_config = config
        .functions
        .get(&inference_config.function_name)
        .map(|f| EvaluationFunctionConfig::from(f.as_ref()))
        .expect("function should exist");

    // Wrap the client in ClientInferenceExecutor for use with evaluations
    let inference_executor = Arc::new(ClientInferenceExecutor::new(tensorzero_client.clone()));

    let external_session_id = "session-123".to_string();
    let external_tag_value = "external-value".to_string();
    let external_tags = HashMap::from([
        (
            "tensorzero::autopilot::session_id".to_string(),
            external_session_id.clone(),
        ),
        ("custom_tag".to_string(), external_tag_value.clone()),
    ]);

    // Set precision targets for both evaluators
    // exact_match: CI half-width <= 0.10
    // topic_starts_with_f: CI half-width <= 0.13
    let mut precision_targets = HashMap::new();
    precision_targets.insert("exact_match".to_string(), 0.20);
    precision_targets.insert("topic_starts_with_f".to_string(), 0.13);

    let core_args = EvaluationCoreArgs {
        inference_executor,
        db: Arc::new(db.clone()),
        function_name: inference_config.function_name.clone(),
        function_config,
        evaluators: inference_config.evaluators.clone(),
        dataset_name: Some(dataset_name.clone()),
        datapoint_ids: Some(vec![]),
        variant: EvaluationVariant::Name("gpt_4o_mini".to_string()),
        evaluation_name: Some(evaluation_name),
        evaluation_run_id,
        inference_cache: CacheEnabledMode::Off,
        concurrency: 5,
        tags: external_tags.clone(),
    };

    // Run with precision targets
    let result = run_evaluation_core_streaming(
        core_args,
        None, // No max_datapoints limit
        precision_targets.clone(),
    )
    .await
    .unwrap();

    // Consume results and track evaluations, computing statistics as we go
    let batcher_join_handles = result.batcher_join_handles.clone();
    let mut receiver = result.receiver;
    let mut exact_match_stats = PerEvaluatorStats::new(true); // Bernoulli evaluator (uses Wilson CI)
    let mut topic_stats = PerEvaluatorStats::new(false); // Treated as float evaluator (uses Wald CI)
    let mut total_datapoints = 0;
    let mut tagged_inference_id = None;

    while let Some(update) = receiver.recv().await {
        if let EvaluationUpdate::Success(info) = update {
            total_datapoints += 1;
            if tagged_inference_id.is_none() {
                tagged_inference_id = Some(info.response.inference_id());
            }

            // Track exact_match values (boolean)
            if let Some(Some(serde_json::Value::Bool(b))) = info.evaluations.get("exact_match") {
                exact_match_stats.push(if *b { 1.0 } else { 0.0 });
            }

            // Track topic_starts_with_f values (boolean, but treated as float for testing Wald CI)
            if let Some(Some(serde_json::Value::Bool(b))) =
                info.evaluations.get("topic_starts_with_f")
            {
                topic_stats.push(if *b { 1.0 } else { 0.0 });
            }
        }
    }

    // Verify min_datapoints constraint (hardcoded to 20 in StoppingManager)
    assert!(
        total_datapoints >= 20,
        "Should process at least min_datapoints (20) datapoints, got {total_datapoints}"
    );

    // Verify that both evaluators achieved their precision targets
    let exact_match_ci = exact_match_stats.ci_half_width();
    let topic_ci = topic_stats.ci_half_width();

    // Assert that achieved precision is within limits
    assert!(
        exact_match_ci.is_some(),
        "exact_match should have computed CI half-width"
    );
    assert!(
        exact_match_ci.unwrap() <= precision_targets["exact_match"],
        "exact_match CI half-width {:.3} should be <= limit {:.3}",
        exact_match_ci.unwrap(),
        precision_targets["exact_match"]
    );

    assert!(
        topic_ci.is_some(),
        "topic_starts_with_f should have computed CI half-width"
    );
    assert!(
        topic_ci.unwrap() <= precision_targets["topic_starts_with_f"],
        "topic_starts_with_f CI half-width {:.3} should be <= limit {:.3}",
        topic_ci.unwrap(),
        precision_targets["topic_starts_with_f"]
    );

    for handle in batcher_join_handles {
        handle
            .await
            .expect("Batch writer should complete before tag assertions");
    }
    db.flush_pending_writes().await;
    sleep(Duration::from_secs(5)).await;

    let inference_id =
        tagged_inference_id.expect("Should capture an inference id to validate tags");
    let StoredInferenceDatabase::Chat(db_inference) = query_inference(&db, &config, inference_id)
        .await
        .expect("Should load evaluation inference from database")
    else {
        panic!("Expected chat inference");
    };
    assert_eq!(
        db_inference.tags["tensorzero::evaluation_run_id"],
        evaluation_run_id.to_string(),
        "Evaluation inferences should include `tensorzero::evaluation_run_id` tags"
    );
    assert_eq!(
        db_inference.tags["tensorzero::evaluation_name"],
        evaluation_name_tag.as_str(),
        "Evaluation inferences should include `tensorzero::evaluation_name` tags"
    );
    assert_eq!(
        db_inference.tags["tensorzero::autopilot::session_id"],
        external_session_id.as_str(),
        "Evaluation inferences should include external tags"
    );
    assert_eq!(
        db_inference.tags["custom_tag"],
        external_tag_value.as_str(),
        "Evaluation inferences should include custom external tags"
    );

    let metric_name = format!(
        "tensorzero::evaluation_name::{}::evaluator_name::topic_starts_with_f",
        evaluation_name_tag.as_str()
    );
    let db_feedback = query_boolean_feedback(&db, inference_id, Some(metric_name.as_str()))
        .await
        .expect("Should load evaluator feedback from database");
    assert_eq!(
        db_feedback.tags["tensorzero::autopilot::session_id"],
        external_session_id.as_str(),
        "Evaluator feedback should include external tags"
    );
    assert_eq!(
        db_feedback.tags["custom_tag"],
        external_tag_value.as_str(),
        "Evaluator feedback should include custom external tags"
    );
    let evaluator_inference_id =
        Uuid::parse_str(&db_feedback.tags["tensorzero::evaluator_inference_id"]).unwrap();
    let StoredInferenceDatabase::Json(evaluator_inference) =
        query_inference(&db, &config, evaluator_inference_id)
            .await
            .expect("Should load judge inference from database")
    else {
        panic!("Expected json inference for evaluator");
    };
    assert_eq!(
        evaluator_inference.tags["tensorzero::evaluation_run_id"],
        evaluation_run_id.to_string(),
        "Judge inferences should include `tensorzero::evaluation_run_id` tags"
    );
    assert_eq!(
        evaluator_inference.tags["tensorzero::evaluation_name"],
        evaluation_name_tag.as_str(),
        "Judge inferences should include `tensorzero::evaluation_name` tags"
    );
    assert_eq!(
        evaluator_inference.tags["tensorzero::autopilot::session_id"],
        external_session_id.as_str(),
        "Judge inferences should include external tags"
    );
    assert_eq!(
        evaluator_inference.tags["custom_tag"],
        external_tag_value.as_str(),
        "Judge inferences should include custom external tags"
    );
}

/// Tests that the CLI interface (`run_evaluation`) correctly respects the `max_datapoints` constraint.
#[tokio::test(flavor = "multi_thread")]
async fn test_cli_args_max_datapoints() {
    init_tracing_for_tests();
    let db = DelegatingDatabaseConnection::new_for_e2e_test().await;
    let dataset_name = format!("good-haiku-data-cli-max-{}", Uuid::now_v7());
    write_chat_fixture_to_dataset(
        &db,
        &PathBuf::from(&format!(
            "{}/../tensorzero-core/fixtures/datasets/chat_datapoint_fixture.jsonl",
            std::env::var("CARGO_MANIFEST_DIR").unwrap()
        )),
        &HashMap::from([("good-haiku-data".to_string(), dataset_name.clone())]),
    )
    .await;

    let config_path = get_e2e_config_path();
    let evaluation_run_id = Uuid::now_v7();

    // Test CLI Args with max_datapoints limit
    let args = Args {
        config_file: config_path,
        gateway_url: None,
        evaluation_name: Some("haiku_with_outputs".to_string()),
        function_name: None,
        evaluator_names: None,
        dataset_name: Some(dataset_name.clone()),
        datapoint_ids: Some(vec![]),
        variant_name: "gpt_4o_mini".to_string(),
        concurrency: 10,
        format: OutputFormat::Jsonl,
        inference_cache: CacheEnabledMode::Off,
        max_datapoints: Some(21),
        precision_targets: vec![],
        cutoffs: vec![],
    };

    let mut output = Vec::new();
    Box::pin(run_evaluation(args, evaluation_run_id, &mut output))
        .await
        .unwrap();

    // Parse output and verify max_datapoints constraint was respected
    let output_str = String::from_utf8(output).unwrap();
    let lines: Vec<&str> = output_str.lines().collect();

    // First line should be RunInfo
    let run_info: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
    let num_datapoints = run_info["num_datapoints"].as_u64().unwrap() as usize;

    // Should be bounded between MIN_DATAPOINTS and max_datapoints (21)
    assert!(
        num_datapoints >= MIN_DATAPOINTS,
        "Should have at least MIN_DATAPOINTS ({MIN_DATAPOINTS}) inferences, got {num_datapoints}"
    );
    assert!(
        num_datapoints <= 21,
        "Should not exceed max_datapoints (20), got {num_datapoints}"
    );
}

/// Tests that the CLI interface (`run_evaluation`) correctly implements adaptive stopping with precision targets.
#[tokio::test(flavor = "multi_thread")]
async fn test_cli_args_precision_targets() {
    init_tracing_for_tests();
    let db = DelegatingDatabaseConnection::new_for_e2e_test().await;
    let dataset_name = format!("good-haiku-data-cli-precision-{}", Uuid::now_v7());
    write_chat_fixture_to_dataset(
        &db,
        &PathBuf::from(&format!(
            "{}/../tensorzero-core/fixtures/datasets/chat_datapoint_fixture.jsonl",
            std::env::var("CARGO_MANIFEST_DIR").unwrap()
        )),
        &HashMap::from([("good-haiku-data".to_string(), dataset_name.clone())]),
    )
    .await;

    let config_path = get_e2e_config_path();
    let evaluation_run_id = Uuid::now_v7();

    // Test CLI Args with precision_targets for adaptive stopping
    // Set a liberal precision target so the test completes quickly
    let args = Args {
        config_file: config_path,
        gateway_url: None,
        evaluation_name: Some("haiku_with_outputs".to_string()),
        function_name: None,
        evaluator_names: None,
        dataset_name: Some(dataset_name.clone()),
        datapoint_ids: Some(vec![]),
        variant_name: "gpt_4o_mini".to_string(),
        concurrency: 10,
        format: OutputFormat::Jsonl,
        inference_cache: CacheEnabledMode::Off,
        max_datapoints: None,
        precision_targets: vec![("exact_match".to_string(), 0.2)],
        cutoffs: vec![],
    };

    let mut output = Vec::new();
    Box::pin(run_evaluation(args, evaluation_run_id, &mut output))
        .await
        .unwrap();

    // Parse output and verify precision target was reached
    let output_str = String::from_utf8(output).unwrap();
    let lines: Vec<&str> = output_str.lines().collect();

    // First line should be RunInfo
    let run_info: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
    let num_datapoints = run_info["num_datapoints"].as_u64().unwrap() as usize;

    // Collect evaluation results and compute CI half-width for exact_match
    let mut exact_match_values = Vec::new();
    for line in lines.iter().skip(1) {
        if let Ok(result) = serde_json::from_str::<serde_json::Value>(line) {
            // Each line (after the first) is a result with evaluations
            if let Some(exact_match) = result["evaluations"]["exact_match"].as_bool() {
                exact_match_values.push(if exact_match { 1.0 } else { 0.0 });
            }
        }
    }

    // Compute CI half-width
    let mut stats = PerEvaluatorStats::default();
    for value in exact_match_values {
        stats.push(value);
    }

    let ci_half_width = stats.ci_half_width();
    assert!(
        ci_half_width.is_some(),
        "Should have computed CI half-width for exact_match"
    );

    // Verify that the CI half-width meets the precision target
    let ci_half_width = ci_half_width.unwrap();
    assert!(
        ci_half_width <= 0.2,
        "CI half-width {ci_half_width:.3} should be <= precision target 0.2"
    );

    // Should have processed at least MIN_DATAPOINTS datapoints
    assert!(
        num_datapoints >= MIN_DATAPOINTS,
        "Should have at least MIN_DATAPOINTS ({MIN_DATAPOINTS}) inferences, got {num_datapoints}"
    );
}

/// Tests the `--function-name` / `--evaluator-names` CLI path, which resolves
/// top-level evaluators by name instead of using a named evaluation config.
#[tokio::test(flavor = "multi_thread")]
async fn run_evaluation_with_function_name_and_evaluator_names() {
    init_tracing_for_tests();
    let db = DelegatingDatabaseConnection::new_for_e2e_test().await;
    let config = get_config().await;
    let dataset_name = format!("extract_entities_top_level-{}", Uuid::now_v7());
    write_json_fixture_to_dataset(
        &db,
        &PathBuf::from(&format!(
            "{}/../tensorzero-core/fixtures/datasets/json_datapoint_fixture.jsonl",
            std::env::var("CARGO_MANIFEST_DIR").unwrap()
        )),
        &HashMap::from([("extract_entities_0.8".to_string(), dataset_name.clone())]),
    )
    .await;

    let config_path = get_e2e_config_path();
    let evaluation_run_id = Uuid::now_v7();

    let args = Args {
        config_file: config_path,
        gateway_url: None,
        evaluation_name: None,
        function_name: Some("extract_entities".to_string()),
        evaluator_names: Some(vec!["exact_match".to_string()]),
        dataset_name: Some(dataset_name.clone()),
        datapoint_ids: Some(vec![]),
        variant_name: "gpt_4o_mini".to_string(),
        concurrency: 10,
        format: OutputFormat::Jsonl,
        inference_cache: CacheEnabledMode::Off,
        max_datapoints: None,
        precision_targets: vec![],
        cutoffs: vec![],
    };

    let mut output = Vec::new();
    Box::pin(run_evaluation(args, evaluation_run_id, &mut output))
        .await
        .expect("Evaluation with --function-name/--evaluator-names should succeed");
    db.flush_pending_writes().await;
    sleep(Duration::from_secs(5)).await;

    let output_str = String::from_utf8(output).unwrap();
    let output_lines: Vec<&str> = output_str.lines().collect();

    // First line is RunInfo
    let run_info: serde_json::Value =
        serde_json::from_str(output_lines[0]).expect("RunInfo should be valid JSON");
    assert_eq!(run_info["evaluation_run_id"], evaluation_run_id.to_string());
    assert!(
        run_info["num_datapoints"].as_u64().unwrap() > 0,
        "Should have datapoints"
    );

    // Parse result lines
    let mut successes = 0;
    for line in output_lines.iter().skip(1) {
        let parsed: EvaluationUpdate =
            serde_json::from_str(line).expect("Each line should be valid JSON");
        let info = match parsed {
            EvaluationUpdate::Success(info) => info,
            EvaluationUpdate::Error(err) => panic!("Unexpected evaluation error: {}", err.message),
            EvaluationUpdate::RunInfo(_) => continue,
        };

        // Should only have exact_match evaluator results
        assert!(
            info.evaluations.contains_key("exact_match"),
            "Should have exact_match evaluation result"
        );
        assert_eq!(
            info.evaluations.len(),
            1,
            "Should only have one evaluator (exact_match)"
        );
        assert!(
            info.evaluator_errors.is_empty(),
            "Should have no evaluator errors"
        );

        // Verify feedback was written with top-level evaluator metric naming
        let inference_id = info.response.inference_id();
        let feedback = query_boolean_feedback(
            &db,
            inference_id,
            Some("tensorzero::evaluator::exact_match"),
        )
        .await
        .expect("Should find boolean feedback for top-level evaluator");

        assert_eq!(
            feedback.metric_name, "tensorzero::evaluator::exact_match",
            "Metric name should use top-level evaluator naming (no evaluation_name prefix)"
        );
        assert_eq!(
            feedback.tags["tensorzero::evaluation_run_id"],
            evaluation_run_id.to_string()
        );
        assert_eq!(feedback.tags["tensorzero::evaluator_name"], "exact_match");
        // Top-level evaluators should not have an evaluation_name tag
        assert!(
            !feedback.tags.contains_key("tensorzero::evaluation_name"),
            "Top-level evaluator feedback should not have evaluation_name tag"
        );

        // Verify the inference is properly tagged (no evaluation_name)
        let StoredInferenceDatabase::Json(db_inference) =
            query_inference(&db, &config, inference_id)
                .await
                .expect("Should find json inference")
        else {
            panic!("Expected json inference");
        };
        assert_eq!(
            db_inference.tags["tensorzero::evaluation_run_id"],
            evaluation_run_id.to_string()
        );
        assert_eq!(db_inference.tags["tensorzero::dataset_name"], dataset_name);
        assert!(
            !db_inference
                .tags
                .contains_key("tensorzero::evaluation_name"),
            "Top-level evaluator inference should not have evaluation_name tag"
        );

        successes += 1;
    }
    assert_eq!(successes, 6, "Should have 6 successful evaluations");
}
