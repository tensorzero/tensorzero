#![cfg_attr(
    test,
    allow(clippy::expect_used, clippy::unwrap_used, clippy::print_stdout)
)]
mod common;
use clap::Parser;
use evaluations::dataset::query_dataset;
use evaluations::evaluators::llm_judge::{run_llm_judge_evaluator, RunLLMJudgeEvaluatorParams};
use evaluations::{Clients, ThrottledTensorZeroClient};
use serde_json::json;
use tensorzero_core::cache::CacheEnabledMode;
use tensorzero_core::client::input_handling::resolved_input_to_client_input;
use tensorzero_core::db::clickhouse::test_helpers::{
    select_inference_evaluation_human_feedback_clickhouse, select_model_inferences_clickhouse,
};
use tensorzero_core::endpoints::datasets::StoredDatapoint;
use tensorzero_core::evaluations::{LLMJudgeConfig, LLMJudgeInputFormat, LLMJudgeOutputType};
use tensorzero_core::function::{FunctionConfig, FunctionConfigJson};
use tensorzero_core::inference::types::{
    StoredInput, StoredInputMessage, StoredInputMessageContent, Text,
};
use tokio::time::sleep;
use url::Url;

use crate::common::write_json_fixture_to_dataset;
use common::{get_tensorzero_client, write_chat_fixture_to_dataset};
use evaluations::{run_evaluation, stats::EvaluationUpdate, Args, OutputFormat};
use std::collections::HashMap;
use std::time::Duration;
use std::{path::PathBuf, sync::Arc};
use tensorzero_core::client::{
    ClientBuilder, ClientBuilderMode, FeedbackParams, InferenceResponse, Role,
};
use tensorzero_core::{
    db::clickhouse::test_helpers::{
        clickhouse_flush_async_insert, get_clickhouse, select_chat_inference_clickhouse,
        select_feedback_by_target_id_clickhouse, select_json_inference_clickhouse,
    },
    inference::types::{ContentBlockChatOutput, JsonInferenceOutput, Usage},
};
use tensorzero_core::{
    endpoints::{
        datasets::{JsonInferenceDatapoint, StoredChatInferenceDatapoint},
        inference::{ChatInferenceResponse, JsonInferenceResponse},
    },
    evaluations::{LLMJudgeIncludeConfig, LLMJudgeOptimize},
};
use tokio::sync::Semaphore;
use uuid::Uuid;

pub fn init_tracing_for_tests() {
    tracing_subscriber::fmt().init();
}

#[tokio::test(flavor = "multi_thread")]
async fn run_evaluations_json() {
    init_tracing_for_tests();
    let clickhouse = get_clickhouse().await;
    let dataset_name = format!("extract_entities_0.8-{}", Uuid::now_v7());
    let tensorzero_client = get_tensorzero_client().await;
    write_json_fixture_to_dataset(
        &PathBuf::from(&format!(
            "{}/../tensorzero-core/fixtures/datasets/json_datapoint_fixture.jsonl",
            std::env::var("CARGO_MANIFEST_DIR").unwrap()
        )),
        &HashMap::from([("extract_entities_0.8".to_string(), dataset_name.clone())]),
    )
    .await;
    let config_path = PathBuf::from(&format!(
        "{}/../tensorzero-core/tests/e2e/tensorzero.toml",
        std::env::var("CARGO_MANIFEST_DIR").unwrap()
    ));
    let evaluation_run_id = Uuid::now_v7();
    let args = || Args {
        config_file: config_path.clone(),
        gateway_url: None,
        evaluation_name: "entity_extraction".to_string(),
        dataset_name: dataset_name.clone(),
        variant_name: "gpt_4o_mini".to_string(),
        concurrency: 10,
        format: OutputFormat::Jsonl,
        // This test relies on the cache (see below), so we need to enable it
        inference_cache: CacheEnabledMode::On,
    };

    let mut output = Vec::new();
    run_evaluation(args(), evaluation_run_id, &mut output)
        .await
        .unwrap();
    clickhouse_flush_async_insert(&clickhouse).await;
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
        let clickhouse_inference = select_json_inference_clickhouse(&clickhouse, inference_id)
            .await
            .unwrap();
        let parsed_response = match parsed.response.clone() {
            InferenceResponse::Json(json_response) => json_response,
            InferenceResponse::Chat(..) => panic!("Chat response not supported"),
        };
        let clickhouse_input: StoredInput =
            serde_json::from_str(clickhouse_inference["input"].as_str().unwrap()).unwrap();
        // Check the input to the inference is the same as the input to the datapoint
        assert_eq!(&clickhouse_input, parsed.datapoint.input());
        let clickhouse_output: JsonInferenceOutput =
            serde_json::from_str(clickhouse_inference["output"].as_str().unwrap()).unwrap();
        // Check the output to the inference is the same as the output in the response
        assert_eq!(clickhouse_output, parsed_response.output);
        // Check the inference is properly tagged
        assert_eq!(
            clickhouse_inference["tags"]["tensorzero::evaluation_run_id"],
            evaluation_run_id.to_string()
        );
        assert_eq!(
            clickhouse_inference["tags"]["tensorzero::datapoint_id"],
            parsed.datapoint.id().to_string()
        );
        assert_eq!(
            clickhouse_inference["tags"]["tensorzero::evaluation_name"],
            "entity_extraction"
        );
        assert_eq!(
            clickhouse_inference["tags"]["tensorzero::dataset_name"],
            dataset_name
        );
        // Check boolean feedback was recorded
        let feedback = select_feedback_by_target_id_clickhouse(
            &clickhouse,
            "BooleanMetricFeedback",
            inference_id,
            None,
        )
        .await
        .unwrap();

        assert_eq!(
            feedback["metric_name"].as_str().unwrap(),
            "tensorzero::evaluation_name::entity_extraction::evaluator_name::exact_match"
        );
        assert_eq!(
            feedback["value"],
            parsed.evaluations["exact_match"]
                .as_ref()
                .unwrap()
                .as_bool()
                .unwrap()
        );
        assert_eq!(
            feedback["tags"]["tensorzero::evaluation_run_id"],
            evaluation_run_id.to_string()
        );
        assert_eq!(
            feedback["tags"]["tensorzero::datapoint_id"],
            parsed.datapoint.id().to_string()
        );

        // Check float feedback was recorded
        let feedback = select_feedback_by_target_id_clickhouse(
            &clickhouse,
            "FloatMetricFeedback",
            inference_id,
            None,
        )
        .await
        .unwrap();

        assert_eq!(
            feedback["metric_name"].as_str().unwrap(),
            "tensorzero::evaluation_name::entity_extraction::evaluator_name::count_sports"
        );
        assert_eq!(
            feedback["value"],
            parsed.evaluations["count_sports"]
                .as_ref()
                .unwrap()
                .as_f64()
                .unwrap()
        );
        assert_eq!(
            feedback["tags"]["tensorzero::evaluation_run_id"],
            evaluation_run_id.to_string()
        );
        let datapoint_id = parsed.datapoint.id();
        assert_eq!(
            feedback["tags"]["tensorzero::datapoint_id"],
            datapoint_id.to_string()
        );
        assert_eq!(
            feedback["tags"]["tensorzero::evaluator_name"],
            "count_sports"
        );
        assert_eq!(
            feedback["tags"]["tensorzero::evaluation_name"],
            "entity_extraction"
        );
        assert!(feedback["tags"]
            .get("tensorzero::derived_from_human_feedback")
            .is_none());
        let evaluator_inference_id = Uuid::parse_str(
            feedback["tags"]["tensorzero::evaluator_inference_id"]
                .as_str()
                .unwrap(),
        )
        .unwrap();
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
        let evaluator_inference =
            select_json_inference_clickhouse(&clickhouse, evaluator_inference_id)
                .await
                .unwrap();
        assert_eq!(
            evaluator_inference["tags"]["tensorzero::evaluation_name"],
            "entity_extraction"
        );
        total_sports += feedback["value"].as_f64().unwrap() as u32;
        parsed_output.push(parsed);
        // Sleep for 5s to make sure the feedback is recorded
        sleep(Duration::from_secs(5)).await;

        let human_feedback = select_inference_evaluation_human_feedback_clickhouse(
            &clickhouse,
            metric_name,
            datapoint_id,
            serde_json::to_string(&clickhouse_output).unwrap().as_str(),
        )
        .await
        .unwrap();
        assert_eq!(human_feedback.value, "0");
        assert_eq!(
            human_feedback.evaluator_inference_id,
            Some(evaluator_inference_id)
        );
    }
    assert_eq!(parsed_output.len(), 6);
    assert_eq!(total_sports, 3);
    sleep(Duration::from_secs(5)).await;

    // Check that the human feedback affects the next eval run results
    // Run the evaluation again but now it should read the human feedback that was sent
    let mut output = Vec::new();
    run_evaluation(args(), evaluation_run_id, &mut output)
        .await
        .unwrap();
    clickhouse_flush_async_insert(&clickhouse).await;
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
        // Grab the feedback from ClickHouse for the second run and make sure it has the human feedback tag
        let clickhouse_feedback = select_feedback_by_target_id_clickhouse(
            &clickhouse,
            "FloatMetricFeedback",
            inference_id,
            None,
        )
        .await
        .unwrap();
        assert_eq!(
            clickhouse_feedback["tags"]["tensorzero::derived_from_human_feedback"],
            "true",
        );
        assert_eq!(
            clickhouse_feedback["tags"]["tensorzero::evaluator_inference_id"],
            evaluator_inference_ids[&parsed.datapoint.id()].to_string()
        );
    }
    assert_eq!(total_sports, 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn run_exact_match_evaluation_chat() {
    init_tracing_for_tests();
    let dataset_name = format!("good-haiku-data-{}", Uuid::now_v7());
    let clickhouse = get_clickhouse().await;
    write_chat_fixture_to_dataset(
        &PathBuf::from(&format!(
            "{}/../tensorzero-core/fixtures/datasets/chat_datapoint_fixture.jsonl",
            std::env::var("CARGO_MANIFEST_DIR").unwrap()
        )),
        &HashMap::from([("good-haiku-data".to_string(), dataset_name.clone())]),
    )
    .await;
    let config_path = PathBuf::from(&format!(
        "{}/../tensorzero-core/tests/e2e/tensorzero.toml",
        std::env::var("CARGO_MANIFEST_DIR").unwrap()
    ));
    let evaluation_run_id = Uuid::now_v7();
    let args = Args {
        config_file: config_path,
        gateway_url: None,
        evaluation_name: "haiku_with_outputs".to_string(),
        dataset_name: dataset_name.clone(),
        variant_name: "gpt_4o_mini".to_string(),
        concurrency: 10,
        format: OutputFormat::Jsonl,
        inference_cache: CacheEnabledMode::Off,
    };

    let mut output = Vec::new();
    run_evaluation(args, evaluation_run_id, &mut output)
        .await
        .unwrap();
    clickhouse_flush_async_insert(&clickhouse).await;
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
        let clickhouse_inference = select_chat_inference_clickhouse(&clickhouse, inference_id)
            .await
            .unwrap();
        let parsed_response = match parsed.response.clone() {
            InferenceResponse::Chat(chat_response) => chat_response,
            InferenceResponse::Json(..) => panic!("Json response not supported"),
        };
        let clickhouse_input: StoredInput =
            serde_json::from_str(clickhouse_inference["input"].as_str().unwrap()).unwrap();
        // Check the input to the inference is the same as the input to the datapoint
        assert_eq!(&clickhouse_input, parsed.datapoint.input());
        let clickhouse_output: Vec<ContentBlockChatOutput> =
            serde_json::from_str(clickhouse_inference["output"].as_str().unwrap()).unwrap();
        // Check the output to the inference is the same as the output in the response
        assert_eq!(clickhouse_output, parsed_response.content);
        // Check the inference is properly tagged
        assert_eq!(
            clickhouse_inference["tags"]["tensorzero::evaluation_run_id"],
            evaluation_run_id.to_string()
        );
        assert_eq!(
            clickhouse_inference["tags"]["tensorzero::datapoint_id"],
            parsed.datapoint.id().to_string()
        );
        assert_eq!(
            clickhouse_inference["tags"]["tensorzero::evaluation_name"],
            "haiku_with_outputs"
        );
        assert_eq!(
            clickhouse_inference["tags"]["tensorzero::dataset_name"],
            dataset_name
        );
        let clickhouse_feedback = select_feedback_by_target_id_clickhouse(
            &clickhouse,
            "BooleanMetricFeedback",
            inference_id,
            None,
        )
        .await
        .unwrap();
        assert_eq!(
            clickhouse_feedback["metric_name"].as_str().unwrap(),
            "tensorzero::evaluation_name::haiku_with_outputs::evaluator_name::exact_match"
        );
        assert_eq!(
            clickhouse_feedback["value"],
            parsed.evaluations["exact_match"]
                .as_ref()
                .unwrap()
                .as_bool()
                .unwrap()
        );
        assert_eq!(
            clickhouse_feedback["tags"]["tensorzero::evaluation_run_id"],
            evaluation_run_id.to_string()
        );
        assert_eq!(
            clickhouse_feedback["tags"]["tensorzero::datapoint_id"],
            parsed.datapoint.id().to_string()
        );
        assert_eq!(
            clickhouse_feedback["tags"]["tensorzero::evaluation_name"],
            "haiku_with_outputs"
        );
        assert_eq!(
            clickhouse_feedback["tags"]["tensorzero::evaluator_name"],
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
    let clickhouse = get_clickhouse().await;
    write_chat_fixture_to_dataset(
        &PathBuf::from(&format!(
            "{}/../tensorzero-core/fixtures/datasets/chat_datapoint_fixture.jsonl",
            std::env::var("CARGO_MANIFEST_DIR").unwrap()
        )),
        &HashMap::from([("good-haikus-no-output".to_string(), dataset_name.clone())]),
    )
    .await;
    let config_path = PathBuf::from(&format!(
        "{}/../tensorzero-core/tests/e2e/tensorzero.toml",
        std::env::var("CARGO_MANIFEST_DIR").unwrap()
    ));
    let tensorzero_client = get_tensorzero_client().await;
    let evaluation_run_id = Uuid::now_v7();
    let args = || Args {
        config_file: config_path.clone(),
        gateway_url: None,
        dataset_name: dataset_name.clone(),
        evaluation_name: "haiku_without_outputs".to_string(),
        variant_name: "gpt_4o_mini".to_string(),
        concurrency: 10,
        format: OutputFormat::Jsonl,
        inference_cache: CacheEnabledMode::On,
    };

    let mut output = Vec::new();
    run_evaluation(args(), evaluation_run_id, &mut output)
        .await
        .unwrap();
    clickhouse_flush_async_insert(&clickhouse).await;
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
        let clickhouse_inference = select_chat_inference_clickhouse(&clickhouse, inference_id)
            .await
            .unwrap();
        let parsed_response = match parsed.response.clone() {
            InferenceResponse::Chat(chat_response) => chat_response,
            InferenceResponse::Json(..) => panic!("Json response not supported"),
        };
        let clickhouse_input: StoredInput =
            serde_json::from_str(clickhouse_inference["input"].as_str().unwrap()).unwrap();
        // Check the input to the inference is the same as the input to the datapoint
        assert_eq!(&clickhouse_input, parsed.datapoint.input());
        let clickhouse_output: Vec<ContentBlockChatOutput> =
            serde_json::from_str(clickhouse_inference["output"].as_str().unwrap()).unwrap();
        // Check the output to the inference is the same as the output in the response
        assert_eq!(clickhouse_output, parsed_response.content);
        // Check the inference is properly tagged
        assert_eq!(
            clickhouse_inference["tags"]["tensorzero::evaluation_run_id"],
            evaluation_run_id.to_string()
        );
        assert_eq!(
            clickhouse_inference["tags"]["tensorzero::datapoint_id"],
            parsed.datapoint.id().to_string()
        );
        assert_eq!(
            clickhouse_inference["tags"]["tensorzero::evaluation_name"],
            "haiku_without_outputs"
        );

        // There should be no Float feedback for this evaluation
        assert!(select_feedback_by_target_id_clickhouse(
            &clickhouse,
            "FloatMetricFeedback",
            inference_id,
            None,
        )
        .await
        .is_none());
        // The exact match evaluation should have value None since there is no output in any of these
        assert!(parsed.evaluations["exact_match"].is_none());

        // There should be Boolean feedback for this evaluation
        let clickhouse_feedback = select_feedback_by_target_id_clickhouse(
            &clickhouse,
            "BooleanMetricFeedback",
            inference_id,
            None,
        )
        .await
        .unwrap();
        assert_eq!(
            clickhouse_feedback["metric_name"].as_str().unwrap(),
            "tensorzero::evaluation_name::haiku_without_outputs::evaluator_name::topic_starts_with_f"
        );
        assert_eq!(
            clickhouse_feedback["value"],
            parsed.evaluations["topic_starts_with_f"]
                .as_ref()
                .unwrap()
                .as_bool()
                .unwrap()
        );
        total_topic_fs += clickhouse_feedback["value"].as_bool().unwrap() as u32;
        assert_eq!(
            clickhouse_feedback["tags"]["tensorzero::evaluation_run_id"],
            evaluation_run_id.to_string()
        );
        assert_eq!(
            clickhouse_feedback["tags"]["tensorzero::datapoint_id"],
            parsed.datapoint.id().to_string()
        );
        assert_eq!(
            clickhouse_feedback["tags"]["tensorzero::evaluation_name"],
            "haiku_without_outputs"
        );
        assert_eq!(
            clickhouse_feedback["tags"]["tensorzero::evaluator_name"],
            "topic_starts_with_f"
        );
        assert!(clickhouse_feedback["tags"]
            .get("tensorzero::derived_from_human_feedback")
            .is_none());
        let evaluator_inference_id = Uuid::parse_str(
            clickhouse_feedback["tags"]["tensorzero::evaluator_inference_id"]
                .as_str()
                .unwrap(),
        )
        .unwrap();
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

        let evaluator_inference =
            select_json_inference_clickhouse(&clickhouse, evaluator_inference_id)
                .await
                .unwrap();
        assert_eq!(
            evaluator_inference["tags"]["tensorzero::evaluation_name"],
            "haiku_without_outputs"
        );
        parsed_output.push(parsed);
    }
    assert_eq!(parsed_output.len(), 10);
    assert_eq!(total_topic_fs, 3);
    sleep(Duration::from_secs(5)).await;
    // Run the evaluation again but now it should read the human feedback that was sent
    let mut output = Vec::new();
    run_evaluation(args(), evaluation_run_id, &mut output)
        .await
        .unwrap();
    clickhouse_flush_async_insert(&clickhouse).await;
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
        // Grab the feedback from ClickHouse for the second run and make sure it has the human feedback tag
        let clickhouse_feedback = select_feedback_by_target_id_clickhouse(
            &clickhouse,
            "BooleanMetricFeedback",
            inference_id,
            None,
        )
        .await
        .unwrap();
        assert_eq!(
            clickhouse_feedback["tags"]["tensorzero::derived_from_human_feedback"],
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
    let clickhouse = get_clickhouse().await;
    write_chat_fixture_to_dataset(
        &PathBuf::from(&format!(
            "{}/../tensorzero-core/fixtures/datasets/chat_datapoint_fixture.jsonl",
            std::env::var("CARGO_MANIFEST_DIR").unwrap()
        )),
        &HashMap::from([("baz".to_string(), dataset_name.clone())]),
    )
    .await;
    let config_path = PathBuf::from(&format!(
        "{}/../tensorzero-core/tests/e2e/tensorzero.toml",
        std::env::var("CARGO_MANIFEST_DIR").unwrap()
    ));
    let evaluation_run_id = Uuid::now_v7();
    let args = Args {
        config_file: config_path,
        gateway_url: None,
        dataset_name: dataset_name.clone(),
        evaluation_name: "images".to_string(),
        variant_name: "honest_answer".to_string(),
        concurrency: 10,
        format: OutputFormat::Jsonl,
        inference_cache: CacheEnabledMode::WriteOnly,
    };

    let mut output = Vec::new();
    run_evaluation(args, evaluation_run_id, &mut output)
        .await
        .unwrap();
    clickhouse_flush_async_insert(&clickhouse).await;
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
        let clickhouse_inference = select_chat_inference_clickhouse(&clickhouse, inference_id)
            .await
            .unwrap();
        let parsed_response = match parsed.response.clone() {
            InferenceResponse::Chat(chat_response) => chat_response,
            InferenceResponse::Json(..) => panic!("Json response not supported"),
        };
        // Check the input to the inference parses as StoredInput
        let _clickhouse_input: StoredInput =
            serde_json::from_str(clickhouse_inference["input"].as_str().unwrap()).unwrap();
        // assert_eq!(&clickhouse_input, parsed.datapoint.input());
        let clickhouse_output: Vec<ContentBlockChatOutput> =
            serde_json::from_str(clickhouse_inference["output"].as_str().unwrap()).unwrap();
        // Check the output to the inference is the same as the output in the response
        assert_eq!(clickhouse_output, parsed_response.content);
        // Check the inference is properly tagged
        assert_eq!(
            clickhouse_inference["tags"]["tensorzero::evaluation_run_id"],
            evaluation_run_id.to_string()
        );
        assert_eq!(
            clickhouse_inference["tags"]["tensorzero::datapoint_id"],
            parsed.datapoint.id().to_string()
        );
        assert_eq!(
            clickhouse_inference["tags"]["tensorzero::evaluation_name"],
            "images"
        );

        // There should be no Float feedback for this evaluation
        assert!(select_feedback_by_target_id_clickhouse(
            &clickhouse,
            "FloatMetricFeedback",
            inference_id,
            None,
        )
        .await
        .is_none());
        // The exact match evaluation should fail
        assert!(!parsed.evaluations["exact_match"]
            .as_ref()
            .unwrap()
            .as_bool()
            .unwrap());

        // There should be Boolean feedback for honest answer
        let clickhouse_feedback = select_feedback_by_target_id_clickhouse(
            &clickhouse,
            "BooleanMetricFeedback",
            inference_id,
            Some("tensorzero::evaluation_name::images::evaluator_name::honest_answer"),
        )
        .await
        .unwrap();
        assert_eq!(
            clickhouse_feedback["metric_name"].as_str().unwrap(),
            "tensorzero::evaluation_name::images::evaluator_name::honest_answer"
        );
        assert_eq!(
            clickhouse_feedback["value"],
            parsed.evaluations["honest_answer"]
                .as_ref()
                .unwrap()
                .as_bool()
                .unwrap()
        );
        total_honest_answers += clickhouse_feedback["value"].as_bool().unwrap() as u32;
        assert_eq!(
            clickhouse_feedback["tags"]["tensorzero::evaluation_run_id"],
            evaluation_run_id.to_string()
        );
        assert_eq!(
            clickhouse_feedback["tags"]["tensorzero::datapoint_id"],
            parsed.datapoint.id().to_string()
        );
        assert_eq!(
            clickhouse_feedback["tags"]["tensorzero::evaluation_name"],
            "images"
        );
        assert_eq!(
            clickhouse_feedback["tags"]["tensorzero::evaluator_name"],
            "honest_answer"
        );
        let evaluator_inference_id = Uuid::parse_str(
            clickhouse_feedback["tags"]["tensorzero::evaluator_inference_id"]
                .as_str()
                .unwrap(),
        )
        .unwrap();
        let evaluator_inference =
            select_json_inference_clickhouse(&clickhouse, evaluator_inference_id)
                .await
                .unwrap();
        assert_eq!(
            evaluator_inference["tags"]["tensorzero::evaluation_name"],
            "images"
        );

        // There should be Boolean feedback for matches reference
        let clickhouse_feedback = select_feedback_by_target_id_clickhouse(
            &clickhouse,
            "BooleanMetricFeedback",
            inference_id,
            Some("tensorzero::evaluation_name::images::evaluator_name::matches_reference"),
        )
        .await
        .unwrap();
        assert_eq!(
            clickhouse_feedback["metric_name"].as_str().unwrap(),
            "tensorzero::evaluation_name::images::evaluator_name::matches_reference"
        );
        assert_eq!(
            clickhouse_feedback["value"],
            parsed.evaluations["matches_reference"]
                .as_ref()
                .unwrap()
                .as_bool()
                .unwrap()
        );
        total_matches_reference += clickhouse_feedback["value"].as_bool().unwrap() as u32;
        assert_eq!(
            clickhouse_feedback["tags"]["tensorzero::evaluation_run_id"],
            evaluation_run_id.to_string()
        );
        assert_eq!(
            clickhouse_feedback["tags"]["tensorzero::datapoint_id"],
            parsed.datapoint.id().to_string()
        );
        assert_eq!(
            clickhouse_feedback["tags"]["tensorzero::evaluation_name"],
            "images"
        );
        assert_eq!(
            clickhouse_feedback["tags"]["tensorzero::evaluator_name"],
            "matches_reference"
        );
        let evaluator_inference_id = Uuid::parse_str(
            clickhouse_feedback["tags"]["tensorzero::evaluator_inference_id"]
                .as_str()
                .unwrap(),
        )
        .unwrap();
        let evaluator_inference =
            select_json_inference_clickhouse(&clickhouse, evaluator_inference_id)
                .await
                .unwrap();
        assert_eq!(
            evaluator_inference["tags"]["tensorzero::evaluation_name"],
            "images"
        );
        let input = evaluator_inference["input"].as_str().unwrap();
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
    let clickhouse = get_clickhouse().await;
    write_chat_fixture_to_dataset(
        &PathBuf::from(&format!(
            "{}/../tensorzero-core/fixtures/datasets/chat_datapoint_fixture.jsonl",
            std::env::var("CARGO_MANIFEST_DIR").unwrap()
        )),
        &HashMap::from([("baz".to_string(), dataset_name.clone())]),
    )
    .await;
    let config_path = PathBuf::from(&format!(
        "{}/../tensorzero-core/tests/e2e/tensorzero.toml",
        std::env::var("CARGO_MANIFEST_DIR").unwrap()
    ));
    let evaluation_run_id = Uuid::now_v7();
    let args = Args {
        config_file: config_path,
        gateway_url: None,
        dataset_name: dataset_name.clone(),
        evaluation_name: "bad_images".to_string(),
        variant_name: "honest_answer".to_string(),
        concurrency: 10,
        format: OutputFormat::Jsonl,
        inference_cache: CacheEnabledMode::Off,
    };

    let mut output = Vec::new();
    run_evaluation(args, evaluation_run_id, &mut output)
        .await
        .unwrap();
    clickhouse_flush_async_insert(&clickhouse).await;
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
        assert_eq!(honest_answer_error, "Image content not supported for LLM judge evaluations with `serialized` input format. If you want image evaluations, try the `messages` input format.");
        let inference_id = parsed.response.inference_id();
        let clickhouse_inference = select_chat_inference_clickhouse(&clickhouse, inference_id)
            .await
            .unwrap();
        let parsed_response = match parsed.response.clone() {
            InferenceResponse::Chat(chat_response) => chat_response,
            InferenceResponse::Json(..) => panic!("Json response not supported"),
        };
        // Check the input to the inference parses as StoreInput
        let _clickhouse_input: StoredInput =
            serde_json::from_str(clickhouse_inference["input"].as_str().unwrap()).unwrap();
        // assert_eq!(&clickhouse_input, parsed.datapoint.input());
        let clickhouse_output: Vec<ContentBlockChatOutput> =
            serde_json::from_str(clickhouse_inference["output"].as_str().unwrap()).unwrap();
        // Check the output to the inference is the same as the output in the response
        assert_eq!(clickhouse_output, parsed_response.content);
        // Check the inference is properly tagged
        assert_eq!(
            clickhouse_inference["tags"]["tensorzero::evaluation_run_id"],
            evaluation_run_id.to_string()
        );
        assert_eq!(
            clickhouse_inference["tags"]["tensorzero::datapoint_id"],
            parsed.datapoint.id().to_string()
        );
        assert_eq!(
            clickhouse_inference["tags"]["tensorzero::evaluation_name"],
            "bad_images"
        );

        // There should be no Float feedback for this evaluation
        assert!(select_feedback_by_target_id_clickhouse(
            &clickhouse,
            "FloatMetricFeedback",
            inference_id,
            None,
        )
        .await
        .is_none());

        // There should be no Boolean feedback for honest answer since it should have failed
        assert!(select_feedback_by_target_id_clickhouse(
            &clickhouse,
            "BooleanMetricFeedback",
            inference_id,
            Some("tensorzero::evaluation_name::images::evaluator_name::honest_answer"),
        )
        .await
        .is_none());
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn run_llm_judge_evaluation_chat_pretty() {
    init_tracing_for_tests();
    let dataset_name = format!("good-haikus-no-output-{}", Uuid::now_v7());
    write_chat_fixture_to_dataset(
        &PathBuf::from(&format!(
            "{}/../tensorzero-core/fixtures/datasets/chat_datapoint_fixture.jsonl",
            std::env::var("CARGO_MANIFEST_DIR").unwrap()
        )),
        &HashMap::from([("good-haikus-no-output".to_string(), dataset_name.clone())]),
    )
    .await;
    let config_path = PathBuf::from(&format!(
        "{}/../tensorzero-core/tests/e2e/tensorzero.toml",
        std::env::var("CARGO_MANIFEST_DIR").unwrap()
    ));
    let evaluation_run_id = Uuid::now_v7();
    let args = Args {
        config_file: config_path,
        gateway_url: None,
        evaluation_name: "haiku_without_outputs".to_string(),
        dataset_name: dataset_name.clone(),
        variant_name: "gpt_4o_mini".to_string(),
        concurrency: 10,
        format: OutputFormat::Pretty,
        inference_cache: CacheEnabledMode::Off,
    };

    let mut output = Vec::new();
    // Let's make sure this threshold passes and the output is reasonable
    run_evaluation(args, evaluation_run_id, &mut output)
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
    let dataset_name = format!("extract_entities_0.8-{}", Uuid::now_v7());
    write_json_fixture_to_dataset(
        &PathBuf::from(&format!(
            "{}/../tensorzero-core/fixtures/datasets/json_datapoint_fixture.jsonl",
            std::env::var("CARGO_MANIFEST_DIR").unwrap()
        )),
        &HashMap::from([("extract_entities_0.8".to_string(), dataset_name.clone())]),
    )
    .await;
    let config_path = PathBuf::from(&format!(
        "{}/../tensorzero-core/tests/e2e/tensorzero.toml",
        std::env::var("CARGO_MANIFEST_DIR").unwrap()
    ));
    let evaluation_run_id = Uuid::now_v7();
    let args = Args {
        config_file: config_path,
        gateway_url: None,
        evaluation_name: "entity_extraction".to_string(),
        dataset_name: dataset_name.clone(),
        variant_name: "gpt_4o_mini".to_string(),
        concurrency: 10,
        format: OutputFormat::Pretty,
        inference_cache: CacheEnabledMode::Off,
    };

    let mut output = Vec::new();
    // Let's make sure this threshold fails and the output is reasonable
    let err = run_evaluation(args, evaluation_run_id, &mut output)
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
    assert!(args
        .to_string()
        .contains("the following required arguments were not provided:"));
    assert!(args
        .to_string()
        .contains("--evaluation-name <EVALUATION_NAME>"));
    assert!(args.to_string().contains("--dataset-name <DATASET_NAME>"));
    assert!(args.to_string().contains("--variant-name <VARIANT_NAME>"));

    // Test required arguments
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
    assert_eq!(args.evaluation_name, "my-evaluation");
    assert_eq!(args.variant_name, "my-variant");
    assert_eq!(args.dataset_name, "my-dataset");
    assert_eq!(args.config_file, PathBuf::from("./config/tensorzero.toml"));
    assert_eq!(args.concurrency, 1);
    assert_eq!(args.gateway_url, None);
    assert_eq!(args.format, OutputFormat::Pretty);
    assert_eq!(args.inference_cache, CacheEnabledMode::On);

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
    ])
    .unwrap();
    assert_eq!(args.evaluation_name, "my-evaluation");
    assert_eq!(args.dataset_name, "my-dataset");
    assert_eq!(args.variant_name, "my-variant");
    assert_eq!(args.config_file, PathBuf::from("/path/to/config.toml"));
    assert_eq!(
        args.gateway_url,
        Some(Url::parse("http://localhost:8080/").unwrap())
    );
    assert_eq!(args.concurrency, 10);
    assert_eq!(args.format, OutputFormat::Jsonl);
    assert_eq!(args.inference_cache, CacheEnabledMode::WriteOnly);
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
    assert!(args
        .to_string()
        .contains("invalid value 'not-a-url' for '--gateway-url <GATEWAY_URL>'"));

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
    assert!(args
        .to_string()
        .contains("invalid value 'invalid' for '--format <FORMAT>'"));
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
    let clickhouse = get_clickhouse().await;
    write_json_fixture_to_dataset(
        &PathBuf::from(&format!(
            "{}/../tensorzero-core/fixtures/datasets/json_datapoint_fixture.jsonl",
            std::env::var("CARGO_MANIFEST_DIR").unwrap()
        )),
        &HashMap::from([("extract_entities_0.8".to_string(), dataset_name.clone())]),
    )
    .await;
    let config_path = PathBuf::from(&format!(
        "{}/../tensorzero-core/tests/e2e/tensorzero.toml",
        std::env::var("CARGO_MANIFEST_DIR").unwrap()
    ));
    let evaluation_run_id = Uuid::now_v7();
    let args = Args {
        config_file: config_path,
        gateway_url: None,
        evaluation_name: "entity_extraction".to_string(),
        dataset_name: dataset_name.clone(),
        variant_name: "dummy_error".to_string(),
        concurrency: 10,
        format: OutputFormat::Jsonl,
        inference_cache: CacheEnabledMode::Off,
    };

    let mut output = Vec::new();
    run_evaluation(args, evaluation_run_id, &mut output)
        .await
        .unwrap();
    clickhouse_flush_async_insert(&clickhouse).await;
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
        assert!(error
            .message
            .contains("Error sending request to Dummy provider"));
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_run_llm_judge_evaluator_chat() {
    init_tracing_for_tests();
    let tensorzero_client = ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
        config_file: Some(PathBuf::from(&format!(
            "{}/../tensorzero-core/tests/e2e/tensorzero.toml",
            std::env::var("CARGO_MANIFEST_DIR").unwrap()
        ))),
        clickhouse_url: None,
        postgres_url: None,
        timeout: None,
        verify_credentials: true,
        allow_batch_writes: true,
    })
    .build()
    .await
    .unwrap();
    let tensorzero_client = ThrottledTensorZeroClient::new(tensorzero_client, Semaphore::new(1));
    let clients = Arc::new(Clients {
        tensorzero_client,
        clickhouse_client: get_clickhouse().await,
    });
    let inference_response = InferenceResponse::Chat(ChatInferenceResponse {
        content: vec![ContentBlockChatOutput::Text(Text {
            text: "Hello, world!".to_string(),
        })],
        original_response: None,
        finish_reason: None,
        episode_id: Uuid::now_v7(),
        inference_id: Uuid::now_v7(),
        usage: Usage {
            input_tokens: 0,
            output_tokens: 0,
        },
        variant_name: "test_variant".to_string(),
    });
    let datapoint = StoredDatapoint::Chat(StoredChatInferenceDatapoint {
        input: StoredInput {
            system: None,
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputMessageContent::Text(Text {
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
        tool_params: None,
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
    };
    let input = resolved_input_to_client_input(
        datapoint
            .input()
            .clone()
            .reresolve(&clients.tensorzero_client)
            .await
            .unwrap(),
    )
    .unwrap();
    let result = run_llm_judge_evaluator(RunLLMJudgeEvaluatorParams {
        inference_response: &inference_response,
        datapoint: &datapoint,
        clients: &clients,
        llm_judge_config: &llm_judge_config,
        evaluation_name: "test_evaluation",
        evaluator_name: "happy_bool",
        evaluation_run_id: Uuid::now_v7(),
        input: &input,
        inference_cache: CacheEnabledMode::Off,
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
        evaluation_name: "test_evaluation",
        evaluator_name: "sad_bool",
        evaluation_run_id: Uuid::now_v7(),
        input: &input,
        inference_cache: CacheEnabledMode::Off,
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
        evaluation_name: "test_evaluation",
        evaluator_name: "zero",
        evaluation_run_id: Uuid::now_v7(),
        input: &input,
        inference_cache: CacheEnabledMode::Off,
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
        evaluation_name: "test_evaluation",
        evaluator_name: "one",
        evaluation_run_id: Uuid::now_v7(),
        input: &input,
        inference_cache: CacheEnabledMode::Off,
    })
    .await
    .unwrap()
    .unwrap();
    assert_eq!(result.value, json!(1));

    // Try without output
    let datapoint = StoredDatapoint::Chat(StoredChatInferenceDatapoint {
        input: StoredInput {
            system: None,
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputMessageContent::Text(Text {
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
        tool_params: None,
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
        evaluation_name: "test_evaluation",
        evaluator_name: "happy_bool",
        evaluation_run_id: Uuid::now_v7(),
        input: &input,
        inference_cache: CacheEnabledMode::Off,
    })
    .await
    .unwrap();
    assert!(result.is_none());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_run_llm_judge_evaluator_json() {
    init_tracing_for_tests();
    let tensorzero_client = get_tensorzero_client().await;
    let tensorzero_client = ThrottledTensorZeroClient::new(tensorzero_client, Semaphore::new(1));
    let clients = Arc::new(Clients {
        tensorzero_client,
        clickhouse_client: get_clickhouse().await,
    });
    let inference_response = InferenceResponse::Json(JsonInferenceResponse {
        output: JsonInferenceOutput {
            parsed: Some(json!({"answer": "LeBron James"})),
            raw: Some("{\"answer\": \"LeBron James\"}".to_string()),
        },
        original_response: None,
        finish_reason: None,
        episode_id: Uuid::now_v7(),
        inference_id: Uuid::now_v7(),
        usage: Usage {
            input_tokens: 0,
            output_tokens: 0,
        },
        variant_name: "test_variant".to_string(),
    });
    let datapoint = StoredDatapoint::Json(JsonInferenceDatapoint {
        input: StoredInput {
            system: None,
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputMessageContent::Text(Text {
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
    };
    let input = resolved_input_to_client_input(
        datapoint
            .input()
            .clone()
            .reresolve(&clients.tensorzero_client)
            .await
            .unwrap(),
    )
    .unwrap();
    let result = run_llm_judge_evaluator(RunLLMJudgeEvaluatorParams {
        inference_response: &inference_response,
        datapoint: &datapoint,
        clients: &clients,
        llm_judge_config: &llm_judge_config,
        evaluation_name: "test_evaluation",
        evaluator_name: "happy_bool",
        evaluation_run_id: Uuid::now_v7(),
        input: &input,
        inference_cache: CacheEnabledMode::Off,
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
        evaluation_name: "test_evaluation",
        evaluator_name: "sad_bool",
        evaluation_run_id: Uuid::now_v7(),
        input: &input,
        inference_cache: CacheEnabledMode::Off,
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
        evaluation_name: "test_evaluation",
        evaluator_name: "zero",
        evaluation_run_id: Uuid::now_v7(),
        input: &input,
        inference_cache: CacheEnabledMode::Off,
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
        evaluation_name: "test_evaluation",
        evaluator_name: "one",
        evaluation_run_id: Uuid::now_v7(),
        input: &input,
        inference_cache: CacheEnabledMode::Off,
    })
    .await
    .unwrap()
    .unwrap();
    assert_eq!(result.value, json!(1));

    // Try without output
    let datapoint = StoredDatapoint::Chat(StoredChatInferenceDatapoint {
        input: StoredInput {
            system: None,
            messages: vec![StoredInputMessage {
                role: Role::User,
                content: vec![StoredInputMessageContent::Text(Text {
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
        tool_params: None,
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
        evaluation_name: "test_evaluation",
        evaluator_name: "happy_bool",
        evaluation_run_id: Uuid::now_v7(),
        input: &input,
        inference_cache: CacheEnabledMode::Off,
    })
    .await
    .unwrap();
    assert!(result.is_none());
}

#[tokio::test(flavor = "multi_thread")]
async fn run_evaluations_best_of_3() {
    init_tracing_for_tests();
    let dataset_name = format!("extract_entities_0.8-{}", Uuid::now_v7());
    let clickhouse = get_clickhouse().await;
    write_json_fixture_to_dataset(
        &PathBuf::from(&format!(
            "{}/../tensorzero-core/fixtures/datasets/json_datapoint_fixture.jsonl",
            std::env::var("CARGO_MANIFEST_DIR").unwrap()
        )),
        &HashMap::from([("extract_entities_0.8".to_string(), dataset_name.clone())]),
    )
    .await;
    let config_path = PathBuf::from(&format!(
        "{}/../tensorzero-core/tests/e2e/tensorzero.toml",
        std::env::var("CARGO_MANIFEST_DIR").unwrap()
    ));
    let evaluation_run_id = Uuid::now_v7();
    let args = Args {
        config_file: config_path,
        gateway_url: None,
        evaluation_name: "best_of_3".to_string(),
        dataset_name: dataset_name.clone(),
        variant_name: "gpt_4o_mini".to_string(),
        concurrency: 10,
        format: OutputFormat::Jsonl,
        inference_cache: CacheEnabledMode::Off,
    };

    let mut output = Vec::new();
    run_evaluation(args, evaluation_run_id, &mut output)
        .await
        .unwrap();
    clickhouse_flush_async_insert(&clickhouse).await;
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
        let clickhouse_inference = select_json_inference_clickhouse(&clickhouse, inference_id)
            .await
            .unwrap();
        let parsed_response = match parsed.response.clone() {
            InferenceResponse::Json(json_response) => json_response,
            InferenceResponse::Chat(..) => panic!("Chat response not supported"),
        };
        let clickhouse_input: StoredInput =
            serde_json::from_str(clickhouse_inference["input"].as_str().unwrap()).unwrap();
        // Check the input to the inference is the same as the input to the datapoint
        assert_eq!(&clickhouse_input, parsed.datapoint.input());
        let clickhouse_output: JsonInferenceOutput =
            serde_json::from_str(clickhouse_inference["output"].as_str().unwrap()).unwrap();
        // Check the output to the inference is the same as the output in the response
        assert_eq!(clickhouse_output, parsed_response.output);
        // Check the inference is properly tagged
        assert_eq!(
            clickhouse_inference["tags"]["tensorzero::evaluation_run_id"],
            evaluation_run_id.to_string()
        );
        assert_eq!(
            clickhouse_inference["tags"]["tensorzero::datapoint_id"],
            parsed.datapoint.id().to_string()
        );
        assert_eq!(
            clickhouse_inference["tags"]["tensorzero::evaluation_name"],
            "best_of_3"
        );
        assert_eq!(
            clickhouse_inference["tags"]["tensorzero::dataset_name"],
            dataset_name
        );
        // Check boolean feedback was recorded
        let feedback = select_feedback_by_target_id_clickhouse(
            &clickhouse,
            "BooleanMetricFeedback",
            inference_id,
            None,
        )
        .await
        .unwrap();

        assert_eq!(
            feedback["metric_name"].as_str().unwrap(),
            "tensorzero::evaluation_name::best_of_3::evaluator_name::llm_judge_bool"
        );
        assert_eq!(
            feedback["value"],
            parsed.evaluations["llm_judge_bool"]
                .as_ref()
                .unwrap()
                .as_bool()
                .unwrap()
        );
        assert_eq!(
            feedback["tags"]["tensorzero::evaluation_run_id"],
            evaluation_run_id.to_string()
        );
        assert_eq!(
            feedback["tags"]["tensorzero::datapoint_id"],
            parsed.datapoint.id().to_string()
        );

        // Check bool feedback was recorded
        let feedback = select_feedback_by_target_id_clickhouse(
            &clickhouse,
            "BooleanMetricFeedback",
            inference_id,
            None,
        )
        .await
        .unwrap();

        assert_eq!(
            feedback["metric_name"].as_str().unwrap(),
            "tensorzero::evaluation_name::best_of_3::evaluator_name::llm_judge_bool"
        );
        assert_eq!(
            feedback["value"],
            parsed.evaluations["llm_judge_bool"]
                .as_ref()
                .unwrap()
                .as_bool()
                .unwrap()
        );
        assert_eq!(
            feedback["tags"]["tensorzero::evaluation_run_id"],
            evaluation_run_id.to_string()
        );
        assert_eq!(
            feedback["tags"]["tensorzero::datapoint_id"],
            parsed.datapoint.id().to_string()
        );
        assert_eq!(
            feedback["tags"]["tensorzero::evaluator_name"],
            "llm_judge_bool"
        );
        assert_eq!(feedback["tags"]["tensorzero::evaluation_name"], "best_of_3");
        let evaluator_inference_id = Uuid::parse_str(
            feedback["tags"]["tensorzero::evaluator_inference_id"]
                .as_str()
                .unwrap(),
        )
        .unwrap();
        let evaluator_inference =
            select_json_inference_clickhouse(&clickhouse, evaluator_inference_id)
                .await
                .unwrap();
        assert_eq!(
            evaluator_inference["tags"]["tensorzero::evaluation_name"],
            "best_of_3"
        );
        parsed_output.push(parsed);

        // Select Model inferences for the evaluator inference id
        let model_inferences =
            select_model_inferences_clickhouse(&clickhouse, evaluator_inference_id)
                .await
                .unwrap();
        let mut happy_count = 0;
        for model_inference in &model_inferences {
            if model_inference["system"].as_str().unwrap().trim()
                == "Return true if you are happy today!"
            {
                happy_count += 1;
            } else {
                assert!(model_inference["system"]
                    .as_str()
                    .unwrap()
                    .starts_with("You are an assistant tasked with re-ranking"));
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
    let clickhouse = get_clickhouse().await;
    let dataset_name = format!("extract_entities_0.8-{}", Uuid::now_v7());
    write_json_fixture_to_dataset(
        &PathBuf::from(&format!(
            "{}/../tensorzero-core/fixtures/datasets/json_datapoint_fixture.jsonl",
            std::env::var("CARGO_MANIFEST_DIR").unwrap()
        )),
        &HashMap::from([("extract_entities_0.8".to_string(), dataset_name.clone())]),
    )
    .await;
    let config_path = PathBuf::from(&format!(
        "{}/../tensorzero-core/tests/e2e/tensorzero.toml",
        std::env::var("CARGO_MANIFEST_DIR").unwrap()
    ));
    let evaluation_run_id = Uuid::now_v7();
    let args = Args {
        config_file: config_path,
        gateway_url: None,
        evaluation_name: "mixture_of_3".to_string(),
        dataset_name: dataset_name.clone(),
        variant_name: "gpt_4o_mini".to_string(),
        concurrency: 10,
        format: OutputFormat::Jsonl,
        inference_cache: CacheEnabledMode::Off,
    };

    let mut output = Vec::new();
    run_evaluation(args, evaluation_run_id, &mut output)
        .await
        .unwrap();
    clickhouse_flush_async_insert(&clickhouse).await;
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
        let clickhouse_inference = select_json_inference_clickhouse(&clickhouse, inference_id)
            .await
            .unwrap();
        let parsed_response = match parsed.response.clone() {
            InferenceResponse::Json(json_response) => json_response,
            InferenceResponse::Chat(..) => panic!("Chat response not supported"),
        };
        let clickhouse_input: StoredInput =
            serde_json::from_str(clickhouse_inference["input"].as_str().unwrap()).unwrap();
        // Check the input to the inference is the same as the input to the datapoint
        assert_eq!(&clickhouse_input, parsed.datapoint.input());
        let clickhouse_output: JsonInferenceOutput =
            serde_json::from_str(clickhouse_inference["output"].as_str().unwrap()).unwrap();
        // Check the output to the inference is the same as the output in the response
        assert_eq!(clickhouse_output, parsed_response.output);
        // Check the inference is properly tagged
        assert_eq!(
            clickhouse_inference["tags"]["tensorzero::evaluation_run_id"],
            evaluation_run_id.to_string()
        );
        assert_eq!(
            clickhouse_inference["tags"]["tensorzero::datapoint_id"],
            parsed.datapoint.id().to_string()
        );
        assert_eq!(
            clickhouse_inference["tags"]["tensorzero::evaluation_name"],
            "mixture_of_3"
        );
        assert_eq!(
            clickhouse_inference["tags"]["tensorzero::dataset_name"],
            dataset_name
        );
        // Check boolean feedback was recorded
        let feedback = select_feedback_by_target_id_clickhouse(
            &clickhouse,
            "BooleanMetricFeedback",
            inference_id,
            None,
        )
        .await
        .unwrap();

        assert_eq!(
            feedback["metric_name"].as_str().unwrap(),
            "tensorzero::evaluation_name::mixture_of_3::evaluator_name::llm_judge_bool"
        );
        assert_eq!(
            feedback["value"],
            parsed.evaluations["llm_judge_bool"]
                .as_ref()
                .unwrap()
                .as_bool()
                .unwrap()
        );
        assert_eq!(
            feedback["tags"]["tensorzero::evaluation_run_id"],
            evaluation_run_id.to_string()
        );
        assert_eq!(
            feedback["tags"]["tensorzero::datapoint_id"],
            parsed.datapoint.id().to_string()
        );

        // Check bool feedback was recorded
        let feedback = select_feedback_by_target_id_clickhouse(
            &clickhouse,
            "BooleanMetricFeedback",
            inference_id,
            None,
        )
        .await
        .unwrap();

        assert_eq!(
            feedback["metric_name"].as_str().unwrap(),
            "tensorzero::evaluation_name::mixture_of_3::evaluator_name::llm_judge_bool"
        );
        assert_eq!(
            feedback["value"],
            parsed.evaluations["llm_judge_bool"]
                .as_ref()
                .unwrap()
                .as_bool()
                .unwrap()
        );
        assert_eq!(
            feedback["tags"]["tensorzero::evaluation_run_id"],
            evaluation_run_id.to_string()
        );
        assert_eq!(
            feedback["tags"]["tensorzero::datapoint_id"],
            parsed.datapoint.id().to_string()
        );
        assert_eq!(
            feedback["tags"]["tensorzero::evaluator_name"],
            "llm_judge_bool"
        );
        assert_eq!(
            feedback["tags"]["tensorzero::evaluation_name"],
            "mixture_of_3"
        );
        let evaluator_inference_id = Uuid::parse_str(
            feedback["tags"]["tensorzero::evaluator_inference_id"]
                .as_str()
                .unwrap(),
        )
        .unwrap();
        let evaluator_inference =
            select_json_inference_clickhouse(&clickhouse, evaluator_inference_id)
                .await
                .unwrap();
        assert_eq!(
            evaluator_inference["tags"]["tensorzero::evaluation_name"],
            "mixture_of_3"
        );
        parsed_output.push(parsed);

        // Select Model inferences for the evaluator inference id
        let model_inferences =
            select_model_inferences_clickhouse(&clickhouse, evaluator_inference_id)
                .await
                .unwrap();
        let mut happy_count = 0;
        for model_inference in &model_inferences {
            if model_inference["system"].as_str().unwrap().trim()
                == "Return true if you are happy today!"
            {
                happy_count += 1;
            } else {
                assert!(model_inference["system"]
                    .as_str()
                    .unwrap()
                    .starts_with("You have been provided with a set of responses from various"));
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
    let clickhouse = get_clickhouse().await;
    let dataset_name = format!("extract_entities_0.8-{}", Uuid::now_v7());
    write_json_fixture_to_dataset(
        &PathBuf::from(&format!(
            "{}/../tensorzero-core/fixtures/datasets/json_datapoint_fixture.jsonl",
            std::env::var("CARGO_MANIFEST_DIR").unwrap()
        )),
        &HashMap::from([("extract_entities_0.8".to_string(), dataset_name.clone())]),
    )
    .await;
    let config_path = PathBuf::from(&format!(
        "{}/../tensorzero-core/tests/e2e/tensorzero.toml",
        std::env::var("CARGO_MANIFEST_DIR").unwrap()
    ));
    let evaluation_run_id = Uuid::now_v7();
    let args = Args {
        config_file: config_path,
        gateway_url: None,
        evaluation_name: "dicl".to_string(),
        dataset_name: dataset_name.clone(),
        variant_name: "gpt_4o_mini".to_string(),
        concurrency: 10,
        format: OutputFormat::Jsonl,
        inference_cache: CacheEnabledMode::Off,
    };

    let mut output = Vec::new();
    run_evaluation(args, evaluation_run_id, &mut output)
        .await
        .unwrap();
    clickhouse_flush_async_insert(&clickhouse).await;
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
        let clickhouse_inference = select_json_inference_clickhouse(&clickhouse, inference_id)
            .await
            .unwrap();
        let parsed_response = match parsed.response.clone() {
            InferenceResponse::Json(json_response) => json_response,
            InferenceResponse::Chat(..) => panic!("Chat response not supported"),
        };
        let clickhouse_input: StoredInput =
            serde_json::from_str(clickhouse_inference["input"].as_str().unwrap()).unwrap();
        // Check the input to the inference is the same as the input to the datapoint
        assert_eq!(&clickhouse_input, parsed.datapoint.input());
        let clickhouse_output: JsonInferenceOutput =
            serde_json::from_str(clickhouse_inference["output"].as_str().unwrap()).unwrap();
        // Check the output to the inference is the same as the output in the response
        assert_eq!(clickhouse_output, parsed_response.output);
        // Check the inference is properly tagged
        assert_eq!(
            clickhouse_inference["tags"]["tensorzero::evaluation_run_id"],
            evaluation_run_id.to_string()
        );
        assert_eq!(
            clickhouse_inference["tags"]["tensorzero::datapoint_id"],
            parsed.datapoint.id().to_string()
        );
        assert_eq!(
            clickhouse_inference["tags"]["tensorzero::evaluation_name"],
            "dicl"
        );
        assert_eq!(
            clickhouse_inference["tags"]["tensorzero::dataset_name"],
            dataset_name
        );
        // Check boolean feedback was recorded
        let feedback = select_feedback_by_target_id_clickhouse(
            &clickhouse,
            "BooleanMetricFeedback",
            inference_id,
            None,
        )
        .await
        .unwrap();

        assert_eq!(
            feedback["metric_name"].as_str().unwrap(),
            "tensorzero::evaluation_name::dicl::evaluator_name::llm_judge_bool"
        );
        assert_eq!(
            feedback["value"],
            parsed.evaluations["llm_judge_bool"]
                .as_ref()
                .unwrap()
                .as_bool()
                .unwrap()
        );
        assert_eq!(
            feedback["tags"]["tensorzero::evaluation_run_id"],
            evaluation_run_id.to_string()
        );
        assert_eq!(
            feedback["tags"]["tensorzero::datapoint_id"],
            parsed.datapoint.id().to_string()
        );

        // Check bool feedback was recorded
        let feedback = select_feedback_by_target_id_clickhouse(
            &clickhouse,
            "BooleanMetricFeedback",
            inference_id,
            None,
        )
        .await
        .unwrap();

        assert_eq!(
            feedback["metric_name"].as_str().unwrap(),
            "tensorzero::evaluation_name::dicl::evaluator_name::llm_judge_bool"
        );
        assert_eq!(
            feedback["value"],
            parsed.evaluations["llm_judge_bool"]
                .as_ref()
                .unwrap()
                .as_bool()
                .unwrap()
        );
        assert_eq!(
            feedback["tags"]["tensorzero::evaluation_run_id"],
            evaluation_run_id.to_string()
        );
        assert_eq!(
            feedback["tags"]["tensorzero::datapoint_id"],
            parsed.datapoint.id().to_string()
        );
        assert_eq!(
            feedback["tags"]["tensorzero::evaluator_name"],
            "llm_judge_bool"
        );
        assert_eq!(feedback["tags"]["tensorzero::evaluation_name"], "dicl");
        let evaluator_inference_id = Uuid::parse_str(
            feedback["tags"]["tensorzero::evaluator_inference_id"]
                .as_str()
                .unwrap(),
        )
        .unwrap();
        let evaluator_inference =
            select_json_inference_clickhouse(&clickhouse, evaluator_inference_id)
                .await
                .unwrap();
        assert_eq!(
            evaluator_inference["tags"]["tensorzero::evaluation_name"],
            "dicl"
        );
        parsed_output.push(parsed);

        // Select Model inferences for the evaluator inference id
        let model_inferences =
            select_model_inferences_clickhouse(&clickhouse, evaluator_inference_id)
                .await
                .unwrap();
        for model_inference in &model_inferences {
            match model_inference["model_name"].as_str().unwrap() {
                "openai::gpt-4o-mini" => {
                    let system = model_inference["system"].as_str().unwrap();
                    assert!(system.starts_with("You are tasked with learning by induction"));
                }
                "text-embedding-3-small" => {
                    assert!(model_inference["system"].is_null());
                    let messages = model_inference["input_messages"].as_str().unwrap();
                    // messages should be a json array of objects
                    assert!(messages.starts_with("[{"));
                }
                _ => {
                    panic!(
                        "Unknown model name: {}",
                        model_inference["model_name"].as_str().unwrap()
                    );
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
    let clickhouse = get_clickhouse().await;
    write_json_fixture_to_dataset(
        &PathBuf::from(&format!(
            "{}/../tensorzero-core/fixtures/datasets/json_datapoint_fixture.jsonl",
            std::env::var("CARGO_MANIFEST_DIR").unwrap()
        )),
        &HashMap::from([("exact_matches_empty".to_string(), dataset_name.clone())]),
    )
    .await;

    let dataset = query_dataset(
        &clickhouse,
        &dataset_name,
        "extract_entities",
        &FunctionConfig::Json(FunctionConfigJson::default()),
    )
    .await
    .unwrap();
    // This ID should not be returned
    let staled_id = Uuid::parse_str("01957bbb-44a8-7490-bfe7-32f8ed2fc797").unwrap();
    let staled_datapoint = dataset.iter().find(|dp| dp.id() == staled_id);
    assert!(staled_datapoint.is_none());
    assert_eq!(dataset.len(), 21);
}
