#![cfg_attr(
    test,
    allow(clippy::expect_used, clippy::unwrap_used, clippy::print_stdout)
)]
mod common;
use clap::Parser;
use std::path::PathBuf;
use url::Url;

use crate::common::write_json_fixture_to_dataset;
use common::write_chat_fixture_to_dataset;
use evals::{run_eval, stats::EvalInfo, Args, OutputFormat};
use tensorzero::InferenceResponse;
use tensorzero_internal::{
    clickhouse::test_helpers::{
        clickhouse_flush_async_insert, get_clickhouse, select_chat_inference_clickhouse,
        select_feedback_by_target_id_clickhouse, select_json_inference_clickhouse,
    },
    inference::types::{ContentBlockChatOutput, JsonInferenceOutput, ResolvedInput},
};
use uuid::Uuid;

#[tokio::test(flavor = "multi_thread")]
async fn run_evals_json() {
    let clickhouse = get_clickhouse().await;
    write_json_fixture_to_dataset(&PathBuf::from(&format!(
        "{}/../tensorzero-internal/fixtures/datasets/json_datapoint_fixture.jsonl",
        std::env::var("CARGO_MANIFEST_DIR").unwrap()
    )))
    .await;
    let config_path = PathBuf::from(&format!(
        "{}/../tensorzero-internal/tests/e2e/tensorzero.toml",
        std::env::var("CARGO_MANIFEST_DIR").unwrap()
    ));
    let eval_run_id = Uuid::now_v7();
    let args = Args {
        config_file: config_path,
        gateway_url: None,
        name: "entity_extraction".to_string(),
        variant: "gpt_4o_mini".to_string(),
        concurrency: 10,
        format: OutputFormat::Jsonl,
    };

    let mut output = Vec::new();
    run_eval(args, eval_run_id, &mut output).await.unwrap();
    clickhouse_flush_async_insert(&clickhouse).await;
    let output_str = String::from_utf8(output).unwrap();
    let mut parsed_output = Vec::new();
    let mut total_the = 0;
    for line in output_str.lines() {
        let parsed: EvalInfo = serde_json::from_str(line).expect("Each line should be valid JSON");
        assert!(parsed.evaluator_errors.is_empty());
        let inference_id = parsed.response.inference_id();
        let clickhouse_inference = select_json_inference_clickhouse(&clickhouse, inference_id)
            .await
            .unwrap();
        let parsed_response = match parsed.response.clone() {
            InferenceResponse::Json(json_response) => json_response,
            InferenceResponse::Chat(..) => panic!("Chat response not supported"),
        };
        let clickhouse_input: ResolvedInput =
            serde_json::from_str(clickhouse_inference["input"].as_str().unwrap()).unwrap();
        // Check the input to the inference is the same as the input to the datapoint
        assert_eq!(&clickhouse_input, parsed.datapoint.input());
        let clickhouse_output: JsonInferenceOutput =
            serde_json::from_str(clickhouse_inference["output"].as_str().unwrap()).unwrap();
        // Check the output to the inference is the same as the output in the response
        assert_eq!(clickhouse_output, parsed_response.output);
        // Check the inference is properly tagged
        assert_eq!(
            clickhouse_inference["tags"]["tensorzero::eval_run_id"],
            eval_run_id.to_string()
        );

        // Check boolean feedback was recorded
        let feedback = select_feedback_by_target_id_clickhouse(
            &clickhouse,
            "BooleanMetricFeedback",
            inference_id,
        )
        .await
        .unwrap();

        assert_eq!(
            feedback["metric_name"].as_str().unwrap(),
            "tensorzero::eval_name::entity_extraction::evaluator_name::exact_match"
        );
        assert_eq!(
            feedback["value"],
            parsed.evaluations["exact_match"]
                .as_ref()
                .unwrap()
                .as_bool()
                .unwrap()
        );

        // Check float feedback was recorded
        let feedback = select_feedback_by_target_id_clickhouse(
            &clickhouse,
            "FloatMetricFeedback",
            inference_id,
        )
        .await
        .unwrap();

        assert_eq!(
            feedback["metric_name"].as_str().unwrap(),
            "tensorzero::eval_name::entity_extraction::evaluator_name::count_sports"
        );
        assert_eq!(
            feedback["value"],
            parsed.evaluations["count_sports"]
                .as_ref()
                .unwrap()
                .as_f64()
                .unwrap()
        );
        total_the += feedback["value"].as_f64().unwrap() as u32;
        parsed_output.push(parsed);
    }
    assert_eq!(parsed_output.len(), 6);
    assert_eq!(total_the, 3);
}

#[tokio::test(flavor = "multi_thread")]
async fn run_exact_match_eval_chat() {
    let clickhouse = get_clickhouse().await;
    write_chat_fixture_to_dataset(&PathBuf::from(&format!(
        "{}/../tensorzero-internal/fixtures/datasets/chat_datapoint_fixture.jsonl",
        std::env::var("CARGO_MANIFEST_DIR").unwrap()
    )))
    .await;
    let config_path = PathBuf::from(&format!(
        "{}/../tensorzero-internal/tests/e2e/tensorzero.toml",
        std::env::var("CARGO_MANIFEST_DIR").unwrap()
    ));
    let eval_run_id = Uuid::now_v7();
    let args = Args {
        config_file: config_path,
        gateway_url: None,
        name: "haiku_with_outputs".to_string(),
        variant: "gpt_4o_mini".to_string(),
        concurrency: 10,
        format: OutputFormat::Jsonl,
    };

    let mut output = Vec::new();
    run_eval(args, eval_run_id, &mut output).await.unwrap();
    clickhouse_flush_async_insert(&clickhouse).await;
    let output_str = String::from_utf8(output).unwrap();
    let mut parsed_output = Vec::new();
    for line in output_str.lines() {
        let parsed: EvalInfo = serde_json::from_str(line).expect("Each line should be valid JSON");
        assert!(parsed.evaluator_errors.is_empty());
        let inference_id = parsed.response.inference_id();
        let clickhouse_inference = select_chat_inference_clickhouse(&clickhouse, inference_id)
            .await
            .unwrap();
        let parsed_response = match parsed.response.clone() {
            InferenceResponse::Chat(chat_response) => chat_response,
            InferenceResponse::Json(..) => panic!("Json response not supported"),
        };
        let clickhouse_input: ResolvedInput =
            serde_json::from_str(clickhouse_inference["input"].as_str().unwrap()).unwrap();
        // Check the input to the inference is the same as the input to the datapoint
        assert_eq!(&clickhouse_input, parsed.datapoint.input());
        let clickhouse_output: Vec<ContentBlockChatOutput> =
            serde_json::from_str(clickhouse_inference["output"].as_str().unwrap()).unwrap();
        // Check the output to the inference is the same as the output in the response
        assert_eq!(clickhouse_output, parsed_response.content);
        // Check the inference is properly tagged
        assert_eq!(
            clickhouse_inference["tags"]["tensorzero::eval_run_id"],
            eval_run_id.to_string()
        );

        let clickhouse_feedback = select_feedback_by_target_id_clickhouse(
            &clickhouse,
            "BooleanMetricFeedback",
            inference_id,
        )
        .await
        .unwrap();
        assert_eq!(
            clickhouse_feedback["metric_name"].as_str().unwrap(),
            "tensorzero::eval_name::haiku_with_outputs::evaluator_name::exact_match"
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
            clickhouse_feedback["tags"]["tensorzero::eval_run_id"],
            eval_run_id.to_string()
        );
        parsed_output.push(parsed);
    }
    assert_eq!(parsed_output.len(), 29);
}

#[tokio::test(flavor = "multi_thread")]
async fn run_llm_judge_eval_chat() {
    let clickhouse = get_clickhouse().await;
    write_chat_fixture_to_dataset(&PathBuf::from(&format!(
        "{}/../tensorzero-internal/fixtures/datasets/chat_datapoint_fixture.jsonl",
        std::env::var("CARGO_MANIFEST_DIR").unwrap()
    )))
    .await;
    let config_path = PathBuf::from(&format!(
        "{}/../tensorzero-internal/tests/e2e/tensorzero.toml",
        std::env::var("CARGO_MANIFEST_DIR").unwrap()
    ));
    let eval_run_id = Uuid::now_v7();
    let args = Args {
        config_file: config_path,
        gateway_url: None,
        name: "haiku_without_outputs".to_string(),
        variant: "gpt_4o_mini".to_string(),
        concurrency: 10,
        format: OutputFormat::Jsonl,
    };

    let mut output = Vec::new();
    run_eval(args, eval_run_id, &mut output).await.unwrap();
    clickhouse_flush_async_insert(&clickhouse).await;
    let output_str = String::from_utf8(output).unwrap();
    let mut parsed_output = Vec::new();
    let mut total_topic_fs = 0;
    for line in output_str.lines() {
        let parsed: EvalInfo = serde_json::from_str(line).expect("Each line should be valid JSON");
        assert!(parsed.evaluator_errors.is_empty());
        let inference_id = parsed.response.inference_id();
        let clickhouse_inference = select_chat_inference_clickhouse(&clickhouse, inference_id)
            .await
            .unwrap();
        let parsed_response = match parsed.response.clone() {
            InferenceResponse::Chat(chat_response) => chat_response,
            InferenceResponse::Json(..) => panic!("Json response not supported"),
        };
        let clickhouse_input: ResolvedInput =
            serde_json::from_str(clickhouse_inference["input"].as_str().unwrap()).unwrap();
        // Check the input to the inference is the same as the input to the datapoint
        assert_eq!(&clickhouse_input, parsed.datapoint.input());
        let clickhouse_output: Vec<ContentBlockChatOutput> =
            serde_json::from_str(clickhouse_inference["output"].as_str().unwrap()).unwrap();
        // Check the output to the inference is the same as the output in the response
        assert_eq!(clickhouse_output, parsed_response.content);
        // Check the inference is properly tagged
        assert_eq!(
            clickhouse_inference["tags"]["tensorzero::eval_run_id"],
            eval_run_id.to_string()
        );

        // There should be no Float feedback for this eval
        assert!(select_feedback_by_target_id_clickhouse(
            &clickhouse,
            "FloatMetricFeedback",
            inference_id,
        )
        .await
        .is_none());
        // The exact match eval should have value None since there is no output in any of these
        assert!(parsed.evaluations["exact_match"].is_none());

        // There should be Boolean feedback for this eval
        let clickhouse_feedback = select_feedback_by_target_id_clickhouse(
            &clickhouse,
            "BooleanMetricFeedback",
            inference_id,
        )
        .await
        .unwrap();
        assert_eq!(
            clickhouse_feedback["metric_name"].as_str().unwrap(),
            "tensorzero::eval_name::haiku_without_outputs::evaluator_name::topic_starts_with_f"
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
            clickhouse_feedback["tags"]["tensorzero::eval_run_id"],
            eval_run_id.to_string()
        );
        parsed_output.push(parsed);
    }
    assert_eq!(parsed_output.len(), 10);
    assert_eq!(total_topic_fs, 3);
}

#[tokio::test(flavor = "multi_thread")]
async fn run_llm_judge_eval_chat_human_readable() {
    write_chat_fixture_to_dataset(&PathBuf::from(&format!(
        "{}/../tensorzero-internal/fixtures/datasets/chat_datapoint_fixture.jsonl",
        std::env::var("CARGO_MANIFEST_DIR").unwrap()
    )))
    .await;
    let config_path = PathBuf::from(&format!(
        "{}/../tensorzero-internal/tests/e2e/tensorzero.toml",
        std::env::var("CARGO_MANIFEST_DIR").unwrap()
    ));
    let eval_run_id = Uuid::now_v7();
    let args = Args {
        config_file: config_path,
        gateway_url: None,
        name: "haiku_without_outputs".to_string(),
        variant: "gpt_4o_mini".to_string(),
        concurrency: 10,
        format: OutputFormat::HumanReadable,
    };

    let mut output = Vec::new();
    // Let's make sure this threshold passes and the output is reasonable
    run_eval(args, eval_run_id, &mut output).await.unwrap();
    let output_str = String::from_utf8(output).unwrap();
    assert!(output_str.contains("count_prepositions: 2.13 ± 0.15"));
    assert!(output_str.contains("exact_match: 0.00 ± 0.00"));
}

#[tokio::test(flavor = "multi_thread")]
async fn run_llm_judge_eval_json_human_readable() {
    write_json_fixture_to_dataset(&PathBuf::from(&format!(
        "{}/../tensorzero-internal/fixtures/datasets/json_datapoint_fixture.jsonl",
        std::env::var("CARGO_MANIFEST_DIR").unwrap()
    )))
    .await;
    let config_path = PathBuf::from(&format!(
        "{}/../tensorzero-internal/tests/e2e/tensorzero.toml",
        std::env::var("CARGO_MANIFEST_DIR").unwrap()
    ));
    let eval_run_id = Uuid::now_v7();
    let args = Args {
        config_file: config_path,
        gateway_url: None,
        name: "entity_extraction".to_string(),
        variant: "gpt_4o_mini".to_string(),
        concurrency: 10,
        format: OutputFormat::HumanReadable,
    };

    let mut output = Vec::new();
    // Let's make sure this threshold fails and the output is reasonable
    let err = run_eval(args, eval_run_id, &mut output).await.unwrap_err();
    let output_str = String::from_utf8(output).unwrap();
    assert!(output_str.contains("count_misc: 1.20 ± 0.24"));
    assert!(output_str.contains("exact_match: 0.50 ± 0.07"));
    let err = err.to_string();
    assert!(err.contains("Failed cutoffs for evaluators:"));
    assert!(err.contains("exact_match (cutoff: 0.6, got: 0.5)"));
}

#[tokio::test]
async fn test_parse_args() {
    // Test default values
    let args = Args::try_parse_from(["test"]).unwrap_err();
    assert!(args
        .to_string()
        .contains("the following required arguments were not provided:"));
    assert!(args.to_string().contains("--name <NAME>"));
    assert!(args.to_string().contains("--variant <VARIANT>"));

    // Test required arguments
    let args =
        Args::try_parse_from(["test", "--name", "my-eval", "--variant", "my-variant"]).unwrap();
    assert_eq!(args.name, "my-eval");
    assert_eq!(args.variant, "my-variant");
    assert_eq!(args.config_file, PathBuf::from("./config/tensorzero.toml"));
    assert_eq!(args.concurrency, 1);
    assert_eq!(args.gateway_url, None);
    assert_eq!(args.format, OutputFormat::HumanReadable);

    // Test all arguments
    let args = Args::try_parse_from([
        "test",
        "--name",
        "my-eval",
        "--variant",
        "my-variant",
        "--config-file",
        "/path/to/config.toml",
        "--gateway-url",
        "http://localhost:8080",
        "--concurrency",
        "10",
        "--format",
        "jsonl",
    ])
    .unwrap();
    assert_eq!(args.name, "my-eval");
    assert_eq!(args.variant, "my-variant");
    assert_eq!(args.config_file, PathBuf::from("/path/to/config.toml"));
    assert_eq!(
        args.gateway_url,
        Some(Url::parse("http://localhost:8080").unwrap())
    );
    assert_eq!(args.concurrency, 10);
    assert_eq!(args.format, OutputFormat::Jsonl);

    // Test invalid URL
    let args = Args::try_parse_from([
        "test",
        "--name",
        "my-eval",
        "--variant",
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
        "--name",
        "my-eval",
        "--variant",
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
async fn test_run_eval_binary() {
    let bin_path = env!("CARGO_BIN_EXE_evals");
    let output = std::process::Command::new(bin_path)
        .output()
        .expect("Failed to execute evals binary");
    let output_str = String::from_utf8(output.stdout).unwrap();
    assert!(output_str.is_empty());
    let stderr_str = String::from_utf8(output.stderr).unwrap();
    assert!(stderr_str.contains("the following required arguments were not provided:"));
}
