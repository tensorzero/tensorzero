#![cfg_attr(
    test,
    allow(clippy::expect_used, clippy::unwrap_used, clippy::print_stdout)
)]
mod common;
use std::path::PathBuf;

use crate::common::write_json_fixture_to_dataset;
use common::write_chat_fixture_to_dataset;
use evals::{run_eval, Args, OutputFormat};
use uuid::Uuid;

#[tokio::test(flavor = "multi_thread")]
async fn run_exact_match_eval_json() {
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
    let output_str = String::from_utf8(output).unwrap();
    let mut parsed_output = Vec::new();
    for line in output_str.lines() {
        let _parsed: serde_json::Value =
            serde_json::from_str(line).expect("Each line should be valid JSON");
        parsed_output.push(_parsed);
    }
    assert_eq!(parsed_output.len(), 50);
}

#[tokio::test(flavor = "multi_thread")]
async fn run_exact_match_eval_chat() {
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
    let output_str = String::from_utf8(output).unwrap();
    let mut parsed_output = Vec::new();
    for line in output_str.lines() {
        let _parsed: serde_json::Value =
            serde_json::from_str(line).expect("Each line should be valid JSON");
        parsed_output.push(_parsed);
    }
    assert_eq!(parsed_output.len(), 29);
}
