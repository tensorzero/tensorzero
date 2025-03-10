#[cfg(test)]
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
        format: OutputFormat::HumanReadable,
    };

    run_eval(args, eval_run_id).await.unwrap();
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
        format: OutputFormat::HumanReadable,
    };

    run_eval(args, eval_run_id).await.unwrap();
}
