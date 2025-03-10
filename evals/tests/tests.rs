#[cfg(test)]
mod common;
use evals::Args;
use uuid::Uuid;

#[tokio::test]
async fn run_exact_match_eval() {
    let eval_run_id = Uuid::now_v7();
    let args = Args {
        config_file: PathBuf::from("tensorzero-internal/tests/e2e/tensorzero.toml"),
        gateway_url: None,
        name: "exact_match".to_string(),
        variant: "gpt_4o_mini".to_string(),
        concurrency: 1,
    };
}
