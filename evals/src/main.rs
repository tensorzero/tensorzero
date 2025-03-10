use anyhow::Result;
use clap::Parser;
use evals::{run_eval, Args};
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<()> {
    let eval_run_id = Uuid::now_v7();
    let args = Args::parse();
    run_eval(args, eval_run_id).await
}
