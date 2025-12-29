use clap::Parser;
use evaluations::{Args, helpers::setup_logging, run_evaluation};
use std::process::ExitCode;
use tracing::instrument;
use uuid::Uuid;

#[tokio::main]
#[instrument]
async fn main() -> ExitCode {
    let evaluation_run_id = Uuid::now_v7();
    let args = Args::parse();
    let mut writer = std::io::stdout();

    if let Some(dataset_name) = &args.dataset_name {
        tracing::info!(
            evaluation_run_id = %evaluation_run_id,
            evaluation_name = %args.evaluation_name,
            dataset_name = %dataset_name,
            variant_name = %args.variant_name,
            concurrency = %args.concurrency,
            "Starting evaluation run"
        );
    } else {
        tracing::info!(
            evaluation_run_id = %evaluation_run_id,
            evaluation_name = %args.evaluation_name,
            num_datapoint_ids = %args.datapoint_ids.as_deref().unwrap_or_default().len(),
            variant_name = %args.variant_name,
            concurrency = %args.concurrency,
            "Starting evaluation run"
        );
    }

    if let Err(error) = setup_logging(&args) {
        tracing::error!(error = %error, "Failed to set up logging");
        return ExitCode::FAILURE;
    }

    let result = Box::pin(run_evaluation(args, evaluation_run_id, &mut writer)).await;

    match &result {
        Ok(()) => {
            tracing::info!(evaluation_run_id = %evaluation_run_id, "Evaluation completed successfully");
            return ExitCode::SUCCESS;
        }
        Err(e) => {
            tracing::error!(evaluation_run_id = %evaluation_run_id, error = %e, "Evaluation failed");
            return ExitCode::FAILURE;
        }
    }
}
