use anyhow::{bail, Result};
use clap::Parser;
use evaluations::{helpers::setup_logging, run_evaluation, Args};
use tracing::{info, instrument};
use uuid::Uuid;

#[tokio::main]
#[instrument]
async fn main() -> Result<()> {
    let evaluation_run_id = Uuid::now_v7();
    let args = Args::parse();
    let mut writer = std::io::stdout();

    // Validate that exactly one of dataset_name or datapoint_ids is provided
    match (&args.dataset_name, args.datapoint_ids.is_empty()) {
        (None, true) => {
            bail!("Either --dataset-name or --datapoint-ids must be provided");
        }
        (Some(_), false) => {
            bail!("Cannot provide both --dataset-name and --datapoint-ids");
        }
        _ => {}
    }

    if let Some(dataset_name) = &args.dataset_name {
        info!(
            evaluation_run_id = %evaluation_run_id,
            evaluation_name = %args.evaluation_name,
            dataset_name = %dataset_name,
            variant_name = %args.variant_name,
            concurrency = %args.concurrency,
            "Starting evaluation run"
        );
    } else {
        info!(
            evaluation_run_id = %evaluation_run_id,
            evaluation_name = %args.evaluation_name,
            num_datapoint_ids = %args.datapoint_ids.len(),
            variant_name = %args.variant_name,
            concurrency = %args.concurrency,
            "Starting evaluation run"
        );
    }

    setup_logging(&args)?;

    let result = run_evaluation(args, evaluation_run_id, &mut writer).await;

    match &result {
        Ok(()) => {
            info!(evaluation_run_id = %evaluation_run_id, "Evaluation completed successfully");
        }
        Err(e) => {
            tracing::error!(evaluation_run_id = %evaluation_run_id, error = %e, "Evaluation failed");
        }
    }

    result
}
