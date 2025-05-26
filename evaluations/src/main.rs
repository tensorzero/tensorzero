use anyhow::Result;
use clap::Parser;
use evaluations::{helpers::setup_logging, run_evaluation, Args};
use serde::Deserialize;
use std::fs;
use std::path::PathBuf;
use uuid::Uuid;

#[derive(Deserialize, Debug)]
struct EvalDefaults {
    evaluation_name: Option<String>,
    dataset_name: Option<String>,
    variant_name: Option<String>,
}

fn load_eval_defaults() -> Option<EvalDefaults> {
    let home = dirs::home_dir()?;
    let path = home.join(".tensorzero_eval_defaults.toml");
    let content = fs::read_to_string(path).ok()?;
    toml::from_str(&content).ok()
}

#[tokio::main]
async fn main() -> Result<()> {
    let evaluation_run_id = Uuid::now_v7();
    let mut args = Args::parse();

    // Load defaults if any of the required fields are missing
    let needs_defaults = args.evaluation_name.is_empty()
        || args.dataset_name.is_empty()
        || args.variant_name.is_empty();
    if needs_defaults {
        if let Some(defaults) = load_eval_defaults() {
            if args.evaluation_name.is_empty() {
                if let Some(val) = defaults.evaluation_name {
                    args.evaluation_name = val;
                }
            }
            if args.dataset_name.is_empty() {
                if let Some(val) = defaults.dataset_name {
                    args.dataset_name = val;
                }
            }
            if args.variant_name.is_empty() {
                if let Some(val) = defaults.variant_name {
                    args.variant_name = val;
                }
            }
        }
    }

    // If still missing, error out
    if args.evaluation_name.is_empty() || args.dataset_name.is_empty() || args.variant_name.is_empty() {
        anyhow::bail!("Error: --evaluation-name, --dataset-name, and --variant-name must be provided either as CLI args or in ~/.tensorzero_eval_defaults.toml");
    }

    let mut writer = std::io::stdout();
    setup_logging(&args)?;
    run_evaluation(args, evaluation_run_id, &mut writer).await
}
