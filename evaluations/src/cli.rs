//! CLI argument definitions for the TensorZero Evaluations binary.
//!
//! This file should remain minimal, containing only CLI argument struct definitions.
//! This constraint exists because CODEOWNERS requires specific review for CLI changes.

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use serde::{Deserialize, Serialize};
use tensorzero_core::cache::CacheEnabledMode;
use url::Url;
use uuid::Uuid;

#[derive(clap::ValueEnum, Clone, Debug, Default, PartialEq, Deserialize, Serialize)]
#[clap(rename_all = "snake_case")]
#[serde(rename_all = "snake_case")]
pub enum OutputFormat {
    #[default]
    Pretty,
    Jsonl,
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
    /// Path to tensorzero.toml.
    #[arg(long, default_value = "./config/tensorzero.toml")]
    pub config_file: PathBuf,

    /// URL of a running TensorZero HTTP gateway server to use for requests. This runs evaluations using that gateway.
    #[arg(long)]
    pub gateway_url: Option<Url>,

    /// Name of the evaluation to run.
    #[arg(short, long)]
    pub evaluation_name: String,

    /// Name of the dataset to run on.
    /// Either dataset_name or datapoint_ids must be provided, but not both.
    #[arg(short, long)]
    pub dataset_name: Option<String>,

    /// Specific datapoint IDs to evaluate (comma-separated).
    /// Either dataset_name or datapoint_ids must be provided, but not both.
    /// Example: --datapoint-ids 01957bbb-44a8-7490-bfe7-32f8ed2fc797,01957bbb-44a8-7490-bfe7-32f8ed2fc798
    #[arg(long, value_delimiter = ',')]
    pub datapoint_ids: Option<Vec<Uuid>>,

    /// Name of the variant to run.
    #[arg(short, long)]
    pub variant_name: String,

    /// Number of concurrent requests to make.
    #[arg(short, long, default_value = "1")]
    pub concurrency: usize,

    #[arg(short, long, default_value = "pretty")]
    pub format: OutputFormat,

    #[arg(long, default_value = "on")]
    pub inference_cache: CacheEnabledMode,

    /// Maximum number of datapoints to evaluate from the dataset.
    #[arg(long)]
    pub max_datapoints: Option<u32>,

    /// Per-evaluator precision targets for adaptive stopping.
    /// Format: evaluator_name=precision_target, comma-separated for multiple evaluators.
    /// Example: --adaptive-stopping-precision exact_match=0.13,llm_judge=0.16
    /// Evaluator stops when confidence interval (CI) half-width (or the maximum width of the two
    /// halves of the CI in the case of asymmetric CIs) <= precision_target.
    #[arg(long = "adaptive-stopping-precision", value_parser = parse_precision_target, value_delimiter = ',', num_args = 0..)]
    pub precision_targets: Vec<(String, f32)>,
}

/// Parse a single precision target in format "evaluator_name=precision_target"
fn parse_precision_target(s: &str) -> Result<(String, f32), String> {
    let s = s.trim();
    if s.is_empty() {
        return Err("Precision target cannot be empty".to_string());
    }

    let parts: Vec<&str> = s.splitn(2, '=').collect();
    if parts.len() != 2 {
        return Err(format!(
            "Invalid precision format: `{s}`. Expected format: evaluator_name=precision_target"
        ));
    }

    let evaluator_name = parts[0].to_string();
    let precision_target = parts[1]
        .parse::<f32>()
        .map_err(|e| format!("Invalid precision value `{}`: {e}", parts[1]))?;

    if precision_target < 0.0 {
        return Err(format!(
            "Precision value must be non-negative, got {precision_target}"
        ));
    }

    Ok((evaluator_name, precision_target))
}
