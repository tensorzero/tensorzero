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

    /// Name of the evaluation to run (legacy mode: evaluators configured in the evaluation).
    /// Mutually exclusive with --function-name / --evaluator-names.
    #[arg(short, long, group = "eval_source")]
    pub evaluation_name: Option<String>,

    /// Name of the function to evaluate. Used with --evaluator-names to run
    /// top-level evaluators without a named evaluation config.
    #[arg(long, requires = "evaluator_names", group = "eval_source")]
    pub function_name: Option<String>,

    /// Comma-separated list of top-level evaluator names to run.
    /// Must be used with --function-name.
    #[arg(long, requires = "function_name", value_delimiter = ',')]
    pub evaluator_names: Option<Vec<String>>,

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

    /// Per-evaluator cutoff thresholds for pass/fail exit status.
    /// Format: evaluator_name=cutoff, comma-separated for multiple evaluators.
    /// Example: --cutoffs exact_match=0.95,llm_judge=0.8
    /// If both this CLI flag and evaluator config `cutoff` are provided
    /// for the same evaluator, the CLI value takes precedence.
    #[arg(long, value_parser = parse_cutoff_target, value_delimiter = ',', num_args = 0..)]
    pub cutoffs: Vec<(String, f32)>,
}

/// Parse a single precision target in format "evaluator_name=precision_target"
fn parse_precision_target(s: &str) -> Result<(String, f32), String> {
    parse_named_non_negative_float(s, "precision")
}

/// Parse a single cutoff target in format "evaluator_name=cutoff"
fn parse_cutoff_target(s: &str) -> Result<(String, f32), String> {
    parse_named_non_negative_float(s, "cutoff")
}

fn parse_named_non_negative_float(input: &str, value_label: &str) -> Result<(String, f32), String> {
    let input = input.trim();
    if input.is_empty() {
        return Err(format!("{value_label} cannot be empty"));
    }

    let parts: Vec<&str> = input.splitn(2, '=').collect();
    if parts.len() != 2 {
        return Err(format!(
            "Invalid {value_label} format: `{input}`. Expected format: evaluator_name=value"
        ));
    }

    let evaluator_name = parts[0].to_string();
    let value = parts[1]
        .parse::<f32>()
        .map_err(|e| format!("Invalid `{value_label}` value `{}`: {e}", parts[1]))?;

    if value < 0.0 {
        return Err(format!(
            "{value_label} value must be non-negative, got {value}",
        ));
    }

    Ok((evaluator_name, value))
}
