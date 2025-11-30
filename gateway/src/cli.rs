//! CLI argument definitions for the TensorZero Gateway.
//!
//! This file should remain minimal, containing only CLI argument struct definitions.
//! This constraint exists because CODEOWNERS requires specific review for CLI changes.

use clap::{Args, Parser};
use std::path::PathBuf;
use tensorzero_core::observability::LogFormat;

#[derive(Parser, Debug)]
#[command(version, about)]
pub struct GatewayArgs {
    /// Use all of the config files matching the specified glob pattern. Incompatible with `--default-config`
    #[arg(long)]
    pub config_file: Option<PathBuf>,

    /// Use a default config file. Incompatible with `--config-file`
    #[arg(long)]
    pub default_config: bool,

    /// Sets the log format used for all gateway logs.
    #[arg(long)]
    #[arg(value_enum)]
    #[clap(default_value_t = LogFormat::default())]
    pub log_format: LogFormat,

    /// These commands trigger some workflow then exit without launching the gateway.
    #[command(flatten)]
    pub early_exit_commands: EarlyExitCommands,
}

#[derive(Args, Debug)]
#[group(multiple = false)]
pub struct EarlyExitCommands {
    /// Run ClickHouse migrations manually then exit.
    #[arg(long, alias = "run-migrations")] // TODO: remove (deprecated)
    pub run_clickhouse_migrations: bool,

    /// Run PostgreSQL migrations manually then exit.
    #[arg(long)]
    pub run_postgres_migrations: bool,

    /// Create an API key then exit.
    #[arg(long)]
    pub create_api_key: bool,
}
