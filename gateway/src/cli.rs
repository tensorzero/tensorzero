//! CLI argument definitions for the TensorZero Gateway.
//!
//! This file should remain minimal, containing only CLI argument struct definitions.
//! This constraint exists because CODEOWNERS requires specific review for CLI changes.

use clap::{Args, Parser};
use std::net::SocketAddr;
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

    /// Sets the socket address the gateway will bind to (e.g., "127.0.0.1:8080").
    #[arg(long)]
    pub bind_address: Option<SocketAddr>,

    /// These commands trigger some workflow then exit without launching the gateway.
    #[command(flatten)]
    pub early_exit_commands: EarlyExitCommands,

    /// Arguments that control the behavior of Postgres migrations.
    #[command(flatten)]
    pub postgres_migration_args: PostgresMigrationArgs,
}

#[derive(Args, Debug)]
#[group(multiple = false)]
pub struct EarlyExitCommands {
    /// Run ClickHouse migrations manually then exit.
    #[arg(long, alias = "run-migrations")] // TODO: remove (deprecated)
    pub run_clickhouse_migrations: bool,

    /// Run Postgres migrations manually then exit.
    #[arg(long)]
    pub run_postgres_migrations: bool,

    /// Create an API key then exit.
    #[arg(long)]
    pub create_api_key: bool,

    /// Disable an API key using its public ID then exit.
    #[arg(long, value_name = "PUBLIC_ID")]
    pub disable_api_key: Option<String>,

    /// Validate the config file then exit.
    #[arg(long)]
    pub validate_and_exit: bool,
}

#[derive(Args, Debug)]
pub struct PostgresMigrationArgs {
    /// Run Postgres migrations for optimizations.
    #[arg(long, default_value_t = false)]
    pub enable_optimization_postgres_migrations: bool,
}
