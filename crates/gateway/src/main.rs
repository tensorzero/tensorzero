#![recursion_limit = "256"]

use clap::Parser;
use futures::{FutureExt, StreamExt};
use mimalloc::MiMalloc;
use secrecy::ExposeSecret;
use sqlx::postgres::PgPoolOptions;
use sqlx::types::chrono::{DateTime, Utc};
use std::fmt::Display;
use std::future::{Future, IntoFuture};
use std::io::ErrorKind;
use std::net::SocketAddr;
use std::process::ExitCode;
use std::time::Duration;
use tensorzero_core::config::{default_flush_interval_ms, default_max_rows};
use tensorzero_core::observability::request_logging::InFlightRequestsData;
use tokio::signal;
use tokio_stream::wrappers::IntervalStream;
use tracing::Level;

use autopilot_worker::{AutopilotWorkerConfig, AutopilotWorkerHandle, spawn_autopilot_worker};
use durable_tools::{EmbeddedClient, WorkerOptions};
use tensorzero_auth::constants::{DEFAULT_ORGANIZATION, DEFAULT_WORKSPACE};
use tensorzero_core::config::{Config, ConfigFileGlob, unwritten::UnwrittenConfig};
use tensorzero_core::db::clickhouse::migration_manager::manual_run_clickhouse_migrations;
use tensorzero_core::db::delegating_connection::PrimaryDatastore;
use tensorzero_core::db::postgres::postgres_setup::{
    check_pgcron_configured_correctly, check_pgvector_configured_correctly,
    check_trigram_indexes_configured_correctly,
};
use tensorzero_core::db::postgres::{PostgresConnectionInfo, manual_run_postgres_migrations};
use tensorzero_core::db::valkey::ValkeyConnectionInfo;
use tensorzero_core::endpoints::status::TENSORZERO_VERSION;
use tensorzero_core::error::{self, Error, ErrorDetails, IMPOSSIBLE_ERROR_MESSAGE};
use tensorzero_core::feature_flags;
use tensorzero_core::observability;
use tensorzero_core::utils::gateway;

mod cli;
mod router;
mod routes;

use cli::GatewayArgs;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// Maximum number of Postgres connections used by the gateway's startup-time
/// pool that loads configuration from the database. `load_config_from_db`
/// requires a leader transaction to be held open while 14 reads run in parallel,
/// so this number must stay above 2 (and we use a number >15 for performance).
const STARTUP_CONFIG_DB_POOL_MAX_CONNECTIONS: u32 = 20;

/// Name of the environment variable holding the Postgres connection string.
const TENSORZERO_POSTGRES_URL_ENV: &str = "TENSORZERO_POSTGRES_URL";

struct StartupConfig {
    unwritten: UnwrittenConfig,
    glob: Option<ConfigFileGlob>,
    from_database: bool,
}

/// Reads `TENSORZERO_POSTGRES_URL` and treats an unset or empty value as absent.
/// A present-but-empty env var is almost always a shell/compose misconfiguration;
/// propagating it forward produces an opaque sqlx dial error, so we normalize it
/// to `None` here and let callers handle the "missing" case uniformly.
fn read_postgres_url_env() -> Option<String> {
    std::env::var(TENSORZERO_POSTGRES_URL_ENV)
        .ok()
        .filter(|url| !url.is_empty())
}

async fn load_startup_config(args: &GatewayArgs) -> Result<StartupConfig, ExitCode> {
    if args.default_config && args.config_file.is_some() {
        tracing::error!("You must not specify both `--config-file` and `--default-config`.");
        return Err(ExitCode::FAILURE);
    }

    if let Some(path) = args.config_file.as_ref() {
        let (unwritten, glob) = load_config_from_path_glob(path).await?;
        return Ok(StartupConfig {
            unwritten,
            glob: Some(glob),
            from_database: false,
        });
    }

    if args.default_config {
        tracing::warn!(
            "No config file provided, so only default functions will be available. Use `--config-file path/to/tensorzero.toml` to specify a config file."
        );
        let unwritten = Config::new_empty()
            .await
            .log_err_pretty("Failed to load default config")?;
        return Ok(StartupConfig {
            unwritten,
            glob: None,
            from_database: false,
        });
    }

    // Load configuration from Postgres only when the operator has explicitly
    // opted in via the `ENABLE_CONFIG_IN_DATABASE` feature flag. An empty
    // database is a valid starting point: the gateway will serve a functional
    // runtime with no functions/variants, and operators populate config
    // through the REST API (or the UI). `TENSORZERO_POSTGRES_URL` by itself
    // is NOT sufficient to take this path — many deployments set that env
    // var for observability or rate-limiting without intending to boot from
    // DB config, and config-in-database is still behind the flag while we
    // harden it.
    if feature_flags::ENABLE_CONFIG_IN_DATABASE.get() {
        let postgres_url = read_postgres_url_env();
        let unwritten = load_startup_config_from_database(postgres_url.as_deref())
            .await
            .log_err_pretty("Failed to load configuration from database")?;
        return Ok(StartupConfig {
            unwritten,
            glob: None,
            from_database: true,
        });
    }

    tracing::error!("You must specify either `--config-file` or `--default-config`.");
    Err(ExitCode::FAILURE)
}

fn merge_db_config_load_errors(errors: Vec<Error>) -> Error {
    let mut errors = errors.into_iter();
    let Some(first_error) = errors.next() else {
        return Error::new(ErrorDetails::InternalError {
            message: format!(
                "merge_db_config_load_errors called with empty error list. {IMPOSSIBLE_ERROR_MESSAGE}"
            ),
        });
    };
    let remaining_messages = errors.map(|error| error.to_string()).collect::<Vec<_>>();
    if remaining_messages.is_empty() {
        return first_error;
    }
    let first_message = first_error.to_string();
    Error::new(ErrorDetails::Config {
        message: format!(
            "Failed to load configuration from database:\n- {}\n- {}",
            first_message,
            remaining_messages.join("\n- ")
        ),
    })
}

async fn load_startup_config_from_database(
    postgres_url: Option<&str>,
) -> Result<UnwrittenConfig, Error> {
    let postgres_url = postgres_url.ok_or_else(|| {
        Error::new(ErrorDetails::PostgresConnectionInitialization {
            message: format!(
                "Missing environment variable `{TENSORZERO_POSTGRES_URL_ENV}` required to load configuration from database."
            ),
        })
    })?;
    // `load_config_from_db` fans out into 14 parallel snapshot readers plus a
    // leader transaction, so the pool must allow at least 15 simultaneous
    // connections. We add a small margin to absorb transient slowness without
    // hitting the default 30s acquire timeout during gateway startup.
    let pool = PgPoolOptions::new()
        .max_connections(STARTUP_CONFIG_DB_POOL_MAX_CONNECTIONS)
        .connect(postgres_url)
        .await
        .map_err(|error| {
            Error::new(ErrorDetails::PostgresConnectionInitialization {
                message: error.to_string(),
            })
        })?;
    Config::load_from_db(&pool, true)
        .await
        .map_err(merge_db_config_load_errors)
}

async fn load_config_from_path_glob(
    path: &std::path::Path,
) -> Result<
    (
        tensorzero_core::config::unwritten::UnwrittenConfig,
        ConfigFileGlob,
    ),
    ExitCode,
> {
    let glob =
        ConfigFileGlob::new_from_path(path).log_err_pretty("Failed to process config file glob")?;
    let glob_display = glob.glob.clone();
    let resolved_paths = glob
        .paths
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join("\n");
    let config = Config::load_and_verify_from_path(&glob)
        .await
        .ok()
        .log_err_pretty(&format!(
            "Failed to load config. Config file glob `{glob_display}` resolved to the following files:\n{resolved_paths}"
        ))?;
    Ok((config, glob))
}

async fn store_config_in_database(
    uninitialized_config: tensorzero_core::config::UninitializedConfig,
    extra_templates: &std::collections::HashMap<String, String>,
) -> Result<(), Error> {
    let postgres_url = std::env::var("TENSORZERO_POSTGRES_URL").map_err(|_| {
        Error::new(ErrorDetails::AppState {
            message: "Missing environment variable `TENSORZERO_POSTGRES_URL`. \
                      `--migrate-config` requires a Postgres connection."
                .to_string(),
        })
    })?;

    let pool = PgPoolOptions::new()
        .connect(&postgres_url)
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::AppState {
                message: format!("Failed to connect to Postgres: {e}"),
            })
        })?;

    let postgres = PostgresConnectionInfo::new_with_pool(pool);
    postgres
        .write_stored_config(
            tensorzero_core::db::postgres::stored_config_writes::WriteStoredConfigParams {
                config: &uninitialized_config,
                creation_source: "migrate-config-cli",
                source_autopilot_session_id: None,
                extra_templates,
            },
        )
        .await?;

    Ok(())
}

#[expect(clippy::print_stdout)]
fn print_key(key: &secrecy::SecretString) {
    println!("{}", key.expose_secret());
}

async fn handle_create_api_key(
    expiration: Option<DateTime<Utc>>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Read the Postgres URL from the environment
    let postgres_url = std::env::var("TENSORZERO_POSTGRES_URL")
        .map_err(|_| "TENSORZERO_POSTGRES_URL environment variable not set")?;

    let now = Utc::now();

    if let Some(expiration_datetime) = expiration
        && expiration_datetime < now
    {
        return Err("Expiration datetime needs to be in the future".into());
    }

    // Create connection pool (alpha version for tensorzero-auth)
    let pool = sqlx::PgPool::connect(&postgres_url).await?;

    // Create the key with default organization and workspace
    let key = tensorzero_auth::postgres::create_key(
        DEFAULT_ORGANIZATION,
        DEFAULT_WORKSPACE,
        None,
        expiration,
        &pool,
    )
    .await?;

    // Print only the API key to stdout for easy machine parsing
    print_key(&key);

    if let Some(expiration) = expiration {
        tracing::debug!("Created API key with expiration: {expiration}");
    } else {
        tracing::debug!("Created API key with no expiration");
    }

    Ok(())
}

async fn run_optimization_postgres_migrations() -> Result<(), Error> {
    let postgres_url = std::env::var("TENSORZERO_POSTGRES_URL").map_err(|_| {
        Error::new(ErrorDetails::PostgresConnectionInitialization {
            message: "Failed to read TENSORZERO_POSTGRES_URL environment variable".to_string(),
        })
    })?;
    let pool = sqlx::PgPool::connect(&postgres_url).await.map_err(|e| {
        Error::new(ErrorDetails::PostgresConnectionInitialization {
            message: e.to_string(),
        })
    })?;

    // The migration error is silently swallowed, because we don't want to require pgvector yet.
    // TODO(#6912): require optimization migrations to run correctly soon.
    if let Err(e) = tensorzero_optimizers::postgres::make_migrator()
        .run(&pool)
        .await
    {
        tracing::warn!(
            "Failed to run Postgres migrations for optimization: {e}. This is non-fatal, but TensorZero will require them soon."
        );
    }

    if let Err(e) = check_pgvector_configured_correctly(&pool).await {
        let msg = e.suppress_logging_of_error_message();
        tracing::warn!(
            "pgvector extension is not configured correctly for your Postgres setup: {msg}. TensorZero will start requiring pgvector soon.",
        );
    }

    Ok(())
}

async fn handle_disable_api_key(public_id: &str) -> Result<(), Box<dyn std::error::Error>> {
    let postgres_url = std::env::var("TENSORZERO_POSTGRES_URL")
        .map_err(|_| "TENSORZERO_POSTGRES_URL environment variable not set")?;
    let pool = sqlx::PgPool::connect(&postgres_url).await?;

    let disabled_time = tensorzero_auth::postgres::disable_key(public_id, &pool).await?;

    tracing::info!("Deleted API key {public_id} at {disabled_time}");

    Ok(())
}

async fn validate_postgres_extensions_for_postgres_primary(
    gateway_handle: &gateway::GatewayHandle,
) -> Result<(), ExitCode> {
    if gateway_handle.app_state.primary_datastore() != PrimaryDatastore::Postgres {
        return Ok(());
    }
    let postgres = gateway_handle.app_state.postgres_connection_info();
    let Some(pgpool) = postgres.get_pool() else {
        tracing::error!(
            "Postgres is configured to be the primary observability backend, but cannot establish a postgres connection."
        );
        return Err(ExitCode::FAILURE);
    };

    let (pgcron_result, trigram_result, pgvector_result) = tokio::join!(
        check_pgcron_configured_correctly(pgpool),
        check_trigram_indexes_configured_correctly(pgpool),
        check_pgvector_configured_correctly(pgpool),
    );

    let mut has_fatal_error = false;

    if let Err(e) = pgcron_result {
        e.log_at_level("Postgres is configured to be the primary observability backend, but pgcron is not configured correctly: ", Level::ERROR);
        has_fatal_error = true;
    }

    if let Err(e) = trigram_result {
        e.log_at_level("Postgres is configured to be the primary observability backend, but trigram indices are not configured correctly: ", Level::ERROR);
        has_fatal_error = true;
    }

    if let Err(e) = pgvector_result {
        e.log_at_level("TensorZero will require pgvector soon for deployments with Postgres, and pgvector is not configured correctly: ", Level::WARN);
    }

    if has_fatal_error {
        return Err(ExitCode::FAILURE);
    }

    Ok(())
}

#[tokio::main]
async fn main() -> ExitCode {
    match Box::pin(run()).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(code) => code,
    }
}

async fn run() -> Result<(), ExitCode> {
    let args = GatewayArgs::parse();

    // Set up logs and metrics immediately, so that we can use `tracing`.
    // OTLP will be enabled based on the config file
    // We start with empty headers and update them after loading the config
    let delayed_log_config = observability::setup_observability(args.log_format.clone(), true)
        .await
        .log_err_pretty("Failed to set up logs")?;

    if args.early_exit_commands.create_api_key {
        handle_create_api_key(args.early_exit_command_arguments.expiration)
            .await
            .log_err_pretty("Failed to create API key")?;
        return Ok(());
    }

    if let Some(public_id) = args.early_exit_commands.disable_api_key {
        handle_disable_api_key(&public_id)
            .await
            .log_err_pretty("Failed to delete API key")?;

        return Ok(());
    }

    if args.early_exit_commands.run_clickhouse_migrations {
        tracing::info!("Applying ClickHouse migrations...");
        manual_run_clickhouse_migrations()
            .await
            .log_err_pretty("Failed to run ClickHouse migrations")?;
        tracing::info!("ClickHouse is ready.");
        return Ok(());
    }

    if args.early_exit_commands.run_postgres_migrations {
        tracing::info!("Applying Postgres migrations...");
        manual_run_postgres_migrations()
            .await
            .log_err_pretty("Failed to run Postgres migrations")?;

        run_optimization_postgres_migrations()
            .await
            .log_err_pretty("Failed to run optimization Postgres migrations.")?;

        tracing::info!("Postgres is ready.");
        return Ok(());
    }

    if let Some(config_path) = args.early_exit_commands.migrate_config.as_ref() {
        let glob = ConfigFileGlob::new_from_path(config_path)
            .log_err_pretty("Failed to process config file glob")?;
        print_configuration_info(Some(&glob));

        // Parse the glob into UninitializedConfig (before load/initialization).
        // TOML path resolution already strips the shared filesystem prefix from
        // template keys while retaining the absolute path separately for runtime use.
        let globbed = tensorzero_core::config::UninitializedConfig::read_toml_config(&glob, false)
            .ok()
            .log_err_pretty("Failed to parse config files")?;
        let uninitialized_config =
            tensorzero_core::config::UninitializedConfig::try_from(globbed.table)
                .ok()
                .log_err_pretty("Failed to deserialize config")?;

        if uninitialized_config
            .gateway
            .as_ref()
            .and_then(|g| g.template_filesystem_access.as_ref())
            .is_some_and(|tfa| tfa.is_active())
        {
            tracing::error!(
                "`template_filesystem_access` is set in the gateway config, but \
                 `--migrate-config` does not support filesystem-based template access. \
                 Remove or disable `gateway.template_filesystem_access` and modify your \
                 templates to remove file imports before migrating config."
            );
            return Err(ExitCode::FAILURE);
        }

        // Validate by running the full load pipeline (ensures the config is valid before storing).
        // Also extracts extra templates discovered from the filesystem — all prompts, whether
        // explicitly specified or dynamically included, must be stored in the database.
        let validated = Config::load_and_verify_from_path(&glob)
            .await
            .ok()
            .log_err_pretty("Config validation failed")?;
        let extra_templates = validated.extra_templates().clone();

        store_config_in_database(uninitialized_config, &extra_templates)
            .await
            .log_err_pretty("Failed to store configuration in the database")?;
        tracing::info!("Configuration stored in the database.");
        return Ok(());
    }

    tracing::info!("Starting TensorZero Gateway {TENSORZERO_VERSION}");

    let StartupConfig {
        unwritten: unwritten_config,
        glob,
        from_database: config_in_database,
    } = load_startup_config(&args).await?;

    let metrics_handle = observability::setup_metrics(Some(&unwritten_config.gateway.metrics))
        .log_err_pretty("Failed to set up metrics")?;

    if unwritten_config.gateway.debug {
        delayed_log_config
            .delayed_debug_logs
            .enable_debug()
            .log_err_pretty("Failed to enable debug logs")?;
    }

    // Note: We only enable OTLP after config file parsing/loading is complete,
    // so that the config file can control whether OTLP is enabled or not.
    // This means that any tracing spans created before this point will not be exported to OTLP.
    // For now, this is fine, as we only ever export spans for inference/batch/feedback requests,
    // which cannot have occurred up until this point.
    // If we ever want to emit earlier OTLP spans, we'll need to come up with a different way
    // of doing OTLP initialization (e.g. buffer spans, and submit them once we know if OTLP should be enabled).
    // See `build_opentelemetry_layer` for the details of exactly what spans we export.
    let export_config = &unwritten_config.gateway.export;
    let otlp_traces_enabled = export_config
        .otlp
        .as_ref()
        .and_then(|o| o.traces.as_ref())
        .and_then(|t| t.enabled)
        .unwrap_or(false);
    if otlp_traces_enabled {
        if std::env::var("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT").is_err() {
            // This makes it easier to run the gateway in local development and CI
            if cfg!(feature = "e2e_tests") {
                tracing::warn!(
                    "Running without explicit `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` environment variable in e2e tests mode."
                );
            } else {
                tracing::error!(
                    "The `gateway.export.otlp.traces.enabled` configuration option is `true`, but environment variable `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` is not set. Please set it to the OTLP endpoint (e.g. `http://localhost:4317`)."
                );
                return Err(ExitCode::FAILURE);
            }
        }

        // Set config-level OTLP headers if we have a tracer wrapper
        if let Some(ref tracer_wrapper) = delayed_log_config.otel_tracer {
            let extra_headers = export_config
                .otlp
                .as_ref()
                .and_then(|o| o.traces.as_ref())
                .and_then(|t| t.extra_headers.as_ref());
            if let Some(headers) = extra_headers
                && !headers.is_empty()
            {
                tracer_wrapper
                    .set_static_otlp_traces_extra_headers(headers)
                    .log_err_pretty("Failed to set OTLP config headers")?;
            }
        }

        match delayed_log_config.delayed_otel {
            Ok(delayed_otel) => {
                delayed_otel
                    .enable_otel()
                    .log_err_pretty("Failed to enable OpenTelemetry")?;
            }
            Err(e) => {
                tracing::error!(
                    "Could not enable OpenTelemetry export due to previous error: `{e}`. Exiting."
                );
                return Err(ExitCode::FAILURE);
            }
        }
    } else if let Err(e) = delayed_log_config.delayed_otel {
        tracing::warn!(
            "[gateway.export.otlp.traces.enabled] is `false`, so ignoring OpenTelemetry error: `{e}`"
        );
    }

    // Collect available tool names for autopilot (single source of truth)
    let available_tools = autopilot_tools::collect_tool_names()
        .await
        .log_err_pretty("Failed to collect autopilot tool names")?;

    // Resolve tool whitelist from config
    let tool_whitelist: std::collections::HashSet<String> =
        match &unwritten_config.autopilot.tool_whitelist {
            Some(list) => list.iter().cloned().collect(),
            None => autopilot_tools::default_whitelisted_tool_names(),
        };

    // Initialize GatewayHandle
    let gateway_handle = gateway::GatewayHandle::new(
        unwritten_config,
        available_tools,
        tool_whitelist,
        config_in_database,
    )
    .await
    .map_err(|e| {
        e.log_at_level("Failed to initialize AppState: ", tracing::Level::ERROR);
        ExitCode::FAILURE
    })?;

    validate_postgres_extensions_for_postgres_primary(&gateway_handle).await?;

    // Start autopilot worker if configured
    let autopilot_worker_handle = spawn_autopilot_worker_if_configured(&gateway_handle).await?;

    // Start tool whitelist approver if configured
    if let Some(client) = gateway_handle.app_state.autopilot_client.clone() {
        let token = gateway_handle.app_state.shutdown_token.clone();
        gateway_handle
            .app_state
            .deferred_tasks
            .spawn(async move { client.run_tool_whitelist_approver(token).await });
    }

    // Create a new observability_enabled_pretty string for the log message below
    let postgres_enabled_pretty =
        get_postgres_status_string(&gateway_handle.app_state.postgres_connection_info());

    let config = gateway_handle.app_state.config().load();

    // Set debug mode
    error::set_debug(config.gateway.debug).log_err_pretty("Failed to set debug mode")?;
    error::set_unstable_error_json(config.gateway.unstable_error_json)
        .log_err_pretty("Failed to set unstable error JSON")?;

    let base_path = config.gateway.base_path.as_deref().unwrap_or("/");
    if !base_path.starts_with("/") {
        tracing::error!("[gateway.base_path] must start with a `/` : `{base_path}`");
        return Err(ExitCode::FAILURE);
    }
    let base_path = base_path.trim_end_matches("/");

    if args.early_exit_commands.validate_and_exit {
        tracing::info!("Configuration is valid. Exiting.");
        return Ok(());
    }

    let (router, in_flight_requests_data) = router::build_axum_router(
        base_path,
        delayed_log_config.otel_tracer.clone(),
        gateway_handle.app_state.clone(),
        metrics_handle,
        gateway_handle.app_state.shutdown_token.clone(),
    )
    .await
    .log_err_pretty("Failed to build router")?;

    // Bind to the socket address specified in the CLI, config, or default to 0.0.0.0:3000
    if args.bind_address.is_some() && config.gateway.bind_address.is_some() {
        tracing::error!(
            "You must only specify one of `--bind-address` (CLI), `TENSORZERO_GATEWAY_BIND_ADDRESS` (environment variable), or `gateway.bind_address` (configuration)."
        );
        return Err(ExitCode::FAILURE);
    }
    let bind_address = args
        .bind_address
        .or(config.gateway.bind_address)
        .unwrap_or_else(|| SocketAddr::from(([0, 0, 0, 0], 3000)));

    let listener = match tokio::net::TcpListener::bind(bind_address).await {
        Ok(listener) => listener,
        Err(e) if e.kind() == ErrorKind::AddrInUse => {
            tracing::error!(
                "Failed to bind to socket address {bind_address}: {e}. Tip: Ensure no other process is using port {} or try a different port.",
                bind_address.port()
            );
            return Err(ExitCode::FAILURE);
        }
        Err(e) => {
            tracing::error!("Failed to bind to socket address {bind_address}: {e}");
            return Err(ExitCode::FAILURE);
        }
    };

    // This will give us the chosen port if the user specified a port of 0
    let actual_bind_address = listener
        .local_addr()
        .log_err_pretty("Failed to get bind address from listener")?;

    // Print the bind address
    tracing::info!("TensorZero Gateway is listening on {actual_bind_address}");

    // Print the base path if set
    if base_path.is_empty() {
        tracing::info!("├ API Base Path: /");
    } else {
        tracing::info!("├ API Base Path: {base_path}");
    }

    // Print the configuration being used
    if glob.is_none() && !args.default_config {
        tracing::info!("├ Configuration: database");
    } else {
        print_configuration_info(glob.as_ref());
    }

    // Print observability backend and ClickHouse status
    let observability_backend = match gateway_handle.app_state.primary_datastore() {
        PrimaryDatastore::ClickHouse => "ClickHouse",
        PrimaryDatastore::Postgres => "Postgres",
        PrimaryDatastore::Disabled => "disabled",
    };
    tracing::info!("├ Observability Backend: {observability_backend}");
    tracing::info!(
        "├ ClickHouse: {}",
        gateway_handle.app_state.clickhouse_connection_info()
    );
    let batch_writes = config
        .gateway
        .observability
        .batch_writes
        .clone()
        .unwrap_or_default();
    if batch_writes.enabled {
        tracing::info!(
            "├ Batch Writes: enabled (flush_interval_ms = {}, max_rows = {})",
            batch_writes
                .flush_interval_ms
                .unwrap_or_else(default_flush_interval_ms),
            batch_writes.max_rows.unwrap_or_else(default_max_rows)
        );
    } else {
        tracing::info!("├ Batch Writes: disabled");
    }
    if config.gateway.observability.async_writes() {
        tracing::info!("├ Async Writes: enabled");
    } else {
        tracing::info!("├ Async Writes: disabled");
    }

    // Print whether postgres is enabled
    tracing::info!("├ Postgres: {postgres_enabled_pretty}");

    // Print whether valkey is enabled
    let valkey_enabled_pretty =
        get_valkey_status_string(&gateway_handle.app_state.valkey_connection_info());
    tracing::info!("├ Valkey: {valkey_enabled_pretty}");
    if std::env::var("TENSORZERO_VALKEY_CACHE_URL").is_ok() {
        let valkey_cache_enabled_pretty =
            get_valkey_status_string(&gateway_handle.app_state.valkey_cache_connection_info());
        tracing::info!("├ Valkey (cache): {valkey_cache_enabled_pretty}");
    }

    if let Some(gateway_url) = config
        .gateway
        .relay
        .as_ref()
        .and_then(|relay| relay.original_config.gateway_url.as_ref())
    {
        tracing::info!("├ Relay mode: enabled (gateway_url = {gateway_url})");
    } else {
        tracing::info!("├ Relay mode: disabled");
    }

    // Print whether Autopilot Worker is enabled
    if autopilot_worker_handle.is_some() {
        tracing::info!("├ Autopilot Worker: enabled");
    } else {
        tracing::info!("├ Autopilot Worker: disabled");
    }

    // Print whether Autopilot Tool Whitelist Approver is enabled
    if let Some(client) = gateway_handle.app_state.autopilot_client.as_ref() {
        let count = client.tool_whitelist.len();
        if count > 0 {
            tracing::info!("├ Autopilot Tool Whitelist Approver: enabled ({count} tools)");
        } else {
            tracing::info!("├ Autopilot Tool Whitelist Approver: disabled (empty whitelist)");
        }
    } else {
        tracing::info!("├ Autopilot Tool Whitelist Approver: disabled");
    }

    // Print whether OpenTelemetry is enabled
    let otlp_traces_enabled = config
        .gateway
        .export
        .otlp
        .as_ref()
        .and_then(|o| o.traces.as_ref())
        .and_then(|t| t.enabled)
        .unwrap_or(false);
    if otlp_traces_enabled {
        tracing::info!("└ OpenTelemetry: enabled");
    } else {
        tracing::info!("└ OpenTelemetry: disabled");
    }

    let shutdown_token = gateway_handle.app_state.shutdown_token.clone();
    let shutdown_token_clone = shutdown_token.clone();
    // This is responsible for starting the shutdown
    #[expect(clippy::disallowed_methods)]
    tokio::spawn(async move {
        shutdown_signal().await;
        shutdown_token_clone.cancel();
    });

    let server_fut = axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_token.clone().cancelled_owned())
        .into_future()
        .map(|r| {
            let _ = r.log_err_pretty("Failed to start server");
        })
        .shared();

    // This is a purely informational logging task, so we don't need to wait for it to finish.
    #[expect(clippy::disallowed_methods)]
    tokio::spawn(monitor_server_shutdown(
        shutdown_token.clone().cancelled_owned(),
        server_fut.clone(),
        in_flight_requests_data,
    ));

    // Wait for the server to finish - this happens once the shutdown signal is received,
    // and after axum completes its graceful shutdown.
    //
    // The overall shutdown happens in multiple phases:
    // 1. The 'shutdown_signal' resolves (e.g. due to a Ctrl-C signal)
    // 2. Axum detects the shutdown signal via `with_graceful_shutdown`.
    //    It stops accepting new requests, and finishes processing existing requests
    // 3. The 'server_fut' future resolves when Axum itself is finished. However,
    //     we still might have running `tokio::task`s with spans that descend from
    //     HTTP request spans (e.g. rate-limiting `return_tickets` calls)
    // 4. When OpenTelemetry is enabled, we call `tracer_wrapper.shutdown()`.
    //    * We first wait for all of the spans descending from HTTP requests to finish.
    //      At this point, no new OTEL-exported spans should be created, or they might
    //      not be exported to OTLP before we exit.
    //    * We then start the shutdown of all our OpenTelemetry exporters, and wait
    //      for them to complete.
    // 5.  Our `GatewayHandle` drops, and blocks on any final remaining shutdown tasks
    //     (e.g. the ClickHouse batch writer task)
    server_fut.await;

    if let Some(tracer_wrapper) = delayed_log_config.otel_tracer {
        tracing::info!("Shutting down OpenTelemetry exporter");
        tracer_wrapper
            .shutdown(delayed_log_config.leak_detector.as_ref())
            .await;
        tracing::info!("OpenTelemetry exporter shut down");
    }
    Ok(())
}

/// A background task that waits for the server shutdown to initiate, and then logs status information every 5 seconds until
/// the server completes its shutdown.
async fn monitor_server_shutdown(
    shutdown_signal: impl Future<Output = ()>,
    server_fut: impl Future<Output = ()>,
    in_flight_requests_data: InFlightRequestsData,
) {
    // First, wait for the shutdown signal
    shutdown_signal.await;
    // The server should now be shutting down, so print a message every 5 seconds until it completes
    IntervalStream::new(tokio::time::interval(Duration::from_secs(5)))
        .take_until(server_fut)
        .for_each(|_| async {
            let counts = in_flight_requests_data
                .current_counts_by_route()
                .collect::<Vec<_>>();

            let total = counts.iter().map(|(_, count)| *count).sum::<u32>();
            tracing::info!(
                "Server shutdown in progress: {} in-flight requests remaining",
                total
            );
            if total > 0 {
                tracing::info!("In-flight requests by route:");
                for (route, count) in counts {
                    if count > 0 {
                        tracing::info!("├ `{route}` -> {count} requests in flight");
                    }
                }
            }
        })
        .await;
    tracing::info!("Server shutdown complete");
}

fn get_postgres_status_string(postgres: &PostgresConnectionInfo) -> String {
    match postgres {
        PostgresConnectionInfo::Disabled => "disabled".to_string(),
        PostgresConnectionInfo::Enabled { .. } => "enabled".to_string(),
        // Mock variant only exists in test builds, but the gateway binary should never use it
        #[cfg(test)]
        #[expect(unreachable_patterns)]
        _ => "test".to_string(),
    }
}

fn get_valkey_status_string(valkey: &ValkeyConnectionInfo) -> String {
    match valkey {
        ValkeyConnectionInfo::Disabled => "disabled".to_string(),
        ValkeyConnectionInfo::Enabled { .. } => "enabled".to_string(),
    }
}

pub async fn shutdown_signal() {
    // If any errors occur in these futures, we log them and return from the future
    // This will cause the `tokio::select!` block to resolve - i.e. we treat it as
    // though a shutdown signal was immediately received.
    let ctrl_c = async {
        let _ = signal::ctrl_c()
            .await
            .log_err_pretty("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        if let Ok(mut sig) = signal::unix::signal(signal::unix::SignalKind::terminate())
            .log_err_pretty("Failed to install SIGTERM handler")
        {
            sig.recv().await;
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    #[cfg(unix)]
    let hangup = async {
        if let Ok(mut sig) = signal::unix::signal(signal::unix::SignalKind::hangup())
            .log_err_pretty("Failed to install SIGHUP handler")
        {
            sig.recv().await;
        }
    };

    #[cfg(not(unix))]
    let hangup = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {
            tracing::info!("Received Ctrl+C signal");
        }
        () = terminate => {
            tracing::info!("Received SIGTERM signal");
        }
        () = hangup => {
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            tracing::info!("Received SIGHUP signal");
        }
    };
}

/// Spawn the durable worker if Postgres is configured.
///
/// The worker processes tasks from the durable queue. It starts whenever Postgres
/// is available, regardless of whether the autopilot API key is set. This allows
/// standalone durable tools (e.g. GEPA) to run without autopilot credentials.
async fn spawn_autopilot_worker_if_configured(
    gateway_handle: &gateway::GatewayHandle,
) -> Result<Option<AutopilotWorkerHandle>, ExitCode> {
    // Only start if Postgres is enabled (needed for durable task queue)
    let pool = match gateway_handle.app_state.postgres_connection_info() {
        PostgresConnectionInfo::Enabled { pool, .. } => pool.clone(),
        PostgresConnectionInfo::Disabled => {
            if std::env::var("TENSORZERO_AUTOPILOT_API_KEY").is_ok() {
                tracing::error!(
                    "`TENSORZERO_AUTOPILOT_API_KEY` is set, but Postgres is not enabled."
                );
                return Err(ExitCode::FAILURE);
            }
            return Ok(None);
        }
        #[cfg(test)]
        #[expect(unreachable_patterns)]
        _ => return Ok(None),
    };

    // Create an embedded TensorZero client using the gateway's state
    let t0_client =
        std::sync::Arc::new(EmbeddedClient::new(gateway_handle.app_state.load_latest()));

    // TODO: decide how we want to do autopilot config.
    let default_max_attempts = 5;
    let worker_options = WorkerOptions {
        poll_interval: Duration::from_secs(1),
        concurrency: 8,
        ..Default::default()
    };
    let config = AutopilotWorkerConfig::new(pool, t0_client, default_max_attempts, worker_options);

    Ok(Some(
        spawn_autopilot_worker(
            &gateway_handle.app_state.deferred_tasks,
            gateway_handle.app_state.shutdown_token.clone(),
            config,
        )
        .await
        .log_err_pretty("Failed to spawn autopilot worker")?,
    ))
}

/// ┌──────────────────────────────────────────────────────────────────────────┐
/// │                           MAIN.RS ESCAPE HATCH                           │
/// └──────────────────────────────────────────────────────────────────────────┘
///
/// We don't allow panic, escape, unwrap, or similar methods in the codebase,
/// except for the private `expect_pretty` method, which is to be used only in
/// main.rs during initialization. After initialization, we expect all code to
/// handle errors gracefully.
///
/// We use `expect_pretty` for better DX when handling errors in main.rs.
/// `expect_pretty` will print an error message and return an `ExitCode`.
/// This is propagated in `main` via `?`, so that we exit the process while still
/// running drop impls.
trait LogErrPretty<T> {
    fn log_err_pretty(self, msg: &str) -> Result<T, ExitCode>;
}

impl<T, E: Display> LogErrPretty<T> for Result<T, E> {
    fn log_err_pretty(self, msg: &str) -> Result<T, ExitCode> {
        match self {
            Ok(value) => Ok(value),
            Err(err) => {
                tracing::error!("{msg}: {err}");
                Err(ExitCode::FAILURE)
            }
        }
    }
}

impl<T> LogErrPretty<T> for Option<T> {
    fn log_err_pretty(self, msg: &str) -> Result<T, ExitCode> {
        match self {
            Some(value) => Ok(value),
            None => {
                tracing::error!("{msg}");
                Err(ExitCode::FAILURE)
            }
        }
    }
}

/// Trait for configuration glob information, so that we can create a mocked version in tests
trait ConfigGlobInfo {
    fn glob(&self) -> &str;
    fn paths(&self) -> &[std::path::PathBuf];
}

impl ConfigGlobInfo for tensorzero_core::config::ConfigFileGlob {
    fn glob(&self) -> &str {
        &self.glob
    }

    fn paths(&self) -> &[std::path::PathBuf] {
        &self.paths
    }
}

fn print_configuration_info(glob: Option<&impl ConfigGlobInfo>) {
    if let Some(glob) = glob {
        match glob.paths().len() {
            0 => {
                tracing::warn!(
                    "├ Configuration: glob `{}` did not match any files.",
                    glob.glob()
                );
            }
            _ => {
                tracing::info!("├ Configuration: glob `{}` resolved to:", glob.glob());

                for (i, path) in glob.paths().iter().enumerate() {
                    if i < glob.paths().len() - 1 {
                        tracing::info!("│ ├ {}", path.to_string_lossy());
                    } else {
                        tracing::info!("│ └ {}", path.to_string_lossy());
                    }
                }
            }
        }
    } else {
        tracing::info!("├ Configuration: default");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // Mock implementation for testing
    struct MockConfigGlob {
        glob: String,
        paths: Vec<PathBuf>,
    }

    impl ConfigGlobInfo for MockConfigGlob {
        fn glob(&self) -> &str {
            &self.glob
        }

        fn paths(&self) -> &[PathBuf] {
            &self.paths
        }
    }

    #[test]
    fn test_print_configuration_info_default() {
        let logs_contain = tensorzero_core::utils::testing::capture_logs();
        let glob: Option<&MockConfigGlob> = None;
        print_configuration_info(glob);

        assert!(logs_contain("├ Configuration: default"));
    }

    #[test]
    fn test_print_configuration_info_glob_no_matches() {
        let logs_contain = tensorzero_core::utils::testing::capture_logs();
        // Create a mock with no paths for testing
        let glob = MockConfigGlob {
            glob: "*.nonexistent".to_string(),
            paths: vec![],
        };

        print_configuration_info(Some(&glob));

        assert!(logs_contain(
            "├ Configuration: glob `*.nonexistent` did not match any files."
        ));
    }

    #[test]
    fn test_print_configuration_info_glob_single_path() {
        let logs_contain = tensorzero_core::utils::testing::capture_logs();
        let glob = MockConfigGlob {
            glob: "config/*.toml".to_string(),
            paths: vec![PathBuf::from("config/app.toml")],
        };

        print_configuration_info(Some(&glob));

        assert!(logs_contain(
            "├ Configuration: glob `config/*.toml` resolved to:"
        ));
        assert!(logs_contain("│ └ config/app.toml"));
    }

    #[test]
    fn test_print_configuration_info_glob_multiple_paths() {
        let logs_contain = tensorzero_core::utils::testing::capture_logs();
        let glob = MockConfigGlob {
            glob: "config/**/*.toml".to_string(),
            paths: vec![
                PathBuf::from("config/app.toml"),
                PathBuf::from("config/database.toml"),
                PathBuf::from("config/prod/settings.toml"),
            ],
        };

        print_configuration_info(Some(&glob));

        assert!(logs_contain(
            "├ Configuration: glob `config/**/*.toml` resolved to:"
        ));
        assert!(logs_contain("│ ├ config/app.toml"));
        assert!(logs_contain("│ ├ config/database.toml"));
        assert!(logs_contain("│ └ config/prod/settings.toml"));
    }
}
