use clap::{Args, Parser};
use futures::{FutureExt, StreamExt};
use mimalloc::MiMalloc;
use secrecy::ExposeSecret;
use std::fmt::Display;
use std::future::{Future, IntoFuture};
use std::io::ErrorKind;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::signal;
use tokio_stream::wrappers::IntervalStream;
use tower_http::metrics::in_flight_requests::InFlightRequestsCounter;

use tensorzero_auth::constants::{DEFAULT_ORGANIZATION, DEFAULT_WORKSPACE};
use tensorzero_core::config::{Config, ConfigFileGlob, ConfigLoadInfo};
use tensorzero_core::db::clickhouse::migration_manager::manual_run_clickhouse_migrations;
use tensorzero_core::db::postgres::{manual_run_postgres_migrations, PostgresConnectionInfo};
use tensorzero_core::endpoints::status::TENSORZERO_VERSION;
use tensorzero_core::error;
use tensorzero_core::observability::{self, LogFormat};
use tensorzero_core::utils::gateway;

mod router;
mod routes;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Parser, Debug)]
#[command(version, about)]
struct GatewayArgs {
    /// Use all of the config files matching the specified glob pattern. Incompatible with `--default-config`
    #[arg(long)]
    config_file: Option<PathBuf>,

    /// Use a default config file. Incompatible with `--config-file`
    #[arg(long)]
    default_config: bool,

    /// Sets the log format used for all gateway logs.
    #[arg(long)]
    #[arg(value_enum)]
    #[clap(default_value_t = LogFormat::default())]
    log_format: LogFormat,

    #[command(flatten)]
    migration_commands: MigrationCommands,
}

#[derive(Args, Debug)]
#[group(multiple = false)]
struct MigrationCommands {
    /// Run ClickHouse migrations manually then exit.
    // TODO: remove
    #[arg(long, alias = "run-migrations")]
    run_clickhouse_migrations: bool,

    /// Run PostgreSQL migrations manually then exit.
    #[arg(long)]
    run_postgres_migrations: bool,

    /// Create an API key then exit.
    #[arg(long)]
    create_api_key: bool,
}

#[expect(clippy::print_stdout)]
fn print_key(key: &secrecy::SecretString) {
    println!("{}", key.expose_secret());
}

async fn handle_create_api_key() -> Result<(), Box<dyn std::error::Error>> {
    // Read the Postgres URL from the environment
    let postgres_url = std::env::var("TENSORZERO_POSTGRES_URL")
        .map_err(|_| "TENSORZERO_POSTGRES_URL environment variable not set")?;

    // Create connection pool (alpha version for tensorzero-auth)
    let pool = sqlx::PgPool::connect(&postgres_url).await?;

    // Create the key with default organization and workspace
    let key =
        tensorzero_auth::postgres::create_key(DEFAULT_ORGANIZATION, DEFAULT_WORKSPACE, None, &pool)
            .await?;

    // Print only the API key to stdout for easy machine parsing
    print_key(&key);

    Ok(())
}

#[tokio::main]
async fn main() {
    let args = GatewayArgs::parse();
    // Set up logs and metrics immediately, so that we can use `tracing`.
    // OTLP will be enabled based on the config file
    // We start with empty headers and update them after loading the config
    let delayed_log_config = observability::setup_observability(args.log_format)
        .await
        .expect_pretty("Failed to set up logs");

    let git_sha = tensorzero_core::built_info::GIT_COMMIT_HASH_SHORT.unwrap_or("unknown");

    if args.migration_commands.create_api_key {
        handle_create_api_key()
            .await
            .expect_pretty("Failed to create API key");
        return;
    }

    if args.migration_commands.run_clickhouse_migrations {
        manual_run_clickhouse_migrations()
            .await
            .expect_pretty("Failed to run ClickHouse migrations");
        tracing::info!("ClickHouse is ready.");
        return;
    }

    if args.migration_commands.run_postgres_migrations {
        manual_run_postgres_migrations()
            .await
            .expect_pretty("Failed to run PostgreSQL migrations");
        tracing::info!("Postgres is ready.");
        return;
    }

    tracing::info!("Starting TensorZero Gateway {TENSORZERO_VERSION} (commit: {git_sha})");

    let metrics_handle = observability::setup_metrics().expect_pretty("Failed to set up metrics");

    // Handle `--config-file` or `--default-config`
    let (config_load_info, glob) = match (args.default_config, args.config_file) {
        (true, Some(_)) => {
            tracing::error!("You must not specify both `--config-file` and `--default-config`.");
            std::process::exit(1);
        }
        (false, None) => {
            tracing::error!("You must specify either `--config-file` or `--default-config`.");
            std::process::exit(1);
        }
        (true, None) => {
            tracing::warn!("No config file provided, so only default functions will be available. Use `--config-file path/to/tensorzero.toml` to specify a config file.");
            (
                Config::new_empty()
                    .await
                    .expect_pretty("Failed to load default config"),
                None,
            )
        }
        (false, Some(path)) => {
            let glob = ConfigFileGlob::new_from_path(&path)
                .expect_pretty("Failed to process config file glob");
            (

                    Config::load_and_verify_from_path(&glob)
                        .await
                        .ok() // Don't print the error here, since it was already printed when it was constructed
                        .expect_pretty(&format!(
                            "Failed to load config. Config file glob `{}` resolved to the following files:\n{}",
                            glob.glob,
                            glob.paths.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join("\n")
                        )),
                Some(glob),
            )
        }
    };
    let ConfigLoadInfo {
        config,
        snapshot: _, // TODO: write the snapshot
    } = config_load_info;
    let config = Arc::new(config);

    if config.gateway.debug {
        delayed_log_config
            .delayed_debug_logs
            .enable_debug()
            .expect_pretty("Failed to enable debug logs");
    }

    // Note: We only enable OTLP after config file parsing/loading is complete,
    // so that the config file can control whether OTLP is enabled or not.
    // This means that any tracing spans created before this point will not be exported to OTLP.
    // For now, this is fine, as we only ever export spans for inference/batch/feedback requests,
    // which cannot have occurred up until this point.
    // If we ever want to emit earlier OTLP spans, we'll need to come up with a different way
    // of doing OTLP initialization (e.g. buffer spans, and submit them once we know if OTLP should be enabled).
    // See `build_opentelemetry_layer` for the details of exactly what spans we export.
    if config.gateway.export.otlp.traces.enabled {
        if std::env::var("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT").is_err() {
            // This makes it easier to run the gateway in local development and CI
            if cfg!(feature = "e2e_tests") {
                tracing::warn!("Running without explicit `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` environment variable in e2e tests mode.");
            } else {
                tracing::error!("The `gateway.export.otlp.traces.enabled` configuration option is `true`, but environment variable `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` is not set. Please set it to the OTLP endpoint (e.g. `http://localhost:4317`).");
                std::process::exit(1);
            }
        }

        // Set config-level OTLP headers if we have a tracer wrapper
        if let Some(ref tracer_wrapper) = delayed_log_config.otel_tracer {
            if !config.gateway.export.otlp.traces.extra_headers.is_empty() {
                tracer_wrapper
                    .set_static_otlp_traces_extra_headers(
                        &config.gateway.export.otlp.traces.extra_headers,
                    )
                    .expect_pretty("Failed to set OTLP config headers");
            }
        }

        match delayed_log_config.delayed_otel {
            Ok(delayed_otel) => {
                delayed_otel
                    .enable_otel()
                    .expect_pretty("Failed to enable OpenTelemetry");
            }
            Err(e) => {
                tracing::error!(
                    "Could not enable OpenTelemetry export due to previous error: `{e}`. Exiting."
                );
                std::process::exit(1);
            }
        }
    } else if let Err(e) = delayed_log_config.delayed_otel {
        tracing::warn!(
            "[gateway.export.otlp.traces.enabled] is `false`, so ignoring OpenTelemetry error: `{e}`"
        );
    }

    // Initialize GatewayHandle
    let gateway_handle = gateway::GatewayHandle::new(config.clone())
        .await
        .expect_pretty("Failed to initialize AppState");

    // Create a new observability_enabled_pretty string for the log message below
    let postgres_enabled_pretty =
        get_postgres_status_string(&gateway_handle.app_state.postgres_connection_info);

    // Set debug mode
    error::set_debug(config.gateway.debug).expect_pretty("Failed to set debug mode");
    error::set_unstable_error_json(config.gateway.unstable_error_json)
        .expect_pretty("Failed to set unstable error JSON");

    let base_path = config.gateway.base_path.as_deref().unwrap_or("/");
    if !base_path.starts_with("/") {
        tracing::error!("[gateway.base_path] must start with a `/` : `{base_path}`");
        std::process::exit(1);
    }
    let base_path = base_path.trim_end_matches("/");

    let (router, in_flight_requests_counter) = router::build_axum_router(
        base_path,
        delayed_log_config.otel_tracer.clone(),
        gateway_handle.app_state.clone(),
        metrics_handle,
    );

    // Bind to the socket address specified in the config, or default to 0.0.0.0:3000
    let bind_address = config
        .gateway
        .bind_address
        .unwrap_or_else(|| SocketAddr::from(([0, 0, 0, 0], 3000)));

    let listener = match tokio::net::TcpListener::bind(bind_address).await {
        Ok(listener) => listener,
        Err(e) if e.kind() == ErrorKind::AddrInUse => {
            tracing::error!(
                "Failed to bind to socket address {bind_address}: {e}. Tip: Ensure no other process is using port {} or try a different port.",
                bind_address.port()
            );
            std::process::exit(1);
        }
        Err(e) => {
            tracing::error!("Failed to bind to socket address {bind_address}: {e}");
            std::process::exit(1);
        }
    };

    // This will give us the chosen port if the user specified a port of 0
    let actual_bind_address = listener
        .local_addr()
        .expect_pretty("Failed to get bind address from listener");

    // Print the bind address
    tracing::info!("TensorZero Gateway is listening on {actual_bind_address}");

    // Print the base path if set
    if base_path.is_empty() {
        tracing::info!("├ API Base Path: /");
    } else {
        tracing::info!("├ API Base Path: {base_path}");
    }

    // Print the configuration being used
    print_configuration_info(glob.as_ref());

    // Print whether observability is enabled
    tracing::info!(
        "├ Observability (ClickHouse): {}",
        gateway_handle.app_state.clickhouse_connection_info
    );
    if config.gateway.observability.batch_writes.enabled {
        tracing::info!(
            "├ Batch Writes: enabled (flush_interval_ms = {}, max_rows = {})",
            config.gateway.observability.batch_writes.flush_interval_ms,
            config.gateway.observability.batch_writes.max_rows
        );
    } else {
        tracing::info!("├ Batch Writes: disabled");
    }

    // Print whether postgres is enabled
    tracing::info!("├ Postgres: {postgres_enabled_pretty}");

    // Print whether OpenTelemetry is enabled
    if config.gateway.export.otlp.traces.enabled {
        tracing::info!("└ OpenTelemetry: enabled");
    } else {
        tracing::info!("└ OpenTelemetry: disabled");
    }

    let shutdown_signal = shutdown_signal().shared();

    let server_fut = axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal.clone())
        .into_future()
        .map(|r| r.expect_pretty("Failed to start server"))
        .shared();

    // This is a purely informational logging task, so we don't need to wait for it to finish.
    #[expect(clippy::disallowed_methods)]
    tokio::spawn(monitor_server_shutdown(
        shutdown_signal,
        server_fut.clone(),
        in_flight_requests_counter,
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
}

/// A background task that waits for the server shutdown to initiate, and then logs status information every 5 seconds until
/// the server completes its shutdown.
async fn monitor_server_shutdown(
    shutdown_signal: impl Future<Output = ()>,
    server_fut: impl Future<Output = ()>,
    in_flight_requests_counter: InFlightRequestsCounter,
) {
    // First, wait for the shutdown signal
    shutdown_signal.await;
    // The server should now be shutting down, so print a message every 5 seconds until it completes
    IntervalStream::new(tokio::time::interval(Duration::from_secs(5)))
        .take_until(server_fut)
        .for_each(|_| async {
            tracing::info!(
                "Server shutdown in progress: {} in-flight requests remaining",
                in_flight_requests_counter.get()
            );
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

pub async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect_pretty("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect_pretty("Failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    #[cfg(unix)]
    let hangup = async {
        signal::unix::signal(signal::unix::SignalKind::hangup())
            .expect_pretty("Failed to install SIGHUP handler")
            .recv()
            .await;
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
/// `expect_pretty` will print an error message and exit with a status code of 1.
trait ExpectPretty<T> {
    fn expect_pretty(self, msg: &str) -> T;
}

impl<T, E: Display> ExpectPretty<T> for Result<T, E> {
    fn expect_pretty(self, msg: &str) -> T {
        match self {
            Ok(value) => value,
            Err(err) => {
                tracing::error!("{msg}: {err}");
                std::process::exit(1);
            }
        }
    }
}

impl<T> ExpectPretty<T> for Option<T> {
    fn expect_pretty(self, msg: &str) -> T {
        match self {
            Some(value) => value,
            None => {
                tracing::error!("{msg}");
                std::process::exit(1);
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
