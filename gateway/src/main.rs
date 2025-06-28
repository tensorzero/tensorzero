use axum::extract::{DefaultBodyLimit, Request};
use axum::http::HeaderValue;
use axum::middleware::Next;
use axum::response::Response;
use axum::routing::{delete, get, post, put};
use axum::Router;
use clap::Parser;
use mimalloc::MiMalloc;
use std::fmt::Display;
use std::io::ErrorKind;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::signal;
use tower_http::trace::{DefaultOnFailure, TraceLayer};
use tracing::Level;

use tensorzero_internal::auth::require_api_key;
use tensorzero_internal::clickhouse::ClickHouseConnectionInfo;
use tensorzero_internal::config_parser::Config;
use tensorzero_internal::endpoints;
use tensorzero_internal::endpoints::status::TENSORZERO_VERSION;
use tensorzero_internal::error;
use tensorzero_internal::gateway_util::{self, AuthenticationInfo};
use tensorzero_internal::observability::{self, LogFormat, RouterExt};
use tensorzero_internal::redis_client::RedisClient;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// Use the `tensorzero.toml` config file at the specified path. Incompatible with `--default-config`
    #[arg(long)]
    config_file: Option<PathBuf>,

    /// Use a default config file. Incompatible with `--config-file`
    #[arg(long)]
    default_config: bool,

    // Hidden flag used by our `Dockerfile` to warn users who have not overridden the default CMD
    #[arg(long)]
    #[clap(hide = true)]
    warn_default_cmd: bool,

    /// Sets the log format used for all gateway logs.
    #[arg(long)]
    #[arg(value_enum)]
    #[clap(default_value_t = LogFormat::default())]
    log_format: LogFormat,

    /// Deprecated: use `--config-file` instead
    tensorzero_toml: Option<PathBuf>,
}

async fn add_version_header(request: Request, next: Next) -> Response {
    #[cfg_attr(not(feature = "e2e_tests"), expect(unused_mut))]
    let mut version = HeaderValue::from_static(TENSORZERO_VERSION);

    #[cfg(feature = "e2e_tests")]
    {
        if request
            .headers()
            .contains_key("x-tensorzero-e2e-version-remove")
        {
            tracing::info!("Removing version header due to e2e header");
            return next.run(request).await;
        }
        if let Some(header_version) = request.headers().get("x-tensorzero-e2e-version-override") {
            tracing::info!("Overriding version header with e2e header: {header_version:?}");
            version = header_version.clone();
        }
    }

    let mut response = next.run(request).await;
    response
        .headers_mut()
        .insert("x-tensorzero-gateway-version", version);
    response
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    // Set up logs and metrics immediately, so that we can use `tracing`.
    // OTLP will be enabled based on the config file
    let delayed_log_config = observability::setup_observability(args.log_format)
        .await
        .expect_pretty("Failed to set up logs");

    let git_sha = tensorzero_internal::built_info::GIT_COMMIT_HASH_SHORT.unwrap_or("unknown");

    tracing::info!("Starting TensorZero Gateway {TENSORZERO_VERSION} (commit: {git_sha})");

    let metrics_handle = observability::setup_metrics().expect_pretty("Failed to set up metrics");

    if args.warn_default_cmd {
        tracing::warn!("Deprecation Warning: Running gateway from Docker container without overriding default CMD. Please override the command to either `--config-file` to specify a custom configuration file (e.g. `--config-file /path/to/tensorzero.toml`) or `--default-config` to use default settings (i.e. no custom functions, metrics, etc.).");
    }

    if args.tensorzero_toml.is_some() && args.config_file.is_some() {
        tracing::error!("Cannot specify both `--config-file` and a positional path argument");
        std::process::exit(1);
    }

    if args.tensorzero_toml.is_some() {
        tracing::warn!(
            "`Specifying a positional path argument is deprecated. Use `--config-file path/to/tensorzero.toml` instead."
        );
    }

    let config_path = args.config_file.or(args.tensorzero_toml);

    if config_path.is_some() && args.default_config {
        tracing::error!("Cannot specify both `--config-file` and `--default-config`");
        std::process::exit(1);
    }

    if !args.default_config && config_path.is_none() {
        tracing::warn!("Running the gateway without any config-related arguments is deprecated. Use `--default-config` to start the gateway with the default config.");
    }

    let config = if let Some(path) = &config_path {
        Arc::new(
            Config::load_and_verify_from_path(Path::new(&path))
                .await
                .ok() // Don't print the error here, since it was already printed when it was constructed
                .expect_pretty("Failed to load config"),
        )
    } else {
        tracing::warn!("No config file provided, so only default functions will be available. Use `--config-file path/to/tensorzero.toml` to specify a config file.");
        Arc::new(Config::default())
    };

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

        delayed_log_config
            .delayed_otel
            .enable_otel()
            .expect_pretty("Failed to enable OpenTelemetry");

        tracing::info!("Enabled OpenTelemetry OTLP export");
    }

    // Initialize AppState
    let app_state = gateway_util::AppStateData::new(config.clone())
        .await
        .expect_pretty("Failed to initialize AppState");

    // Setup Redis client if authentication is enabled
    if let AuthenticationInfo::Enabled(ref auth) = app_state.authentication_info {
        if let Ok(redis_url) = std::env::var("TENSORZERO_REDIS_URL") {
            if !redis_url.is_empty() {
                let redis_client =
                    RedisClient::new(&redis_url, app_state.clone(), auth.clone()).await;
                redis_client
                    .start()
                    .await
                    .expect_pretty("Failed to start Redis client");
            } else {
                tracing::warn!(
                    "TENSORZERO_REDIS_URL is empty, so Redis client will not be started"
                );
            }
        }
    }

    // Create a new observability_enabled_pretty string for the log message below
    let observability_enabled_pretty = match &app_state.clickhouse_connection_info {
        ClickHouseConnectionInfo::Disabled => "disabled".to_string(),
        ClickHouseConnectionInfo::Mock { healthy, .. } => {
            format!("mocked (healthy={healthy})")
        }
        ClickHouseConnectionInfo::Production { database, .. } => {
            format!("enabled (database: {database})")
        }
    };

    // Create authentication status string for logging
    let authentication_enabled_pretty = match &app_state.authentication_info {
        AuthenticationInfo::Disabled => "disabled",
        AuthenticationInfo::Enabled(_) => "enabled",
    };

    // Set debug mode
    error::set_debug(config.gateway.debug).expect_pretty("Failed to set debug mode");

    // OpenAI-compatible routes (conditionally authenticated)
    let openai_routes = Router::new()
        .route(
            "/v1/chat/completions",
            post(endpoints::openai_compatible::inference_handler),
        )
        .route(
            "/v1/embeddings",
            post(endpoints::openai_compatible::embedding_handler),
        )
        .route(
            "/v1/moderations",
            post(endpoints::openai_compatible::moderation_handler),
        );

    // Apply authentication middleware only if authentication is enabled
    let openai_routes = match &app_state.authentication_info {
        AuthenticationInfo::Enabled(auth) => openai_routes.layer(
            axum::middleware::from_fn_with_state(auth.clone(), require_api_key),
        ),
        AuthenticationInfo::Disabled => openai_routes,
    };

    // Routes that don't require authentication
    let public_routes = Router::new()
        .route("/inference", post(endpoints::inference::inference_handler))
        .route(
            "/batch_inference",
            post(endpoints::batch_inference::start_batch_inference_handler),
        )
        .route(
            "/batch_inference/{batch_id}",
            get(endpoints::batch_inference::poll_batch_inference_handler),
        )
        .route(
            "/batch_inference/{batch_id}/inference/{inference_id}",
            get(endpoints::batch_inference::poll_batch_inference_handler),
        )
        .route("/feedback", post(endpoints::feedback::feedback_handler))
        // Everything above this layer has OpenTelemetry tracing enabled
        // Note - we do *not* attach a `OtelInResponseLayer`, as this seems to be incorrect according to the W3C Trace Context spec
        // (the only response header is `traceresponse` for a completed trace)
        .apply_otel_http_trace_layer()
        // Everything below the Otel layers does not have OpenTelemetry tracing enabled
        .route(
            "/datasets/{dataset_name}/datapoints/bulk",
            post(endpoints::datasets::bulk_insert_datapoints_handler),
        )
        .route(
            "/datasets/{dataset_name}/datapoints/{datapoint_id}",
            delete(endpoints::datasets::delete_datapoint_handler),
        )
        .route(
            "/datasets/{dataset_name}/datapoints",
            get(endpoints::datasets::list_datapoints_handler),
        )
        .route(
            "/datasets/{dataset_name}/datapoints/{datapoint_id}",
            get(endpoints::datasets::get_datapoint_handler),
        )
        .route(
            "/internal/datasets/{dataset_name}/datapoints",
            post(endpoints::datasets::insert_from_existing_datapoint_handler),
        )
        .route(
            "/internal/datasets/{dataset_name}/datapoints/{datapoint_id}",
            put(endpoints::datasets::update_datapoint_handler),
        )
        .route(
            "/internal/object_storage",
            get(endpoints::object_storage::get_object_handler),
        )
        .route(
            "/dynamic_evaluation_run",
            post(endpoints::dynamic_evaluation_run::dynamic_evaluation_run_handler),
        )
        .route(
            "/dynamic_evaluation_run/{run_id}/episode",
            post(endpoints::dynamic_evaluation_run::dynamic_evaluation_run_episode_handler),
        )
        .route("/status", get(endpoints::status::status_handler))
        .route("/health", get(endpoints::status::health_handler))
        .route(
            "/metrics",
            get(move || std::future::ready(metrics_handle.render())),
        );

    let router = Router::new()
        .merge(openai_routes)
        .merge(public_routes)
        .fallback(endpoints::fallback::handle_404)
        .layer(axum::middleware::from_fn(add_version_header))
        .layer(DefaultBodyLimit::max(100 * 1024 * 1024)) // increase the default body limit from 2MB to 100MB
        // Note - this is intentionally *not* used by our OTEL exporter (it creates a span without any `http.` or `otel.` fields)
        // This is only used to output request/response information to our logs
        // OTEL exporting is done by the `OtelAxumLayer` above, which is only enabled for certain routes (and includes much more information)
        // We log failed requests messages at 'DEBUG', since we already have our own error-logging code,
        .layer(TraceLayer::new_for_http().on_failure(DefaultOnFailure::new().level(Level::DEBUG)))
        .with_state(app_state);

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

    let config_path_pretty = if let Some(path) = &config_path {
        format!("config file `{}`", path.to_string_lossy())
    } else {
        "no config file".to_string()
    };

    tracing::info!(
        "TensorZero Gateway is listening on {actual_bind_address} with {config_path_pretty}, observability {observability_enabled_pretty}, and authentication {authentication_enabled_pretty}.",
    );

    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .expect_pretty("Failed to start server");
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
        _ = ctrl_c => {
            tracing::info!("Received Ctrl+C signal");
        }
        _ = terminate => {
            tracing::info!("Received SIGTERM signal");
        }
        _ = hangup => {
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
