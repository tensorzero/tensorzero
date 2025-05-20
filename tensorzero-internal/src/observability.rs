use axum::extract::MatchedPath;
use axum::Router;
use clap::ValueEnum;
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use opentelemetry::trace::TracerProvider as _;
use opentelemetry::KeyValue;
use opentelemetry_sdk::trace::{SdkTracerProvider, SpanExporter};
use opentelemetry_sdk::Resource;
use tower_http::trace::TraceLayer;
use tracing::level_filters::LevelFilter;
use tracing::Span;
use tracing_opentelemetry::OpenTelemetrySpanExt;
use tracing_subscriber::layer::Filter;
use tracing_subscriber::{filter, EnvFilter, Registry};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, Layer};

use crate::error::{Error, ErrorDetails};

#[derive(Clone, Debug, Default, ValueEnum)]
pub enum LogFormat {
    #[default]
    Pretty,
    Json,
}

// Builds the internal OpenTelemetry layer, without any filtering applied.
fn internal_build_otel_layer<T: SpanExporter + 'static>(
    override_exporter: Option<T>,
) -> Result<impl Layer<Registry>, Error> {
    let mut provider = SdkTracerProvider::builder().with_resource(
        Resource::builder_empty()
            .with_attribute(KeyValue::new(
                opentelemetry_semantic_conventions::resource::SERVICE_NAME,
                "tensorzero-gateway",
            ))
            .build(),
    );

    if let Some(exporter) = override_exporter {
        provider = provider.with_simple_exporter(exporter)
    } else {
        provider = provider.with_batch_exporter(
            opentelemetry_otlp::SpanExporter::builder()
                .with_tonic()
                .build()
                .map_err(|e| {
                    Error::new(ErrorDetails::Observability {
                        message: format!("Failed to create OTLP exporter: {e}"),
                    })
                })?,
        );
    }
    let provider = provider.build();
    let tracer = provider.tracer("tensorzero");
    opentelemetry::global::set_tracer_provider(provider.clone());
    Ok(tracing_opentelemetry::layer()
        .with_tracer(tracer)
        .with_level(true))
}

/// Creates an OpenTelemetry export layer. This layer is disabled by default,
/// and can be dynamically enabled using the returned `DelayedOtelEnableHandle`.
/// See the `DelayedOtelEnableHandle` docs for more details.
///
/// The `override_exporter` parameter can be used to prove a custom `SpanExporter`.
/// This is used by `install_capturing_otel_exporter` during e2e tests to capture
/// all emitted spans.
///
/// If `override_exporter` is `None`, the default OTLP exporter will be used,
/// which is configured via environment variables (e..g `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`):
/// https://github.com/open-telemetry/opentelemetry-specification/blob/main/specification/protocol/exporter.md#endpoint-urls-for-otlphttp
pub fn build_opentelemetry_layer<T: SpanExporter + 'static>(
    override_exporter: Option<T>,
) -> Result<(DelayedOtelEnableHandle, impl Layer<Registry>), Error> {
    let (otel_reload_filter, reload_handle) = tracing_subscriber::reload::Layer::new(Box::new(
        LevelFilter::OFF,
    )
        as Box<dyn Filter<_> + Send + Sync>);

    let base_otel_layer = internal_build_otel_layer(override_exporter)?;

    let delayed_enable = DelayedOtelEnableHandle {
        enable_cb: Box::new(move || {
            // Only register the propagator if we actually enabled OTEL.
            // This means that the `traceparent` and `tracestate` headers will only be added
            // to outgoing requests using the propagator if OTEL is actually enabled.
            init_tracing_opentelemetry::init_propagator().map_err(|e| {
                Error::new(ErrorDetails::Observability {
                    message: format!("Failed to initialize OTLP propagator: {e}"),
                })
            })?;
            // Avoid exposing all of our internal spans, as we don't want customers to start depending on them.
            // We only expose spans that explicitly contain field prefixed with "http." or "otel."
            // For example, `#[instrument(fields(otel.name = "my_otel_name"))]` will be exported
            let filter = filter::filter_fn(|metadata| {
                metadata.fields().iter().any(|field| {
                    field.name().starts_with("http.") || field.name().starts_with("otel.")
                })
            });

            reload_handle
                .modify(|l| {
                    *l = Box::new(filter);
                })
                .map_err(|e| {
                    Error::new(ErrorDetails::Observability {
                        message: format!("Failed to enable OTLP exporter: {e}"),
                    })
                })?;
            Ok(())
        }),
    };
    Ok((
        delayed_enable,
        // Note - we *must* use the `tracing_opentelemetry` (without it being wrapped in a reloadable layer)
        // due to https://github.com/tokio-rs/tracing-opentelemetry/issues/121
        // We attach a reloadable filter, which we use to start exporting spans when `delayed_enable` is called.
        // This means that we unconditionally construct the `tracing_opentelemetry` layer,
        // (including the batch exporter), which will just end being unused if OTEL exporting is disabled.
        base_otel_layer.with_filter(otel_reload_filter),
    ))
}

/// A helper trait to apply layers to a `Router`.
/// Without this trait, we would need to write something like:
/// ```rust
/// // in `tensorzero-internal`
/// fn make_my_layer() -> SomeType
///
/// // in `tensorzero-gateway`
/// router.layer(tensorzero_internal::make_my_layer())
/// ```
///
/// However, writing the return type for `make_my_layer` can be very complicated
/// due to all of the generic bounds used by `axum` and `tower`.
///
/// To make things simpler, we define a helper trait, which allows us to call
/// functions on our `router`. Inside the helper method, we can call
/// `router.layer(some_layer)` as if we were writing everything inline
/// in `tensorzero-gateway`, without every needing to name a return type.
pub trait RouterExt<S> {
    fn apply_otel_http_trace_layer(self) -> Self;
}

impl<S: Clone + Send + Sync + 'static> RouterExt<S> for Router<S> {
    /// Makes a `TraceLayer` specialized for OpenTelemetry traces.
    /// This is only applied to certain routes (e.g. `/inference`), and
    /// is *not* used to log requests to the console.
    fn apply_otel_http_trace_layer(self) -> Self {
        fn make_span<B>(req: &http::Request<B>) -> Span {
            // Based on `OtelAxumLayer`. We use `TraceLayer` from `tower-http` instead, as it extends
            // the span to cover the entire response (including sending the full SSE stream).
            let span =
                tracing_opentelemetry_instrumentation_sdk::http::http_server::make_span_from_request(
                    req,
                );
            let route = req.extensions().get::<MatchedPath>().map(|mp| mp.as_str());
            if let Some(route) = route {
                span.record("http.route", route);
            }
            let method = tracing_opentelemetry_instrumentation_sdk::http::http_method(req.method());
            span.record(
                "otel.name",
                format!("{method} {}", route.unwrap_or_default()).trim(),
            );
            span.set_parent(
                tracing_opentelemetry_instrumentation_sdk::http::extract_context(req.headers()),
            );
            span
        }
        self.layer(TraceLayer::new_for_http().make_span_with(make_span))
    }
}

/// A handle produced by `build_opentelemetry_layer` to allow enabling the OTEL layer
/// after tracing as been initialized.
/// Background: During gateway initialization, we need to:
/// * Set up the global tracing subscriber
/// * Log some startup info (e.g. version, git hash)
/// * Try to load and parse the config file from disk
///
/// The config file is responsible for controlling whether OTEL is enabled,
/// but we want to use `tracing` before and during config file parsing.
///
/// The solution is to use `tracing_subscriber::reload` to create a reloadable layer.
/// a wrapped layer, which can later be enabled based on the config file value.
/// The gateway unconditionally registers the layer returned by `build_opentelemetry_layer`,
/// and later determines whether to call `enable_otel` based on the config file.
pub struct DelayedOtelEnableHandle {
    enable_cb: Box<dyn FnOnce() -> Result<(), Error> + Send + Sync>,
}

impl DelayedOtelEnableHandle {
    pub fn enable_otel(self) -> Result<(), Error> {
        (self.enable_cb)()
    }
}

pub struct DelayedDebugLogs {
    enable_cb: Box<dyn FnOnce() -> Result<(), Error> + Send + Sync>,
}

impl DelayedDebugLogs {
    pub fn enable_debug(self) -> Result<(), Error> {
        (self.enable_cb)()
    }
}

/// A handle produced by `setup_logs` that allows updating some configuration values after logging has been initialized.
/// This allows us to use the following pattern in the gateway:
/// 1. Enable logging with some default (verbose) settings
/// 2. Deserialize the config file (`tracing::*` macros will work at this point)
/// 3. Update the logging configuration based on the deserialized config file (e.g. `gateway.debug = true`)
pub struct DelayedLogConfig {
    pub delayed_otel: DelayedOtelEnableHandle,
    pub delayed_debug_logs: DelayedDebugLogs,
}

/// This is used when `gateway.debug` is `false` and `RUST_LOG` is not set
const DEFAULT_GATEWAY_NON_DEBUG_DIRECTIVES: &str = "warn,gateway=info,tensorzero_internal=info";
/// This is used when `gateway.debug` is `true` and `RUST_LOG` is not set
const DEFAULT_GATEWAY_DEBUG_DIRECTIVES: &str =
    "warn,gateway=debug,tensorzero_internal=debug,tower_http::trace=debug";

/// Set up logging (including the necessary layers for OpenTelemtry exporting)
///
/// This does *not* actually enable OTEL exporting - you must use the returned
/// `DelayedOtelEnableHandle` to turn on exporting. This two-step approach is
/// needed because we need to initialize the tracing Registry before parsing
/// the config file (so that we can log errors during config file parsing),
/// but the parsed config file determines whether OTEL is enabled.
///
/// The priority for our logging configuration is:
/// 1. If `RUST_LOG` is set, use it verbatim, ignoring everything else
/// 2. If `gateway.debug` is set in the config file, use `DEFAULT_GATEWAY_DEBUG_DIRECTIVES`
/// 3. Otherwise, use `DEFAULT_GATEWAY_NON_DEBUG_DIRECTIVES`
///
/// The case of unset `RUST_LOG` and `gateway.debug = true` is special:
/// We initialize our filter with `DEFAULT_GATEWAY_NON_DEBUG_DIRECTIVES`,
/// and then later override it (with `DelayedDebugLogs`) to `DEFAULT_GATEWAY_DEBUG_DIRECTIVES`
/// This allows us to still see warnings/errors that occur during config file parsing.
///
/// In all other cases, the filter is set once during initialization, and then never changed.
///
/// Strictly speaking, this does not need to be an async function.
/// However, the call to `build_opentelemetry_layer` requires a Tokio runtime,
/// so marking this function as async makes it clear to callers that they need to
/// be in an async context.
pub async fn setup_observability(log_format: LogFormat) -> Result<DelayedLogConfig, Error> {
    let env_var_name = "RUST_LOG";
    let has_env_var = std::env::var(env_var_name).is_ok();

    let default_debug_filter = EnvFilter::builder()
        .parse(DEFAULT_GATEWAY_DEBUG_DIRECTIVES)
        .map_err(|e| {
            Error::new(ErrorDetails::InternalError {
                message: format!(
                    "Failed to parse internal debug directives - this should never happen: {e}"
                ),
            })
        })?;

    // If the `RUST_LOG` env var is set, then use it as our filter.
    // Otherwise, use the default non-debug directives (which might later get overridden to DEFAULT_GATEWAY_DEBUG_DIRECTIVES
    // using the `update_log_level` handle).
    let base_filter = if has_env_var {
        EnvFilter::builder()
            .with_env_var(env_var_name)
            .from_env()
            .map_err(|e| {
                Error::new(ErrorDetails::Observability {
                    message: format!("Invalid `{env_var_name}` environment variable: {e}"),
                })
            })?
    } else {
        EnvFilter::builder()
            .parse(DEFAULT_GATEWAY_NON_DEBUG_DIRECTIVES)
            .map_err(|e| {
                Error::new(ErrorDetails::InternalError {
                    message: format!("Failed to parse internal non-debug directives - this should never happen: {e}"),
                })
            })?
    };

    let (log_level, update_log_level) = tracing_subscriber::reload::Layer::new(base_filter);

    let log_layer = match log_format {
        LogFormat::Pretty => {
            Box::new(tracing_subscriber::fmt::layer()) as Box<dyn Layer<_> + Send + Sync>
        }
        LogFormat::Json => Box::new(tracing_subscriber::fmt::layer().json()),
    };

    // We need to provide a dummy generic parameter to satisfy the compiler
    let (delayed_otel, otel_layer) =
        build_opentelemetry_layer::<opentelemetry_otlp::SpanExporter>(None)?;

    tracing_subscriber::registry()
        .with(otel_layer)
        .with(log_layer.with_filter(log_level))
        .init();

    // If `RUST_LOG` is explicitly set, it takes precedence over `gateway.debug`,
    // so we return a no-op `DelayedDebugLogs` handle.
    let delayed_debug_logs = if has_env_var {
        DelayedDebugLogs {
            enable_cb: Box::new(|| Ok(())),
        }
    } else {
        DelayedDebugLogs {
            enable_cb: Box::new(move || {
                update_log_level
                    .modify(move |l| {
                        *l = default_debug_filter;
                    })
                    .map_err(|e| {
                        Error::new(ErrorDetails::Observability {
                            message: format!("Failed to update log level: {e}"),
                        })
                    })
            }),
        }
    };
    Ok(DelayedLogConfig {
        delayed_otel,
        delayed_debug_logs,
    })
}

/// Set up Prometheus metrics exporter
pub fn setup_metrics() -> Result<PrometheusHandle, Error> {
    PrometheusBuilder::new().install_recorder().map_err(|e| {
        Error::new(ErrorDetails::Observability {
            message: format!("Failed to install Prometheus exporter: {e}"),
        })
    })
}
