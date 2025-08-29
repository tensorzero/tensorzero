use std::hash::Hasher;
use std::str::FromStr;

use axum::extract::MatchedPath;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::{middleware, Router};
use clap::ValueEnum;
use http::HeaderMap;
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use moka::sync::Cache;
use opentelemetry::trace::{Tracer, TracerProvider as _};
use opentelemetry::{Context, KeyValue};
use opentelemetry_otlp::tonic_types::metadata::MetadataMap;
use opentelemetry_otlp::WithTonicConfig;
use opentelemetry_sdk::trace::SdkTracer;
use opentelemetry_sdk::trace::{SdkTracerProvider, SpanExporter};
use opentelemetry_sdk::Resource;
use std::hash::Hash;
use tokio_util::task::TaskTracker;
use tonic::metadata::{AsciiMetadataKey, MetadataValue};
use tower_http::trace::TraceLayer;
use tracing::level_filters::LevelFilter;
use tracing::Span;
use tracing_opentelemetry::{OpenTelemetrySpanExt, PreSampledTracer};
use tracing_subscriber::layer::Filter;
use tracing_subscriber::{filter, EnvFilter, Registry};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, Layer};
use uuid::Uuid;

use crate::error::{Error, ErrorDetails};

#[derive(Clone, Debug, Default, ValueEnum)]
pub enum LogFormat {
    #[default]
    Pretty,
    Json,
}

#[derive(Clone, Debug)]
struct CustomTracerKey {
    extra_headers: MetadataMap,
}

impl Hash for CustomTracerKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let CustomTracerKey { extra_headers } = self;
        extra_headers.as_ref().iter().for_each(|(key, value)| {
            key.hash(state);
            state.write_u8(0);
            value.hash(state);
            state.write_u8(0);
        });
    }
}

impl PartialEq for CustomTracerKey {
    fn eq(&self, other: &Self) -> bool {
        let CustomTracerKey { extra_headers } = self;
        extra_headers.as_ref() == other.extra_headers.as_ref()
    }
}

impl Eq for CustomTracerKey {}

#[derive(Clone, Debug)]
struct CustomTracer {
    inner: SdkTracer,
    provider: SdkTracerProvider,
}

pub struct TracerWrapper {
    default_tracer: SdkTracer,
    default_provider: SdkTracerProvider,
    custom_tracers: Cache<CustomTracerKey, CustomTracer>,
    shutdown_tasks: TaskTracker,
}

fn build_tracer<T: SpanExporter + 'static>(
    metadata: MetadataMap,
    override_exporter: Option<T>,
) -> Result<(SdkTracerProvider, SdkTracer), Error> {
    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_metadata(metadata)
        .build()
        .map_err(|e| {
            Error::new(ErrorDetails::Observability {
                message: format!("Failed to create OTLP exporter: {e}"),
            })
        })?;

    let mut builder = SdkTracerProvider::builder().with_resource(
        Resource::builder_empty()
            .with_attribute(KeyValue::new(
                opentelemetry_semantic_conventions::resource::SERVICE_NAME,
                "tensorzero-gateway",
            ))
            .build(),
    );

    if let Some(exporter) = override_exporter {
        builder = builder.with_simple_exporter(exporter);
    } else {
        builder = builder.with_batch_exporter(exporter);
    }
    let provider = builder.build();

    let tracer = provider.tracer("tensorzero");
    Ok((provider, tracer))
}

impl Tracer for TracerWrapper {
    type Span = <SdkTracer as Tracer>::Span;

    fn build_with_context(
        &self,
        builder: opentelemetry::trace::SpanBuilder,
        parent_cx: &opentelemetry::Context,
    ) -> Self::Span {
        if let Some(key) = parent_cx.get::<CustomTracerKey>() {
            let tracer = self.custom_tracers.try_get_with_by_ref(key, || {
                // We need to provide a dummy generic parameter to satisfy the compiler
                let (provider, tracer) = build_tracer::<opentelemetry_otlp::SpanExporter>(
                    key.extra_headers.clone(),
                    None,
                )?;
                Ok::<_, Error>(CustomTracer {
                    inner: tracer,
                    provider,
                })
            });
            match tracer {
                Ok(tracer) => {
                    return tracer.inner.build_with_context(builder, parent_cx);
                }
                Err(e) => {
                    tracing::error!("Failed to create custom tracer for span {builder:?}: {e}");
                    return self.default_tracer.build_with_context(builder, parent_cx);
                }
            }
        }

        self.default_tracer.build_with_context(builder, parent_cx)
    }

    fn start<T>(&self, name: T) -> Self::Span
    where
        T: Into<std::borrow::Cow<'static, str>>,
    {
        self.default_tracer.start(name)
    }

    fn start_with_context<T>(&self, name: T, parent_cx: &opentelemetry::Context) -> Self::Span
    where
        T: Into<std::borrow::Cow<'static, str>>,
    {
        self.default_tracer.start_with_context(name, parent_cx)
    }

    fn span_builder<T>(&self, name: T) -> opentelemetry::trace::SpanBuilder
    where
        T: Into<std::borrow::Cow<'static, str>>,
    {
        self.default_tracer.span_builder(name)
    }

    fn build(&self, builder: opentelemetry::trace::SpanBuilder) -> Self::Span {
        self.default_tracer.build(builder)
    }

    fn in_span<T, F, N>(&self, name: N, f: F) -> T
    where
        F: FnOnce(opentelemetry::Context) -> T,
        N: Into<std::borrow::Cow<'static, str>>,
        Self::Span: Send + Sync + 'static,
    {
        self.default_tracer.in_span(name, f)
    }
}

impl PreSampledTracer for TracerWrapper {
    fn sampled_context(
        &self,
        data: &mut tracing_opentelemetry::OtelData,
    ) -> opentelemetry::Context {
        self.default_tracer.sampled_context(data)
    }

    fn new_trace_id(&self) -> opentelemetry::trace::TraceId {
        self.default_tracer.new_trace_id()
    }

    fn new_span_id(&self) -> opentelemetry::trace::SpanId {
        self.default_tracer.new_span_id()
    }
}

struct OtelLayerData<T: Layer<Registry>> {
    layer: T,
    wrapper: TracerWrapper,
}

// Builds the internal OpenTelemetry layer, without any filtering applied.
fn internal_build_otel_layer<T: SpanExporter + 'static>(
    override_exporter: Option<T>,
) -> Result<OtelLayerData<impl Layer<Registry>>, Error> {
    let (provider, tracer) = build_tracer(MetadataMap::new(), override_exporter)?;
    opentelemetry::global::set_tracer_provider(provider.clone());
    let shutdown_tasks = TaskTracker::new();
    let shutdown_tasks_clone = shutdown_tasks.clone();
    let wrapper = TracerWrapper {
        default_tracer: tracer,
        default_provider: provider,
        custom_tracers: Cache::builder()
            .eviction_listener(move |_key, val: CustomTracer, _reason| {
                shutdown_tasks_clone.spawn(shutdown_otel(val.provider));
            })
            .build(),
        shutdown_tasks,
    };

    // Cloning of these types internally peresrves a reference - we don't need our own `Arc` here
    let cloned_wrapper = TracerWrapper {
        default_tracer: wrapper.default_tracer.clone(),
        default_provider: wrapper.default_provider.clone(),
        custom_tracers: wrapper.custom_tracers.clone(),
        shutdown_tasks: wrapper.shutdown_tasks.clone(),
    };
    Ok(OtelLayerData {
        layer: tracing_opentelemetry::layer()
            .with_tracer(wrapper)
            .with_level(true),
        wrapper: cloned_wrapper,
    })
}

async fn shutdown_otel(provider: SdkTracerProvider) -> Result<(), Error> {
    tokio::task::spawn_blocking(move || {
        let id = Uuid::now_v7();
        tracing::debug!(tracer_id = id.to_string(), "Shutting down custom tracer");
        provider.shutdown().map_err(|e| {
            Error::new(ErrorDetails::Observability {
                message: format!("Failed to shutdown OpenTelemetry: {e}"),
            })
        })?;
        tracing::debug!(tracer_id = id.to_string(), "Custom tracer shut down");
        Ok::<_, Error>(())
    })
    .await
    .map_err(|e| {
        Error::new(ErrorDetails::Observability {
            message: format!("Failed to wait on OpenTelemetry shutdown: {e}"),
        })
    })??;
    Ok(())
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
) -> Result<(DelayedOtelEnableHandle, impl Layer<Registry>, TracerWrapper), Error> {
    let (otel_reload_filter, reload_handle) = tracing_subscriber::reload::Layer::new(Box::new(
        LevelFilter::OFF,
    )
        as Box<dyn Filter<_> + Send + Sync>);

    let OtelLayerData {
        layer: base_otel_layer,
        wrapper,
    } = internal_build_otel_layer(override_exporter)?;

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
                if metadata.is_event() {
                    matches!(metadata.level(), &tracing::Level::ERROR)
                } else {
                    metadata.fields().iter().any(|field| {
                        field.name().starts_with("http.") || field.name().starts_with("otel.")
                    })
                }
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
        wrapper,
    ))
}

/// A helper trait to apply layers to a `Router`.
/// Without this trait, we would need to write something like:
/// ```rust
/// // in `tensorzero-core`
/// fn make_my_layer() -> SomeType
///
/// // in `tensorzero-gateway`
/// router.layer(tensorzero_core::make_my_layer())
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

/// A special header prefix used to attach an additional header to the OTLP export for this trace.
/// The format is: `tensorzero-otlp-traces-extra-header--HEADER_NAME: HEADER_VALUE`
/// For each header with the `tensorzero-otlp-traces-extra-header--`, we add `HEADER_NAME: HEADER_VALUE`
/// to the OTLP export HTTP/gRPC request headers.
///
/// When an incoming request has a `tensorzero-otlp-traces-extra-header--`, we handle it through
/// the following sequence of events:
/// 1. The `tensorzero_otlp_headers_middleware` function extracts the headers from the request,
///    validates them, and attaches a `CustomTracerKey` to the request extensions.
///    This needs to be a separate Axum middleware layer so that we can reject the request
///    if the headers fail to parse.
/// 2. The tracing layer in `apply_otel_http_trace_layer` checks for a `CustomTracerKey` in the
///    request extensions. If it's present, it's set in the OpenTelemetry `Context` when we construct
///    our `tracing::Span`.
/// 3. The `tracing-opentelemetry` library propagates this `Context` to all descendant spans,
///    ensuring that the entire tree of spans will have our `CustomTracerKey` available.
/// 4. When a `tracing::Span` is closed, our `TracerWrapper` is called by `tracing-opentelemetry`
///    We check for the presence of our `CustomTracerKey` in the opentelemetry `Context` -
///    if it's present, then we get or create a new `SdkTrace` with the provided extra headers
///    set in the OTLP exporter. The custom tracer is otherwise identical to the standard one
///    (it exports to the same URL at the same interval).
///
///    If a `CustomTracerKey` is not present, we use the default OTLP tracer
///    (which exports without any extra headers set).
///
/// 5. The custom `SdkTracer` is preseved in a `moka::Cache` for subsequent requests.
const TENSORZERO_OTLP_HEADERS_PREFIX: &str = "tensorzero-otlp-traces-extra-header-";

fn extract_tensorzero_headers(headers: &HeaderMap) -> Result<Option<CustomTracerKey>, Error> {
    let mut metadata = MetadataMap::new();
    for (name, value) in headers {
        if let Some(suffix) = name.as_str().strip_prefix(TENSORZERO_OTLP_HEADERS_PREFIX) {
            let key: AsciiMetadataKey = suffix.parse().map_err(|e| {
                Error::new(ErrorDetails::Observability {
                    message: format!("Failed to parse `{TENSORZERO_OTLP_HEADERS_PREFIX}` header `{suffix}` as valid metadata key: {e}"),
                })
            })?;
            let value = MetadataValue::from_str(&value.to_str().map_err(|e| {
                Error::new(ErrorDetails::Observability {
                    message: format!("Failed to parse `{TENSORZERO_OTLP_HEADERS_PREFIX}` header `{suffix}` value as valid string: {e}"),
                })
            })?).map_err(|e| {
                Error::new(ErrorDetails::Observability {
                    message: format!("Failed to parse `{TENSORZERO_OTLP_HEADERS_PREFIX}` header `{suffix}` value as valid metadata value: {e}"),
                })
            })?;
            eprintln!("Adding header `{key}` with value `{value:?}`");
            metadata.insert(key, value);
        }
    }
    if !metadata.is_empty() {
        return Ok(Some(CustomTracerKey {
            extra_headers: metadata,
        }));
    }
    Ok(None)
}

async fn tensorzero_otlp_headers_middleware(
    mut req: axum::extract::Request,
    next: Next,
) -> Response {
    match extract_tensorzero_headers(req.headers()) {
        Ok(Some(custom_tracer_key)) => {
            req.extensions_mut().insert(custom_tracer_key);
        }
        Ok(None) => {}
        Err(e) => {
            return e.into_response();
        }
    }
    next.run(req).await
}

impl<S: Clone + Send + Sync + 'static> RouterExt<S> for Router<S> {
    /// Makes a `TraceLayer` specialized for OpenTelemetry traces.
    /// This is only applied to certain routes (e.g. `/inference`), and
    /// is *not* used to log requests to the console.
    fn apply_otel_http_trace_layer(self) -> Self {
        fn make_span<B>(req: &http::Request<B>) -> Span {
            // Based on `OtelAxumLayer`. We use `TraceLayer` from `tower-http` instead, as it extends
            // the span to cover the entire response (including sending the full SSE stream).`

            // If we need to use a custom otel `Tracer`, then attach an `CustomTracerKey` to the OTEL context.
            // We check for a `CustomTracerKey` `TracerWrapper`, and use it to dispatch to a
            // dynamically-created `SdkTracer` with additional headers set.
            let mut in_context = None;
            if let Some(custom_tracer_key) = req.extensions().get::<CustomTracerKey>() {
                in_context = Some(Context::current_with_value(custom_tracer_key.clone()).attach());
            }
            let span =
                tracing_opentelemetry_instrumentation_sdk::http::http_server::make_span_from_request(
                    req,
                );
            span.set_parent(
                tracing_opentelemetry_instrumentation_sdk::http::extract_context(req.headers()),
            );
            drop(in_context);

            let route = req
                .extensions()
                .get::<MatchedPath>()
                .map(MatchedPath::as_str);
            if let Some(route) = route {
                span.record("http.route", route);
            }

            let method = tracing_opentelemetry_instrumentation_sdk::http::http_method(req.method());
            span.record(
                "otel.name",
                format!("{method} {}", route.unwrap_or_default()).trim(),
            );
            span
        }
        self.layer(TraceLayer::new_for_http().make_span_with(make_span))
            .layer(middleware::from_fn(tensorzero_otlp_headers_middleware))
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
pub struct ObservabilityHandle {
    /// We allow the OTEL layer creation to fail (e.g. if an invalid `OTEL_` environment variable is set)
    /// The HTTP gateway will exit if OTLP was explicitly enabled through the config,
    /// while the embedded gateway will do nothing (as it never actually tries to enable
    /// OTEL exporting via `delayed_otel`).
    /// **NOTE** - since the `Error` will have been constructed before we've initialized
    /// `tracing_subscriber`, it will *not* be automatically logged.
    /// Instead, consumers that care about OTEL (currently only the HTTP gateway)
    /// must manually log the error.
    pub delayed_otel: Result<DelayedOtelEnableHandle, Error>,
    pub delayed_debug_logs: DelayedDebugLogs,
    pub otel_tracer: Option<TracerWrapper>,
}

impl TracerWrapper {
    pub async fn shutdown(self) {
        // First, spawn shutdown tasks for all of our custom tracers.
        // This might happen in parallel for the same custom tracer, but opentelemetry
        // documents that it's safe to call `shutdown` multiple times.
        for (_key, tracer) in &self.custom_tracers {
            self.shutdown_tasks
                .spawn(shutdown_otel(tracer.provider.clone()));
        }
        self.shutdown_tasks
            .spawn(shutdown_otel(self.default_provider.clone()));
        // Then, wait for all of the custom tracers to shut down.
        self.shutdown_tasks.close();
        self.shutdown_tasks.wait().await;
    }
}

/// This is used when `gateway.debug` is `false` and `RUST_LOG` is not set
const DEFAULT_GATEWAY_NON_DEBUG_DIRECTIVES: &str = "warn,gateway=info,tensorzero_core=info";
/// This is used when `gateway.debug` is `true` and `RUST_LOG` is not set
const DEFAULT_GATEWAY_DEBUG_DIRECTIVES: &str =
    "warn,gateway=debug,tensorzero_core=debug,tower_http::trace=debug";

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
pub async fn setup_observability(log_format: LogFormat) -> Result<ObservabilityHandle, Error> {
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
    let otel_data = build_opentelemetry_layer::<opentelemetry_otlp::SpanExporter>(None);
    let (delayed_otel, otel_layer, tracer_wrapper) = match otel_data {
        Ok((delayed_otel, otel_layer, tracer_wrapper)) => {
            (Ok(delayed_otel), Some(otel_layer), Some(tracer_wrapper))
        }
        Err(e) => (Err(e), None, None),
    };
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
    Ok(ObservabilityHandle {
        delayed_otel,
        delayed_debug_logs,
        otel_tracer: tracer_wrapper,
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
