//! TensorZero observability code.
//!
//! This module contains code for three inter-related observability systems:
//! 1. `tracing` span configuration and logging
//! 2. OpenTelemetry span exporting, via `tracing-opentelemetry`
//! 3. Prometheus metric exporting.
//!
//! The main entrypoint for this module are:
//!
//! * `setup_observability` - registers a global Tracing subscriber, with an (initially disabled) OpenTelemetry layer attached
//! * `ObservabilityHandle` - produced by `setup_observability`, and used to handle delayed tracing configuration, and OTLP shutdown.
//!   We need to set up `tracing` before we parse our config file (so that we can log warnings and errors during config parsing),
//!   but the config file itself controls OTLP exporting and debug logging. The `ObservabilityHandle` type provides callbacks
//!   which we conditionally invoke after we've parsed our config file, and before starting the gateway.
//! * `setup_metrics` - builds Prometheus metrics exporter
//!
//! As part of our opentelemetry handling, we support forwarding custom HTTP headers to the OTLP export endpoint.
//! This requires several interconnected steps:
//! 1. A client makes a request to a traced-enabled TensorZero HTTP endpoint (e.g. POST /inference),
//!    with header(s) prefixed with `tensorzero-otlp-traces-extra-header-`.
//!    For example, `tensorzero-otlp-traces-extra-header-my-first-header: my-first-value`.
//! 2. Our `tensorzero_otlp_headers_middleware` Axum middleware detects these custom headers,
//!    and constructs a `CustomTracerKey` with the header name and value pairs.
//!    The middleware rejects the request if the headers fail to parse as a `tonic::metadata::MetadataMap`
//!    (this is the type that we will ultimately pass to the OTLP exporter).
//!    We attach this `CustomTracerKey` to the http request extensions.
//! 3. Our `make_span` function in `apply_otel_http_trace_layer` detects the `CustomTracerKey` in the request extensions,
//!    and inserts it into the opentelemetry `Context` when we construct our `tracing::Span` for the overall HTTP request.
//!    The `tracing-opentelemetry` library will propagate this `Context` to all descendant spans, so we only need to do
//!    this for the root HTTP request span.
//! 4. When any `tracing::Span` is closed, our `TracerWrapper::build_with_context` is called by `tracing-opentelemetry`
//!    (since we registered it when creating the OpenTelemetry layer).
//!    If our `CustomTracerKey` is present in the `Context`, then we perform a (cached) creation of a new `SdkTracer`,
//!    with the custom headers from the `CustomTracerKey` set in the OTLP exporter. We need to create an entirely new
//!    tracer, since the `metadata` (which controls the custom headers) can only be set at creation time.
//!    If we don't have a `CustomTracerKey` in the `Context`, then we use our default `SdkTracer` (which doesn't
//!    have any custom headers set).
//! 5. The OpenTelemetry `Span` is built using a `SdkTracer` with our custom metadata attached, so the custom
//!    headers will be set when that span is exported. Since we cache the `SdkTracer`s, multiple requests that
//!    have exactly the same custom headers set can share the same `SdkTracer`, and benefit from things like
//!    batched exporting.
//!
//! We store our `CustomTracerKey` in two 'context' objects:
//! * The `http::Request` extensions map on the HTTP request, so that our middleware can pass information
//!   to our `make_span` function.
//! * The OpenTelemetry `Context`, which is captured by the `tracing-opentelemetry` library when we create a new span,
//!   and passed along to `TracerWrapper::build_with_context` when the span is closed and exported.
use std::borrow::Cow;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Duration;

use once_cell::sync::OnceCell;

use axum::extract::MatchedPath;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::{middleware, Router};
use clap::ValueEnum;
use http::HeaderMap;
use metrics::{describe_counter, Unit};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use moka::sync::Cache;
use opentelemetry::trace::Status;
use opentelemetry::trace::{Tracer, TracerProvider as _};
use opentelemetry::{Context, KeyValue};
use opentelemetry_otlp::tonic_types::metadata::MetadataMap;
use opentelemetry_otlp::WithTonicConfig;
use opentelemetry_sdk::trace::SdkTracer;
use opentelemetry_sdk::trace::{SdkTracerProvider, SpanExporter};
use opentelemetry_sdk::Resource;
use std::str::FromStr;
use tokio_util::task::TaskTracker;
use tonic::metadata::AsciiMetadataKey;
use tonic::metadata::MetadataValue;
use tracing::level_filters::LevelFilter;
use tracing::Span;
use tracing_futures::Instrument;
use tracing_opentelemetry::OpenTelemetrySpanExt;
use tracing_opentelemetry::PreSampledTracer;
use tracing_subscriber::layer::Filter;
use tracing_subscriber::{filter, EnvFilter, Registry};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, Layer};
use uuid::Uuid;

use crate::error::{Error, ErrorDetails};
use crate::observability::tracing_bug::apply_filter_fixing_tracing_bug;

pub mod tracing_bug;

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

/// A special wrapper to dispatch to different `Tracer` implementations based on our `Context` and `Span`
/// being exported.
/// By default, we forward to `default_tracer`/`default_provider`. When we have a `CustomTracerKey` in our `Context`
/// (due to `tensorzero-otlp-traces-extra-header-` being set in the incoming request),
/// we forward to a (cached) dynamically-created `CustomTracer`, which has extra headers set in the OTLP exporter.
pub struct TracerWrapper {
    default_tracer: SdkTracer,
    default_provider: SdkTracerProvider,
    // Static headers from the config that are always included (can be overridden by dynamic headers)
    // Wrapped in Arc<OnceCell> so we can set them once after initialization (e.g. in the gateway after loading config)
    static_otlp_traces_extra_headers: Arc<OnceCell<MetadataMap>>,
    // We need to build a new `CustomTracer` for each unique list of extra headers,
    // since export headers can only be configured at the `Tracer` level.
    // We use a `moka` Cache to handle automatic eviction (see `internal_build_otel_layer` for
    // where we register an eviction_listener).
    custom_tracers: Cache<CustomTracerKey, CustomTracer>,
    // Shutdown tasks for all of the `CustomTracer`s that have been evicted from our cache.
    // We use a `TaskTracer` to avoid accumulating memory for each finished `CustomTracer` -
    // memory is freed immediately when a shutdown tasks exists. We wait on all remaining tasks
    // in `TracerWrapper::shutdown`
    shutdown_tasks: TaskTracker,
}

// Adds our self-signed certificate to the TLS config for Tonic
// This is used in e2e test mode so that we can test gRPC export over TLS to
// our local OTLP collector.
#[cfg(feature = "e2e_tests")]
fn add_local_self_signed_cert(
    tls_config: tonic::transport::ClientTlsConfig,
) -> tonic::transport::ClientTlsConfig {
    static CERT: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/e2e/self-signed-certs/otlp-collector.crt"
    ));
    tls_config.ca_certificate(tonic::transport::Certificate::from_pem(CERT))
}

/// Builds a new `SdkTracerProvider`, which will attach the extra headers from `metadata`
/// to outgoing OTLP export requests.
fn build_tracer<T: SpanExporter + 'static>(
    metadata: MetadataMap,
    override_exporter: Option<T>,
) -> Result<(SdkTracerProvider, SdkTracer), Error> {
    let tls_config = tonic::transport::ClientTlsConfig::new().with_enabled_roots();
    #[cfg(feature = "e2e_tests")]
    let tls_config = add_local_self_signed_cert(tls_config);

    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_metadata(metadata)
        .with_tls_config(tls_config)
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

    // This is the only method where we dispatch to a different `Tracer` - all other methods
    // just forward to `default_tracer`/`default_provider`.
    // This is fine, since `build_with_context` is the only method used by `tracing-opentelemetry`
    // when building an OpenTelemetry span from a `tracing::Span`.
    fn build_with_context(
        &self,
        builder: opentelemetry::trace::SpanBuilder,
        parent_cx: &opentelemetry::Context,
    ) -> Self::Span {
        // Check if we have any headers (config or dynamic)
        let static_otlp_traces_extra_headers = self.static_otlp_traces_extra_headers.get();
        let dynamic_key = parent_cx.get::<CustomTracerKey>();

        // If we have any headers (config OR dynamic), create a custom tracer with merged headers
        let has_otlp_traces_extra_headers =
            static_otlp_traces_extra_headers.is_some() || dynamic_key.is_some();

        if has_otlp_traces_extra_headers {
            // This is the potentially expensive part - we need to dynamically create a new `SdkTracer`.
            // If this ends up causing performance issues (due to thrashing the `custom_tracers` cache,
            // or `build_tracer` becoming expensive), then we should do the following:
            // 1. Make a new `SpanWrapper` enum that we set as the `Span` associated type for `TracerWrapper`.
            // 2. When we have a `CustomTracerKey` in the `Context`, store the `builder` and `parent_cx` in the `SpanWrapper`,
            //    and don't immediately create the `SdkTracer`.
            // 3. In the `Drop` impl for `SpanWrapper`, call `tokio::task::spawn_blocking`, and perform the cache
            //    lookup and nested `build_with_context` inside the closure.

            // Merge config headers with dynamic headers (dynamic takes precedence)
            let mut merged_headers = static_otlp_traces_extra_headers
                .cloned()
                .unwrap_or_default();

            if let Some(key) = dynamic_key {
                for key_value_ref in key.extra_headers.iter() {
                    match key_value_ref {
                        tonic::metadata::KeyAndValueRef::Ascii(k, v) => {
                            merged_headers.insert(k.clone(), v.clone());
                        }
                        tonic::metadata::KeyAndValueRef::Binary(k, _) => {
                            tracing::warn!(
                                "Non-ASCII header encountered in `extra_headers` and ignored: {k}",
                            );
                        }
                    }
                }
            }

            // Create a new key with the merged headers
            let merged_key = CustomTracerKey {
                extra_headers: merged_headers,
            };

            let tracer = self.custom_tracers.try_get_with_by_ref(&merged_key, || {
                // We need to provide a dummy generic parameter to satisfy the compiler
                let (provider, tracer) = build_tracer::<opentelemetry_otlp::SpanExporter>(
                    merged_key.extra_headers.clone(),
                    None,
                )?;
                Ok::<_, Error>(CustomTracer {
                    inner: tracer,
                    provider,
                })
            });

            if let Ok(tracer) = tracer {
                return tracer.inner.build_with_context(builder, parent_cx);
            } else if let Err(e) = tracer {
                tracing::error!("Failed to create custom tracer for span {builder:?}: {e}");
            }
            return self.default_tracer.build_with_context(builder, parent_cx);
        }

        // No headers at all, use default tracer
        self.default_tracer.build_with_context(builder, parent_cx)
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
// The default tracer is always built with empty headers. Config headers are stored separately
// and applied when building spans. Use `TracerWrapper::set_static_otlp_traces_extra_headers` to set headers after initialization.
fn internal_build_otel_layer<T: SpanExporter + 'static>(
    override_exporter: Option<T>,
) -> Result<OtelLayerData<impl Layer<Registry>>, Error> {
    // Default tracer always has empty headers
    let (provider, tracer) = build_tracer(MetadataMap::new(), override_exporter)?;
    opentelemetry::global::set_tracer_provider(provider.clone());
    let shutdown_tasks = TaskTracker::new();
    let shutdown_tasks_clone = shutdown_tasks.clone();
    // Initialize empty - will be set once later via set_static_otlp_traces_extra_headers
    let config_headers = Arc::new(OnceCell::new());
    let wrapper = TracerWrapper {
        default_tracer: tracer,
        default_provider: provider,
        static_otlp_traces_extra_headers: config_headers.clone(),
        custom_tracers: Cache::builder()
            .max_capacity(32)
            // Expire entries that have been idle for 1 hour
            .time_to_idle(Duration::from_secs(60 * 60))
            .eviction_listener(move |_key, val: CustomTracer, _reason| {
                // When we evict a `CustomTracer` from the cache, shut it down, and add the shutdown task
                // to `shutdown_tasks` so that we can wait for all of the custom tracers to shut down
                // during gateway shutdown.
                shutdown_tasks_clone.spawn(shutdown_otel(val.provider));
            })
            .build(),
        shutdown_tasks,
    };

    // Cloning of these types internally preserves a reference - we don't need our own `Arc` here
    let cloned_wrapper = TracerWrapper {
        default_tracer: wrapper.default_tracer.clone(),
        default_provider: wrapper.default_provider.clone(),
        static_otlp_traces_extra_headers: wrapper.static_otlp_traces_extra_headers.clone(),
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

/// Shuts down the provided `SdkTracerProvider`, and asynchronously waits for the shutdown to complete.
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
                        message: format!("Failed to enable OTLP exporter: {e:?}"),
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
        apply_filter_fixing_tracing_bug(base_otel_layer, otel_reload_filter),
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
/// The format is: `tensorzero-otlp-traces-extra-header-HEADER_NAME: HEADER_VALUE`
/// For each header with the `tensorzero-otlp-traces-extra-header-`, we add `HEADER_NAME: HEADER_VALUE`
/// to the OTLP export HTTP/gRPC request headers.
///
/// When an incoming request has a `tensorzero-otlp-traces-extra-header-`, we handle it through
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
/// 5. The custom `SdkTracer` is preserved in a `moka::Cache` for subsequent requests.
const TENSORZERO_OTLP_HEADERS_PREFIX: &str = "tensorzero-otlp-traces-extra-header-";

/// Converts a HashMap of config headers to a MetadataMap
fn config_headers_to_metadata(
    config_headers: &HashMap<String, String>,
) -> Result<MetadataMap, Error> {
    let mut metadata = MetadataMap::new();
    for (name, value) in config_headers {
        let key: AsciiMetadataKey = name.parse().map_err(|e| {
            Error::new(ErrorDetails::Observability {
                message: format!(
                    "Failed to parse config header `{name}` as valid metadata key: {e}"
                ),
            })
        })?;
        let value = MetadataValue::from_str(value).map_err(|e| {
            Error::new(ErrorDetails::Observability {
                message: format!(
                    "Failed to parse config header `{name}` value as valid metadata value: {e}"
                ),
            })
        })?;
        metadata.insert(key, value);
    }
    Ok(metadata)
}

// Removes all of the headers prefixed with `TENSORZERO_OTLP_HEADERS_PREFIX`.
// If any are present, constructs a `CustomTracerKey` with all of the  matching header/value pairs
// (with `TENSORZERO_OTLP_HEADERS_PREFIX` removed from the header name).
fn extract_tensorzero_headers(headers: &HeaderMap) -> Result<Option<CustomTracerKey>, Error> {
    let mut metadata = MetadataMap::new();
    for (name, value) in headers {
        if let Some(suffix) = name.as_str().strip_prefix(TENSORZERO_OTLP_HEADERS_PREFIX) {
            let key: AsciiMetadataKey = suffix.parse().map_err(|e| {
                Error::new(ErrorDetails::Observability {
                    message: format!("Failed to parse `{TENSORZERO_OTLP_HEADERS_PREFIX}` header `{suffix}` as valid metadata key: {e}"),
                })
            })?;
            let value = MetadataValue::from_str(value.to_str().map_err(|e| {
                Error::new(ErrorDetails::Observability {
                    message: format!("Failed to parse `{TENSORZERO_OTLP_HEADERS_PREFIX}` header `{suffix}` value as valid string: {e}"),
                })
            })?).map_err(|e| {
                Error::new(ErrorDetails::Observability {
                    message: format!("Failed to parse `{TENSORZERO_OTLP_HEADERS_PREFIX}` header `{suffix}` value as valid metadata value: {e}"),
                })
            })?;
            metadata.insert(key, value);
        }
    }
    if !metadata.is_empty() {
        tracing::debug!("Using custom OTLP headers: {:?}", metadata);
        return Ok(Some(CustomTracerKey {
            extra_headers: metadata,
        }));
    }
    Ok(None)
}

/// Creates the top-level span for an incoming HTTP request, with the correct
/// OpenTelemetry context attached
fn make_span<B>(req: &http::Request<B>, key: Option<CustomTracerKey>) -> Span {
    // Based on `OtelAxumLayer`.
    // If we need to use a custom otel `Tracer`, then attach an `CustomTracerKey` to the OTEL context.
    // We check for a `CustomTracerKey` in `TracerWrapper`, and use it to dispatch to a
    // dynamically-created `SdkTracer` with additional headers set.
    let mut in_context = None;
    if let Some(custom_tracer_key) = key {
        in_context = Some(Context::current_with_value(custom_tracer_key).attach());
    }
    let span =
        tracing_opentelemetry_instrumentation_sdk::http::http_server::make_span_from_request(req);
    span.set_parent(
        tracing_opentelemetry_instrumentation_sdk::http::extract_context(req.headers()),
    );
    // We need our custom context to be active for both the span creation and setting the span parent.
    // This ensures that `tracing-opentelemetry` will record our custom Context (with the `CustomTracerKey`)
    // for the newly created span, and all of its descendants.
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

/// Attach information from our HTTP response to the original span for the overall
/// HTTP request processing
fn handle_response<B>(res: &Response<B>, span: &Span) {
    // We cast this to an i64, so that tracing-opentelemetry will record it as an integer
    // rather than a string
    span.record("http.response.status_code", res.status().as_u16() as i64);
    if res.status().is_server_error() {
        if let Some(error) = res.extensions().get::<Error>() {
            span.set_status(Status::Error {
                description: Cow::Owned(error.to_string()),
            });
        } else {
            // Don't set a description for non-TensorZero errors,
            // since we don't know what a nice description should look like
            span.set_status(Status::Error {
                description: Cow::Owned(String::new()),
            });
        }
    }
}

/// Applies a span to an incoming HTTP request.
/// Also handles OTLP-related headers (`traceparent`/`tracestate`) and custom TensorZero OTLP headers
async fn tensorzero_tracing_middleware(req: axum::extract::Request, next: Next) -> Response {
    let custom_tracer_key = match extract_tensorzero_headers(req.headers()) {
        Ok(key) => key,
        Err(e) => {
            return e.into_response();
        }
    };

    // Note - we intentionally create this span this *after* `extract_tensorzero_headers`
    // As a result, if we reject a request due to a failure to parse custom OTLP headers,
    // we will *not* create an OpenTelemetry span. Custom headers can be required to correctly
    // process an OpenTelemetry span (e.g. to set the Arize API key), so this is correct behavior.
    let span = make_span(&req, custom_tracer_key);
    let response = next.run(req).instrument(span.clone()).await;
    handle_response(&response, &span);
    response
}

impl<S: Clone + Send + Sync + 'static> RouterExt<S> for Router<S> {
    /// Makes a `TraceLayer` specialized for OpenTelemetry traces.
    /// This is only applied to certain routes (e.g. `/inference`), and
    /// is *not* used to log requests to the console.
    fn apply_otel_http_trace_layer(self) -> Self {
        self.layer(middleware::from_fn(tensorzero_tracing_middleware))
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
    /// Set the config headers that will be merged with dynamic headers.
    /// This can only be called once after initialization (e.g. in the gateway after loading config).
    pub fn set_static_otlp_traces_extra_headers(
        &self,
        headers: &HashMap<String, String>,
    ) -> Result<(), Error> {
        let metadata_map = config_headers_to_metadata(headers)?;
        self.static_otlp_traces_extra_headers
            .set(metadata_map)
            .map_err(|_| {
                Error::new(ErrorDetails::Observability {
                    message: "Failed to set static OTLP headers: already initialized".to_string(),
                })
            })
    }

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

/// Set up logging (including the necessary layers for OpenTelemetry exporting)
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
    // We need to provide a dummy generic parameter to satisfy the compiler
    setup_observability_with_exporter_override::<opentelemetry_otlp::SpanExporter>(log_format, None)
        .await
}

pub async fn setup_observability_with_exporter_override<T: SpanExporter + 'static>(
    log_format: LogFormat,
    exporter_override: Option<T>,
) -> Result<ObservabilityHandle, Error> {
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

    let otel_data = build_opentelemetry_layer(exporter_override);
    let (delayed_otel, otel_layer, tracer_wrapper) = match otel_data {
        Ok((delayed_otel, otel_layer, tracer_wrapper)) => {
            (Ok(delayed_otel), Some(otel_layer), Some(tracer_wrapper))
        }
        Err(e) => (Err(e), None, None),
    };
    // IMPORTANT: If you add any new layers here that have per-layer filtering applied
    // you *MUST* call `apply_filter_fixing_tracing_bug` instead of `layer.with_filter(filter)`
    // See the docs for `apply_filter_fixing_tracing_bug` for more details.
    tracing_subscriber::registry()
        .with(otel_layer)
        .with(apply_filter_fixing_tracing_bug(log_layer, log_level))
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
    let metrics_handle = PrometheusBuilder::new().install_recorder().map_err(|e| {
        Error::new(ErrorDetails::Observability {
            message: format!("Failed to install Prometheus exporter: {e}"),
        })
    })?;

    // Register the expected metrics along with their types and docstrings
    describe_counter!(
        "request_count",
        Unit::Count,
        "Requests handled by TensorZero (deprecated: use `tensorzero_requests_total` instead)",
    );

    describe_counter!(
        "tensorzero_requests_total",
        Unit::Count,
        "Requests handled by TensorZero",
    );

    describe_counter!(
        "inference_count",
        Unit::Count,
        "Inferences performed by TensorZero (deprecated: use `tensorzero_inferences_total` instead)",
    );

    describe_counter!(
        "tensorzero_inferences_total",
        Unit::Count,
        "Inferences performed by TensorZero",
    );

    Ok(metrics_handle)
}
