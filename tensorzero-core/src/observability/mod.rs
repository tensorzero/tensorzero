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
//!    with header(s) prefixed with:
//! * `tensorzero-otlp-traces-extra-header-`: For example, `tensorzero-otlp-traces-extra-header-my-first-header: my-first-value`.
//! * `tensorzero-otlp-traces-extra-resource-`: For example, `tensorzero-otlp-traces-extra-resource-my-first-resource: my-first-value`.
//! 2. Our `tensorzero_tracing_middleware` Axum middleware detects these custom headers,
//!    and rejects the request if the headers fail to parse as a `tonic::metadata::MetadataMap`
//!    (this is the type that we will ultimately pass to the OTLP exporter).
//! 3. We perform a (cached) creation of a `CustomTracer` with the `MetadataMap` we just parsed.
//!    Incoming requests with identical custom OpenTelemetry headers will have the same cache key,
//!    and can share the same `Arc<CustomTracer>`. This is a performance optimization - we use a `moka` cache
//!    with eviction to prevent an unbounded amount of memory from being used for different `CustomTracer` instances.
//! 4. We attach a `CustomTracerContextEntry` (which holds our `Arc<CustomTracer>`) to the span's opentelemetry `Context`.
//!    This is automatically propagated to descendant spans by `tracing-opentelemetry`. Once all of descendant spans are dropped
//!    (i.e. all background processing for our request is finished), the `CustomTracer` will get automatically dropped,
//!    which triggers shutdown in our `Drop` impl for `CustomTracer`
//! 4. When a span is exported using `TracerWrapper::build_with_context`, we inspect the `Context` for a `CustomTracerContextEntry`.
//!    If present, we use the wrapped `CustomTracer`, which will cause the span to get exported using the correct
//!    custom HTTP headers and OpenTelemetry resources. Otherwise, we our default `SdkTracer`, which doesn't attach any custom headers
//!    or extra OpenTelemetry resources.
use futures::StreamExt;
use std::borrow::Cow;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Duration;

use once_cell::sync::OnceCell;
#[cfg(feature = "e2e_tests")]
use opentelemetry::ContextGuard;
use tokio_stream::wrappers::IntervalStream;
use tracing::field::Empty;

use crate::observability::exporter_wrapper::TensorZeroExporterWrapper;
use crate::observability::span_leak_detector::SpanLeakDetector;
use axum::extract::MatchedPath;
use axum::extract::State;
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
use tokio_util::task::task_tracker::TaskTrackerToken;
use tokio_util::task::TaskTracker;
use tonic::metadata::AsciiMetadataKey;
use tonic::metadata::MetadataValue;
use tracing::level_filters::LevelFilter;
use tracing::{Metadata, Span};
use tracing_futures::Instrument;
use tracing_opentelemetry::OpenTelemetrySpanExt;
use tracing_opentelemetry_instrumentation_sdk::http::{
    http_flavor, http_host, url_scheme, user_agent,
};
use tracing_subscriber::layer::Filter;
use tracing_subscriber::{filter, EnvFilter, Registry};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, Layer};
use uuid::Uuid;

use crate::error::{Error, ErrorDetails};
use crate::observability::tracing_bug::apply_filter_fixing_tracing_bug;

mod exporter_wrapper;
pub mod request_logging;
mod span_leak_detector;
pub mod tracing_bug;

#[derive(Clone, Debug, Default, ValueEnum)]
pub enum LogFormat {
    #[default]
    Pretty,
    Json,
}

#[derive(Clone, Debug)]
struct CustomTracerKey {
    // Extra headers to use for outgoing OTLP export requests.
    // These will be set as headers in the gRPc request made by `tonic`
    extra_headers: MetadataMap,
    // Extra OpenTelemetry resources (https://opentelemetry.io/docs/languages/js/resources/)
    // These will be set as attributes on *all* spans (not just top-level spans)
    // exported by a `CustomTracer`
    extra_resources: Vec<KeyValue>,
    extra_attributes: Vec<KeyValue>,
}

impl Hash for CustomTracerKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let CustomTracerKey {
            extra_headers,
            extra_resources,
            extra_attributes,
        } = self;
        // We add null byte separators to keep the data prefix-free: https://doc.rust-lang.org/std/hash/trait.Hash.html#prefix-collisions
        extra_headers.as_ref().iter().for_each(|(key, value)| {
            key.hash(state);
            state.write_u8(0);
            value.hash(state);
            state.write_u8(0);
        });
        state.write_u8(0);
        extra_resources.hash(state);
        extra_attributes.hash(state);
    }
}

impl PartialEq for CustomTracerKey {
    fn eq(&self, other: &Self) -> bool {
        let CustomTracerKey {
            extra_headers,
            extra_resources,
            extra_attributes,
        } = self;
        extra_headers.as_ref() == other.extra_headers.as_ref()
            && extra_resources == &other.extra_resources
            && extra_attributes == &other.extra_attributes
    }
}

impl Eq for CustomTracerKey {}

#[derive(Clone, Debug)]
struct CustomTracer {
    inner: SdkTracer,
    provider: Option<SdkTracerProvider>,
    // This comes from our `TracerWrapper` - when we drop a `CustomTracer`,
    // we add the shutdown future to `shutdown_tasks`, so that we can wait
    // on all custom tracers to shut down.
    shutdown_tasks: TaskTracker,
}

impl Drop for CustomTracer {
    fn drop(&mut self) {
        // Shut down the tracer in the background
        // When the entire gateway shut downs, we'll wait on `shutdown_tasks`
        // (our `shutdown_tasks` is a clone of an existing `TaskTracer`)
        // to make sure that all custom tracers have finished exporting
        // before we exit.
        if let Some(provider) = self.provider.take() {
            self.shutdown_tasks.spawn(shutdown_otel(provider));
        }
    }
}

/// Our entry in the opentelemetry `Context` for spans with a custom tracer.
/// When an incoming HTTP request has custom OpenTelemetry headers attached,
/// we insert a `CustomTracerContextEntry` into the span's `Context`, which
/// gets propagated by `tracing-opentelemetry` to all child spans.
/// When the span gets exported in `TracerWrapper::build_with_context`,
/// we check for the presence of `CustomTracerContextEntry`, and use the
/// `inner` field to perform the export.
///
/// By storing our `CustomTracer` in the context, we ensure that it gets dropped
/// automatically once all of the descendant spans are dropped. See
/// `CustomTracer` for more information
struct CustomTracerContextEntry {
    inner: Arc<CustomTracer>,
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
    custom_tracers: Cache<CustomTracerKey, Arc<CustomTracer>>,
    // Shutdown tasks for all of the `CustomTracer`s that have been evicted from our cache.
    // We use a `TaskTracer` to avoid accumulating memory for each finished `CustomTracer` -
    // memory is freed immediately when a shutdown tasks exists. We wait on all remaining tasks
    // in `TracerWrapper::shutdown`
    shutdown_tasks: TaskTracker,
    // See `InFlightSpan` for more information.
    in_flight_spans: TaskTracker,
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

impl TracerWrapper {
    fn get_or_create_custom_tracer(
        &self,
        key: &CustomTracerKey,
        context: Context,
    ) -> Result<Context, Error> {
        // This is the potentially expensive part - we need to dynamically create a new `SdkTracer`.
        // If this ends up causing performance issues (due to thrashing the `custom_tracers` cache,
        // or `build_tracer` becoming expensive), then we should do the following:
        // 1. Make a new `SpanWrapper` enum that we set as the `Span` associated type for `TracerWrapper`.
        // 2. When we have a `CustomTracerKey` in the `Context`, store the `builder` and `parent_cx` in the `SpanWrapper`,
        //    and don't immediately create the `SdkTracer`.
        // 3. In the `Drop` impl for `SpanWrapper`, call `tokio::task::spawn_blocking`, and perform the cache
        //    lookup and nested `build_with_context` inside the closure.
        let tracer = self
            .custom_tracers
            .try_get_with_by_ref(key, || {
                // We need to provide a dummy generic parameter to satisfy the compiler
                let (provider, tracer) =
                    build_tracer::<opentelemetry_otlp::SpanExporter>(key.clone(), None)?;
                Ok::<_, Error>(Arc::new(CustomTracer {
                    inner: tracer,
                    provider: Some(provider),
                    shutdown_tasks: self.shutdown_tasks.clone(),
                }))
            })
            .map_err(Arc::unwrap_or_clone)?;
        Ok(context.with_value(CustomTracerContextEntry { inner: tracer }))
    }
}

/// Builds a new `SdkTracerProvider`, which will attach the extra headers from `metadata`
/// to outgoing OTLP export requests.
fn build_tracer<T: SpanExporter + 'static>(
    key: CustomTracerKey,
    override_exporter: Option<T>,
) -> Result<(SdkTracerProvider, SdkTracer), Error> {
    let tls_config = tonic::transport::ClientTlsConfig::new().with_enabled_roots();
    #[cfg(feature = "e2e_tests")]
    let tls_config = add_local_self_signed_cert(tls_config);

    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_metadata(key.extra_headers)
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
            .with_attributes(key.extra_resources)
            .build(),
    );

    if let Some(override_exporter) = override_exporter {
        builder = builder.with_simple_exporter(TensorZeroExporterWrapper::new(
            override_exporter,
            key.extra_attributes,
        ));
    } else {
        builder = builder.with_batch_exporter(TensorZeroExporterWrapper::new(
            exporter,
            key.extra_attributes,
        ));
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
        if let Some(key) = parent_cx.get::<CustomTracerContextEntry>() {
            key.inner.inner.build_with_context(builder, parent_cx)
        } else {
            self.default_tracer.build_with_context(builder, parent_cx)
        }
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
    // Default tracer always has empty headers and no extra resources
    let (provider, tracer) = build_tracer(
        CustomTracerKey {
            extra_headers: MetadataMap::new(),
            extra_resources: vec![],
            extra_attributes: vec![],
        },
        override_exporter,
    )?;
    opentelemetry::global::set_tracer_provider(provider.clone());
    let shutdown_tasks = TaskTracker::new();
    // Initialize empty - will be set once later via set_static_otlp_traces_extra_headers
    let config_headers = Arc::new(OnceCell::new());
    let wrapper = TracerWrapper {
        default_tracer: tracer,
        default_provider: provider,
        static_otlp_traces_extra_headers: config_headers.clone(),
        // This cache stores `Arc<CustomTracer>`, so we don't need a custom eviction handler
        // Once all clones of an `Arc` are dropped (including those stored in opentelemetry `Context`
        // objects associated with various `Span`s), the `CustomTracer::drop` method will automatically get called,
        // which handles shutting down the custom tracer
        custom_tracers: Cache::builder()
            .max_capacity(32)
            // Expire entries that have been idle for 1 hour
            .time_to_idle(Duration::from_secs(60 * 60))
            .build(),
        shutdown_tasks,
        in_flight_spans: TaskTracker::new(),
    };

    // Cloning of these types internally preserves a reference - we don't need our own `Arc` here
    let cloned_wrapper = TracerWrapper {
        default_tracer: wrapper.default_tracer.clone(),
        default_provider: wrapper.default_provider.clone(),
        static_otlp_traces_extra_headers: wrapper.static_otlp_traces_extra_headers.clone(),
        custom_tracers: wrapper.custom_tracers.clone(),
        shutdown_tasks: wrapper.shutdown_tasks.clone(),
        in_flight_spans: wrapper.in_flight_spans.clone(),
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
        provider.shutdown_with_timeout(Duration::MAX).map_err(|e| {
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
/// which is configured via environment variables (e.g. `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`):
/// https://github.com/open-telemetry/opentelemetry-specification/blob/main/specification/protocol/exporter.md#endpoint-urls-for-otlphttp
fn build_opentelemetry_layer<T: SpanExporter + 'static>(
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
            // We only expose spans that explicitly contain field prefixed with "otel."
            // For example, `#[instrument(fields(otel.name = "my_otel_name"))]` will be exported
            fn accept_errors_and_otel(metadata: &Metadata<'_>) -> bool {
                if metadata.is_event() {
                    matches!(metadata.level(), &tracing::Level::ERROR)
                } else {
                    // We only expose spans that explicitly contain field prefixed with "otel.",
                    // *and* that are a descendant of a top-level HTTP span (as determined by the presence of an `InFlightSpan` in the context).
                    // This ensures that we can call into instrumented code (e.g. `Variant::infer`, or any authorization middleware)
                    // from a non-OTEL http route (e.g. a ui route that internally makes some inferences) without causing
                    // parent-less OTEL spans to get emitted by the instrumented code.
                    metadata
                        .fields()
                        .iter()
                        .any(|field| field.name().starts_with("otel."))
                        && Context::map_current(|c| c.get::<InFlightOtelOnlySpan>().is_some())
                }
            }

            // We mark this as a dynamic filter so that `tracing` doesn't cache the result
            // (it depends on the current call stack via the opentelemetry `Context`),
            // not just the static call site of the immediate span/event being filtered.
            #[allow(unused_mut, clippy::allow_attributes)]
            let mut filter =
                filter::dynamic_filter_fn(|metadata, _context| accept_errors_and_otel(metadata));

            #[cfg(any(test, feature = "e2e_tests"))]
            {
                if tracing_bug::DISABLE_TRACING_BUG_WORKAROUND
                    .load(std::sync::atomic::Ordering::Relaxed)
                {
                    // When we're attempting to reproduce the tracing bug, we turn *on* callsite caching.
                    // This is effectively the same behavior as when 'filter::filter_fn' is used instead of
                    // 'filter::dynamic_filter_fn'.
                    // We do this to avoid needing to do anything weird in the main production code
                    // (this entire block is only compiled in test code)
                    filter = filter.with_callsite_filter(|metadata| {
                        if accept_errors_and_otel(metadata) {
                            tracing::subscriber::Interest::always()
                        } else {
                            tracing::subscriber::Interest::never()
                        }
                    });
                }
            }

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
    fn apply_top_level_otel_http_trace_layer(
        self,
        otel_tracer: Option<Arc<TracerWrapper>>,
        otel_enabled_routes: OtelEnabledRoutes,
    ) -> Self;
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
const TENSORZERO_OTLP_RESOURCE_PREFIX: &str = "tensorzero-otlp-traces-extra-resource-";
const TENSORZERO_OTLP_ATTRIBUTE_PREFIX: &str = "tensorzero-otlp-traces-extra-attribute-";

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

fn json_to_otel_value(value: serde_json::Value) -> Result<opentelemetry::Value, Error> {
    match value {
        serde_json::Value::Null => Err(Error::new(ErrorDetails::InvalidRequest {
            message: "Null is not a valid OpenTelemetry attribute value".to_string(),
        })),
        serde_json::Value::Bool(value) => Ok(opentelemetry::Value::Bool(value)),
        serde_json::Value::Number(_) => Err(Error::new(ErrorDetails::InvalidRequest {
            message: "Numbers are not yet supported for OpenTelemetry attributes values"
                .to_string(),
        })),
        serde_json::Value::String(value) => Ok(opentelemetry::Value::String(value.into())),
        serde_json::Value::Array(_) => Err(Error::new(ErrorDetails::InvalidRequest {
            message: "Arrays are not yet supported for OpenTelemetry attribute values".to_string(),
        })),
        serde_json::Value::Object(_) => Err(Error::new(ErrorDetails::InvalidRequest {
            message: "JSON objects are not valid OpenTelemetry attribute values".to_string(),
        })),
    }
}

// Removes all of the headers prefixed with `TENSORZERO_OTLP_HEADERS_PREFIX`.
// If any are present (or we have static config headers), constructs a `CustomTracerKey` with all of the  matching header/value pairs
// (with `TENSORZERO_OTLP_HEADERS_PREFIX` removed from the header name).
// We also apply any static custom OTLP headers set in the `TracerWrapper`.
fn extract_tensorzero_headers(
    tracer_wrapper: &TracerWrapper,
    headers: &HeaderMap,
) -> Result<Option<CustomTracerKey>, Error> {
    // Merge config headers with dynamic headers (dynamic takes precedence)
    let mut metadata = tracer_wrapper
        .static_otlp_traces_extra_headers
        .get()
        .cloned()
        .unwrap_or_default();
    let mut extra_resources = vec![];
    let mut extra_attributes = vec![];
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
        if let Some(suffix) = name.as_str().strip_prefix(TENSORZERO_OTLP_RESOURCE_PREFIX) {
            let key = suffix.to_string();
            let value = value.to_str().map_err(|e| {
                Error::new(ErrorDetails::InvalidRequest {
                    message: format!("Failed to parse `{TENSORZERO_OTLP_RESOURCE_PREFIX}` header `{suffix}` value as valid string: {e}"),
                })
            })?.to_string();
            extra_resources.push(KeyValue::new(key, value));
        }
        if let Some(suffix) = name.as_str().strip_prefix(TENSORZERO_OTLP_ATTRIBUTE_PREFIX) {
            let key = suffix.to_string();
            let value = value.to_str().map_err(|e| {
                Error::new(ErrorDetails::InvalidRequest {
                    message: format!("Failed to parse `{TENSORZERO_OTLP_ATTRIBUTE_PREFIX}` header `{suffix}` value as valid string: {e}"),
                })
            })?;
            let value_json = serde_json::from_str::<serde_json::Value>(value).map_err(|e| {
                Error::new(ErrorDetails::InvalidRequest {
                    message: format!("Failed to parse `{TENSORZERO_OTLP_ATTRIBUTE_PREFIX}` header `{suffix}` value as valid JSON: {e}"),
                })
            })?;
            let value_otel = json_to_otel_value(value_json).map_err(|e| {
                Error::new(ErrorDetails::InvalidRequest {
                    message: format!("Failed to convert `{TENSORZERO_OTLP_ATTRIBUTE_PREFIX}` header `{suffix}` value to OpenTelemetry attribute value: {e}"),
                })
            })?;
            extra_attributes.push(KeyValue::new(key, value_otel));
        }
    }
    if !metadata.is_empty() || !extra_resources.is_empty() || !extra_attributes.is_empty() {
        tracing::debug!(
            "Using custom OTLP configuration: metadata={:?}, extra_resources={:?}, extra_attributes={:?}",
            metadata,
            extra_resources,
            extra_attributes
        );
        return Ok(Some(CustomTracerKey {
            extra_headers: metadata,
            extra_resources,
            extra_attributes,
        }));
    }
    Ok(None)
}

/// An opentelemetry `Context` value that keeps track of in-flight spans.
/// When we create a new top-level OpenTelemetry span (regardless of whether or not we have custom headers),
/// we attach an `InFlightSpan` to the `Context`
/// The `tracing-opentelemetry` library will propagate the `Context` to all descendant spans,
/// which ensures that our `InFlightSpan` is only dropped once all descendant spans are dropped.
/// In particular, it will be dropped after any background tasks that have otel-enabled spans (e.g. rate-limiting `return_tickets` calls)
/// have finished executing, even if they outlive the original HTTP connection.
///
/// During gateway shutdown, we wait for the parent `TaskTracker` to finish before shutting down our OTEL exporters,
/// which ensures that we only try to shut down all of our OTEL exporters after all in-flight spans have finished.
///
/// Note that this is *only* created for OpenTelemetry-enabled routes, and will not be created at all when OTEL is disabled.
/// Its purpose it to allow us to delay OTEL shutdown until we know that we won't miss exporting any spans.
///
/// If you want to ensure that a generic tokio task is processed before the gateway shuts down,
/// then you should spawn a task on `AppStateData.deferred_tasks`.
pub struct InFlightOtelOnlySpan {
    // This field just holds on to the token until the `InFlightSpan` is dropped.
    #[expect(dead_code)]
    token: TaskTrackerToken,
}

/// Enters into a fake HTTP request context for testing purposes, which will allow OTEL spans to be reported
/// This is used by OTEL tests that use an embedded client - since they don't go through our axum router,
/// we would normally (correctly) suppress any nested OTEL spans (e.g. `function_inference`)
/// This method simulates the relevant parts of our axum otel logic
/// We also have fully end-to-end tests that use a live gateway, so this is just to allow us to write
/// in-memory tests
#[cfg(feature = "e2e_tests")]
pub fn enter_fake_http_request_otel() -> ContextGuard {
    Context::current()
        .with_value(InFlightOtelOnlySpan {
            token: TaskTracker::new().token(),
        })
        .attach()
}

/// Creates the top-level span for an incoming HTTP request, with the correct
/// OpenTelemetry context attached
/// This span is *only* used for OpenTelemetry - we have a separate `TracerLayer`
/// we attach to print nice HTTP request logs to the console
fn make_otel_http_span<B>(
    req: &http::Request<B>,
    key: Option<CustomTracerKey>,
    tracer_wrapper: &TracerWrapper,
) -> Result<Span, Error> {
    // Based on `OtelAxumLayer`.
    // If we need to use a custom otel `Tracer`, then attach an `CustomTracerKey` to the OTEL context.
    // We check for a `CustomTracerKey` in `TracerWrapper`, and use it to dispatch to a
    // dynamically-created `SdkTracer` with additional headers set.
    let mut context =
        tracing_opentelemetry_instrumentation_sdk::http::extract_context(req.headers())
            // See the docs on `InFlightSpan` for more information.
            .with_value(InFlightOtelOnlySpan {
                token: tracer_wrapper.in_flight_spans.token(),
            });
    // If we had custom OTEL headers, and we've enabled otel export, then create a custom tracer
    // that attaches our custom headers on export.
    // This is stored in the span's `Context`, which is automatically propagated to descendants.
    // When a span is exported to OpenTelemetry, we'll look for
    if let Some(custom_tracer_key) = key {
        context = tracer_wrapper.get_or_create_custom_tracer(&custom_tracer_key, context)?;
    }
    let _guard = context.attach();

    let route = req
        .extensions()
        .get::<MatchedPath>()
        .map(MatchedPath::as_str);

    let http_method = req.method();

    // Copied from `tracing_opentelemetry_instrumentation_sdk::http::http_server::make_span_from_request` (https://github.com/davidB/tracing-opentelemetry-instrumentation-sdk/blob/5a64c55228645be87f21c628093dbd044104a10a/tracing-opentelemetry-instrumentation-sdk/src/http/http_server.rs#L10)
    // with `otel.name` added (so that it can be detected by our filtering layer)
    let span = tracing::info_span!(
        "HTTP request",
        http.request.method = %http_method,
        http.route = Empty, // to set by router of "webframework" after
        network.protocol.version = %http_flavor(req.version()),
        server.address = http_host(req),
        // server.port = req.uri().port(),
        http.client.address = Empty, //%$request.connection_info().realip_remote_addr().unwrap_or(""),
        user_agent.original = user_agent(req),
        http.response.status_code = Empty, // to set on response
        url.path = req.uri().path(),
        url.query = req.uri().query(),
        url.scheme = url_scheme(req.uri()),
        otel.kind = ?opentelemetry::trace::SpanKind::Server,
        otel.status_code = Empty, // to set on response
        trace_id = Empty, // to set on response
        request_id = Empty, // to set
        exception.message = Empty, // to set on response
        "span.type" = "web", // non-official open-telemetry key, only supported by Datadog
        otel.name = format!("{} {}", req.method(), route.unwrap_or_default()).trim(),
    );

    if let Some(route) = route {
        span.record("http.route", route);
    }

    Ok(span)
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

/// Applies an OpenTelemetry span to an incoming HTTP request.
/// Also handles OTLP-related headers (`traceparent`/`tracestate`) and custom TensorZero OTLP headers
///
/// We use this middleware to wrap *all* of our routes, not just OpenTelemetry-enabled routes.
/// This allows us to run this middleware before any other middleware, such as authorization.
/// As a result, the duration (and child spans) of this span include *all* processing associated with the request.
/// For example, authorization might require a Postgres lookup - we want to include this duration (and any associated spans
/// inside the top-level HTTP span that we export, even though this logic runs *before* we reach the route handler - in particular,
/// before any middleware that's applied in the 'middle' of the stack (e.g. only on certain routes).
///
/// Since it wraps all routes, we need to detect if the target route should actually create an OpenTelemetry span.
/// This is done through the `otel_enabled_routes`, which is initially constructed when we build our Axum router.
async fn tensorzero_otel_tracing_middleware(
    State(TracingMiddlewareState {
        tracer_wrapper,
        otel_enabled_routes,
    }): State<TracingMiddlewareState>,
    req: axum::extract::Request,
    next: Next,
) -> Response {
    // We parse headers even if the route is not OpenTelemetry-enabled, to prevent users from sending invalid headers
    // to a route (and then having their code break if we later decide to enable OpenTelemetry for that route).
    let custom_tracer_key = match extract_tensorzero_headers(&tracer_wrapper, req.headers()) {
        Ok(key) => key,
        Err(e) => {
            return e.into_response();
        }
    };

    let route = req
        .extensions()
        .get::<MatchedPath>()
        .map(MatchedPath::as_str);

    // If this is an OpenTelemetry-enabled route, then wrap the route handling in a new span
    // See the docstring on this method for why we need this check
    if let Some(route) = route {
        if otel_enabled_routes.routes.contains(&route) {
            // Note - we intentionally create this span this *after* `extract_tensorzero_headers`
            // As a result, if we reject a request due to a failure to parse custom OTLP headers,
            // we will *not* create an OpenTelemetry span. Custom headers can be required to correctly
            // process an OpenTelemetry span (e.g. to set the Arize API key), so this is correct behavior.
            let span = match make_otel_http_span(&req, custom_tracer_key, &tracer_wrapper) {
                Ok(span) => span,
                Err(e) => {
                    return e.into_response();
                }
            };
            let response = next.run(req).instrument(span.clone()).await;
            handle_response(&response, &span);
            return response;
        }
    }

    // Otherwise, just process the request without creating a span.
    // Since `make_otel_http_span` didn't run, we won't have an `InFlightOtelOnlySpan` in the context,
    next.run(req).await
}

pub struct OtelEnabledRoutes {
    // The list of routes (i.e. the strings passed to `Router.route`)
    // that have OpenTelemetry span exporting enabled.
    // This is constructed by `build_otel_enabled_routes` - we have a small number of routes,
    // so we use a Vec rather than a HashSet
    pub routes: Vec<&'static str>,
}

#[derive(Clone)]
pub struct TracingMiddlewareState {
    tracer_wrapper: Arc<TracerWrapper>,
    otel_enabled_routes: Arc<OtelEnabledRoutes>,
}

impl<S: Clone + Send + Sync + 'static> RouterExt<S> for Router<S> {
    /// Creates tracing spans for HTTP requests, specialized for OpenTelemetry traces
    /// Note that this is applied to *all* routes, not just OpenTelemetry-enabled routes.
    /// The `otel_enabled_routes` parameter is used to determine whether to create a span for the request.
    /// See the docs on `tensorzero_otel_tracing_middleware` for more details.
    fn apply_top_level_otel_http_trace_layer(
        self,
        otel_tracer: Option<Arc<TracerWrapper>>,
        otel_enabled_routes: OtelEnabledRoutes,
    ) -> Self {
        // If OpenTelemetry is disable, then we don't need to create extra spans
        if let Some(tracer) = otel_tracer {
            self.layer(middleware::from_fn_with_state(
                TracingMiddlewareState {
                    tracer_wrapper: tracer,
                    otel_enabled_routes: Arc::new(otel_enabled_routes),
                },
                tensorzero_otel_tracing_middleware,
            ))
        } else {
            self
        }
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
    pub otel_tracer: Option<Arc<TracerWrapper>>,
    // In `e2e_tests` mode, we enable a `SpanLeakDetector` to detect spans that were not closed when the gateway finished shutting down.
    pub leak_detector: Option<SpanLeakDetector>,
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

    /// Shuts down all OpenTelemetry exporters.
    /// This ensures that the batch exporter flushes all pending spans.
    /// No new requests should be *started* after this method is called:
    /// * In the HTTP gateway, we call this after the axum server has exited
    /// * If we ever support OTEL in an embedded client, it should be called
    ///   in a method that takes `self` by value (to prevent starting any new requests afterwards)
    ///
    /// This method will correctly wait for any processing related to *previous* requests to finish
    /// (e.g. rate-limiting `return_tickets` calls) to finish before shutting down the exporters.
    pub async fn shutdown(&self, leak_detector: Option<&SpanLeakDetector>) {
        // See the docs on `InFlightSpan` for more information.
        wait_for_tasks_with_logging(&self.in_flight_spans, "request processing", leak_detector)
            .await;
        // Now that all of our OpenTelemetry spans have closed (including spans in background tasks),
        // shut down all of our custom tracers.
        // This might happen in parallel for the same custom tracer (if moka evicts its cache entry), but opentelemetry
        // documents that it's safe to call `shutdown` multiple times.
        for (_key, tracer) in &self.custom_tracers {
            if let Some(provider) = &tracer.provider {
                self.shutdown_tasks.spawn(shutdown_otel(provider.clone()));
            }
        }
        // Also shut down our default tracer.
        self.shutdown_tasks
            .spawn(shutdown_otel(self.default_provider.clone()));
        // Then, wait for all all of the shutdown tasks to finish.
        wait_for_tasks_with_logging(
            &self.shutdown_tasks,
            "trace exporter shutdown",
            leak_detector,
        )
        .await;
    }
}

// Helper function that waits for a TaskTracker to finish, logging the current task count every 5 seconds.
async fn wait_for_tasks_with_logging(
    tasks: &TaskTracker,
    name: &str,
    leak_detector: Option<&SpanLeakDetector>,
) {
    tasks.close();
    IntervalStream::new(tokio::time::interval(Duration::from_secs(5)))
        .take_until(tasks.wait())
        .for_each(|_| async {
            tracing::info!(
                "Waiting for {name} tasks to finish: {} tasks remaining",
                tasks.len()
            );
            if let Some(leak_detector) = leak_detector {
                leak_detector.print_active_spans();
            }
        })
        .await;
    tracing::info!("{name} tasks finished");
}

/// This is used when `gateway.debug` is `false` and `RUST_LOG` is not set
const DEFAULT_GATEWAY_NON_DEBUG_DIRECTIVES: &str = "warn,gateway=info,tensorzero_core=info";
/// This is used when `gateway.debug` is `true` and `RUST_LOG` is not set
const DEFAULT_GATEWAY_DEBUG_DIRECTIVES: &str = "warn,gateway=debug,tensorzero_core=debug";

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

    let leak_detector = if cfg!(feature = "e2e_tests") {
        Some(SpanLeakDetector::new())
    } else {
        None
    };

    let otel_data = build_opentelemetry_layer(exporter_override);
    let (delayed_otel, otel_layer, tracer_wrapper) = match otel_data {
        Ok((delayed_otel, otel_layer, tracer_wrapper)) => (
            Ok(delayed_otel),
            Some(otel_layer),
            Some(Arc::new(tracer_wrapper)),
        ),
        Err(e) => (Err(e), None, None),
    };

    // IMPORTANT: If you add any new layers here that have per-layer filtering applied
    // you *MUST* call `apply_filter_fixing_tracing_bug` instead of `layer.with_filter(filter)`
    // See the docs for `apply_filter_fixing_tracing_bug` for more details.
    tracing_subscriber::registry()
        .with(otel_layer)
        .with(apply_filter_fixing_tracing_bug(log_layer, log_level))
        .with(leak_detector.clone())
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
        leak_detector,
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
