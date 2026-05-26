pub mod error;
pub mod exporter_wrapper;
pub mod span_leak_detector;
pub mod tracing_bug;

pub use error::ObservabilityError;

use std::borrow::Cow;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::MatchedPath;
use axum::extract::State;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::{Router, middleware};
use clap::ValueEnum;
use futures::StreamExt;
use http::HeaderMap;
use metrics::{Unit, describe_counter, describe_histogram};
use metrics_exporter_prometheus::{Matcher, PrometheusBuilder, PrometheusHandle};
use moka::sync::Cache;
use once_cell::sync::OnceCell;
#[cfg(feature = "e2e_tests")]
use opentelemetry::ContextGuard;
use opentelemetry::trace::Status;
use opentelemetry::trace::{Tracer, TracerProvider as _};
use opentelemetry::{Context, KeyValue};
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_otlp::WithHttpConfig;
use opentelemetry_otlp::WithTonicConfig;
use opentelemetry_otlp::tonic_types::metadata::MetadataMap;
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::trace::SdkTracer;
use opentelemetry_sdk::trace::{SdkTracerProvider, SpanExporter};
use tensorzero_overhead::OverheadTimingLayer;
use tokio_stream::wrappers::IntervalStream;
use tokio_util::task::TaskTracker;
use tokio_util::task::task_tracker::TaskTrackerToken;
use tonic::metadata::AsciiMetadataKey;
use tonic::metadata::MetadataValue;
use tracing::field::Empty;
use tracing::level_filters::LevelFilter;
use tracing::{Metadata, Span};
use tracing_futures::Instrument;
use tracing_opentelemetry::OpenTelemetrySpanExt;
use tracing_opentelemetry_instrumentation_sdk::http::{
    http_flavor, http_host, url_scheme, user_agent,
};
use tracing_subscriber::layer::Filter;
use tracing_subscriber::{EnvFilter, Registry, filter};
use tracing_subscriber::{Layer, layer::SubscriberExt, util::SubscriberInitExt};
use uuid::Uuid;

use crate::error::ObservabilityError as Error;
use crate::exporter_wrapper::TensorZeroExporterWrapper;
use crate::span_leak_detector::SpanLeakDetector;
use crate::tracing_bug::apply_filter_fixing_tracing_bug;

#[derive(Clone, Debug, Default, ValueEnum)]
pub enum LogFormat {
    #[default]
    Pretty,
    Json,
}

/// Transport used for OTLP trace export. Mirrors the OpenTelemetry
/// `OTEL_EXPORTER_OTLP_PROTOCOL` spec values, restricted to the two transports
/// we actually wire up: gRPC (over tonic) and HTTP with protobuf payloads.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, ValueEnum)]
pub enum OtlpProtocol {
    /// gRPC over HTTP/2 using `tonic`. The default OTLP transport.
    #[default]
    Grpc,
    /// HTTP/1.1 with protobuf-encoded payloads (`http/protobuf` in the OTel
    /// spec). Uses `reqwest` under the hood.
    HttpBinary,
}

/// Knobs that used to be hardcoded inside this crate. Callers pass these in so
/// the same plumbing can drive any gateway — `service.name`, the tracer name,
/// the overhead histogram label, and the env-filter directives all come from
/// here instead of `"tensorzero-gateway"` / `"tensorzero"` constants.
#[derive(Clone, Debug)]
pub struct ObservabilitySettings {
    /// Value of the `service.name` OpenTelemetry resource attribute.
    pub service_name: &'static str,
    /// Name passed to `TracerProvider::tracer(...)` when building the default
    /// tracer. Surfaces as the `otel.library.name` attribute on emitted spans.
    pub tracer_name: &'static str,
    /// Histogram name recorded by [`OverheadTimingLayer`]. Only used when
    /// `register_overhead_layer` is true.
    pub overhead_metric_name: &'static str,
    /// Base prefix for the custom OTLP header families that callers attach to
    /// incoming requests. The full prefixes are formed by joining this value, a
    /// `-` separator, and a fixed suffix, e.g.
    /// `{otlp_header_prefix}-otlp-traces-extra-header-`,
    /// `{otlp_header_prefix}-otlp-traces-extra-resource-`, and
    /// `{otlp_header_prefix}-otlp-traces-extra-attribute-`. Defaults to
    /// [`DEFAULT_OTLP_HEADER_PREFIX`] (`"tensorzero"`); downstream consumers
    /// can rebrand by setting their own value (without a trailing `-`).
    pub otlp_header_prefix: &'static str,
    /// Default `EnvFilter` directives used when `RUST_LOG` is unset and debug
    /// logging is not enabled. Should be a comma-separated set of
    /// `target=level` pairs (e.g. `"warn,gateway=info"`).
    pub default_log_directives: &'static str,
    /// Default `EnvFilter` directives used when `RUST_LOG` is unset and the
    /// caller invokes `DelayedDebugLogs::enable_debug`. Same format as
    /// `default_log_directives`.
    pub debug_log_directives: &'static str,
    /// If true, registers the [`OverheadTimingLayer`]. Set to false for
    /// non-HTTP entry points (e.g. embedded clients) where no top-level
    /// overhead span exists.
    pub register_overhead_layer: bool,
    /// Transport to use for OTLP trace export. Controls whether the
    /// underlying `SpanExporter` is built with `.with_tonic()` (gRPC) or
    /// `.with_http()` (HTTP/protobuf). The endpoint and other knobs are still
    /// read from the standard `OTEL_*` environment variables.
    pub otlp_protocol: OtlpProtocol,
}

/// TensorZero gateway defaults — preserves the strings that lived inside this
/// crate before it was generalized. New consumers should construct their own
/// [`ObservabilitySettings`] with project-specific names.
pub const TENSORZERO_DEFAULTS: ObservabilitySettings = ObservabilitySettings {
    service_name: "tensorzero-gateway",
    tracer_name: "tensorzero",
    overhead_metric_name: "tensorzero_inference_latency_overhead_seconds",
    otlp_header_prefix: DEFAULT_OTLP_HEADER_PREFIX,
    default_log_directives: "warn,gateway=info,tensorzero_core=info,tensorzero_otel=info",
    debug_log_directives: "warn,gateway=debug,tensorzero_core=debug,tensorzero_otel=debug",
    register_overhead_layer: true,
    otlp_protocol: OtlpProtocol::Grpc,
};

/// Same as [`TENSORZERO_DEFAULTS`] but with `register_overhead_layer = false`,
/// for use in embedded-client style entry points (e.g. the Python bindings).
pub const TENSORZERO_EMBEDDED_DEFAULTS: ObservabilitySettings = ObservabilitySettings {
    service_name: TENSORZERO_DEFAULTS.service_name,
    tracer_name: TENSORZERO_DEFAULTS.tracer_name,
    overhead_metric_name: TENSORZERO_DEFAULTS.overhead_metric_name,
    otlp_header_prefix: TENSORZERO_DEFAULTS.otlp_header_prefix,
    default_log_directives: TENSORZERO_DEFAULTS.default_log_directives,
    debug_log_directives: TENSORZERO_DEFAULTS.debug_log_directives,
    register_overhead_layer: false,
    otlp_protocol: TENSORZERO_DEFAULTS.otlp_protocol,
};

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
    // Captured settings (service.name, tracer name, etc.) — used when this
    // wrapper lazily builds a `CustomTracer` for a request that asks for extra
    // OTLP headers via `tensorzero-otlp-traces-extra-header-*`.
    settings: ObservabilitySettings,
}

impl TracerWrapper {
    /// Returns the `TaskTracker` used to keep top-level HTTP spans alive until
    /// their (possibly background-spawned) descendants have closed. Consumers
    /// that own the HTTP middleware should issue per-request tokens by calling
    /// `.token()` on this tracker so graceful shutdown can wait for them.
    pub fn in_flight_spans(&self) -> &TaskTracker {
        &self.in_flight_spans
    }
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
        "/../tensorzero-core/tests/e2e/self-signed-certs/otlp-collector.crt"
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
                let (provider, tracer) = build_tracer::<opentelemetry_otlp::SpanExporter>(
                    key.clone(),
                    None,
                    &self.settings,
                )?;
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
    settings: &ObservabilitySettings,
) -> Result<(SdkTracerProvider, SdkTracer), Error> {
    let CustomTracerKey {
        extra_headers,
        extra_resources,
        extra_attributes,
    } = key;

    // Per the OTel spec, `OTEL_SERVICE_NAME` overrides any service.name set
    // in code. Honor that — callers' `settings.service_name` is the default,
    // not a hardcoded floor.
    let service_name: String = std::env::var("OTEL_SERVICE_NAME")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| settings.service_name.to_owned());
    let mut builder = SdkTracerProvider::builder().with_resource(
        Resource::builder()
            .with_attribute(KeyValue::new(
                opentelemetry_semantic_conventions::resource::SERVICE_NAME,
                service_name,
            ))
            .with_attributes(extra_resources)
            .build(),
    );

    if let Some(override_exporter) = override_exporter {
        builder = builder.with_simple_exporter(TensorZeroExporterWrapper::new(
            override_exporter,
            extra_attributes,
        ));
    } else {
        builder = match settings.otlp_protocol {
            OtlpProtocol::Grpc => {
                let tls_config = tonic::transport::ClientTlsConfig::new().with_enabled_roots();
                #[cfg(feature = "e2e_tests")]
                let tls_config = add_local_self_signed_cert(tls_config);
                let exporter = opentelemetry_otlp::SpanExporter::builder()
                    .with_tonic()
                    .with_metadata(extra_headers)
                    .with_tls_config(tls_config)
                    .build()
                    .map_err(|e| {
                        Error::observability(format!("Failed to create OTLP gRPC exporter: {e}"))
                    })?;
                builder
                    .with_batch_exporter(TensorZeroExporterWrapper::new(exporter, extra_attributes))
            }
            OtlpProtocol::HttpBinary => {
                let headers = metadata_to_http_headers(&extra_headers)?;
                let exporter = opentelemetry_otlp::SpanExporter::builder()
                    .with_http()
                    .with_headers(headers)
                    .with_protocol(opentelemetry_otlp::Protocol::HttpBinary)
                    .build()
                    .map_err(|e| {
                        Error::observability(format!("Failed to create OTLP HTTP exporter: {e}"))
                    })?;
                builder
                    .with_batch_exporter(TensorZeroExporterWrapper::new(exporter, extra_attributes))
            }
        };
    }
    let provider = builder.build();

    let tracer = provider.tracer(settings.tracer_name);
    Ok((provider, tracer))
}

/// Convert a tonic `MetadataMap` (the gRPC OTLP exporter's native header shape)
/// into the `HashMap<String, String>` that the HTTP OTLP exporter expects.
/// Binary metadata values (`*-bin` keys) aren't valid HTTP header values and
/// will produce an error rather than being silently dropped.
fn metadata_to_http_headers(metadata: &MetadataMap) -> Result<HashMap<String, String>, Error> {
    let mut out = HashMap::with_capacity(metadata.len());
    for (name, value) in metadata.as_ref() {
        let value_str = value.to_str().map_err(|e| {
            Error::observability(format!(
                "Failed to convert OTLP header `{}` value to string for HTTP export: {e}",
                name.as_str()
            ))
        })?;
        out.insert(name.as_str().to_string(), value_str.to_string());
    }
    Ok(out)
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
    settings: &ObservabilitySettings,
) -> Result<OtelLayerData<impl Layer<Registry>>, Error> {
    // Default tracer always has empty headers and no extra resources
    let (provider, tracer) = build_tracer(
        CustomTracerKey {
            extra_headers: MetadataMap::new(),
            extra_resources: vec![],
            extra_attributes: vec![],
        },
        override_exporter,
        settings,
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
        settings: settings.clone(),
    };

    // Cloning of these types internally preserves a reference - we don't need our own `Arc` here
    let cloned_wrapper = TracerWrapper {
        default_tracer: wrapper.default_tracer.clone(),
        default_provider: wrapper.default_provider.clone(),
        static_otlp_traces_extra_headers: wrapper.static_otlp_traces_extra_headers.clone(),
        custom_tracers: wrapper.custom_tracers.clone(),
        shutdown_tasks: wrapper.shutdown_tasks.clone(),
        in_flight_spans: wrapper.in_flight_spans.clone(),
        settings: wrapper.settings.clone(),
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
        provider
            .shutdown_with_timeout(Duration::MAX)
            .map_err(|e| Error::observability(format!("Failed to shutdown OpenTelemetry: {e}")))?;
        tracing::debug!(tracer_id = id.to_string(), "Custom tracer shut down");
        Ok::<_, Error>(())
    })
    .await
    .map_err(|e| {
        Error::observability(format!("Failed to wait on OpenTelemetry shutdown: {e}"))
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
    settings: &ObservabilitySettings,
) -> Result<(DelayedOtelEnableHandle, impl Layer<Registry>, TracerWrapper), Error> {
    let (otel_reload_filter, reload_handle) = tracing_subscriber::reload::Layer::new(Box::new(
        LevelFilter::OFF,
    )
        as Box<dyn Filter<_> + Send + Sync>);

    let OtelLayerData {
        layer: base_otel_layer,
        wrapper,
    } = internal_build_otel_layer(override_exporter, settings)?;

    let delayed_enable = DelayedOtelEnableHandle {
        enable_cb: Box::new(move || {
            // Only register the propagator if we actually enabled OTEL.
            // This means that the `traceparent` and `tracestate` headers will only be added
            // to outgoing requests using the propagator if OTEL is actually enabled.
            init_tracing_opentelemetry::init_propagator().map_err(|e| {
                Error::observability(format!("Failed to initialize OTLP propagator: {e}"))
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
                if crate::tracing_bug::DISABLE_TRACING_BUG_WORKAROUND
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
                    Error::observability(format!("Failed to enable OTLP exporter: {e:?}"))
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
        error_extractor: ResponseErrorExtractor,
    ) -> Self;
}

/// A special header prefix used to attach an additional header to the OTLP export for this trace.
/// The format is: `{prefix}-otlp-traces-extra-header-HEADER_NAME: HEADER_VALUE`, where `{prefix}`
/// is [`ObservabilitySettings::otlp_header_prefix`] (default `tensorzero`).
/// For each header with the `{prefix}-otlp-traces-extra-header-` prefix, we add `HEADER_NAME: HEADER_VALUE`
/// to the OTLP export HTTP/gRPC request headers.
///
/// When an incoming request has a `{prefix}-otlp-traces-extra-header-` header, we handle it through
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
///
/// The leading `tensorzero` portion of these prefixes is configurable via
/// [`ObservabilitySettings::otlp_header_prefix`]; the `-` separator and the
/// fixed suffixes below are not.
const OTLP_TRACES_EXTRA_HEADER_SUFFIX: &str = "otlp-traces-extra-header-";
const OTLP_TRACES_EXTRA_RESOURCE_SUFFIX: &str = "otlp-traces-extra-resource-";
const OTLP_TRACES_EXTRA_ATTRIBUTE_SUFFIX: &str = "otlp-traces-extra-attribute-";

/// Default value for [`ObservabilitySettings::otlp_header_prefix`]. Downstream
/// consumers can override this to rebrand the custom OTLP header families. The
/// `-` separator before the suffix is appended automatically, so this value
/// should not include a trailing `-`.
pub const DEFAULT_OTLP_HEADER_PREFIX: &str = "tensorzero";

// The full custom OTLP header-family prefixes for a default (`tensorzero`)
// deployment. These are kept as public constants so clients targeting a default
// TensorZero gateway can prefix their headers without recomputing the strings.
// A gateway built with a custom `otlp_header_prefix` derives its own prefixes at
// runtime via `OtlpHeaderPrefixes::from_base`; the unit tests below assert these
// constants stay in sync with `DEFAULT_OTLP_HEADER_PREFIX` plus the suffixes.
pub const TENSORZERO_OTLP_HEADERS_PREFIX: &str = "tensorzero-otlp-traces-extra-header-";
pub const TENSORZERO_OTLP_RESOURCE_PREFIX: &str = "tensorzero-otlp-traces-extra-resource-";
pub const TENSORZERO_OTLP_ATTRIBUTE_PREFIX: &str = "tensorzero-otlp-traces-extra-attribute-";

/// The three custom OTLP header-family prefixes, derived from a configurable
/// base prefix (e.g. `tensorzero`), a `-` separator, and the fixed suffixes.
/// Built per request in [`extract_tensorzero_headers`] from
/// [`ObservabilitySettings::otlp_header_prefix`].
struct OtlpHeaderPrefixes {
    header: String,
    resource: String,
    attribute: String,
}

impl OtlpHeaderPrefixes {
    fn from_base(base: &str) -> Self {
        // The `-` separator between the base prefix and the suffix is appended
        // here, so callers configure the base without a trailing `-`.
        Self {
            header: format!("{base}-{OTLP_TRACES_EXTRA_HEADER_SUFFIX}"),
            resource: format!("{base}-{OTLP_TRACES_EXTRA_RESOURCE_SUFFIX}"),
            attribute: format!("{base}-{OTLP_TRACES_EXTRA_ATTRIBUTE_SUFFIX}"),
        }
    }
}

/// Converts a HashMap of config headers to a MetadataMap
fn config_headers_to_metadata(
    config_headers: &HashMap<String, String>,
) -> Result<MetadataMap, Error> {
    let mut metadata = MetadataMap::new();
    for (name, value) in config_headers {
        let key: AsciiMetadataKey = name.parse().map_err(|e| {
            Error::observability(format!(
                "Failed to parse config header `{name}` as valid metadata key: {e}"
            ))
        })?;
        let value = MetadataValue::from_str(value).map_err(|e| {
            Error::observability(format!(
                "Failed to parse config header `{name}` value as valid metadata value: {e}"
            ))
        })?;
        metadata.insert(key, value);
    }
    Ok(metadata)
}

fn json_to_otel_value(value: serde_json::Value) -> Result<opentelemetry::Value, Error> {
    match value {
        serde_json::Value::Null => Err(Error::invalid_request(
            "Null is not a valid OpenTelemetry attribute value".to_string(),
        )),
        serde_json::Value::Bool(value) => Ok(opentelemetry::Value::Bool(value)),
        serde_json::Value::Number(_) => Err(Error::invalid_request(
            "Numbers are not yet supported for OpenTelemetry attributes values".to_string(),
        )),
        serde_json::Value::String(value) => Ok(opentelemetry::Value::String(value.into())),
        serde_json::Value::Array(_) => Err(Error::invalid_request(
            "Arrays are not yet supported for OpenTelemetry attribute values".to_string(),
        )),
        serde_json::Value::Object(_) => Err(Error::invalid_request(
            "JSON objects are not valid OpenTelemetry attribute values".to_string(),
        )),
    }
}

// Removes all of the headers prefixed with the `otlp-traces-extra-header-` family
// (using the configurable base prefix from `ObservabilitySettings::otlp_header_prefix`).
// If any are present (or we have static config headers), constructs a `CustomTracerKey` with all of the matching header/value pairs
// (with the prefix removed from the header name).
// We also apply any static custom OTLP headers set in the `TracerWrapper`.
fn extract_tensorzero_headers(
    tracer_wrapper: &TracerWrapper,
    headers: &HeaderMap,
) -> Result<Option<CustomTracerKey>, Error> {
    let OtlpHeaderPrefixes {
        header: header_prefix,
        resource: resource_prefix,
        attribute: attribute_prefix,
    } = OtlpHeaderPrefixes::from_base(tracer_wrapper.settings.otlp_header_prefix);
    // Merge config headers with dynamic headers (dynamic takes precedence)
    let mut metadata = tracer_wrapper
        .static_otlp_traces_extra_headers
        .get()
        .cloned()
        .unwrap_or_default();
    let mut extra_resources = vec![];
    let mut extra_attributes = vec![];
    for (name, value) in headers {
        if let Some(suffix) = name.as_str().strip_prefix(header_prefix.as_str()) {
            let key: AsciiMetadataKey = suffix.parse().map_err(|e| {
                Error::observability(format!(
                    "Failed to parse `{header_prefix}` header `{suffix}` as valid metadata key: {e}"
                ))
            })?;
            let value = MetadataValue::from_str(value.to_str().map_err(|e| {
                Error::observability(format!("Failed to parse `{header_prefix}` header `{suffix}` value as valid string: {e}"))
            })?).map_err(|e| {
                Error::observability(format!("Failed to parse `{header_prefix}` header `{suffix}` value as valid metadata value: {e}"))
            })?;
            metadata.insert(key, value);
        }
        if let Some(suffix) = name.as_str().strip_prefix(resource_prefix.as_str()) {
            let key = suffix.to_string();
            let value = value.to_str().map_err(|e| {
                Error::invalid_request(format!("Failed to parse `{resource_prefix}` header `{suffix}` value as valid string: {e}"))
            })?.to_string();
            extra_resources.push(KeyValue::new(key, value));
        }
        if let Some(suffix) = name.as_str().strip_prefix(attribute_prefix.as_str()) {
            let key = suffix.to_string();
            let value = value.to_str().map_err(|e| {
                Error::invalid_request(format!("Failed to parse `{attribute_prefix}` header `{suffix}` value as valid string: {e}"))
            })?;
            let value_json = serde_json::from_str::<serde_json::Value>(value).map_err(|e| {
                Error::invalid_request(format!("Failed to parse `{attribute_prefix}` header `{suffix}` value as valid JSON: {e}"))
            })?;
            let value_otel = json_to_otel_value(value_json).map_err(|e| {
                Error::invalid_request(format!("Failed to convert `{attribute_prefix}` header `{suffix}` value to OpenTelemetry attribute value: {e}"))
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

impl InFlightOtelOnlySpan {
    /// Wrap a `TaskTrackerToken` in an `InFlightOtelOnlySpan` marker that the
    /// OTel filter recognizes. Typically used by consumers that drive their
    /// own top-level HTTP middleware (instead of `apply_top_level_otel_http_trace_layer`)
    /// and want their server-side spans to flow through OTel export.
    /// The token should come from `TracerWrapper::in_flight_spans().token()`.
    pub fn new(token: TaskTrackerToken) -> Self {
        Self { token }
    }
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

/// Callback that lets a consumer extract a human-readable error description
/// from response extensions. Used by [`handle_response`] to mark spans with
/// [`Status::Error`] when the response carries an application-specific error
/// type. Return `None` to fall back to status-code-based detection.
///
/// The callback is plain `fn` (not `Box<dyn Fn>`) so the middleware state
/// stays `Clone`-cheap and `'static`. Consumers that need richer state should
/// stash it in response extensions and extract it here.
pub type ResponseErrorExtractor = fn(&http::Extensions) -> Option<String>;

/// Default error extractor: never marks a response-extension-based error.
/// Server-side (5xx) status still triggers a generic `Status::Error`.
pub fn default_response_error_extractor(_ext: &http::Extensions) -> Option<String> {
    None
}

/// Attach information from our HTTP response to the original span for the overall
/// HTTP request processing
fn handle_response<B>(res: &Response<B>, span: &Span, error_extractor: ResponseErrorExtractor) {
    // We cast this to an i64, so that tracing-opentelemetry will record it as an integer
    // rather than a string
    span.record("http.response.status_code", res.status().as_u16() as i64);

    if let Some(description) = error_extractor(res.extensions()) {
        span.set_status(Status::Error {
            description: Cow::Owned(description),
        });
    } else if res.status().is_server_error() {
        // Don't set a description for non-application errors,
        // since we don't know what a nice description should look like
        span.set_status(Status::Error {
            description: Cow::Owned(String::new()),
        });
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
        error_extractor,
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
    if let Some(route) = route
        && otel_enabled_routes.routes.contains(&route)
    {
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
        handle_response(&response, &span, error_extractor);
        return response;
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
    error_extractor: ResponseErrorExtractor,
}

impl<S: Clone + Send + Sync + 'static> RouterExt<S> for Router<S> {
    /// Creates tracing spans for HTTP requests, specialized for OpenTelemetry traces
    /// Note that this is applied to *all* routes, not just OpenTelemetry-enabled routes.
    /// The `otel_enabled_routes` parameter is used to determine whether to create a span for the request.
    /// See the docs on `tensorzero_otel_tracing_middleware` for more details.
    ///
    /// `error_extractor` is invoked on each response's extensions to find an
    /// application-specific error description. Pass [`default_response_error_extractor`]
    /// (or `|_| None`) if you don't need this hook.
    fn apply_top_level_otel_http_trace_layer(
        self,
        otel_tracer: Option<Arc<TracerWrapper>>,
        otel_enabled_routes: OtelEnabledRoutes,
        error_extractor: ResponseErrorExtractor,
    ) -> Self {
        // If OpenTelemetry is disable, then we don't need to create extra spans
        if let Some(tracer) = otel_tracer {
            self.layer(middleware::from_fn_with_state(
                TracingMiddlewareState {
                    tracer_wrapper: tracer,
                    otel_enabled_routes: Arc::new(otel_enabled_routes),
                    error_extractor,
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
                Error::observability(
                    "Failed to set static OTLP headers: already initialized".to_string(),
                )
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
/// 2. If `gateway.debug` is set in the config file, use `settings.debug_log_directives`
/// 3. Otherwise, use `settings.default_log_directives`
///
/// The case of unset `RUST_LOG` and `gateway.debug = true` is special:
/// We initialize our filter with `settings.default_log_directives`,
/// and then later override it (with `DelayedDebugLogs`) to `settings.debug_log_directives`.
/// This allows us to still see warnings/errors that occur during config file parsing.
///
/// In all other cases, the filter is set once during initialization, and then never changed.
///
/// Strictly speaking, this does not need to be an async function.
/// However, the call to `build_opentelemetry_layer` requires a Tokio runtime,
/// so marking this function as async makes it clear to callers that they need to
/// be in an async context.
pub async fn setup_observability(
    log_format: LogFormat,
    settings: ObservabilitySettings,
) -> Result<ObservabilityHandle, Error> {
    // We need to provide a dummy generic parameter to satisfy the compiler
    setup_observability_with_exporter_override::<opentelemetry_otlp::SpanExporter>(
        log_format, None, settings,
    )
    .await
}

#[expect(clippy::unused_async)]
pub async fn setup_observability_with_exporter_override<T: SpanExporter + 'static>(
    log_format: LogFormat,
    exporter_override: Option<T>,
    settings: ObservabilitySettings,
) -> Result<ObservabilityHandle, Error> {
    let env_var_name = "RUST_LOG";
    let has_env_var = std::env::var(env_var_name).is_ok();

    let default_debug_filter = EnvFilter::builder()
        .parse(settings.debug_log_directives)
        .map_err(|e| {
            Error::internal(format!(
                "Failed to parse internal debug directives - this should never happen: {e}"
            ))
        })?;

    // If the `RUST_LOG` env var is set, then use it as our filter.
    // Otherwise, use the default non-debug directives (which might later get overridden to settings.debug_log_directives
    // using the `update_log_level` handle).
    let base_filter = if has_env_var {
        EnvFilter::builder()
            .with_env_var(env_var_name)
            .from_env()
            .map_err(|e| {
                Error::observability(format!(
                    "Invalid `{env_var_name}` environment variable: {e}"
                ))
            })?
    } else {
        EnvFilter::builder()
            .parse(settings.default_log_directives)
            .map_err(|e| {
                Error::internal(format!(
                    "Failed to parse internal non-debug directives - this should never happen: {e}"
                ))
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

    let otel_data = build_opentelemetry_layer(exporter_override, &settings);
    let (delayed_otel, otel_layer, tracer_wrapper) = match otel_data {
        Ok((delayed_otel, otel_layer, tracer_wrapper)) => (
            Ok(delayed_otel),
            Some(otel_layer),
            Some(Arc::new(tracer_wrapper)),
        ),
        Err(e) => (Err(e), None, None),
    };

    // This layer only makes sense when we construct top-level HTTP overhead-tracking spans
    let overhead_timing_layer = settings
        .register_overhead_layer
        .then(|| OverheadTimingLayer::new(settings.overhead_metric_name));

    // IMPORTANT: If you add any new layers here that have per-layer filtering applied
    // you *MUST* call `apply_filter_fixing_tracing_bug` instead of `layer.with_filter(filter)`
    // See the docs for `apply_filter_fixing_tracing_bug` for more details.
    tracing_subscriber::registry()
        .with(otel_layer)
        .with(apply_filter_fixing_tracing_bug(log_layer, log_level))
        .with(leak_detector.clone())
        .with(overhead_timing_layer)
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
                    .map_err(|e| Error::observability(format!("Failed to update log level: {e}")))
            }),
        }
    };

    // This is needed for for the `redis` crate to work in HTTPs mode
    // We call this in `setup_observability` since this is an 'application'-style
    // entrypoint (e.g. any eventual external consumers of the Rust client will
    // *not* call this, allowing the top-level application to decide the crypto provider.)
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();

    Ok(ObservabilityHandle {
        delayed_otel,
        delayed_debug_logs,
        otel_tracer: tracer_wrapper,
        leak_detector,
    })
}

/// Set up Prometheus metrics exporter.
///
/// `inference_latency_overhead_buckets` should contain the histogram buckets for the
/// `tensorzero_inference_latency_overhead_seconds` metric. Pass an empty slice to disable the metric.
pub fn setup_metrics(
    inference_latency_overhead_buckets: &[f64],
) -> Result<PrometheusHandle, Error> {
    let mut builder = PrometheusBuilder::new();

    if !inference_latency_overhead_buckets.is_empty() {
        // Set buckets for the metric
        builder = builder
            .set_buckets_for_metric(
                Matcher::Full("tensorzero_inference_latency_overhead_seconds".to_string()),
                inference_latency_overhead_buckets,
            )
            .map_err(|e| Error::observability(format!("Failed to set histogram buckets: {e}")))?;
    }

    let metrics_handle = builder
        .install_recorder()
        .map_err(|e| Error::observability(format!("Failed to install Prometheus exporter: {e}")))?;
    let handle_clone = metrics_handle.clone();
    // Metrics are pull-based via the `/metrics` endpoint, so we don't
    // need to do anything on shutdown - we don't track this task because
    // it's a best-effort background task.
    #[expect(clippy::disallowed_methods)]
    tokio::spawn(async move {
        loop {
            // metrics-exporter-prometheus defaults to 5 seconds for `upkeep_timeout`
            // when using `install()`
            tokio::time::sleep(Duration::from_secs(5)).await;
            handle_clone.run_upkeep();
        }
    });

    // Register the expected metrics along with their types and docstrings
    describe_counter!(
        "tensorzero_requests_total",
        Unit::Count,
        "Requests handled by TensorZero",
    );

    describe_counter!(
        "tensorzero_inferences_total",
        Unit::Count,
        "Inferences performed by TensorZero",
    );

    describe_counter!(
        "tensorzero_input_tokens_total",
        Unit::Count,
        "Input tokens consumed by TensorZero inferences",
    );

    describe_counter!(
        "tensorzero_output_tokens_total",
        Unit::Count,
        "Output tokens consumed by TensorZero inferences",
    );

    if !inference_latency_overhead_buckets.is_empty() {
        describe_histogram!(
            "tensorzero_inference_latency_overhead_seconds",
            Unit::Seconds,
            "Overhead of TensorZero on HTTP requests. You can customize buckets using `gateway.metrics.tensorzero_inference_latency_overhead_seconds_buckets` in the configuration."
        );
    }

    Ok(metrics_handle)
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_OTLP_HEADER_PREFIX, OtlpHeaderPrefixes, TENSORZERO_DEFAULTS,
        TENSORZERO_OTLP_ATTRIBUTE_PREFIX, TENSORZERO_OTLP_HEADERS_PREFIX,
        TENSORZERO_OTLP_RESOURCE_PREFIX,
    };
    use googletest::prelude::*;

    /// The public `TENSORZERO_OTLP_*_PREFIX` constants are used by clients targeting a
    /// default gateway. Ensure they stay in sync with the prefixes derived from
    /// `DEFAULT_OTLP_HEADER_PREFIX` plus the fixed suffixes, so a default gateway and a
    /// default client agree on the header names.
    #[gtest]
    fn default_prefix_constants_match_derived_prefixes() {
        let prefixes = OtlpHeaderPrefixes::from_base(DEFAULT_OTLP_HEADER_PREFIX);
        expect_that!(prefixes.header, eq(TENSORZERO_OTLP_HEADERS_PREFIX));
        expect_that!(prefixes.resource, eq(TENSORZERO_OTLP_RESOURCE_PREFIX));
        expect_that!(prefixes.attribute, eq(TENSORZERO_OTLP_ATTRIBUTE_PREFIX));
    }

    /// The default `ObservabilitySettings` should use the default base prefix.
    #[gtest]
    fn tensorzero_defaults_use_default_prefix() {
        expect_that!(
            TENSORZERO_DEFAULTS.otlp_header_prefix,
            eq(DEFAULT_OTLP_HEADER_PREFIX)
        );
    }

    /// A custom base prefix is reflected in all three derived header families, with the
    /// `-` separator appended automatically (the base carries no trailing `-`).
    #[gtest]
    fn custom_prefix_is_applied_to_all_families() {
        let prefixes = OtlpHeaderPrefixes::from_base("acme");
        expect_that!(prefixes.header, eq("acme-otlp-traces-extra-header-"));
        expect_that!(prefixes.resource, eq("acme-otlp-traces-extra-resource-"));
        expect_that!(prefixes.attribute, eq("acme-otlp-traces-extra-attribute-"));
    }
}
