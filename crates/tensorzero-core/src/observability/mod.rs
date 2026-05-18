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

mod exporter_wrapper;
pub mod genai_conventions;
pub mod internal_metrics;
pub mod openinference_conventions;
pub mod request_logging;
mod span_leak_detector;
pub mod tracing_bug;
mod tracing_otel;

pub use tracing_otel::*;
