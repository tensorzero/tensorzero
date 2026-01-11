//! This module implements an 'overhead' metric (exposed through our `/metrics` endpoint)
//! At a high level, the metric tracks a user-controlled duration associated with a span
//!  (e.g. the time taken to process an HTTP request and send the response body),
//!  subtracting off all of the time during which an 'external' span (e.g. an outgoing model HTTP request)
//!  was running. This calculated value represents the overhead added by TensorZero, as seen by external HTTP
//!  clients.
//!
//! Overhead tracking is controlled by setting the following three attributes on tracing spans:
//! * `tensorzero.overhead.track` - Enables overhead tracking for the span.
//! * `OverheadSpanExt.set_latency_and_record` - this is a method on a `Span`, rather than an attribute.
//!   It sets the total latency of a span with overhead tracking enabled, and records the overhead metric.
//!   This must be called on the same span that has `tensorzero.overhead.track` applied.
//! * `tensorzero.overhead.external_span` - Indicates that the span is an 'external' span, and should be excluded from the overhead metric.
//!   This should be set on descendants of the span with the `tensorzero.overhead.track` attribute,
//!   (the actual value is ignored, but should be set to 'true' to make it clear what's going on)
//!
//!
//!  We currently use these attributes to track overhead on HTTP request processing spans.
//!   For example, an HTTP request might result in the following spans:
//!
//! [ request {tensorzero.overhead.track}                                                 ]
//!   [ function_call ]
//!     [model_inference]
//!       [model_provider_inference]
//!         [Outgoing HTTP request {tensorzero.overhead.external_span = true}]
//!                                                                    [send_response]
//!                                                                                    [update_rate_limiting]
//!
//!
//! In this example, we call `set_latency_and_record` value after [`send_response`] finished
//! (after we've finished sending our response body to the client.)
//! The overhead value is then computed as follows:
//! * The initial time interval starts at the instant the `request` span was created, and spans the duration set by the call to `set_latency_and_record`.
//!   Note that this does *not* include the time taken by `update_rate_limiting` - even though the `request` span was still active,
//!   the explicit `set_latency_and_record` call parameter only included the time taken until `send_response` finished,
//!   since any background processing that occurs afterward will not affect the latency seen by the original user making the HTTP request.
//! * We subtract off the time taken in the 'Outgoing HTTP request' span, since it's a descendant of the `request` span,
//!   and has 'tensorzero.overhead.external_span' attribute applied.
//!
//! Notable features:
//! * The reported metric is a histogram - each recorded value has an associated 'kind' label,
//!   controlled by the `tensorzero.overhead.track` span attribute on the top-level span.
//! * The initial duration is controlled by calling `set_latency_and_record`
//!   (on the same span as the `tensorzero.overhead.track` attribute)
//!   Notably, this will generally be *shorter* than the duration of the entire span
//!   (as measured by the time between `on_new_span` and `on_close`).
//!   We currently call `set_latency_and_record` with the response body
//!   finish time (when the response `Body` has no more chunks left).
//!   The overall HTTP span (e.g. `POST /inference`) can stay open well beyond this time -
//!   it's used for background tasks like async ClickHouse writes and rate-limit accounting,
//!   However, since these tasks occur after the response body has been sent,
//!   we don't want to count them towards the overhead metric, as they don't affect the total
//!   request time as seen by the external HTTP client.
//! * Overlapping 'external' spans are handled correctly - we track a set of disjoint time intervals
//!   representing all of the time intervals where at least one 'external' span was running.
//!   The total 'external' duration is the sum of the durations of all of the disjoint intervals.
//!   For example, a `best_of_n` variant might create the following external HTTP spans:
//!
//!   ```
//!   [ ---- Candidate 1 ---- ]
//!      [ --------- Candidate 2 ----------- ]
//!     [ --------- Candidate 3 --------- ]
//!                                              [ --- Judge ---]
//!   ```
//!
//! In this example, at least one external span was active from the time Candidate 1 started
//! until Candidate 2 finished, so that entire time interval is excluded from the overhead metric.
//! The time taken to call the judge is also excluded from our overhead metric
//! However, the time between Candidate 2 finishing
//! and the judge starting *is* counted as overhead, since no external HTTP requests were active during that time.
//!
//! The metric calculation is controlled entirely through `tracing` span attributes.
//!
use std::time::{Duration, Instant};

use crate::error::IMPOSSIBLE_ERROR_MESSAGE;
use crate::observability::disjoint_intervals::DisjointIntervals;
use metrics::Label;
use tracing::{
    Span, Subscriber,
    span::{Attributes, Id},
};
use tracing_subscriber::{
    Layer, Registry,
    layer::Context,
    registry::{ExtensionsMut, LookupSpan, SpanData},
};

/// Computes the 'overhead' of TensorZero,
/// defined as the duration of a certain span (e.g. `POST /inference`) minus
/// the duration of any specially-tagged 'external' spans (e.g. outgoing model HTTP requests).
/// The 'external' spans may run in parallel (e.g. for best_of_n/mixture_of_n), so we record
/// the time interval for each of the 'external' spans, and account for overlap
pub struct OverheadTimingLayer {
    _private: (),
}

impl OverheadTimingLayer {
    #[expect(clippy::new_without_default)]
    pub fn new() -> Self {
        Self { _private: () }
    }
}

/// Gets the tracing-subscriber `ExtensionsMut` for a span, printing an error to stdout
/// if we could not obtain the span data
fn with_span_extensions(span: &Span, f: impl FnOnce(ExtensionsMut<'_>)) {
    let res = span.with_subscriber(|(id, dispatch)| {
        if let Some(registry) = dispatch.downcast_ref::<Registry>() {
            if let Some(span_data) = registry.span_data(id) {
                f(span_data.extensions_mut());
            } else {
                error_within_tracing(&format!("No span data found for span {id:?}"));
            }
        } else {
            error_within_tracing(&format!("No registry found for span {id:?}"));
        }
    });
    if res.is_none() {
        error_within_tracing(&format!("Failed to get subscriber for span {span:?}"));
    }
}
pub trait OverheadSpanExt {
    /// Sets the total inference latency associated with the span, and record the overhead metric
    /// (subtracting off any 'external' spans that have *finished*).
    /// Note that any still in-progress 'external' spans will count towards the overhead metric.
    /// We may revisit this decision depending on whether or not we add more uses of `TensorzeroHttpClient`
    /// during inference.
    /// Any key/value pairs in `extra_labels` will be added to the `tensorzero_inference_latency_overhead_seconds` histogram metric.
    /// NOTE: This method will print an error if called on a span that does not have the `tensorzero.overhead.track` attribute applied.
    fn set_inference_latency_and_record(&self, elapsed: Duration, extra_labels: Option<Vec<Label>>);
}

impl OverheadSpanExt for Span {
    fn set_inference_latency_and_record(
        &self,
        elapsed: Duration,
        extra_labels: Option<Vec<Label>>,
    ) {
        with_span_extensions(self, |mut extensions| {
            if let Some(overhead_data) = extensions.remove::<OverheadSpanData>() {
                let excluded_time: Duration = overhead_data
                    .excluded_intervals
                    .into_disjoint_intervals()
                    .iter()
                    .map(|i| *i.end() - *i.start())
                    .sum();
                let overhead = elapsed.checked_sub(excluded_time).unwrap_or_else(|| {
                    error_within_tracing(
                        "Excluded time exceeded elapsed span duration; clamping overhead to zero",
                    );
                    Duration::ZERO
                });

                if let Some(extra_labels) = extra_labels {
                    metrics::histogram!(
                        "tensorzero_inference_latency_overhead_seconds",
                        extra_labels.clone()
                    )
                    .record(overhead.as_secs_f64());
                    // DEPRECATED (2026.2+): Also emit under the old name for backward compatibility
                    metrics::histogram!(
                        "tensorzero_inference_latency_overhead_seconds_histogram",
                        extra_labels
                    )
                    .record(overhead.as_secs_f64());
                } else {
                    metrics::histogram!("tensorzero_inference_latency_overhead_seconds")
                        .record(overhead.as_secs_f64());
                    // DEPRECATED (2026.2+): Also emit under the old name for backward compatibility
                    metrics::histogram!("tensorzero_inference_latency_overhead_seconds_histogram")
                        .record(overhead.as_secs_f64());
                }
            } else {
                error_within_tracing(&format!("No OverheadSpanData found for span {self:?}"));
            }
        });
    }
}

/// This span attribute indicates that we should track overhead for the span.
pub const TENSORZERO_TRACK_OVERHEAD_ATTRIBUTE_NAME: &str = "tensorzero.overhead.track";
/// NOTE - the value of this attribute is ignored - setting to 'false' will still enable it
pub const TENSORZERO_EXTERNAL_SPAN_ATTRIBUTE_NAME: &str = "tensorzero.overhead.external_span";

impl<S: Subscriber> Layer<S> for OverheadTimingLayer
where
    for<'a> S: LookupSpan<'a>,
{
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        for attr in attrs.fields() {
            if attr.name() == TENSORZERO_TRACK_OVERHEAD_ATTRIBUTE_NAME {
                if let Some(data) = ctx.span(id) {
                    data.extensions_mut().insert(OverheadSpanData {
                        excluded_intervals: DisjointIntervals::new(),
                    });
                } else {
                    error_within_tracing(&format!(
                        "Missing span data for `{TENSORZERO_TRACK_OVERHEAD_ATTRIBUTE_NAME}` span {id:?}"
                    ));
                }
                // A span can have at most one of our two attributes, so we can exit after seeing one
                break;
            } else if attr.name() == TENSORZERO_EXTERNAL_SPAN_ATTRIBUTE_NAME {
                // We don't care about the value of `TENSORZERO_EXTERNAL_SPAN_ATTRIBUTE_NAME`, just that the attribute exists
                if let Some(data) = ctx.span(id) {
                    data.extensions_mut().insert(ExternalSpanData {
                        start_time: Instant::now(),
                    });
                } else {
                    error_within_tracing(&format!(
                        "Missing span data for `tensorzero.overhead.external_span` span {id:?}"
                    ));
                }
                // A span can have at most one of our two attributes, so we can exit after seeing one
                break;
            }
        }
    }

    fn on_close(&self, id: Id, ctx: Context<'_, S>) {
        if let Some(data) = ctx.span(&id)
            && let Some(external_data) = data.extensions().get::<ExternalSpanData>()
        {
            if let Some(scope) = ctx.span_scope(&id) {
                // Skip over the current span
                for parent in scope.skip(1) {
                    let mut parent_exts = parent.extensions_mut();
                    if let Some(overhead_data) = parent_exts.get_mut::<OverheadSpanData>() {
                        overhead_data
                            .excluded_intervals
                            .add_interval(external_data.start_time..=Instant::now());
                        // For now, we only support at most one `tensorzero.track_overhead` span per trace
                        // (we currently only set this at the top level, when receiving an incoming HTTP request)
                        return;
                    }
                }
                // We might create 'external' spans outside of an overhead-tracked span
                // (e.g. when using TensorzeroHttpClient in a background task), so don't error if
                // we didn't find a parent span with `OverheadSpanData`
            } else {
                error_within_tracing(
                    format!("Failed to find span scope for external span {id:?}").as_str(),
                );
            }
        }
    }
}

// To avoid complicated re-entrant logging, we use `eprintln` for any errors that occur within our tracing code
// Constructing our `Error` type or using `tracing::error!` would cause our `OverheadTimingLayer` to potentially
// get called re-entrantly (while the original span is still active).
fn error_within_tracing(message: &str) {
    #![expect(clippy::print_stderr)]
    eprintln!(
        "ERROR: Internal error in TensorZero tracing code: {message}. {IMPOSSIBLE_ERROR_MESSAGE}"
    );
}

/// Marks spans with the `tensorzero.overhead.track` attribute applied.
pub struct OverheadSpanData {
    excluded_intervals: DisjointIntervals<Instant>,
}

/// Marks spans with the `tensorzero.overhead.external_span` attribute applied.
pub struct ExternalSpanData {
    start_time: Instant,
}
