use std::{
    cell::Cell,
    sync::{
        Arc,
        atomic::{AtomicU32, Ordering},
    },
    time::{Duration, Instant},
};

use http::StatusCode;
use metrics::Label;
use tracing::Span;

use crate::OverheadSpanExt;

/// A *response* extension used to pass data from a route handler to `request_logging_middleware`
/// See the `inference` route handler for an example of how to use this.
#[derive(Clone)]
pub struct HttpMetricData {
    /// Extra labels to add to the `tensorzero_inference_latency_overhead_seconds` metric.
    /// We currently use this to attach `function_name`, `variant_name`, and `model_name` labels
    /// when recording the overhead of `/inference` requests
    pub extra_overhead_labels: Vec<Label>,
}

/// A drop guard that logs a message on drop if `start_time` is set.
pub struct ConnectionDropGuard {
    // The counter to decrement when the request is finished
    count_per_route: Arc<AtomicU32>,
    latency_span: Option<Span>,
    request_logging_data: Option<HttpMetricData>,
    span: Span,
    start_time: Instant,
    finished_with_latency: Cell<Option<Duration>>,
    status: Option<StatusCode>,
}

impl ConnectionDropGuard {
    pub fn new(
        count_per_route: Arc<AtomicU32>,
        latency_span: Option<Span>,
        span: Span,
        start_time: Instant,
    ) -> Self {
        Self {
            count_per_route,
            latency_span,
            request_logging_data: None,
            span,
            start_time,
            finished_with_latency: Cell::new(None),
            status: None,
        }
    }

    pub fn set_request_logging_data(&mut self, data: Option<HttpMetricData>) {
        self.request_logging_data = data;
    }

    pub fn set_status(&mut self, status: StatusCode) {
        self.status = Some(status);
    }

    // Mark the guard as explicitly finished by the server.
    // This suppresses the warning that we would otherwise log in the `Drop` impl.
    // Note that we call this method even if the server produces an error - the purpose
    // of this method is to detect early drops, when the server didn't produce a response of any kind.
    pub fn mark_finished(&self) {
        // Calculate the elapsed time when we've finished sending the response to
        // the client - this is the latency that we want to log to users,
        // and use for computing the `tensorzero_inference_latency_overhead_seconds` metric
        self.finished_with_latency
            .set(Some(self.start_time.elapsed()));
    }
}

impl Drop for ConnectionDropGuard {
    fn drop(&mut self) {
        let _guard = self.span.enter();
        self.count_per_route.fetch_sub(1, Ordering::Relaxed);
        let latency_duration = if let Some(finished_with_latency) = self.finished_with_latency.get()
        {
            if let Some(latency_span) = &self.latency_span {
                // Only update the 'overhead' metric when the request finished, so that we
                // can accurately subtract off the time taken for 'external' spans.
                latency_span.set_inference_latency_and_record(
                    finished_with_latency,
                    self.request_logging_data
                        .take()
                        .map(|data| data.extra_overhead_labels),
                );
            }
            finished_with_latency
        } else {
            // If we didn't explicitly mark the request as 'finished' (due to the connection
            // getting dropped early), then use the current time to compute the latency.
            self.start_time.elapsed()
        };

        let latency = format!("{} ms", latency_duration.as_millis());

        // If we did not explicitly set 'finished', then `ConnectionDropGuard` was dropped before the response was sent
        // We log a warning and the latency of the request.
        if self.finished_with_latency.get().is_none() {
            tracing::warn!(
                %latency,
                "Client closed the connection before the response was sent",
            );
        }

        // We might have a status code even if the client closed the connection early
        // (e.g. if we were sending an SSE stream)
        if let Some(status) = self.status {
            tracing::debug!(
                %latency,
                status = i32::from(status.as_u16()),
                success = status.is_success(),
                "finished processing request"
            );
        }
    }
}
