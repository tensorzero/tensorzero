use std::time::Duration;

use opentelemetry::trace::Status;
use opentelemetry_sdk::{
    error::OTelSdkResult,
    trace::{SpanData, SpanExporter},
    Resource,
};

/// Wraps a `SpanExporter`, and changes `Status::Unset` to `Status::Ok` on all spans.
/// From https://opentelemetry.io/docs/concepts/signals/traces/#span-status :
/// "What Ok does is represent an unambiguous “final call” on the status of a span that has been explicitly set by a user."
/// We set the status just before it gets exported, so we know that no more errors can occur within the (already finished) span.
///
/// This produces nicer traces on platforms like Arize
#[derive(Debug)]
pub struct ExporterWrapperUnsetToOk<T: SpanExporter> {
    inner: T,
}

impl<T: SpanExporter> ExporterWrapperUnsetToOk<T> {
    pub fn new(inner: T) -> Self {
        Self { inner }
    }
}

// We need to forward all methods to the underlying exporter
#[deny(clippy::missing_trait_methods)]
impl<T: SpanExporter> SpanExporter for ExporterWrapperUnsetToOk<T> {
    async fn export(&self, mut batch: Vec<SpanData>) -> OTelSdkResult {
        for span in &mut batch {
            match span.status {
                Status::Unset => {
                    span.status = Status::Ok;
                }
                Status::Ok | Status::Error { .. } => {}
            }
        }
        self.inner.export(batch).await
    }

    fn shutdown_with_timeout(&mut self, timeout: Duration) -> OTelSdkResult {
        self.inner.shutdown_with_timeout(timeout)
    }

    fn shutdown(&mut self) -> OTelSdkResult {
        self.inner.shutdown()
    }

    fn force_flush(&mut self) -> OTelSdkResult {
        self.inner.force_flush()
    }

    fn set_resource(&mut self, resource: &Resource) {
        self.inner.set_resource(resource);
    }
}
