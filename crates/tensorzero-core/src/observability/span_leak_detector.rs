use moka::sync::Cache;
use tracing::{Metadata, Subscriber};
use tracing_subscriber::{Layer, registry::LookupSpan};

/// A tracing layer that tracks active spans, which can be used to detect leaked spans.
/// Currently, we only use this in e2e tests, since we haven't evaluated the performance impact.
// This can be `Clone` since cloning a moka `Cache` just creates a reference to the same cache
#[derive(Clone, Debug)]
pub struct SpanLeakDetector {
    spans: moka::sync::Cache<tracing::span::Id, CapturedSpanData>,
}

impl SpanLeakDetector {
    #[allow(clippy::new_without_default, clippy::allow_attributes)]
    pub fn new() -> Self {
        Self {
            spans: Cache::builder().build(),
        }
    }

    pub fn print_active_spans(&self) {
        let entries = self
            .spans
            .iter()
            .map(|(_, v)| v.debug_string.clone())
            .collect::<Vec<_>>();
        if entries.is_empty() {
            return;
        }
        tracing::warn!(
            "The following spans are still active:\n{}",
            entries.join("\n")
        );
    }
}

#[derive(Clone, Debug)]
struct CapturedSpanData {
    debug_string: String,
    // We can use these fields if we want to filter out any spans before printing them
    #[expect(dead_code)]
    metadata: &'static Metadata<'static>,
}

impl<S: Subscriber> Layer<S> for SpanLeakDetector
where
    for<'lookup> S: LookupSpan<'lookup>,
{
    fn on_new_span(
        &self,
        attrs: &tracing::span::Attributes<'_>,
        id: &tracing::span::Id,
        ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let parent_id_name = ctx
            .span(id)
            .and_then(|span| span.parent())
            .map(|parent| (parent.id(), parent.name()));
        self.spans.insert(
            id.clone(),
            CapturedSpanData {
                debug_string: format!("{attrs:?} (dynamic_parent: {parent_id_name:?})"),
                metadata: attrs.metadata(),
            },
        );
    }

    fn on_close(&self, id: tracing::span::Id, _ctx: tracing_subscriber::layer::Context<'_, S>) {
        self.spans.remove(&id);
    }
}
