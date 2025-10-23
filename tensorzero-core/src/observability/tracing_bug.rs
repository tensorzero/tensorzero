use tracing::Subscriber;
use tracing_subscriber::{filter, layer::Filter, registry::LookupSpan, Layer};

#[cfg(any(test, feature = "e2e_tests"))]
pub static DISABLE_TRACING_BUG_WORKAROUND: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Attempt to work around the tracing bug https://github.com/tokio-rs/tracing/issues/2519
/// When using per-layer filters (which we need in order to output different tracing logs
/// to standard output vs OpenTelemetry), `tracing` can lose events/spans when a crate
/// calls `tracing::enabled!` (or the equivalent via the `log` crate with the `tracing` integration).
///
/// We first encountered this when added `sqlx`, which internally uses `tracing::enabled!`
/// However, any crate anywhere in our dependency graph could trigger this issue.
///
/// The issue itself is the collision of several different optimizations within `tracing`:
/// As far as I can tell, the issue occurs like this:
/// 1. All of the filters return `Interest::always()` for a particular `register_callsite`.
/// 2. Some other crate calls `tracing::enabled!` for a disabled event (e.g. a debug event from `sqlx`)
///    for the first time
/// 3. The `tracing` crate calls `Layer::enabled` for our `Filtered` layer (created with `layer.with_filter(filter)`)
/// 4. The `Filtered` layer implementation uses some very sketchy thread-local usage -
///    it sets a thread-local to the filter result for that event:
///    https://github.com/tokio-rs/tracing/blob/c297a37096ae1562d453624984aa5f924ab490a1/tracing-subscriber/src/filter/layer_filters/mod.rs#L767-L769
///
/// 5. Since this was due to a `tracing::enabled!` call, we don't try to emit any span/event
/// 6. If the next event is for an `Interest::always()` callsite, we'll skip calling `Layer::enabled`
///    and immediately emit the event/span.
///    The `Filtered` layer will call `did_enabled`, which will take and reset the thread-local value:
///    https://github.com/tokio-rs/tracing/blob/c297a37096ae1562d453624984aa5f924ab490a1/tracing-subscriber/src/filter/layer_filters/mod.rs#L767-L769
///
/// 6. As a result, an unrelated event/span ends up using the 'disabled' filter result, and gets discarded.
///    Since the thread-local value was reset to 'true', we'll start processing events/spans again.
///
/// Our current workaround is to prevent `Interest::always()` from getting used, by register a `filter::dynamic_filter_fn`.
/// As stated in the docs for `https://docs.rs/tracing-subscriber/latest/tracing_subscriber/filter/struct.DynFilterFn.html#method.with_callsite_filter`:
/// "By default, DynFilterFn assumes that, because the filter may depend dynamically on the current span context, its result should never be cached."
///
/// We intentionally do *not* call `with_callsite_filter`, so the filter result will never be cached (e.g. `Interest::sometimes` will be used)
/// This ensures that every tracing span/event is preceded by a call to `Layer.enabled`, which will set the `Filtered`
/// thread-local to its correct value for the event. In particular, this will overwrite any wrong value set by
/// a previous `tracing::enabled!` call
///
/// **NOTE** - this workaround might not fix the issue entirely, as the `tracing` callsite/layer/filter interactions are extremely complicated.
/// However, it's a strict improvement over the previous behavior, as `tracing_subscriber::filter::DynFilterFn` is explicitly documented
/// as performing less caching than `tracing_subscriber::filter::Filtered`:
/// https://docs.rs/tracing-subscriber/0.3.20/tracing_subscriber/filter/struct.DynFilterFn.html#method.with_callsite_filter
#[expect(clippy::multiple_bound_locations)] // Clippy false positive
pub fn apply_filter_fixing_tracing_bug<
    S: Subscriber,
    L: Layer<S> + Send + Sync,
    F: Filter<S> + Send + Sync + 'static,
>(
    layer: L,
    filter: F,
) -> impl Layer<S>
where
    for<'a> S: LookupSpan<'a>,
{
    #[cfg(any(test, feature = "e2e_tests"))]
    {
        if DISABLE_TRACING_BUG_WORKAROUND.load(std::sync::atomic::Ordering::Relaxed) {
            // The implementation of `apply_filter_fixing_tracing_bug` is the one place
            // where we actually want to call `Layer::with_filter`
            #[expect(clippy::disallowed_methods)]
            return layer.with_filter(filter).boxed();
        }
    }
    // The implementation of `apply_filter_fixing_tracing_bug` is the one place
    // where we actually want to call `Layer::with_filter`
    #[expect(clippy::disallowed_methods)]
    layer
        .with_filter(filter::dynamic_filter_fn(move |metadata, _context| {
            filter.enabled(metadata, _context)
        }))
        .boxed()
}

#[cfg(test)]
#[expect(clippy::print_stdout)]
mod tests {
    use crate::observability::{self, enter_fake_http_request_otel};
    use gag::BufferRedirect;
    use serde_json::json;
    use std::io::Read;

    fn log_message(message: &str) {
        // We need to have a single invocation of the 'tracing::info!' macro,
        // so that the tracing call-site get reused.
        // Note - our test for the bug reproduction/fix relies on emitting a span (and seeing/not seeing the span information
        // attached to the message), rather than directly checking if a 'tracing::info!' event gets swallowed.
        // The `tracing` crate's logic around caching tracing callsite `Interest` (https://docs.rs/tracing-core/latest/tracing_core/callsite/index.html)
        // is ridiculously complicated, so it can actually be quite difficult to observe wrong stdout output
        // I haven't determined exactly why can we can only reproduce missing spans (and not missing events),
        // but I believe the issue is related to the per-callsite caching behavior.
        // The `tracing::info!` call always seems to re-evaluate `layer.enabled`, implying that the callsite `Interest` is `Interest::sometimes`.
        // However, the `tracing::info_span!` call does *not* gets `layer.enabled` called, which allows the wrong thread-local value
        // from a `log::log_enabled!` call to get used when the event gets emitted.
        let _guard =
            tracing::info_span!(target: "gateway", "My span", otel.name = "my_otel_name", outer_message = message).entered();
        tracing::info!(target: "gateway","My message: {message}");
    }

    #[tokio::test]
    async fn test_reproduce_tracing_bug_stdout() {
        run_bad_tracing_code(true).await;
    }

    #[tokio::test]
    async fn test_workaround_tracing_bug_stdout() {
        run_bad_tracing_code(false).await;
    }

    async fn run_bad_tracing_code(should_see_bug: bool) {
        if should_see_bug {
            observability::tracing_bug::DISABLE_TRACING_BUG_WORKAROUND
                .store(true, std::sync::atomic::Ordering::SeqCst);
        }
        let handle = observability::setup_observability(observability::LogFormat::Json)
            .await
            .unwrap();
        handle.delayed_otel.unwrap().enable_otel().unwrap();
        let buf = BufferRedirect::stdout().unwrap();

        let _guard = enter_fake_http_request_otel();

        log_message("First message");
        // The problematic call from https://github.com/tensorzero/tensorzero/issues/3715
        let _ = log::log_enabled!(target: "fake-target", log::Level::Info);
        // This message will have missing span information if we've set 'DISABLE_TRACING_BUG_WORKAROUND'
        // In production code (where DISABLE_TRACING_BUG_WORKAROUND is compiled out), we should see the span information
        log_message("Second message");

        // Stop capturing stdout
        let mut output = buf.into_inner();
        let mut stdout = String::new();
        output.read_to_string(&mut stdout).unwrap();

        let mut lines = stdout
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .collect::<Vec<_>>();

        println!("Stdout: ```\n{stdout}\n```");

        for line in &mut lines {
            line.as_object_mut().unwrap().remove("timestamp");
        }

        assert_eq!(
            lines[0],
            json!({
                "level": "INFO",
                "fields": {
                    "message": "My message: First message"
                },
                "target": "gateway",
                "spans": [
                    {
                        "otel.name": "my_otel_name",
                        "outer_message": "First message",
                        "name": "My span"
                    }
                ],
                "span": {
                    "otel.name": "my_otel_name",
                    "outer_message": "First message",
                    "name": "My span"
                }
            })
        );
        // If we're trying to reproduce the bug, then we should be missing span information
        // for the message
        if should_see_bug {
            assert_eq!(
                lines[1],
                json!({
                    "level": "INFO",
                    "fields": {
                        "message": "My message: Second message"
                    },
                    "target": "gateway"
                })
            );
        // If we're using the workaround, then we should see the span information
        } else {
            assert_eq!(
                lines[1],
                json!({
                    "level": "INFO",
                    "fields": {
                        "message": "My message: Second message"
                    },
                    "target": "gateway",
                    "spans": [
                        {
                            "otel.name": "my_otel_name",
                            "outer_message": "Second message",
                            "name": "My span"
                        }
                    ],
                    "span": {
                        "otel.name": "my_otel_name",
                        "outer_message": "Second message",
                        "name": "My span"
                    }
                })
            );
        }

        assert_eq!(lines.len(), 2);
    }
}
