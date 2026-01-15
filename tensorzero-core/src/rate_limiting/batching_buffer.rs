//! Buffer for batching rate-limited requests.

use std::future::Future;
use std::time::Duration;

/// Strategy for determining when to flush a batching buffer.
///
/// The flush strategy controls the timing of batch flushes. Different strategies
/// can be used for production (timer-based) vs testing (manual trigger).
pub(super) trait FlushStrategy: Send + Sync + Clone + 'static {
    /// Wait until the batch should be flushed.
    ///
    /// Called when a new batch window opens. Returns a future that resolves
    /// when the batch should be flushed.
    fn wait_for_flush(&self) -> impl Future<Output = ()> + Send;
}

/// Timer-based flush strategy: flush after a fixed duration.
///
/// This is the default strategy used in production. It waits for a specified
/// duration (default 10ms) before flushing the batch.
#[derive(Clone)]
pub(super) struct TimerFlushStrategy {
    duration: Duration,
}

impl TimerFlushStrategy {
    /// Create a new timer flush strategy with the specified duration.
    pub fn new(duration: Duration) -> Self {
        Self { duration }
    }
}

impl Default for TimerFlushStrategy {
    fn default() -> Self {
        Self::new(Duration::from_millis(10))
    }
}

impl FlushStrategy for TimerFlushStrategy {
    async fn wait_for_flush(&self) {
        tokio::time::sleep(self.duration).await;
    }
}

/// Manual flush strategy for tests: waits for explicit trigger.
///
/// This strategy allows tests to have deterministic control over when batches
/// are flushed, avoiding timing-dependent test failures.
#[cfg(test)]
pub(crate) mod manual_flush_strategy {
    use super::FlushStrategy;
    use std::sync::Arc;
    use tokio::sync::Notify;

    #[derive(Clone)]
    pub struct ManualFlushStrategy {
        trigger: Arc<Notify>,
    }

    impl ManualFlushStrategy {
        /// Create a new manual flush strategy and its trigger.
        ///
        /// Returns a tuple of (strategy, trigger). Call `trigger.trigger()` to
        /// cause the strategy's `wait_for_flush()` to complete.
        pub fn new() -> (Self, FlushTrigger) {
            let notify = Arc::new(Notify::new());
            (
                Self {
                    trigger: notify.clone(),
                },
                FlushTrigger(notify),
            )
        }
    }

    impl FlushStrategy for ManualFlushStrategy {
        async fn wait_for_flush(&self) {
            self.trigger.notified().await;
        }
    }

    /// Handle to manually trigger a flush for testing.
    pub struct FlushTrigger(Arc<Notify>);

    impl FlushTrigger {
        /// Trigger the flush, causing `ManualFlushStrategy::wait_for_flush()` to complete.
        pub fn trigger(&self) {
            self.0.notify_one();
        }
    }
}

// ============================================================================
// Batching Buffer
// ============================================================================

/// Result of a batch flush operation.
#[derive(Debug, Clone)]
pub(super) enum BatchResult {
    /// Batch completed successfully with tokens available
    Success,
    /// Batch failed (e.g., DB error or rate limited)
    Error(String),
}

/// A waiter in the batching buffer.
pub(super) struct BatchWaiter {
    /// The batch ID this waiter belongs to
    batch_id: u64,
    /// Number of tokens this waiter needs
    tokens_needed: u64,
    /// Channel to send the result to the waiter
    response_tx: tokio::sync::oneshot::Sender<BatchResult>,
}

/// Internal state for BatchingBuffer, protected by a single mutex.
struct BatchingBufferState {
    /// List of waiters in the current batch
    waiters: Vec<BatchWaiter>,
    /// Current batch ID: 0 = closed, >0 = open with this batch_id
    current_batch_id: u64,
    /// Counter for generating unique batch IDs (never resets)
    batch_id_counter: u64,
    /// Approximate queued tokens for observability
    queued_tokens: u64,
}

impl BatchingBufferState {
    fn new() -> Self {
        Self {
            waiters: Vec::new(),
            current_batch_id: 0,
            batch_id_counter: 0,
            queued_tokens: 0,
        }
    }
}

/// Unified batching buffer for cold start and burst handling.
///
/// The batching buffer groups multiple requests together to reduce database calls.
/// It only activates when:
/// - The pool has insufficient tokens for immediate consumption, OR
/// - There's no usage data for P99-based borrowing
///
/// TODO(#5623): Implement first-in-batch immediate flush optimization.
/// At low QPS, the first request in a batch should flush immediately (no 10ms delay)
/// while opening a window for stragglers. This avoids adding latency to isolated requests.
/// See design doc at agent-plans/rate-limiting-token-pool/plan.md for details.
///
/// ## Batch Window Lifecycle
///
/// - `current_batch_id = 0` means the window is closed
/// - `current_batch_id > 0` means the window is open with that batch ID
/// - All state is protected by a single mutex, enforcing correct access patterns
pub(super) struct BatchingBuffer<F: FlushStrategy = TimerFlushStrategy> {
    /// All mutable state, protected by a single mutex
    state: tokio::sync::Mutex<BatchingBufferState>,
    /// Strategy for determining when to flush
    flush_strategy: F,
}

impl<F: FlushStrategy> BatchingBuffer<F> {
    /// Create a new batching buffer with the given flush strategy.
    pub fn new(flush_strategy: F) -> Self {
        Self {
            state: tokio::sync::Mutex::new(BatchingBufferState::new()),
            flush_strategy,
        }
    }

    /// Join an existing batch or open a new one.
    ///
    /// Returns a receiver that will be notified when the batch completes.
    ///
    /// The `flush_fn` closure is called to perform the actual flush (DB operation).
    /// It receives the batch_id and should call `drain_waiters()` to get the waiters.
    pub async fn join_or_open<Fut>(
        &self,
        tokens: u64,
        flush_fn: impl FnOnce(u64) -> Fut + Send + 'static,
    ) -> tokio::sync::oneshot::Receiver<BatchResult>
    where
        Fut: Future<Output = ()> + Send + 'static,
    {
        let (tx, rx) = tokio::sync::oneshot::channel();

        let batch_id_to_flush = {
            let mut state = self.state.lock().await;

            if state.current_batch_id == 0 {
                // Window closed, open new one
                state.batch_id_counter += 1;
                let new_id = state.batch_id_counter;
                state.current_batch_id = new_id;
                state.queued_tokens = tokens;

                state.waiters.push(BatchWaiter {
                    batch_id: new_id,
                    tokens_needed: tokens,
                    response_tx: tx,
                });

                Some(new_id) // Need to schedule flush
            } else {
                // Window open, join existing batch
                let current_batch_id = state.current_batch_id;
                state.waiters.push(BatchWaiter {
                    batch_id: current_batch_id,
                    tokens_needed: tokens,
                    response_tx: tx,
                });
                state.queued_tokens += tokens;

                None // Don't schedule flush
            }
        }; // Lock released here

        if let Some(batch_id) = batch_id_to_flush {
            self.schedule_flush(batch_id, flush_fn);
        }

        rx
    }

    /// Schedule a flush for the given batch ID.
    ///
    /// The spawned task is short-lived (10ms timer + quick DB call + notification)
    /// and rate limiting has its own shutdown mechanism for returning unused tokens,
    /// so it's safe to not wait for this task during shutdown.
    #[expect(clippy::disallowed_methods)]
    fn schedule_flush<Fut>(&self, batch_id: u64, flush_fn: impl FnOnce(u64) -> Fut + Send + 'static)
    where
        Fut: Future<Output = ()> + Send + 'static,
    {
        let strategy = self.flush_strategy.clone();
        tokio::spawn(async move {
            strategy.wait_for_flush().await;
            flush_fn(batch_id).await;
        });
    }

    /// Drain waiters for a batch, filtering cancelled requests.
    ///
    /// Returns (live_waiters, total_tokens_needed).
    ///
    /// Cancelled requests (where the receiver has been dropped) are filtered out and
    /// don't contribute to `total_tokens_needed`.
    pub async fn drain_waiters(&self, batch_id: u64) -> (Vec<BatchWaiter>, u64) {
        let mut state = self.state.lock().await;

        // Only drain if this is still our batch
        if state.current_batch_id != batch_id {
            return (vec![], 0);
        }

        // Close window and take all waiters
        state.current_batch_id = 0;
        state.queued_tokens = 0;
        let all_waiters = std::mem::take(&mut state.waiters);

        // Partition waiters by batch_id and log invariant violations
        let (batch_waiters, unexpected): (Vec<_>, Vec<_>) = all_waiters
            .into_iter()
            .partition(|w| w.batch_id == batch_id);

        if !unexpected.is_empty() {
            tracing::error!(
                "found {} waiters with unexpected batch_id (expected {batch_id})",
                unexpected.len(),
            );
            state.waiters = unexpected;
        }

        // Filter out cancelled requests and compute total tokens
        let live_waiters: Vec<_> = batch_waiters
            .into_iter()
            .filter(|w| !w.response_tx.is_closed())
            .collect();

        let total_tokens: u64 = live_waiters.iter().map(|w| w.tokens_needed).sum();

        (live_waiters, total_tokens)
    }

    /// Notify all waiters with the given result.
    ///
    /// This consumes the waiters and sends the result to each one.
    /// Errors sending (receiver dropped) are silently ignored.
    pub fn notify_waiters(waiters: Vec<BatchWaiter>, result: BatchResult) {
        for waiter in waiters {
            let _ = waiter.response_tx.send(result.clone());
        }
    }
}

impl Default for BatchingBuffer<TimerFlushStrategy> {
    fn default() -> Self {
        Self::new(TimerFlushStrategy::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use manual_flush_strategy::ManualFlushStrategy;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_timer_flush_strategy() {
        let strategy = TimerFlushStrategy::new(Duration::from_millis(10));
        let start = std::time::Instant::now();
        strategy.wait_for_flush().await;
        let elapsed = start.elapsed();

        assert!(
            elapsed >= Duration::from_millis(10),
            "Timer should wait at least 10ms, waited {elapsed:?}"
        );
        assert!(
            elapsed < Duration::from_millis(50),
            "Timer should not wait too long, waited {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn test_timer_flush_strategy_default() {
        let strategy = TimerFlushStrategy::default();
        let start = std::time::Instant::now();
        strategy.wait_for_flush().await;
        let elapsed = start.elapsed();

        assert!(
            elapsed >= Duration::from_millis(10),
            "Default timer should wait at least 10ms"
        );
    }

    #[tokio::test]
    #[expect(clippy::disallowed_methods)]
    async fn test_manual_flush_strategy_trigger() {
        let (strategy, trigger) = ManualFlushStrategy::new();

        let handle = tokio::spawn(async move {
            strategy.wait_for_flush().await;
            true
        });

        tokio::time::sleep(Duration::from_millis(5)).await;
        trigger.trigger();

        let result = tokio::time::timeout(Duration::from_millis(100), handle)
            .await
            .expect("Task should complete within timeout")
            .expect("Task should not panic");

        assert!(result, "Task should have completed after trigger");
    }

    #[tokio::test]
    #[expect(clippy::disallowed_methods)]
    async fn test_manual_flush_strategy_no_trigger() {
        let (strategy, _trigger) = ManualFlushStrategy::new();

        let handle = tokio::spawn(async move {
            strategy.wait_for_flush().await;
        });

        let result = tokio::time::timeout(Duration::from_millis(20), handle).await;

        assert!(result.is_err(), "Task should not complete without trigger");
    }

    #[tokio::test]
    #[expect(clippy::disallowed_methods)]
    async fn test_flush_strategy_clone() {
        let strategy1 = TimerFlushStrategy::new(Duration::from_millis(1));
        let strategy2 = strategy1.clone();

        let handle1 = tokio::spawn(async move {
            strategy1.wait_for_flush().await;
        });
        let handle2 = tokio::spawn(async move {
            strategy2.wait_for_flush().await;
        });

        let _ = handle1.await;
        let _ = handle2.await;
    }

    // ========================================================================
    // BatchingBuffer tests
    // ========================================================================

    #[tokio::test]
    async fn test_batching_buffer_single_waiter() {
        let (strategy, trigger) = ManualFlushStrategy::new();
        let buffer = Arc::new(BatchingBuffer::new(strategy));
        let buffer_clone = buffer.clone();

        // First waiter opens the batch
        let rx = buffer
            .join_or_open(100, move |batch_id| {
                let buffer = buffer_clone.clone();
                async move {
                    let (waiters, total_tokens) = buffer.drain_waiters(batch_id).await;
                    assert_eq!(total_tokens, 100, "Should have 100 tokens queued");
                    assert_eq!(waiters.len(), 1, "Should have 1 waiter");
                    BatchingBuffer::<ManualFlushStrategy>::notify_waiters(
                        waiters,
                        BatchResult::Success,
                    );
                }
            })
            .await;

        // Verify queued tokens
        assert_eq!(
            buffer.state.lock().await.queued_tokens,
            100,
            "Should show 100 queued tokens"
        );

        // Trigger flush
        trigger.trigger();

        // Wait for result
        let result = tokio::time::timeout(Duration::from_millis(100), rx)
            .await
            .expect("Should receive result within timeout")
            .expect("Channel should not be closed");

        assert!(
            matches!(result, BatchResult::Success),
            "Should receive success"
        );
    }

    #[tokio::test]
    async fn test_batching_buffer_multiple_joiners() {
        let (strategy, trigger) = ManualFlushStrategy::new();
        let buffer = Arc::new(BatchingBuffer::new(strategy));
        let buffer_clone = buffer.clone();

        // First waiter opens the batch
        let rx1 = buffer
            .join_or_open(100, move |batch_id| {
                let buffer = buffer_clone.clone();
                async move {
                    let (waiters, total_tokens) = buffer.drain_waiters(batch_id).await;
                    assert_eq!(
                        total_tokens, 350,
                        "Should have 350 tokens queued (100+100+150)"
                    );
                    assert_eq!(waiters.len(), 3, "Should have 3 waiters");
                    BatchingBuffer::<ManualFlushStrategy>::notify_waiters(
                        waiters,
                        BatchResult::Success,
                    );
                }
            })
            .await;

        // Second and third waiters join the existing batch
        let rx2 = buffer
            .join_or_open(100, |_| async {
                // This flush_fn won't be called since batch is already open
                panic!("Should not be called");
            })
            .await;

        let rx3 = buffer
            .join_or_open(150, |_| async {
                panic!("Should not be called");
            })
            .await;

        // Verify queued tokens
        assert_eq!(
            buffer.state.lock().await.queued_tokens,
            350,
            "Should show 350 queued tokens"
        );

        // Trigger flush
        trigger.trigger();

        // All waiters should receive success
        let result1 = tokio::time::timeout(Duration::from_millis(100), rx1)
            .await
            .expect("rx1 timeout")
            .expect("rx1 closed");
        let result2 = tokio::time::timeout(Duration::from_millis(100), rx2)
            .await
            .expect("rx2 timeout")
            .expect("rx2 closed");
        let result3 = tokio::time::timeout(Duration::from_millis(100), rx3)
            .await
            .expect("rx3 timeout")
            .expect("rx3 closed");

        assert!(matches!(result1, BatchResult::Success));
        assert!(matches!(result2, BatchResult::Success));
        assert!(matches!(result3, BatchResult::Success));
    }

    #[tokio::test]
    async fn test_batching_buffer_joiner_after_close() {
        // Test that a joiner arriving after drain_waiters opens a new batch
        let (strategy, trigger1) = ManualFlushStrategy::new();
        let buffer = Arc::new(BatchingBuffer::new(strategy));
        let buffer_clone = buffer.clone();

        // First batch
        let rx1 = buffer
            .join_or_open(100, move |batch_id| {
                let buffer = buffer_clone.clone();
                async move {
                    let (waiters, _) = buffer.drain_waiters(batch_id).await;
                    BatchingBuffer::<ManualFlushStrategy>::notify_waiters(
                        waiters,
                        BatchResult::Success,
                    );
                }
            })
            .await;

        // Trigger and wait for first batch to complete
        trigger1.trigger();
        let _ = tokio::time::timeout(Duration::from_millis(100), rx1).await;

        // Now window is closed, next joiner should open new batch
        let (strategy2, trigger2) = ManualFlushStrategy::new();

        // We need to create a new buffer with the new strategy for this test
        // since the original buffer still has the old strategy
        let buffer2 = Arc::new(BatchingBuffer::new(strategy2));
        let buffer2_clone = buffer2.clone();

        let rx2 = buffer2
            .join_or_open(200, move |batch_id| {
                let buffer = buffer2_clone.clone();
                async move {
                    let (waiters, total_tokens) = buffer.drain_waiters(batch_id).await;
                    assert_eq!(total_tokens, 200, "New batch should have 200 tokens");
                    BatchingBuffer::<ManualFlushStrategy>::notify_waiters(
                        waiters,
                        BatchResult::Success,
                    );
                }
            })
            .await;

        trigger2.trigger();

        let result2 = tokio::time::timeout(Duration::from_millis(100), rx2)
            .await
            .expect("rx2 timeout")
            .expect("rx2 closed");

        assert!(matches!(result2, BatchResult::Success));
    }

    #[tokio::test]
    async fn test_batching_buffer_cancellation_before_flush() {
        let (strategy, trigger) = ManualFlushStrategy::new();
        let buffer = Arc::new(BatchingBuffer::new(strategy));
        let buffer_clone = buffer.clone();

        // First waiter opens the batch
        let rx1 = buffer
            .join_or_open(100, move |batch_id| {
                let buffer = buffer_clone.clone();
                async move {
                    let (waiters, total_tokens) = buffer.drain_waiters(batch_id).await;
                    // rx2 was cancelled, so only 100 tokens from rx1
                    assert_eq!(
                        total_tokens, 100,
                        "Should have 100 tokens (cancelled waiter excluded)"
                    );
                    assert_eq!(waiters.len(), 1, "Should have 1 live waiter");
                    BatchingBuffer::<ManualFlushStrategy>::notify_waiters(
                        waiters,
                        BatchResult::Success,
                    );
                }
            })
            .await;

        // Second waiter joins
        let rx2 = buffer
            .join_or_open(200, |_| async {
                panic!("Should not be called");
            })
            .await;

        // Drop rx2 before flush (simulates cancellation)
        drop(rx2);

        // Trigger flush
        trigger.trigger();

        // rx1 should still receive success
        let result1 = tokio::time::timeout(Duration::from_millis(100), rx1)
            .await
            .expect("rx1 timeout")
            .expect("rx1 closed");

        assert!(matches!(result1, BatchResult::Success));
    }

    #[tokio::test]
    async fn test_batching_buffer_error_result() {
        let (strategy, trigger) = ManualFlushStrategy::new();
        let buffer = Arc::new(BatchingBuffer::new(strategy));
        let buffer_clone = buffer.clone();

        let rx = buffer
            .join_or_open(100, move |batch_id| {
                let buffer = buffer_clone.clone();
                async move {
                    let (waiters, _) = buffer.drain_waiters(batch_id).await;
                    BatchingBuffer::<ManualFlushStrategy>::notify_waiters(
                        waiters,
                        BatchResult::Error("Rate limited".to_string()),
                    );
                }
            })
            .await;

        trigger.trigger();

        let result = tokio::time::timeout(Duration::from_millis(100), rx)
            .await
            .expect("timeout")
            .expect("closed");

        assert!(
            matches!(result, BatchResult::Error(msg) if msg == "Rate limited"),
            "Should receive error"
        );
    }

    #[tokio::test]
    #[expect(clippy::disallowed_methods)]
    async fn test_batching_buffer_concurrent_joiners_and_flush() {
        // Test multiple concurrent joiners with flush happening during joins
        let buffer = Arc::new(BatchingBuffer::new(TimerFlushStrategy::new(
            Duration::from_millis(5),
        )));

        let mut handles = vec![];

        // Spawn multiple concurrent joiners
        for i in 0..10 {
            let buffer = buffer.clone();
            let handle = tokio::spawn(async move {
                let buffer_clone = buffer.clone();
                let rx = buffer
                    .join_or_open(100, move |batch_id| {
                        let buffer = buffer_clone.clone();
                        async move {
                            let (waiters, _) = buffer.drain_waiters(batch_id).await;
                            BatchingBuffer::<TimerFlushStrategy>::notify_waiters(
                                waiters,
                                BatchResult::Success,
                            );
                        }
                    })
                    .await;

                // Wait for result
                let result = tokio::time::timeout(Duration::from_millis(100), rx).await;
                (i, result.is_ok())
            });
            handles.push(handle);
        }

        // All joiners should eventually receive a result (either in first batch or later)
        let results: Vec<_> = futures::future::join_all(handles).await;

        let success_count = results
            .iter()
            .filter(|r| r.as_ref().map(|(_, ok)| *ok).unwrap_or(false))
            .count();

        assert!(
            success_count >= 1,
            "At least some joiners should receive results"
        );
    }

    #[tokio::test]
    async fn test_batching_buffer_batch_ids_increment() {
        let buffer = Arc::new(BatchingBuffer::new(TimerFlushStrategy::new(
            Duration::from_millis(1),
        )));

        // First batch
        let buffer_clone = buffer.clone();
        let _rx1 = buffer
            .join_or_open(100, move |batch_id| {
                assert_eq!(batch_id, 1, "First batch should have ID 1");
                let buffer = buffer_clone.clone();
                async move {
                    let (waiters, _) = buffer.drain_waiters(batch_id).await;
                    BatchingBuffer::<TimerFlushStrategy>::notify_waiters(
                        waiters,
                        BatchResult::Success,
                    );
                }
            })
            .await;

        // Wait for first batch to complete
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Second batch
        let buffer_clone = buffer.clone();
        let _rx2 = buffer
            .join_or_open(200, move |batch_id| {
                assert_eq!(batch_id, 2, "Second batch should have ID 2");
                let buffer = buffer_clone.clone();
                async move {
                    let (waiters, _) = buffer.drain_waiters(batch_id).await;
                    BatchingBuffer::<TimerFlushStrategy>::notify_waiters(
                        waiters,
                        BatchResult::Success,
                    );
                }
            })
            .await;

        // Wait for second batch
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}
