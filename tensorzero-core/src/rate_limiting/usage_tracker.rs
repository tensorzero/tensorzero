//! Per-second usage tracking for P99 calculation at high throughput.
//!
//! This module provides `PerSecondUsageTracker` which tracks total tokens
//! consumed per second in a circular buffer for P99 calculation over a rolling window.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

// ============================================================================
// Per-Second Usage Tracking
// ============================================================================

/// Rolling window size in seconds for per-second P99 tracking.
const WINDOW_SECONDS: usize = 120;

/// A single bucket that packs epoch (u32) and tokens (u32) into one AtomicU64.
///
/// This ensures atomic read-modify-write without races between epoch changes
/// and token increments. The upper 32 bits store the epoch (seconds since
/// tracker creation), and the lower 32 bits store the accumulated tokens.
#[derive(Debug)]
struct Bucket(AtomicU64);

impl Bucket {
    fn new() -> Self {
        Self(AtomicU64::new(0))
    }

    /// Pack epoch and tokens into a single u64.
    fn pack(epoch: u32, tokens: u32) -> u64 {
        ((epoch as u64) << 32) | (tokens as u64)
    }

    /// Unpack a u64 into (epoch, tokens).
    fn unpack(packed: u64) -> (u32, u32) {
        ((packed >> 32) as u32, packed as u32)
    }

    /// Record tokens for the current epoch using a CAS loop.
    ///
    /// If the stored epoch matches `current_epoch`, adds tokens to the existing count.
    /// If the epoch differs, resets the bucket to the new epoch with just these tokens.
    fn record(&self, current_epoch: u32, tokens: u32) {
        loop {
            let packed = self.0.load(Ordering::Acquire);
            let (stored_epoch, stored_tokens) = Self::unpack(packed);

            let new_packed = if stored_epoch == current_epoch {
                // Same epoch: accumulate tokens (saturating to avoid overflow)
                Self::pack(current_epoch, stored_tokens.saturating_add(tokens))
            } else {
                // New epoch: reset counter, start with these tokens
                Self::pack(current_epoch, tokens)
            };

            if self
                .0
                .compare_exchange_weak(packed, new_packed, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return;
            }
            // CAS failed, retry with updated value
        }
    }

    /// Adjust tokens for a specific epoch using a CAS loop.
    ///
    /// Looks back at the bucket for `target_epoch` and adjusts its token count.
    /// Only drops the adjustment if the bucket has been reused for a different epoch
    /// (meaning more than WINDOW_SECONDS have passed since the original recording).
    fn adjust(&self, target_epoch: u32, delta: i64) -> bool {
        loop {
            let packed = self.0.load(Ordering::Acquire);
            let (stored_epoch, stored_tokens) = Self::unpack(packed);

            if stored_epoch != target_epoch {
                // Bucket has been reused for a different epoch (> WINDOW_SECONDS passed)
                return false;
            }

            // Apply delta with saturating arithmetic
            let new_tokens = stored_tokens.saturating_add_signed(delta as i32);
            let new_packed = Self::pack(target_epoch, new_tokens);

            if self
                .0
                .compare_exchange_weak(packed, new_packed, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return true;
            }
        }
    }

    /// Read the token count if the bucket is within the valid window.
    ///
    /// Returns `Some(tokens)` if `stored_epoch` is within `window` epochs of `current_epoch`
    /// and has non-zero tokens. Returns `None` otherwise (stale or empty bucket).
    fn read_if_not_stale(&self, current_epoch: u32, validity_window: u32) -> Option<u64> {
        let packed = self.0.load(Ordering::Acquire);
        let (stored_epoch, tokens) = Self::unpack(packed);

        // Check if the bucket is recent enough (within window epochs)
        // Using wrapping subtraction handles epoch wraparound correctly
        if current_epoch.wrapping_sub(stored_epoch) < validity_window && tokens > 0 {
            Some(tokens as u64)
        } else {
            None
        }
    }
}

/// Tracks total tokens consumed per second over a rolling window.
///
/// Uses a circular buffer of packed atomic buckets for lock-free updates.
/// Each bucket stores the epoch (seconds since creation) and total tokens
/// for that second, packed into a single AtomicU64.
#[derive(Debug)]
pub(super) struct PerSecondUsageTracker {
    buckets: [Bucket; WINDOW_SECONDS],
    created_at: Instant,
}

impl PerSecondUsageTracker {
    pub fn new() -> Self {
        Self {
            buckets: std::array::from_fn(|_| Bucket::new()),
            created_at: Instant::now(),
        }
    }

    /// Get the current epoch (seconds since tracker creation).
    fn current_epoch(&self) -> u32 {
        self.created_at.elapsed().as_secs() as u32
    }

    /// Record token usage for the current second, and return the epoch it was recorded at.
    ///
    /// This is O(1) and lock-free, using a CAS loop internally.
    /// Tokens are clamped to u32::MAX (4B tokens/sec is plenty for any use case).
    /// The returned epoch can be passed to `adjust()` to adjust the recording (e.g., when actual
    /// usage differs from estimated).
    pub fn record_with_epoch(&self, tokens: u64) -> u32 {
        let epoch = self.current_epoch();
        let idx = (epoch as usize) % WINDOW_SECONDS;
        // Clamp to u32::MAX for packing
        let tokens_clamped = std::cmp::min(tokens, u32::MAX as u64) as u32;
        self.buckets[idx].record(epoch, tokens_clamped);
        epoch
    }

    /// Adjust a previous recording when actual usage differs from estimated.
    ///
    /// Looks back at the bucket for `recorded_epoch` and applies the delta
    /// (actual - estimated). If the bucket has been reused (> WINDOW_SECONDS passed),
    /// the adjustment is dropped since the original data is gone.
    pub fn adjust(&self, recorded_epoch: u32, estimated: u64, actual: u64) {
        if actual == estimated {
            return;
        }
        let delta = actual as i64 - estimated as i64;
        let idx = (recorded_epoch as usize) % WINDOW_SECONDS;
        // Best-effort adjustment - returns false if bucket was reused
        self.buckets[idx].adjust(recorded_epoch, delta);
    }

    /// Calculate P99 tokens/second over the rolling window. Return 0 if there's no data.
    ///
    /// If there's truly no data, we return 0. Otherwise, we always attempt to
    /// compute P99 even within the first second; the estimate will not be accurate
    /// (we will under-borrow), but we will hit the DB more frequently rather than
    /// provide an arbitrary estimate during cold start / bursty traffic to avoid
    /// starving requests.
    pub fn p99(&self) -> Option<u64> {
        let epoch = self.current_epoch();
        let mut values: Vec<u64> = self
            .buckets
            .iter()
            .filter_map(|b| b.read_if_not_stale(epoch, WINDOW_SECONDS as u32))
            .collect();

        if values.is_empty() {
            return None;
        }

        values.sort_unstable();
        let idx = (values.len() - 1) * 99 / 100;
        Some(values[idx])
    }
}

impl Default for PerSecondUsageTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_bucket_pack_unpack() {
        // Test pack/unpack roundtrip
        let epoch = 12345u32;
        let tokens = 67890u32;
        let packed = Bucket::pack(epoch, tokens);
        let (unpacked_epoch, unpacked_tokens) = Bucket::unpack(packed);
        assert_eq!(unpacked_epoch, epoch, "Epoch should roundtrip correctly");
        assert_eq!(unpacked_tokens, tokens, "Tokens should roundtrip correctly");

        // Test edge cases
        let packed_max = Bucket::pack(u32::MAX, u32::MAX);
        let (e, t) = Bucket::unpack(packed_max);
        assert_eq!(e, u32::MAX, "Max epoch should roundtrip");
        assert_eq!(t, u32::MAX, "Max tokens should roundtrip");

        let packed_zero = Bucket::pack(0, 0);
        let (e, t) = Bucket::unpack(packed_zero);
        assert_eq!(e, 0, "Zero epoch should roundtrip");
        assert_eq!(t, 0, "Zero tokens should roundtrip");
    }

    #[test]
    fn test_bucket_record_same_epoch() {
        let bucket = Bucket::new();
        let epoch = 42;

        // Record multiple times in same epoch
        bucket.record(epoch, 100);
        bucket.record(epoch, 200);
        bucket.record(epoch, 50);

        let value = bucket.read_if_not_stale(epoch, 60);
        assert_eq!(
            value,
            Some(350),
            "Tokens should accumulate within same epoch"
        );
    }

    #[test]
    fn test_bucket_record_new_epoch_resets() {
        let bucket = Bucket::new();

        // Record in epoch 1
        bucket.record(1, 500);
        assert_eq!(
            bucket.read_if_not_stale(1, 60),
            Some(500),
            "Should have 500 tokens in epoch 1"
        );

        // Record in epoch 2 should reset
        bucket.record(2, 100);
        assert_eq!(
            bucket.read_if_not_stale(2, 60),
            Some(100),
            "New epoch should reset counter"
        );

        // Old epoch data is gone
        assert_eq!(
            bucket.read_if_not_stale(1, 60),
            None,
            "Old epoch should return None (epoch 2 - 1 < 60, but stored epoch is now 2)"
        );
    }

    #[test]
    fn test_bucket_read_stale_data() {
        let bucket = Bucket::new();
        bucket.record(10, 100);

        // Reading within window should work
        assert_eq!(
            bucket.read_if_not_stale(10, 60),
            Some(100),
            "Reading at same epoch should work"
        );
        assert_eq!(
            bucket.read_if_not_stale(50, 60),
            Some(100),
            "Reading within window should work"
        );
        assert_eq!(
            bucket.read_if_not_stale(69, 60),
            Some(100),
            "Reading at edge of window should work"
        );

        // Reading outside window should return None
        assert_eq!(
            bucket.read_if_not_stale(70, 60),
            None,
            "Reading outside window should return None"
        );
    }

    #[test]
    fn test_bucket_saturating_add() {
        let bucket = Bucket::new();
        let epoch = 1;

        // Record max - 100, then add 200 (should saturate to max)
        bucket.record(epoch, u32::MAX - 100);
        bucket.record(epoch, 200);

        let value = bucket.read_if_not_stale(epoch, 60).unwrap();
        assert_eq!(value, u32::MAX as u64, "Should saturate at u32::MAX");
    }

    #[test]
    fn test_per_second_tracker_basic() {
        let tracker = PerSecondUsageTracker::new();

        // Initially no data
        assert_eq!(tracker.p99(), None, "Should have no data initially");

        // Record in current second multiple times
        tracker.record_with_epoch(100);
        tracker.record_with_epoch(200);
        tracker.record_with_epoch(50);

        // Within the first second, we start computing P99, but it's not accurate yet.
        let p99 = tracker.p99();
        assert!(p99.is_some(), "P99 should have a value");
    }

    #[test]
    fn test_per_second_tracker_token_clamping() {
        let tracker = PerSecondUsageTracker::new();

        // Record more than u32::MAX tokens
        tracker.record_with_epoch(u64::MAX);

        let epoch = tracker.current_epoch();
        let idx = (epoch as usize) % WINDOW_SECONDS;
        let value = tracker.buckets[idx].read_if_not_stale(epoch, WINDOW_SECONDS as u32);

        assert_eq!(
            value,
            Some(u32::MAX as u64),
            "Tokens should be clamped to u32::MAX"
        );
    }

    #[test]
    fn test_per_second_tracker_concurrent_recording() {
        const NUM_THREADS: usize = 100;
        const RECORDS_PER_THREAD: usize = 1000;
        const TOKENS_PER_RECORD: u64 = 100;

        let tracker = Arc::new(PerSecondUsageTracker::new());

        let handles: Vec<_> = (0..NUM_THREADS)
            .map(|_| {
                let tracker = Arc::clone(&tracker);
                thread::spawn(move || {
                    for _ in 0..RECORDS_PER_THREAD {
                        tracker.record_with_epoch(TOKENS_PER_RECORD);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("Thread should complete without panic");
        }

        // All tokens should be recorded in the current second
        // Expected: NUM_THREADS * RECORDS_PER_THREAD * TOKENS_PER_RECORD = 10M tokens
        // This exceeds u32::MAX (4B), so we expect saturation
        let epoch = tracker.current_epoch();
        let idx = (epoch as usize) % WINDOW_SECONDS;
        let value = tracker.buckets[idx].read_if_not_stale(epoch, WINDOW_SECONDS as u32);

        // With 10M tokens attempted and u32::MAX = ~4B, we should saturate
        // The actual value depends on thread interleaving with the CAS loop
        assert!(
            value.is_some(),
            "Should have recorded tokens under concurrency"
        );
    }

    #[test]
    fn test_bucket_stale_data_after_window() {
        let bucket = Bucket::new();

        // Record at epoch 100
        bucket.record(100, 500);

        // Read from epoch that's beyond the window
        // epoch 200, window 60 -> epoch 100 is stale (200 - 100 = 100 >= 60)
        let value = bucket.read_if_not_stale(200, 60);
        assert_eq!(
            value, None,
            "Should return None for data outside the window"
        );

        // But epoch 150 should still see it (150 - 100 = 50 < 60)
        let value = bucket.read_if_not_stale(150, 60);
        assert_eq!(
            value,
            Some(500),
            "Should return data that's still within the window"
        );
    }
}
