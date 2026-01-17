//! Bucketed histogram for tracking usage distribution at high throughput.
//!
//! At high QPS (e.g., 10k QPS), exact P99 tracking via VecDeque would require
//! O(n log n) sorting over millions of entries. This module provides O(1) recording
//! and O(NUM_BUCKETS) percentile calculation using logarithmic buckets.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use super::RateLimitResource;

/// Number of buckets in the histogram (covers 0 to 2^20 = 1M tokens)
const NUM_BUCKETS: usize = 21;

/// Lock-free histogram for tracking usage distribution.
/// Uses logarithmic buckets: bucket i covers values in [2^(i-1), 2^i) for i > 0.
/// Bucket 0 covers [0, 1).
#[derive(Debug)]
pub(super) struct UsageHistogram {
    token_buckets: [AtomicU64; NUM_BUCKETS],
    inference_buckets: [AtomicU64; NUM_BUCKETS],
    total_samples: AtomicU64,
}

impl Default for UsageHistogram {
    fn default() -> Self {
        Self::new()
    }
}

impl UsageHistogram {
    fn new() -> Self {
        Self {
            token_buckets: std::array::from_fn(|_| AtomicU64::new(0)),
            inference_buckets: std::array::from_fn(|_| AtomicU64::new(0)),
            total_samples: AtomicU64::new(0),
        }
    }

    /// Get bucket index for a value using log2.
    /// Bucket 0: [0, 1), Bucket 1: [1, 2), Bucket 2: [2, 4), Bucket 3: [4, 8), etc.
    fn bucket_index(value: u64) -> usize {
        if value == 0 {
            return 0;
        }
        // Position of highest set bit (1-indexed)
        let highest_bit = 64 - value.leading_zeros() as usize;
        highest_bit.min(NUM_BUCKETS - 1)
    }

    /// Upper bound value for a bucket index.
    pub(super) fn bucket_upper_bound(index: usize) -> u64 {
        if index == 0 {
            1
        } else if index >= 64 {
            u64::MAX
        } else {
            1u64 << index
        }
    }

    /// Record a usage sample - O(1), lock-free.
    fn record(&self, tokens: u64, model_inferences: u64) {
        let token_idx = Self::bucket_index(tokens);
        let inference_idx = Self::bucket_index(model_inferences);

        self.token_buckets[token_idx].fetch_add(1, Ordering::Relaxed);
        self.inference_buckets[inference_idx].fetch_add(1, Ordering::Relaxed);
        self.total_samples.fetch_add(1, Ordering::Relaxed);
    }

    /// Calculate approximate percentile from buckets - O(NUM_BUCKETS).
    /// Returns the upper bound of the bucket containing the percentile.
    fn percentile(&self, buckets: &[AtomicU64; NUM_BUCKETS], pct: f64) -> Option<u64> {
        let total = self.total_samples.load(Ordering::Relaxed);
        if total == 0 {
            return None;
        }

        let target = ((total as f64) * pct).ceil() as u64;
        let mut cumulative = 0u64;

        for (i, bucket) in buckets.iter().enumerate() {
            cumulative += bucket.load(Ordering::Relaxed);
            if cumulative >= target {
                return Some(Self::bucket_upper_bound(i));
            }
        }

        Some(Self::bucket_upper_bound(NUM_BUCKETS - 1))
    }

    fn p99_tokens(&self) -> Option<u64> {
        self.percentile(&self.token_buckets, 0.99)
    }

    fn p99_inferences(&self) -> Option<u64> {
        self.percentile(&self.inference_buckets, 0.99)
    }

    pub(super) fn sample_count(&self) -> u64 {
        self.total_samples.load(Ordering::Relaxed)
    }

    /// Reset all counts to zero.
    fn reset(&self) {
        for bucket in &self.token_buckets {
            bucket.store(0, Ordering::Relaxed);
        }
        for bucket in &self.inference_buckets {
            bucket.store(0, Ordering::Relaxed);
        }
        self.total_samples.store(0, Ordering::Relaxed);
    }

    pub(super) fn token_buckets(&self) -> &[AtomicU64; NUM_BUCKETS] {
        &self.token_buckets
    }

    pub(super) fn inference_buckets(&self) -> &[AtomicU64; NUM_BUCKETS] {
        &self.inference_buckets
    }
}

/// Rolling window usage tracker using two histograms.
/// Alternates between histograms to maintain approximate rolling window.
#[derive(Debug)]
pub(super) struct RollingUsageTracker {
    histograms: [UsageHistogram; 2],
    current_index: AtomicUsize,
    /// Last rotation time stored as nanoseconds since `created_at` for fast atomic read
    last_rotation_nanos: AtomicU64,
    /// Lock for the rare rotation path (only acquired when actually rotating)
    rotation_lock: Mutex<()>,
    rotation_interval: Duration,
    /// Total window duration (used for rate calculation)
    window_duration: Duration,
    /// Time when the tracker was created (for rate calculation during warm-up)
    created_at: Instant,
}

impl RollingUsageTracker {
    pub(super) fn new(window_duration: Duration) -> Self {
        let now = Instant::now();
        Self {
            histograms: [UsageHistogram::new(), UsageHistogram::new()],
            current_index: AtomicUsize::new(0),
            last_rotation_nanos: AtomicU64::new(0),
            rotation_lock: Mutex::new(()),
            rotation_interval: window_duration / 2,
            window_duration,
            created_at: now,
        }
    }

    /// Record usage - O(1), lock-free for the common path.
    pub(super) fn record(&self, tokens: u64, model_inferences: u64) {
        let idx = self.current_index.load(Ordering::Relaxed);
        self.histograms[idx].record(tokens, model_inferences);
    }

    /// Maybe rotate histograms if enough time has passed.
    /// Uses atomic check on hot path; only acquires lock when rotating (rare).
    fn maybe_rotate(&self) {
        let elapsed_nanos = self.created_at.elapsed().as_nanos() as u64;
        let last_rotation = self.last_rotation_nanos.load(Ordering::Relaxed);
        let interval_nanos = self.rotation_interval.as_nanos() as u64;

        // Quick atomic check - no lock needed for common case
        if elapsed_nanos.saturating_sub(last_rotation) < interval_nanos {
            return;
        }

        // Rare case: might need to rotate. Try to acquire lock without blocking.
        let Ok(_guard) = self.rotation_lock.try_lock() else {
            // Someone else is rotating, skip
            return;
        };

        // Double-check after acquiring lock (another thread may have just rotated)
        let last_rotation = self.last_rotation_nanos.load(Ordering::Acquire);
        if elapsed_nanos.saturating_sub(last_rotation) < interval_nanos {
            return;
        }

        // Perform rotation
        let old_idx = self.current_index.load(Ordering::Relaxed);
        let new_idx = 1 - old_idx;

        self.histograms[new_idx].reset();
        self.current_index.store(new_idx, Ordering::Release);
        self.last_rotation_nanos
            .store(elapsed_nanos, Ordering::Release);
    }

    /// Get P99 for the specified resource, using data from both histograms.
    pub(super) fn p99(&self, resource: RateLimitResource) -> Option<u64> {
        self.maybe_rotate();

        let idx = self.current_index.load(Ordering::Relaxed);
        let current = &self.histograms[idx];
        let previous = &self.histograms[1 - idx];

        // Prefer current if it has enough samples, otherwise combine
        let current_samples = current.sample_count();
        let previous_samples = previous.sample_count();

        if current_samples >= 100 {
            match resource {
                RateLimitResource::Token => current.p99_tokens(),
                RateLimitResource::ModelInference => current.p99_inferences(),
            }
        } else if current_samples + previous_samples > 0 {
            // Combine both histograms for better estimate
            self.combined_p99(resource)
        } else {
            None
        }
    }

    /// Compute P99 from combined histogram data.
    fn combined_p99(&self, resource: RateLimitResource) -> Option<u64> {
        let idx = self.current_index.load(Ordering::Relaxed);
        let current = &self.histograms[idx];
        let previous = &self.histograms[1 - idx];

        let total = current.sample_count() + previous.sample_count();
        if total == 0 {
            return None;
        }

        let target = ((total as f64) * 0.99).ceil() as u64;
        let mut cumulative = 0u64;

        let (current_buckets, previous_buckets) = match resource {
            RateLimitResource::Token => (current.token_buckets(), previous.token_buckets()),
            RateLimitResource::ModelInference => {
                (current.inference_buckets(), previous.inference_buckets())
            }
        };

        for i in 0..NUM_BUCKETS {
            cumulative += current_buckets[i].load(Ordering::Relaxed);
            cumulative += previous_buckets[i].load(Ordering::Relaxed);
            if cumulative >= target {
                return Some(UsageHistogram::bucket_upper_bound(i));
            }
        }

        Some(UsageHistogram::bucket_upper_bound(NUM_BUCKETS - 1))
    }

    /// Estimate request rate per second based on samples collected.
    ///
    /// Uses the actual elapsed time since tracker creation (capped at window_duration)
    /// to provide accurate rate estimation even during warm-up.
    pub(super) fn request_rate_per_second(&self) -> f64 {
        let idx = self.current_index.load(Ordering::Relaxed);
        let current = &self.histograms[idx];
        let previous = &self.histograms[1 - idx];

        let total_samples = current.sample_count() + previous.sample_count();
        if total_samples == 0 {
            return 0.0;
        }

        // Use actual elapsed time, capped at window_duration
        // This gives accurate rate during warm-up when we have less than window_duration of data
        let elapsed = self.created_at.elapsed();
        let effective_duration = elapsed.min(self.window_duration);
        let seconds = effective_duration.as_secs_f64().max(1.0); // Avoid division by zero

        total_samples as f64 / seconds
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bucket_index() {
        assert_eq!(UsageHistogram::bucket_index(0), 0);
        assert_eq!(UsageHistogram::bucket_index(1), 1);
        assert_eq!(UsageHistogram::bucket_index(2), 2);
        assert_eq!(UsageHistogram::bucket_index(3), 2);
        assert_eq!(UsageHistogram::bucket_index(4), 3);
        assert_eq!(UsageHistogram::bucket_index(7), 3);
        assert_eq!(UsageHistogram::bucket_index(8), 4);
        assert_eq!(UsageHistogram::bucket_index(1000), 10);
        assert_eq!(UsageHistogram::bucket_index(1024), 11);
    }

    #[test]
    fn test_bucket_upper_bound() {
        assert_eq!(UsageHistogram::bucket_upper_bound(0), 1);
        assert_eq!(UsageHistogram::bucket_upper_bound(1), 2);
        assert_eq!(UsageHistogram::bucket_upper_bound(2), 4);
        assert_eq!(UsageHistogram::bucket_upper_bound(3), 8);
        assert_eq!(UsageHistogram::bucket_upper_bound(10), 1024);
    }

    #[test]
    fn test_histogram_record_and_percentile() {
        let hist = UsageHistogram::new();

        // Record 100 samples with values 1-100
        for i in 1..=100 {
            hist.record(i, i);
        }

        assert_eq!(hist.sample_count(), 100);

        // P99 should be in a bucket that covers value ~99
        let p99 = hist.p99_tokens().unwrap();
        assert!(p99 >= 64, "P99 should be at least 64, got {p99}");
        assert!(p99 <= 128, "P99 should be at most 128, got {p99}");
    }

    #[test]
    fn test_rolling_tracker_record() {
        let tracker = RollingUsageTracker::new(Duration::from_secs(60));

        for i in 1..=50 {
            tracker.record(i, i);
        }

        // Should have some P99 value
        let p99 = tracker.p99(RateLimitResource::Token);
        assert!(p99.is_some(), "Should have P99 after recording samples");
    }

    #[test]
    fn test_rolling_tracker_request_rate() {
        let tracker = RollingUsageTracker::new(Duration::from_secs(60));

        // Initially no samples, rate should be 0
        assert_eq!(
            tracker.request_rate_per_second(),
            0.0,
            "Rate should be 0 with no samples"
        );

        // Record 100 samples
        for _ in 0..100 {
            tracker.record(10, 1);
        }

        // Rate should be positive (100 samples / elapsed time)
        // Since this test runs quickly, elapsed time is ~0, so rate calculation uses 1s minimum
        let rate = tracker.request_rate_per_second();
        assert!(
            rate > 0.0,
            "Rate should be positive after recording samples"
        );
        assert!(
            rate <= 100.0,
            "Rate should be at most 100 req/s (100 samples in ~1s minimum)"
        );
    }
}
