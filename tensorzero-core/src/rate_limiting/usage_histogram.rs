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
    last_rotation: Mutex<Instant>,
    rotation_interval: Duration,
}

impl RollingUsageTracker {
    pub(super) fn new(window_duration: Duration) -> Self {
        Self {
            histograms: [UsageHistogram::new(), UsageHistogram::new()],
            current_index: AtomicUsize::new(0),
            last_rotation: Mutex::new(Instant::now()),
            rotation_interval: window_duration / 2,
        }
    }

    /// Record usage - O(1), lock-free for the common path.
    pub(super) fn record(&self, tokens: u64, model_inferences: u64) {
        let idx = self.current_index.load(Ordering::Relaxed);
        self.histograms[idx].record(tokens, model_inferences);
    }

    /// Maybe rotate histograms if enough time has passed.
    /// Called occasionally (e.g., when computing P99).
    fn maybe_rotate(&self) {
        let now = Instant::now();

        // Quick check without lock
        let should_rotate = self
            .last_rotation
            .lock()
            .ok()
            .map(|l| now.duration_since(*l) >= self.rotation_interval)
            .unwrap_or(false);

        if !should_rotate {
            return;
        }

        // Double-check with lock held for rotation
        let Ok(mut last) = self.last_rotation.lock() else {
            return;
        };

        if now.duration_since(*last) < self.rotation_interval {
            return;
        }

        let old_idx = self.current_index.load(Ordering::Relaxed);
        let new_idx = 1 - old_idx;

        self.histograms[new_idx].reset();
        self.current_index.store(new_idx, Ordering::Relaxed);
        *last = now;
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
}
