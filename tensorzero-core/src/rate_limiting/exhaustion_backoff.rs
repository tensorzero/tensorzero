//! Exhaustion backoff for rate limit pools.
//!
//! When the database reports no tokens available (rate limit exhausted), we enter
//! backoff mode to avoid spamming the database with repeated requests that will
//! all be rejected anyway.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Tracks exhaustion state for exponential backoff when rate limit is depleted.
///
/// When the database reports no tokens available (rate limit exhausted), we enter
/// backoff mode to avoid spamming the database with repeated requests that will
/// all be rejected anyway. During backoff:
/// - Requests are rejected immediately without hitting the database
/// - Backoff duration grows exponentially (10ms → 20ms → 40ms → ... → 1000ms)
/// - When tokens become available again, backoff resets
#[derive(Debug)]
pub(super) struct ExhaustionBackoff {
    /// Time until which we should reject requests without hitting DB.
    /// Stored as nanoseconds since `created_at` for atomic operations.
    reject_until_nanos: AtomicU64,
    /// Current backoff duration in nanoseconds (exponential growth).
    backoff_nanos: AtomicU64,
    /// Reference time for computing elapsed nanoseconds.
    created_at: Instant,
}

impl ExhaustionBackoff {
    /// Initial backoff duration when entering exhaustion state.
    const INITIAL_BACKOFF_MS: u64 = 10;
    /// Maximum backoff duration (cap on exponential growth).
    const MAX_BACKOFF_MS: u64 = 1000;
    /// Multiplier for exponential backoff growth.
    const BACKOFF_MULTIPLIER: u64 = 2;

    pub fn new() -> Self {
        Self {
            reject_until_nanos: AtomicU64::new(0),
            backoff_nanos: AtomicU64::new(Self::INITIAL_BACKOFF_MS * 1_000_000),
            created_at: Instant::now(),
        }
    }

    /// Get current elapsed time in nanoseconds since creation.
    fn now_nanos(&self) -> u64 {
        self.created_at.elapsed().as_nanos() as u64
    }

    /// Check if we're currently in backoff period and should reject immediately.
    pub fn should_reject(&self) -> bool {
        let now = self.now_nanos();
        now < self.reject_until_nanos.load(Ordering::Acquire)
    }

    /// Called when DB reports exhaustion (0 tokens available).
    /// Enters or extends backoff period with exponential growth.
    pub fn on_exhausted(&self) {
        let now = self.now_nanos();
        let current_backoff = self.backoff_nanos.load(Ordering::Relaxed);

        // Set reject_until to now + backoff
        let new_reject_until = now.saturating_add(current_backoff);
        self.reject_until_nanos
            .store(new_reject_until, Ordering::Release);

        // Grow backoff exponentially, capped at MAX
        let new_backoff = (current_backoff.saturating_mul(Self::BACKOFF_MULTIPLIER))
            .min(Self::MAX_BACKOFF_MS * 1_000_000);
        self.backoff_nanos.store(new_backoff, Ordering::Relaxed);
    }

    /// Called when DB reports tokens available.
    /// Resets backoff state so requests proceed normally.
    pub fn on_available(&self) {
        self.reject_until_nanos.store(0, Ordering::Release);
        self.backoff_nanos
            .store(Self::INITIAL_BACKOFF_MS * 1_000_000, Ordering::Relaxed);
    }
}

impl Default for ExhaustionBackoff {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    #[test]
    fn test_exhaustion_backoff_initial_state() {
        let backoff = ExhaustionBackoff::new();
        assert!(
            !backoff.should_reject(),
            "Should not reject initially (no exhaustion)"
        );
    }

    #[test]
    fn test_exhaustion_backoff_on_exhausted() {
        let backoff = ExhaustionBackoff::new();

        // Trigger exhaustion
        backoff.on_exhausted();

        // Should now be in backoff
        assert!(
            backoff.should_reject(),
            "Should reject immediately after exhaustion"
        );

        // Backoff should have grown
        let backoff_nanos = backoff.backoff_nanos.load(Ordering::Relaxed);
        assert_eq!(
            backoff_nanos,
            ExhaustionBackoff::INITIAL_BACKOFF_MS
                * 1_000_000
                * ExhaustionBackoff::BACKOFF_MULTIPLIER,
            "Backoff should have doubled after first exhaustion"
        );
    }

    #[test]
    fn test_exhaustion_backoff_on_available_resets() {
        let backoff = ExhaustionBackoff::new();

        // Trigger exhaustion multiple times to grow backoff
        backoff.on_exhausted();
        backoff.on_exhausted();
        backoff.on_exhausted();

        // Now reset
        backoff.on_available();

        // Should no longer reject
        assert!(
            !backoff.should_reject(),
            "Should not reject after on_available"
        );

        // Backoff should be reset to initial
        let backoff_nanos = backoff.backoff_nanos.load(Ordering::Relaxed);
        assert_eq!(
            backoff_nanos,
            ExhaustionBackoff::INITIAL_BACKOFF_MS * 1_000_000,
            "Backoff should be reset to initial after on_available"
        );
    }

    #[test]
    fn test_exhaustion_backoff_exponential_growth() {
        let backoff = ExhaustionBackoff::new();

        // Initial backoff
        let initial = ExhaustionBackoff::INITIAL_BACKOFF_MS * 1_000_000;
        assert_eq!(backoff.backoff_nanos.load(Ordering::Relaxed), initial);

        // First exhaustion: 10ms -> 20ms
        backoff.on_exhausted();
        assert_eq!(
            backoff.backoff_nanos.load(Ordering::Relaxed),
            initial * 2,
            "First exhaustion should double backoff"
        );

        // Second exhaustion: 20ms -> 40ms
        backoff.on_exhausted();
        assert_eq!(
            backoff.backoff_nanos.load(Ordering::Relaxed),
            initial * 4,
            "Second exhaustion should double again"
        );

        // Third exhaustion: 40ms -> 80ms
        backoff.on_exhausted();
        assert_eq!(
            backoff.backoff_nanos.load(Ordering::Relaxed),
            initial * 8,
            "Third exhaustion should double again"
        );
    }

    #[test]
    fn test_exhaustion_backoff_max_cap() {
        let backoff = ExhaustionBackoff::new();
        let max_nanos = ExhaustionBackoff::MAX_BACKOFF_MS * 1_000_000;

        // Trigger exhaustion many times to hit the cap
        for _ in 0..20 {
            backoff.on_exhausted();
        }

        let current = backoff.backoff_nanos.load(Ordering::Relaxed);
        assert_eq!(
            current, max_nanos,
            "Backoff should be capped at MAX_BACKOFF_MS"
        );

        // One more exhaustion should not exceed cap
        backoff.on_exhausted();
        let after = backoff.backoff_nanos.load(Ordering::Relaxed);
        assert_eq!(
            after, max_nanos,
            "Backoff should remain at cap after additional exhaustion"
        );
    }

    #[tokio::test]
    async fn test_exhaustion_backoff_expires() {
        let backoff = ExhaustionBackoff::new();

        // Trigger exhaustion
        backoff.on_exhausted();
        assert!(backoff.should_reject(), "Should reject after exhaustion");

        // Wait for initial backoff (10ms) to expire
        tokio::time::sleep(Duration::from_millis(15)).await;

        // Should no longer reject (backoff expired)
        assert!(
            !backoff.should_reject(),
            "Should not reject after backoff period expires"
        );
    }
}
