//! Internal metrics (currently reported to Howdy)
//! These are intentionally *not* persisted to the databsae across restarts
//! We use atomic counters, rather than go through the `metrics` crate,
//! as this turns out to be simpler than stripping out the labels we use when normally
//! recording metrics.

use std::sync::atomic::AtomicU64;

pub static TENSORZERO_INFERENCES_TOTAL: AtomicU64 = AtomicU64::new(0);
pub static TENSORZERO_FEEDBACKS_TOTAL: AtomicU64 = AtomicU64::new(0);
pub static TENSORZERO_INPUT_TOKENS_TOTAL: AtomicU64 = AtomicU64::new(0);
pub static TENSORZERO_OUTPUT_TOKENS_TOTAL: AtomicU64 = AtomicU64::new(0);
