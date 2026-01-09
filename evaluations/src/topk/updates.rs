//! Types for streaming top-k evaluation progress via durable `emit_event`.
//!
//! These types are emitted during task execution to enable:
//! - Real-time UI updates showing progress and confidence bounds
//! - CLI progress feedback
//! - Inter-task coordination (other tasks can `await_event` for completion)
//! - External webhook/API notifications

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::{GlobalStoppingReason, VariantStatus};
use crate::betting_confidence_sequences::MeanBettingConfidenceSequence;

/// Update type for streaming top-k evaluation progress.
///
/// Events are emitted with these naming conventions:
/// - `topk_progress:{task_id}:{batch_idx}` - Batch progress (0, 1, 2, ..., N-1)
/// - `topk_progress:{task_id}:{N}` - Task completion (also emitted at batch index N)
/// - `topk_completed:{task_id}` - Task completion (same payload as above)
///
/// The completion event is emitted under both names so that clients can either:
/// 1. Await `topk_completed:{task_id}` directly for completion, or
/// 2. Poll sequential batch events and receive completion at index N
///
/// Note: The `task_id` is the durable task ID (from `spawn_result.task_id`), which is
/// known externally before the task starts. The payload contains `evaluation_run_id`
/// which is generated inside the task.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TopKUpdate {
    /// Progress update after each batch completes
    BatchProgress(BatchProgressUpdate),
    /// Task completed (success or failure)
    Completed(CompletedUpdate),
}

/// Progress update emitted after processing each batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchProgressUpdate {
    /// Unique identifier for this evaluation run
    pub evaluation_run_id: Uuid,
    /// Current batch index (0-based)
    pub batch_index: usize,
    /// Total number of datapoints processed so far
    pub num_datapoints_processed: usize,
    /// Total number of datapoints in the dataset
    pub total_datapoints: usize,
    /// Lightweight summary of variant performance (mean estimates and CI bounds)
    pub variant_summaries: HashMap<String, VariantSummary>,
    /// Current status of each variant
    pub variant_statuses: HashMap<String, VariantStatus>,
    /// Number of variants still actively being evaluated
    pub num_active_variants: usize,
}

/// Lightweight summary of a variant's performance for streaming.
///
/// This is a reduced form of `MeanBettingConfidenceSequence` suitable for
/// transmission without the full internal state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantSummary {
    /// Point estimate of the mean
    pub mean_est: f64,
    /// Lower bound of the confidence sequence
    pub cs_lower: f64,
    /// Upper bound of the confidence sequence
    pub cs_upper: f64,
    /// Number of observations
    pub count: u64,
}

impl From<&MeanBettingConfidenceSequence> for VariantSummary {
    fn from(cs: &MeanBettingConfidenceSequence) -> Self {
        Self {
            mean_est: cs.mean_est,
            cs_lower: cs.cs_lower,
            cs_upper: cs.cs_upper,
            count: cs.count,
        }
    }
}

/// Completion update emitted when the task finishes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletedUpdate {
    /// Unique identifier for this evaluation run
    pub evaluation_run_id: Uuid,
    /// Why the evaluation stopped
    pub stopping_reason: GlobalStoppingReason,
    /// Total number of datapoints processed
    pub num_datapoints_processed: usize,
    /// Final status of each variant
    pub final_variant_statuses: HashMap<String, VariantStatus>,
}
