use std::collections::HashMap;
use tokio_util::sync::CancellationToken;
use tracing::info;

use crate::stats::PerEvaluatorStats;

const MIN_DATAPOINTS: usize = 20;

/// Manager for adaptive stopping logic during evaluation
pub struct StoppingManager {
    precision_limits: Option<HashMap<String, f32>>,
    min_datapoints: usize,
    cancellation_tokens: Option<HashMap<String, CancellationToken>>,
    evaluator_stats: Option<HashMap<String, PerEvaluatorStats>>,
}

impl StoppingManager {
    /// Create a new StoppingManager with precision limits and cancellation tokens
    ///
    /// If precision_limits are provided, this creates evaluator stats for tracking.
    /// The cancellation tokens are used to check and cancel evaluators when precision
    /// limits are met.
    pub fn new(
        precision_limits: Option<HashMap<String, f32>>,
        cancellation_tokens: Option<HashMap<String, CancellationToken>>,
    ) -> Self {
        // Only create stats if precision limits are enabled
        let evaluator_stats = precision_limits.as_ref().map(|precision_map| {
            precision_map
                .keys()
                .map(|name| (name.clone(), PerEvaluatorStats::default()))
                .collect()
        });

        Self {
            precision_limits,
            min_datapoints: MIN_DATAPOINTS,
            cancellation_tokens,
            evaluator_stats,
        }
    }

    /// Update per-evaluator statistics with new evaluation results
    pub fn update_stats<E>(
        &mut self,
        evaluation_result: &HashMap<String, Result<Option<serde_json::Value>, E>>,
    ) {
        if let Some(stats) = self.evaluator_stats.as_mut() {
            for (evaluator_name, eval_result) in evaluation_result {
                // Only track stats for evaluators with stopping conditions
                if let Some(evaluator_stats) = stats.get_mut(evaluator_name) {
                    match eval_result {
                        Ok(Some(serde_json::Value::Number(n))) => {
                            if let Some(value) = n.as_f64() {
                                evaluator_stats.push(value as f32);
                            }
                        }
                        Ok(Some(serde_json::Value::Bool(b))) => {
                            evaluator_stats.push(if *b { 1.0 } else { 0.0 });
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    /// Cancel tokens for evaluators that have converged within their precision limits
    ///
    /// Checks each evaluator's CI half-width against its precision limit and cancels
    /// the token if the evaluator has converged. Only checks after min_datapoints
    /// have been completed.
    pub fn cancel_converged_evaluators(&self, completed_inferences: usize) {
        // Only check after min_datapoints have been completed
        if completed_inferences < self.min_datapoints {
            return;
        }

        if let (Some(stats), Some(precision_map), Some(tokens)) = (
            self.evaluator_stats.as_ref(),
            self.precision_limits.as_ref(),
            self.cancellation_tokens.as_ref(),
        ) {
            // Check each evaluator's stopping condition
            for (evaluator_name, evaluator_stats) in stats {
                if let Some(precision_limit) = precision_map.get(evaluator_name) {
                    if let Some(ci_half_width) = evaluator_stats.ci_half_width() {
                        if ci_half_width <= *precision_limit {
                            if let Some(token) = tokens.get(evaluator_name) {
                                if !token.is_cancelled() {
                                    info!(
                                        evaluator_name = %evaluator_name,
                                        ci_half_width = ci_half_width,
                                        precision_limit = precision_limit,
                                        count = evaluator_stats.count(),
                                        mean = ?evaluator_stats.mean(),
                                        "Stopping evaluator: CI half-width <= precision_limit"
                                    );
                                    token.cancel();
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Check if all evaluators have been stopped
    ///
    /// Returns true if all cancellation tokens are cancelled, false otherwise.
    /// If no cancellation tokens exist (no adaptive stopping), returns false.
    pub fn all_evaluators_stopped(&self) -> bool {
        self.cancellation_tokens
            .as_ref()
            .is_some_and(|tokens| tokens.values().all(|token| token.is_cancelled()))
    }
}
