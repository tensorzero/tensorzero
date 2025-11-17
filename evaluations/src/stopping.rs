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
    /// The cancellation tokens are cloned (they're Arc-based) and used to check
    /// and cancel evaluators when precision limits are met.
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

    /// Update statistics with new evaluation results
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stopping_manager_no_precision_limits() {
        let manager = StoppingManager::new(None, None);

        // With no precision limits, all_evaluators_stopped should always return false
        assert!(!manager.all_evaluators_stopped());

        // Cancel should be a no-op
        manager.cancel_converged_evaluators(100);
        assert!(!manager.all_evaluators_stopped());
    }

    #[test]
    fn test_stopping_manager_with_precision_limits() {
        let mut precision_limits = HashMap::new();
        precision_limits.insert("evaluator1".to_string(), 0.1);

        let mut tokens = HashMap::new();
        tokens.insert("evaluator1".to_string(), CancellationToken::new());

        let manager = StoppingManager::new(Some(precision_limits), Some(tokens));

        // Initially no evaluators should be stopped
        assert!(!manager.all_evaluators_stopped());
    }

    #[test]
    fn test_update_stats() {
        let mut precision_limits = HashMap::new();
        precision_limits.insert("evaluator1".to_string(), 0.1);

        let tokens = HashMap::new();

        let mut manager = StoppingManager::new(Some(precision_limits), Some(tokens));

        // Add some evaluation results
        let mut results: HashMap<String, Result<Option<serde_json::Value>, String>> =
            HashMap::new();
        results.insert(
            "evaluator1".to_string(),
            Ok(Some(serde_json::Value::Number(
                serde_json::Number::from_f64(0.8).unwrap(),
            ))),
        );

        manager.update_stats(&results);

        // Stats should be updated (we can't directly check but at least verify no panic)
        // In a real scenario, we'd need to expose stats for testing or test indirectly
    }

    #[test]
    fn test_cancel_converged_evaluators_before_min_datapoints() {
        let mut precision_limits = HashMap::new();
        precision_limits.insert("evaluator1".to_string(), 0.1);

        let mut tokens = HashMap::new();
        let token = CancellationToken::new();
        tokens.insert("evaluator1".to_string(), token.clone());

        let manager = StoppingManager::new(Some(precision_limits), Some(tokens));

        // Should not cancel before min_datapoints (20)
        manager.cancel_converged_evaluators(10);
        assert!(!token.is_cancelled());
    }

    #[test]
    fn test_cancel_converged_evaluators_after_convergence() {
        let mut precision_limits = HashMap::new();
        // Set a very high precision limit so it converges easily
        precision_limits.insert("evaluator1".to_string(), 100.0);

        let mut tokens = HashMap::new();
        let token = CancellationToken::new();
        tokens.insert("evaluator1".to_string(), token.clone());

        let mut manager = StoppingManager::new(Some(precision_limits), Some(tokens));

        // Add enough consistent values to converge
        for _ in 0..25 {
            let mut results: HashMap<String, Result<Option<serde_json::Value>, String>> =
                HashMap::new();
            results.insert(
                "evaluator1".to_string(),
                Ok(Some(serde_json::Value::Bool(true))),
            );
            manager.update_stats(&results);
        }

        // Should not cancel before calling cancel_converged_evaluators
        assert!(!token.is_cancelled());

        // Now check stopping after min_datapoints
        manager.cancel_converged_evaluators(25);

        // Should be cancelled now due to convergence with very high precision limit
        assert!(token.is_cancelled());
        assert!(manager.all_evaluators_stopped());
    }

    #[test]
    fn test_all_evaluators_stopped_with_multiple_evaluators() {
        let mut precision_limits = HashMap::new();
        precision_limits.insert("evaluator1".to_string(), 0.1);
        precision_limits.insert("evaluator2".to_string(), 0.1);

        let mut tokens = HashMap::new();
        let token1 = CancellationToken::new();
        let token2 = CancellationToken::new();
        tokens.insert("evaluator1".to_string(), token1.clone());
        tokens.insert("evaluator2".to_string(), token2.clone());

        let manager = StoppingManager::new(Some(precision_limits), Some(tokens));

        // Initially none stopped
        assert!(!manager.all_evaluators_stopped());

        // Cancel one
        token1.cancel();
        assert!(!manager.all_evaluators_stopped());

        // Cancel both
        token2.cancel();
        assert!(manager.all_evaluators_stopped());
    }

    #[test]
    fn test_evaluator_not_in_precision_limits() {
        let mut precision_limits = HashMap::new();
        precision_limits.insert("evaluator1".to_string(), 0.1);

        let mut tokens = HashMap::new();
        tokens.insert("evaluator1".to_string(), CancellationToken::new());
        tokens.insert("evaluator2".to_string(), CancellationToken::new());

        let mut manager = StoppingManager::new(Some(precision_limits), Some(tokens));

        // Add results for evaluator2 (not in precision_limits)
        let mut results: HashMap<String, Result<Option<serde_json::Value>, String>> =
            HashMap::new();
        results.insert(
            "evaluator2".to_string(),
            Ok(Some(serde_json::Value::Bool(true))),
        );

        // Should not panic even though evaluator2 is not being tracked
        manager.update_stats(&results);
    }
}
