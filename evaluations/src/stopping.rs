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
    /// Create a new StoppingManager with precision limits and evaluator names
    ///
    /// If precision_limits are provided, this creates:
    /// - Cancellation tokens for all evaluators
    /// - Statistics trackers for evaluators with precision limits
    pub fn new(precision_limits: Option<HashMap<String, f32>>, evaluator_names: &[String]) -> Self {
        // Create cancellation tokens and stats only if precision limits are enabled
        let (cancellation_tokens, evaluator_stats) =
            if let Some(ref precision_map) = precision_limits {
                // Create tokens for all evaluators
                let tokens: HashMap<String, CancellationToken> = evaluator_names
                    .iter()
                    .map(|name| (name.clone(), CancellationToken::new()))
                    .collect();

                // Create stats only for evaluators with precision limits
                let stats: HashMap<String, PerEvaluatorStats> = precision_map
                    .keys()
                    .map(|name| (name.clone(), PerEvaluatorStats::default()))
                    .collect();

                (Some(tokens), Some(stats))
            } else {
                (None, None)
            };

        Self {
            precision_limits,
            min_datapoints: MIN_DATAPOINTS,
            cancellation_tokens,
            evaluator_stats,
        }
    }

    /// Get a reference to the cancellation tokens for use by evaluation tasks
    ///
    /// Returns None if no precision limits were configured (no adaptive stopping).
    pub fn get_tokens(&self) -> Option<&HashMap<String, CancellationToken>> {
        self.cancellation_tokens.as_ref()
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
        let evaluator_names = vec!["evaluator1".to_string(), "evaluator2".to_string()];
        let manager = StoppingManager::new(None, &evaluator_names);

        // With no precision limits, all_evaluators_stopped should always return false
        assert!(!manager.all_evaluators_stopped());

        // get_tokens should return None
        assert!(manager.get_tokens().is_none());

        // Cancel should be a no-op
        manager.cancel_converged_evaluators(100);
        assert!(!manager.all_evaluators_stopped());
    }

    #[test]
    fn test_stopping_manager_with_precision_limits() {
        let mut precision_limits = HashMap::new();
        precision_limits.insert("evaluator1".to_string(), 0.1);

        let evaluator_names = vec!["evaluator1".to_string()];
        let manager = StoppingManager::new(Some(precision_limits), &evaluator_names);

        // Initially no evaluators should be stopped
        assert!(!manager.all_evaluators_stopped());

        // get_tokens should return Some
        assert!(manager.get_tokens().is_some());
        assert_eq!(manager.get_tokens().unwrap().len(), 1);
    }

    #[test]
    fn test_update_stats() {
        let mut precision_limits = HashMap::new();
        precision_limits.insert("evaluator1".to_string(), 0.1);

        let evaluator_names = vec!["evaluator1".to_string()];
        let mut manager = StoppingManager::new(Some(precision_limits), &evaluator_names);

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

        let evaluator_names = vec!["evaluator1".to_string()];
        let manager = StoppingManager::new(Some(precision_limits), &evaluator_names);

        // Should not cancel before min_datapoints (20)
        manager.cancel_converged_evaluators(10);

        // Get the token to check if it's cancelled
        let tokens = manager.get_tokens().unwrap();
        let token = tokens.get("evaluator1").unwrap();
        assert!(!token.is_cancelled());
    }

    #[test]
    fn test_cancel_converged_evaluators_after_convergence() {
        let mut precision_limits = HashMap::new();
        // Set a very high precision limit so it converges easily
        precision_limits.insert("evaluator1".to_string(), 100.0);

        let evaluator_names = vec!["evaluator1".to_string()];
        let mut manager = StoppingManager::new(Some(precision_limits), &evaluator_names);

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

        // Get the token to check its state
        let tokens = manager.get_tokens().unwrap();
        let token = tokens.get("evaluator1").unwrap();

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

        let evaluator_names = vec!["evaluator1".to_string(), "evaluator2".to_string()];
        let manager = StoppingManager::new(Some(precision_limits), &evaluator_names);

        // Initially none stopped
        assert!(!manager.all_evaluators_stopped());

        // Get tokens and cancel them manually (simulating convergence)
        let tokens = manager.get_tokens().unwrap();
        let token1 = tokens.get("evaluator1").unwrap();
        let token2 = tokens.get("evaluator2").unwrap();

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

        let evaluator_names = vec!["evaluator1".to_string(), "evaluator2".to_string()];
        let mut manager = StoppingManager::new(Some(precision_limits), &evaluator_names);

        // Verify that tokens exist for both evaluators (even though only evaluator1 has precision limit)
        let tokens = manager.get_tokens().unwrap();
        assert!(tokens.contains_key("evaluator1"));
        assert!(tokens.contains_key("evaluator2"));

        // Add results for evaluator2 (not in precision_limits)
        let mut results: HashMap<String, Result<Option<serde_json::Value>, String>> =
            HashMap::new();
        results.insert(
            "evaluator2".to_string(),
            Ok(Some(serde_json::Value::Bool(true))),
        );

        // Should not panic even though evaluator2 is not being tracked for stats
        manager.update_stats(&results);
    }
}
