use std::collections::HashMap;
use std::ops::Deref;
use tensorzero_core::evaluations::EvaluatorConfig;
use tokio_util::sync::CancellationToken;
use tracing::info;

use crate::stats::PerEvaluatorStats;

pub const MIN_DATAPOINTS: usize = 20;

/// Newtype wrapper for cancellation tokens
///
/// An empty map indicates no adaptive stopping is configured.
#[derive(Debug, Clone, Default)]
pub struct CancellationTokens(HashMap<String, CancellationToken>);

impl CancellationTokens {
    /// Create a new CancellationTokens from evaluator names
    pub fn new(evaluator_names: &[String]) -> Self {
        Self(
            evaluator_names
                .iter()
                .map(|name| (name.clone(), CancellationToken::new()))
                .collect(),
        )
    }

    /// Check if all tokens are cancelled
    pub fn all_cancelled(&self) -> bool {
        !self.is_empty() && self.values().all(|token| token.is_cancelled())
    }
}

impl Deref for CancellationTokens {
    type Target = HashMap<String, CancellationToken>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Manager for adaptive stopping logic during evaluation
pub struct StoppingManager {
    precision_targets: HashMap<String, f32>,
    min_datapoints: usize,
    cancellation_tokens: CancellationTokens,
    evaluator_stats: HashMap<String, PerEvaluatorStats>,
}

impl StoppingManager {
    /// Create a new StoppingManager with precision targets and evaluator configs
    ///
    /// If precision_targets is non-empty, this creates:
    /// - Cancellation tokens for all evaluators
    /// - Statistics trackers for evaluators with precision targets
    ///
    /// If precision_targets is empty (no adaptive stopping):
    /// - No cancellation tokens are created
    /// - No statistics are tracked
    pub fn new(
        evaluators: &HashMap<String, EvaluatorConfig>,
        precision_targets: HashMap<String, f32>,
    ) -> Self {
        // Create cancellation tokens and stats only if precision targets are enabled
        let (cancellation_tokens, evaluator_stats) = if precision_targets.is_empty() {
            (CancellationTokens::default(), HashMap::new())
        } else {
            // Create tokens for all evaluators
            let evaluator_names: Vec<String> = evaluators.keys().cloned().collect();
            let tokens = CancellationTokens::new(&evaluator_names);

            // Create stats only for evaluators with precision targets
            let stats: HashMap<String, PerEvaluatorStats> = precision_targets
                .keys()
                .filter_map(|name| {
                    evaluators
                        .get(name)
                        .map(|config| (name.clone(), PerEvaluatorStats::new(config.is_bernoulli())))
                })
                .collect();

            (tokens, stats)
        };

        Self {
            precision_targets,
            min_datapoints: MIN_DATAPOINTS,
            cancellation_tokens,
            evaluator_stats,
        }
    }

    /// Get a reference to the cancellation tokens for use by evaluation tasks
    pub fn get_tokens(&self) -> &CancellationTokens {
        &self.cancellation_tokens
    }

    /// Update statistics with new evaluation results
    pub fn update_stats<E>(
        &mut self,
        evaluation_result: &HashMap<String, Result<Option<serde_json::Value>, E>>,
    ) {
        for (evaluator_name, eval_result) in evaluation_result {
            // Only track stats for evaluators with stopping conditions
            if let Some(evaluator_stats) = self.evaluator_stats.get_mut(evaluator_name) {
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

    /// Cancel tokens for evaluators that have converged to their precision targets
    ///
    /// Checks each evaluator's CI half-width (max width of the two halves of the CI)
    /// against its precision target and cancels the token if the evaluator has converged.
    /// Only checks after inferences have been completed for at least min_datapoints.
    pub fn cancel_converged_evaluators(&self, num_completed_datapoints: usize) {
        // Only check after min_datapoints have been completed
        if num_completed_datapoints < self.min_datapoints {
            return;
        }

        // If no precision targets configured, nothing to do
        if self.precision_targets.is_empty() {
            return;
        }

        let tokens = &self.cancellation_tokens;

        // Check each evaluator's stopping condition
        for (evaluator_name, evaluator_stats) in &self.evaluator_stats {
            if let Some(precision_target) = self.precision_targets.get(evaluator_name)
                && let Some(ci_half_width) = evaluator_stats.ci_half_width()
                && ci_half_width <= *precision_target
                && let Some(token) = tokens.get(evaluator_name)
                && !token.is_cancelled()
            {
                info!(
                    evaluator_name = %evaluator_name,
                    ci_half_width = ci_half_width,
                    precision_target = precision_target,
                    count = evaluator_stats.count(),
                    mean = ?evaluator_stats.mean(),
                    "Stopping evaluator: CI half-width <= precision_target"
                );
                token.cancel();
            }
        }
    }

    /// Check if all evaluators have been stopped
    ///
    /// Returns true if all cancellation tokens are cancelled, false otherwise.
    /// If no tokens exist (no adaptive stopping), returns false.
    pub fn all_evaluators_stopped(&self) -> bool {
        self.cancellation_tokens.all_cancelled()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensorzero_core::evaluations::{EvaluatorConfig::ExactMatch, ExactMatchConfig};

    #[test]
    fn test_stopping_manager_no_precision_targets() {
        let mut evaluators = HashMap::new();
        evaluators.insert(
            "evaluator1".to_string(),
            ExactMatch(ExactMatchConfig::default()),
        );
        evaluators.insert(
            "evaluator2".to_string(),
            ExactMatch(ExactMatchConfig::default()),
        );

        let manager = StoppingManager::new(&evaluators, HashMap::new());

        // With no precision targets, all_evaluators_stopped should always return false
        assert!(!manager.all_evaluators_stopped());

        // get_tokens should return empty map
        assert!(manager.get_tokens().is_empty());

        // Cancel should be a no-op
        manager.cancel_converged_evaluators(100);
        assert!(!manager.all_evaluators_stopped());
    }

    #[test]
    fn test_stopping_manager_with_precision_targets() {
        let mut precision_targets = HashMap::new();
        precision_targets.insert("evaluator1".to_string(), 0.1);

        let mut evaluators = HashMap::new();
        evaluators.insert(
            "evaluator1".to_string(),
            ExactMatch(ExactMatchConfig::default()),
        );

        let manager = StoppingManager::new(&evaluators, precision_targets);

        // Initially no evaluators should be stopped
        assert!(!manager.all_evaluators_stopped());

        // get_tokens should return non-empty map
        assert_eq!(manager.get_tokens().len(), 1);
        assert!(manager.get_tokens().contains_key("evaluator1"));
    }

    #[test]
    fn test_cancel_converged_evaluators_before_min_datapoints() {
        let mut precision_targets = HashMap::new();
        precision_targets.insert("evaluator1".to_string(), 0.1);

        let mut evaluators = HashMap::new();
        evaluators.insert(
            "evaluator1".to_string(),
            ExactMatch(ExactMatchConfig::default()),
        );

        let manager = StoppingManager::new(&evaluators, precision_targets);

        // Should not cancel before min_datapoints (20)
        manager.cancel_converged_evaluators(10);

        // Get the token to check if it's cancelled
        let tokens = manager.get_tokens();
        let token = tokens.get("evaluator1").unwrap();
        assert!(!token.is_cancelled());
    }

    #[test]
    fn test_cancel_converged_evaluators_after_convergence() {
        let mut precision_targets = HashMap::new();
        // Set a very liberal precision limit so it converges easily
        precision_targets.insert("evaluator1".to_string(), 100.0);

        let mut evaluators = HashMap::new();
        evaluators.insert(
            "evaluator1".to_string(),
            ExactMatch(ExactMatchConfig::default()),
        );

        let mut manager = StoppingManager::new(&evaluators, precision_targets);

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
        let tokens = manager.get_tokens();
        let token = tokens.get("evaluator1").unwrap();

        // Should not cancel before calling cancel_converged_evaluators
        assert!(!token.is_cancelled());

        // Now check stopping after min_datapoints
        manager.cancel_converged_evaluators(25);

        // Should be cancelled now due to convergence with very liberal precision target
        assert!(token.is_cancelled());
        assert!(manager.all_evaluators_stopped());
    }

    #[test]
    fn test_all_evaluators_stopped_with_multiple_evaluators() {
        let mut precision_targets = HashMap::new();
        precision_targets.insert("evaluator1".to_string(), 0.1);
        precision_targets.insert("evaluator2".to_string(), 0.1);

        let mut evaluators = HashMap::new();
        evaluators.insert(
            "evaluator1".to_string(),
            ExactMatch(ExactMatchConfig::default()),
        );
        evaluators.insert(
            "evaluator2".to_string(),
            ExactMatch(ExactMatchConfig::default()),
        );

        let manager = StoppingManager::new(&evaluators, precision_targets);

        // Initially none stopped
        assert!(!manager.all_evaluators_stopped());

        // Get tokens and cancel them manually (simulating convergence)
        let tokens = manager.get_tokens();
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
    fn test_evaluator_not_in_precision_targets() {
        let mut precision_targets = HashMap::new();
        precision_targets.insert("evaluator1".to_string(), 0.1);

        let mut evaluators = HashMap::new();
        evaluators.insert(
            "evaluator1".to_string(),
            ExactMatch(ExactMatchConfig::default()),
        );
        evaluators.insert(
            "evaluator2".to_string(),
            ExactMatch(ExactMatchConfig::default()),
        );

        let mut manager = StoppingManager::new(&evaluators, precision_targets);

        // Verify that tokens exist for both evaluators (even though only evaluator1 has precision limit)
        let tokens = manager.get_tokens();
        assert!(tokens.contains_key("evaluator1"));
        assert!(tokens.contains_key("evaluator2"));

        // Add results for evaluator2 (not in precision_targets)
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
