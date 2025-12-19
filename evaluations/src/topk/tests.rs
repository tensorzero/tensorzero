use super::*;
use crate::DatapointVariantError;
use crate::DatapointVariantResult;
use crate::betting_confidence_sequences::{WealthProcessGridPoints, WealthProcesses};
use crate::types::EvaluationVariant;
use serde_json::{Value, json};
use std::sync::Arc;
use tensorzero_core::client::Input;
use tensorzero_core::endpoints::datasets::{ChatInferenceDatapoint, Datapoint};
use tensorzero_core::endpoints::inference::{ChatInferenceResponse, InferenceResponse};
use tensorzero_core::inference::types::{ContentBlockChatOutput, Text, Usage};
use tensorzero_core::tool::DynamicToolParams;

// ============================================================================
// Test Helpers
// ============================================================================

/// Helper to create a confidence sequence with specified bounds (for testing stopping conditions)
fn mock_cs_with_bounds(
    name: &str,
    cs_lower: f64,
    cs_upper: f64,
) -> (String, MeanBettingConfidenceSequence) {
    (
        name.to_string(),
        MeanBettingConfidenceSequence {
            name: name.to_string(),
            mean_regularized: (cs_lower + cs_upper) / 2.0,
            variance_regularized: 0.1,
            count: 100,
            mean_est: (cs_lower + cs_upper) / 2.0,
            cs_lower,
            cs_upper,
            alpha: 0.05,
            wealth: WealthProcesses {
                grid: WealthProcessGridPoints::Resolution(101),
                wealth_upper: vec![1.0; 101],
                wealth_lower: vec![1.0; 101],
            },
        },
    )
}

/// Helper to create a fresh confidence sequence with no observations (for testing updates)
fn mock_fresh_cs(name: &str) -> MeanBettingConfidenceSequence {
    MeanBettingConfidenceSequence {
        name: name.to_string(),
        mean_regularized: 0.5,
        variance_regularized: 0.25,
        count: 0,
        mean_est: 0.5,
        cs_lower: 0.0,
        cs_upper: 1.0,
        alpha: 0.05,
        wealth: WealthProcesses {
            grid: WealthProcessGridPoints::Resolution(101),
            wealth_upper: vec![1.0; 101],
            wealth_lower: vec![1.0; 101],
        },
    }
}

/// Helper to create a mock DatapointVariantResult for testing.
/// Takes a variant name and evaluation results (evaluator_name -> value).
fn mock_success(
    datapoint_id: Uuid,
    variant_name: &str,
    eval_results: HashMap<String, Result<Option<Value>>>,
) -> BatchItemResult {
    // Create a minimal ChatInferenceDatapoint
    let datapoint = Datapoint::Chat(ChatInferenceDatapoint {
        dataset_name: "test_dataset".to_string(),
        function_name: "test_function".to_string(),
        id: datapoint_id,
        episode_id: None,
        input: Input::default(),
        output: None,
        tool_params: DynamicToolParams::default(),
        tags: None,
        auxiliary: String::new(),
        source_inference_id: None,
        staled_at: None,
        updated_at: "2024-01-01T00:00:00Z".to_string(),
        is_deleted: false,
        is_custom: false,
        name: None,
    });

    // Create a minimal InferenceResponse
    let inference_response = InferenceResponse::Chat(ChatInferenceResponse {
        inference_id: Uuid::now_v7(),
        episode_id: Uuid::now_v7(),
        variant_name: variant_name.to_string(),
        content: vec![ContentBlockChatOutput::Text(Text {
            text: "test output".to_string(),
        })],
        usage: Usage {
            input_tokens: Some(0),
            output_tokens: Some(0),
        },
        original_response: None,
        finish_reason: None,
    });

    BatchItemResult::Success(Box::new(DatapointVariantResult {
        datapoint: Arc::new(datapoint),
        variant: Arc::new(EvaluationVariant::Name(variant_name.to_string())),
        inference_response: Arc::new(inference_response),
        evaluation_result: eval_results,
    }))
}

/// A simple scoring function that extracts each variant's first evaluator value as its score
struct FirstEvaluatorScore;

impl ScoringFunction for FirstEvaluatorScore {
    fn score(&self, evaluations: &HashMap<String, &EvaluationResult>) -> HashMap<String, f64> {
        let mut scores = HashMap::new();
        for (variant_name, eval_result) in evaluations {
            // Return the first Ok(Some(value)) we find, converted to f64
            for result in eval_result.values() {
                if let Ok(Some(value)) = result
                    && let Some(num) = value.as_f64()
                {
                    scores.insert(variant_name.clone(), num);
                    break;
                }
            }
        }
        scores
    }
}

// ============================================================================
// Tests for compute_updates
// ============================================================================

/// Test that compute_updates handles empty results gracefully (no changes to CS maps).
#[test]
fn test_compute_updates_empty_results() {
    let scoring_fn = FirstEvaluatorScore;
    let mut variant_performance: HashMap<String, MeanBettingConfidenceSequence> =
        [mock_cs_with_bounds("test_variant", 0.3, 0.7)]
            .into_iter()
            .collect();
    let mut variant_failures: HashMap<String, MeanBettingConfidenceSequence> =
        [mock_cs_with_bounds("test_variant", 0.1, 0.4)]
            .into_iter()
            .collect();
    let mut evaluator_failures: HashMap<String, MeanBettingConfidenceSequence> =
        [mock_cs_with_bounds("evaluator1", 0.05, 0.2)]
            .into_iter()
            .collect();

    let results_by_datapoint: BatchResultsByDatapoint = HashMap::new();
    let result = compute_updates(
        &results_by_datapoint,
        &scoring_fn,
        &mut variant_performance,
        &mut variant_failures,
        &mut evaluator_failures,
    );

    assert!(result.is_ok());

    // Verify variant_performance is completely unchanged
    let vp = &variant_performance["test_variant"];
    assert_eq!(vp.name, "test_variant");
    assert_eq!(vp.mean_regularized, 0.5); // (0.3 + 0.7) / 2
    assert_eq!(vp.variance_regularized, 0.1);
    assert_eq!(vp.count, 100);
    assert_eq!(vp.mean_est, 0.5);
    assert_eq!(vp.cs_lower, 0.3);
    assert_eq!(vp.cs_upper, 0.7);
    assert_eq!(vp.alpha, 0.05);
    assert_eq!(vp.wealth.wealth_upper, vec![1.0; 101]);
    assert_eq!(vp.wealth.wealth_lower, vec![1.0; 101]);

    // Verify variant_failures is completely unchanged
    let vf = &variant_failures["test_variant"];
    assert_eq!(vf.name, "test_variant");
    assert_eq!(vf.mean_regularized, 0.25); // (0.1 + 0.4) / 2
    assert_eq!(vf.variance_regularized, 0.1);
    assert_eq!(vf.count, 100);
    assert_eq!(vf.mean_est, 0.25);
    assert_eq!(vf.cs_lower, 0.1);
    assert_eq!(vf.cs_upper, 0.4);
    assert_eq!(vf.alpha, 0.05);
    assert_eq!(vf.wealth.wealth_upper, vec![1.0; 101]);
    assert_eq!(vf.wealth.wealth_lower, vec![1.0; 101]);

    // Verify evaluator_failures is completely unchanged
    let ef = &evaluator_failures["evaluator1"];
    assert_eq!(ef.name, "evaluator1");
    assert_eq!(ef.mean_regularized, 0.125); // (0.05 + 0.2) / 2
    assert_eq!(ef.variance_regularized, 0.1);
    assert_eq!(ef.count, 100);
    assert_eq!(ef.mean_est, 0.125);
    assert_eq!(ef.cs_lower, 0.05);
    assert_eq!(ef.cs_upper, 0.2);
    assert_eq!(ef.alpha, 0.05);
    assert_eq!(ef.wealth.wealth_upper, vec![1.0; 101]);
    assert_eq!(ef.wealth.wealth_lower, vec![1.0; 101]);
}

/// Test that variant-level errors (BatchItemResult::Error) update failure CS but not performance/evaluator CS.
#[test]
fn test_compute_updates_variant_failures() {
    let scoring_fn = FirstEvaluatorScore;
    let mut variant_performance: HashMap<String, MeanBettingConfidenceSequence> =
        [("test_variant".to_string(), mock_fresh_cs("test_variant"))]
            .into_iter()
            .collect();
    let mut variant_failures: HashMap<String, MeanBettingConfidenceSequence> =
        [("test_variant".to_string(), mock_fresh_cs("test_variant"))]
            .into_iter()
            .collect();
    let mut evaluator_failures: HashMap<String, MeanBettingConfidenceSequence> =
        [("evaluator1".to_string(), mock_fresh_cs("evaluator1"))]
            .into_iter()
            .collect();

    // Create two errors for two different datapoints
    let datapoint_id_1 = Uuid::now_v7();
    let datapoint_id_2 = Uuid::now_v7();

    // Structure: datapoint_id -> variant_name -> result
    let results_by_datapoint: BatchResultsByDatapoint = [
        (
            datapoint_id_1,
            [(
                "test_variant".to_string(),
                BatchItemResult::Error(DatapointVariantError {
                    datapoint_id: datapoint_id_1,
                    variant: None,
                    message: "test error 1".to_string(),
                }),
            )]
            .into_iter()
            .collect(),
        ),
        (
            datapoint_id_2,
            [(
                "test_variant".to_string(),
                BatchItemResult::Error(DatapointVariantError {
                    datapoint_id: datapoint_id_2,
                    variant: None,
                    message: "test error 2".to_string(),
                }),
            )]
            .into_iter()
            .collect(),
        ),
    ]
    .into_iter()
    .collect();

    let result = compute_updates(
        &results_by_datapoint,
        &scoring_fn,
        &mut variant_performance,
        &mut variant_failures,
        &mut evaluator_failures,
    );

    assert!(result.is_ok());
    // variant_failures should have 2 observations (both 1.0 = failure)
    assert_eq!(variant_failures["test_variant"].count, 2);
    // Performance should not be updated since there were no successes
    assert_eq!(variant_performance["test_variant"].count, 0);
    // evaluator_failures should not be updated (variant failed before evaluation ran)
    assert_eq!(evaluator_failures["evaluator1"].count, 0);
}

/// Test that results for variants not in the CS maps are silently ignored.
#[test]
fn test_compute_updates_missing_variant_in_map() {
    let scoring_fn = FirstEvaluatorScore;
    // Don't include "test_variant" in confidence sequence maps - should handle gracefully
    let mut variant_performance: HashMap<String, MeanBettingConfidenceSequence> = HashMap::new();
    let mut variant_failures: HashMap<String, MeanBettingConfidenceSequence> = HashMap::new();
    let mut evaluator_failures: HashMap<String, MeanBettingConfidenceSequence> = HashMap::new();

    let datapoint_id = Uuid::now_v7();
    let results_by_datapoint: BatchResultsByDatapoint = [(
        datapoint_id,
        [(
            "test_variant".to_string(),
            BatchItemResult::Error(DatapointVariantError {
                datapoint_id,
                variant: None,
                message: "test error".to_string(),
            }),
        )]
        .into_iter()
        .collect(),
    )]
    .into_iter()
    .collect();

    // Should not panic, just skip updates for missing variants
    let result = compute_updates(
        &results_by_datapoint,
        &scoring_fn,
        &mut variant_performance,
        &mut variant_failures,
        &mut evaluator_failures,
    );

    assert!(result.is_ok());
    // No maps should be modified since variant wasn't in any of them
    assert!(variant_performance.is_empty());
    assert!(variant_failures.is_empty());
    assert!(evaluator_failures.is_empty());
}

/// Test that successful evaluations update performance and evaluator CS correctly.
#[test]
fn test_compute_updates_successful_evaluations() {
    let scoring_fn = FirstEvaluatorScore;
    let mut variant_performance: HashMap<String, MeanBettingConfidenceSequence> = [
        ("variant_a".to_string(), mock_fresh_cs("variant_a")),
        ("variant_b".to_string(), mock_fresh_cs("variant_b")),
    ]
    .into_iter()
    .collect();
    let mut variant_failures: HashMap<String, MeanBettingConfidenceSequence> = [
        ("variant_a".to_string(), mock_fresh_cs("variant_a")),
        ("variant_b".to_string(), mock_fresh_cs("variant_b")),
    ]
    .into_iter()
    .collect();
    let mut evaluator_failures: HashMap<String, MeanBettingConfidenceSequence> =
        [("evaluator1".to_string(), mock_fresh_cs("evaluator1"))]
            .into_iter()
            .collect();

    let datapoint_id = Uuid::now_v7();

    // Both variants succeed with different scores
    let results_by_datapoint: BatchResultsByDatapoint = [(
        datapoint_id,
        [
            (
                "variant_a".to_string(),
                mock_success(
                    datapoint_id,
                    "variant_a",
                    [("evaluator1".to_string(), Ok(Some(json!(0.8))))]
                        .into_iter()
                        .collect(),
                ),
            ),
            (
                "variant_b".to_string(),
                mock_success(
                    datapoint_id,
                    "variant_b",
                    [("evaluator1".to_string(), Ok(Some(json!(0.6))))]
                        .into_iter()
                        .collect(),
                ),
            ),
        ]
        .into_iter()
        .collect(),
    )]
    .into_iter()
    .collect();

    let result = compute_updates(
        &results_by_datapoint,
        &scoring_fn,
        &mut variant_performance,
        &mut variant_failures,
        &mut evaluator_failures,
    );

    assert!(result.is_ok());

    // Check variant_a performance (observation = 0.8)
    // mean_regularized = (0.5 * 1 + 0.8) / 2 = 0.65
    // variance_regularized = (0.25 * 1 + (0.8 - 0.65)^2) / 2 = 0.13625
    // mean_est = 0.8
    let vp_a = &variant_performance["variant_a"];
    assert_eq!(vp_a.count, 1);
    assert!((vp_a.mean_regularized - 0.65).abs() < 1e-10);
    assert!((vp_a.variance_regularized - 0.13625).abs() < 1e-10);
    assert!((vp_a.mean_est - 0.8).abs() < 1e-10);

    // Check variant_b performance (observation = 0.6)
    // mean_regularized = (0.5 * 1 + 0.6) / 2 = 0.55
    // variance_regularized = (0.25 * 1 + (0.6 - 0.55)^2) / 2 = 0.12625
    // mean_est = 0.6
    let vp_b = &variant_performance["variant_b"];
    assert_eq!(vp_b.count, 1);
    assert!((vp_b.mean_regularized - 0.55).abs() < 1e-10);
    assert!((vp_b.variance_regularized - 0.12625).abs() < 1e-10);
    assert!((vp_b.mean_est - 0.6).abs() < 1e-10);

    // Check variant_a failures (observation = 0.0, success)
    // mean_regularized = (0.5 * 1 + 0.0) / 2 = 0.25
    // variance_regularized = (0.25 * 1 + (0.0 - 0.25)^2) / 2 = 0.15625
    // mean_est = 0.0
    let vf_a = &variant_failures["variant_a"];
    assert_eq!(vf_a.count, 1);
    assert!((vf_a.mean_regularized - 0.25).abs() < 1e-10);
    assert!((vf_a.variance_regularized - 0.15625).abs() < 1e-10);
    assert!((vf_a.mean_est - 0.0).abs() < 1e-10);

    // Check variant_b failures (observation = 0.0, success)
    let vf_b = &variant_failures["variant_b"];
    assert_eq!(vf_b.count, 1);
    assert!((vf_b.mean_regularized - 0.25).abs() < 1e-10);
    assert!((vf_b.variance_regularized - 0.15625).abs() < 1e-10);
    assert!((vf_b.mean_est - 0.0).abs() < 1e-10);

    // Check evaluator1 failures (2 observations, both = 0.0, success)
    // After first observation (0.0):
    //   mean_regularized = (0.5 * 1 + 0.0) / 2 = 0.25
    //   variance_regularized = (0.25 * 1 + (0.0 - 0.25)^2) / 2 = 0.15625
    // After second observation (0.0):
    //   mean_regularized = (0.25 * 2 + 0.0) / 3 = 0.16666...
    //   variance_regularized = (0.15625 * 2 + (0.0 - 0.16666...)^2) / 3 = 0.11319...
    // mean_est = (0.0 + 0.0) / 2 = 0.0
    let ef = &evaluator_failures["evaluator1"];
    assert_eq!(ef.count, 2);
    assert!((ef.mean_regularized - 1.0 / 6.0).abs() < 1e-10);
    assert!(
        (ef.variance_regularized - (0.15625 * 2.0 + (1.0_f64 / 6.0).powi(2)) / 3.0).abs() < 1e-10
    );
    assert!((ef.mean_est - 0.0).abs() < 1e-10);
}

/// Test that evaluator errors within successful inferences update evaluator failure CS.
#[test]
fn test_compute_updates_evaluator_failures() {
    let scoring_fn = FirstEvaluatorScore;
    let mut variant_performance: HashMap<String, MeanBettingConfidenceSequence> =
        [("test_variant".to_string(), mock_fresh_cs("test_variant"))]
            .into_iter()
            .collect();
    let mut variant_failures: HashMap<String, MeanBettingConfidenceSequence> =
        [("test_variant".to_string(), mock_fresh_cs("test_variant"))]
            .into_iter()
            .collect();
    let mut evaluator_failures: HashMap<String, MeanBettingConfidenceSequence> = [
        ("evaluator1".to_string(), mock_fresh_cs("evaluator1")),
        ("evaluator2".to_string(), mock_fresh_cs("evaluator2")),
    ]
    .into_iter()
    .collect();

    let datapoint_id = Uuid::now_v7();

    // Variant succeeds but one evaluator fails
    let results_by_datapoint: BatchResultsByDatapoint = [(
        datapoint_id,
        [(
            "test_variant".to_string(),
            mock_success(
                datapoint_id,
                "test_variant",
                [
                    ("evaluator1".to_string(), Ok(Some(json!(0.7)))),
                    (
                        "evaluator2".to_string(),
                        Err(anyhow::anyhow!("evaluator error")),
                    ),
                ]
                .into_iter()
                .collect(),
            ),
        )]
        .into_iter()
        .collect(),
    )]
    .into_iter()
    .collect();

    let result = compute_updates(
        &results_by_datapoint,
        &scoring_fn,
        &mut variant_performance,
        &mut variant_failures,
        &mut evaluator_failures,
    );

    assert!(result.is_ok());
    // Variant succeeded so variant_failures should have 1 observation (0.0)
    assert_eq!(variant_failures["test_variant"].count, 1);
    // Performance should be updated (scoring function uses first successful evaluator)
    assert_eq!(variant_performance["test_variant"].count, 1);
    // evaluator1 succeeded, evaluator2 failed
    assert_eq!(evaluator_failures["evaluator1"].count, 1);
    assert_eq!(evaluator_failures["evaluator2"].count, 1);
}

/// Test mixed scenario: one variant succeeds, one fails - verify correct CS updates for each.
#[test]
fn test_compute_updates_mixed_success_and_failure() {
    let scoring_fn = FirstEvaluatorScore;
    let mut variant_performance: HashMap<String, MeanBettingConfidenceSequence> = [
        ("variant_a".to_string(), mock_fresh_cs("variant_a")),
        ("variant_b".to_string(), mock_fresh_cs("variant_b")),
    ]
    .into_iter()
    .collect();
    let mut variant_failures: HashMap<String, MeanBettingConfidenceSequence> = [
        ("variant_a".to_string(), mock_fresh_cs("variant_a")),
        ("variant_b".to_string(), mock_fresh_cs("variant_b")),
    ]
    .into_iter()
    .collect();
    let mut evaluator_failures: HashMap<String, MeanBettingConfidenceSequence> =
        [("evaluator1".to_string(), mock_fresh_cs("evaluator1"))]
            .into_iter()
            .collect();

    let datapoint_id = Uuid::now_v7();

    // variant_a succeeds, variant_b fails
    let results_by_datapoint: BatchResultsByDatapoint = [(
        datapoint_id,
        [
            (
                "variant_a".to_string(),
                mock_success(
                    datapoint_id,
                    "variant_a",
                    [("evaluator1".to_string(), Ok(Some(json!(0.9))))]
                        .into_iter()
                        .collect(),
                ),
            ),
            (
                "variant_b".to_string(),
                BatchItemResult::Error(DatapointVariantError {
                    datapoint_id,
                    variant: None,
                    message: "variant error".to_string(),
                }),
            ),
        ]
        .into_iter()
        .collect(),
    )]
    .into_iter()
    .collect();

    let result = compute_updates(
        &results_by_datapoint,
        &scoring_fn,
        &mut variant_performance,
        &mut variant_failures,
        &mut evaluator_failures,
    );

    assert!(result.is_ok());
    // variant_a succeeded: performance and failures updated
    assert_eq!(variant_performance["variant_a"].count, 1);
    assert_eq!(variant_failures["variant_a"].count, 1);
    // variant_b failed: only failures updated (count as failure), no performance
    assert_eq!(variant_performance["variant_b"].count, 0);
    assert_eq!(variant_failures["variant_b"].count, 1);
    // evaluator only ran for variant_a
    assert_eq!(evaluator_failures["evaluator1"].count, 1);
}

// ============================================================================
// Tests for check_topk_stopping
// ============================================================================

/// Test that empty input returns no stopping (graceful handling).
#[test]
fn test_check_topk_stopping_empty() {
    let variant_performance: HashMap<String, MeanBettingConfidenceSequence> = HashMap::new();
    let result = check_topk_stopping(&variant_performance, 1, 1, None).unwrap();
    assert!(!result.stopped);
    assert!(result.k.is_none());
    assert!(result.top_variants.is_empty());
}

/// Test that k_min=0 returns an error.
#[test]
fn test_check_topk_stopping_k_min_zero_errors() {
    let variant_performance: HashMap<String, MeanBettingConfidenceSequence> =
        [mock_cs_with_bounds("a", 0.5, 0.7)].into_iter().collect();

    let result = check_topk_stopping(&variant_performance, 0, 1, None);
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("k_min must be > 0")
    );
}

/// Test that k_max < k_min returns an error.
#[test]
fn test_check_topk_stopping_k_max_less_than_k_min_errors() {
    let variant_performance: HashMap<String, MeanBettingConfidenceSequence> =
        [mock_cs_with_bounds("a", 0.5, 0.7)].into_iter().collect();

    let result = check_topk_stopping(&variant_performance, 2, 1, None);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("k_max"));
}

/// Test that k_min > num_variants returns no stopping (graceful handling).
#[test]
fn test_check_topk_stopping_k_min_exceeds_num_variants() {
    let variant_performance: HashMap<String, MeanBettingConfidenceSequence> =
        [mock_cs_with_bounds("a", 0.5, 0.7)].into_iter().collect();

    let result = check_topk_stopping(&variant_performance, 5, 5, None).unwrap();
    assert!(!result.stopped);
    assert!(result.k.is_none());
    assert!(result.top_variants.is_empty());
}

/// Test that negative epsilon returns an error.
#[test]
fn test_check_topk_stopping_negative_epsilon_errors() {
    let variant_performance: HashMap<String, MeanBettingConfidenceSequence> = [
        mock_cs_with_bounds("a", 0.7, 0.9),
        mock_cs_with_bounds("b", 0.3, 0.5),
    ]
    .into_iter()
    .collect();

    let result = check_topk_stopping(&variant_performance, 1, 1, Some(-0.1));
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("epsilon"));
}

/// Test top-1 identification when one variant clearly dominates all others.
#[test]
fn test_check_topk_stopping_clear_winner() {
    // Variant A: [0.7, 0.9] - clearly better
    // Variant B: [0.3, 0.5] - clearly worse
    // Variant C: [0.2, 0.4] - clearly worse
    // A's lower bound (0.7) exceeds B's upper (0.5) and C's upper (0.4)
    // So A beats 2 variants, which is >= (3 - 1) = 2, so top-1 is identified
    let variant_performance: HashMap<String, MeanBettingConfidenceSequence> = [
        mock_cs_with_bounds("a", 0.7, 0.9),
        mock_cs_with_bounds("b", 0.3, 0.5),
        mock_cs_with_bounds("c", 0.2, 0.4),
    ]
    .into_iter()
    .collect();

    let result = check_topk_stopping(&variant_performance, 1, 1, None).unwrap();
    assert!(result.stopped);
    assert_eq!(result.k, Some(1));
    assert_eq!(result.top_variants.len(), 1);
    assert!(result.top_variants.contains(&"a".to_string()));
}

/// Test top-2 identification when two variants clearly beat a third.
#[test]
fn test_check_topk_stopping_top2() {
    // Variant A: [0.7, 0.9] - in top 2
    // Variant B: [0.6, 0.8] - in top 2
    // Variant C: [0.2, 0.4] - clearly worse
    // A's lower (0.7) > C's upper (0.4), so A beats 1
    // B's lower (0.6) > C's upper (0.4), so B beats 1
    // For top-2, each needs to beat >= (3 - 2) = 1 variant
    let variant_performance: HashMap<String, MeanBettingConfidenceSequence> = [
        mock_cs_with_bounds("a", 0.7, 0.9),
        mock_cs_with_bounds("b", 0.6, 0.8),
        mock_cs_with_bounds("c", 0.2, 0.4),
    ]
    .into_iter()
    .collect();

    let result = check_topk_stopping(&variant_performance, 2, 2, None).unwrap();
    assert!(result.stopped);
    assert_eq!(result.k, Some(2));
    assert_eq!(result.top_variants.len(), 2);
    assert!(result.top_variants.contains(&"a".to_string()));
    assert!(result.top_variants.contains(&"b".to_string()));
}

/// Test that overlapping confidence intervals prevent top-1 identification.
#[test]
fn test_check_topk_stopping_overlapping_intervals_no_stop() {
    // All intervals overlap significantly - can't distinguish
    // Variant A: [0.4, 0.7]
    // Variant B: [0.45, 0.65]
    // Variant C: [0.5, 0.6]
    // No lower bound exceeds any upper bound, so no one "beats" anyone
    let variant_performance: HashMap<String, MeanBettingConfidenceSequence> = [
        mock_cs_with_bounds("a", 0.4, 0.7),
        mock_cs_with_bounds("b", 0.45, 0.65),
        mock_cs_with_bounds("c", 0.5, 0.6),
    ]
    .into_iter()
    .collect();

    let result = check_topk_stopping(&variant_performance, 1, 1, None).unwrap();
    assert!(!result.stopped);
    assert!(result.k.is_none());
}

/// Test that the largest viable k is returned when checking a range.
#[test]
fn test_check_topk_stopping_k_range() {
    // Variant A: [0.8, 0.9] - beats B and C
    // Variant B: [0.5, 0.7] - beats C only
    // Variant C: [0.2, 0.4] - beats no one
    // A beats 2 (both B and C's uppers are below 0.8)
    // B beats 1 (C's upper 0.4 < B's lower 0.5)
    // C beats 0
    //
    // For k=1: need to beat >= 2. Only A qualifies. Top-1 found.
    // For k=2: need to beat >= 1. A and B qualify. Top-2 found.
    let variant_performance: HashMap<String, MeanBettingConfidenceSequence> = [
        mock_cs_with_bounds("a", 0.8, 0.9),
        mock_cs_with_bounds("b", 0.5, 0.7),
        mock_cs_with_bounds("c", 0.2, 0.4),
    ]
    .into_iter()
    .collect();

    // When k_min=1, k_max=2, should return k=2 (largest k that works)
    let result = check_topk_stopping(&variant_performance, 1, 2, None).unwrap();
    assert!(result.stopped);
    assert_eq!(result.k, Some(2));
    assert_eq!(result.top_variants.len(), 2);

    // When k_min=1, k_max=1, should return k=1
    let result = check_topk_stopping(&variant_performance, 1, 1, None).unwrap();
    assert!(result.stopped);
    assert_eq!(result.k, Some(1));
    assert_eq!(result.top_variants.len(), 1);
    assert!(result.top_variants.contains(&"a".to_string()));
}

/// Test that a single variant is trivially identified as top-1.
#[test]
fn test_check_topk_single_variant() {
    // Single variant - should be identified as top-1
    // It needs to beat >= (1 - 1) = 0 variants, which it does trivially
    let variant_performance: HashMap<String, MeanBettingConfidenceSequence> =
        [mock_cs_with_bounds("a", 0.5, 0.7)].into_iter().collect();

    let result = check_topk_stopping(&variant_performance, 1, 1, None).unwrap();
    assert!(result.stopped);
    assert_eq!(result.k, Some(1));
    assert_eq!(result.top_variants, vec!["a".to_string()]);
}

/// Test that check_topk wrapper correctly calls check_topk_stopping with k_min = k_max.
#[test]
fn test_check_topk_wrapper() {
    let variant_performance: HashMap<String, MeanBettingConfidenceSequence> = [
        mock_cs_with_bounds("a", 0.7, 0.9),
        mock_cs_with_bounds("b", 0.3, 0.5),
        mock_cs_with_bounds("c", 0.2, 0.4),
    ]
    .into_iter()
    .collect();

    let result = check_topk(&variant_performance, 1, None).unwrap();
    assert!(result.stopped);
    assert_eq!(result.k, Some(1));
}

/// Test that epsilon tolerance enables top-1 stopping for nearly-separated intervals.
#[test]
fn test_check_topk_stopping_epsilon_enables_stopping() {
    // Variant A: [0.48, 0.7] - lower bound just below B's upper
    // Variant B: [0.3, 0.5] - upper bound at 0.5
    // Variant C: [0.2, 0.4] - clearly worse
    //
    // Without epsilon: A's lower (0.48) < B's upper (0.5), so A doesn't beat B
    // A only beats C (0.48 > 0.4), so num_beaten = 1
    // For top-1, need to beat >= 2, so no stopping
    //
    // With epsilon = 0.05: A beats B if 0.5 - 0.05 < 0.48, i.e., 0.45 < 0.48 ✓
    // A now beats both B and C, so top-1 is identified
    let variant_performance: HashMap<String, MeanBettingConfidenceSequence> = [
        mock_cs_with_bounds("a", 0.48, 0.7),
        mock_cs_with_bounds("b", 0.3, 0.5),
        mock_cs_with_bounds("c", 0.2, 0.4),
    ]
    .into_iter()
    .collect();

    // Without epsilon, can't identify top-1
    let result = check_topk_stopping(&variant_performance, 1, 1, None).unwrap();
    assert!(!result.stopped);

    // With epsilon = 0.05, top-1 is identified
    let result = check_topk_stopping(&variant_performance, 1, 1, Some(0.05)).unwrap();
    assert!(result.stopped);
    assert_eq!(result.k, Some(1));
    assert!(result.top_variants.contains(&"a".to_string()));
}

/// Test that epsilon tolerance enables identifying a larger top-k set.
#[test]
fn test_check_topk_stopping_epsilon_enables_larger_k() {
    // Variant A: [0.7, 0.9] - clearly best
    // Variant B: [0.48, 0.65] - B's lower (0.48) just below C's upper (0.5)
    // Variant C: [0.3, 0.5] - middle
    // Variant D: [0.1, 0.3] - clearly worst
    //
    // Without epsilon:
    // A beats 3 (all uppers < 0.7)
    // B beats 1 (only D's upper 0.3 < 0.48)
    // C beats 1 (only D's upper 0.3 < 0.3? No, 0.3 is not < 0.3)
    // Actually C's lower is 0.3, D's upper is 0.3, so C beats 0 (0.3 is not < 0.3)
    // For top-2, need to beat >= 2. Only A qualifies.
    //
    // With epsilon = 0.05:
    // B beats C if 0.5 - 0.05 < 0.48, i.e., 0.45 < 0.48 ✓
    // B beats D if 0.3 - 0.05 < 0.48, i.e., 0.25 < 0.48 ✓
    // So B now beats 2, qualifying for top-2
    let variant_performance: HashMap<String, MeanBettingConfidenceSequence> = [
        mock_cs_with_bounds("a", 0.7, 0.9),
        mock_cs_with_bounds("b", 0.48, 0.65),
        mock_cs_with_bounds("c", 0.3, 0.5),
        mock_cs_with_bounds("d", 0.1, 0.3),
    ]
    .into_iter()
    .collect();

    // Without epsilon, can identify top-1 but not top-2
    let result = check_topk_stopping(&variant_performance, 1, 1, None).unwrap();
    assert!(result.stopped);
    assert_eq!(result.k, Some(1));

    let result = check_topk_stopping(&variant_performance, 2, 2, None).unwrap();
    assert!(!result.stopped);

    // With epsilon = 0.05, can identify top-2
    let result = check_topk_stopping(&variant_performance, 2, 2, Some(0.05)).unwrap();
    assert!(result.stopped);
    assert_eq!(result.k, Some(2));
    assert!(result.top_variants.contains(&"a".to_string()));
    assert!(result.top_variants.contains(&"b".to_string()));
}

/// Test that epsilon=0.0 behaves identically to epsilon=None.
#[test]
fn test_check_topk_stopping_epsilon_zero_same_as_none() {
    let variant_performance: HashMap<String, MeanBettingConfidenceSequence> = [
        mock_cs_with_bounds("a", 0.7, 0.9),
        mock_cs_with_bounds("b", 0.3, 0.5),
        mock_cs_with_bounds("c", 0.2, 0.4),
    ]
    .into_iter()
    .collect();

    let result_none = check_topk_stopping(&variant_performance, 1, 2, None).unwrap();
    let result_zero = check_topk_stopping(&variant_performance, 1, 2, Some(0.0)).unwrap();

    assert_eq!(result_none.stopped, result_zero.stopped);
    assert_eq!(result_none.k, result_zero.k);
    // Note: top_variants order might differ, so just check same elements
    assert_eq!(
        result_none.top_variants.len(),
        result_zero.top_variants.len()
    );
}

/// Test that a very large epsilon makes all variants beat all others.
#[test]
fn test_check_topk_stopping_large_epsilon_all_beat_all() {
    // With a very large epsilon, every variant "beats" every other variant
    // (since ub - epsilon will be negative for any reasonable upper bound)
    let variant_performance: HashMap<String, MeanBettingConfidenceSequence> = [
        mock_cs_with_bounds("a", 0.1, 0.2),
        mock_cs_with_bounds("b", 0.3, 0.4),
        mock_cs_with_bounds("c", 0.5, 0.6),
    ]
    .into_iter()
    .collect();

    // With epsilon = 1.0, all variants beat the other 2 variants
    // (a variant never counts itself as beaten)
    // For top-1, need >= 2. All qualify, so we get a top-3 (largest k checked first)
    let result = check_topk_stopping(&variant_performance, 1, 3, Some(1.0)).unwrap();
    assert!(result.stopped);
    assert_eq!(result.k, Some(3));
    assert_eq!(result.top_variants.len(), 3);
}

/// Test that a variant never counts itself as beaten.
#[test]
fn test_check_topk_stopping_variant_does_not_beat_itself() {
    // With 2 variants and large epsilon, each variant beats the other but not itself.
    // So each variant beats exactly 1 other variant.
    // For top-1: need to beat >= 1. Both qualify.
    // For top-2: need to beat >= 0. Both qualify.
    // The key point: with epsilon=1.0, if variants counted themselves, they'd each
    // beat 2 variants and top-1 would be identified. But they shouldn't count themselves.
    let variant_performance: HashMap<String, MeanBettingConfidenceSequence> = [
        mock_cs_with_bounds("a", 0.1, 0.2),
        mock_cs_with_bounds("b", 0.3, 0.4),
    ]
    .into_iter()
    .collect();

    // With epsilon = 1.0, each variant beats exactly 1 other (not itself)
    // For top-1, need to beat >= 1. Both qualify, so we get top-2.
    let result = check_topk_stopping(&variant_performance, 1, 2, Some(1.0)).unwrap();
    assert!(result.stopped);
    assert_eq!(result.k, Some(2));
    assert_eq!(result.top_variants.len(), 2);

    // If variants incorrectly counted themselves, each would beat 2 variants,
    // and top-1 would be viable (need >= 1 beaten). Verify top-1 does NOT
    // incorrectly select just one variant.
    let result = check_topk_stopping(&variant_performance, 1, 1, Some(1.0)).unwrap();
    // Both variants beat exactly 1 other, so top-1 requires beating >= 1, which both do.
    // Since both qualify but we can only pick 1, this should still stop with k=1.
    // The important thing is the count is 1 (not 2).
    assert!(result.stopped);
    assert_eq!(result.k, Some(1));
}

/// Test that k_max caps the returned k, even when larger k values are viable.
#[test]
fn test_check_topk_stopping_returns_largest_viable_k() {
    // Variant A: [0.8, 0.95] - beats all 4 others
    // Variant B: [0.7, 0.85] - beats C, D, E (3 others)
    // Variant C: [0.5, 0.65] - beats D, E (2 others)
    // Variant D: [0.3, 0.45] - beats E (1 other)
    // Variant E: [0.1, 0.25] - beats none
    //
    // For top-1: need to beat >= 4. Only A qualifies.
    // For top-2: need to beat >= 3. A and B qualify.
    // For top-3: need to beat >= 2. A, B, C qualify.
    // For top-4: need to beat >= 1. A, B, C, D qualify.
    // For top-5: need to beat >= 0. All qualify.
    let variant_performance: HashMap<String, MeanBettingConfidenceSequence> = [
        mock_cs_with_bounds("a", 0.8, 0.95),
        mock_cs_with_bounds("b", 0.7, 0.85),
        mock_cs_with_bounds("c", 0.5, 0.65),
        mock_cs_with_bounds("d", 0.3, 0.45),
        mock_cs_with_bounds("e", 0.1, 0.25),
    ]
    .into_iter()
    .collect();

    // k_min=1, k_max=5: should return k=5 (largest viable)
    let result = check_topk_stopping(&variant_performance, 1, 5, None).unwrap();
    assert!(result.stopped);
    assert_eq!(result.k, Some(5));
    assert_eq!(result.top_variants.len(), 5);

    // k_min=1, k_max=3: should return k=3 (largest viable within range)
    let result = check_topk_stopping(&variant_performance, 1, 3, None).unwrap();
    assert!(result.stopped);
    assert_eq!(result.k, Some(3));
    assert_eq!(result.top_variants.len(), 3);
    assert!(result.top_variants.contains(&"a".to_string()));
    assert!(result.top_variants.contains(&"b".to_string()));
    assert!(result.top_variants.contains(&"c".to_string()));

    // k_min=1, k_max=2: should return k=2
    let result = check_topk_stopping(&variant_performance, 1, 2, None).unwrap();
    assert!(result.stopped);
    assert_eq!(result.k, Some(2));
    assert_eq!(result.top_variants.len(), 2);
    assert!(result.top_variants.contains(&"a".to_string()));
    assert!(result.top_variants.contains(&"b".to_string()));

    // k_min=2, k_max=4: should return k=4 (largest viable within range)
    let result = check_topk_stopping(&variant_performance, 2, 4, None).unwrap();
    assert!(result.stopped);
    assert_eq!(result.k, Some(4));
    assert_eq!(result.top_variants.len(), 4);
}

/// Test that no stopping occurs when no k in the range is viable.
#[test]
fn test_check_topk_stopping_k_range_no_viable_k() {
    // Variant A: [0.4, 0.7]
    // Variant B: [0.35, 0.65]
    // Variant C: [0.3, 0.6]
    // All intervals overlap significantly - no one beats anyone
    //
    // For any k < 3, we need variants to beat others, but none do.
    // Only k=3 works (need to beat >= 0).
    let variant_performance: HashMap<String, MeanBettingConfidenceSequence> = [
        mock_cs_with_bounds("a", 0.4, 0.7),
        mock_cs_with_bounds("b", 0.35, 0.65),
        mock_cs_with_bounds("c", 0.3, 0.6),
    ]
    .into_iter()
    .collect();

    // k_min=1, k_max=2: neither k=1 nor k=2 is viable, so no stopping
    let result = check_topk_stopping(&variant_performance, 1, 2, None).unwrap();
    assert!(!result.stopped);
    assert!(result.k.is_none());

    // k_min=1, k_max=3: k=3 is viable (all variants beat >= 0 others)
    let result = check_topk_stopping(&variant_performance, 1, 3, None).unwrap();
    assert!(result.stopped);
    assert_eq!(result.k, Some(3));

    // k_min=3, k_max=3: k=3 is viable
    let result = check_topk_stopping(&variant_performance, 3, 3, None).unwrap();
    assert!(result.stopped);
    assert_eq!(result.k, Some(3));
}

/// Test a "gap" scenario where k=1 and k=3 are viable but k=2 is not.
#[test]
fn test_check_topk_stopping_k_range_partial_viability() {
    // Variant A: [0.7, 0.9] - beats B and C
    // Variant B: [0.4, 0.6] - beats no one (C's upper 0.55 > B's lower 0.4)
    // Variant C: [0.35, 0.55] - beats no one
    //
    // A beats 2 (both uppers < 0.7)
    // B beats 0
    // C beats 0
    //
    // For top-1: need >= 2. Only A qualifies. ✓
    // For top-2: need >= 1. Only A qualifies (1 variant). ✗
    // For top-3: need >= 0. All qualify. ✓
    let variant_performance: HashMap<String, MeanBettingConfidenceSequence> = [
        mock_cs_with_bounds("a", 0.7, 0.9),
        mock_cs_with_bounds("b", 0.4, 0.6),
        mock_cs_with_bounds("c", 0.35, 0.55),
    ]
    .into_iter()
    .collect();

    // k_min=1, k_max=3: k=3 is largest viable
    let result = check_topk_stopping(&variant_performance, 1, 3, None).unwrap();
    assert!(result.stopped);
    assert_eq!(result.k, Some(3));

    // k_min=1, k_max=2: only k=1 is viable (k=2 fails)
    let result = check_topk_stopping(&variant_performance, 1, 2, None).unwrap();
    assert!(result.stopped);
    assert_eq!(result.k, Some(1));
    assert_eq!(result.top_variants, vec!["a".to_string()]);

    // k_min=2, k_max=2: k=2 is not viable, no stopping
    let result = check_topk_stopping(&variant_performance, 2, 2, None).unwrap();
    assert!(!result.stopped);

    // k_min=2, k_max=3: k=3 is viable
    let result = check_topk_stopping(&variant_performance, 2, 3, None).unwrap();
    assert!(result.stopped);
    assert_eq!(result.k, Some(3));
}

/// Test tiebreaking by mean_est when num_beaten is equal.
#[test]
fn test_check_topk_stopping_tiebreak_by_mean_est() {
    // With large epsilon, all variants beat all others (num_beaten = 2 for each)
    // Tiebreaker should be mean_est descending
    // Note: mock_cs_with_bounds sets mean_est = (cs_lower + cs_upper) / 2
    let variant_performance: HashMap<String, MeanBettingConfidenceSequence> = [
        mock_cs_with_bounds("a", 0.1, 0.3), // mean_est = 0.2
        mock_cs_with_bounds("b", 0.4, 0.6), // mean_est = 0.5
        mock_cs_with_bounds("c", 0.7, 0.9), // mean_est = 0.8
    ]
    .into_iter()
    .collect();

    // With epsilon = 1.0, all beat all others
    // For top-1, should return "c" (highest mean_est)
    let result = check_topk_stopping(&variant_performance, 1, 1, Some(1.0)).unwrap();
    assert!(result.stopped);
    assert_eq!(result.k, Some(1));
    assert_eq!(result.top_variants.len(), 1);
    assert!(result.top_variants.contains(&"c".to_string()));

    // For top-2, should return "c" and "b" (top two by mean_est)
    let result = check_topk_stopping(&variant_performance, 2, 2, Some(1.0)).unwrap();
    assert!(result.stopped);
    assert_eq!(result.k, Some(2));
    assert_eq!(result.top_variants.len(), 2);
    assert!(result.top_variants.contains(&"c".to_string()));
    assert!(result.top_variants.contains(&"b".to_string()));
}

/// Test tiebreaking by cs_lower when mean_est is equal.
#[test]
fn test_check_topk_stopping_tiebreak_by_cs_lower() {
    // Create variants with same mean_est but different cs_lower
    // mean_est = (cs_lower + cs_upper) / 2, so we need to adjust both bounds
    let mut variant_performance: HashMap<String, MeanBettingConfidenceSequence> = HashMap::new();

    // All have mean_est = 0.5, but different cs_lower
    let (name_a, mut cs_a) = mock_cs_with_bounds("a", 0.3, 0.7); // mean_est = 0.5, cs_lower = 0.3
    cs_a.mean_est = 0.5;
    variant_performance.insert(name_a, cs_a);

    let (name_b, mut cs_b) = mock_cs_with_bounds("b", 0.4, 0.6); // mean_est = 0.5, cs_lower = 0.4
    cs_b.mean_est = 0.5;
    variant_performance.insert(name_b, cs_b);

    let (name_c, mut cs_c) = mock_cs_with_bounds("c", 0.2, 0.8); // mean_est = 0.5, cs_lower = 0.2
    cs_c.mean_est = 0.5;
    variant_performance.insert(name_c, cs_c);

    // With epsilon = 1.0, all beat all others
    // For top-1, should return "b" (highest cs_lower since mean_est is tied)
    let result = check_topk_stopping(&variant_performance, 1, 1, Some(1.0)).unwrap();
    assert!(result.stopped);
    assert_eq!(result.k, Some(1));
    assert_eq!(result.top_variants.len(), 1);
    assert!(result.top_variants.contains(&"b".to_string()));
}

/// Test that identical variants at the k boundary cause an enlarged result set.
#[test]
fn test_check_topk_stopping_ties_at_boundary_enlarges_set() {
    // Create 4 variants: one clear winner, then 3 identical variants
    let mut variant_performance: HashMap<String, MeanBettingConfidenceSequence> = HashMap::new();

    // Clear winner
    let (name_a, cs_a) = mock_cs_with_bounds("a", 0.8, 0.9);
    variant_performance.insert(name_a, cs_a);

    // Three identical variants (same bounds, same mean_est)
    let (name_b, cs_b) = mock_cs_with_bounds("b", 0.4, 0.6);
    variant_performance.insert(name_b, cs_b);

    let (name_c, cs_c) = mock_cs_with_bounds("c", 0.4, 0.6);
    variant_performance.insert(name_c, cs_c);

    let (name_d, cs_d) = mock_cs_with_bounds("d", 0.4, 0.6);
    variant_performance.insert(name_d, cs_d);

    // With epsilon = 1.0, all beat all others (num_beaten = 3 each)
    // For top-2: "a" is clearly first, but b/c/d are tied for second
    // Should return all 4 variants (enlarged set)
    let result = check_topk_stopping(&variant_performance, 2, 2, Some(1.0)).unwrap();
    assert!(result.stopped);
    assert_eq!(result.k, Some(2));
    assert_eq!(result.top_variants.len(), 4); // Enlarged due to ties
    assert!(result.top_variants.contains(&"a".to_string()));
    assert!(result.top_variants.contains(&"b".to_string()));
    assert!(result.top_variants.contains(&"c".to_string()));
    assert!(result.top_variants.contains(&"d".to_string()));
}

/// Test that non-identical variants at the k boundary do NOT enlarge the set.
#[test]
fn test_check_topk_stopping_no_ties_at_boundary() {
    // Create 4 variants with distinct mean_est values
    let variant_performance: HashMap<String, MeanBettingConfidenceSequence> = [
        mock_cs_with_bounds("a", 0.7, 0.9), // mean_est = 0.8
        mock_cs_with_bounds("b", 0.5, 0.7), // mean_est = 0.6
        mock_cs_with_bounds("c", 0.3, 0.5), // mean_est = 0.4
        mock_cs_with_bounds("d", 0.1, 0.3), // mean_est = 0.2
    ]
    .into_iter()
    .collect();

    // With epsilon = 1.0, all beat all others
    // For top-2, should return exactly 2 variants (a and b by mean_est)
    let result = check_topk_stopping(&variant_performance, 2, 2, Some(1.0)).unwrap();
    assert!(result.stopped);
    assert_eq!(result.k, Some(2));
    assert_eq!(result.top_variants.len(), 2); // No enlargement
    assert!(result.top_variants.contains(&"a".to_string()));
    assert!(result.top_variants.contains(&"b".to_string()));
}

/// Test identical variants (all have same bounds).
#[test]
fn test_check_topk_stopping_all_identical_variants() {
    // All variants have identical confidence sequences
    let variant_performance: HashMap<String, MeanBettingConfidenceSequence> = [
        mock_cs_with_bounds("a", 0.4, 0.6),
        mock_cs_with_bounds("b", 0.4, 0.6),
        mock_cs_with_bounds("c", 0.4, 0.6),
    ]
    .into_iter()
    .collect();

    // With epsilon = 0, no one beats anyone (intervals are identical)
    // Only k=3 is viable (need to beat >= 0)
    let result = check_topk_stopping(&variant_performance, 1, 3, None).unwrap();
    assert!(result.stopped);
    assert_eq!(result.k, Some(3));
    assert_eq!(result.top_variants.len(), 3);

    // With epsilon = 1.0, all beat all others, but for top-1, all are tied
    // Should return all 3 (enlarged set)
    let result = check_topk_stopping(&variant_performance, 1, 1, Some(1.0)).unwrap();
    assert!(result.stopped);
    assert_eq!(result.k, Some(1));
    assert_eq!(result.top_variants.len(), 3); // All tied
}
