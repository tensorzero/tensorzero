/// Tests for public evaluation stats API
/// These tests verify that evaluation statistics types and utility functions
/// are accessible from external consumers of the evaluations crate.

#[test]
fn test_evaluation_types_are_public() {
    use evaluations::{EvaluationInfo, EvaluationStats, EvaluatorStats, mean, std_deviation};

    // If this compiles, the types are public
    let _: Option<EvaluatorStats> = None;
    let _: Option<EvaluationInfo> = None;
    let _: Option<EvaluationStats> = None;

    // Test utility functions are accessible
    let data = vec![1.0, 2.0, 3.0];
    let m = mean(&data);
    let s = std_deviation(&data);

    assert!(m.is_some());
    assert!(s.is_some());
}

#[test]
fn test_evaluator_stats_clone_and_debug() {
    use evaluations::EvaluatorStats;

    let stats = EvaluatorStats {
        mean: 0.85,
        stderr: 0.05,
        count: 100,
    };

    // Test Clone
    let cloned = stats.clone();
    assert_eq!(cloned.mean, stats.mean);
    assert_eq!(cloned.stderr, stats.stderr);
    assert_eq!(cloned.count, stats.count);

    // Test Debug
    let debug_str = format!("{stats:?}");
    assert!(debug_str.contains("mean"));
    assert!(debug_str.contains("0.85"));
}

#[test]
fn test_mean_calculation() {
    use evaluations::mean;

    // Test normal case
    assert_eq!(mean(&[1.0, 2.0, 3.0]), Some(2.0));

    // Test empty array
    assert_eq!(mean(&[]), None);

    // Test single element
    assert_eq!(mean(&[5.0]), Some(5.0));

    // Test larger dataset
    let data = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    assert_eq!(mean(&data), Some(30.0));
}

#[test]
fn test_std_deviation_calculation() {
    use evaluations::std_deviation;

    // Test normal case with known standard deviation
    let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    let std = std_deviation(&data).unwrap();
    // The standard deviation of this dataset is 2.0
    assert!((std - 2.0).abs() < 0.1);

    // Test empty array
    assert_eq!(std_deviation(&[]), None);

    // Test single element (std dev should be 0)
    let std = std_deviation(&[5.0]).unwrap();
    assert!(std.abs() < 0.001);

    // Test dataset with no variance
    let data = vec![3.0, 3.0, 3.0, 3.0];
    let std = std_deviation(&data).unwrap();
    assert!(std.abs() < 0.001);
}

#[test]
fn test_mean_and_std_dev_together() {
    use evaluations::{mean, std_deviation};

    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let m = mean(&data).unwrap();
    let s = std_deviation(&data).unwrap();

    // Mean should be 3.0
    assert!((m - 3.0).abs() < 0.001);

    // Standard deviation should be approximately 1.414
    assert!((s - 1.414).abs() < 0.01);
}
