/// Helper function to generate a minimal Track-and-Stop config for testing.
///
/// # Arguments
/// * `metric_name` - Name of the metric to use
/// * `candidate_variants` - List of variant names to test
/// * `fallback_variants` - List of fallback variant names
/// * `min_samples_per_variant` - Minimum samples before bandit sampling starts
/// * `delta` - Confidence parameter (0, 1)
/// * `epsilon` - Sub-optimality tolerance (>= 0)
/// * `update_period_s` - How often to update probabilities (seconds)
///
/// Returns a complete TOML config string with:
/// - A dummy model provider
/// - A function with the specified variants
/// - A metric
/// - Track-and-Stop experimentation config
pub fn make_track_and_stop_config(
    metric_name: &str,
    candidate_variants: &[&str],
    fallback_variants: &[&str],
    min_samples_per_variant: u64,
    delta: f64,
    epsilon: f64,
    update_period_s: u64,
) -> String {
    let variant_configs: Vec<String> = candidate_variants
        .iter()
        .map(|variant| {
            format!(
                r#"
[functions.test_function.variants.{variant}]
type = "chat_completion"
model = "test_model"
"#
            )
        })
        .collect();

    let candidate_variants_list = candidate_variants
        .iter()
        .map(|v| format!("\"{v}\""))
        .collect::<Vec<_>>()
        .join(", ");

    let fallback_variants_list = fallback_variants
        .iter()
        .map(|v| format!("\"{v}\""))
        .collect::<Vec<_>>()
        .join(", ");

    format!(
        r#"
[models.test_model]
routing = ["dummy"]

[models.test_model.providers.dummy]
type = "dummy"
model_name = "test"

[metrics.{metric_name}]
type = "float"
optimize = "max"
level = "inference"

[functions.test_function]
type = "chat"
{variants}

[functions.test_function.experimentation]
type = "track_and_stop"
metric = "{metric_name}"
candidate_variants = [{candidate_variants_list}]
fallback_variants = [{fallback_variants_list}]
min_samples_per_variant = {min_samples_per_variant}
delta = {delta}
epsilon = {epsilon}
update_period_s = {update_period_s}
"#,
        variants = variant_configs.join("\n"),
    )
}

#[test]
fn test_make_track_and_stop_config_generates_valid_toml() {
    let config = make_track_and_stop_config(
        "accuracy",
        &["variant_a", "variant_b", "variant_c"],
        &["variant_a", "variant_b"],
        10,
        0.05,
        0.1,
        300,
    );

    // Verify it's valid TOML
    let parsed: toml::Value =
        toml::from_str(&config).expect("Generated config should be valid TOML");

    // Verify key sections exist
    assert!(parsed.get("models").is_some());
    assert!(parsed.get("metrics").is_some());
    assert!(parsed.get("functions").is_some());

    // Verify experimentation config
    let exp = parsed
        .get("functions")
        .and_then(|f| f.get("test_function"))
        .and_then(|tf| tf.get("experimentation"))
        .expect("Experimentation config should exist");

    assert_eq!(
        exp.get("type").and_then(|v| v.as_str()),
        Some("track_and_stop")
    );
    assert_eq!(exp.get("metric").and_then(|v| v.as_str()), Some("accuracy"));
    assert_eq!(
        exp.get("min_samples_per_variant")
            .and_then(toml::Value::as_integer),
        Some(10)
    );
    assert_eq!(exp.get("delta").and_then(toml::Value::as_float), Some(0.05));
    assert_eq!(
        exp.get("epsilon").and_then(toml::Value::as_float),
        Some(0.1)
    );
    assert_eq!(
        exp.get("update_period_s").and_then(toml::Value::as_integer),
        Some(300)
    );
}

// TODO: Helper function to generate Bernoulli or Gaussian bandit environments
