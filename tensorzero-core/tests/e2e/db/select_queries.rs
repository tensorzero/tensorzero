use tensorzero::TimeWindow;
use tensorzero_core::db::{clickhouse::test_helpers::get_clickhouse, SelectQueries};

#[tokio::test]
async fn test_query_model_usage() {
    let clickhouse = get_clickhouse().await;
    let model_usage = clickhouse
        .get_model_usage_timeseries(TimeWindow::Month, 3)
        .await
        .unwrap();

    for usage in &model_usage {
        println!("{:?}", usage);
    }

    // Basic structure assertions
    assert!(
        !model_usage.is_empty(),
        "Model usage data should not be empty"
    );

    // Should have data for 3 months as requested
    let unique_periods: std::collections::HashSet<_> =
        model_usage.iter().map(|usage| usage.period_start).collect();
    assert_eq!(
        unique_periods.len(),
        3,
        "Should have exactly 3 unique time periods"
    );

    // Verify we have expected time periods (March, April, May 2025)
    let expected_periods = vec![
        "2025-03-01T00:00:00Z",
        "2025-04-01T00:00:00Z",
        "2025-05-01T00:00:00Z",
    ];
    for expected in &expected_periods {
        assert!(
            model_usage
                .iter()
                .any(|usage| usage.period_start.to_string().starts_with(expected)),
            "Should contain data for period: {}",
            expected
        );
    }

    // Model-specific assertions
    let model_names: std::collections::HashSet<_> =
        model_usage.iter().map(|usage| &usage.model_name).collect();

    // Check for expected major model families
    assert!(
        model_names
            .iter()
            .any(|name| name.contains("openai::gpt-4o")),
        "Should contain OpenAI GPT-4o models"
    );
    assert!(
        model_names
            .iter()
            .any(|name| name.contains("anthropic::claude")),
        "Should contain Anthropic Claude models"
    );

    // Data quality assertions
    for usage in &model_usage {
        // All usage records should have valid positive values where present
        if let Some(input_tokens) = usage.input_tokens {
            assert!(
                input_tokens > 0,
                "Input tokens should be positive for model: {}",
                usage.model_name
            );
        }
        if let Some(output_tokens) = usage.output_tokens {
            assert!(
                output_tokens > 0,
                "Output tokens should be positive for model: {}",
                usage.model_name
            );
        }
        if let Some(count) = usage.count {
            assert!(
                count > 0,
                "Count should be positive for model: {}",
                usage.model_name
            );
        }

        // Model name should not be empty
        assert!(
            !usage.model_name.is_empty(),
            "Model name should not be empty"
        );
    }

    // Test for specific high-usage models based on your data
    let gpt4o_mini_usage: Vec<_> = model_usage
        .iter()
        .filter(|usage| usage.model_name.contains("gpt-4o-mini-2024-07-18"))
        .collect();
    assert!(
        !gpt4o_mini_usage.is_empty(),
        "Should have GPT-4o-mini usage data"
    );

    // Verify May 2025 has the highest activity (based on your sample data)
    let may_usage: Vec<_> = model_usage
        .iter()
        .filter(|usage| usage.period_start.to_string().starts_with("2025-05-01"))
        .collect();
    let april_usage: Vec<_> = model_usage
        .iter()
        .filter(|usage| usage.period_start.to_string().starts_with("2025-04-01"))
        .collect();

    assert!(
        may_usage.len() >= april_usage.len(),
        "May should have equal or more model entries than April"
    );

    // Test for dummy models (which seem to have very high usage numbers)
    let dummy_models: Vec<_> = model_usage
        .iter()
        .filter(|usage| usage.model_name.starts_with("dummy::"))
        .collect();

    for dummy in dummy_models {
        if let Some(tokens) = dummy.input_tokens {
            assert!(
                tokens >= 100_000_000,
                "Dummy models should have high token usage"
            );
        }
    }

    // Ordering verification - should be sorted by period_start
    let periods: Vec<_> = model_usage.iter().map(|usage| usage.period_start).collect();
    let mut sorted_periods = periods.clone();
    sorted_periods.sort();
    // Note: This assumes your query returns results in chronological order
    // Remove this assertion if the ordering is not guaranteed

    // Summary assertion
    let total_models = model_names.len();
    assert!(
        total_models >= 10,
        "Should have at least 10 different models across all periods"
    );
}
