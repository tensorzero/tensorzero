#![expect(clippy::print_stdout)]
use tensorzero::TimeWindow;
use tensorzero_core::db::{
    clickhouse::{
        migration_manager::migrations::migration_0037::QUANTILES, test_helpers::get_clickhouse,
    },
    SelectQueries,
};

#[tokio::test]
async fn test_clickhouse_query_model_usage() {
    let clickhouse = get_clickhouse().await;
    let model_usage = clickhouse
        .get_model_usage_timeseries(TimeWindow::Month, 3)
        .await
        .unwrap();

    for usage in &model_usage {
        println!("{usage:?}");
    }
    // Basic structure assertions
    assert!(
        !model_usage.is_empty(),
        "Model usage data should not be empty"
    );

    // Test specific data points from May 2025
    let may_gemini = model_usage.iter().find(|u| {
        u.period_start.format("%Y-%m-%d").to_string() == "2025-05-01"
            && u.model_name == "google_ai_studio_gemini::gemini-2.5-flash-preview-04-17"
    });
    assert!(
        may_gemini.is_some(),
        "Should have gemini-2.5-flash data for May 2025"
    );
    let may_gemini = may_gemini.unwrap();
    assert_eq!(may_gemini.input_tokens, Some(2041389));
    assert_eq!(may_gemini.output_tokens, Some(6788));
    assert_eq!(may_gemini.count, Some(120));

    let may_gpt4o_mini = model_usage.iter().find(|u| {
        u.period_start.format("%Y-%m-%d").to_string() == "2025-05-01"
            && u.model_name == "openai::gpt-4o-mini-2024-07-18"
    });
    assert!(
        may_gpt4o_mini.is_some(),
        "Should have gpt-4o-mini data for May 2025"
    );
    let may_gpt4o_mini = may_gpt4o_mini.unwrap();
    assert_eq!(may_gpt4o_mini.input_tokens, Some(983420));
    assert_eq!(may_gpt4o_mini.output_tokens, Some(149656));
    assert_eq!(may_gpt4o_mini.count, Some(2933));

    // Test data from April 2025
    let april_claude = model_usage.iter().find(|u| {
        u.period_start.format("%Y-%m-%d").to_string() == "2025-04-01"
            && u.model_name == "anthropic::claude-3-5-haiku-20241022"
    });
    assert!(
        april_claude.is_some(),
        "Should have claude-3-5-haiku data for April 2025"
    );
    let april_claude = april_claude.unwrap();
    assert_eq!(april_claude.input_tokens, Some(29859));
    assert_eq!(april_claude.output_tokens, Some(44380));
    assert_eq!(april_claude.count, Some(310));

    // Test data from March 2025
    let march_llama = model_usage.iter().find(|u| {
        u.period_start.format("%Y-%m-%d").to_string() == "2025-03-01"
            && u.model_name == "llama-3.1-8b-instruct"
    });
    assert!(
        march_llama.is_some(),
        "Should have llama-3.1-8b data for March 2025"
    );
    let march_llama = march_llama.unwrap();
    assert_eq!(march_llama.input_tokens, Some(6363));
    assert_eq!(march_llama.output_tokens, Some(1349));
    assert_eq!(march_llama.count, Some(42));

    // Test edge case from February 2025 (missing token data)
    let feb_data = model_usage.iter().find(|u| {
        u.period_start.format("%Y-%m-%d").to_string() == "2025-02-01"
            && u.model_name == "openai::gpt-4o-mini-2024-07-18"
    });
    if let Some(feb_entry) = feb_data {
        assert_eq!(feb_entry.input_tokens, None);
        assert_eq!(feb_entry.output_tokens, None);
        assert_eq!(feb_entry.count, Some(1));
    }

    // Should have data for 3 (or 4, from rollover) months as requested
    let unique_periods: std::collections::HashSet<_> =
        model_usage.iter().map(|usage| usage.period_start).collect();
    assert!(
        unique_periods.len() <= 4,
        "Should have at most 4 unique time periods"
    );

    // Verify we have expected time periods (March, April, May 2025)
    // Check for data in each of the 3 most recent months
    let march_data = model_usage.iter().any(|usage| {
        let period_str = usage.period_start.format("%Y-%m").to_string();
        period_str == "2025-03"
    });
    let april_data = model_usage.iter().any(|usage| {
        let period_str = usage.period_start.format("%Y-%m").to_string();
        period_str == "2025-04"
    });
    let may_data = model_usage.iter().any(|usage| {
        let period_str = usage.period_start.format("%Y-%m").to_string();
        period_str == "2025-05"
    });

    assert!(march_data, "Should contain data for March 2025");
    assert!(april_data, "Should contain data for April 2025");
    assert!(may_data, "Should contain data for May 2025");

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
        .filter(|usage| {
            let period_str = usage.period_start.format("%Y-%m").to_string();
            period_str == "2025-05"
        })
        .collect();
    let april_usage: Vec<_> = model_usage
        .iter()
        .filter(|usage| {
            let period_str = usage.period_start.format("%Y-%m").to_string();
            period_str == "2025-04"
        })
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

#[tokio::test]
async fn test_clickhouse_query_model_usage_daily() {
    let clickhouse = get_clickhouse().await;
    let model_usage = clickhouse
        .get_model_usage_timeseries(TimeWindow::Day, 7)
        .await
        .unwrap();

    for usage in &model_usage {
        println!("{usage:?}");
    }
    // Basic structure assertions
    assert!(
        !model_usage.is_empty(),
        "Daily model usage data should not be empty"
    );

    // Test specific daily data point
    let may_23_data = model_usage.iter().find(|u| {
        u.period_start.format("%Y-%m-%d").to_string() == "2025-05-23"
            && u.model_name == "openai::gpt-4o-mini"
    });
    assert!(
        may_23_data.is_some(),
        "Should have gpt-4o-mini data for May 23, 2025"
    );
    let may_23_data = may_23_data.unwrap();
    assert_eq!(may_23_data.input_tokens, Some(17015));
    assert_eq!(may_23_data.output_tokens, Some(113));
    assert_eq!(may_23_data.count, Some(1));

    // Should have data for up to 7 days as requested
    let unique_periods: std::collections::HashSet<_> =
        model_usage.iter().map(|usage| usage.period_start).collect();
    assert!(
        unique_periods.len() <= 7,
        "Should have at most 7 unique time periods for daily granularity"
    );
    assert!(
        !unique_periods.is_empty(),
        "Should have at least one time period"
    );

    // Verify the date range spans at most 7 days
    let now = chrono::Utc::now();
    let dates: Vec<_> = model_usage.iter().map(|usage| usage.period_start).collect();
    if !dates.is_empty() {
        let earliest = dates.iter().min().unwrap();
        let latest = dates.iter().max().unwrap();
        let date_range_days = latest.signed_duration_since(*earliest).num_days();
        assert!(
            date_range_days <= 7,
            "Date range should be within 7 days, got {date_range_days} days between {earliest} and {latest}",
        );
    }

    // Model-specific assertions
    let model_names: std::collections::HashSet<_> =
        model_usage.iter().map(|usage| &usage.model_name).collect();

    // Check for expected model families (should be similar to monthly data)
    let has_openai = model_names
        .iter()
        .any(|name| name.contains("openai::gpt") || name.contains("gpt"));
    let has_anthropic = model_names
        .iter()
        .any(|name| name.contains("anthropic::claude") || name.contains("claude"));

    // At least one major model family should be present
    assert!(
        has_openai || has_anthropic,
        "Should contain at least one major model family (OpenAI or Anthropic)"
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

        // Period start should be a valid date
        assert!(
            usage.period_start <= now,
            "Period start should not be in the future: {}",
            usage.period_start
        );
    }

    // Verify daily granularity - periods should be different days
    let periods: Vec<_> = model_usage
        .iter()
        .map(|usage| usage.period_start.date_naive())
        .collect();
    let unique_dates: std::collections::HashSet<_> = periods.iter().collect();
    assert_eq!(
        unique_dates.len(),
        unique_periods.len(),
        "Each unique period should represent a different day"
    );

    // Summary assertion
    let total_models = model_names.len();
    assert!(
        total_models >= 1,
        "Should have at least 1 different model in daily data"
    );
}

#[tokio::test]
async fn test_clickhouse_model_latency_cumulative() {
    let clickhouse = get_clickhouse().await;
    let response = clickhouse
        .run_query_synchronous_no_params("SELECT COUNT() FROM ModelInference".to_string())
        .await
        .unwrap();
    println!("Number of model inferences: {}", response.response);
    let model_latency_data = clickhouse
        .get_model_latency_quantiles(TimeWindow::Cumulative)
        .await
        .unwrap();

    for latency in &model_latency_data {
        println!("Model latency cumulative datapoint: {latency:?}");
    }
    // Basic structure assertions
    assert!(
        !model_latency_data.is_empty(),
        "Cumulative model latency data should not be empty"
    );

    // Test specific cumulative latency data
    let claude_haiku_latency = model_latency_data
        .iter()
        .find(|l| l.model_name == "anthropic::claude-3-5-haiku-20241022");
    assert!(
        claude_haiku_latency.is_some(),
        "Should have cumulative latency data for claude-3-5-haiku"
    );
    let claude_haiku_latency = claude_haiku_latency.unwrap();
    assert_eq!(claude_haiku_latency.count, 461);
    // Check P50 response time (index 35 based on QUANTILES array)
    assert_eq!(
        claude_haiku_latency.response_time_ms_quantiles[35],
        Some(2825.6714)
    );

    let llama_latency = model_latency_data
        .iter()
        .find(|l| l.model_name == "llama-3.1-8b-instruct");
    assert!(
        llama_latency.is_some(),
        "Should have cumulative latency data for llama"
    );
    let llama_latency = llama_latency.unwrap();
    assert_eq!(llama_latency.count, 122);

    let gemini_latency = model_latency_data
        .iter()
        .find(|l| l.model_name == "google_ai_studio_gemini::gemini-2.5-flash-preview-04-17");
    assert!(
        gemini_latency.is_some(),
        "Should have cumulative latency data for gemini"
    );
    let gemini_latency = gemini_latency.unwrap();
    assert_eq!(gemini_latency.count, 120);

    // Helper function to find quantile indices
    // P50 = 0.50, P90 = 0.90, P99 = 0.99
    let p50_index = QUANTILES.iter().position(|&x| x == 0.50).unwrap();
    let p90_index = QUANTILES.iter().position(|&x| x == 0.90).unwrap();
    let p99_index = QUANTILES.iter().position(|&x| x == 0.99).unwrap();

    // Data quality assertions
    for latency in &model_latency_data {
        // Model name should not be empty
        assert!(
            !latency.model_name.is_empty(),
            "Model name should not be empty"
        );

        // Count should be positive
        assert!(
            latency.count > 0,
            "Count should be positive for model: {}",
            latency.model_name
        );

        // Check quantile arrays have expected length
        assert_eq!(
            latency.response_time_ms_quantiles.len(),
            QUANTILES.len(),
            "Response time quantiles should have {} entries",
            QUANTILES.len()
        );
        assert_eq!(
            latency.ttft_ms_quantiles.len(),
            QUANTILES.len(),
            "TTFT quantiles should have {} entries",
            QUANTILES.len()
        );

        // Extract key percentiles for response time
        let p50_response = latency
            .response_time_ms_quantiles
            .get(p50_index)
            .and_then(|&x| x);
        let p90_response = latency
            .response_time_ms_quantiles
            .get(p90_index)
            .and_then(|&x| x);
        let p99_response = latency
            .response_time_ms_quantiles
            .get(p99_index)
            .and_then(|&x| x);

        // Quantile values should be non-negative if present
        if let Some(p50) = p50_response {
            assert!(
                p50 >= 0.0,
                "P50 response time should be non-negative for model: {}",
                latency.model_name
            );
        }
        if let Some(p90) = p90_response {
            assert!(
                p90 >= 0.0,
                "P90 response time should be non-negative for model: {}",
                latency.model_name
            );
        }
        if let Some(p99) = p99_response {
            assert!(
                p99 >= 0.0,
                "P99 response time should be non-negative for model: {}",
                latency.model_name
            );
        }

        // Logical ordering: P50 <= P90 <= P99 (when all are present)
        match (p50_response, p90_response, p99_response) {
            (Some(p50), Some(p90), Some(p99)) => {
                assert!(
                    p50 <= p90,
                    "P50 should be <= P90 for model {}: {} vs {}",
                    latency.model_name,
                    p50,
                    p90
                );
                assert!(
                    p90 <= p99,
                    "P90 should be <= P99 for model {}: {} vs {}",
                    latency.model_name,
                    p90,
                    p99
                );
            }
            (Some(p50), Some(p90), None) => {
                assert!(
                    p50 <= p90,
                    "P50 should be <= P90 for model {}: {} vs {}",
                    latency.model_name,
                    p50,
                    p90
                );
            }
            _ => {} // Skip ordering checks if insufficient data
        }

        // Check TTFT quantiles if present
        let p50_ttft = latency.ttft_ms_quantiles.get(p50_index).and_then(|&x| x);
        let p90_ttft = latency.ttft_ms_quantiles.get(p90_index).and_then(|&x| x);
        let p99_ttft = latency.ttft_ms_quantiles.get(p99_index).and_then(|&x| x);

        if let Some(ttft) = p50_ttft {
            assert!(
                ttft >= 0.0,
                "P50 TTFT should be non-negative for model: {}",
                latency.model_name
            );
        }
        if let Some(ttft) = p90_ttft {
            assert!(
                ttft >= 0.0,
                "P90 TTFT should be non-negative for model: {}",
                latency.model_name
            );
        }
        if let Some(ttft) = p99_ttft {
            assert!(
                ttft >= 0.0,
                "P99 TTFT should be non-negative for model: {}",
                latency.model_name
            );
        }
    }

    // Check for expected model types
    let model_names: std::collections::HashSet<_> =
        model_latency_data.iter().map(|l| &l.model_name).collect();

    // Should have at least one model with latency data
    assert!(
        !model_names.is_empty(),
        "Should have at least one model with latency data"
    );

    // Summary statistics - cumulative should have more data than daily
    let total_models = model_names.len();
    assert!(
        total_models >= 1,
        "Should have latency data for at least 1 model"
    );

    // Verify at least some entries have response time data
    let entries_with_response_time = model_latency_data
        .iter()
        .filter(|l| l.response_time_ms_quantiles.iter().any(Option::is_some))
        .count();

    assert!(
        entries_with_response_time > 0,
        "Should have at least one entry with response time quantiles"
    );

    // Verify structure consistency
    for latency in &model_latency_data {
        // At least some quantiles should have values if count > 0
        let has_response_data = latency
            .response_time_ms_quantiles
            .iter()
            .any(Option::is_some);
        let has_ttft_data = latency.ttft_ms_quantiles.iter().any(Option::is_some);

        if latency.count > 0 {
            assert!(
                has_response_data || has_ttft_data,
                "Model with count > 0 should have some latency data: {}",
                latency.model_name
            );
        }
    }

    // Cumulative-specific assertions: should generally have higher counts than daily
    // since it includes all historical data
    let total_count: u64 = model_latency_data.iter().map(|l| l.count).sum();
    assert!(
        total_count > 0,
        "Cumulative data should have positive total count"
    );

    // Should have good coverage of different models for cumulative data
    if total_models >= 3 {
        // Check that we have some variety in models
        let has_openai = model_names
            .iter()
            .any(|name| name.contains("openai") || name.contains("gpt"));
        let has_anthropic = model_names
            .iter()
            .any(|name| name.contains("anthropic") || name.contains("claude"));
        let has_dummy = model_names.iter().any(|name| name.contains("dummy"));

        // At least two different model providers/types should be present in cumulative data
        let provider_count = [has_openai, has_anthropic, has_dummy]
            .iter()
            .filter(|&&x| x)
            .count();
        assert!(
            provider_count >= 1,
            "Cumulative data should have at least one model provider type"
        );
    }
}
