#![allow(clippy::print_stdout)]
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tensorzero::{
    ClientInferenceParams, ClientInput, ClientInputMessage, ClientInputMessageContent,
    InferenceOutput, Role, TensorZeroError,
};
use tensorzero_core::endpoints::inference::{ChatCompletionInferenceParams, InferenceParams};
use tensorzero_core::inference::types::TextKind;
use uuid::Uuid;

// ===== HELPER FUNCTIONS =====

fn generate_rate_limit_config(rules: &[&str]) -> String {
    let rules_toml = rules.join("\n\n");
    format!(
        r#"
[rate_limiting]
enabled = true

{rules_toml}

[models]

[models."dummy"]
routing = ["dummy"]

[models."dummy".providers.dummy]
type = "dummy"
model_name = "dummy"

[functions]

[functions.basic_test]
type = "chat"

[functions.basic_test.variants.default]
type = "chat_completion"
model = "dummy"
"#,
    )
}

async fn make_request_with_tags(
    client: &tensorzero::Client,
    tags: HashMap<String, String>,
) -> Result<InferenceOutput, TensorZeroError> {
    client
        .inference(ClientInferenceParams {
            function_name: Some("basic_test".to_string()),
            input: ClientInput {
                system: None,
                messages: vec![ClientInputMessage {
                    role: Role::User,
                    content: vec![ClientInputMessageContent::Text(TextKind::Text {
                        text: "Hello".to_string(),
                    })],
                }],
            },
            params: InferenceParams {
                chat_completion: ChatCompletionInferenceParams {
                    max_tokens: Some(100),
                    ..Default::default()
                },
            },
            tags,
            ..Default::default()
        })
        .await
}

fn assert_rate_limit_exceeded(result: Result<InferenceOutput, TensorZeroError>) {
    match result {
        Err(TensorZeroError::Http {
            status_code, text, ..
        }) => {
            // Rate limit errors can return various status codes (429, 502, etc.)
            if let Some(text) = text {
                if text.contains("rate limit exceeded")
                    || text.contains("Rate limit exceeded")
                    || text.contains("RateLimitExceeded")
                {
                    // Expected - this is a rate limit error
                } else {
                    panic!(
                        "Expected rate limit exceeded error, got status {status_code}: {text:?}",
                    );
                }
            } else {
                panic!("Expected rate limit exceeded error, got status {status_code} with no text",);
            }
        }
        Err(TensorZeroError::Other { source, .. }) => {
            let error_str = source.to_string();
            if error_str.contains("rate limit exceeded") || error_str.contains("RateLimitExceeded")
            {
                // Expected
            } else {
                panic!("Expected rate limit exceeded error, got: {source:?}");
            }
        }
        Err(e) => {
            let error_str = e.to_string();
            if error_str.contains("rate limit exceeded") || error_str.contains("RateLimitExceeded")
            {
                // Expected
            } else {
                panic!("Expected rate limit exceeded error, got: {e:?}");
            }
        }
        Ok(output) => panic!("Expected rate limit exceeded, but request succeeded: {output:?}",),
    }
}

// ===== BASIC TAG-BASED TESTS =====

#[tokio::test(flavor = "multi_thread")]
async fn test_rate_limiting_concrete_tag_value() {
    let id = Uuid::now_v7();
    let config = generate_rate_limit_config(&[&format!(
        r#"[[rate_limiting.rules]]
model_inferences_per_minute = {{ capacity = 5, refill_rate = 3 }}
always = true
scope = [
    {{ tag_key = "test1_customer_id_{id}", tag_value = "customer_alpha" }}
]"#
    )]);

    let client =
        tensorzero::test_helpers::make_embedded_gateway_with_config_and_postgres(&config).await;
    let tags_match = HashMap::from([(
        format!("test1_customer_id_{id}"),
        "customer_alpha".to_string(),
    )]);
    let tags_no_match = HashMap::from([(
        format!("test1_customer_id_{id}"),
        "customer_beta".to_string(),
    )]);

    // First request should succeed
    let result1 = make_request_with_tags(&client, tags_match.clone()).await;
    if let Err(ref e) = result1 {
        println!("First request failed: {e:?}");
    }
    assert!(result1.is_ok(), "First request should succeed: {result1:?}",);

    // Second request should succeed
    let result2 = make_request_with_tags(&client, tags_match.clone()).await;
    assert!(result2.is_ok(), "Second request should succeed");

    // Third request should succeed
    let result3 = make_request_with_tags(&client, tags_match.clone()).await;
    assert!(result3.is_ok(), "Third request should succeed");

    // Fourth request should fail (exceeded capacity of 3, but we've only made 3 requests)
    // Let's make 2 more to reach the limit of 5
    let result4 = make_request_with_tags(&client, tags_match.clone()).await;
    assert!(result4.is_ok(), "Fourth request should succeed");

    let result5 = make_request_with_tags(&client, tags_match.clone()).await;
    assert!(result5.is_ok(), "Fifth request should succeed");

    // Sixth request should fail (exceeded capacity of 5)
    let result6 = make_request_with_tags(&client, tags_match.clone()).await;
    assert_rate_limit_exceeded(result6);

    // Request with different customer_id should succeed (not affected by limit)
    let result7 = make_request_with_tags(&client, tags_no_match).await;
    assert!(
        result7.is_ok(),
        "Request with different customer_id should succeed"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_rate_limiting_total_tag_value() {
    let id = Uuid::now_v7();
    let config = generate_rate_limit_config(&[&format!(
        r#"[[rate_limiting.rules]]
model_inferences_per_minute = {{ capacity = 3, refill_rate = 3 }}
always = true
scope = [
    {{ tag_key = "test2_tenant_id_{id}", tag_value = "tensorzero::total" }}
]"#
    )]);

    let client =
        tensorzero::test_helpers::make_embedded_gateway_with_config_and_postgres(&config).await;
    let tags1 = HashMap::from([(format!("test2_tenant_id_{id}"), "tenant_gamma".to_string())]);
    let tags2 = HashMap::from([(format!("test2_tenant_id_{id}"), "tenant_delta".to_string())]);
    let no_tags = HashMap::new();

    // First three requests should succeed (aggregate limit of 3)
    let result1 = make_request_with_tags(&client, tags1.clone()).await;
    assert!(result1.is_ok());

    let result2 = make_request_with_tags(&client, tags2.clone()).await;
    assert!(result2.is_ok());

    let result3 = make_request_with_tags(&client, tags1.clone()).await;
    assert!(result3.is_ok());

    // Fourth request should fail
    let result4 = make_request_with_tags(&client, tags2.clone()).await;
    assert_rate_limit_exceeded(result4);

    // Request without the tag should succeed (not subject to limit)
    let result5 = make_request_with_tags(&client, no_tags).await;
    assert!(result5.is_ok());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_rate_limiting_each_tag_value() {
    let id = Uuid::now_v7();
    let config = generate_rate_limit_config(&[&format!(
        r#"[[rate_limiting.rules]]
model_inferences_per_minute = {{ capacity = 2, refill_rate = 2 }}
always = true
scope = [
    {{ tag_key = "test3_workspace_id_{id}", tag_value = "tensorzero::each" }}
]"#
    )]);

    let client =
        tensorzero::test_helpers::make_embedded_gateway_with_config_and_postgres(&config).await;
    let tags_workspace1 = HashMap::from([(
        format!("test3_workspace_id_{id}"),
        "workspace_epsilon".to_string(),
    )]);
    let tags_workspace2 = HashMap::from([(
        format!("test3_workspace_id_{id}"),
        "workspace_zeta".to_string(),
    )]);

    // Each workspace gets their own separate limit of 2
    let result1a = make_request_with_tags(&client, tags_workspace1.clone()).await;
    assert!(result1a.is_ok());

    let result1b = make_request_with_tags(&client, tags_workspace1.clone()).await;
    assert!(result1b.is_ok());

    let result2a = make_request_with_tags(&client, tags_workspace2.clone()).await;
    assert!(result2a.is_ok());

    let result2b = make_request_with_tags(&client, tags_workspace2.clone()).await;
    assert!(result2b.is_ok());

    // Third request for workspace1 should fail
    let result1c = make_request_with_tags(&client, tags_workspace1.clone()).await;
    assert_rate_limit_exceeded(result1c);

    // Third request for workspace2 should fail
    let result2c = make_request_with_tags(&client, tags_workspace2.clone()).await;
    assert_rate_limit_exceeded(result2c);
}

// ===== PRIORITY TESTS =====

#[tokio::test(flavor = "multi_thread")]
async fn test_rate_limiting_priority_always() {
    let id = Uuid::now_v7();
    let config = generate_rate_limit_config(&[
        &format!(
            r#"[[rate_limiting.rules]]
model_inferences_per_minute = {{ capacity = 3, refill_rate = 2 }}
priority = 1
scope = [
    {{ tag_key = "test4_account_id_{id}", tag_value = "account_eta" }}
]"#
        ),
        &format!(
            r#"[[rate_limiting.rules]]
model_inferences_per_minute = {{ capacity = 8, refill_rate = 5 }}
always = true
scope = [
    {{ tag_key = "test4_service_id_{id}", tag_value = "service_theta" }}
]"#
        ),
    ]);

    let client =
        tensorzero::test_helpers::make_embedded_gateway_with_config_and_postgres(&config).await;
    // Tags that match BOTH rules - should apply both rules (always + priority)
    let tags_both = HashMap::from([
        (format!("test4_account_id_{id}"), "account_eta".to_string()),
        (
            format!("test4_service_id_{id}"),
            "service_theta".to_string(),
        ),
    ]);

    // Tags that match only the always rule
    let tags_always_only = HashMap::from([(
        format!("test4_service_id_{id}"),
        "service_theta".to_string(),
    )]);

    // Both rules apply for tags_both - most restrictive limit wins (capacity=3)
    let result1 = make_request_with_tags(&client, tags_both.clone()).await;
    assert!(
        result1.is_ok(),
        "First request with both rules should succeed"
    );

    let result2 = make_request_with_tags(&client, tags_both.clone()).await;
    assert!(
        result2.is_ok(),
        "Second request with both rules should succeed"
    );

    let result3 = make_request_with_tags(&client, tags_both.clone()).await;
    assert!(
        result3.is_ok(),
        "Third request with both rules should succeed"
    );

    // Fourth request should fail due to priority=1 rule limit of 3
    let result4 = make_request_with_tags(&client, tags_both.clone()).await;
    assert_rate_limit_exceeded(result4);

    // Request with only always rule should succeed (higher capacity=8)
    let result5 = make_request_with_tags(&client, tags_always_only).await;
    assert!(
        result5.is_ok(),
        "Request with only always rule should succeed"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_rate_limiting_priority_numeric() {
    // Test with different scopes that can overlap but aren't identical
    let id = Uuid::now_v7();
    let config = generate_rate_limit_config(&[
        &format!(
            r#"[[rate_limiting.rules]]
model_inferences_per_minute = {{ capacity = 2, refill_rate = 2 }}
priority = 1
scope = [
    {{ tag_key = "test5_team_id_{id}", tag_value = "team_iota" }}
]"#
        ),
        &format!(
            r#"[[rate_limiting.rules]]
model_inferences_per_minute = {{ capacity = 5, refill_rate = 5 }}
priority = 3
scope = [
    {{ tag_key = "test5_project_id_{id}", tag_value = "project_kappa" }}
]"#
        ),
        &format!(
            r#"[[rate_limiting.rules]]
model_inferences_per_minute = {{ capacity = 1, refill_rate = 1 }}
priority = 2
scope = [
    {{ tag_key = "test5_environment_id_{id}", tag_value = "env_lambda" }}
]"#
        ),
    ]);

    let client =
        tensorzero::test_helpers::make_embedded_gateway_with_config_and_postgres(&config).await;

    // Test 1: Request that matches only the highest priority rule (priority=3)
    let tags_highest_only = HashMap::from([(
        format!("test5_project_id_{id}"),
        "project_kappa".to_string(),
    )]);

    for i in 0..5 {
        let result = make_request_with_tags(&client, tags_highest_only.clone()).await;
        assert!(
            result.is_ok(),
            "Request {} with highest priority only should succeed",
            i + 1
        );
    }

    // Sixth request should fail (exceeded capacity=5)
    let result = make_request_with_tags(&client, tags_highest_only.clone()).await;
    assert_rate_limit_exceeded(result);

    // Test 2: Request that matches multiple rules - only highest priority should apply
    let tags_multiple = HashMap::from([
        (format!("test5_team_id_{id}"), "team_iota".to_string()), // matches priority=1, capacity=2
        (
            format!("test5_project_id_{id}"),
            "project_kappa".to_string(),
        ), // matches priority=3, capacity=5
        (
            format!("test5_environment_id_{id}"),
            "env_lambda".to_string(),
        ), // matches priority=2, capacity=1
    ]);

    // Should be limited by highest priority rule (priority=3, capacity=5)
    // But we already used 5, so this should fail immediately
    let result = make_request_with_tags(&client, tags_multiple).await;
    assert_rate_limit_exceeded(result);
}

// ===== RESOURCE TYPE TESTS =====
#[tokio::test(flavor = "multi_thread")]
async fn test_rate_limiting_tokens() {
    let id = Uuid::now_v7();
    let config = generate_rate_limit_config(&[&format!(
        r#"[[rate_limiting.rules]]
tokens_per_minute = {{ capacity = 500, refill_rate = 150 }}
always = true
scope = [
    {{ tag_key = "test_tokens_user_id_{id}", tag_value = "123" }}
]"#
    )]);

    let client =
        tensorzero::test_helpers::make_embedded_gateway_with_config_and_postgres(&config).await;
    let tags = HashMap::from([(format!("test_tokens_user_id_{id}"), "123".to_string())]);

    // Make several requests - should succeed until token limit is reached
    // Each request borrows roughly 100 tokens
    for i in 0..38 {
        match make_request_with_tags(&client, tags.clone()).await {
            Ok(_) => (),
            Err(_) => {
                // Should hit limit after 37 requests due to token consumption
                // (11 * 37 = 407, the borrowed amount is 102, 407 + 102 = 509 > 500)
                assert!(
                    i == 37,
                    "Should hit exactly 37  requests before hitting token limit (got {i})"
                );
                break;
            }
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_rate_limiting_multiple_resources() {
    let id = Uuid::now_v7();
    let config = generate_rate_limit_config(&[&format!(
        r#"[[rate_limiting.rules]]
tokens_per_minute = {{ capacity = 10000, refill_rate = 100 }}
model_inferences_per_minute = {{ capacity = 3, refill_rate = 3 }}
always = true
scope = [
    {{ tag_key = "test_multi_user_id_{id}", tag_value = "123" }}
]"#
    )]);

    let client =
        tensorzero::test_helpers::make_embedded_gateway_with_config_and_postgres(&config).await;
    let tags = HashMap::from([(format!("test_multi_user_id_{id}"), "123".to_string())]);

    // Should be limited by model_inferences_per_minute (3) rather than tokens (100)
    let result1 = make_request_with_tags(&client, tags.clone()).await;
    assert!(result1.is_ok());

    let result2 = make_request_with_tags(&client, tags.clone()).await;
    assert!(result2.is_ok());

    let result3 = make_request_with_tags(&client, tags.clone()).await;
    assert!(result3.is_ok());

    // Fourth request should fail due to model inference limit
    let result4 = make_request_with_tags(&client, tags.clone()).await;
    assert_rate_limit_exceeded(result4);
}

// ===== CONCURRENT REQUEST TESTS =====

#[tokio::test(flavor = "multi_thread")]
async fn test_rate_limiting_concurrent_requests() {
    let id = Uuid::now_v7();
    let config = generate_rate_limit_config(&[&format!(
        r#"[[rate_limiting.rules]]
model_inferences_per_minute = {{ capacity = 5, refill_rate = 5 }}
always = true
scope = [
    {{ tag_key = "user_id_{id}", tag_value = "123" }}
]"#
    )]);

    let client = Arc::new(
        tensorzero::test_helpers::make_embedded_gateway_with_config_and_postgres(&config).await,
    );
    let tags = HashMap::from([(format!("user_id_{id}"), "123".to_string())]);

    // Launch 10 concurrent requests
    let handles: Vec<_> = (0..10)
        .map(|_| {
            let client_clone = client.clone();
            let tags_clone = tags.clone();
            tokio::spawn(async move { make_request_with_tags(&client_clone, tags_clone).await })
        })
        .collect();

    // Wait for all requests to complete
    let results: Vec<_> = futures::future::join_all(handles)
        .await
        .into_iter()
        .map(|handle_result| handle_result.unwrap())
        .collect();

    // Count successes and failures
    let success_count = results.iter().filter(|r| r.is_ok()).count();
    let failure_count = results.iter().filter(|r| r.is_err()).count();

    // Should have exactly 5 successes and 5 failures due to atomic rate limiting
    assert_eq!(
        success_count, 5,
        "Should have exactly 5 successful requests"
    );
    assert_eq!(failure_count, 5, "Should have exactly 5 failed requests");

    // All failures should be rate limit exceeded errors
    for result in results.iter().filter(|r| r.is_err()) {
        if let Err(e) = result {
            match e {
                TensorZeroError::Http {
                    status_code, text, ..
                } => {
                    // Rate limit errors can be 429, 502, or other status codes
                    if let Some(text) = text {
                        assert!(
                            text.contains("rate limit exceeded")
                                || text.contains("Rate limit exceeded"),
                            "HTTP failures should be rate limit errors, got status {status_code}: {text:?}",
                        );
                    }
                }
                _ => {
                    let error_str = e.to_string();
                    assert!(
                        error_str.contains("rate limit exceeded")
                            || error_str.contains("RateLimitExceeded"),
                        "All failures should be rate limit exceeded errors, got: {e:?}",
                    );
                }
            }
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_rate_limiting_concurrent_different_tags() {
    let id = Uuid::now_v7();
    let config = generate_rate_limit_config(&[&format!(
        r#"[[rate_limiting.rules]]
model_inferences_per_minute = {{ capacity = 2, refill_rate = 2 }}
always = true
scope = [
    {{ tag_key = "user_id_{id}", tag_value = "tensorzero::each" }}
]"#
    )]);

    let client = Arc::new(
        tensorzero::test_helpers::make_embedded_gateway_with_config_and_postgres(&config).await,
    );

    // Launch concurrent requests for different users
    let handles: Vec<_> = (0..6)
        .map(|i| {
            let client_clone = client.clone();
            let tags = HashMap::from([(format!("user_id_{id}"), format!("user{}", i % 3))]);
            tokio::spawn(async move { (i % 3, make_request_with_tags(&client_clone, tags).await) })
        })
        .collect();

    // Wait for all requests to complete
    let results: Vec<_> = futures::future::join_all(handles)
        .await
        .into_iter()
        .map(|handle_result| handle_result.unwrap())
        .collect();

    // Group results by user
    let mut user_results = HashMap::new();
    for (user_id, result) in results {
        user_results
            .entry(user_id)
            .or_insert_with(Vec::new)
            .push(result);
    }

    // Each user should have exactly 2 requests (some succeed, some fail based on timing)
    for (user_id, user_results) in user_results {
        assert_eq!(user_results.len(), 2, "Each user should have 2 requests");

        // At least one should succeed, but due to concurrency both might succeed or one might fail
        let success_count = user_results.iter().filter(|r| r.is_ok()).count();
        assert!(
            success_count >= 1,
            "User {user_id} should have at least 1 success",
        );
    }
}

// ===== COMPLEX SCOPE TESTS =====

#[tokio::test(flavor = "multi_thread")]
async fn test_rate_limiting_multiple_scopes() {
    let id = Uuid::now_v7();
    let config = generate_rate_limit_config(&[&format!(
        r#"[[rate_limiting.rules]]
model_inferences_per_minute = {{ capacity = 3, refill_rate = 3 }}
always = true
scope = [
    {{ tag_key = "test6_application_id_{id}", tag_value = "app1" }},
    {{ tag_key = "test6_user_id_{id}", tag_value = "tensorzero::total" }}
]"#
    )]);

    let client =
        tensorzero::test_helpers::make_embedded_gateway_with_config_and_postgres(&config).await;

    let tags_match = HashMap::from([
        (format!("test6_application_id_{id}"), "app1".to_string()),
        (format!("test6_user_id_{id}"), "user123".to_string()),
    ]);

    let tags_partial = HashMap::from([
        (format!("test6_application_id_{id}"), "app2".to_string()),
        (format!("test6_user_id_{id}"), "user123".to_string()),
    ]);

    let tags_missing = HashMap::from([(format!("test6_application_id_{id}"), "app1".to_string())]);

    // Requests that match both scope requirements should be limited
    for _ in 0..3 {
        let result = make_request_with_tags(&client, tags_match.clone()).await;
        assert!(
            result.is_ok(),
            "Matching requests within limit should succeed"
        );
    }

    let result = make_request_with_tags(&client, tags_match.clone()).await;
    assert_rate_limit_exceeded(result);

    // Requests that don't match all scope requirements should not be limited
    let result = make_request_with_tags(&client, tags_partial.clone()).await;
    assert!(result.is_ok(), "Partial match should not be rate limited");

    let result = make_request_with_tags(&client, tags_missing.clone()).await;
    assert!(result.is_ok(), "Missing tag should not be rate limited");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_rate_limiting_missing_tags() {
    let id = Uuid::now_v7();
    let config = generate_rate_limit_config(&[&format!(
        r#"[[rate_limiting.rules]]
model_inferences_per_minute = {{ capacity = 1, refill_rate = 1 }}
always = true
scope = [
    {{ tag_key = "test7_required_tag_{id}", tag_value = "value" }}
]"#
    )]);

    let client =
        tensorzero::test_helpers::make_embedded_gateway_with_config_and_postgres(&config).await;

    let tags_with_required =
        HashMap::from([(format!("test7_required_tag_{id}"), "value".to_string())]);
    let tags_without_required =
        HashMap::from([(format!("test7_other_tag_{id}"), "value".to_string())]);
    let no_tags = HashMap::new();

    // Request with required tag should be limited
    let result1 = make_request_with_tags(&client, tags_with_required.clone()).await;
    assert!(result1.is_ok());

    let result2 = make_request_with_tags(&client, tags_with_required.clone()).await;
    assert_rate_limit_exceeded(result2);

    // Requests without the required tag should not be limited
    for _ in 0..5 {
        let result = make_request_with_tags(&client, tags_without_required.clone()).await;
        assert!(
            result.is_ok(),
            "Request without required tag should not be limited"
        );

        let result = make_request_with_tags(&client, no_tags.clone()).await;
        assert!(result.is_ok(), "Request with no tags should not be limited");
    }
}

// ===== EDGE CASES AND ERROR HANDLING =====

#[tokio::test(flavor = "multi_thread")]
async fn test_rate_limiting_zero_limit() {
    let id = Uuid::now_v7();
    let config = generate_rate_limit_config(&[&format!(
        r#"[[rate_limiting.rules]]
model_inferences_per_minute = 0
always = true
scope = [
    {{ tag_key = "test8_user_id_{id}", tag_value = "123" }}
]"#
    )]);

    let client =
        tensorzero::test_helpers::make_embedded_gateway_with_config_and_postgres(&config).await;
    let tags = HashMap::from([(format!("test8_user_id_{id}"), "123".to_string())]);

    // First request should fail immediately due to zero limit
    let result = make_request_with_tags(&client, tags.clone()).await;
    assert_rate_limit_exceeded(result);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_rate_limiting_disabled() {
    let id = Uuid::now_v7();
    let config = &format!(
        r#"
[rate_limiting]
enabled = false

[[rate_limiting.rules]]
model_inferences_per_minute = 1
always = true
scope = [
    {{ tag_key = "test9_user_id_{id}", tag_value = "123" }}
]

[models]

[models."dummy"]
routing = ["dummy"]

[models."dummy".providers.dummy]
type = "dummy"
model_name = "dummy"

[functions]

[functions.basic_test]
type = "chat"

[functions.basic_test.variants.default]
type = "chat_completion"
model = "dummy"
"#
    );

    let client =
        tensorzero::test_helpers::make_embedded_gateway_with_config_and_postgres(config).await;
    let tags = HashMap::from([(format!("test9_user_id_{id}"), "123".to_string())]);

    // All requests should succeed when rate limiting is disabled
    for _ in 0..10 {
        let result = make_request_with_tags(&client, tags.clone()).await;
        assert!(
            result.is_ok(),
            "All requests should succeed when rate limiting is disabled"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_rate_limiting_bucket_configuration() {
    let id = Uuid::now_v7();
    let config = generate_rate_limit_config(&[&format!(
        r#"[[rate_limiting.rules]]
model_inferences_per_minute = {{ capacity = 10, refill_rate = 2 }}
always = true
scope = [
    {{ tag_key = "test10_user_id_{id}", tag_value = "123" }}
]"#
    )]);

    let client =
        tensorzero::test_helpers::make_embedded_gateway_with_config_and_postgres(&config).await;
    let tags = HashMap::from([(format!("test10_user_id_{id}"), "123".to_string())]);

    // Should be able to make requests up to the capacity (10)
    let mut success_count = 0;
    for _ in 0..15 {
        match make_request_with_tags(&client, tags.clone()).await {
            Ok(_) => success_count += 1,
            Err(_) => break,
        }
    }

    // Should have succeeded for some requests but hit the capacity limit
    assert!(success_count > 0, "Should have some successful requests");
    assert!(success_count <= 10, "Should not exceed capacity of 10");

    // Next request should fail
    let result = make_request_with_tags(&client, tags.clone()).await;
    assert_rate_limit_exceeded(result);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_rate_limiting_large_limit() {
    let id = Uuid::now_v7();
    let config = generate_rate_limit_config(&[&format!(
        r#"[[rate_limiting.rules]]
model_inferences_per_minute = {{ capacity = 1000000, refill_rate = 1000000 }}
always = true
scope = [
    {{ tag_key = "test11_user_id_{id}", tag_value = "123" }}
]"#
    )]);

    let client =
        tensorzero::test_helpers::make_embedded_gateway_with_config_and_postgres(&config).await;
    let tags = HashMap::from([(format!("test11_user_id_{id}"), "123".to_string())]);

    // Should be able to make many requests without hitting the limit
    for _ in 0..100 {
        let result = make_request_with_tags(&client, tags.clone()).await;
        assert!(result.is_ok(), "Should succeed with very large limit");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_rate_limiting_time_intervals() {
    // Test different time intervals work correctly
    let id = Uuid::now_v7();
    let config = generate_rate_limit_config(&[&format!(
        r#"[[rate_limiting.rules]]
model_inferences_per_second = {{ capacity = 2, refill_rate = 2 }}
always = true
scope = [
    {{ tag_key = "test12_user_id_{id}", tag_value = "123" }}
]"#
    )]);

    let client =
        tensorzero::test_helpers::make_embedded_gateway_with_config_and_postgres(&config).await;
    let tags = HashMap::from([(format!("test12_user_id_{id}"), "123".to_string())]);

    // Should be able to make 2 requests per second
    let result1 = make_request_with_tags(&client, tags.clone()).await;
    assert!(result1.is_ok());

    let result2 = make_request_with_tags(&client, tags.clone()).await;
    assert!(result2.is_ok());

    // Third request should fail
    let result3 = make_request_with_tags(&client, tags.clone()).await;
    assert_rate_limit_exceeded(result3);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_rate_limiting_no_rules() {
    let config = r#"
[rate_limiting]
enabled = true

[models]

[models."dummy"]
routing = ["dummy"]

[models."dummy".providers.dummy]
type = "dummy"
model_name = "dummy"

[functions]

[functions.basic_test]
type = "chat"

[functions.basic_test.variants.default]
type = "chat_completion"
model = "dummy"
"#;

    let client =
        tensorzero::test_helpers::make_embedded_gateway_with_config_and_postgres(config).await;
    let id = Uuid::now_v7();
    let tags = HashMap::from([(format!("test13_user_id_{id}"), "123".to_string())]);

    // Should succeed when no rules are defined
    for _ in 0..10 {
        let result = make_request_with_tags(&client, tags.clone()).await;
        assert!(
            result.is_ok(),
            "Should succeed when no rate limiting rules are defined"
        );
    }
}
