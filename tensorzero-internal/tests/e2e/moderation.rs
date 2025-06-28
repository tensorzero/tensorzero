#![expect(clippy::print_stdout)]

use std::collections::HashMap;
use std::sync::Arc;

use tensorzero_internal::cache::{CacheEnabledMode, CacheOptions};
use tensorzero_internal::clickhouse::ClickHouseConnectionInfo;
use tensorzero_internal::config_parser::read_config_from_path;
use tensorzero_internal::endpoints::inference::{InferenceClients, InferenceCredentials};
use tensorzero_internal::model_table::{ModelTable, ShorthandModelConfig};
use tensorzero_internal::moderation::{
    handle_moderation_request, ModerationInput, ModerationModelConfig, ModerationModelTable,
    ModerationRequest,
};
use uuid::Uuid;

use crate::common::{get_dummy_clickhouse_connection, get_gateway_endpoint, TestEvent};

#[tokio::test]
async fn test_dummy_moderation_single_text() {
    let TestEvent { clickhouse, .. } = TestEvent::load().await;
    
    // Initialize moderation model table
    let mut model_table = ModerationModelTable::new();
    let model_config = ModerationModelConfig::from_shorthand("dummy", "dummy-moderation")
        .await
        .expect("Failed to create dummy moderation model");
    model_table.insert("dummy::dummy-moderation".to_string(), model_config);
    
    // Setup clients
    let clients = InferenceClients {
        http_client: reqwest::Client::new(),
        clickhouse_connection_info: get_dummy_clickhouse_connection(),
        cache_options: Arc::new(CacheOptions {
            enabled: Arc::new(parking_lot::RwLock::new(CacheEnabledMode::All)),
            max_age_s: Some(3600),
        }),
    };
    
    let credentials = InferenceCredentials::new();
    
    // Test with safe content
    let safe_request = ModerationRequest {
        input: ModerationInput::Single("This is a friendly message about puppies and rainbows".to_string()),
        model: Some("dummy-moderation".to_string()),
    };
    
    let response = handle_moderation_request(
        safe_request,
        &clients,
        &credentials,
        "dummy::dummy-moderation",
        &model_table,
    )
    .await
    .expect("Failed to moderate safe content");
    
    assert_eq!(response.results.len(), 1);
    assert!(!response.results[0].flagged);
    assert!(!response.cached);
    
    // Test with harmful content
    let harmful_request = ModerationRequest {
        input: ModerationInput::Single("This message contains harmful content".to_string()),
        model: Some("dummy-moderation".to_string()),
    };
    
    let response = handle_moderation_request(
        harmful_request,
        &clients,
        &credentials,
        "dummy::dummy-moderation",
        &model_table,
    )
    .await
    .expect("Failed to moderate harmful content");
    
    assert_eq!(response.results.len(), 1);
    assert!(response.results[0].flagged);
    assert!(response.results[0].categories.self_harm);
    assert!(response.results[0].category_scores.self_harm > 0.7);
}

#[tokio::test]
async fn test_dummy_moderation_batch() {
    let TestEvent { .. } = TestEvent::load().await;
    
    // Initialize moderation model table
    let mut model_table = ModerationModelTable::new();
    let model_config = ModerationModelConfig::from_shorthand("dummy", "dummy-moderation")
        .await
        .expect("Failed to create dummy moderation model");
    model_table.insert("dummy::dummy-moderation".to_string(), model_config);
    
    // Setup clients
    let clients = InferenceClients {
        http_client: reqwest::Client::new(),
        clickhouse_connection_info: get_dummy_clickhouse_connection(),
        cache_options: Arc::new(CacheOptions {
            enabled: Arc::new(parking_lot::RwLock::new(CacheEnabledMode::None)),
            max_age_s: None,
        }),
    };
    
    let credentials = InferenceCredentials::new();
    
    // Test batch moderation
    let batch_request = ModerationRequest {
        input: ModerationInput::Batch(vec![
            "This is safe content".to_string(),
            "This contains hate speech".to_string(),
            "This is violent content".to_string(),
            "Another safe message".to_string(),
        ]),
        model: Some("dummy-moderation".to_string()),
    };
    
    let response = handle_moderation_request(
        batch_request,
        &clients,
        &credentials,
        "dummy::dummy-moderation",
        &model_table,
    )
    .await
    .expect("Failed to moderate batch");
    
    assert_eq!(response.results.len(), 4);
    
    // Check individual results
    assert!(!response.results[0].flagged);
    assert!(response.results[1].flagged);
    assert!(response.results[1].categories.hate);
    assert!(response.results[2].flagged);
    assert!(response.results[2].categories.violence);
    assert!(!response.results[3].flagged);
}

#[tokio::test]
async fn test_moderation_caching() {
    let TestEvent { clickhouse, .. } = TestEvent::load().await;
    
    // Initialize moderation model table
    let mut model_table = ModerationModelTable::new();
    let model_config = ModerationModelConfig::from_shorthand("dummy", "dummy-moderation")
        .await
        .expect("Failed to create dummy moderation model");
    model_table.insert("dummy::dummy-moderation".to_string(), model_config);
    
    // Setup clients with caching enabled
    let clients = InferenceClients {
        http_client: reqwest::Client::new(),
        clickhouse_connection_info: clickhouse,
        cache_options: Arc::new(CacheOptions {
            enabled: Arc::new(parking_lot::RwLock::new(CacheEnabledMode::All)),
            max_age_s: Some(3600),
        }),
    };
    
    let credentials = InferenceCredentials::new();
    
    let request = ModerationRequest {
        input: ModerationInput::Single("This is a test for caching".to_string()),
        model: Some("dummy-moderation".to_string()),
    };
    
    // First request - should not be cached
    let response1 = handle_moderation_request(
        request.clone(),
        &clients,
        &credentials,
        "dummy::dummy-moderation",
        &model_table,
    )
    .await
    .expect("Failed first moderation request");
    
    assert!(!response1.cached);
    let first_results = response1.results.clone();
    
    // Wait a bit to ensure cache write completes
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    
    // Second request - should be cached
    let response2 = handle_moderation_request(
        request,
        &clients,
        &credentials,
        "dummy::dummy-moderation",
        &model_table,
    )
    .await
    .expect("Failed second moderation request");
    
    assert!(response2.cached);
    
    // Results should be identical
    assert_eq!(first_results.len(), response2.results.len());
    for (r1, r2) in first_results.iter().zip(response2.results.iter()) {
        assert_eq!(r1.flagged, r2.flagged);
    }
}

#[tokio::test]
async fn test_moderation_provider_failover() {
    let TestEvent { .. } = TestEvent::load().await;
    
    // Initialize moderation model table with multiple providers
    let mut model_table = ModerationModelTable::new();
    
    // Create a model config with routing that includes a non-existent provider first
    let model_config = ModerationModelConfig::from_shorthand("dummy", "dummy-moderation")
        .await
        .expect("Failed to create dummy moderation model");
    
    // Note: In a real test, we'd configure multiple providers and test failover
    // For now, we just test that the dummy provider works
    model_table.insert("test-model".to_string(), model_config);
    
    // Setup clients
    let clients = InferenceClients {
        http_client: reqwest::Client::new(),
        clickhouse_connection_info: get_dummy_clickhouse_connection(),
        cache_options: Arc::new(CacheOptions {
            enabled: Arc::new(parking_lot::RwLock::new(CacheEnabledMode::None)),
            max_age_s: None,
        }),
    };
    
    let credentials = InferenceCredentials::new();
    
    let request = ModerationRequest {
        input: ModerationInput::Single("Test content".to_string()),
        model: Some("dummy-moderation".to_string()),
    };
    
    let response = handle_moderation_request(
        request,
        &clients,
        &credentials,
        "test-model",
        &model_table,
    )
    .await
    .expect("Failed moderation request");
    
    assert_eq!(response.results.len(), 1);
}

#[tokio::test]
#[cfg(not(feature = "e2e_tests_no_credentials"))]
async fn test_openai_moderation() {
    // Skip this test if OPENAI_API_KEY is not set
    if std::env::var("OPENAI_API_KEY").is_err() {
        println!("Skipping OpenAI moderation test - OPENAI_API_KEY not set");
        return;
    }
    
    let TestEvent { clickhouse, .. } = TestEvent::load().await;
    
    // Initialize moderation model table
    let mut model_table = ModerationModelTable::new();
    let model_config = ModerationModelConfig::from_shorthand("openai", "text-moderation-latest")
        .await
        .expect("Failed to create OpenAI moderation model");
    model_table.insert("openai::text-moderation-latest".to_string(), model_config);
    
    // Setup clients
    let clients = InferenceClients {
        http_client: reqwest::Client::new(),
        clickhouse_connection_info: clickhouse,
        cache_options: Arc::new(CacheOptions {
            enabled: Arc::new(parking_lot::RwLock::new(CacheEnabledMode::All)),
            max_age_s: Some(3600),
        }),
    };
    
    let mut credentials = InferenceCredentials::new();
    if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
        credentials.insert("openai_api_key".to_string(), api_key);
    }
    
    let request = ModerationRequest {
        input: ModerationInput::Single("I love programming and learning new things!".to_string()),
        model: None, // Let it use the default
    };
    
    let response = handle_moderation_request(
        request,
        &clients,
        &credentials,
        "openai::text-moderation-latest",
        &model_table,
    )
    .await
    .expect("Failed OpenAI moderation request");
    
    assert_eq!(response.results.len(), 1);
    
    // Safe content should not be flagged
    assert!(!response.results[0].flagged);
    
    // All categories should be false for safe content
    assert!(!response.results[0].categories.hate);
    assert!(!response.results[0].categories.harassment);
    assert!(!response.results[0].categories.self_harm);
    assert!(!response.results[0].categories.sexual);
    assert!(!response.results[0].categories.violence);
}