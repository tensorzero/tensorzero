use std::collections::HashMap;
use tensorzero_internal::cache::{CacheEnabledMode, CacheOptions};
use tensorzero_internal::clickhouse::test_helpers::get_clickhouse;
use tensorzero_internal::endpoints::inference::{InferenceClients, InferenceCredentials};
use tensorzero_internal::model_table::ShorthandModelConfig;
use tensorzero_internal::moderation::{
    handle_moderation_request, ModerationInput, ModerationModelConfig, ModerationModelTable,
    ModerationRequest,
};

#[tokio::test]
async fn test_dummy_moderation_single_text() {
    let clickhouse = get_clickhouse().await;

    // Initialize moderation model table
    let model_config = ModerationModelConfig::from_shorthand("dummy", "dummy-moderation")
        .await
        .expect("Failed to create dummy moderation model");
    let mut model_table_map = HashMap::new();
    model_table_map.insert("dummy::dummy-moderation".to_string().into(), model_config);
    let model_table = ModerationModelTable::try_from(model_table_map).unwrap();

    // Setup clients
    let http_client = reqwest::Client::new();
    let cache_options = CacheOptions {
        enabled: CacheEnabledMode::On,
        max_age_s: Some(3600),
    };
    let credentials = InferenceCredentials::new();
    let clients = InferenceClients {
        http_client: &http_client,
        clickhouse_connection_info: &clickhouse,
        credentials: &credentials,
        cache_options: &cache_options,
    };

    // Test with safe content
    let safe_request = ModerationRequest {
        input: ModerationInput::Single(
            "This is a friendly message about puppies and rainbows".to_string(),
        ),
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

    // Test with harmful content
    let harmful_request = ModerationRequest {
        input: ModerationInput::Single("This content contains harmful language".to_string()),
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
    assert!(response.results[0].categories.harassment);
}

#[tokio::test]
async fn test_dummy_moderation_batch() {
    let clickhouse = get_clickhouse().await;

    // Initialize moderation model table
    let model_config = ModerationModelConfig::from_shorthand("dummy", "dummy-moderation")
        .await
        .expect("Failed to create dummy moderation model");
    let mut model_table_map = HashMap::new();
    model_table_map.insert("dummy::dummy-moderation".to_string().into(), model_config);
    let model_table = ModerationModelTable::try_from(model_table_map).unwrap();

    // Setup clients
    let http_client = reqwest::Client::new();
    let cache_options = CacheOptions {
        enabled: CacheEnabledMode::Off,
        max_age_s: None,
    };
    let credentials = InferenceCredentials::new();
    let clients = InferenceClients {
        http_client: &http_client,
        clickhouse_connection_info: &clickhouse,
        credentials: &credentials,
        cache_options: &cache_options,
    };

    // Test batch moderation
    let batch_request = ModerationRequest {
        input: ModerationInput::Batch(vec![
            "Safe content".to_string(),
            "Content with violent themes".to_string(),
            "Another safe message".to_string(),
            "Content with hate speech".to_string(),
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
    .expect("Failed to moderate batch content");

    assert_eq!(response.results.len(), 4);
    assert!(!response.results[0].flagged);
    assert!(response.results[1].flagged);
    assert!(response.results[1].categories.violence);
    assert!(!response.results[2].flagged);
    assert!(response.results[3].flagged);
    assert!(response.results[3].categories.hate);
}

#[tokio::test]
async fn test_moderation_caching() {
    let clickhouse = get_clickhouse().await;

    // Initialize moderation model table
    let model_config = ModerationModelConfig::from_shorthand("dummy", "dummy-moderation")
        .await
        .expect("Failed to create dummy moderation model");
    let mut model_table_map = HashMap::new();
    model_table_map.insert("dummy::dummy-moderation".to_string().into(), model_config);
    let model_table = ModerationModelTable::try_from(model_table_map).unwrap();

    // Setup clients with caching enabled
    let http_client = reqwest::Client::new();
    let cache_options = CacheOptions {
        enabled: CacheEnabledMode::On,
        max_age_s: Some(3600),
    };
    let credentials = InferenceCredentials::new();
    let clients = InferenceClients {
        http_client: &http_client,
        clickhouse_connection_info: &clickhouse,
        credentials: &credentials,
        cache_options: &cache_options,
    };

    // Make the same request twice
    let request = ModerationRequest {
        input: ModerationInput::Single("Test content for caching".to_string()),
        model: Some("dummy-moderation".to_string()),
    };

    let response1 = handle_moderation_request(
        request.clone(),
        &clients,
        &credentials,
        "dummy::dummy-moderation",
        &model_table,
    )
    .await
    .expect("Failed first moderation request");

    let response2 = handle_moderation_request(
        request,
        &clients,
        &credentials,
        "dummy::dummy-moderation",
        &model_table,
    )
    .await
    .expect("Failed second moderation request");

    // Results should be identical
    assert_eq!(response1.results, response2.results);
}

#[tokio::test]
async fn test_moderation_fallback() {
    let clickhouse = get_clickhouse().await;

    // Initialize moderation model table with multiple providers
    let primary_config = ModerationModelConfig::from_shorthand("dummy", "dummy-moderation")
        .await
        .expect("Failed to create primary model");
    let fallback_config = ModerationModelConfig::from_shorthand("dummy", "dummy-moderation-2")
        .await
        .expect("Failed to create fallback model");

    let mut model_table_map = HashMap::new();
    model_table_map.insert("primary::moderation".to_string().into(), primary_config);
    model_table_map.insert("fallback::moderation".to_string().into(), fallback_config);
    let model_table = ModerationModelTable::try_from(model_table_map).unwrap();

    // Setup clients
    let http_client = reqwest::Client::new();
    let cache_options = CacheOptions {
        enabled: CacheEnabledMode::Off,
        max_age_s: None,
    };
    let credentials = InferenceCredentials::new();
    let clients = InferenceClients {
        http_client: &http_client,
        clickhouse_connection_info: &clickhouse,
        credentials: &credentials,
        cache_options: &cache_options,
    };

    // Test fallback behavior
    let request = ModerationRequest {
        input: ModerationInput::Single("Test content".to_string()),
        model: Some("primary::moderation".to_string()),
    };

    let response = handle_moderation_request(
        request,
        &clients,
        &credentials,
        "primary::moderation",
        &model_table,
    )
    .await
    .expect("Failed moderation request");

    assert_eq!(response.results.len(), 1);
}

#[tokio::test]
#[ignore] // Run with --ignored to test with real OpenAI API
async fn test_openai_moderation() {
    let clickhouse = get_clickhouse().await;

    // Skip if no OpenAI API key
    let api_key = match std::env::var("OPENAI_API_KEY") {
        Ok(key) => key,
        Err(_) => {
            return;
        }
    };

    // Initialize OpenAI moderation model
    let model_config = ModerationModelConfig::from_shorthand("openai", "omni-moderation-latest")
        .await
        .expect("Failed to create OpenAI moderation model");
    let mut model_table_map = HashMap::new();
    model_table_map.insert(
        "openai::omni-moderation-latest".to_string().into(),
        model_config,
    );
    let model_table = ModerationModelTable::try_from(model_table_map).unwrap();

    // Setup clients with OpenAI credentials
    let http_client = reqwest::Client::new();
    let cache_options = CacheOptions {
        enabled: CacheEnabledMode::On,
        max_age_s: Some(3600),
    };
    let mut credentials = InferenceCredentials::new();
    credentials.insert("OPENAI_API_KEY".to_string(), api_key.into());

    let clients = InferenceClients {
        http_client: &http_client,
        clickhouse_connection_info: &clickhouse,
        credentials: &credentials,
        cache_options: &cache_options,
    };

    // Test with safe content
    let safe_request = ModerationRequest {
        input: ModerationInput::Single("I love programming in Rust!".to_string()),
        model: Some("omni-moderation-latest".to_string()),
    };

    let response = handle_moderation_request(
        safe_request,
        &clients,
        &credentials,
        "openai::omni-moderation-latest",
        &model_table,
    )
    .await
    .expect("Failed to moderate safe content with OpenAI");

    assert_eq!(response.results.len(), 1);
    assert!(!response.results[0].flagged);

    // Test batch moderation
    let batch_request = ModerationRequest {
        input: ModerationInput::Batch(vec![
            "The weather is nice today".to_string(),
            "I enjoy reading books".to_string(),
        ]),
        model: Some("omni-moderation-latest".to_string()),
    };

    let response = handle_moderation_request(
        batch_request,
        &clients,
        &credentials,
        "openai::omni-moderation-latest",
        &model_table,
    )
    .await
    .expect("Failed to moderate batch content with OpenAI");

    assert_eq!(response.results.len(), 2);
    assert!(!response.results[0].flagged);
    assert!(!response.results[1].flagged);
}
