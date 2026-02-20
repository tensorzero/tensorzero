use futures::future::join_all;
use std::collections::HashMap;
use std::sync::Arc;
use tempfile::NamedTempFile;
use tensorzero::{
    Client, ClientBuilder, ClientBuilderMode, ClientInferenceParams, FeedbackParams,
    InferenceOutput, InferenceResponse, Input, InputMessage, InputMessageContent, PostgresConfig,
    Role,
};
use tensorzero_core::config::Namespace;
use tensorzero_core::db::clickhouse::ClickHouseConnectionInfo;
use tensorzero_core::db::clickhouse::test_helpers::CLICKHOUSE_URL;
use tensorzero_core::db::feedback::FeedbackQueries;
use tensorzero_core::db::postgres::PostgresConnectionInfo;
use tensorzero_core::db::test_helpers::TestDatabaseHelpers;
use tensorzero_core::inference::types::Text;
use tensorzero_core::variant::chat_completion::UninitializedChatCompletionConfig;
use tokio::time::Duration;
use url::Url;
use uuid::Uuid;

use crate::clickhouse::{DeleteDbOnDrop, get_clean_clickhouse};
use crate::experimentation::track_and_stop::BernoulliBandit;

// ============================================================================
// Helpers
// ============================================================================

async fn make_embedded_gateway_with_clean_clickhouse(
    config: &str,
) -> (
    Client,
    ClickHouseConnectionInfo,
    PostgresConnectionInfo,
    DeleteDbOnDrop,
) {
    let postgres_url = std::env::var("TENSORZERO_POSTGRES_URL")
        .expect("TENSORZERO_POSTGRES_URL must be set for tests that require Postgres");

    let (clickhouse, guard) = get_clean_clickhouse(false).await;

    clickhouse
        .create_database_and_migrations_table()
        .await
        .expect("failed to create ClickHouse database for embedded gateway tests");

    let tmp_config = NamedTempFile::new().unwrap();
    std::fs::write(tmp_config.path(), config).unwrap();

    let database = clickhouse.database();
    let mut clickhouse_url = Url::parse(&CLICKHOUSE_URL).unwrap();
    clickhouse_url.set_path(database);
    let clickhouse_url_string = clickhouse_url.to_string();

    let client = ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
        config_file: Some(tmp_config.path().to_owned()),
        clickhouse_url: Some(clickhouse_url_string),
        postgres_config: Some(PostgresConfig::Url(postgres_url.clone())),
        valkey_url: None,
        timeout: None,
        verify_credentials: true,
        allow_batch_writes: true,
    })
    .build()
    .await
    .unwrap();

    let postgres = crate::db::get_test_postgres().await;

    (client, clickhouse, postgres, guard)
}

fn make_namespace_test_config() -> String {
    r#"
[models.test_model]
routing = ["test"]

[models.test_model.providers.test]
type = "dummy"
model_name = "test"

[functions.test_function]
type = "chat"

[functions.test_function.variants.variant_a]
type = "chat_completion"
model = "test_model"

[functions.test_function.variants.variant_b]
type = "chat_completion"
model = "test_model"

[functions.test_function.variants.variant_c]
type = "chat_completion"
model = "test_model"

[functions.test_function.experimentation]
type = "uniform"

[functions.test_function.experimentation.namespaces.mobile]
type = "static_weights"
candidate_variants = {"variant_a" = 1.0}

[functions.test_function.experimentation.namespaces.web]
type = "static_weights"
candidate_variants = {"variant_b" = 1.0}
"#
    .to_string()
}

async fn do_inference(client: &Client, namespace: Option<&str>) -> (uuid::Uuid, String) {
    let output = client
        .inference(ClientInferenceParams {
            function_name: Some("test_function".to_string()),
            namespace: namespace.map(|ns| Namespace::new(ns).unwrap()),
            input: Input {
                system: None,
                messages: vec![InputMessage {
                    role: Role::User,
                    content: vec![InputMessageContent::Text(Text {
                        text: "test".to_string(),
                    })],
                }],
            },
            ..Default::default()
        })
        .await
        .unwrap();

    let InferenceOutput::NonStreaming(InferenceResponse::Chat(response)) = output else {
        panic!("Expected non-streaming chat response");
    };

    (response.inference_id, response.variant_name.clone())
}

// ============================================================================
// Tests
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
async fn test_namespace_base_config_used_without_namespace() {
    let config = make_namespace_test_config();
    let (client, _clickhouse, _postgres, _guard) =
        make_embedded_gateway_with_clean_clickhouse(&config).await;

    let sample_size = 100;
    let mut counts: HashMap<String, usize> = HashMap::new();
    for _ in 0..sample_size {
        let (_, variant_name) = do_inference(&client, None).await;
        *counts.entry(variant_name).or_insert(0) += 1;
    }

    // With uniform sampling over 3 variants, we expect all three to appear
    assert!(
        counts.len() >= 2,
        "Uniform sampling without namespace should produce multiple variants, got: {counts:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_namespace_specific_config_used() {
    let config = make_namespace_test_config();
    let (client, _clickhouse, _postgres, _guard) =
        make_embedded_gateway_with_clean_clickhouse(&config).await;

    let sample_size = 20;
    for _ in 0..sample_size {
        let (_, variant_name) = do_inference(&client, Some("mobile")).await;
        assert_eq!(
            variant_name, "variant_a",
            "With namespace `mobile` (static_weights A=1.0), all inferences should use variant_a"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_namespace_different_configs() {
    let config = make_namespace_test_config();
    let (client, _clickhouse, _postgres, _guard) =
        make_embedded_gateway_with_clean_clickhouse(&config).await;

    let sample_size = 20;
    for _ in 0..sample_size {
        let (_, variant_name) = do_inference(&client, Some("web")).await;
        assert_eq!(
            variant_name, "variant_b",
            "With namespace `web` (static_weights B=1.0), all inferences should use variant_b"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_namespace_unknown_falls_back_to_base() {
    let config = make_namespace_test_config();
    let (client, _clickhouse, _postgres, _guard) =
        make_embedded_gateway_with_clean_clickhouse(&config).await;

    let sample_size = 100;
    let mut counts: HashMap<String, usize> = HashMap::new();
    for _ in 0..sample_size {
        let (_, variant_name) = do_inference(&client, Some("unknown_ns")).await;
        *counts.entry(variant_name).or_insert(0) += 1;
    }

    // Should fall back to base uniform config, producing multiple variants
    assert!(
        counts.len() >= 2,
        "Unknown namespace should fall back to uniform base config, got: {counts:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_namespace_stored_as_tag() {
    let config = make_namespace_test_config();
    let (client, clickhouse, _postgres, _guard) =
        make_embedded_gateway_with_clean_clickhouse(&config).await;

    let (inference_id, _) = do_inference(&client, Some("mobile")).await;

    // Flush ClickHouse writes and wait for them to be visible
    clickhouse.flush_pending_writes().await;
    clickhouse.sleep_for_writes_to_be_visible().await;
    tokio::time::sleep(Duration::from_millis(1000)).await;

    // Query ClickHouse for the tag
    let query = format!(
        "SELECT tags['tensorzero::namespace'] AS ns FROM ChatInference WHERE id = '{inference_id}' FORMAT JSONEachRow"
    );
    let response = clickhouse
        .run_query_synchronous_no_params(query)
        .await
        .expect("ClickHouse query should succeed");
    let rows: Vec<serde_json::Value> = response
        .response
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(
        rows.len(),
        1,
        "Should find exactly one row for the inference"
    );
    assert_eq!(
        rows[0]["ns"].as_str().unwrap(),
        "mobile",
        "The `tensorzero::namespace` tag should be stored as `mobile`"
    );
}

// ============================================================================
// Model Namespace Tests — Config Validation
// ============================================================================

/// Helper to build a client and expect a config error (gateway fails to start)
async fn expect_config_error(config: &str) -> String {
    let tmp_config = NamedTempFile::new().unwrap();
    std::fs::write(tmp_config.path(), config).unwrap();

    ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
        config_file: Some(tmp_config.path().to_owned()),
        clickhouse_url: Some(CLICKHOUSE_URL.clone()),
        postgres_config: None,
        valkey_url: None,
        timeout: None,
        verify_credentials: true,
        allow_batch_writes: true,
    })
    .build()
    .await
    .unwrap_err()
    .to_string()
}

/// Namespaced model variant in base experimentation config → error
#[tokio::test(flavor = "multi_thread")]
async fn test_namespaced_model_variant_in_base_config_rejected() {
    let config = r#"
[models.namespaced_model]
routing = ["test"]
namespace = "mobile"

[models.namespaced_model.providers.test]
type = "dummy"
model_name = "test"

[functions.test_fn]
type = "chat"

[functions.test_fn.variants.variant_a]
type = "chat_completion"
model = "namespaced_model"

[functions.test_fn.experimentation]
type = "uniform"
"#;

    let err = expect_config_error(config).await;
    assert!(
        err.contains("namespace") && err.contains("mobile") && err.contains("variant_a"),
        "Expected error about namespaced model in base config, got: {err}"
    );
}

/// Namespaced model variant in wrong namespace config → error
#[tokio::test(flavor = "multi_thread")]
async fn test_namespaced_model_variant_in_wrong_namespace_rejected() {
    let config = r#"
[models.namespaced_model]
routing = ["test"]
namespace = "mobile"

[models.namespaced_model.providers.test]
type = "dummy"
model_name = "test"

[models.regular_model]
routing = ["test"]

[models.regular_model.providers.test]
type = "dummy"
model_name = "test"

[functions.test_fn]
type = "chat"

[functions.test_fn.variants.variant_a]
type = "chat_completion"
model = "namespaced_model"

[functions.test_fn.variants.variant_b]
type = "chat_completion"
model = "regular_model"

[functions.test_fn.experimentation]
type = "static_weights"
candidate_variants = {"variant_b" = 1.0}

[functions.test_fn.experimentation.namespaces.web]
type = "static_weights"
candidate_variants = {"variant_a" = 1.0}

[functions.test_fn.experimentation.namespaces.mobile]
type = "static_weights"
candidate_variants = {"variant_a" = 1.0}
"#;

    let err = expect_config_error(config).await;
    assert!(
        err.contains("namespace") && err.contains("web") && err.contains("variant_a"),
        "Expected error about wrong namespace, got: {err}"
    );
}

/// Namespaced model variant with no experimentation block (legacy uniform) → error
#[tokio::test(flavor = "multi_thread")]
async fn test_namespaced_model_variant_no_experimentation_rejected() {
    let config = r#"
[models.namespaced_model]
routing = ["test"]
namespace = "mobile"

[models.namespaced_model.providers.test]
type = "dummy"
model_name = "test"

[functions.test_fn]
type = "chat"

[functions.test_fn.variants.variant_a]
type = "chat_completion"
model = "namespaced_model"
"#;

    let err = expect_config_error(config).await;
    assert!(
        err.contains("namespace") && err.contains("mobile") && err.contains("variant_a"),
        "Expected error about namespaced model in default experimentation, got: {err}"
    );
}

/// Valid: namespaced model variant only in matching namespace → OK
#[tokio::test(flavor = "multi_thread")]
async fn test_namespaced_model_variant_in_matching_namespace_ok() {
    let config = r#"
[models.namespaced_model]
routing = ["test"]
namespace = "mobile"

[models.namespaced_model.providers.test]
type = "dummy"
model_name = "test"

[models.regular_model]
routing = ["test"]

[models.regular_model.providers.test]
type = "dummy"
model_name = "test"

[functions.test_function]
type = "chat"

[functions.test_function.variants.variant_a]
type = "chat_completion"
model = "namespaced_model"

[functions.test_function.variants.variant_b]
type = "chat_completion"
model = "regular_model"

[functions.test_function.experimentation]
type = "static_weights"
candidate_variants = {"variant_b" = 1.0}

[functions.test_function.experimentation.namespaces.mobile]
type = "static_weights"
candidate_variants = {"variant_a" = 1.0}
"#;

    let (client, _clickhouse, _postgres, _guard) =
        make_embedded_gateway_with_clean_clickhouse(config).await;

    // Make sure the gateway started and we can do an inference
    let (_, variant_name) = do_inference(&client, Some("mobile")).await;
    assert_eq!(
        variant_name, "variant_a",
        "Mobile namespace should use variant_a (namespaced model)"
    );
}

/// BestOfN variant with evaluator in namespace B and candidates using namespace A model,
/// reachable from namespace B → error (candidate models must also be namespace-compatible)
#[tokio::test(flavor = "multi_thread")]
async fn test_best_of_n_candidate_namespace_mismatch_rejected() {
    let config = r#"
[models.mobile_model]
routing = ["test"]
namespace = "mobile"

[models.mobile_model.providers.test]
type = "dummy"
model_name = "test"

[models.web_model]
routing = ["test"]
namespace = "web"

[models.web_model.providers.test]
type = "dummy"
model_name = "test"

[models.regular_model]
routing = ["test"]

[models.regular_model.providers.test]
type = "dummy"
model_name = "test"

[functions.test_fn]
type = "chat"

# Candidate variant that uses a mobile-namespaced model
[functions.test_fn.variants.mobile_candidate]
type = "chat_completion"
model = "mobile_model"

# Regular variant for base experimentation
[functions.test_fn.variants.regular_variant]
type = "chat_completion"
model = "regular_model"

# BestOfN variant with web evaluator but mobile candidate
[functions.test_fn.variants.bon_variant]
type = "experimental_best_of_n_sampling"
candidates = ["mobile_candidate"]

[functions.test_fn.variants.bon_variant.evaluator]
model = "web_model"

[functions.test_fn.experimentation]
type = "static_weights"
candidate_variants = {"regular_variant" = 1.0}

# mobile_candidate only in mobile namespace (correct)
[functions.test_fn.experimentation.namespaces.mobile]
type = "static_weights"
candidate_variants = {"mobile_candidate" = 1.0}

# bon_variant in web namespace — should fail because it indirectly uses mobile_model
[functions.test_fn.experimentation.namespaces.web]
type = "static_weights"
candidate_variants = {"bon_variant" = 1.0}
"#;

    let err = expect_config_error(config).await;
    assert!(
        err.contains("namespace") && err.contains("mobile") && err.contains("bon_variant"),
        "Expected error about BestOfN candidate using namespaced model, got: {err}"
    );
}

/// MixtureOfN variant with fuser in namespace B and candidates using namespace A model,
/// reachable from namespace B → error
#[tokio::test(flavor = "multi_thread")]
async fn test_mixture_of_n_candidate_namespace_mismatch_rejected() {
    let config = r#"
[models.mobile_model]
routing = ["test"]
namespace = "mobile"

[models.mobile_model.providers.test]
type = "dummy"
model_name = "test"

[models.web_model]
routing = ["test"]
namespace = "web"

[models.web_model.providers.test]
type = "dummy"
model_name = "test"

[models.regular_model]
routing = ["test"]

[models.regular_model.providers.test]
type = "dummy"
model_name = "test"

[functions.test_fn]
type = "chat"

# Candidate variant that uses a mobile-namespaced model
[functions.test_fn.variants.mobile_candidate]
type = "chat_completion"
model = "mobile_model"

# Regular variant for base experimentation
[functions.test_fn.variants.regular_variant]
type = "chat_completion"
model = "regular_model"

# MixtureOfN variant with web fuser but mobile candidate
[functions.test_fn.variants.mon_variant]
type = "experimental_mixture_of_n"
candidates = ["mobile_candidate"]

[functions.test_fn.variants.mon_variant.fuser]
model = "web_model"

[functions.test_fn.experimentation]
type = "static_weights"
candidate_variants = {"regular_variant" = 1.0}

# mobile_candidate only in mobile namespace (correct)
[functions.test_fn.experimentation.namespaces.mobile]
type = "static_weights"
candidate_variants = {"mobile_candidate" = 1.0}

# mon_variant in web namespace — should fail because it indirectly uses mobile_model
[functions.test_fn.experimentation.namespaces.web]
type = "static_weights"
candidate_variants = {"mon_variant" = 1.0}
"#;

    let err = expect_config_error(config).await;
    assert!(
        err.contains("namespace") && err.contains("mobile") && err.contains("mon_variant"),
        "Expected error about MixtureOfN candidate using namespaced model, got: {err}"
    );
}

// ============================================================================
// Model Namespace Tests — Inference-time Validation
// ============================================================================

/// Helper config for inference-time namespace tests.
/// Has a namespaced model (`mobile_model` with namespace `mobile`) and a regular model.
/// The experimentation config puts `mobile_variant` ONLY in `mobile` namespace
/// and `regular_variant` in base.
fn make_inference_namespace_test_config() -> String {
    r#"
[models.mobile_model]
routing = ["test"]
namespace = "mobile"

[models.mobile_model.providers.test]
type = "dummy"
model_name = "test"

[models.regular_model]
routing = ["test"]

[models.regular_model.providers.test]
type = "dummy"
model_name = "test"

[functions.test_function]
type = "chat"

[functions.test_function.variants.mobile_variant]
type = "chat_completion"
model = "mobile_model"

[functions.test_function.variants.regular_variant]
type = "chat_completion"
model = "regular_model"

[functions.test_function.experimentation]
type = "static_weights"
candidate_variants = {"regular_variant" = 1.0}

[functions.test_function.experimentation.namespaces.mobile]
type = "static_weights"
candidate_variants = {"mobile_variant" = 1.0}
"#
    .to_string()
}

fn make_test_input() -> Input {
    Input {
        system: None,
        messages: vec![InputMessage {
            role: Role::User,
            content: vec![InputMessageContent::Text(Text {
                text: "test".to_string(),
            })],
        }],
    }
}

/// model_name path: model has namespace, request matches → OK
#[tokio::test(flavor = "multi_thread")]
async fn test_model_name_namespace_match_ok() {
    let config = make_inference_namespace_test_config();
    let (client, _clickhouse, _postgres, _guard) =
        make_embedded_gateway_with_clean_clickhouse(&config).await;

    let output = client
        .inference(ClientInferenceParams {
            model_name: Some("mobile_model".to_string()),
            namespace: Some(Namespace::new("mobile").unwrap()),
            input: make_test_input(),
            ..Default::default()
        })
        .await;

    assert!(
        output.is_ok(),
        "model_name with matching namespace should succeed"
    );
}

/// model_name path: model has namespace, request doesn't match → 400
#[tokio::test(flavor = "multi_thread")]
async fn test_model_name_namespace_mismatch_rejected() {
    let config = make_inference_namespace_test_config();
    let (client, _clickhouse, _postgres, _guard) =
        make_embedded_gateway_with_clean_clickhouse(&config).await;

    let err = client
        .inference(ClientInferenceParams {
            model_name: Some("mobile_model".to_string()),
            namespace: Some(Namespace::new("web").unwrap()),
            input: make_test_input(),
            ..Default::default()
        })
        .await
        .unwrap_err()
        .to_string();

    assert!(
        err.contains("namespace") && err.contains("mobile"),
        "Expected namespace mismatch error, got: {err}"
    );
}

/// model_name path: model has namespace, no request namespace → 400
#[tokio::test(flavor = "multi_thread")]
async fn test_model_name_namespace_none_rejected() {
    let config = make_inference_namespace_test_config();
    let (client, _clickhouse, _postgres, _guard) =
        make_embedded_gateway_with_clean_clickhouse(&config).await;

    let err = client
        .inference(ClientInferenceParams {
            model_name: Some("mobile_model".to_string()),
            input: make_test_input(),
            ..Default::default()
        })
        .await
        .unwrap_err()
        .to_string();

    assert!(
        err.contains("namespace") && err.contains("mobile"),
        "Expected namespace error for None namespace, got: {err}"
    );
}

/// model_name path: model has no namespace, request has namespace → OK (no restriction)
#[tokio::test(flavor = "multi_thread")]
async fn test_model_name_no_namespace_with_request_namespace_ok() {
    let config = make_inference_namespace_test_config();
    let (client, _clickhouse, _postgres, _guard) =
        make_embedded_gateway_with_clean_clickhouse(&config).await;

    let output = client
        .inference(ClientInferenceParams {
            model_name: Some("regular_model".to_string()),
            namespace: Some(Namespace::new("mobile").unwrap()),
            input: make_test_input(),
            ..Default::default()
        })
        .await;

    assert!(
        output.is_ok(),
        "Unnamespaced model should work with any request namespace"
    );
}

/// Pinned variant: model has namespace, request matches → OK
#[tokio::test(flavor = "multi_thread")]
async fn test_pinned_variant_namespace_match_ok() {
    let config = make_inference_namespace_test_config();
    let (client, _clickhouse, _postgres, _guard) =
        make_embedded_gateway_with_clean_clickhouse(&config).await;

    let output = client
        .inference(ClientInferenceParams {
            function_name: Some("test_function".to_string()),
            variant_name: Some("mobile_variant".to_string()),
            namespace: Some(Namespace::new("mobile").unwrap()),
            input: make_test_input(),
            ..Default::default()
        })
        .await;

    assert!(
        output.is_ok(),
        "Pinned variant with matching namespace should succeed"
    );
}

/// Pinned variant: model has namespace, request doesn't match → 400
#[tokio::test(flavor = "multi_thread")]
async fn test_pinned_variant_namespace_mismatch_rejected() {
    let config = make_inference_namespace_test_config();
    let (client, _clickhouse, _postgres, _guard) =
        make_embedded_gateway_with_clean_clickhouse(&config).await;

    let err = client
        .inference(ClientInferenceParams {
            function_name: Some("test_function".to_string()),
            variant_name: Some("mobile_variant".to_string()),
            namespace: Some(Namespace::new("web").unwrap()),
            input: make_test_input(),
            ..Default::default()
        })
        .await
        .unwrap_err()
        .to_string();

    assert!(
        err.contains("namespace") && err.contains("mobile"),
        "Expected namespace mismatch error for pinned variant, got: {err}"
    );
}

/// Pinned variant: model has namespace, no request namespace → 400
#[tokio::test(flavor = "multi_thread")]
async fn test_pinned_variant_namespace_none_rejected() {
    let config = make_inference_namespace_test_config();
    let (client, _clickhouse, _postgres, _guard) =
        make_embedded_gateway_with_clean_clickhouse(&config).await;

    let err = client
        .inference(ClientInferenceParams {
            function_name: Some("test_function".to_string()),
            variant_name: Some("mobile_variant".to_string()),
            input: make_test_input(),
            ..Default::default()
        })
        .await
        .unwrap_err()
        .to_string();

    assert!(
        err.contains("namespace") && err.contains("mobile"),
        "Expected namespace error for pinned variant with None namespace, got: {err}"
    );
}

/// Dynamic variant: model has namespace, request matches → OK
#[tokio::test(flavor = "multi_thread")]
async fn test_dynamic_variant_namespace_match_ok() {
    let config = make_inference_namespace_test_config();
    let (client, _clickhouse, _postgres, _guard) =
        make_embedded_gateway_with_clean_clickhouse(&config).await;

    let output = client
        .inference(ClientInferenceParams {
            function_name: Some("test_function".to_string()),
            namespace: Some(Namespace::new("mobile").unwrap()),
            input: make_test_input(),
            dryrun: Some(true),
            internal_dynamic_variant_config: Some(
                tensorzero_core::config::UninitializedVariantInfo {
                    inner: tensorzero_core::config::UninitializedVariantConfig::ChatCompletion(
                        UninitializedChatCompletionConfig {
                            model: "mobile_model".into(),
                            ..Default::default()
                        },
                    ),
                    timeouts: None,
                },
            ),
            ..Default::default()
        })
        .await;

    assert!(
        output.is_ok(),
        "Dynamic variant with matching namespace should succeed"
    );
}

/// Dynamic variant: model has namespace, request doesn't match → 400
#[tokio::test(flavor = "multi_thread")]
async fn test_dynamic_variant_namespace_mismatch_rejected() {
    let config = make_inference_namespace_test_config();
    let (client, _clickhouse, _postgres, _guard) =
        make_embedded_gateway_with_clean_clickhouse(&config).await;

    let err = client
        .inference(ClientInferenceParams {
            function_name: Some("test_function".to_string()),
            namespace: Some(Namespace::new("web").unwrap()),
            input: make_test_input(),
            dryrun: Some(true),
            internal_dynamic_variant_config: Some(
                tensorzero_core::config::UninitializedVariantInfo {
                    inner: tensorzero_core::config::UninitializedVariantConfig::ChatCompletion(
                        UninitializedChatCompletionConfig {
                            model: "mobile_model".into(),
                            ..Default::default()
                        },
                    ),
                    timeouts: None,
                },
            ),
            ..Default::default()
        })
        .await
        .unwrap_err()
        .to_string();

    assert!(
        err.contains("namespace") && err.contains("mobile"),
        "Expected namespace mismatch error for dynamic variant, got: {err}"
    );
}

/// Dynamic variant: model has namespace, no request namespace → 400
#[tokio::test(flavor = "multi_thread")]
async fn test_dynamic_variant_namespace_none_rejected() {
    let config = make_inference_namespace_test_config();
    let (client, _clickhouse, _postgres, _guard) =
        make_embedded_gateway_with_clean_clickhouse(&config).await;

    let err = client
        .inference(ClientInferenceParams {
            function_name: Some("test_function".to_string()),
            input: make_test_input(),
            dryrun: Some(true),
            internal_dynamic_variant_config: Some(
                tensorzero_core::config::UninitializedVariantInfo {
                    inner: tensorzero_core::config::UninitializedVariantConfig::ChatCompletion(
                        UninitializedChatCompletionConfig {
                            model: "mobile_model".into(),
                            ..Default::default()
                        },
                    ),
                    timeouts: None,
                },
            ),
            ..Default::default()
        })
        .await
        .unwrap_err()
        .to_string();

    assert!(
        err.contains("namespace") && err.contains("mobile"),
        "Expected namespace error for dynamic variant with None namespace, got: {err}"
    );
}

// ============================================================================
// Track-and-Stop Namespace Tests
// ============================================================================

/// Delay after ClickHouse flush for inferences (milliseconds).
const CLICKHOUSE_FLUSH_DELAY_MS: u64 = 1000;

/// Delay after ClickHouse flush for feedback to allow background update (milliseconds).
const BACKGROUND_UPDATE_DELAY_MS: u64 = 1000;

fn make_namespace_track_and_stop_config() -> String {
    r#"
gateway.unstable_disable_feedback_target_validation = true

[models.test_model]
routing = ["test"]

[models.test_model.providers.test]
type = "dummy"
model_name = "test"

[metrics.test_metric]
type = "boolean"
optimize = "max"
level = "inference"

[functions.test_function]
type = "chat"

[functions.test_function.variants.variant_a]
type = "chat_completion"
model = "test_model"

[functions.test_function.variants.variant_b]
type = "chat_completion"
model = "test_model"

[functions.test_function.experimentation]
type = "uniform"

[functions.test_function.experimentation.namespaces.mobile]
type = "track_and_stop"
metric = "test_metric"
candidate_variants = ["variant_a", "variant_b"]
min_samples_per_variant = 10
delta = 0.05
epsilon = 0.01
update_period_s = 1
"#
    .to_string()
}

/// Run a batch of inferences with a namespace and return (inference_id, variant_name) pairs.
async fn run_namespace_inference_batch(
    client: &Arc<Client>,
    count: usize,
    namespace: Option<&str>,
) -> Vec<(Uuid, String)> {
    let tasks: Vec<_> = (0..count)
        .map(|_| {
            let client = client.clone();
            let ns = namespace.map(|s| s.to_string());
            async move { do_inference(&client, ns.as_deref()).await }
        })
        .collect();
    join_all(tasks).await
}

/// Send boolean feedback for a batch of inferences using a Bernoulli bandit.
async fn send_namespace_feedback(
    client: &Arc<Client>,
    inference_results: &[(Uuid, String)],
    bandit: &BernoulliBandit,
    metric_name: &str,
) {
    for (inference_id, variant_name) in inference_results {
        let reward = bandit.sample(variant_name);
        client
            .feedback(FeedbackParams {
                inference_id: Some(*inference_id),
                metric_name: metric_name.to_string(),
                value: serde_json::json!(reward),
                ..Default::default()
            })
            .await
            .unwrap();
    }
}

/// Test that track_and_stop within a namespace converges to the winning variant.
///
/// This is the most important e2e test for the namespace track_and_stop feature.
/// It verifies the full pipeline: config loading -> inference routing -> namespace tag ->
/// feedback -> background task -> namespace-filtered ClickHouse query -> probability
/// update -> convergence.
#[tokio::test(flavor = "multi_thread")]
async fn test_namespace_track_and_stop_convergence() {
    let config = make_namespace_track_and_stop_config();
    let (client, clickhouse, _postgres, _guard) =
        Box::pin(make_embedded_gateway_with_clean_clickhouse(&config)).await;
    let client = Arc::new(client);

    // Set up bandit with a very clear winner: variant_a = 0.95, variant_b = 0.10
    let bandit = BernoulliBandit::new(vec![("variant_a", 0.95), ("variant_b", 0.10)], Some(42));

    let num_initial_batches = 2;
    let inferences_per_batch = 300;

    // Phase 1: Run inference + feedback batches with namespace="mobile" to train the model
    for _batch in 0..num_initial_batches {
        let inference_results =
            run_namespace_inference_batch(&client, inferences_per_batch, Some("mobile")).await;

        clickhouse.flush_pending_writes().await;
        tokio::time::sleep(Duration::from_millis(CLICKHOUSE_FLUSH_DELAY_MS)).await;

        send_namespace_feedback(&client, &inference_results, &bandit, "test_metric").await;

        clickhouse.flush_pending_writes().await;
        tokio::time::sleep(Duration::from_millis(BACKGROUND_UPDATE_DELAY_MS)).await;
    }

    // Phase 2: Run inferences with namespace="mobile" and verify convergence to variant_a
    let verification_count = 50;
    let verification_results =
        run_namespace_inference_batch(&client, verification_count, Some("mobile")).await;
    let mut variant_counts: HashMap<String, usize> = HashMap::new();
    for (_, variant_name) in &verification_results {
        *variant_counts.entry(variant_name.clone()).or_insert(0) += 1;
    }

    let variant_a_count = variant_counts.get("variant_a").copied().unwrap_or(0);
    assert_eq!(
        variant_a_count, verification_count,
        "Expected 100% of namespace `mobile` inferences to converge to variant_a (the winner), \
         but got distribution: {variant_counts:?}"
    );

    // Phase 3: Verify that non-namespaced inferences still use uniform distribution
    let base_results = run_namespace_inference_batch(&client, 50, None).await;
    let mut base_counts: HashMap<String, usize> = HashMap::new();
    for (_, variant_name) in &base_results {
        *base_counts.entry(variant_name.clone()).or_insert(0) += 1;
    }

    assert!(
        base_counts.len() >= 2,
        "Base config (no namespace) should still use uniform distribution, got: {base_counts:?}"
    );
}

/// Verify that namespace-filtered `get_feedback_by_variant` returns the correct counts.
///
/// This is extracted so it can be run against both ClickHouse and Postgres connections.
async fn verify_namespace_feedback_filtering(
    conn: &impl FeedbackQueries,
    db_name: &str,
    expected_mobile_count: u64,
    expected_total_count: u64,
) {
    // Query feedback filtered by namespace="mobile"
    let mobile_feedback = conn
        .get_feedback_by_variant("test_metric", "test_function", None, Some("mobile"), None)
        .await
        .unwrap_or_else(|e| {
            panic!("[{db_name}] Namespace-filtered feedback query should succeed: {e}")
        });

    // We should only get feedback from mobile-tagged inferences
    let total_mobile_count: u64 = mobile_feedback.iter().map(|f| f.count).sum();
    assert_eq!(
        total_mobile_count, expected_mobile_count,
        "[{db_name}] Namespace-filtered query should return exactly {expected_mobile_count} mobile inferences, \
         but got {total_mobile_count}. Feedback: {mobile_feedback:?}"
    );

    // Query feedback WITHOUT namespace filter (should include all inferences)
    let all_feedback = conn
        .get_feedback_by_variant("test_metric", "test_function", None, None, None)
        .await
        .unwrap_or_else(|e| panic!("[{db_name}] Unfiltered feedback query should succeed: {e}"));

    let total_all_count: u64 = all_feedback.iter().map(|f| f.count).sum();
    assert_eq!(
        total_all_count, expected_total_count,
        "[{db_name}] Unfiltered query should return all {expected_total_count} inferences, \
         but got {total_all_count}. Feedback: {all_feedback:?}"
    );
}

/// Shared logic for the namespace feedback query filter test.
///
/// Sends inferences + feedback through the embedded gateway, then verifies
/// the namespace-filtered `get_feedback_by_variant` query against the provided connection.
/// The connection should be the primary datastore (whichever database the gateway writes to).
async fn run_namespace_feedback_query_filter_test(
    client: Arc<Client>,
    clickhouse: &ClickHouseConnectionInfo,
    verify_conn: &impl FeedbackQueries,
    db_name: &str,
) {
    // Send inferences + feedback for namespace="mobile"
    let mobile_results = run_namespace_inference_batch(&client, 30, Some("mobile")).await;
    clickhouse.flush_pending_writes().await;
    tokio::time::sleep(Duration::from_millis(CLICKHOUSE_FLUSH_DELAY_MS)).await;

    // All mobile feedback: variant_a = true (1.0), variant_b = false (0.0)
    for (inference_id, variant_name) in &mobile_results {
        let value = if variant_name == "variant_a" {
            serde_json::json!(true)
        } else {
            serde_json::json!(false)
        };
        client
            .feedback(FeedbackParams {
                inference_id: Some(*inference_id),
                metric_name: "test_metric".to_string(),
                value,
                ..Default::default()
            })
            .await
            .unwrap();
    }

    // Send inferences + feedback with NO namespace (should not appear in namespace queries)
    let base_results = run_namespace_inference_batch(&client, 30, None).await;
    clickhouse.flush_pending_writes().await;
    tokio::time::sleep(Duration::from_millis(CLICKHOUSE_FLUSH_DELAY_MS)).await;

    // All base feedback: all true (different from mobile variant_b feedback)
    for (inference_id, _) in &base_results {
        client
            .feedback(FeedbackParams {
                inference_id: Some(*inference_id),
                metric_name: "test_metric".to_string(),
                value: serde_json::json!(true),
                ..Default::default()
            })
            .await
            .unwrap();
    }

    clickhouse.flush_pending_writes().await;
    tokio::time::sleep(Duration::from_millis(CLICKHOUSE_FLUSH_DELAY_MS)).await;

    let expected_mobile = mobile_results.len() as u64;
    let expected_total = (mobile_results.len() + base_results.len()) as u64;

    verify_namespace_feedback_filtering(verify_conn, db_name, expected_mobile, expected_total)
        .await;
}

/// Test that the namespace-filtered `get_feedback_by_variant` query correctly
/// returns only feedback for inferences tagged with the specified namespace (ClickHouse).
#[tokio::test(flavor = "multi_thread")]
async fn test_namespace_feedback_query_filters_correctly_clickhouse() {
    let config = make_namespace_track_and_stop_config();
    let (client, clickhouse, _postgres, _guard) =
        Box::pin(make_embedded_gateway_with_clean_clickhouse(&config)).await;

    run_namespace_feedback_query_filter_test(
        Arc::new(client),
        &clickhouse,
        &clickhouse,
        "ClickHouse",
    )
    .await;
}

/// Test that the namespace-filtered `get_feedback_by_variant` query correctly
/// returns only feedback for inferences tagged with the specified namespace (Postgres).
///
/// Sets `ENABLE_POSTGRES_AS_PRIMARY_DATASTORE=true` so the embedded gateway writes to Postgres.
#[tokio::test(flavor = "multi_thread")]
async fn test_namespace_feedback_query_filters_correctly_postgres() {
    // Must be set before the first flag access (OnceLock caches on first read)
    tensorzero_unsafe_helpers::set_env_var_tests_only(
        "TENSORZERO_INTERNAL_FLAG_ENABLE_POSTGRES_AS_PRIMARY_DATASTORE",
        "true",
    );

    let config = make_namespace_track_and_stop_config();
    let (client, clickhouse, postgres, _guard) =
        Box::pin(make_embedded_gateway_with_clean_clickhouse(&config)).await;

    run_namespace_feedback_query_filter_test(Arc::new(client), &clickhouse, &postgres, "Postgres")
        .await;
}
