//! E2E tests for `load_config_from_db` (config-in-db read path).

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use googletest::prelude::*;
use sqlx::PgPool;
use uuid::Uuid;

use tensorzero_core::config::editable::{config_to_toml, config_to_toml_with_errors};
use tensorzero_core::config::path::ResolvedTomlPathData;
use tensorzero_core::config::{
    Namespace, NonStreamingTimeouts, StreamingTimeouts, TimeoutsConfig, UninitializedConfig,
    UninitializedFunctionConfig, UninitializedFunctionConfigJson, UninitializedSchemas,
    UninitializedToolConfig, UninitializedVariantConfig, UninitializedVariantInfo,
};
use tensorzero_core::db::postgres::PostgresConnectionInfo;
use tensorzero_core::db::postgres::function_config_writes::WriteFunctionConfigParams;
use tensorzero_core::db::postgres::stored_config_queries::load_config_from_db;
use tensorzero_core::evaluations::{
    ExactMatchConfig, UninitializedEvaluationConfig, UninitializedEvaluatorConfig,
    UninitializedInferenceEvaluationConfig,
};
use tensorzero_core::inference::types::extra_body::{
    ExtraBodyConfig, ExtraBodyReplacement, ExtraBodyReplacementKind,
};
use tensorzero_core::inference::types::extra_headers::{
    ExtraHeader, ExtraHeaderKind, ExtraHeadersConfig,
};
use tensorzero_core::utils::retries::RetryConfig;
use tensorzero_core::variant::chat_completion::{
    UninitializedChatCompletionConfig, UninitializedChatTemplate, UninitializedChatTemplates,
};
use tensorzero_stored_config::{
    StoredEvaluationConfig, StoredEvaluatorConfig, StoredExactMatchConfig, StoredFileRef,
    StoredFunctionConfig, StoredInferenceEvaluationConfig, StoredJsonFunctionConfig,
    StoredToolConfig, StoredVariantRef,
};

fn empty_config() -> UninitializedConfig {
    // Mirrors what `load_config_from_db` returns for an empty database:
    // every singleton is populated with its default, every collection is
    // an empty map, and `object_storage` is `None` because there is no
    // default object storage config.
    UninitializedConfig {
        gateway: Some(Default::default()),
        clickhouse: Some(Default::default()),
        postgres: Some(Default::default()),
        rate_limiting: Some(Default::default()),
        object_storage: None,
        models: Some(HashMap::new()),
        embedding_models: Some(HashMap::new()),
        functions: Some(HashMap::new()),
        metrics: Some(HashMap::new()),
        tools: Some(HashMap::new()),
        evaluations: Some(HashMap::new()),
        provider_types: Some(Default::default()),
        optimizers: Some(HashMap::new()),
        autopilot: Some(Default::default()),
    }
}

fn fake_template(key: &str, body: &str) -> ResolvedTomlPathData {
    ResolvedTomlPathData::new_fake_path(key.to_string(), body.to_string())
}

fn sample_extra_body() -> ExtraBodyConfig {
    ExtraBodyConfig {
        data: vec![ExtraBodyReplacement {
            pointer: "/temperature".to_string(),
            kind: ExtraBodyReplacementKind::Value(serde_json::json!(0.25)),
        }],
    }
}

fn sample_extra_headers() -> ExtraHeadersConfig {
    ExtraHeadersConfig {
        data: vec![ExtraHeader {
            name: "x-test".to_string(),
            kind: ExtraHeaderKind::Value("true".to_string()),
        }],
    }
}

fn sample_function() -> UninitializedFunctionConfig {
    UninitializedFunctionConfig::Json(UninitializedFunctionConfigJson {
        variants: HashMap::from([(
            "chat".to_string(),
            UninitializedVariantInfo {
                inner: UninitializedVariantConfig::ChatCompletion(
                    UninitializedChatCompletionConfig {
                        weight: Some(1.0),
                        model: Arc::<str>::from("openai::gpt-5-mini"),
                        system_template: Some(fake_template(
                            "functions.test.variants.chat.system_template",
                            "{% include 'shared/header' %}\nSystem {{ input }}",
                        )),
                        user_template: Some(fake_template(
                            "functions.test.variants.chat.user_template",
                            "User {{ input }}",
                        )),
                        assistant_template: None,
                        input_wrappers: None,
                        templates: UninitializedChatTemplates {
                            inner: HashMap::from([(
                                "header".to_string(),
                                UninitializedChatTemplate {
                                    path: fake_template("shared/header", "Shared header"),
                                },
                            )]),
                        },
                        temperature: Some(0.2),
                        top_p: Some(0.9),
                        max_tokens: Some(256),
                        presence_penalty: Some(0.1),
                        frequency_penalty: Some(0.2),
                        seed: Some(7),
                        stop_sequences: Some(vec!["DONE".to_string()]),
                        reasoning_effort: Some("medium".to_string()),
                        service_tier: None,
                        thinking_budget_tokens: Some(32),
                        verbosity: Some("low".to_string()),
                        json_mode: None,
                        retries: RetryConfig {
                            num_retries: 2,
                            max_delay_s: 1.5,
                        },
                        extra_body: Some(sample_extra_body()),
                        extra_headers: Some(sample_extra_headers()),
                    },
                ),
                timeouts: Some(TimeoutsConfig {
                    non_streaming: Some(NonStreamingTimeouts {
                        total_ms: Some(5_000),
                    }),
                    streaming: Some(StreamingTimeouts {
                        ttft_ms: Some(1_000),
                        total_ms: Some(8_000),
                    }),
                }),
                namespace: Some(Namespace::new("alpha").expect("test namespace should be valid")),
            },
        )]),
        system_schema: Some(fake_template(
            "functions.test.system_schema",
            "{\"type\":\"object\"}",
        )),
        user_schema: None,
        assistant_schema: None,
        schemas: UninitializedSchemas::from_paths(HashMap::from([(
            "custom".to_string(),
            fake_template("functions.test.schemas.custom", "{\"type\":\"number\"}"),
        )])),
        output_schema: Some(fake_template(
            "functions.test.output_schema",
            "{\"type\":\"string\"}",
        )),
        description: Some("JSON test function".to_string()),
        experimentation: None,
        evaluators: HashMap::new(),
    })
}

fn sample_tool() -> UninitializedToolConfig {
    UninitializedToolConfig {
        description: "Search docs".to_string(),
        parameters: fake_template(
            "tools.search.parameters",
            "{\"type\":\"object\",\"properties\":{\"query\":{\"type\":\"string\"}}}",
        ),
        name: Some("search".to_string()),
        strict: true,
    }
}

fn sample_evaluation() -> UninitializedEvaluationConfig {
    #[expect(deprecated)]
    let exact_match = ExactMatchConfig { cutoff: Some(0.9) };
    UninitializedEvaluationConfig::Inference(UninitializedInferenceEvaluationConfig {
        evaluators: HashMap::from([(
            "exact".to_string(),
            UninitializedEvaluatorConfig::ExactMatch(exact_match),
        )]),
        function_name: "test".to_string(),
        description: Some("Exact-match evaluation".to_string()),
    })
}

async fn insert_file(pool: &PgPool, file_path: &str, source_body: &str) -> Uuid {
    let id = Uuid::now_v7();
    let content_hash = blake3::hash(source_body.as_bytes()).as_bytes().to_vec();
    sqlx::query(
        "INSERT INTO tensorzero.stored_files \
         (id, file_path, source_body, content_hash, creation_source) \
         VALUES ($1, $2, $3, $4, $5)",
    )
    .bind(id)
    .bind(file_path)
    .bind(source_body)
    .bind(content_hash)
    .bind("test")
    .execute(pool)
    .await
    .expect("stored file insert should succeed");
    id
}

/// Smoke test for the zero-config gateway boot path: a freshly-migrated
/// database with no rows should load into an `UninitializedConfig` whose
/// singletons are populated with their defaults and whose collections are
/// empty. This is the shape the gateway sees when it starts with only a
/// Postgres connection and an otherwise empty database.
#[sqlx::test(migrator = "tensorzero_stored_config::postgres::MIGRATOR")]
async fn load_config_from_db_returns_defaults_on_empty_database(pool: PgPool) {
    let loaded = load_config_from_db(&pool)
        .await
        .expect("loading an empty database should succeed");
    // No rows to load means no per-row loading errors.
    assert_that!(loaded.loading_errors.is_empty(), eq(true));
    assert_that!(
        loaded.config,
        matches_pattern!(UninitializedConfig {
            gateway: some(anything()),
            clickhouse: some(anything()),
            postgres: some(anything()),
            rate_limiting: some(anything()),
            object_storage: none(),
            models: some(is_empty()),
            embedding_models: some(is_empty()),
            functions: some(is_empty()),
            metrics: some(is_empty()),
            tools: some(is_empty()),
            evaluations: some(is_empty()),
            provider_types: some(anything()),
            optimizers: some(is_empty()),
            autopilot: some(anything()),
        })
    );
}

#[sqlx::test(migrator = "tensorzero_stored_config::postgres::MIGRATOR")]
async fn load_config_from_db_round_trips_written_function_configs(pool: PgPool) {
    let postgres = PostgresConnectionInfo::new_with_pool(pool.clone());
    let function = sample_function();
    postgres
        .write_function_config(WriteFunctionConfigParams {
            function_name: "test",
            config: &function,
            expected_current_version_id: None,
            creation_source: "test",
            source_autopilot_session_id: None,
            extra_templates: &HashMap::new(),
        })
        .await
        .expect("write path should succeed");

    let loaded = load_config_from_db(&pool)
        .await
        .expect("DB load should succeed");

    let mut expected = empty_config();
    expected
        .functions
        .get_or_insert_with(HashMap::new)
        .insert("test".to_string(), sample_function());
    assert_that!(&loaded.config, eq(&expected));
    assert_that!(loaded.loading_errors.is_empty(), eq(true));
}

#[sqlx::test(migrator = "tensorzero_stored_config::postgres::MIGRATOR")]
async fn load_config_from_db_loads_top_level_tools_and_evaluations(pool: PgPool) {
    let tool_file_id = insert_file(
        &pool,
        "tools.search.parameters",
        "{\"type\":\"object\",\"properties\":{\"query\":{\"type\":\"string\"}}}",
    )
    .await;

    sqlx::query(
        "INSERT INTO tensorzero.tools_configs (id, name, schema_revision, config) \
         VALUES ($1, $2, $3, $4)",
    )
    .bind(Uuid::now_v7())
    .bind("search")
    .bind(1_i32)
    .bind(
        serde_json::to_value(StoredToolConfig {
            description: "Search docs".to_string(),
            parameters: StoredFileRef {
                file_version_id: tool_file_id,
                file_path: "tools.search.parameters".to_string(),
            },
            name: Some("search".to_string()),
            strict: true,
        })
        .expect("tool config should serialize"),
    )
    .execute(&pool)
    .await
    .expect("tool row insert should succeed");

    sqlx::query(
        "INSERT INTO tensorzero.evaluations_configs (id, name, schema_revision, config) \
         VALUES ($1, $2, $3, $4)",
    )
    .bind(Uuid::now_v7())
    .bind("exact_eval")
    .bind(1_i32)
    .bind(
        serde_json::to_value(StoredEvaluationConfig::Inference(
            StoredInferenceEvaluationConfig {
                evaluators: Some(BTreeMap::from([(
                    "exact".to_string(),
                    StoredEvaluatorConfig::ExactMatch(StoredExactMatchConfig { cutoff: Some(0.9) }),
                )])),
                function_name: "test".to_string(),
                description: Some("Exact-match evaluation".to_string()),
            },
        ))
        .expect("evaluation config should serialize"),
    )
    .execute(&pool)
    .await
    .expect("evaluation row insert should succeed");

    let loaded = load_config_from_db(&pool)
        .await
        .expect("DB load should succeed");

    assert_that!(
        loaded.config.tools.as_ref().and_then(|t| t.get("search")),
        some(eq(&sample_tool()))
    );
    assert_that!(
        loaded
            .config
            .evaluations
            .as_ref()
            .and_then(|e| e.get("exact_eval")),
        some(eq(&sample_evaluation()))
    );
}

#[sqlx::test(migrator = "tensorzero_stored_config::postgres::MIGRATOR")]
async fn load_config_from_db_skips_invalid_collection_rows_and_broken_functions(pool: PgPool) {
    let postgres = PostgresConnectionInfo::new_with_pool(pool.clone());
    let function = sample_function();
    postgres
        .write_function_config(WriteFunctionConfigParams {
            function_name: "good",
            config: &function,
            expected_current_version_id: None,
            creation_source: "test",
            source_autopilot_session_id: None,
            extra_templates: &HashMap::new(),
        })
        .await
        .expect("good function write should succeed");

    sqlx::query(
        "INSERT INTO tensorzero.models_configs (id, name, schema_revision, config) \
         VALUES ($1, $2, $3, $4)",
    )
    .bind(Uuid::now_v7())
    .bind("broken-model")
    .bind(999_i32)
    .bind(serde_json::json!({}))
    .execute(&pool)
    .await
    .expect("bad model row insert should succeed");

    let broken_function_id = Uuid::now_v7();
    let missing_variant_version_id = Uuid::now_v7();
    sqlx::query(
        "INSERT INTO tensorzero.function_configs \
         (id, name, function_type, schema_revision, config, creation_source) \
         VALUES ($1, $2, $3, $4, $5, $6)",
    )
    .bind(broken_function_id)
    .bind("broken")
    .bind("json")
    .bind(1_i32)
    .bind(
        serde_json::to_value(StoredFunctionConfig::Json(StoredJsonFunctionConfig {
            variants: Some(BTreeMap::from([(
                "chat".to_string(),
                StoredVariantRef {
                    variant_version_id: missing_variant_version_id,
                },
            )])),
            system_schema: None,
            user_schema: None,
            assistant_schema: None,
            schemas: None,
            output_schema: None,
            description: Some("broken function".to_string()),
            experimentation: None,
            evaluators: None,
        }))
        .expect("broken function config should serialize"),
    )
    .bind("test")
    .execute(&pool)
    .await
    .expect("broken function row insert should succeed");

    let loaded = load_config_from_db(&pool)
        .await
        .expect("DB load should succeed");

    // The broken model is skipped — models map is empty.
    assert_that!(
        loaded.config.models.as_ref().is_none_or(|m| m.is_empty()),
        eq(true)
    );
    // The "good" function loads normally.
    assert_that!(
        loaded.config.functions.as_ref().and_then(|f| f.get("good")),
        some(eq(&sample_function()))
    );
    // The "broken" function loads but with an empty variants map (the missing
    // variant reference is collected as a per-variant error, not a fatal failure).
    let broken = loaded
        .config
        .functions
        .as_ref()
        .and_then(|f| f.get("broken"));
    assert_that!(broken.is_some(), eq(true));

    // Both the broken model and the missing variant should be in loading_errors.
    assert_that!(loaded.loading_errors.len(), ge(2usize));
    assert_that!(
        loaded
            .loading_errors
            .iter()
            .any(|e| e.kind == "model" && e.name == "broken-model"),
        eq(true)
    );
    // The missing variant should appear as a variant-kind error.
    assert_that!(
        loaded.loading_errors.iter().any(|e| e.kind == "variant"),
        eq(true)
    );
}

#[sqlx::test(migrator = "tensorzero_stored_config::postgres::MIGRATOR")]
async fn load_config_from_db_collects_invalid_singleton_rows_as_loading_errors(pool: PgPool) {
    sqlx::query(
        "INSERT INTO tensorzero.gateway_configs (id, schema_revision, config) \
         VALUES ($1, $2, $3)",
    )
    .bind(Uuid::now_v7())
    .bind(999_i32)
    .bind(serde_json::json!({}))
    .execute(&pool)
    .await
    .expect("bad gateway row insert should succeed");

    // Invalid singleton rows no longer abort the load — they produce loading errors and
    // fall back to Default::default() so the gateway can still start up.
    let loaded = load_config_from_db(&pool)
        .await
        .expect("load should succeed despite broken singleton row");

    assert_that!(loaded.loading_errors.len(), eq(1));
    assert_that!(loaded.loading_errors[0].kind, eq("gateway_config"));
    assert_that!(
        loaded.loading_errors[0].error,
        contains_substring("unsupported schema revision 999 for `gateway_config`")
    );
}

/// Regression test for the 7d fix: the apply handler's CAS check must compute
/// the same base_signature that GET returned, even when the DB has broken
/// rows. Both paths must serialize via `config_to_toml_with_errors(config,
/// loading_errors)` — if either side drifts back to plain `config_to_toml`,
/// the signatures will diverge any time `loading_errors` is non-empty and
/// every save will fail with a false CAS conflict.
///
/// This test locks in the contract that two independent loads of the same
/// database state produce byte-identical annotated TOML.
#[sqlx::test(migrator = "tensorzero_stored_config::postgres::MIGRATOR")]
async fn config_to_toml_with_errors_is_stable_across_loads_for_cas(pool: PgPool) {
    // Broken singleton: invalid schema revision on gateway row.
    sqlx::query(
        "INSERT INTO tensorzero.gateway_configs (id, schema_revision, config) \
         VALUES ($1, $2, $3)",
    )
    .bind(Uuid::now_v7())
    .bind(999_i32)
    .bind(serde_json::json!({}))
    .execute(&pool)
    .await
    .expect("bad gateway row insert should succeed");

    // Broken collection row: model with unsupported schema revision.
    sqlx::query(
        "INSERT INTO tensorzero.models_configs (id, name, schema_revision, config) \
         VALUES ($1, $2, $3, $4)",
    )
    .bind(Uuid::now_v7())
    .bind("broken-model")
    .bind(999_i32)
    .bind(serde_json::json!({}))
    .execute(&pool)
    .await
    .expect("bad model row insert should succeed");

    // Load twice in a row. The DB state is identical, so both annotated TOML
    // strings and their path_contents maps must be byte-identical. If the
    // apply handler ever recomputes via plain `config_to_toml`, the second
    // string will drop the `# BROKEN:` fragments and the CAS signature will
    // mismatch on every save.
    let first = load_config_from_db(&pool)
        .await
        .expect("first load should succeed");
    let second = load_config_from_db(&pool)
        .await
        .expect("second load should succeed");

    assert_that!(first.loading_errors.len(), eq(2));
    assert_that!(second.loading_errors.len(), eq(2));

    let (first_toml, first_paths) =
        config_to_toml_with_errors(&first.config, &first.loading_errors)
            .expect("first annotated serialization should succeed");
    let (second_toml, second_paths) =
        config_to_toml_with_errors(&second.config, &second.loading_errors)
            .expect("second annotated serialization should succeed");

    assert_that!(&first_toml, eq(&second_toml));
    assert_that!(&first_paths, eq(&second_paths));

    // The annotated TOML must actually contain the broken-item markers,
    // otherwise this test would pass vacuously if both sides regressed to
    // plain `config_to_toml`.
    assert_that!(&first_toml, contains_substring("# BROKEN ("));
    assert_that!(
        &first_toml,
        contains_substring(r#"# BROKEN (gateway_config "gateway_config")"#)
    );
    assert_that!(
        &first_toml,
        contains_substring(r#"# BROKEN (model "broken-model")"#)
    );
}

/// Locks in the rationale for `apply_config_toml_handler` re-loading from the
/// DB after commit and using `config_to_toml_with_errors` (rather than plain
/// `config_to_toml`) for the response. When the DB still has broken rows that
/// survived the apply (e.g. a malformed singleton the user did not overwrite),
/// the two functions produce different output, so the apply response signature
/// must be computed via the same function GET uses — otherwise consumers
/// reusing the apply response's `base_signature` for their next save would hit
/// a false CAS conflict on the very next call.
#[sqlx::test(migrator = "tensorzero_stored_config::postgres::MIGRATOR")]
async fn plain_and_annotated_toml_diverge_when_loading_errors_present(pool: PgPool) {
    sqlx::query(
        "INSERT INTO tensorzero.gateway_configs (id, schema_revision, config) \
         VALUES ($1, $2, $3)",
    )
    .bind(Uuid::now_v7())
    .bind(999_i32)
    .bind(serde_json::json!({}))
    .execute(&pool)
    .await
    .expect("bad gateway row insert should succeed");

    let loaded = load_config_from_db(&pool)
        .await
        .expect("load should succeed even with broken row");
    assert_that!(loaded.loading_errors.len(), eq(1));

    let (plain_toml, _) =
        config_to_toml(&loaded.config).expect("plain serialization should succeed");
    let (annotated_toml, _) = config_to_toml_with_errors(&loaded.config, &loaded.loading_errors)
        .expect("annotated serialization should succeed");

    assert_that!(&plain_toml, not(eq(&annotated_toml)));
    assert_that!(&plain_toml, not(contains_substring("# BROKEN (")));
    assert_that!(&annotated_toml, contains_substring("# BROKEN ("));
}
