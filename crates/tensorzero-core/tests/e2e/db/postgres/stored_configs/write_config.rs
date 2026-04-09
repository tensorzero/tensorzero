//! E2E tests for writing whole `UninitializedConfig`s to Postgres (config-in-db).

use std::collections::HashMap;

use googletest::assert_that;
use googletest::matchers::*;
use googletest_matchers::{matches_json_literal, partially};
use serde::de::DeserializeOwned;
use sqlx::{PgPool, Row};

use tensorzero_core::config::{
    AutopilotConfig, ClickHouseConfig, MetricConfig, MetricConfigLevel, MetricConfigOptimize,
    MetricConfigType, PostgresConfig, UninitializedConfig, UninitializedFunctionConfig,
    UninitializedToolConfig, path::ResolvedTomlPathData,
};
use tensorzero_core::db::postgres::PostgresConnectionInfo;
use tensorzero_core::db::postgres::stored_config_queries::load_config_from_db;
use tensorzero_core::db::postgres::stored_config_writes::WriteStoredConfigParams;
use tensorzero_core::embeddings::UninitializedEmbeddingModelConfig;
use tensorzero_core::evaluations::UninitializedEvaluationConfig;
use tensorzero_core::inference::types::storage::StorageKind;
use tensorzero_core::model::UninitializedModelConfig;
use tensorzero_core::optimization::UninitializedOptimizerInfo;
use tensorzero_core::rate_limiting::UninitializedRateLimitingConfig;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn deserialize_from_json<T: DeserializeOwned>(value: serde_json::Value) -> T {
    serde_json::from_value(value).expect("test value should deserialize")
}

fn fake_template(key: &str, body: &str) -> ResolvedTomlPathData {
    ResolvedTomlPathData::new_fake_path(key.to_string(), body.to_string())
}

fn default_write_params(config: &UninitializedConfig) -> WriteStoredConfigParams<'_> {
    // The extra-templates HashMap is only consumed by the function-config path,
    // which these tests exercise only incidentally. A `'static` empty map keeps
    // the lifetime bookkeeping simple.
    static EMPTY_TEMPLATES: std::sync::OnceLock<HashMap<String, String>> =
        std::sync::OnceLock::new();
    WriteStoredConfigParams {
        config,
        creation_source: "ui",
        source_autopilot_session_id: None,
        extra_templates: EMPTY_TEMPLATES.get_or_init(HashMap::new),
        path_prefix_to_strip: None,
    }
}

async fn fetch_latest_singleton_config(pool: &PgPool, table: &str) -> serde_json::Value {
    let query = format!("SELECT config FROM tensorzero.{table} ORDER BY created_at DESC LIMIT 1");
    let row = sqlx::query(sqlx::AssertSqlSafe(query))
        .fetch_one(pool)
        .await
        .unwrap_or_else(|e| panic!("singleton row in `{table}` should exist: {e}"));
    row.get("config")
}

async fn count_rows(pool: &PgPool, table: &str) -> i64 {
    let query = format!("SELECT COUNT(*)::BIGINT FROM tensorzero.{table}");
    sqlx::query_scalar(sqlx::AssertSqlSafe(query))
        .fetch_one(pool)
        .await
        .unwrap_or_else(|e| panic!("count query on `{table}` should succeed: {e}"))
}

async fn fetch_named_config(pool: &PgPool, table: &str, name: &str) -> serde_json::Value {
    let query = format!("SELECT config FROM tensorzero.{table} WHERE name = $1");
    let row = sqlx::query(sqlx::AssertSqlSafe(query))
        .bind(name)
        .fetch_one(pool)
        .await
        .unwrap_or_else(|e| panic!("named row `{name}` in `{table}` should exist: {e}"));
    row.get("config")
}

// ── Tests: empty config ───────────────────────────────────────────────────────

/// An `UninitializedConfig::default()` has every field set to `None`, so
/// `write_stored_config` should succeed and leave every config table empty.
#[sqlx::test(migrator = "tensorzero_stored_config::postgres::MIGRATOR")]
async fn write_stored_config_empty_config_writes_no_rows(pool: PgPool) {
    let postgres = PostgresConnectionInfo::new_with_pool(pool.clone());
    let config = UninitializedConfig::default();
    postgres
        .write_stored_config(default_write_params(&config))
        .await
        .expect("empty config should write successfully");

    for table in [
        "gateway_configs",
        "clickhouse_configs",
        "postgres_configs",
        "object_storage_configs",
        "rate_limiting_configs",
        "autopilot_configs",
        "provider_types_configs",
        "models_configs",
        "embedding_models_configs",
        "metrics_configs",
        "optimizers_configs",
        "tools_configs",
        "evaluations_configs",
        "function_configs",
        "stored_files",
    ] {
        assert_that!(count_rows(&pool, table).await, eq(0));
    }
}

// ── Tests: singleton tables ───────────────────────────────────────────────────

/// Populate every singleton-backed config field (gateway, clickhouse, postgres,
/// object_storage, rate_limiting, autopilot, provider_types) and assert that
/// one row is written to each singleton table with the expected contents.
#[sqlx::test(migrator = "tensorzero_stored_config::postgres::MIGRATOR")]
async fn write_stored_config_persists_singleton_configs(pool: PgPool) {
    let postgres = PostgresConnectionInfo::new_with_pool(pool.clone());

    let gateway = deserialize_from_json(serde_json::json!({
        "bind_address": "127.0.0.1:3000",
        "debug": true,
        "observability": { "enabled": true, "async_writes": true },
    }));
    let rate_limiting: UninitializedRateLimitingConfig = deserialize_from_json(serde_json::json!({
        "enabled": true,
        "backend": "postgres",
        "default_nano_cost": 1_500_000_000u64,
        "rules": [
            {
                "limits": [
                    {
                        "resource": "model_inference",
                        "interval": "minute",
                        "capacity": 100,
                        "refill_rate": 10
                    }
                ],
                "scope": [
                    { "tag_key": "user_id", "tag_value": "tensorzero::each" }
                ],
                "priority": { "Priority": 3 }
            }
        ]
    }));
    let provider_types = deserialize_from_json(serde_json::json!({
        "openai": {
            "defaults": { "api_key_location": "env::OPENAI_API_KEY" }
        }
    }));

    #[expect(deprecated)]
    let config = UninitializedConfig {
        gateway: Some(gateway),
        clickhouse: Some(ClickHouseConfig {
            disable_automatic_migrations: Some(true),
        }),
        postgres: Some(PostgresConfig {
            enabled: None,
            connection_pool_size: Some(17),
            inference_metadata_retention_days: Some(30),
            inference_data_retention_days: Some(7),
        }),
        object_storage: Some(StorageKind::Disabled),
        rate_limiting: Some(rate_limiting),
        autopilot: Some(AutopilotConfig {
            tool_whitelist: Some(vec!["ls".to_string(), "cat".to_string()]),
        }),
        provider_types: Some(provider_types),
        ..Default::default()
    };

    postgres
        .write_stored_config(default_write_params(&config))
        .await
        .expect("singleton writes should succeed");

    assert_that!(
        fetch_latest_singleton_config(&pool, "gateway_configs").await,
        matches_json_literal!({
            "bind_address": "127.0.0.1:3000",
            "debug": true,
            "observability": {
                "enabled": true,
                "async_writes": true,
            },
        })
    );

    assert_that!(
        fetch_latest_singleton_config(&pool, "clickhouse_configs").await,
        matches_json_literal!({ "disable_automatic_migrations": true })
    );

    assert_that!(
        fetch_latest_singleton_config(&pool, "postgres_configs").await,
        matches_json_literal!({
            "connection_pool_size": 17,
            "inference_metadata_retention_days": 30,
            "inference_data_retention_days": 7,
        })
    );

    assert_that!(
        fetch_latest_singleton_config(&pool, "object_storage_configs").await,
        matches_json_literal!({ "type": "disabled" })
    );

    assert_that!(
        fetch_latest_singleton_config(&pool, "rate_limiting_configs").await,
        matches_json_literal!({
            "enabled": true,
            "backend": "postgres",
            "default_nano_cost": 1_500_000_000u64,
            "rules": [
                {
                    "limits": [
                        {
                            "resource": "model_inference",
                            "interval": "minute",
                            "capacity": 100,
                            "refill_rate": 10
                        }
                    ],
                    "scope": {
                        "scopes": [
                            {
                                "type": "tag",
                                "tag_key": "user_id",
                                "tag_value": { "type": "each" }
                            }
                        ]
                    },
                    "priority": { "type": "priority", "value": 3 }
                }
            ]
        })
    );

    assert_that!(
        fetch_latest_singleton_config(&pool, "autopilot_configs").await,
        matches_json_literal!({ "tool_whitelist": ["ls", "cat"] })
    );

    // Provider types: `CredentialLocationWithFallback` is internally tagged, so
    // the stored JSON nests `type: single` around a `StoredCredentialLocation`.
    assert_that!(
        fetch_latest_singleton_config(&pool, "provider_types_configs").await,
        matches_json_literal!({
            "openai": {
                "defaults": {
                    "api_key_location": {
                        "type": "single",
                        "location": { "type": "env", "value": "OPENAI_API_KEY" }
                    }
                }
            }
        })
    );
}

/// Re-running `write_stored_config` with a config that has singleton fields
/// set should append a new row to each singleton table (singleton tables are
/// append-only — the latest row wins on read, previous rows are retained as
/// an audit trail).
#[sqlx::test(migrator = "tensorzero_stored_config::postgres::MIGRATOR")]
async fn write_stored_config_singletons_are_append_only(pool: PgPool) {
    let postgres = PostgresConnectionInfo::new_with_pool(pool.clone());

    let config = UninitializedConfig {
        clickhouse: Some(ClickHouseConfig {
            disable_automatic_migrations: Some(false),
        }),
        ..Default::default()
    };
    postgres
        .write_stored_config(default_write_params(&config))
        .await
        .expect("first write should succeed");
    postgres
        .write_stored_config(default_write_params(&config))
        .await
        .expect("second write should succeed");

    assert_that!(count_rows(&pool, "clickhouse_configs").await, eq(2));
}

// ── Tests: named-collection tables ────────────────────────────────────────────

/// Populate models, embedding models, metrics, and optimizers and assert that
/// each named-collection table receives one row per entry with the expected
/// values.
#[sqlx::test(migrator = "tensorzero_stored_config::postgres::MIGRATOR")]
async fn write_stored_config_persists_named_collections(pool: PgPool) {
    let postgres = PostgresConnectionInfo::new_with_pool(pool.clone());

    let model: UninitializedModelConfig = deserialize_from_json(serde_json::json!({
        "routing": ["dummy_provider"],
        "providers": {
            "dummy_provider": {
                "type": "dummy",
                "model_name": "dummy-model"
            }
        }
    }));
    let embedding_model: UninitializedEmbeddingModelConfig =
        deserialize_from_json(serde_json::json!({
            "routing": ["embed_provider"],
            "providers": {
                "embed_provider": {
                    "type": "openai",
                    "model_name": "text-embedding-3-large"
                }
            }
        }));
    let dicl_optimizer: UninitializedOptimizerInfo = deserialize_from_json(serde_json::json!({
        "type": "dicl",
        "embedding_model": "embed_provider",
        "variant_name": "variant_a",
        "function_name": "function_a",
        "k": 5
    }));

    let config = UninitializedConfig {
        models: Some(HashMap::from([(
            std::sync::Arc::<str>::from("model_a"),
            model,
        )])),
        embedding_models: Some(HashMap::from([(
            std::sync::Arc::<str>::from("embed_a"),
            embedding_model,
        )])),
        metrics: Some(HashMap::from([(
            "quality".to_string(),
            MetricConfig {
                r#type: MetricConfigType::Float,
                optimize: MetricConfigOptimize::Max,
                level: MetricConfigLevel::Inference,
                description: Some("Quality score".to_string()),
            },
        )])),
        optimizers: Some(HashMap::from([(
            "dicl_optimizer".to_string(),
            dicl_optimizer,
        )])),
        ..Default::default()
    };

    postgres
        .write_stored_config(default_write_params(&config))
        .await
        .expect("named collection writes should succeed");

    // Models.
    assert_that!(count_rows(&pool, "models_configs").await, eq(1));
    assert_that!(
        fetch_named_config(&pool, "models_configs", "model_a").await,
        partially(matches_json_literal!({
            "routing": ["dummy_provider"],
            "providers": {
                "dummy_provider": {
                    "provider": { "type": "dummy", "model_name": "dummy-model" }
                }
            }
        }))
    );

    // Embedding models.
    assert_that!(count_rows(&pool, "embedding_models_configs").await, eq(1));
    assert_that!(
        fetch_named_config(&pool, "embedding_models_configs", "embed_a").await,
        partially(matches_json_literal!({
            "routing": ["embed_provider"],
            "providers": {
                "embed_provider": {
                    "provider": { "type": "openai", "model_name": "text-embedding-3-large" }
                }
            }
        }))
    );

    // Metrics.
    assert_that!(count_rows(&pool, "metrics_configs").await, eq(1));
    assert_that!(
        fetch_named_config(&pool, "metrics_configs", "quality").await,
        matches_json_literal!({
            "type": "float",
            "optimize": "max",
            "level": "inference",
            "description": "Quality score",
        })
    );

    // Optimizers.
    assert_that!(count_rows(&pool, "optimizers_configs").await, eq(1));
    assert_that!(
        fetch_named_config(&pool, "optimizers_configs", "dicl_optimizer").await,
        partially(matches_json_literal!({
            "type": "dicl",
            "embedding_model": "embed_provider",
            "variant_name": "variant_a",
            "function_name": "function_a",
            "k": 5,
        }))
    );
}

/// Upserting a named-collection config with the same name should replace the
/// prior row rather than insert a duplicate.
#[sqlx::test(migrator = "tensorzero_stored_config::postgres::MIGRATOR")]
async fn write_stored_config_named_collections_upsert_by_name(pool: PgPool) {
    let postgres = PostgresConnectionInfo::new_with_pool(pool.clone());

    let config_v1 = UninitializedConfig {
        metrics: Some(HashMap::from([(
            "quality".to_string(),
            MetricConfig {
                r#type: MetricConfigType::Boolean,
                optimize: MetricConfigOptimize::Min,
                level: MetricConfigLevel::Episode,
                description: Some("v1".to_string()),
            },
        )])),
        ..Default::default()
    };
    postgres
        .write_stored_config(default_write_params(&config_v1))
        .await
        .expect("first write should succeed");

    let config_v2 = UninitializedConfig {
        metrics: Some(HashMap::from([(
            "quality".to_string(),
            MetricConfig {
                r#type: MetricConfigType::Float,
                optimize: MetricConfigOptimize::Max,
                level: MetricConfigLevel::Inference,
                description: Some("v2".to_string()),
            },
        )])),
        ..Default::default()
    };
    postgres
        .write_stored_config(default_write_params(&config_v2))
        .await
        .expect("second write should succeed");

    assert_that!(count_rows(&pool, "metrics_configs").await, eq(1));
    assert_that!(
        fetch_named_config(&pool, "metrics_configs", "quality").await,
        matches_json_literal!({
            "type": "float",
            "optimize": "max",
            "level": "inference",
            "description": "v2",
        })
    );
}

// ── Tests: tools with stored files ────────────────────────────────────────────

/// Writing a tool should persist the tool row referencing a stored file
/// row that contains the tool's parameters schema.
#[sqlx::test(migrator = "tensorzero_stored_config::postgres::MIGRATOR")]
async fn write_stored_config_persists_tool_with_file(pool: PgPool) {
    let postgres = PostgresConnectionInfo::new_with_pool(pool.clone());

    let tool = UninitializedToolConfig {
        description: "Look up weather".to_string(),
        parameters: fake_template(
            "tools.get_weather.parameters",
            "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}}}",
        ),
        name: Some("get_weather".to_string()),
        strict: true,
    };
    let config = UninitializedConfig {
        tools: Some(HashMap::from([("get_weather".to_string(), tool)])),
        ..Default::default()
    };

    postgres
        .write_stored_config(default_write_params(&config))
        .await
        .expect("tool write should succeed");

    assert_that!(count_rows(&pool, "tools_configs").await, eq(1));
    assert_that!(count_rows(&pool, "stored_files").await, eq(1));

    // The tool's `parameters.file_version_id` is a freshly-generated UUID,
    // so we only assert on the fields we can predict.
    assert_that!(
        fetch_named_config(&pool, "tools_configs", "get_weather").await,
        partially(matches_json_literal!({
            "description": "Look up weather",
            "name": "get_weather",
            "strict": true,
            "parameters": {
                "file_path": "tools.get_weather.parameters",
            },
        }))
    );

    // Verify the stored file row exists for the referenced path and points
    // back to the same `file_version_id` the tool stored.
    let stored_tool = fetch_named_config(&pool, "tools_configs", "get_weather").await;
    let template_version_id = stored_tool
        .get("parameters")
        .and_then(|v| v.get("file_version_id"))
        .and_then(serde_json::Value::as_str)
        .expect("tool should reference a stored file version id");
    let template_row = sqlx::query(
        "SELECT id, file_path, source_body FROM tensorzero.stored_files \
         WHERE file_path = $1",
    )
    .bind("tools.get_weather.parameters")
    .fetch_one(&pool)
    .await
    .expect("stored file row should exist");
    let template_id: uuid::Uuid = template_row.get("id");
    assert_that!(template_id.to_string(), eq(template_version_id));
    let source_body: String = template_row.get("source_body");
    assert_that!(source_body, contains_substring("\"city\""));
}

// ── Tests: evaluations with stored files ──────────────────────────────────────

/// Writing an evaluation with an LLM-judge variant should persist the
/// evaluation row and one stored file row per variant.
#[sqlx::test(migrator = "tensorzero_stored_config::postgres::MIGRATOR")]
async fn write_stored_config_persists_evaluation_with_file(pool: PgPool) {
    let postgres = PostgresConnectionInfo::new_with_pool(pool.clone());

    // `UninitializedEvaluationConfig` has a custom deserializer that reads
    // `type = "inference"` and the evaluator keys inline, so JSON is the
    // cleanest way to build one from an integration test.
    let evaluation: UninitializedEvaluationConfig = deserialize_from_json(serde_json::json!({
        "type": "inference",
        "function_name": "basic_test",
        "description": "End-to-end evaluation",
        "evaluators": {
            "judge": {
                "type": "llm_judge",
                "input_format": "serialized",
                "output_type": "boolean",
                "optimize": "max",
                "variants": {
                    "judge_chat": {
                        "type": "chat_completion",
                        "active": true,
                        "model": "openai::gpt-5-mini",
                        "system_instructions": {
                            "__tensorzero_remapped_path":
                                "evaluations.my_eval.evaluators.judge.variants.judge_chat.system_instructions",
                            "__data": "Judge this response carefully"
                        },
                        "json_mode": "strict"
                    }
                }
            }
        }
    }));

    // The evaluation references `basic_test`, so the full-config validation
    // performed at the top of `write_stored_config` requires the function to
    // exist. A minimal chat function with a shortcut-model variant is enough.
    let basic_test: UninitializedFunctionConfig = deserialize_from_json(serde_json::json!({
        "type": "chat",
        "variants": {
            "default": {
                "type": "chat_completion",
                "model": "dummy::good",
            }
        }
    }));

    let config = UninitializedConfig {
        functions: Some(HashMap::from([("basic_test".to_string(), basic_test)])),
        evaluations: Some(HashMap::from([("my_eval".to_string(), evaluation)])),
        ..Default::default()
    };

    postgres
        .write_stored_config(default_write_params(&config))
        .await
        .expect("evaluation write should succeed");

    assert_that!(count_rows(&pool, "evaluations_configs").await, eq(1));
    assert_that!(count_rows(&pool, "stored_files").await, eq(1));

    assert_that!(
        fetch_named_config(&pool, "evaluations_configs", "my_eval").await,
        partially(matches_json_literal!({
            "type": "inference",
            "function_name": "basic_test",
            "description": "End-to-end evaluation",
            "evaluators": {
                "judge": {
                    "type": "llm_judge",
                    "input_format": "serialized",
                    "output_type": "boolean",
                    "optimize": "max",
                    "variants": {
                        "judge_chat": {
                            "variant": {
                                "type": "chat_completion",
                                "active": true,
                                "model": "openai::gpt-5-mini",
                                "json_mode": "strict",
                                "system_instructions": {
                                    "file_path": "evaluations.my_eval.evaluators.judge.variants.judge_chat.system_instructions",
                                },
                            },
                        },
                    },
                },
            },
        }))
    );

    let template_row = sqlx::query("SELECT file_path, source_body FROM tensorzero.stored_files")
        .fetch_one(&pool)
        .await
        .expect("stored file row should exist");
    let file_path: String = template_row.get("file_path");
    assert_that!(
        file_path,
        eq("evaluations.my_eval.evaluators.judge.variants.judge_chat.system_instructions")
    );
    let source_body: String = template_row.get("source_body");
    assert_that!(source_body, eq("Judge this response carefully"));
}

// ── Tests: read-back via load_config_from_db ──────────────────────────────────

/// Write a representative `UninitializedConfig` covering one singleton, one
/// named-collection entry of each kind, plus a tool with a stored file,
/// and assert that `load_config_from_db` returns those values.
#[sqlx::test(migrator = "tensorzero_stored_config::postgres::MIGRATOR")]
async fn write_stored_config_round_trips_via_load_config_from_db(pool: PgPool) {
    let postgres = PostgresConnectionInfo::new_with_pool(pool.clone());

    let model: UninitializedModelConfig = deserialize_from_json(serde_json::json!({
        "routing": ["dummy_provider"],
        "providers": {
            "dummy_provider": {
                "type": "dummy",
                "model_name": "dummy-model"
            }
        }
    }));
    let embedding_model: UninitializedEmbeddingModelConfig =
        deserialize_from_json(serde_json::json!({
            "routing": ["embed_provider"],
            "providers": {
                "embed_provider": {
                    "type": "openai",
                    "model_name": "text-embedding-3-large"
                }
            }
        }));
    let metric = MetricConfig {
        r#type: MetricConfigType::Float,
        optimize: MetricConfigOptimize::Max,
        level: MetricConfigLevel::Inference,
        description: Some("Quality score".to_string()),
    };
    let tool = UninitializedToolConfig {
        description: "Look up weather".to_string(),
        parameters: fake_template(
            "tools.get_weather.parameters",
            "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}}}",
        ),
        name: Some("get_weather".to_string()),
        strict: true,
    };

    #[expect(deprecated)]
    let config = UninitializedConfig {
        clickhouse: Some(ClickHouseConfig {
            disable_automatic_migrations: Some(true),
        }),
        postgres: Some(PostgresConfig {
            enabled: None,
            connection_pool_size: Some(17),
            inference_metadata_retention_days: Some(30),
            inference_data_retention_days: Some(7),
        }),
        object_storage: Some(StorageKind::Disabled),
        autopilot: Some(AutopilotConfig {
            tool_whitelist: Some(vec!["ls".to_string(), "cat".to_string()]),
        }),
        models: Some(HashMap::from([(
            std::sync::Arc::<str>::from("model_a"),
            model.clone(),
        )])),
        embedding_models: Some(HashMap::from([(
            std::sync::Arc::<str>::from("embed_a"),
            embedding_model.clone(),
        )])),
        metrics: Some(HashMap::from([("quality".to_string(), metric.clone())])),
        tools: Some(HashMap::from([("get_weather".to_string(), tool.clone())])),
        ..Default::default()
    };

    postgres
        .write_stored_config(default_write_params(&config))
        .await
        .expect("write should succeed");

    let loaded = load_config_from_db(&pool)
        .await
        .expect("DB load should succeed");

    assert_that!(
        loaded
            .clickhouse
            .as_ref()
            .and_then(|c| c.disable_automatic_migrations),
        some(eq(true))
    );
    assert_that!(
        loaded
            .postgres
            .as_ref()
            .and_then(|p| p.connection_pool_size),
        some(eq(17))
    );
    assert_that!(
        loaded.object_storage.as_ref(),
        some(eq(&StorageKind::Disabled))
    );
    assert_that!(
        loaded
            .autopilot
            .as_ref()
            .and_then(|a| a.tool_whitelist.as_deref()),
        some(elements_are![eq("ls"), eq("cat")])
    );
    assert_that!(
        loaded
            .models
            .as_ref()
            .and_then(|m| m.get(&std::sync::Arc::<str>::from("model_a"))),
        some(eq(&model))
    );
    assert_that!(
        loaded
            .embedding_models
            .as_ref()
            .and_then(|m| m.get(&std::sync::Arc::<str>::from("embed_a"))),
        some(eq(&embedding_model))
    );
    assert_that!(
        loaded.metrics.as_ref().and_then(|m| m.get("quality")),
        some(eq(&metric))
    );
    assert_that!(
        loaded.tools.as_ref().and_then(|t| t.get("get_weather")),
        some(eq(&tool))
    );
}

// ── Tests: advisory lock ──────────────────────────────────────────────────────

/// `write_stored_config` should fail with a clear error if another client is
/// holding the global stored-config advisory lock. This verifies that the
/// lock acquisition path in `write_stored_config_in_tx` actually serializes
/// concurrent writers via `pg_try_advisory_xact_lock`.
#[sqlx::test(migrator = "tensorzero_stored_config::postgres::MIGRATOR")]
async fn write_stored_config_fails_when_advisory_lock_held(pool: PgPool) {
    let postgres = PostgresConnectionInfo::new_with_pool(pool.clone());

    // Reproduce the lock key: BLAKE3 hash of `tensorzero::stored_config::global`,
    // first 8 bytes as little-endian i64.
    let hash = blake3::hash(b"tensorzero::stored_config::global");
    let lock_key_bytes: [u8; 8] = hash.as_bytes()[..8]
        .try_into()
        .expect("BLAKE3 output is always 32 bytes");
    let lock_key = i64::from_le_bytes(lock_key_bytes);

    // Hold the lock from a separate transaction on a separate connection.
    let mut blocking_tx = pool
        .begin()
        .await
        .expect("blocking transaction should start");
    let acquired: bool = sqlx::query_scalar("SELECT pg_try_advisory_xact_lock($1)")
        .bind(lock_key)
        .fetch_one(&mut *blocking_tx)
        .await
        .expect("blocking lock acquisition should succeed");
    assert_that!(acquired, eq(true));

    // While the lock is held elsewhere, the writer should fail fast rather
    // than block (we use `pg_try_advisory_xact_lock`).
    let config = UninitializedConfig::default();
    let err = postgres
        .write_stored_config(default_write_params(&config))
        .await
        .expect_err("write should fail while another client holds the lock");
    assert_that!(
        format!("{err}"),
        contains_substring("another client is writing the config")
    );

    // Releasing the lock (by ending the blocking transaction) should let a
    // subsequent write succeed.
    blocking_tx
        .rollback()
        .await
        .expect("rollback should release the advisory lock");
    postgres
        .write_stored_config(default_write_params(&config))
        .await
        .expect("write should succeed once the lock is released");
}
