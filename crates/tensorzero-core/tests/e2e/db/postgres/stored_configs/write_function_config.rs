//! E2E tests for writing function configs to Postgres (config-in-db).

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use googletest::assert_that;
use googletest::matchers::*;
use serde::de::DeserializeOwned;
use sqlx::{PgPool, Row};
use tensorzero_types::inference_params::{JsonMode, ServiceTier};
use uuid::Uuid;

use tensorzero_core::config::{
    Namespace, UninitializedFunctionConfig, UninitializedFunctionConfigJson, UninitializedSchemas,
    UninitializedVariantConfig, UninitializedVariantInfo, path::ResolvedTomlPathData,
};
use tensorzero_core::config::{NonStreamingTimeouts, StreamingTimeouts, TimeoutsConfig};
use tensorzero_core::db::postgres::PostgresConnectionInfo;
use tensorzero_core::db::postgres::function_config_writes::WriteFunctionConfigParams;
use tensorzero_core::db::postgres::stored_config_queries::load_config_from_db;
use tensorzero_core::evaluations::{
    ExactMatchConfig, LLMJudgeIncludeConfig, LLMJudgeInputFormat, LLMJudgeOptimize,
    LLMJudgeOutputType, UninitializedEvaluatorConfig,
    UninitializedLLMJudgeChatCompletionVariantConfig, UninitializedLLMJudgeConfig,
    UninitializedLLMJudgeDiclVariantConfig, UninitializedLLMJudgeVariantConfig,
    UninitializedLLMJudgeVariantInfo,
};
use tensorzero_core::experimentation::adaptive_experimentation::{
    AdaptiveExperimentationAlgorithm, UninitializedAdaptiveExperimentationConfig,
};
use tensorzero_core::experimentation::{
    StaticExperimentationConfig, UninitializedExperimentationConfig,
    UninitializedExperimentationConfigWithNamespaces,
};
use tensorzero_core::inference::types::extra_body::{
    ExtraBodyConfig, ExtraBodyReplacement, ExtraBodyReplacementKind,
};
use tensorzero_core::inference::types::extra_headers::{
    ExtraHeader, ExtraHeaderKind, ExtraHeadersConfig,
};
use tensorzero_core::utils::retries::RetryConfig;
use tensorzero_core::variant::best_of_n_sampling::{
    UninitializedBestOfNEvaluatorConfig, UninitializedBestOfNSamplingConfig,
};
use tensorzero_core::variant::chain_of_thought::UninitializedChainOfThoughtConfig;
use tensorzero_core::variant::chat_completion::{
    UninitializedChatCompletionConfig, UninitializedChatTemplate,
};
use tensorzero_core::variant::dicl::UninitializedDiclConfig;
use tensorzero_core::variant::mixture_of_n::{
    UninitializedFuserConfig, UninitializedMixtureOfNConfig,
};
use tensorzero_stored_config::{
    StoredEvaluatorConfig, StoredFunctionConfig, StoredVariantConfig, StoredVariantVersionConfig,
};

fn fake_template(key: &str, body: &str) -> ResolvedTomlPathData {
    ResolvedTomlPathData::new_fake_path(key.to_string(), body.to_string())
}

fn deserialize_from_json<T: DeserializeOwned>(value: serde_json::Value) -> T {
    serde_json::from_value(value).expect("test value should deserialize")
}

fn sample_input_wrappers() -> tensorzero_core::variant::chat_completion::UninitializedInputWrappers
{
    deserialize_from_json(serde_json::json!({
        "user": {
            "__tensorzero_remapped_path": "functions.test.variants.chat.input_wrappers.user",
            "__data": "[[{{ input }}]]"
        },
        "system": {
            "__tensorzero_remapped_path": "functions.test.variants.chat.input_wrappers.system",
            "__data": "<<{{ system }}>>"
        }
    }))
}

fn sample_track_and_stop_config()
-> tensorzero_core::experimentation::track_and_stop::UninitializedTrackAndStopExperimentationConfig
{
    deserialize_from_json(serde_json::json!({
        "metric": "quality",
        "candidate_variants": ["chat", "cot"],
        "fallback_variants": ["dicl"],
        "min_samples_per_variant": 12,
        "delta": 0.1,
        "epsilon": 0.01,
        "update_period_s": 60,
        "min_prob": 0.02,
        "max_samples_per_variant": 1000
    }))
}

fn sample_extra_body() -> ExtraBodyConfig {
    ExtraBodyConfig {
        data: vec![ExtraBodyReplacement {
            pointer: "/temperature".to_string(),
            kind: ExtraBodyReplacementKind::Value(serde_json::json!(0.2)),
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

fn sample_chat_completion(
    system_template: ResolvedTomlPathData,
) -> UninitializedChatCompletionConfig {
    UninitializedChatCompletionConfig {
        weight: Some(1.0),
        model: Arc::<str>::from("openai::gpt-5"),
        system_template: Some(system_template),
        user_template: Some(fake_template(
            "functions.test.variants.chat.user_template",
            "User {{ input }}",
        )),
        assistant_template: Some(fake_template(
            "functions.test.variants.chat.assistant_template",
            "Assistant {{ response }}",
        )),
        input_wrappers: Some(sample_input_wrappers()),
        templates: tensorzero_core::variant::chat_completion::UninitializedChatTemplates {
            inner: HashMap::from([
                (
                    "body".to_string(),
                    UninitializedChatTemplate {
                        path: fake_template(
                            "functions.test.variants.chat.templates.body",
                            "Body {{ body }}",
                        ),
                    },
                ),
                (
                    "shared_header".to_string(),
                    UninitializedChatTemplate {
                        path: fake_template("shared/header", "Shared header"),
                    },
                ),
            ]),
        },
        temperature: Some(0.1),
        top_p: Some(0.9),
        max_tokens: Some(512),
        presence_penalty: Some(0.2),
        frequency_penalty: Some(0.3),
        seed: Some(42),
        stop_sequences: Some(vec!["STOP".to_string()]),
        reasoning_effort: Some("high".to_string()),
        service_tier: Some(ServiceTier::Auto),
        thinking_budget_tokens: Some(128),
        verbosity: Some("medium".to_string()),
        json_mode: Some(JsonMode::Strict),
        retries: RetryConfig {
            num_retries: 2,
            max_delay_s: 1.5,
        },
        extra_body: Some(sample_extra_body()),
        extra_headers: Some(sample_extra_headers()),
    }
}

#[expect(deprecated)]
fn sample_json_function() -> UninitializedFunctionConfig {
    let chat_system = fake_template(
        "functions.test.variants.chat.system_template",
        "{% include 'shared/header' %}\nSystem {{ input }}",
    );
    let chat_completion = sample_chat_completion(chat_system);
    UninitializedFunctionConfig::Json(UninitializedFunctionConfigJson {
        variants: HashMap::from([
            (
                "chat".to_string(),
                UninitializedVariantInfo {
                    inner: UninitializedVariantConfig::ChatCompletion(chat_completion.clone()),
                    timeouts: Some(TimeoutsConfig {
                        non_streaming: Some(NonStreamingTimeouts {
                            total_ms: Some(5000),
                        }),
                        streaming: Some(StreamingTimeouts {
                            ttft_ms: Some(1000),
                            total_ms: Some(8000),
                        }),
                    }),
                    namespace: Some(Namespace::new("alpha").expect("namespace should be valid")),
                },
            ),
            (
                "best".to_string(),
                UninitializedVariantInfo {
                    inner: UninitializedVariantConfig::BestOfNSampling(
                        UninitializedBestOfNSamplingConfig {
                            weight: Some(0.5),
                            timeout_s: Some(5.0),
                            candidates: vec!["chat".to_string(), "dicl".to_string()],
                            evaluator: UninitializedBestOfNEvaluatorConfig {
                                inner: sample_chat_completion(fake_template(
                                    "functions.test.variants.best.evaluator.system_template",
                                    "Judge best",
                                )),
                            },
                        },
                    ),
                    timeouts: None,
                    namespace: None,
                },
            ),
            (
                "mixture".to_string(),
                UninitializedVariantInfo {
                    inner: UninitializedVariantConfig::MixtureOfN(UninitializedMixtureOfNConfig {
                        weight: Some(0.25),
                        timeout_s: Some(7.0),
                        candidates: vec!["chat".to_string(), "best".to_string()],
                        fuser: UninitializedFuserConfig {
                            inner: sample_chat_completion(fake_template(
                                "functions.test.variants.mixture.fuser.system_template",
                                "Fuse mixture",
                            )),
                        },
                    }),
                    timeouts: None,
                    namespace: None,
                },
            ),
            (
                "dicl".to_string(),
                UninitializedVariantInfo {
                    inner: UninitializedVariantConfig::Dicl(UninitializedDiclConfig {
                        weight: Some(0.1),
                        embedding_model: "openai::text-embedding-3-large".to_string(),
                        k: 4,
                        model: "openai::gpt-5-mini".to_string(),
                        system_instructions: Some(fake_template(
                            "functions.test.variants.dicl.system_instructions",
                            "DICL system",
                        )),
                        temperature: Some(0.4),
                        top_p: Some(0.8),
                        stop_sequences: Some(vec!["DONE".to_string()]),
                        presence_penalty: Some(0.1),
                        frequency_penalty: Some(0.2),
                        max_tokens: Some(256),
                        seed: Some(11),
                        reasoning_effort: Some("medium".to_string()),
                        thinking_budget_tokens: Some(64),
                        verbosity: Some("low".to_string()),
                        json_mode: Some(JsonMode::On),
                        extra_body: Some(sample_extra_body()),
                        retries: RetryConfig {
                            num_retries: 1,
                            max_delay_s: 2.0,
                        },
                        extra_headers: Some(sample_extra_headers()),
                        max_distance: Some(0.7),
                    }),
                    timeouts: None,
                    namespace: None,
                },
            ),
            (
                "cot".to_string(),
                UninitializedVariantInfo {
                    inner: UninitializedVariantConfig::ChainOfThought(
                        UninitializedChainOfThoughtConfig {
                            inner: sample_chat_completion(fake_template(
                                "functions.test.variants.cot.system_template",
                                "CoT system",
                            )),
                        },
                    ),
                    timeouts: None,
                    namespace: None,
                },
            ),
        ]),
        system_schema: Some(fake_template(
            "functions.test.system_schema",
            "{\"type\":\"string\"}",
        )),
        user_schema: Some(fake_template(
            "functions.test.user_schema",
            "{\"type\":\"string\"}",
        )),
        assistant_schema: None,
        schemas: UninitializedSchemas::from_paths(HashMap::from([(
            "custom".to_string(),
            fake_template("functions.test.schemas.custom", "{\"type\":\"number\"}"),
        )])),
        output_schema: Some(fake_template(
            "functions.test.output_schema",
            "{\"type\":\"object\",\"additionalProperties\":false}",
        )),
        description: Some("JSON test function".to_string()),
        experimentation: Some(UninitializedExperimentationConfigWithNamespaces {
            base: UninitializedExperimentationConfig::Static(StaticExperimentationConfig {
                candidate_variants: tensorzero_core::experimentation::WeightedVariants::from_map(
                    BTreeMap::from([("chat".to_string(), 0.7), ("best".to_string(), 0.3)]),
                ),
                fallback_variants: vec!["dicl".to_string()],
            }),
            namespaces: HashMap::from([(
                "beta".to_string(),
                UninitializedExperimentationConfig::Adaptive(
                    UninitializedAdaptiveExperimentationConfig {
                        algorithm: Some(AdaptiveExperimentationAlgorithm::TrackAndStop),
                        inner: sample_track_and_stop_config(),
                    },
                ),
            )]),
        }),
        evaluators: HashMap::from([
            (
                "exact".to_string(),
                UninitializedEvaluatorConfig::ExactMatch(ExactMatchConfig { cutoff: Some(0.9) }),
            ),
            (
                "judge".to_string(),
                UninitializedEvaluatorConfig::LLMJudge(UninitializedLLMJudgeConfig {
                    input_format: Some(LLMJudgeInputFormat::Serialized),
                    variants: HashMap::from([
                        (
                            "judge_chat".to_string(),
                            UninitializedLLMJudgeVariantInfo {
                                inner: UninitializedLLMJudgeVariantConfig::ChatCompletion(
                                    UninitializedLLMJudgeChatCompletionVariantConfig {
                                        active: Some(true),
                                        model: Arc::<str>::from("openai::gpt-5-mini"),
                                        system_instructions: fake_template(
                                            "functions.test.evaluators.judge.variants.judge_chat.system_instructions",
                                            "Judge system",
                                        ),
                                        temperature: Some(0.2),
                                        top_p: Some(0.7),
                                        max_tokens: Some(128),
                                        presence_penalty: Some(0.1),
                                        frequency_penalty: Some(0.2),
                                        seed: Some(9),
                                        json_mode: JsonMode::Strict,
                                        stop_sequences: Some(vec!["END".to_string()]),
                                        reasoning_effort: Some("high".to_string()),
                                        service_tier: Some(ServiceTier::Default),
                                        thinking_budget_tokens: Some(32),
                                        verbosity: Some("high".to_string()),
                                        retries: RetryConfig {
                                            num_retries: 3,
                                            max_delay_s: 4.0,
                                        },
                                        extra_body: Some(sample_extra_body()),
                                        extra_headers: Some(sample_extra_headers()),
                                    },
                                ),
                                timeouts: None,
                            },
                        ),
                        (
                            "judge_dicl".to_string(),
                            UninitializedLLMJudgeVariantInfo {
                                inner: UninitializedLLMJudgeVariantConfig::Dicl(
                                    UninitializedLLMJudgeDiclVariantConfig {
                                        active: Some(false),
                                        embedding_model: "openai::text-embedding-3-large"
                                            .to_string(),
                                        k: 3,
                                        model: "openai::gpt-5-mini".to_string(),
                                        system_instructions: Some(fake_template(
                                            "functions.test.evaluators.judge.variants.judge_dicl.system_instructions",
                                            "Judge DICL system",
                                        )),
                                        temperature: Some(0.4),
                                        top_p: Some(0.8),
                                        presence_penalty: Some(0.1),
                                        frequency_penalty: Some(0.2),
                                        max_tokens: Some(64),
                                        seed: Some(7),
                                        json_mode: Some(JsonMode::On),
                                        stop_sequences: Some(vec!["STOP".to_string()]),
                                        extra_body: Some(sample_extra_body()),
                                        retries: RetryConfig {
                                            num_retries: 1,
                                            max_delay_s: 2.5,
                                        },
                                        extra_headers: Some(sample_extra_headers()),
                                    },
                                ),
                                timeouts: Some(TimeoutsConfig {
                                    non_streaming: Some(NonStreamingTimeouts {
                                        total_ms: Some(3000),
                                    }),
                                    streaming: Some(StreamingTimeouts::default()),
                                }),
                            },
                        ),
                    ]),
                    output_type: LLMJudgeOutputType::Boolean,
                    optimize: LLMJudgeOptimize::Max,
                    include: Some(LLMJudgeIncludeConfig {
                        reference_output: true,
                    }),
                    cutoff: Some(0.75),
                    description: Some("Judge config".to_string()),
                }),
            ),
        ]),
    })
}

async fn load_function_version(pool: &PgPool, id: Uuid) -> StoredFunctionConfig {
    let row = sqlx::query("SELECT config FROM tensorzero.function_configs WHERE id = $1")
        .bind(id)
        .fetch_one(pool)
        .await
        .expect("function version should exist");
    serde_json::from_value(row.get("config")).expect("stored function config should deserialize")
}

async fn load_variant_version(pool: &PgPool, id: Uuid) -> StoredVariantVersionConfig {
    let row = sqlx::query("SELECT config FROM tensorzero.variant_configs WHERE id = $1")
        .bind(id)
        .fetch_one(pool)
        .await
        .expect("variant version should exist");
    serde_json::from_value(row.get("config")).expect("stored variant config should deserialize")
}

#[sqlx::test(migrator = "tensorzero_stored_config::postgres::MIGRATOR")]
async fn write_function_config_persists_expected_rows(pool: PgPool) {
    let postgres = PostgresConnectionInfo::new_with_pool(pool.clone());
    let config = sample_json_function();
    let result = postgres
        .write_function_config(WriteFunctionConfigParams {
            function_name: "test",
            config: &config,
            expected_current_version_id: None,
            creation_source: "ui",
            source_autopilot_session_id: None,
            extra_templates: &HashMap::new(),
        })
        .await
        .expect("write path should succeed");

    let stored_function = load_function_version(&pool, result.function_version_id).await;
    let StoredFunctionConfig::Json(stored_function) = stored_function else {
        panic!("expected JSON function config");
    };
    assert_that!(
        stored_function.description.as_deref(),
        some(eq("JSON test function"))
    );
    assert_that!(
        stored_function.variants.as_ref().map(BTreeMap::len),
        some(eq(5))
    );
    assert_that!(
        stored_function.schemas.as_ref().map(BTreeMap::len),
        some(eq(1))
    );
    assert_that!(
        stored_function
            .output_schema
            .as_ref()
            .map(|schema| schema.file_path.as_str()),
        some(eq("functions.test.output_schema"))
    );
    assert_that!(
        stored_function.evaluators.as_ref().map(BTreeMap::len),
        some(eq(2))
    );
    let judge = match stored_function
        .evaluators
        .as_ref()
        .and_then(|evaluators| evaluators.get("judge"))
    {
        Some(StoredEvaluatorConfig::LLMJudge(judge)) => judge,
        _ => panic!("expected stored LLM judge config"),
    };
    assert_that!(judge.variants.as_ref().map(BTreeMap::len), some(eq(2)));
    assert_that!(
        judge
            .include
            .as_ref()
            .map(|include| include.reference_output),
        some(eq(true))
    );

    let chat_variant_id = *result
        .variant_version_ids
        .get("chat")
        .expect("chat variant should be written");
    let chat_variant = load_variant_version(&pool, chat_variant_id).await;
    let StoredVariantConfig::ChatCompletion(chat_variant) = chat_variant.variant else {
        panic!("expected chat completion variant");
    };
    assert_that!(
        chat_variant
            .system_template
            .as_ref()
            .map(|template| template.file_path.as_str()),
        some(eq("functions.test.variants.chat.system_template"))
    );
    assert_that!(
        chat_variant.templates.as_ref().map(BTreeMap::len),
        some(eq(2))
    );
    assert_that!(
        chat_variant.retries.as_ref().map(|retry| retry.num_retries),
        some(eq(2))
    );

    let prompt_count: i64 =
        sqlx::query_scalar("SELECT COUNT(*)::BIGINT FROM tensorzero.stored_files")
            .fetch_one(&pool)
            .await
            .expect("prompt count query should succeed");
    assert_that!(prompt_count, ge(12));
}

/// Write a function config and then read the whole DB back via
/// `load_config_from_db`, asserting that the loaded `UninitializedConfig`
/// contains a function equal to the one we wrote.
#[sqlx::test(migrator = "tensorzero_stored_config::postgres::MIGRATOR")]
async fn write_function_config_round_trips_via_load_config_from_db(pool: PgPool) {
    let postgres = PostgresConnectionInfo::new_with_pool(pool.clone());
    let config = sample_json_function();
    postgres
        .write_function_config(WriteFunctionConfigParams {
            function_name: "test",
            config: &config,
            expected_current_version_id: None,
            creation_source: "ui",
            source_autopilot_session_id: None,
            extra_templates: &HashMap::new(),
        })
        .await
        .expect("write should succeed");

    let loaded = load_config_from_db(&pool)
        .await
        .expect("DB load should succeed");

    assert_that!(
        loaded.functions.as_ref().and_then(|f| f.get("test")),
        some(eq(&sample_json_function()))
    );
}

#[sqlx::test(migrator = "tensorzero_stored_config::postgres::MIGRATOR")]
async fn write_function_config_compare_and_swap(pool: PgPool) {
    let postgres = PostgresConnectionInfo::new_with_pool(pool);
    let config = sample_json_function();

    // Initial write with no expected version (first write for this function).
    let first_result = postgres
        .write_function_config(WriteFunctionConfigParams {
            function_name: "test",
            config: &config,
            expected_current_version_id: None,
            creation_source: "ui",
            source_autopilot_session_id: None,
            extra_templates: &HashMap::new(),
        })
        .await
        .expect("initial write should succeed");
    let first_version_id = first_result.function_version_id;

    // Second write with the correct expected version succeeds.
    let second_result = postgres
        .write_function_config(WriteFunctionConfigParams {
            function_name: "test",
            config: &config,
            expected_current_version_id: Some(first_version_id),
            creation_source: "ui",
            source_autopilot_session_id: None,
            extra_templates: &HashMap::new(),
        })
        .await
        .expect("write with correct expected version should succeed");
    assert_that!(second_result.function_version_id, not(eq(first_version_id)));

    // Third write with the stale first version ID fails.
    let error = postgres
        .write_function_config(WriteFunctionConfigParams {
            function_name: "test",
            config: &config,
            expected_current_version_id: Some(first_version_id),
            creation_source: "ui",
            source_autopilot_session_id: None,
            extra_templates: &HashMap::new(),
        })
        .await
        .expect_err("write with stale expected version should fail");
    assert_that!(
        error.to_string(),
        contains_substring("updated during your edit")
    );
}

/// Write a function with 2 variants, then add a third variant and update experimentation.
/// The two original variants should be reused (same IDs via content-hash dedup),
/// while a new variant and a new function version are created.
#[sqlx::test(migrator = "tensorzero_stored_config::postgres::MIGRATOR")]
async fn write_function_config_reuses_unchanged_variants(pool: PgPool) {
    let postgres = PostgresConnectionInfo::new_with_pool(pool.clone());

    // Build a simple JSON function with 2 chat completion variants.
    let variant_a = UninitializedVariantInfo {
        inner: UninitializedVariantConfig::ChatCompletion(sample_chat_completion(fake_template(
            "functions.reuse.variants.a.system_template",
            "Variant A system",
        ))),
        timeouts: None,
        namespace: None,
    };
    let variant_b = UninitializedVariantInfo {
        inner: UninitializedVariantConfig::ChatCompletion(sample_chat_completion(fake_template(
            "functions.reuse.variants.b.system_template",
            "Variant B system",
        ))),
        timeouts: None,
        namespace: None,
    };

    let config_v1 = UninitializedFunctionConfig::Json(UninitializedFunctionConfigJson {
        variants: HashMap::from([
            ("a".to_string(), variant_a.clone()),
            ("b".to_string(), variant_b.clone()),
        ]),
        system_schema: None,
        user_schema: None,
        assistant_schema: None,
        schemas: UninitializedSchemas::default(),
        output_schema: None,
        description: Some("reuse test v1".to_string()),
        experimentation: None,
        evaluators: HashMap::new(),
    });

    let result_v1 = postgres
        .write_function_config(WriteFunctionConfigParams {
            function_name: "reuse_test",
            config: &config_v1,
            expected_current_version_id: None,
            creation_source: "ui",
            source_autopilot_session_id: None,
            extra_templates: &HashMap::new(),
        })
        .await
        .expect("v1 write should succeed");

    let variant_a_id_v1 = *result_v1
        .variant_version_ids
        .get("a")
        .expect("variant a should be written");
    let variant_b_id_v1 = *result_v1
        .variant_version_ids
        .get("b")
        .expect("variant b should be written");

    // Add a third variant and set up experimentation referencing all three.
    let variant_c = UninitializedVariantInfo {
        inner: UninitializedVariantConfig::ChatCompletion(sample_chat_completion(fake_template(
            "functions.reuse.variants.c.system_template",
            "Variant C system",
        ))),
        timeouts: None,
        namespace: None,
    };

    let config_v2 = UninitializedFunctionConfig::Json(UninitializedFunctionConfigJson {
        variants: HashMap::from([
            ("a".to_string(), variant_a),
            ("b".to_string(), variant_b),
            ("c".to_string(), variant_c),
        ]),
        system_schema: None,
        user_schema: None,
        assistant_schema: None,
        schemas: UninitializedSchemas::default(),
        output_schema: None,
        description: Some("reuse test v2".to_string()),
        experimentation: Some(UninitializedExperimentationConfigWithNamespaces {
            base: UninitializedExperimentationConfig::Static(StaticExperimentationConfig {
                candidate_variants: tensorzero_core::experimentation::WeightedVariants::from_map(
                    BTreeMap::from([
                        ("a".to_string(), 0.4),
                        ("b".to_string(), 0.3),
                        ("c".to_string(), 0.3),
                    ]),
                ),
                fallback_variants: vec!["a".to_string()],
            }),
            namespaces: HashMap::new(),
        }),
        evaluators: HashMap::new(),
    });

    let result_v2 = postgres
        .write_function_config(WriteFunctionConfigParams {
            function_name: "reuse_test",
            config: &config_v2,
            expected_current_version_id: Some(result_v1.function_version_id),
            creation_source: "ui",
            source_autopilot_session_id: None,
            extra_templates: &HashMap::new(),
        })
        .await
        .expect("v2 write should succeed");

    // Unchanged variants should have the same IDs (content-hash dedup).
    assert_that!(
        result_v2.variant_version_ids.get("a").copied(),
        some(eq(variant_a_id_v1))
    );
    assert_that!(
        result_v2.variant_version_ids.get("b").copied(),
        some(eq(variant_b_id_v1))
    );

    // The new variant should have a different ID.
    let variant_c_id = *result_v2
        .variant_version_ids
        .get("c")
        .expect("variant c should be written");
    assert_that!(variant_c_id, not(eq(variant_a_id_v1)));
    assert_that!(variant_c_id, not(eq(variant_b_id_v1)));

    // A new function version should have been created.
    assert_that!(
        result_v2.function_version_id,
        not(eq(result_v1.function_version_id))
    );

    // Verify only 3 total variant rows exist for this function (not 5).
    let variant_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*)::BIGINT FROM tensorzero.variant_configs WHERE function_name = 'reuse_test'",
    )
    .fetch_one(&pool)
    .await
    .expect("variant count query should succeed");
    assert_that!(variant_count, eq(3));
}
