#![allow(clippy::panic, clippy::panic_in_result_fn)]
//! End-to-end tests for the TypeScript judge evaluator.
//!
//! These tests exercise the full config → evaluation pipeline: a TensorZero
//! config defines `[evaluations.*]` blocks whose evaluators are of type
//! `typescript`, each pointing at a `.ts` fixture. The test:
//!
//! 1. loads the real e2e config (which includes
//!    `tensorzero.functions.ts_judge_eval.toml`),
//! 2. inserts a minimal dataset against the `ts_judge_eval` chat function
//!    (backed by `dummy::good` — deterministic output, no API dependency),
//! 3. constructs a `TypescriptJudgeExecutor` from a real `RlmPool` +
//!    `TsCheckerPool`, and
//! 4. calls `run_evaluation_core_streaming` end-to-end, asserting the
//!    resulting `EvaluationUpdate` stream matches the expected outcome for
//!    each scenario (success, TypeScript exception, deliberate OOM).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use evaluations::evaluators::typescript_judge::TypescriptJudgeExecutor;
use evaluations::{
    ClientInferenceExecutor, EvaluationCoreArgs, EvaluationFunctionConfig, EvaluationVariant,
    run_evaluation_core_streaming,
    stats::{EvaluationInfo, EvaluationUpdate},
};
use tensorzero::test_helpers::make_embedded_gateway;
use tensorzero_core::cache::CacheEnabledMode;
use tensorzero_core::db::datasets::DatasetQueries;
use tensorzero_core::db::delegating_connection::DelegatingDatabaseConnection;
use tensorzero_core::db::stored_datapoint::{StoredChatInferenceDatapoint, StoredDatapoint};
use tensorzero_core::db::test_helpers::TestDatabaseHelpers;
use tensorzero_core::evaluations::EvaluationConfig;
use tokio::time::sleep;
use uuid::Uuid;

use crate::common::{
    build_test_ts_executor, get_config, init_tracing_for_tests, query_boolean_feedback,
    query_float_feedback,
};

/// Build a minimal chat datapoint for the `ts_judge_eval` function.
///
/// Input is a single "hello" user message. No schema/template — the function
/// definition in `tensorzero.functions.ts_judge_eval.toml` omits both.
fn make_datapoint(dataset_name: &str) -> StoredChatInferenceDatapoint {
    let input: tensorzero_core::inference::types::stored_input::StoredInput =
        serde_json::from_value(serde_json::json!({
            "messages": [
                {
                    "role": "user",
                    "content": [{ "type": "text", "text": "hello" }]
                }
            ]
        }))
        .expect("valid StoredInput");
    StoredChatInferenceDatapoint {
        dataset_name: dataset_name.to_string(),
        function_name: "ts_judge_eval".to_string(),
        id: Uuid::now_v7(),
        episode_id: Some(Uuid::now_v7()),
        input,
        output: None,
        tool_params: None,
        tags: None,
        is_custom: true,
        source_inference_id: None,
        staled_at: None,
        name: None,
        snapshot_hash: None,
        is_deleted: false,
        auxiliary: String::new(),
        updated_at: String::new(),
    }
}

/// Construct `EvaluationCoreArgs` referencing the named evaluation from the
/// loaded config, scoped to the single datapoint we inserted above.
fn core_args_for_evaluation(
    evaluation_name: &str,
    datapoint_id: Uuid,
    config: &tensorzero_core::config::Config,
    db: &DelegatingDatabaseConnection,
    tensorzero_client: tensorzero_core::client::Client,
    ts_executor: TypescriptJudgeExecutor,
) -> EvaluationCoreArgs {
    let evaluation_config = config
        .evaluations
        .get(evaluation_name)
        .unwrap_or_else(|| panic!("evaluation `{evaluation_name}` missing from config"));
    let EvaluationConfig::Inference(inference_config) = &**evaluation_config;
    let function_config = config
        .functions
        .get(&inference_config.function_name)
        .map(|f| EvaluationFunctionConfig::from(f.as_ref()))
        .expect("ts_judge_eval function should exist");
    EvaluationCoreArgs {
        inference_executor: Arc::new(ClientInferenceExecutor::new(tensorzero_client)),
        db: Arc::new(db.clone()),
        function_name: inference_config.function_name.clone(),
        function_config,
        evaluators: inference_config.evaluators.clone(),
        evaluation_name: Some(evaluation_name.to_string()),
        evaluation_run_id: Uuid::now_v7(),
        dataset_name: None,
        datapoint_ids: Some(vec![datapoint_id]),
        variant: EvaluationVariant::Name("dummy".to_string()),
        concurrency: 1,
        inference_cache: CacheEnabledMode::Off,
        tags: HashMap::new(),
        ts_executor,
    }
}

/// Run a single-datapoint evaluation end-to-end and collect all
/// `EvaluationUpdate::Success` payloads. Panics on any `Error` / `FatalError`
/// update (those represent pipeline failures unrelated to evaluator outcome —
/// e.g. inference errors — which we don't want to hide).
async fn run_single_evaluation(args: EvaluationCoreArgs) -> Vec<EvaluationInfo> {
    let result = run_evaluation_core_streaming(args, None, HashMap::new())
        .await
        .expect("run_evaluation_core_streaming should start successfully");
    let mut receiver = result.receiver;
    let mut successes = Vec::new();
    while let Some(update) = receiver.recv().await {
        match update {
            EvaluationUpdate::Success(info) => successes.push(info),
            EvaluationUpdate::Error(err) => {
                panic!("unexpected evaluation error: {}", err.message);
            }
            EvaluationUpdate::FatalError(msg) => {
                panic!("fatal evaluation error: {msg}");
            }
            EvaluationUpdate::RunInfo(_) => {}
        }
    }
    successes
}

// ────────────────────────────────────────────────────────────────────────────
// Success path
// ────────────────────────────────────────────────────────────────────────────

/// Happy path: both evaluators in `typescript_judge_success` run to
/// completion. `ts_always_true` returns `true`, `ts_output_length` returns
/// the character length of the dummy model's text block. Both should
/// persist feedback keyed by the evaluator metric names.
#[tokio::test(flavor = "multi_thread")]
async fn test_typescript_judge_success() {
    init_tracing_for_tests();
    let db = DelegatingDatabaseConnection::new_for_e2e_test().await;
    let config = get_config().await;
    let tensorzero_client = make_embedded_gateway().await;

    let dataset_name = format!("ts_judge_success-{}", Uuid::now_v7());
    let datapoint = make_datapoint(&dataset_name);
    let datapoint_id = datapoint.id;
    db.insert_datapoints(&[StoredDatapoint::Chat(datapoint)])
        .await
        .expect("insert_datapoints should succeed");
    db.flush_pending_writes().await;

    let ts_executor = build_test_ts_executor().await;
    let args = core_args_for_evaluation(
        "typescript_judge_success",
        datapoint_id,
        &config,
        &db,
        tensorzero_client,
        ts_executor,
    );

    let successes = run_single_evaluation(args).await;
    assert_eq!(
        successes.len(),
        1,
        "expected exactly one evaluated datapoint"
    );
    let info = &successes[0];

    assert!(
        info.evaluator_errors.is_empty(),
        "expected no evaluator errors on happy path, got: {:?}",
        info.evaluator_errors
    );

    // `ts_always_true` → JSON boolean `true`
    let always_true = info
        .evaluations
        .get("ts_always_true")
        .expect("ts_always_true result missing")
        .as_ref()
        .expect("ts_always_true should produce a value");
    assert_eq!(always_true, &serde_json::json!(true));

    // `ts_output_length` → positive float (length of dummy response text)
    let len_value = info
        .evaluations
        .get("ts_output_length")
        .expect("ts_output_length result missing")
        .as_ref()
        .expect("ts_output_length should produce a value");
    let len = len_value
        .as_f64()
        .expect("ts_output_length should be numeric");
    assert!(
        len > 0.0,
        "dummy model output should be non-empty, got {len}"
    );

    // Wait for async feedback writes, then confirm both metrics landed.
    db.flush_pending_writes().await;
    sleep(Duration::from_secs(5)).await;
    let inference_id = info.response.inference_id();

    let bool_metric =
        "tensorzero::evaluation_name::typescript_judge_success::evaluator_name::ts_always_true";
    let bool_feedback = query_boolean_feedback(&db, inference_id, Some(bool_metric))
        .await
        .expect("boolean feedback for ts_always_true should be recorded");
    assert!(bool_feedback.value, "ts_always_true should persist `true`");

    let float_metric =
        "tensorzero::evaluation_name::typescript_judge_success::evaluator_name::ts_output_length";
    let float_feedback = query_float_feedback(&db, inference_id, Some(float_metric))
        .await
        .expect("float feedback for ts_output_length should be recorded");
    assert!(
        (float_feedback.value - len).abs() < 1e-6,
        "float feedback {} should match evaluator return {len}",
        float_feedback.value
    );
}

// ────────────────────────────────────────────────────────────────────────────
// Exception path
// ────────────────────────────────────────────────────────────────────────────

/// The `ts_throws` evaluator throws inside `tensorzero_evaluator`. The
/// evaluator should surface as an error (captured in `evaluator_errors`) and
/// NOT record feedback. The inference itself still succeeds.
#[tokio::test(flavor = "multi_thread")]
async fn test_typescript_judge_exception() {
    init_tracing_for_tests();
    let db = DelegatingDatabaseConnection::new_for_e2e_test().await;
    let config = get_config().await;
    let tensorzero_client = make_embedded_gateway().await;

    let dataset_name = format!("ts_judge_exception-{}", Uuid::now_v7());
    let datapoint = make_datapoint(&dataset_name);
    let datapoint_id = datapoint.id;
    db.insert_datapoints(&[StoredDatapoint::Chat(datapoint)])
        .await
        .expect("insert_datapoints should succeed");
    db.flush_pending_writes().await;

    let ts_executor = build_test_ts_executor().await;
    let args = core_args_for_evaluation(
        "typescript_judge_exception",
        datapoint_id,
        &config,
        &db,
        tensorzero_client,
        ts_executor,
    );

    let successes = run_single_evaluation(args).await;
    assert_eq!(
        successes.len(),
        1,
        "expected exactly one evaluated datapoint"
    );
    let info = &successes[0];

    let err = info
        .evaluator_errors
        .get("ts_throws")
        .expect("ts_throws should have produced an evaluator error");
    assert_eq!(
        err,
        "TypeScript evaluator error: JS runtime error: Event loop error in \
         <typescript_judge>: Uncaught (in promise) Error: \
         tensorzero-evaluator-test: deliberate failure"
    );

    // No value should have been produced, hence no feedback.
    assert!(
        info.evaluations
            .get("ts_throws")
            .and_then(|v| v.as_ref())
            .is_none(),
        "ts_throws should not have recorded a value"
    );

    db.flush_pending_writes().await;
    sleep(Duration::from_secs(5)).await;
    let inference_id = info.response.inference_id();
    let bool_metric =
        "tensorzero::evaluation_name::typescript_judge_exception::evaluator_name::ts_throws";
    assert!(
        query_boolean_feedback(&db, inference_id, Some(bool_metric))
            .await
            .is_none(),
        "no feedback should be written when the evaluator throws"
    );
}

// ────────────────────────────────────────────────────────────────────────────
// OOM path
// ────────────────────────────────────────────────────────────────────────────

/// The `ts_oom` evaluator allocates ~10MB in a JS array. With the executor's
/// 10MB heap cap, this trips the near-heap-limit callback, terminates the
/// isolate, and surfaces as an evaluator error. As with exceptions, no
/// feedback should be written.
#[tokio::test(flavor = "multi_thread")]
async fn test_typescript_judge_oom() {
    init_tracing_for_tests();
    let db = DelegatingDatabaseConnection::new_for_e2e_test().await;
    let config = get_config().await;
    let tensorzero_client = make_embedded_gateway().await;

    let dataset_name = format!("ts_judge_oom-{}", Uuid::now_v7());
    let datapoint = make_datapoint(&dataset_name);
    let datapoint_id = datapoint.id;
    db.insert_datapoints(&[StoredDatapoint::Chat(datapoint)])
        .await
        .expect("insert_datapoints should succeed");
    db.flush_pending_writes().await;

    let ts_executor = build_test_ts_executor().await;
    let args = core_args_for_evaluation(
        "typescript_judge_oom",
        datapoint_id,
        &config,
        &db,
        tensorzero_client,
        ts_executor,
    );

    let successes = run_single_evaluation(args).await;
    assert_eq!(
        successes.len(),
        1,
        "expected exactly one evaluated datapoint"
    );
    let info = &successes[0];

    let err = info
        .evaluator_errors
        .get("ts_oom")
        .expect("ts_oom should have produced an evaluator error");
    let err_lower = err.to_lowercase();
    assert!(
        err_lower.contains("out-of-memory") || err_lower.contains("out of memory"),
        "evaluator error should mention OOM, got: {err}"
    );

    assert!(
        info.evaluations
            .get("ts_oom")
            .and_then(|v| v.as_ref())
            .is_none(),
        "ts_oom should not have recorded a value"
    );

    db.flush_pending_writes().await;
    sleep(Duration::from_secs(5)).await;
    let inference_id = info.response.inference_id();
    let bool_metric = "tensorzero::evaluation_name::typescript_judge_oom::evaluator_name::ts_oom";
    assert!(
        query_boolean_feedback(&db, inference_id, Some(bool_metric))
            .await
            .is_none(),
        "no feedback should be written when the evaluator OOMs"
    );
}

// ────────────────────────────────────────────────────────────────────────────
// Missing `typescript_file`
// ────────────────────────────────────────────────────────────────────────────

/// Config loading must fail cleanly when `typescript_file` points at a
/// non-existent path. The error should name the offending field and the path
/// that couldn't be read, so users get an actionable message. This is
/// enforced at config-parse time (path walker), not at evaluator-dispatch
/// time — so no database, gateway, or evaluator runtime is needed.
#[tokio::test(flavor = "multi_thread")]
async fn test_typescript_judge_missing_file() {
    init_tracing_for_tests();

    let temp_dir = tempfile::tempdir().expect("failed to create tempdir");
    let config_path = temp_dir.path().join("tensorzero.toml");
    std::fs::write(
        &config_path,
        r#"
[evaluations.missing_ts_judge]
type = "inference"
function_name = "does_not_exist"

[evaluations.missing_ts_judge.evaluators.ts]
type = "typescript"
typescript_file = "./nonexistent.ts"
output_type = "boolean"
optimize = "max"
"#,
    )
    .expect("failed to write config");

    let err = tensorzero::ClientBuilder::new(tensorzero::ClientBuilderMode::EmbeddedGateway {
        config_file: Some(config_path.clone()),
        clickhouse_url: None,
        postgres_config: None,
        valkey_url: None,
        timeout: None,
        verify_credentials: false,
        allow_batch_writes: false,
    })
    .build()
    .await
    .expect_err("config load should fail when typescript_file is missing");

    // `base` is the parent dir of the config file (i.e. the tempdir itself);
    // reference it through the actual tempdir path so the assertion is
    // robust across platforms and random tempdir names.
    let base = temp_dir.path().display();
    let config_display = config_path.display();
    let expected = format!(
        "Failed to parse config: Internal TensorZero Error: \
         `evaluations.missing_ts_judge.evaluators.ts.typescript_file`: \
         Failed to resolve path `./nonexistent.ts` (base: `{base}`): \
         No such file or directory (os error 2). \
         Config file glob `{config_display}` resolved to the following files:\n\
         {config_display}"
    );
    assert_eq!(err.to_string(), expected);
}
