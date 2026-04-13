//! TypeScript judge evaluator.
//!
//! The real implementation depends on `ts-executor-pool` (which pulls in
//! `v8` / `deno_core`). Some platforms — notably the Python wheels we
//! publish for `manylinux2014` and `musllinux` — cannot link against
//! `rusty_v8` because it requires either a newer glibc (for `memfd_create`)
//! or a prebuilt static lib that upstream does not publish.
//!
//! When the `ts-executor-pool` feature is disabled, this module exposes a
//! stub `TypescriptJudgeExecutor` that satisfies the rest of the crate's
//! type signatures, but `run_typescript_judge_evaluator` errors on first
//! call. Callers (CLI, gateway, autopilot, e2e tests) keep the feature on;
//! only `tensorzero-python` opts out.

#[cfg(feature = "ts-executor-pool")]
mod imp {
    use std::sync::Arc;
    use std::time::Duration;

    use anyhow::Result;
    use serde_json::Value;
    use tensorzero_core::client::{InferenceResponse, Input};
    use tensorzero_core::endpoints::inference::InferenceResponse as CoreInferenceResponse;
    use tensorzero_core::evaluations::{TypescriptJudgeConfig, TypescriptJudgeOutputType};
    use ts_executor_pool::ExtraInferenceTags;
    use ts_executor_pool::runtime::RlmPool;
    use ts_executor_pool::tensorzero_client::TensorZeroClient;
    use ts_executor_pool::ts_checker::{TsCheckerPool, build_code_execution_ambient_declarations};

    /// Wall-clock timeout for a single TypeScript judge evaluation. Includes
    /// typecheck + execution.
    const TYPESCRIPT_JUDGE_TIMEOUT: Duration = Duration::from_secs(30);

    /// Per-isolate V8 heap limit for TypeScript judge evaluators.
    ///
    /// Judge code is user-provided and runs with no durable-task checkpointing or
    /// external resources — there is no legitimate reason it should retain large
    /// amounts of memory. A tight cap keeps runaway allocations (e.g. infinite
    /// loops filling arrays) from affecting the gateway. The near-heap-limit
    /// callback in `ts-executor-pool` catches allocations that cross this bound
    /// and terminates the isolate cleanly.
    const TYPESCRIPT_JUDGE_HEAP_LIMIT_MIB: usize = 10;

    /// Resources required to run TypeScript judge evaluators.
    ///
    /// The evaluator uses the shared `ts-executor-pool` sandbox: a pool of
    /// SES/V8 isolates (via `RlmPool`) plus a TypeScript typechecker
    /// (`TsCheckerPool`). Neither the pool nor the checker should be rebuilt
    /// per evaluation — construct once and clone into `Clients`.
    #[derive(Clone)]
    pub struct TypescriptJudgeExecutor {
        pool: RlmPool,
        ts_checker: Arc<TsCheckerPool>,
    }

    impl TypescriptJudgeExecutor {
        /// Build a new executor with a tight V8 heap cap
        /// (`TYPESCRIPT_JUDGE_HEAP_LIMIT_MIB`) applied to every isolate spawned
        /// from `pool`. Call once at process startup and clone the result into
        /// `Clients` for reuse across evaluations.
        pub fn new(pool: RlmPool, ts_checker: Arc<TsCheckerPool>) -> anyhow::Result<Self> {
            let pool = pool
                .with_heap_limit_mib(TYPESCRIPT_JUDGE_HEAP_LIMIT_MIB)
                .map_err(|e| {
                    anyhow::anyhow!("Failed to configure TypeScript judge heap limit: {e}")
                })?;
            Ok(Self { pool, ts_checker })
        }

        /// Build an executor with default pool sizes (1 runtime slot, 1 TS
        /// checker worker) and the standard 10 MiB heap cap. Intended for
        /// callers that don't need to tune concurrency — e.g. the CLI and
        /// tests. Async because `TsCheckerPool::new` restores a V8 snapshot on
        /// a blocking thread.
        pub async fn with_defaults() -> anyhow::Result<Self> {
            let pool = RlmPool::new(1)
                .map_err(|e| anyhow::anyhow!("Failed to build RlmPool for TS judge: {e}"))?;
            let ts_checker =
                Arc::new(TsCheckerPool::new(1).await.map_err(|e| {
                    anyhow::anyhow!("Failed to build TsCheckerPool for TS judge: {e}")
                })?);
            Self::new(pool, ts_checker)
        }
    }

    /// No-op `TensorZeroClient` used when spawning code-execution runtimes for
    /// TypeScript judge evaluators.
    ///
    /// `RlmPool::spawn_code_runtime` requires a `TensorZeroClient` because the
    /// runtime's `OpState` stores it for any `llm_query`-style ops — but in
    /// `RuntimeMode::CodeExecution` those ops are not endowed on the sandbox
    /// globals, so JS code cannot reach them. Hence this stub is never
    /// actually invoked during an evaluation; it exists solely to satisfy the
    /// type requirement.
    struct NoopTensorZeroClient;

    #[async_trait::async_trait]
    impl TensorZeroClient for NoopTensorZeroClient {
        async fn inference(
            &self,
            _params: tensorzero_core::client::ClientInferenceParams,
        ) -> Result<CoreInferenceResponse, Box<dyn std::error::Error + Send + Sync>> {
            Err("TensorZero inference is not available inside typescript judge evaluators".into())
        }
    }

    /// Compose the full ambient declarations passed to the TS typechecker.
    ///
    /// Pieces, in order:
    /// 1. ES stdlib + `console` + `FINAL` from
    ///    [`build_code_execution_ambient_declarations`]
    /// 2. `Input` (and its transitive deps) from `tensorzero-ts-types`
    /// 3. `ContentBlockChatOutput` (and its transitive deps) from `tensorzero-ts-types`
    ///
    /// The `tensorzero-ts-types` bundles are generated at build time from the
    /// `ts-rs`-emitted `.ts` files under `crates/tensorzero-node/lib/bindings/`,
    /// so the declarations stay in sync with the Rust types automatically.
    /// `declare const` is prepended so the generated `type ...` / `interface ...`
    /// statements appear as ambient declarations to the typechecker. Bundles are
    /// topologically sorted, which can cause duplicate declarations when two
    /// root types share dependencies; wrapping each bundle in `declare global
    /// { ... }` doesn't help, so we let the checker's implicit deduplication
    /// handle the common case (identical declarations) — if that ever turns out
    /// to conflict for a specific pair of bundles, we can split them across
    /// separate input files to the checker.
    fn evaluator_ambient_declarations() -> String {
        let stdlib = build_code_execution_ambient_declarations(&[]);
        format!(
            "{stdlib}\n\
             // --- TensorZero evaluator ambient types (from `tensorzero-ts-types`) ---\n\
             {input}\n\n\
             {content}\n",
            input = tensorzero_ts_types::INPUT.as_str(),
            content = tensorzero_ts_types::CONTENT_BLOCK_CHAT_OUTPUT.as_str(),
        )
    }

    pub async fn run_typescript_judge_evaluator(
        inference_response: &InferenceResponse,
        input: &Input,
        config: &TypescriptJudgeConfig,
        executor: &TypescriptJudgeExecutor,
    ) -> Result<Option<Value>> {
        // 1. Serialize input and output as JSON
        let input_json = serde_json::to_string(input)?;
        let output_json = match inference_response {
            InferenceResponse::Chat(chat) => serde_json::to_string(&chat.content)?,
            InferenceResponse::Json(json_resp) => {
                // Wrap JSON output as a single text content block for a consistent interface
                let raw = json_resp.output.raw.as_deref().unwrap_or("");
                let text_block = serde_json::json!([{
                    "type": "text",
                    "text": raw
                }]);
                serde_json::to_string(&text_block)?
            }
        };

        // 2. Build wrapper code that calls the user's evaluator function.
        let wrapper_code = format!(
            "{user_code}\n\n\
             const __t0_input: Input = {input_json};\n\
             const __t0_output: ContentBlockChatOutput[] = {output_json};\n\
             const __t0_result = await tensorzero_evaluator(__t0_input, __t0_output);\n\
             FINAL(JSON.stringify(__t0_result));\n",
            user_code = config.typescript_code,
            input_json = input_json,
            output_json = output_json,
        );

        // 3. Spawn a fresh code-execution runtime and typecheck+run the wrapped TS.
        let handle = executor
            .pool
            .spawn_code_runtime(
                Arc::new(NoopTensorZeroClient),
                ExtraInferenceTags::default(),
                executor.ts_checker.clone(),
                None,
            )
            .await
            .map_err(|e| anyhow::anyhow!("Failed to spawn TypeScript evaluator runtime: {e}"))?;

        let ambient = evaluator_ambient_declarations();
        let exec_result = handle
            .execute_typescript_block(
                executor.ts_checker.as_ref(),
                "<typescript_judge>".to_string(),
                &ambient,
                &wrapper_code,
                TYPESCRIPT_JUDGE_TIMEOUT,
            )
            .await;

        // Always try to shut down the runtime cleanly; log rather than propagate
        // any shutdown error so we don't mask the real execution error.
        if let Err(e) = handle.shutdown().await {
            tracing::error!(error = %e, "Failed to shut down TypeScript judge runtime");
        }

        let exec_result = exec_result
            .map_err(|e| anyhow::anyhow!("TypeScript evaluator execution failed: {e}"))?;

        let final_value = match exec_result.result {
            Ok(Ok(final_answer)) => final_answer,
            Ok(Err(cf)) => {
                return Err(anyhow::anyhow!(
                    "TypeScript evaluator terminated with control flow: {cf:?}"
                ));
            }
            Err(e) => {
                return Err(anyhow::anyhow!("TypeScript evaluator error: {e}"));
            }
        };

        // 4. Parse the result
        parse_evaluator_result(final_value, config.output_type)
    }

    fn parse_evaluator_result(
        result: Option<String>,
        output_type: TypescriptJudgeOutputType,
    ) -> Result<Option<Value>> {
        match result {
            Some(value_str) => {
                let parsed: serde_json::Value = serde_json::from_str(&value_str)
                    .map_err(|e| anyhow::anyhow!("Failed to parse evaluator result: {e}"))?;

                match output_type {
                    TypescriptJudgeOutputType::Boolean => {
                        if let Some(b) = parsed.as_bool() {
                            Ok(Some(Value::Bool(b)))
                        } else {
                            Err(anyhow::anyhow!(
                                "TypeScript evaluator returned `{parsed}`, but `output_type` is `boolean`"
                            ))
                        }
                    }
                    TypescriptJudgeOutputType::Float => {
                        if let Some(n) = parsed.as_f64() {
                            Ok(Some(Value::Number(
                                serde_json::Number::from_f64(n).ok_or_else(|| {
                                    anyhow::anyhow!(
                                        "TypeScript evaluator returned non-finite float: {n}"
                                    )
                                })?,
                            )))
                        } else {
                            Err(anyhow::anyhow!(
                                "TypeScript evaluator returned `{parsed}`, but `output_type` is `float`"
                            ))
                        }
                    }
                }
            }
            None => Err(anyhow::anyhow!(
                "TypeScript evaluator did not return a result (FINAL() was not called)"
            )),
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use googletest::prelude::*;
        use serde_json::json;
        use tensorzero_core::evaluations::TypescriptJudgeOptimize;

        fn make_boolean_config() -> TypescriptJudgeConfig {
            TypescriptJudgeConfig {
                typescript_code: "function tensorzero_evaluator() { return true; }".to_string(),
                output_type: TypescriptJudgeOutputType::Boolean,
                optimize: TypescriptJudgeOptimize::Max,
            }
        }

        fn make_float_config() -> TypescriptJudgeConfig {
            TypescriptJudgeConfig {
                typescript_code: "function tensorzero_evaluator() { return 0.75; }".to_string(),
                output_type: TypescriptJudgeOutputType::Float,
                optimize: TypescriptJudgeOptimize::Max,
            }
        }

        #[gtest]
        fn test_boolean_true_result() {
            let result =
                parse_evaluator_result(Some("true".to_string()), make_boolean_config().output_type);
            let value = result.expect("should succeed").expect("should have value");
            assert_eq!(value, json!(true));
        }

        #[gtest]
        fn test_boolean_false_result() {
            let result = parse_evaluator_result(
                Some("false".to_string()),
                make_boolean_config().output_type,
            );
            let value = result.expect("should succeed").expect("should have value");
            assert_eq!(value, json!(false));
        }

        #[gtest]
        fn test_float_result() {
            let result =
                parse_evaluator_result(Some("0.75".to_string()), make_float_config().output_type);
            let value = result.expect("should succeed").expect("should have value");
            assert_eq!(value, json!(0.75));
        }

        #[gtest]
        fn test_type_mismatch_float_for_boolean() {
            let result =
                parse_evaluator_result(Some("0.5".to_string()), make_boolean_config().output_type);
            assert_that!(result, err(displays_as(contains_substring("output_type"))));
        }

        #[gtest]
        fn test_type_mismatch_boolean_for_float() {
            let result =
                parse_evaluator_result(Some("true".to_string()), make_float_config().output_type);
            assert_that!(result, err(displays_as(contains_substring("output_type"))));
        }

        #[gtest]
        fn test_no_final_called() {
            let result = parse_evaluator_result(None, make_boolean_config().output_type);
            assert_that!(result, err(displays_as(contains_substring("FINAL()"))));
        }

        #[gtest]
        fn test_invalid_json_result() {
            let result = parse_evaluator_result(
                Some("not_valid_json".to_string()),
                make_boolean_config().output_type,
            );
            assert_that!(
                result,
                err(displays_as(contains_substring("Failed to parse")))
            );
        }

        #[gtest]
        fn test_integer_as_float() {
            let result =
                parse_evaluator_result(Some("42".to_string()), make_float_config().output_type);
            let value = result.expect("should succeed").expect("should have value");
            assert_eq!(value, json!(42.0));
        }

        #[gtest]
        fn test_string_result_rejected_for_boolean() {
            let result = parse_evaluator_result(
                Some(r#""hello""#.to_string()),
                make_boolean_config().output_type,
            );
            assert_that!(result, err(displays_as(contains_substring("output_type"))));
        }
    }
}

#[cfg(not(feature = "ts-executor-pool"))]
mod imp {
    //! Stub implementation used when the `ts-executor-pool` feature is
    //! disabled. The struct exists so that `Clients` and friends can hold
    //! a `ts_executor` field on every build, but the runtime entry point
    //! always errors.

    use anyhow::Result;
    use serde_json::Value;
    use tensorzero_core::client::{InferenceResponse, Input};
    use tensorzero_core::evaluations::TypescriptJudgeConfig;

    /// Stub executor: holds no state and exists purely so that types in
    /// `Clients` / `EvaluationCoreArgs` remain identical regardless of
    /// feature flags. Constructing one is a no-op and never fails.
    #[derive(Clone, Default)]
    pub struct TypescriptJudgeExecutor;

    impl TypescriptJudgeExecutor {
        /// Always succeeds. Provided so callers (CLI, gateway, Python
        /// bindings) can keep the same call site whether or not the
        /// `ts-executor-pool` feature is enabled.
        // `async` mirrors the real implementation's signature so callers
        // (which `.await` this) compile identically in both configurations.
        #[expect(clippy::unused_async)]
        pub async fn with_defaults() -> anyhow::Result<Self> {
            Ok(Self)
        }
    }

    // `async` mirrors the real implementation's signature; callers `.await`
    // this in both configurations.
    #[expect(clippy::unused_async)]
    pub async fn run_typescript_judge_evaluator(
        _inference_response: &InferenceResponse,
        _input: &Input,
        _config: &TypescriptJudgeConfig,
        _executor: &TypescriptJudgeExecutor,
    ) -> Result<Option<Value>> {
        Err(anyhow::anyhow!(
            "TypeScript judge evaluators are not available in this build of TensorZero \
             (the `evaluations` crate was built without the `ts-executor-pool` feature)."
        ))
    }
}

pub use imp::*;
