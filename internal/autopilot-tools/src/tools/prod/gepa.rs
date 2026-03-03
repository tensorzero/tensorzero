//! Tool for running GEPA (Genetic Evolution with Pareto Analysis) optimization
//! as a durable task with per-iteration sub-step checkpointing.

use std::borrow::Cow;
use std::time::Duration;

use async_trait::async_trait;
use durable_tools::{NonControlToolError, TaskTool, ToolContext, ToolMetadata, ToolResult};
use schemars::Schema;
use tensorzero_optimizers::gepa::{
    GepaAnalyzeParams, GepaCleanupParams, GepaEvalAndUpdateParams, GepaEvalParentParams,
    GepaIterUpdateResult, GepaMutateParams, GepaSampleParams, GepaSetupParams, GepaSetupResult,
    GepaSideInfo, GepaToolOutput, GepaToolParams,
};

/// Tool for running GEPA optimization with per-iteration sub-step checkpointing.
///
/// Each phase (setup, 5 sub-steps per iteration, cleanup) is a checkpointed step
/// so the durable framework can resume from the last completed step on restart.
///
/// This is a standalone tool — it uses `GepaSideInfo` (not `AutopilotSideInfo`)
/// and is registered via `visit_standalone_task_tool`, bypassing the autopilot
/// result-publishing wrapper.
pub struct GepaTool;

impl ToolMetadata for GepaTool {
    type SideInfo = GepaSideInfo;
    type Output = GepaToolOutput;
    type LlmParams = GepaToolParams;

    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("gepa_optimization")
    }

    fn description(&self) -> Cow<'static, str> {
        Cow::Borrowed(
            "Run GEPA (Genetic Evolution with Pareto Analysis) prompt optimization \
             with per-iteration sub-step checkpointing for durability.",
        )
    }

    fn timeout(&self) -> Duration {
        Duration::from_secs(86400)
    }

    fn parameters_schema(&self) -> ToolResult<Schema> {
        let schema = schemars::generate::SchemaSettings::default()
            .into_generator()
            .into_root_schema_for::<GepaToolParams>();
        serde_json::from_value(serde_json::to_value(schema).map_err(|e| {
            NonControlToolError::SchemaGeneration {
                message: e.to_string(),
            }
        })?)
        .map_err(|e| {
            NonControlToolError::SchemaGeneration {
                message: e.to_string(),
            }
            .into()
        })
    }

    fn strict(&self) -> bool {
        false
    }

    #[cfg(feature = "ts-bindings")]
    fn llm_params_ts_bundle() -> tensorzero_ts_types::TsTypeBundle {
        tensorzero_ts_types::UNIT
    }

    #[cfg(feature = "ts-bindings")]
    fn llm_params_ts_bundle_type_name() -> String {
        "void".to_string()
    }

    #[cfg(feature = "ts-bindings")]
    fn output_ts_bundle() -> tensorzero_ts_types::TsTypeBundle {
        tensorzero_ts_types::UNIT
    }

    #[cfg(feature = "ts-bindings")]
    fn output_ts_bundle_type_name() -> String {
        "void".to_string()
    }
}

#[async_trait]
impl TaskTool for GepaTool {
    type ExtraState = ();

    async fn execute(
        &self,
        llm_params: GepaToolParams,
        side_info: GepaSideInfo,
        ctx: &mut ToolContext,
    ) -> ToolResult<GepaToolOutput> {
        // Step 1: Setup — validate config, create val dataset, evaluate initial variants
        let setup_params = GepaSetupParams {
            gepa_config: llm_params.gepa_config,
            train_examples: side_info.train_examples,
            val_examples: side_info.val_examples,
        };

        let setup_result: GepaSetupResult = ctx
            .step("setup", setup_params, |params, state| async move {
                state
                    .t0_client()
                    .gepa_setup(params)
                    .await
                    .map_err(|e| anyhow::Error::msg(e.to_string()))
            })
            .await?;

        // Destructure for the iteration loop
        let max_iterations = setup_result.max_iterations;
        let original_variant_names = setup_result.original_variant_names;
        let mut checkpoint = setup_result.checkpoint;
        let mut temporary_datasets = setup_result.temporary_datasets;

        // Step 2: Per-iteration sub-steps
        for iteration in 0..(max_iterations as usize) {
            // Sub-step 2a: Sample parent and create minibatch dataset
            let sample_params = GepaSampleParams {
                checkpoint,
                train_examples: setup_result.train_examples.clone(),
                iteration,
                gepa_config: setup_result.gepa_config.clone(),
                val_dataset_name: setup_result.val_dataset_name.clone(),
                temporary_datasets,
                per_variant_concurrency: setup_result.per_variant_concurrency,
                run_id: setup_result.run_id,
            };

            let sample_result = ctx
                .step(
                    &format!("iter_{iteration}_sample"),
                    sample_params,
                    |params, state| async move {
                        state
                            .t0_client()
                            .gepa_iter_sample(params)
                            .await
                            .map_err(|e| anyhow::Error::msg(e.to_string()))
                    },
                )
                .await?;

            let sample_continue = match sample_result {
                tensorzero_optimizers::gepa::GepaSampleResult::Continue(data) => *data,
                tensorzero_optimizers::gepa::GepaSampleResult::SkipIteration(skip) => {
                    checkpoint = skip.checkpoint;
                    temporary_datasets = skip.temporary_datasets;
                    continue;
                }
            };

            // Sub-step 2b: Evaluate parent on minibatch
            let eval_parent_params = GepaEvalParentParams {
                parent: sample_continue.parent,
                mutation_dataset_name: sample_continue.mutation_dataset_name,
                gepa_config: setup_result.gepa_config.clone(),
                iteration,
                checkpoint: sample_continue.checkpoint,
                temporary_datasets: sample_continue.temporary_datasets,
                val_dataset_name: setup_result.val_dataset_name.clone(),
                per_variant_concurrency: setup_result.per_variant_concurrency,
            };

            let eval_parent_result = ctx
                .step(
                    &format!("iter_{iteration}_eval_parent"),
                    eval_parent_params,
                    |params, state| async move {
                        state
                            .t0_client()
                            .gepa_iter_eval_parent(params)
                            .await
                            .map_err(|e| anyhow::Error::msg(e.to_string()))
                    },
                )
                .await?;

            let eval_parent_continue = match eval_parent_result {
                tensorzero_optimizers::gepa::GepaEvalParentResult::Continue(data) => *data,
                tensorzero_optimizers::gepa::GepaEvalParentResult::SkipIteration(skip) => {
                    checkpoint = skip.checkpoint;
                    temporary_datasets = skip.temporary_datasets;
                    continue;
                }
            };

            // Sub-step 2c: Analyze parent inferences
            let analyze_params = GepaAnalyzeParams {
                parent: eval_parent_continue.parent,
                parent_evaluation_infos: eval_parent_continue.parent_evaluation_infos,
                parent_evaluation_stats: eval_parent_continue.parent_evaluation_stats,
                mutation_dataset_name: eval_parent_continue.mutation_dataset_name,
                gepa_config: setup_result.gepa_config.clone(),
                iteration,
                checkpoint: eval_parent_continue.checkpoint,
                temporary_datasets: eval_parent_continue.temporary_datasets,
                val_dataset_name: setup_result.val_dataset_name.clone(),
                per_variant_concurrency: setup_result.per_variant_concurrency,
            };

            let analyze_result = ctx
                .step(
                    &format!("iter_{iteration}_analyze"),
                    analyze_params,
                    |params, state| async move {
                        state
                            .t0_client()
                            .gepa_iter_analyze(params)
                            .await
                            .map_err(|e| anyhow::Error::msg(e.to_string()))
                    },
                )
                .await?;

            let analyze_continue = match analyze_result {
                tensorzero_optimizers::gepa::GepaAnalyzeResult::Continue(data) => *data,
                tensorzero_optimizers::gepa::GepaAnalyzeResult::SkipIteration(skip) => {
                    checkpoint = skip.checkpoint;
                    temporary_datasets = skip.temporary_datasets;
                    continue;
                }
            };

            // Sub-step 2d: Mutate parent to produce child variant
            let mutate_params = GepaMutateParams {
                parent: analyze_continue.parent,
                parent_analyses: analyze_continue.parent_analyses,
                parent_evaluation_stats: analyze_continue.parent_evaluation_stats,
                mutation_dataset_name: analyze_continue.mutation_dataset_name,
                gepa_config: setup_result.gepa_config.clone(),
                iteration,
                checkpoint: analyze_continue.checkpoint,
                temporary_datasets: analyze_continue.temporary_datasets,
                val_dataset_name: setup_result.val_dataset_name.clone(),
                per_variant_concurrency: setup_result.per_variant_concurrency,
            };

            let mutate_result = ctx
                .step(
                    &format!("iter_{iteration}_mutate"),
                    mutate_params,
                    |params, state| async move {
                        state
                            .t0_client()
                            .gepa_iter_mutate(params)
                            .await
                            .map_err(|e| anyhow::Error::msg(e.to_string()))
                    },
                )
                .await?;

            let mutate_continue = match mutate_result {
                tensorzero_optimizers::gepa::GepaMutateResult::Continue(data) => *data,
                tensorzero_optimizers::gepa::GepaMutateResult::SkipIteration(skip) => {
                    checkpoint = skip.checkpoint;
                    temporary_datasets = skip.temporary_datasets;
                    continue;
                }
            };

            // Sub-step 2e: Evaluate child, compare, conditional val eval, update frontier
            let update_params = GepaEvalAndUpdateParams {
                child: mutate_continue.child,
                parent_evaluation_stats: mutate_continue.parent_evaluation_stats,
                mutation_dataset_name: mutate_continue.mutation_dataset_name,
                gepa_config: setup_result.gepa_config.clone(),
                iteration,
                checkpoint: mutate_continue.checkpoint,
                temporary_datasets: mutate_continue.temporary_datasets,
                val_dataset_name: setup_result.val_dataset_name.clone(),
                per_variant_concurrency: setup_result.per_variant_concurrency,
            };

            let update_result: GepaIterUpdateResult = ctx
                .step(
                    &format!("iter_{iteration}_update"),
                    update_params,
                    |params, state| async move {
                        state
                            .t0_client()
                            .gepa_iter_eval_and_update(params)
                            .await
                            .map_err(|e| anyhow::Error::msg(e.to_string()))
                    },
                )
                .await?;

            checkpoint = update_result.checkpoint;
            temporary_datasets = update_result.temporary_datasets;
        }

        // Step 3: Cleanup — delete temporary datasets, extract final variants
        let cleanup_params = GepaCleanupParams {
            checkpoint,
            temporary_datasets,
            original_variant_names,
        };

        let output: GepaToolOutput = ctx
            .step("cleanup", cleanup_params, |params, state| async move {
                state
                    .t0_client()
                    .gepa_cleanup(params)
                    .await
                    .map_err(|e| anyhow::Error::msg(e.to_string()))
            })
            .await?;

        Ok(output)
    }
}
