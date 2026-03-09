//! Durable GEPA task tool with checkpointed sub-steps.
//!
//! Implements the GEPA algorithm as a `TaskTool` so it can be spawned via
//! `SpawnClient` and polled for results. Each major operation (setup, initial
//! evaluation, per-iteration eval/mutate/update) is wrapped in a
//! `ctx.step()` call for fine-grained checkpointing.
//!
//! Step params are lightweight — we never clone the full `SetupResult` into
//! subsequent steps. Instead, each step receives only the fields it needs.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use durable_tools::{
    RunEvaluationParams, RunEvaluationResponse, TaskTool, ToolAppState, ToolContext, ToolMetadata,
    ToolResult,
};
use futures::future::join_all;
use rand::seq::{IndexedRandom, SliceRandom};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use serde::Deserialize;
use serde_json::from_value;
use tensorzero_core::cache::CacheEnabledMode;
use tensorzero_core::client::ClientInferenceParams;
use tensorzero_core::config::{
    UninitializedConfig, UninitializedVariantConfig, UninitializedVariantInfo,
};
use tensorzero_core::endpoints::datasets::v1::types::ListDatapointsRequest;
use tensorzero_core::endpoints::inference::InferenceResponse;
use tensorzero_core::evaluations::{EvaluationConfig, EvaluatorConfig};
use tensorzero_core::inference::types::{
    ContentBlockChatOutput, InputExt, InputMessage, InputMessageContent, Role, Template, Text,
};
use tensorzero_core::optimization::gepa::{GEPAConfig, GepaEvaluatorStats};
use tensorzero_core::variant::chat_completion::UninitializedChatCompletionConfig;
use tokio::sync::Semaphore;
use uuid::Uuid;

use autopilot_client::AutopilotSideInfo;
use tensorzero_optimizers::gepa::analyze::{
    Analysis, Inference, build_analyze_input, create_analyze_variant_config,
    strip_signatures_from_chat_output, strip_signatures_from_stored_input,
};
use tensorzero_optimizers::gepa::durable::types::{
    EvalAnalyzeMutateResult, EvalAnalyzeMutateStepParams, EvalResult, EvalStepParams,
    GepaToolOutput, GepaToolParams, InitEvalStepParams, MutationResult, ResolvedGEPAConfig,
    SetupResult,
};
use tensorzero_optimizers::gepa::evaluate::{EvaluatorName, VariantName, VariantScores};
use tensorzero_optimizers::gepa::pareto::{Candidate, ParetoFrontier, is_improvement};
use tensorzero_optimizers::gepa::validate::FunctionContext;

use crate::error::AutopilotToolError;

// ── Shared metadata helpers ─────────────────────────────────────────────

const GEPA_DESCRIPTION: &str = "Run GEPA (Genetic Evolution with Pareto Analysis) prompt optimization. \
     Iteratively improves prompt templates using multi-objective Pareto optimization.";

const GEPA_TIMEOUT: Duration = Duration::from_secs(86400); // 24 hours

// ── Autopilot-visible tool (visit_task_tool) ────────────────────────────

/// GEPA tool registered via `visit_task_tool` — visible to the autopilot LLM.
pub struct GepaTool;

impl ToolMetadata for GepaTool {
    type SideInfo = AutopilotSideInfo;
    type Output = GepaToolOutput;
    type LlmParams = GepaToolParams;

    #[cfg(feature = "ts-bindings")]
    fn llm_params_ts_bundle() -> tensorzero_ts_types::TsTypeBundle {
        tensorzero_ts_types::GEPA_TOOL_PARAMS
    }

    #[cfg(feature = "ts-bindings")]
    fn llm_params_ts_bundle_type_name() -> String {
        "GepaToolParams".to_string()
    }

    #[cfg(feature = "ts-bindings")]
    fn output_ts_bundle() -> tensorzero_ts_types::TsTypeBundle {
        tensorzero_ts_types::GEPA_TOOL_OUTPUT
    }

    #[cfg(feature = "ts-bindings")]
    fn output_ts_bundle_type_name() -> String {
        "GepaToolOutput".to_string()
    }

    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("gepa")
    }

    fn description(&self) -> Cow<'static, str> {
        Cow::Borrowed(GEPA_DESCRIPTION)
    }

    fn timeout(&self) -> Duration {
        GEPA_TIMEOUT
    }

    fn strict(&self) -> bool {
        false
    }
}

#[async_trait]
impl TaskTool for GepaTool {
    type ExtraState = ();

    async fn execute(
        &self,
        llm_params: GepaToolParams,
        _side_info: AutopilotSideInfo,
        ctx: &mut ToolContext,
    ) -> ToolResult<GepaToolOutput> {
        Box::pin(execute_gepa(llm_params, ctx)).await
    }
}

// ── Standalone tool (visit_standalone_task_tool) ────────────────────────

/// GEPA tool registered via `visit_standalone_task_tool` — for the HTTP endpoint.
/// Not visible to the autopilot server; no result publishing.
pub struct StandaloneGepaTool;

impl ToolMetadata for StandaloneGepaTool {
    type SideInfo = ();
    type Output = GepaToolOutput;
    type LlmParams = GepaToolParams;

    #[cfg(feature = "ts-bindings")]
    fn llm_params_ts_bundle() -> tensorzero_ts_types::TsTypeBundle {
        tensorzero_ts_types::GEPA_TOOL_PARAMS
    }

    #[cfg(feature = "ts-bindings")]
    fn llm_params_ts_bundle_type_name() -> String {
        "GepaToolParams".to_string()
    }

    #[cfg(feature = "ts-bindings")]
    fn output_ts_bundle() -> tensorzero_ts_types::TsTypeBundle {
        tensorzero_ts_types::GEPA_TOOL_OUTPUT
    }

    #[cfg(feature = "ts-bindings")]
    fn output_ts_bundle_type_name() -> String {
        "GepaToolOutput".to_string()
    }

    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed("standalone_gepa")
    }

    fn description(&self) -> Cow<'static, str> {
        Cow::Borrowed(GEPA_DESCRIPTION)
    }

    fn timeout(&self) -> Duration {
        GEPA_TIMEOUT
    }

    fn strict(&self) -> bool {
        false
    }
}

#[async_trait]
impl TaskTool for StandaloneGepaTool {
    type ExtraState = ();

    async fn execute(
        &self,
        llm_params: GepaToolParams,
        _side_info: (),
        ctx: &mut ToolContext,
    ) -> ToolResult<GepaToolOutput> {
        Box::pin(execute_gepa(llm_params, ctx)).await
    }
}

// ── Shared execution logic ──────────────────────────────────────────────

async fn execute_gepa(
    llm_params: GepaToolParams,
    ctx: &mut ToolContext,
) -> ToolResult<GepaToolOutput> {
    // ── Step "setup": validate config, resolve datasets ─────────
    let setup: SetupResult = ctx.step("setup", llm_params.clone(), setup_step).await?;

    let evaluator_configs = &setup.evaluator_configs;

    tracing::info!(
        "Starting GEPA optimization for function '{}' with {} train and {} val datapoints",
        setup.gepa_config.function_name,
        setup.train_datapoint_ids.len(),
        setup.val_datapoint_ids.len()
    );

    tracing::info!(
        "Initialized with {} baseline variants: {:?}",
        setup.original_variants.len(),
        setup.original_variants.keys().collect::<Vec<_>>()
    );

    // ── Step "init_eval": evaluate all initial variants ─────────
    tracing::info!(
        "Evaluating {} initial variants on validation set",
        setup.original_variants.len()
    );

    let init_eval_params = InitEvalStepParams {
        evaluation_name: setup.gepa_config.evaluation_name.clone(),
        datapoint_ids: setup.val_datapoint_ids.clone(),
        variants: setup.original_variants.clone(),
        max_concurrency: setup.gepa_config.max_concurrency,
    };
    let init_scores: HashMap<VariantName, VariantScores> = ctx
        .step("init_eval", init_eval_params, init_eval_step)
        .await?;

    tracing::info!(
        "Initial evaluation complete: collected validation scores for {} variants",
        init_scores.len()
    );

    if init_scores.is_empty() {
        return Err(AutopilotToolError::validation(
            "No validation scores collected for initial variants",
        )
        .into());
    }

    // Derive deterministic seeds from the checkpointed rng_seed.
    // This ensures identical RNG state on task resumption.
    let mut master_rng = StdRng::seed_from_u64(setup.rng_seed);

    // Build initial Pareto frontier
    let mut pareto_frontier = ParetoFrontier::new(
        setup.val_datapoint_ids.clone(),
        evaluator_configs,
        Some(master_rng.random::<u64>()),
    );

    let initial_candidates: HashMap<VariantName, Candidate> = init_scores
        .into_iter()
        .filter_map(|(name, scores)| {
            setup.original_variants.get(&name).map(|config| {
                (
                    name,
                    Candidate {
                        variant: config.clone(),
                        scores,
                    },
                )
            })
        })
        .collect();

    pareto_frontier
        .update(initial_candidates)
        .map_err(|e| AutopilotToolError::validation(e.to_string()))?;

    let original_variant_names: std::collections::HashSet<String> =
        setup.original_variants.keys().cloned().collect();

    let max_iterations = setup.gepa_config.max_iterations as usize;

    // Initialize RNG for minibatch sampling (derived from checkpointed seed)
    let mut sampling_rng = StdRng::from_rng(&mut master_rng);

    // ── Main iteration loop ─────────────────────────────────────
    for iteration in 0..max_iterations {
        ctx.heartbeat(Some(Duration::from_secs(600))).await?;

        // Sample parent from Pareto frontier
        let parent = match pareto_frontier.sample_by_frequency() {
            Ok(variant) => variant,
            Err(err) => {
                tracing::warn!(
                    "Skipping iteration {iteration} because no candidates were available: {err}"
                );
                continue;
            }
        };

        tracing::info!(
            "GEPA iteration {iteration}: selected parent variant '{}'",
            parent.name
        );

        // Sample minibatch datapoint IDs from training dataset
        let batch_size = setup
            .gepa_config
            .batch_size
            .min(setup.train_datapoint_ids.len());
        let sampled_ids: Vec<Uuid> = setup
            .train_datapoint_ids
            .sample(&mut sampling_rng, batch_size)
            .copied()
            .collect();

        // ── Step: evaluate parent + analyze + mutate ─────────────
        tracing::info!(
            "GEPA iteration {iteration}: eval+analyze+mutate for parent '{}' on minibatch ({batch_size} datapoints)",
            parent.name
        );

        let eval_analyze_mutate: Option<EvalAnalyzeMutateResult> = ctx
            .step(
                &format!("iter_{iteration}_eval_analyze_mutate"),
                EvalAnalyzeMutateStepParams {
                    evaluation_name: setup.gepa_config.evaluation_name.clone(),
                    datapoint_ids: sampled_ids.clone(),
                    variant_name: parent.name.clone(),
                    variant_config: parent.config.clone(),
                    function_context: setup.function_context.clone(),
                    gepa_config: setup.gepa_config.clone(),
                    iteration: iteration as u32,
                    max_concurrency: setup.gepa_config.max_concurrency,
                },
                eval_analyze_mutate_step,
            )
            .await?;

        let Some(result) = eval_analyze_mutate else {
            tracing::warn!(
                "GEPA iteration {iteration}: eval+analyze+mutate returned no results, skipping"
            );
            continue;
        };

        let parent_eval = result.parent_eval;

        tracing::debug!(
            "GEPA iteration {iteration}: parent '{}' minibatch evaluation stats: {:#?}",
            parent.name,
            parent_eval.stats
        );

        let Some(mutation) = result.mutation else {
            tracing::warn!("GEPA iteration {iteration}: mutation failed, skipping");
            continue;
        };

        tracing::info!(
            "GEPA iteration {iteration}: mutation complete, evaluating child variant '{}'",
            mutation.child_name
        );

        // ── Step: evaluate child on minibatch ────────────────────
        tracing::info!(
            "GEPA iteration {iteration}: evaluating child variant '{}' on minibatch",
            mutation.child_name
        );

        let child_eval: Option<EvalResult> = ctx
            .step(
                &format!("iter_{iteration}_eval_child"),
                EvalStepParams {
                    evaluation_name: setup.gepa_config.evaluation_name.clone(),
                    dataset_name: None,
                    datapoint_ids: Some(sampled_ids),
                    variant_name: mutation.child_name.clone(),
                    variant_config: mutation.child_config.clone(),
                    max_concurrency: setup.gepa_config.max_concurrency,
                },
                eval_variant_step,
            )
            .await?;

        let Some(child_eval) = child_eval else {
            tracing::warn!(
                "GEPA iteration {iteration}: child evaluation returned no results, skipping"
            );
            continue;
        };

        tracing::debug!(
            "GEPA iteration {iteration}: child '{}' minibatch evaluation stats: {:#?}",
            mutation.child_name,
            child_eval.stats
        );

        tracing::info!(
            "GEPA iteration {iteration}: child variant '{}' minibatch evaluation complete",
            mutation.child_name
        );

        // Check if child improves on parent (minibatch comparison)
        let child_improves =
            is_improvement(&parent_eval.stats, &child_eval.stats, evaluator_configs);

        if !child_improves {
            tracing::info!("GEPA iteration {iteration}: child did not improve on parent, skipping");
            continue;
        }

        // ── Step: evaluate child on validation set ───────────────
        tracing::info!(
            "GEPA iteration {iteration}: evaluating child variant '{}' on validation set",
            mutation.child_name
        );

        let child_val_eval: Option<EvalResult> = ctx
            .step(
                &format!("iter_{iteration}_eval_child_val"),
                EvalStepParams {
                    evaluation_name: setup.gepa_config.evaluation_name.clone(),
                    dataset_name: None,
                    datapoint_ids: Some(setup.val_datapoint_ids.clone()),
                    variant_name: mutation.child_name.clone(),
                    variant_config: mutation.child_config.clone(),
                    max_concurrency: setup.gepa_config.max_concurrency,
                },
                eval_variant_step,
            )
            .await?;

        if let Some(val_eval) = child_val_eval {
            tracing::debug!(
                "GEPA iteration {iteration}: child '{}' validation evaluation stats: {:#?}",
                mutation.child_name,
                val_eval.stats
            );
            tracing::info!(
                "GEPA iteration {iteration}: child variant '{}' validation scores collected ({} datapoints)",
                mutation.child_name,
                val_eval.scores.len()
            );

            let mut candidate = HashMap::new();
            candidate.insert(
                mutation.child_name.clone(),
                Candidate {
                    variant: mutation.child_config.clone(),
                    scores: val_eval.scores,
                },
            );
            match pareto_frontier.update(candidate) {
                Ok(()) => {
                    tracing::info!(
                        "GEPA iteration {iteration}: Pareto frontier updated; pool size: {}",
                        pareto_frontier.variant_configs().len()
                    );
                }
                Err(err) => {
                    tracing::warn!(
                        "GEPA iteration {iteration}: failed to update Pareto frontier: {err}"
                    );
                }
            }
        } else {
            tracing::warn!(
                "GEPA iteration {iteration}: validation evaluation returned no results for child '{}'",
                mutation.child_name
            );
        }
    }

    // Filter out original variants
    let mut new_variants = pareto_frontier.variant_configs().clone();
    new_variants.retain(|name, _| !original_variant_names.contains(name));

    tracing::info!(
        "GEPA optimization complete: created {} new variant(s)",
        new_variants.len()
    );
    tracing::debug!("New variants: {:#?}", new_variants);

    // Build statistics from the Pareto frontier, filtered to new variants only
    let mut statistics = build_statistics(&pareto_frontier, evaluator_configs);
    statistics.retain(|name, _| !original_variant_names.contains(name));

    Ok(GepaToolOutput {
        variants: new_variants,
        statistics,
    })
}

// ── Step functions ──────────────────────────────────────────────────────

/// Setup step: validate config, resolve initial variants, create val dataset.
async fn setup_step(params: GepaToolParams, state: ToolAppState) -> anyhow::Result<SetupResult> {
    let client = state.t0_client();

    // Get live config for validation
    let config_response = client
        .get_config_snapshot(None)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to get config: {e}"))?;

    let uninitialized_config: UninitializedConfig =
        serde_json::from_value(config_response.config.clone())
            .map_err(|e| anyhow::anyhow!("Failed to deserialize config snapshot: {e}"))?;

    // Build GEPAConfig from params
    let evaluation_name = params.evaluation_name.clone().ok_or_else(|| {
        anyhow::anyhow!(
            "`evaluation_name` is required (inline `evaluators` mode is not yet supported)"
        )
    })?;

    let gepa_config = GEPAConfig {
        function_name: params.function_name.clone(),
        evaluation_name: evaluation_name.clone(),
        initial_variants: params.initial_variants.clone(),
        variant_prefix: params.variant_prefix.clone(),
        batch_size: params.batch_size.unwrap_or(5),
        max_iterations: params.max_iterations,
        max_concurrency: params.max_concurrency.unwrap_or(10),
        analysis_model: params.analysis_model.clone(),
        mutation_model: params.mutation_model.clone(),
        seed: params.seed,
        timeout: 300,
        include_inference_for_mutation: params.include_inference_for_mutation.unwrap_or(true),
        retries: Default::default(),
        max_tokens: None,
    };

    // Validate config using the uninitialized config path
    let function_context =
        tensorzero_optimizers::gepa::validate::validate_gepa_config_uninitialized(
            &gepa_config,
            &uninitialized_config,
        )
        .map_err(|e| anyhow::anyhow!("Config validation failed: {e}"))?;

    // Get initial variant configs
    let original_variants =
        tensorzero_optimizers::gepa::validate::get_uninitialized_variant_configs(
            &gepa_config,
            &function_context,
        )
        .map_err(|e| anyhow::anyhow!("Failed to get initial variants: {e}"))?;

    // Get evaluator configs for Pareto analysis
    let evaluator_configs = match &*function_context.evaluation_config {
        EvaluationConfig::Inference(cfg) => cfg.evaluators.clone(),
    };

    let run_id = Uuid::now_v7();

    // Resolve datasets and datapoint IDs.
    // Two modes:
    //   1. Separate datasets: `train_dataset_name` + `val_dataset_name`
    //   2. Auto-split: single `dataset_name` shuffled and split 50/50
    let has_separate_datasets =
        params.train_dataset_name.is_some() || params.val_dataset_name.is_some();

    let (train_dataset_name, train_datapoint_ids, val_dataset_name, val_datapoint_ids) =
        if has_separate_datasets {
            let train_name = params
                .train_dataset_name
                .clone()
                .or_else(|| params.dataset_name.clone())
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "Either `dataset_name` or `train_dataset_name` must be provided"
                    )
                })?;

            let val_name = params
                .val_dataset_name
                .clone()
                .or_else(|| params.dataset_name.clone())
                .ok_or_else(|| {
                    anyhow::anyhow!("Either `dataset_name` or `val_dataset_name` must be provided")
                })?;

            let train_datapoints = client
                .list_datapoints(
                    train_name.clone(),
                    ListDatapointsRequest {
                        limit: Some(u32::MAX),
                        ..Default::default()
                    },
                )
                .await
                .map_err(|e| anyhow::anyhow!("Failed to list training datapoints: {e}"))?;

            let train_ids: Vec<Uuid> = train_datapoints.datapoints.iter().map(|d| d.id()).collect();
            if train_ids.is_empty() {
                return Err(anyhow::anyhow!(
                    "Training dataset `{train_name}` contains no datapoints"
                ));
            }

            let val_datapoints = client
                .list_datapoints(
                    val_name.clone(),
                    ListDatapointsRequest {
                        limit: Some(u32::MAX),
                        ..Default::default()
                    },
                )
                .await
                .map_err(|e| anyhow::anyhow!("Failed to list val datapoints: {e}"))?;

            let val_ids: Vec<Uuid> = val_datapoints.datapoints.iter().map(|d| d.id()).collect();
            if val_ids.is_empty() {
                return Err(anyhow::anyhow!(
                    "Validation dataset `{val_name}` contains no datapoints"
                ));
            }

            (train_name, train_ids, val_name, val_ids)
        } else {
            // Auto-split: single dataset shuffled 50/50
            let dataset_name = params.dataset_name.clone().ok_or_else(|| {
                anyhow::anyhow!(
                    "One of `dataset_name`, `train_dataset_name`, or `val_dataset_name` must be provided"
                )
            })?;

            let all_datapoints = client
                .list_datapoints(
                    dataset_name.clone(),
                    ListDatapointsRequest {
                        limit: Some(u32::MAX),
                        ..Default::default()
                    },
                )
                .await
                .map_err(|e| anyhow::anyhow!("Failed to list datapoints: {e}"))?;

            let mut all_ids: Vec<Uuid> = all_datapoints.datapoints.iter().map(|d| d.id()).collect();
            if all_ids.len() < 2 {
                return Err(anyhow::anyhow!(
                    "Dataset `{dataset_name}` needs at least 2 datapoints for auto-split"
                ));
            }

            // Shuffle deterministically for reproducibility
            let mut split_rng = match gepa_config.seed {
                Some(seed) => StdRng::seed_from_u64(seed as u64),
                None => StdRng::from_rng(&mut rand::rng()),
            };
            all_ids.shuffle(&mut split_rng);

            let split_point = all_ids.len() / 2;
            let train_ids = all_ids[..split_point].to_vec();
            let val_ids = all_ids[split_point..].to_vec();

            tracing::info!(
                "Auto-split dataset `{dataset_name}`: {} train, {} val datapoints",
                train_ids.len(),
                val_ids.len(),
            );

            (dataset_name.clone(), train_ids, dataset_name, val_ids)
        };

    // Generate a deterministic seed for all post-setup RNG usage.
    // This is checkpointed in SetupResult, ensuring identical RNG state on resume.
    let rng_seed: u64 = match gepa_config.seed {
        Some(seed) => seed as u64,
        None => rand::rng().random::<u64>(),
    };

    let resolved_config = ResolvedGEPAConfig {
        function_name: gepa_config.function_name,
        evaluation_name: gepa_config.evaluation_name,
        initial_variants: gepa_config.initial_variants,
        variant_prefix: gepa_config.variant_prefix,
        batch_size: gepa_config.batch_size,
        max_iterations: gepa_config.max_iterations,
        max_concurrency: gepa_config.max_concurrency,
        analysis_model: gepa_config.analysis_model,
        mutation_model: gepa_config.mutation_model,
        seed: gepa_config.seed,
        include_inference_for_mutation: gepa_config.include_inference_for_mutation,
    };

    Ok(SetupResult {
        function_context: function_context.to_serializable(),
        original_variants,
        train_dataset_name,
        train_datapoint_ids,
        val_dataset_name,
        val_datapoint_ids,
        evaluator_configs,
        run_id,
        gepa_config: resolved_config,
        rng_seed,
    })
}

/// Initial evaluation step: evaluate all initial variants on validation set.
async fn init_eval_step(
    params: InitEvalStepParams,
    state: ToolAppState,
) -> anyhow::Result<HashMap<VariantName, VariantScores>> {
    let client = state.t0_client();
    let mut all_scores: HashMap<VariantName, VariantScores> = HashMap::new();

    for (variant_name, variant_config) in &params.variants {
        let variant_info = UninitializedVariantInfo {
            inner: UninitializedVariantConfig::ChatCompletion(variant_config.clone()),
            timeouts: None,
        };

        let eval_params = RunEvaluationParams {
            evaluation_name: params.evaluation_name.clone(),
            dataset_name: None,
            datapoint_ids: Some(params.datapoint_ids.clone()),
            variant_name: variant_name.clone(),
            concurrency: params.max_concurrency as usize,
            inference_cache: CacheEnabledMode::Off,
            max_datapoints: None,
            precision_targets: HashMap::new(),
            include_datapoint_results: true,
            tags: HashMap::new(),
            internal_dynamic_variant_config: Some(variant_info),
            include_evaluation_infos: false,
        };

        match client.run_evaluation(eval_params).await {
            Ok(response) => {
                let scores = extract_scores_from_response(&response);
                if !scores.is_empty() {
                    all_scores.insert(variant_name.clone(), scores);
                }
            }
            Err(e) => {
                tracing::warn!("Init eval failed for variant `{variant_name}`: {e}");
            }
        }
    }

    Ok(all_scores)
}

/// Evaluate a variant on a dataset (used for parent, child, and validation evaluations).
async fn eval_variant_step(
    params: EvalStepParams,
    state: ToolAppState,
) -> anyhow::Result<Option<EvalResult>> {
    let client = state.t0_client();

    let variant_info = UninitializedVariantInfo {
        inner: UninitializedVariantConfig::ChatCompletion(params.variant_config),
        timeouts: None,
    };

    let eval_params = RunEvaluationParams {
        evaluation_name: params.evaluation_name,
        dataset_name: params.dataset_name,
        datapoint_ids: params.datapoint_ids,
        variant_name: params.variant_name.clone(),
        concurrency: params.max_concurrency as usize,
        inference_cache: CacheEnabledMode::Off,
        max_datapoints: None,
        precision_targets: HashMap::new(),
        include_datapoint_results: true,
        tags: HashMap::new(),
        internal_dynamic_variant_config: Some(variant_info),
        include_evaluation_infos: false,
    };

    match client.run_evaluation(eval_params).await {
        Ok(response) => {
            let scores = extract_scores_from_response(&response);
            if scores.is_empty() {
                return Ok(None);
            }
            Ok(Some(EvalResult {
                variant_name: params.variant_name,
                scores,
                stats: response.stats,
            }))
        }
        Err(e) => {
            tracing::warn!(
                "Evaluation failed for variant `{}`: {e}",
                params.variant_name
            );
            Ok(None)
        }
    }
}

/// Combined eval + analyze + mutate step.
///
/// Runs parent evaluation with `include_evaluation_infos: true` to get full
/// `EvaluationInfo` objects, then analyzes them in-memory, then mutates using
/// those analyses. This keeps large inference data in-memory only — never
/// checkpointed.
async fn eval_analyze_mutate_step(
    params: EvalAnalyzeMutateStepParams,
    state: ToolAppState,
) -> anyhow::Result<Option<EvalAnalyzeMutateResult>> {
    let client = state.t0_client();

    // ── 1. Evaluate parent with evaluation_infos ────────────────
    let variant_info = UninitializedVariantInfo {
        inner: UninitializedVariantConfig::ChatCompletion(params.variant_config.clone()),
        timeouts: None,
    };

    let eval_params = RunEvaluationParams {
        evaluation_name: params.evaluation_name.clone(),
        dataset_name: None,
        datapoint_ids: Some(params.datapoint_ids.clone()),
        variant_name: params.variant_name.clone(),
        concurrency: params.max_concurrency as usize,
        inference_cache: CacheEnabledMode::Off,
        max_datapoints: None,
        precision_targets: HashMap::new(),
        include_datapoint_results: true,
        tags: HashMap::new(),
        internal_dynamic_variant_config: Some(variant_info),
        include_evaluation_infos: true,
    };

    let response = match client.run_evaluation(eval_params).await {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(
                "Evaluation failed for variant `{}`: {e}",
                params.variant_name
            );
            return Ok(None);
        }
    };

    let scores = extract_scores_from_response(&response);
    if scores.is_empty() {
        return Ok(None);
    }

    let parent_eval = EvalResult {
        variant_name: params.variant_name.clone(),
        scores,
        stats: response.stats,
    };

    let evaluation_infos = response.evaluation_infos.unwrap_or_default();

    // ── 2. Reconstruct GEPAConfig and FunctionContext ────────────
    let gepa_config = reconstruct_gepa_config(&params.gepa_config);

    let config_response = client
        .get_config_snapshot(None)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to get config: {e}"))?;

    let uninitialized_config: UninitializedConfig =
        serde_json::from_value(config_response.config.clone())
            .map_err(|e| anyhow::anyhow!("Failed to deserialize config snapshot: {e}"))?;

    let function_context = params
        .function_context
        .load(
            &params.gepa_config.function_name,
            &uninitialized_config.metrics,
        )
        .map_err(|e| anyhow::anyhow!("Failed to load function context: {e}"))?;

    // ── 3. Analyze inferences ───────────────────────────────────
    let analyses = durable_analyze_inferences(
        &**client,
        &evaluation_infos,
        &function_context,
        &params.variant_config,
        &gepa_config,
    )
    .await;

    let analyses = match analyses {
        Ok(a) => a,
        Err(e) => {
            tracing::warn!(
                "GEPA iteration {}: analysis failed for parent `{}`: {e}. Proceeding with empty analyses.",
                params.iteration,
                params.variant_name,
            );
            Vec::new()
        }
    };

    tracing::info!(
        "GEPA iteration {}: analyzed {}/{} inferences for parent `{}`",
        params.iteration,
        analyses.len(),
        evaluation_infos.len(),
        params.variant_name,
    );

    // ── 4. Mutate using analyses ────────────────────────────────
    let mutate_variant_config =
        tensorzero_optimizers::gepa::mutate::create_mutate_variant_config(&gepa_config);

    let mutate_input = tensorzero_optimizers::gepa::mutate::build_mutate_input(
        &analyses,
        &function_context,
        &params.variant_config,
    )
    .map_err(|e| anyhow::anyhow!("Failed to build mutate input: {e}"))?;

    let mutate_inference_params = ClientInferenceParams {
        function_name: Some("tensorzero::optimization::gepa::mutate".to_string()),
        input: tensorzero_core::client::Input {
            messages: vec![InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::Template(Template {
                    name: "user".to_string(),
                    arguments: mutate_input,
                })],
            }],
            system: None,
        },
        dryrun: Some(true),
        internal: true,
        internal_dynamic_variant_config: Some(UninitializedVariantInfo {
            inner: UninitializedVariantConfig::ChatCompletion(mutate_variant_config),
            timeouts: None,
        }),
        ..Default::default()
    };

    let mutate_response = match client.inference(mutate_inference_params).await {
        Ok(response) => response,
        Err(e) => {
            tracing::warn!(
                "GEPA iteration {}: mutation inference failed for parent `{}`: {e}",
                params.iteration,
                params.variant_name,
            );
            return Ok(Some(EvalAnalyzeMutateResult {
                parent_eval,
                mutation: None,
            }));
        }
    };

    let mutation = match parse_mutate_response(
        &mutate_response,
        &params.variant_name,
        &params.variant_config,
        &gepa_config,
        params.iteration as usize,
    )? {
        Some(result) => Some(result),
        None => {
            tracing::warn!(
                "GEPA iteration {}: failed to parse mutation response for parent `{}`",
                params.iteration,
                params.variant_name,
            );
            None
        }
    };

    Ok(Some(EvalAnalyzeMutateResult {
        parent_eval,
        mutation,
    }))
}

/// Analyze inferences using the `TensorZeroClient::inference()` API.
///
/// Mirrors `analyze_inferences()` from the sequential path but uses the
/// `TensorZeroClient` trait (returns `InferenceResponse` directly) instead
/// of the SDK `Client` (returns `InferenceOutput`).
///
/// Graceful degradation: individual failures are logged and skipped.
/// Returns error only if all analyses fail.
async fn durable_analyze_inferences(
    client: &dyn durable_tools::TensorZeroClient,
    evaluation_infos: &[evaluations::stats::EvaluationInfo],
    function_context: &FunctionContext,
    variant_config: &UninitializedChatCompletionConfig,
    gepa_config: &GEPAConfig,
) -> anyhow::Result<Vec<Analysis>> {
    if evaluation_infos.is_empty() {
        return Ok(Vec::new());
    }

    let max_concurrency = gepa_config.max_concurrency as usize;

    tracing::info!(
        "Analyzing {} inferences using model '{}' with max concurrency {}",
        evaluation_infos.len(),
        gepa_config.analysis_model,
        max_concurrency
    );

    let semaphore = Arc::new(Semaphore::new(max_concurrency));

    let analysis_futures: Vec<_> = evaluation_infos
        .iter()
        .map(|eval_info| {
            let semaphore = Arc::clone(&semaphore);
            durable_analyze_single_inference(
                semaphore,
                client,
                function_context,
                variant_config,
                gepa_config,
                eval_info,
            )
        })
        .collect();

    let results = join_all(analysis_futures).await;

    let (successes, failures): (Vec<_>, Vec<_>) = results
        .into_iter()
        .enumerate()
        .partition(|(_, result)| result.is_ok());

    let successes: Vec<Analysis> = successes
        .into_iter()
        .filter_map(|(_, result)| result.ok())
        .collect();

    for (index, result) in &failures {
        if let Err(e) = result {
            tracing::warn!(
                "Analysis failed for inference {}/{}: {}",
                index + 1,
                evaluation_infos.len(),
                e
            );
        }
    }

    if successes.is_empty() {
        let first_error = failures
            .into_iter()
            .find_map(|(_, r)| r.err())
            .map(|e| e.to_string())
            .unwrap_or_else(|| "Unknown error".to_string());
        return Err(anyhow::anyhow!(
            "All {} analyses failed. First error: {}",
            evaluation_infos.len(),
            first_error
        ));
    }

    if failures.is_empty() {
        tracing::info!(
            "Successfully completed all {} analyses",
            evaluation_infos.len()
        );
    } else {
        tracing::warn!(
            "Completed {}/{} analyses successfully ({} failed)",
            successes.len(),
            evaluation_infos.len(),
            failures.len()
        );
    }

    Ok(successes)
}

/// Analyze a single inference using the durable TensorZero client.
async fn durable_analyze_single_inference(
    semaphore: Arc<Semaphore>,
    client: &dyn durable_tools::TensorZeroClient,
    function_context: &FunctionContext,
    variant_config: &UninitializedChatCompletionConfig,
    gepa_config: &GEPAConfig,
    eval_info: &evaluations::stats::EvaluationInfo,
) -> anyhow::Result<Analysis> {
    let _permit = semaphore
        .acquire()
        .await
        .map_err(|e| anyhow::anyhow!("Failed to acquire semaphore: {e}"))?;

    let analyze_config = create_analyze_variant_config(gepa_config);
    let analyze_variant_config = UninitializedVariantInfo {
        inner: UninitializedVariantConfig::ChatCompletion(analyze_config),
        timeouts: None,
    };

    let arguments = build_analyze_input(eval_info, function_context, variant_config)
        .map_err(|e| anyhow::anyhow!("Failed to build analyze input: {e}"))?;

    let params = ClientInferenceParams {
        function_name: Some("tensorzero::optimization::gepa::analyze".to_string()),
        input: tensorzero_core::client::Input {
            messages: vec![InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::Template(Template {
                    name: "user".to_string(),
                    arguments,
                })],
            }],
            system: None,
        },
        dryrun: Some(true),
        internal: true,
        internal_dynamic_variant_config: Some(analyze_variant_config),
        ..Default::default()
    };

    let response = client
        .inference(params)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to call analyze function: {e}"))?;

    let InferenceResponse::Chat(chat_response) = &response else {
        return Err(anyhow::anyhow!(
            "analyze function is defined as Chat, cannot return JSON"
        ));
    };

    if chat_response.content.len() > 1 {
        tracing::warn!(
            "Analyze function returned {} content blocks, expected 1. Using first Text block.",
            chat_response.content.len()
        );
    }

    let text_block = chat_response.content.iter().find_map(|block| {
        if let ContentBlockChatOutput::Text(Text { text }) = block {
            Some(text.clone())
        } else {
            None
        }
    });

    let analysis_text = text_block.ok_or_else(|| {
        anyhow::anyhow!(
            "Expected at least one Text content block from analyze function, found none"
        )
    })?;

    tracing::debug!("Generated analysis: {}", analysis_text);

    let inference = if gepa_config.include_inference_for_mutation {
        let mut stored_input = eval_info
            .datapoint
            .input()
            .clone()
            .into_stored_input_without_file_handling()
            .map_err(|e| anyhow::anyhow!("Failed to convert input: {e}"))?;
        strip_signatures_from_stored_input(&mut stored_input);

        let output_value = match &eval_info.response {
            InferenceResponse::Chat(chat_response) => {
                let mut content = chat_response.content.clone();
                strip_signatures_from_chat_output(&mut content);
                serde_json::to_value(&content)?
            }
            InferenceResponse::Json(json_response) => serde_json::to_value(&json_response.output)?,
        };

        Some(Inference {
            input: stored_input,
            output: output_value,
        })
    } else {
        None
    };

    Ok(Analysis {
        inference,
        analysis: analysis_text,
    })
}

/// Reconstruct a `GEPAConfig` from the resolved config stored in checkpoints.
fn reconstruct_gepa_config(resolved: &ResolvedGEPAConfig) -> GEPAConfig {
    GEPAConfig {
        function_name: resolved.function_name.clone(),
        evaluation_name: resolved.evaluation_name.clone(),
        initial_variants: resolved.initial_variants.clone(),
        variant_prefix: resolved.variant_prefix.clone(),
        batch_size: resolved.batch_size,
        max_iterations: resolved.max_iterations,
        max_concurrency: resolved.max_concurrency,
        analysis_model: resolved.analysis_model.clone(),
        mutation_model: resolved.mutation_model.clone(),
        seed: resolved.seed,
        timeout: 300,
        include_inference_for_mutation: resolved.include_inference_for_mutation,
        retries: Default::default(),
        max_tokens: None,
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Helper struct to deserialize the mutation JSON response.
#[derive(Debug, Deserialize)]
struct MutateResponsePayload {
    templates: Vec<MutateTemplateEntry>,
}

#[derive(Debug, Deserialize)]
struct MutateTemplateEntry {
    name: String,
    content: String,
}

/// Parse the mutation inference response to extract child variant templates.
///
/// Mirrors the parsing logic in `mutate_variant()` (sequential path) but works
/// with the `InferenceResponse` returned by `TensorZeroClient::inference()`.
fn parse_mutate_response(
    response: &InferenceResponse,
    parent_name: &str,
    parent_config: &UninitializedChatCompletionConfig,
    gepa_config: &GEPAConfig,
    iteration: usize,
) -> anyhow::Result<Option<MutationResult>> {
    // Extract JSON content from the response
    let InferenceResponse::Json(json_response) = response else {
        return Err(anyhow::anyhow!(
            "mutate function is defined as Json, cannot return Chat"
        ));
    };

    // Try to get parsed output first, otherwise parse the raw output
    let output_value = json_response
        .output
        .parsed
        .clone()
        .or_else(|| {
            json_response
                .output
                .raw
                .as_ref()
                .and_then(|raw| serde_json::from_str(raw).ok())
        })
        .ok_or_else(|| anyhow::anyhow!("Mutate function returned no parsed or raw output"))?;

    // Deserialize the response
    let payload: MutateResponsePayload = from_value(output_value)
        .map_err(|e| anyhow::anyhow!("Failed to deserialize mutate output: {e}"))?;

    // Check for duplicate template names
    let template_names: Vec<&str> = payload.templates.iter().map(|t| t.name.as_str()).collect();
    let unique_names: std::collections::HashSet<&str> = template_names.iter().copied().collect();
    if template_names.len() != unique_names.len() {
        return Err(anyhow::anyhow!(
            "Mutate function returned duplicate template names"
        ));
    }

    // Convert to HashMap
    let templates: HashMap<String, String> = payload
        .templates
        .into_iter()
        .map(|entry| (entry.name, entry.content))
        .collect();

    // Validate all parent templates are present
    for parent_template_name in parent_config.templates.inner.keys() {
        if !templates.contains_key(parent_template_name) {
            return Err(anyhow::anyhow!(
                "Mutate function did not return template `{parent_template_name}` present in parent"
            ));
        }
    }

    // Validate no extra templates were added
    for template_name in templates.keys() {
        if !parent_config.templates.inner.contains_key(template_name) {
            return Err(anyhow::anyhow!(
                "Mutate function returned unexpected template `{template_name}` not present in parent"
            ));
        }
    }

    // Generate child variant name
    let child_name = format!(
        "{}-iter-{}-{}",
        gepa_config.variant_prefix.as_deref().unwrap_or("gepa"),
        iteration,
        parent_name
    );

    // Build child config from parent
    use tensorzero_core::config::path::ResolvedTomlPathData;
    use tensorzero_core::variant::chat_completion::UninitializedChatTemplate;

    let mut child_config = parent_config.clone();
    child_config.retries = gepa_config.retries;
    child_config.templates.inner.clear();

    for (template_name, content) in templates {
        child_config.templates.inner.insert(
            template_name.clone(),
            UninitializedChatTemplate {
                path: ResolvedTomlPathData::new_fake_path(
                    format!("gepa_mutated/{child_name}/{template_name}.minijinja"),
                    content,
                ),
            },
        );
    }

    Ok(Some(MutationResult {
        child_name,
        child_config,
    }))
}

/// Extract per-datapoint scores from an evaluation response.
fn extract_scores_from_response(response: &RunEvaluationResponse) -> VariantScores {
    let mut scores: VariantScores = HashMap::new();
    if let Some(datapoint_results) = &response.datapoint_results {
        for result in datapoint_results {
            if !result.success {
                continue;
            }
            let datapoint_scores: HashMap<EvaluatorName, Option<f32>> = result
                .evaluations
                .iter()
                .map(|(name, value)| (name.clone(), value.map(|v| v as f32)))
                .collect();
            scores.insert(result.datapoint_id, datapoint_scores);
        }
    }
    scores
}

/// Build statistics map from Pareto frontier for the output.
fn build_statistics(
    frontier: &ParetoFrontier,
    evaluator_configs: &HashMap<String, EvaluatorConfig>,
) -> HashMap<String, HashMap<String, GepaEvaluatorStats>> {
    let mut stats: HashMap<String, HashMap<String, GepaEvaluatorStats>> = HashMap::new();

    for variant_name in frontier.variant_configs().keys() {
        if let Some(scores) = frontier.variant_scores_map().get(variant_name) {
            let mut evaluator_stats: HashMap<String, GepaEvaluatorStats> = HashMap::new();

            for evaluator_name in evaluator_configs.keys() {
                let mut values: Vec<f64> = Vec::new();
                for datapoint_scores in scores.values() {
                    if let Some(Some(score)) = datapoint_scores.get(evaluator_name) {
                        values.push(*score as f64);
                    }
                }

                if !values.is_empty() {
                    let count = values.len();
                    let mean = values.iter().sum::<f64>() / count as f64;
                    let variance = if count > 1 {
                        values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (count - 1) as f64
                    } else {
                        0.0
                    };
                    let stdev = variance.sqrt();
                    evaluator_stats.insert(
                        evaluator_name.clone(),
                        GepaEvaluatorStats { mean, stdev, count },
                    );
                }
            }

            if !evaluator_stats.is_empty() {
                stats.insert(variant_name.clone(), evaluator_stats);
            }
        }
    }

    stats
}
