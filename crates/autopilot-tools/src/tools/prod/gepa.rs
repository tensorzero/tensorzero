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
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use durable_tools::{
    Heartbeater, RunEvaluationParams, RunEvaluationResponse, StepState, TaskTool, ToolAppState,
    ToolContext, ToolMetadata, ToolResult,
};
use rand::seq::{IndexedRandom, SliceRandom};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use tensorzero_core::cache::CacheEnabledMode;
use tensorzero_core::client::ClientInferenceParams;
use tensorzero_core::config::{
    UninitializedConfig, UninitializedVariantConfig, UninitializedVariantInfo,
};
use tensorzero_core::endpoints::datasets::v1::types::ListDatapointsRequest;
use tensorzero_core::endpoints::inference::InferenceResponse;
use tensorzero_core::evaluations::{EvaluationConfig, EvaluatorConfig};
use tensorzero_core::optimization::gepa::{GEPAConfig, GepaEvaluatorStats};
use uuid::Uuid;

use autopilot_client::AutopilotSideInfo;
use tensorzero_optimizers::gepa::GepaClient;
use tensorzero_optimizers::gepa::durable::types::{
    EvalAnalyzeMutateResult, EvalAnalyzeMutateStepParams, EvalResult, EvalStepParams,
    GepaToolOutput, GepaToolParams, InitEvalStepParams, MutationResult, ResolvedGEPAConfig,
    SetupResult,
};
use tensorzero_optimizers::gepa::evaluate::{EvaluatorName, VariantName, VariantScores};
use tensorzero_optimizers::gepa::pareto::{Candidate, ParetoFrontier, is_improvement};

use crate::error::AutopilotToolError;

/// Adapter that implements [`GepaClient`] for the durable `TensorZeroClient` trait.
///
/// Wraps `&dyn TensorZeroClient` so the shared GEPA functions
/// (`analyze_inferences`, `mutate_variant`) can be called from the durable path.
struct DurableGepaClient<'a>(&'a dyn durable_tools::TensorZeroClient);

impl GepaClient for DurableGepaClient<'_> {
    async fn inference(
        &self,
        params: ClientInferenceParams,
    ) -> Result<InferenceResponse, tensorzero_core::error::Error> {
        self.0.inference(params).await.map_err(|e| {
            tensorzero_core::error::Error::new(tensorzero_core::error::ErrorDetails::Inference {
                message: format!("{e}"),
            })
        })
    }
}

// ── Internal step params ────────────────────────────────────────────────

/// Wraps `GepaToolParams` with durable-generated values for the setup step.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct SetupStepParams {
    llm_params: GepaToolParams,
    run_id: Uuid,
    rng_seed: u64,
}

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
    // Generate durable values before the setup step so they are
    // deterministic across task re-runs.
    let run_id = ctx.uuid7().await?;
    let rng_seed: u64 = match llm_params.seed {
        Some(seed) => seed as u64,
        None => ctx.rand().await?.to_bits(),
    };

    // ── Step "setup": validate config, resolve datasets ─────────
    let setup_params = SetupStepParams {
        llm_params: llm_params.clone(),
        run_id,
        rng_seed,
    };
    let setup: SetupResult = ctx.step("setup", setup_params, setup_step).await?;

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
    let init_scores: BTreeMap<VariantName, VariantScores> = ctx
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
async fn setup_step(
    params: SetupStepParams,
    step_state: StepState<ToolAppState>,
) -> anyhow::Result<SetupResult> {
    let run_id = params.run_id;
    let rng_seed = params.rng_seed;
    let params = params.llm_params;
    let client = step_state.state.t0_client();

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

    // Resolve datasets and datapoint IDs.
    // Two modes:
    //   1. Separate datasets: `train_dataset_name` + `val_dataset_name`
    //   2. Auto-split: single `dataset_name` shuffled and split 50/50
    let max_datapoints = params.max_datapoints.unwrap_or(1000);
    let fetch_limit = max_datapoints.saturating_add(1);

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

            if train_name == val_name {
                tracing::warn!(
                    "Train and validation datasets resolve to the same name `{train_name}`. \
                     Consider using separate datasets or `dataset_name` for auto-split."
                );
            }

            let train_datapoints = client
                .list_datapoints(
                    train_name.clone(),
                    ListDatapointsRequest {
                        limit: Some(fetch_limit),
                        ..Default::default()
                    },
                )
                .await
                .map_err(|e| anyhow::anyhow!("Failed to list training datapoints: {e}"))?;

            let mut train_ids: Vec<Uuid> =
                train_datapoints.datapoints.iter().map(|d| d.id()).collect();
            if train_ids.is_empty() {
                return Err(anyhow::anyhow!(
                    "Training dataset `{train_name}` contains no datapoints"
                ));
            }
            if train_ids.len() > max_datapoints as usize {
                tracing::warn!(
                    "Training dataset `{train_name}` has more than {max_datapoints} datapoints, truncating"
                );
                train_ids.truncate(max_datapoints as usize);
            }

            let val_datapoints = client
                .list_datapoints(
                    val_name.clone(),
                    ListDatapointsRequest {
                        limit: Some(fetch_limit),
                        ..Default::default()
                    },
                )
                .await
                .map_err(|e| anyhow::anyhow!("Failed to list val datapoints: {e}"))?;

            let mut val_ids: Vec<Uuid> = val_datapoints.datapoints.iter().map(|d| d.id()).collect();
            if val_ids.is_empty() {
                return Err(anyhow::anyhow!(
                    "Validation dataset `{val_name}` contains no datapoints"
                ));
            }
            if val_ids.len() > max_datapoints as usize {
                tracing::warn!(
                    "Validation dataset `{val_name}` has more than {max_datapoints} datapoints, truncating"
                );
                val_ids.truncate(max_datapoints as usize);
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
                        limit: Some(fetch_limit),
                        ..Default::default()
                    },
                )
                .await
                .map_err(|e| anyhow::anyhow!("Failed to list datapoints: {e}"))?;

            let mut all_ids: Vec<Uuid> = all_datapoints.datapoints.iter().map(|d| d.id()).collect();
            if all_ids.len() > max_datapoints as usize {
                tracing::warn!(
                    "Dataset `{dataset_name}` has more than {max_datapoints} datapoints, truncating"
                );
                all_ids.truncate(max_datapoints as usize);
            }
            if all_ids.len() < 2 {
                return Err(anyhow::anyhow!(
                    "Dataset `{dataset_name}` needs at least 2 datapoints for auto-split"
                ));
            }

            // Shuffle deterministically using the durable rng_seed
            let mut split_rng = StdRng::seed_from_u64(rng_seed);
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
    step_state: StepState<ToolAppState>,
) -> anyhow::Result<BTreeMap<VariantName, VariantScores>> {
    let heartbeater: Arc<dyn Heartbeater> = step_state.heartbeater.clone();
    let client = step_state.state.t0_client();
    let mut all_scores: BTreeMap<VariantName, VariantScores> = BTreeMap::new();

    for (variant_name, variant_config) in &params.variants {
        let variant_info = UninitializedVariantInfo {
            inner: UninitializedVariantConfig::ChatCompletion(variant_config.clone()),
            timeouts: None,
            namespace: None,
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

        match client
            .run_evaluation(eval_params, heartbeater.clone())
            .await
        {
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
    step_state: StepState<ToolAppState>,
) -> anyhow::Result<Option<EvalResult>> {
    let heartbeater: Arc<dyn Heartbeater> = step_state.heartbeater.clone();
    let client = step_state.state.t0_client();

    let variant_info = UninitializedVariantInfo {
        inner: UninitializedVariantConfig::ChatCompletion(params.variant_config),
        timeouts: None,
        namespace: None,
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

    match client
        .run_evaluation(eval_params, heartbeater.clone())
        .await
    {
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
    step_state: StepState<ToolAppState>,
) -> anyhow::Result<Option<EvalAnalyzeMutateResult>> {
    let heartbeater: Arc<dyn Heartbeater> = step_state.heartbeater.clone();
    let client = step_state.state.t0_client();

    // ── 1. Evaluate parent with evaluation_infos ────────────────
    let variant_info = UninitializedVariantInfo {
        inner: UninitializedVariantConfig::ChatCompletion(params.variant_config.clone()),
        timeouts: None,
        namespace: None,
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

    let response = match client
        .run_evaluation(eval_params, heartbeater.clone())
        .await
    {
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
    let gepa_client = DurableGepaClient(&**client);
    let analyses = match tensorzero_optimizers::gepa::analyze::analyze_inferences(
        &gepa_client,
        &evaluation_infos,
        &function_context,
        &params.variant_config,
        &gepa_config,
    )
    .await
    {
        Ok(a) => a,
        Err(e) => {
            tracing::warn!(
                "GEPA iteration {}: analysis failed for parent `{}`: {e}. Skipping iteration.",
                params.iteration,
                params.variant_name,
            );
            return Ok(None);
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
    let parent = tensorzero_optimizers::gepa::GEPAVariant {
        name: params.variant_name.clone(),
        config: params.variant_config.clone(),
    };
    let mutation = match tensorzero_optimizers::gepa::mutate::mutate_variant(
        &gepa_client,
        &analyses,
        &function_context,
        &parent,
        &gepa_config,
        params.iteration as usize,
    )
    .await
    {
        Ok(child) => Some(MutationResult {
            child_name: child.name,
            child_config: child.config,
        }),
        Err(e) => {
            tracing::warn!(
                "GEPA iteration {}: mutation failed for parent `{}`: {e}",
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
