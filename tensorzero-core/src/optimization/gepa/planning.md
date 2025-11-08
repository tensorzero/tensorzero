# GEPA Optimizer Implementation Plan

This document outlines the implementation of the GEPA (Genetic Evolution with Pareto Analysis) optimization algorithm for TensorZero.

## Architecture Overview

GEPA uses TensorZero's built-in functions feature to simplify the implementation:

1. **Built-in Functions**: Two built-in functions (`tensorzero::optimization::gepa::analyze` and `tensorzero::optimization::gepa::mutate`) handle analysis and mutation using `internal_dynamic_variant_config`
2. **No Separate Clients**: All inference goes through the existing inference infrastructure - no need to spin up separate gateway instances
3. **Dynamic Variant Evaluation**: Candidate variants are evaluated using `internal_dynamic_variant_config` without modifying the actual config
4. **No Working Config**: Variants are tracked in a `HashMap` and passed dynamically - no need to maintain a separate filtered config
5. **Synchronous Operation**: GEPA completes during `launch()` like DICL (no async polling needed)
6. **Multiple Variant Output**: Returns a Pareto frontier of variants via new `OptimizerOutput::Variants` enum variant
7. **Optimizer Crate**: GEPA will be implemented in the `tensorzero-optimizers` crate (see below), which can depend on the `evaluations` crate for scoring variants

## Architecture Analysis: Optimizer Crate Extraction

### The Solution: GEPA in tensorzero-optimizers

GEPA requires running evaluations to score candidate variants during optimization. The `evaluations` crate depends on `tensorzero-core`, which would create a circular dependency if GEPA (inside core) tried to use evaluations directly.

**The elegant solution**: Extract optimizer implementations to a separate `tensorzero-optimizers` crate using Rust's orphan rule pattern.

**New architecture:**
```
tensorzero-core (types + HTTP handlers + re-exports)
      ↑ depends on
tensorzero-optimizers (traits + implementations including GEPA)
      ↑ depends on
evaluations
      ↑ depends on
tensorzero-core (types only)
```

### How It Works

**Key Insight**: Rust's orphan rule permits implementing a trait on a type when you own either the trait OR the type. We split optimizer code into:

1. **Types (stay in tensorzero-core)**:
   - `OptimizerConfig` enum (for serde deserialization)
   - `OptimizationJobHandle` enum
   - `GEPAConfig`, `GEPAJobHandle` structs
   - All config structs

2. **Behavior (moves to tensorzero-optimizers)**:
   - `trait Optimizer`, `trait JobHandle`
   - `impl Optimizer for GEPAConfig`
   - `impl JobHandle for GEPAJobHandle`
   - All helper functions and business logic

**Why this works:**
- ✅ Serde deserialization works (types in core, tagged enum intact)
- ✅ No circular dependency (optimizers → evaluations → core types only)
- ✅ GEPA can directly use evaluations crate (no subprocess needed)
- ✅ Clean separation of concerns

### Implementation Approach for GEPA

**GEPA will be implemented in `tensorzero-optimizers` crate:**

1. **Config types** in `tensorzero-core/src/optimization/gepa.rs`:
   - `GEPAConfig` (initialized config)
   - `UninitializedGEPAConfig` (deserializable config)
   - `GEPAJobHandle` (job handle returned by launch)

2. **Implementation** in `tensorzero-optimizers/src/gepa/`:
   - `impl Optimizer for GEPAConfig`
   - `impl JobHandle for GEPAJobHandle`
   - All helper functions (`evaluate_variants`, `mutate`, `filter_candidates`, etc.)
   - **Direct use of evaluations crate** for scoring variants

3. **Built-in functions** in `tensorzero-core/src/config/built_in.rs`:
   - `tensorzero::optimization::gepa::analyze` (chat function with tools)
   - `tensorzero::optimization::gepa::mutate` (JSON function for template generation)
   - Config files embedded from `tensorzero-core/src/optimization/gepa/config/`

**Benefits:**
- ✅ GEPA can call evaluations directly (no subprocess overhead)
- ✅ Clean architecture with proper separation of concerns
- ✅ Evaluations remain independently testable
- ✅ No complex orchestration or IPC needed
- ✅ Type safety maintained throughout

### Implementation Prerequisites

**GEPA implementation depends on optimizer extraction:**

1. **Blocking dependency**: Optimizer crate extraction must be completed first
   - Detailed plan: `tensorzero-core/src/optimization/OPTIMIZER_CRATE_EXTRACTION.md`
   - Estimated effort: 2-4 hours
   - Target branch: `andrew/optimizer-crate-extraction`

2. **After extraction completes**:
   - GEPA types can be added to `tensorzero-core/src/optimization/gepa.rs`
   - GEPA implementation goes in `tensorzero-optimizers/src/gepa/`
   - GEPA can import from `evaluations` crate directly

3. **No workarounds needed**:
   - No subprocess spawning
   - No IPC mechanisms
   - No evaluation logic duplication
   - Clean, idiomatic Rust code

**Timeline**: GEPA implementation can begin immediately after optimizer extraction PR merges.

## GEPAConfig

```rust
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(test, derive(ts_rs::TS))]
#[cfg_attr(test, ts(export))]
pub struct GEPAConfig {
    /// Name of the function being optimized
    pub function_name: String,

    /// Name of the evaluation used to score candidate variants
    pub evaluation_name: String,

    /// Optional list of variant_names to initialize GEPA with.
    /// If None, will use all variants defined for the function.
    pub initial_variants: Option<Vec<String>>,

    /// Prefix for the name of the new optimized variants
    pub variant_prefix: Option<String>,

    /// Number of training samples to analyze per iteration
    pub batch_size: usize,

    /// Maximum number of training iterations (default: 1)
    pub max_iterations: u32,

    /// Maximum number of concurrent inference calls (default: 10)
    pub max_concurrency: u32,

    /// Model for analysis/prediction (e.g., "openai::gpt-5-mini")
    pub analysis_model: String,

    /// Model for mutation (e.g., "openai::gpt-5")
    pub mutation_model: String,

    /// Optional random seed for reproducibility
    pub seed: Option<u32>,

    /// Client timeout in seconds for TensorZero gateway operations (default: 300)
    pub timeout: u64,
}

fn default_batch_size() -> usize {
    5
}

fn default_max_iterations() -> u32 {
    1
}

fn default_max_concurrency() -> u32 {
    10
}

fn default_timeout() -> u64 {
    300
}

fn default_analysis_model() -> String {
    "openai::gpt-5-mini".to_string()
}

fn default_mutation_model() -> String {
    "openai::gpt-5".to_string()
}

#[cfg_attr(test, derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, Serialize)]
#[cfg_attr(test, ts(export))]
#[cfg_attr(feature = "pyo3", pyclass(str, name = "GEPAConfig"))]
pub struct UninitializedGEPAConfig {
    pub function_name: String,
    pub evaluation_name: String,
    pub initial_variants: Option<Vec<String>>,
    pub variant_prefix: Option<String>,

    #[serde(default = "default_batch_size")]
    pub batch_size: usize,

    #[serde(default = "default_max_iterations")]
    pub max_iterations: u32,

    #[serde(default = "default_max_concurrency")]
    pub max_concurrency: u32,

    #[serde(default = "default_analysis_model")]
    pub analysis_model: String,

    #[serde(default = "default_mutation_model")]
    pub mutation_model: String,

    pub seed: Option<u32>,

    #[serde(default = "default_timeout")]
    pub timeout: u64,
}

impl Default for UninitializedGEPAConfig {
    fn default() -> Self {
        Self {
            function_name: String::new(),
            evaluation_name: String::new(),
            initial_variants: None,
            variant_prefix: None,
            batch_size: default_batch_size(),
            max_iterations: default_max_iterations(),
            max_concurrency: default_max_concurrency(),
            analysis_model: default_analysis_model(),
            mutation_model: default_mutation_model(),
            seed: None,
            timeout: default_timeout(),
        }
    }
}

impl std::fmt::Display for UninitializedGEPAConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}
```

## GEPAJobHandle

```rust
#[cfg_attr(test, derive(ts_rs::TS))]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(test, ts(export))]
#[cfg_attr(feature = "pyo3", pyclass(str))]
pub struct GEPAJobHandle {
    /// Map of variant names to their configurations
    /// This represents the final Pareto frontier of optimized variants
    pub variant_configs: HashMap<String, UninitializedChatCompletionConfig>,
}

impl std::fmt::Display for GEPAJobHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

impl JobHandle for GEPAJobHandle {
    async fn poll(
        &self,
        client: &TensorzeroHttpClient,
        credentials: &InferenceCredentials,
        _default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<OptimizationJobInfo, Error> {
        // GEPA optimization is synchronous, so it's always complete once launched
        // This matches the Python implementation where the job completes in launch()
        let _ = (client, credentials);

        // Return the Pareto frontier of variant configurations
        // TODO: OptimizerOutput needs new variant added:
        //   Variants(HashMap<String, UninitializedVariantConfig>)
        // Current OptimizerOutput only supports single Variant or Model
        Ok(OptimizationJobInfo::Completed {
            output: OptimizerOutput::Variants(self.variant_configs.clone()),
        })
    }
}
```

## Optimizer

```rust
impl Optimizer for GEPAConfig {
    type Handle = GEPAJobHandle;

    async fn launch(
        &self,
        client: &TensorzeroHttpClient,
        train_examples: Vec<RenderedSample>,
        val_examples: Option<Vec<RenderedSample>>,
        credentials: &InferenceCredentials,
        clickhouse_connection_info: &ClickHouseConnectionInfo,
        config: &Config,
    ) -> Result<Self::Handle, Error> {
        // Generate job ID
        let job_id = uuid7();
        let job_id_short = extract_last_6_hex_chars(&job_id);

        // set seed

        // Validate validation examples (required for GEPA)
        let val_examples = val_examples.ok_or_else(|| {
            Error::new("val_examples are required for GEPA optimization")
        })?;
        }
        // Validate and filter examples
        // Raises error if all examples are dropped
        let train_examples = validate_and_filter_examples(train_examples)?;
        let val_examples = validate_and_filter_examples(val_examples)?;

        // Validate config (function and evaluation exist)
        validate_config(
            config,
            &self.function_name,
            &self.evaluation_name,
        )?;

        // Check if ClickHouse is available (required for GEPA)
        if clickhouse_connection_info.client_type() == ClickHouseClientType::Disabled {
            return Err(Error::new(ErrorDetails::AppState {
                message: "GEPA requires ClickHouse to be enabled for evaluation and storage"
                    .to_string(),
            }));
        }

        // Built-in functions (tensorzero::optimization::gepa::analyze and
        // tensorzero::optimization::gepa::mutate) are used for analysis and mutation.
        // They are called with internal_dynamic_variant_config at runtime.
        //
        // The function being optimized is also called with internal_dynamic_variant_config
        // for each candidate variant during evaluation.
        //
        // No separate clients are needed - all inference goes through the existing
        // inference infrastructure.

        // Get candidate variants from config (filtered by initial_variants if specified)
        let mut candidate_variants: HashMap<String, UninitializedVariantInfo> =
            get_variants_from_config(config, &self.function_name, &self.initial_variants)?;

        // Create validation dataset
        let val_dataset_name = format!("gepa_pareto_data_job_{job_id_short}");
        create_evaluation_dataset(
            clickhouse_connection_info,
            &val_examples,
            &val_dataset_name,
        )?;

        // Evaluate initial variants on validation set (cache these scores)
        let initial_val_scores: HashMap<String, EvaluationResults> = evaluate_variants(
            client,
            credentials,
            clickhouse_connection_info,
            config,
            &self.function_name,
            &self.evaluation_name,
            &candidate_variants,
            &val_dataset_name,
            self.max_concurrency,
        )?;

        // Filter out variants that failed to evaluate
        let mut val_scores: HashMap<String, EvaluationResults> = initial_val_scores
            .into_iter()
            .filter_map(|(name, result)| result.map(|r| (name, r)))
            .collect();

        // Filter candidates before starting iterations (instance-wise Pareto dominance)
        let (mut candidate_variants, mut frequencies) = filter_candidates(
            candidate_variants,
            &val_scores,
            config,
            &self.evaluation_name,
        )?;

        // Prune val_scores to only contain filtered candidates
        val_scores.retain(|k, _| candidate_variants.contains_key(k));

        for i in 0..self.max_iterations {

            // Sample mini-batch from training examples
            let batch_examples = random_sample(
                &train_examples,
                min(self.batch_size, train_examples.len()),
            );
            let batch_dataset_name = format!("gepa_batch_{}_job_{}", i, job_id_short);
            create_evaluation_dataset(
                clickhouse_connection_info,
                &batch_examples,
                &batch_dataset_name,
            )?;

            // Sample variant to mutate (proportional to instance-wise Pareto frequency)
            let candidate_variant_name = sample_by_frequency(&frequencies)?;

            // Generate mutation name
            let prefix = self.variant_prefix.as_ref()
                .map(|p| format!("{}_", p))
                .unwrap_or_default();
            let candidate_mutation_name = format!(
                "{}gepa_a_{}_m_{}_job_{}_step_{}",
                prefix,
                self.analysis_model.replace("::", "_"),
                self.mutation_model.replace("::", "_"),
                job_id_short,
                i
            );

            // Mutate the selected variant
            let candidate_variant = candidate_variants.get(&candidate_variant_name).unwrap();
            let candidate_mutation: Option<UninitializedVariantInfo> = mutate(
                client,
                credentials,
                clickhouse_connection_info,
                config,
                &self.function_name,
                candidate_variant,
                &batch_examples,
                &self.analysis_model,
                &self.mutation_model,
                &job_id,
                self.max_concurrency,
            );

            // Skip if mutation failed
            let Some(candidate_mutation) = candidate_mutation else {
                log::warn!("Iteration {}/{}: Mutation failed, skipping",
                          i + 1, self.max_iterations);
                continue;
            };

            // Add mutation to candidate variants (no config rebuild needed)
            candidate_variants.insert(candidate_mutation_name.clone(), candidate_mutation);

            // Evaluate both original and mutation on batch
            let batch_variant_configs: HashMap<String, UninitializedVariantInfo> = [
                (candidate_variant_name.clone(), candidate_variants[&candidate_variant_name].clone()),
                (candidate_mutation_name.clone(), candidate_variants[&candidate_mutation_name].clone()),
            ].into_iter().collect();

            let batch_scores = evaluate_variants(
                client,
                credentials,
                clickhouse_connection_info,
                config,
                &self.function_name,
                &self.evaluation_name,
                &batch_variant_configs,
                &batch_dataset_name,
                self.max_concurrency,
            )?;

            // Check if mutation is an improvement (global Pareto dominance on batch)
            if is_improvement(
                &batch_scores,
                &candidate_variant_name,
                &candidate_mutation_name,
                config,
                &self.evaluation_name,
            )? {
                // Accept mutation - evaluate on validation set
                let mutation_variant_map: HashMap<String, UninitializedVariantInfo> =
                    [(candidate_mutation_name.clone(), candidate_variants[&candidate_mutation_name].clone())]
                    .into_iter().collect();

                let mutation_val_score = evaluate_variants(
                    client,
                    credentials,
                    clickhouse_connection_info,
                    config,
                    &self.function_name,
                    &self.evaluation_name,
                    &mutation_variant_map,
                    &val_dataset_name,
                    self.max_concurrency,
                )?;
                val_scores.extend(mutation_val_score);
            } else {
                // Reject mutation (not an improvement on batch)
                log::info!("Iteration {}/{}: Rejecting mutation '{}'",
                          i + 1, self.max_iterations, candidate_mutation_name);

                // Remove mutation (config update and client rebuild happen after filtering)
                candidate_variants.remove(&candidate_mutation_name);
            }

            // Filter candidates at end of iteration (instance-wise Pareto dominance)
            let (filtered_variants, new_frequencies) = filter_candidates(
                candidate_variants,
                &val_scores,
                config,
                &self.evaluation_name,
            )?;
            candidate_variants = filtered_variants;
            frequencies = new_frequencies;

            // Prune val_scores
            val_scores.retain(|k, _| candidate_variants.contains_key(k));

        }

        // Convert final candidate variants to UninitializedChatCompletionConfig format
        let variant_configs: HashMap<String, UninitializedChatCompletionConfig> =
            convert_variants_to_output(
                &candidate_variants,
                config,
                &self.function_name,
            )?;

        // Return job handle with final Pareto frontier
        // GEPA is synchronous, so polling immediately returns Completed
        Ok(GEPAJobHandle {
            variant_configs,
        })

    }
}
```

## Helper Functions and Data Types

### validate_and_filter_examples

```rust
/// Validate and filter examples, dropping invalid samples
/// Returns error if ALL examples are invalid
fn validate_and_filter_examples(
    examples: Vec<RenderedSample>
) -> Result<Vec<RenderedSample>, Error>;
```

### validate_config

```rust
/// Validate that the function and evaluation exist in the config
///
/// Validates:
/// - function_name exists in config.functions
/// - evaluation_name exists in config.evaluations
///
/// Returns error if function or evaluation not found
fn validate_config(
    config: &Config,
    function_name: &str,
    evaluation_name: &str,
) -> Result<(), Error>;
```

### get_variants_from_config

```rust
/// Extract variants from config for a given function
/// If initial_variants is specified, only include those variants
/// Returns HashMap<variant_name, variant_config>
/// Returns error if function not found or if specified initial_variants don't exist
fn get_variants_from_config(
    config: &Config,
    function_name: &str,
    initial_variants: &Option<Vec<String>>,
) -> Result<HashMap<String, UninitializedVariantInfo>, Error>;
```

### create_evaluation_dataset

```rust
/// Create evaluation dataset from samples in ClickHouse
/// Uses the existing dataset storage infrastructure
fn create_evaluation_dataset(
    clickhouse: &ClickHouseConnectionInfo,
    samples: &[RenderedSample],
    dataset_name: &str,
) -> Result<(), Error>;
```

### evaluate_variants

```rust
/// Evaluate multiple variants on a dataset
/// Returns HashMap<variant_name, Option<evaluation_results>>
/// None indicates evaluation failure for that variant (graceful degradation)
///
/// Uses internal_dynamic_variant_config to evaluate each variant without
/// modifying the actual config. Each variant is run against the dataset
/// and evaluated using the specified evaluation.
fn evaluate_variants(
    client: &TensorzeroHttpClient,
    credentials: &InferenceCredentials,
    clickhouse: &ClickHouseConnectionInfo,
    config: &Config,
    function_name: &str,
    evaluation_name: &str,
    variant_configs: &HashMap<String, UninitializedVariantInfo>,
    dataset_name: &str,
    max_concurrency: u32,
) -> Result<HashMap<String, Option<EvaluationResults>>, Error>;
```

### EvaluationResults

Lightweight structure for managing evaluation outputs

```rust
#[derive(Clone, Debug)]
pub struct EvaluationResults {
    /// Per-datapoint evaluation results
    /// Outer key: datapoint_id (String)
    /// Inner key: evaluator_name (String)
    /// Value: Option<f32> - None if evaluation failed for that datapoint
    ///
    /// Example:
    /// {
    ///   "dp_001": {"exact_match": Some(1.0), "rouge": Some(0.85)},
    ///   "dp_002": {"exact_match": Some(0.0), "rouge": None},  // rouge eval failed
    /// }
    ///
    /// Used by filter_candidates() to compute instance-wise Pareto dominance.
    pub per_datapoint: HashMap<String, HashMap<String, Option<f32>>>,

    /// Aggregated statistics across all datapoints
    /// Key: evaluator_name
    /// Value: MetricStats with mean/stderr/count
    ///
    /// Example:
    /// {
    ///   "exact_match": MetricStats { mean: 0.85, stderr: 0.02, count: 100 },
    ///   "rouge": MetricStats { mean: 0.73, stderr: 0.03, count: 98 },  // 2 failed
    /// }
    ///
    /// Used for logging, debugging, and potentially global Pareto comparisons.
    pub metrics: HashMap<String, MetricStats>,
}

/// Statistics for a single metric/evaluator across a dataset
#[derive(Clone, Debug)]
pub struct MetricStats {
    /// Mean value of the metric across all samples
    pub mean: f32,

    /// Standard error of the mean
    pub stderr: f32,

    /// Number of samples evaluated
    pub count: usize,
}
```

### filter_candidates

```rust
/// Filter candidates using instance-wise Pareto dominance
/// Returns (filtered_variants, frequencies) where frequencies track
/// how many instances each variant dominates on
///
/// Uses the evaluation config to determine objective directions (higher_is_better)
fn filter_candidates(
    candidates: HashMap<String, UninitializedVariantInfo>,
    scores: &HashMap<String, EvaluationResults>,
    config: &Config,
    evaluation_name: &str,
) -> Result<(HashMap<String, UninitializedVariantInfo>, HashMap<String, usize>), Error>;
```

### sample_by_frequency

```rust
/// Sample a variant name proportional to its frequency
fn sample_by_frequency(
    frequencies: &HashMap<String, usize>
) -> Result<String, Error>;
```

### mutate

```rust
/// Generate mutation of a variant using GEPA analysis and mutation
/// Returns None if mutation generation fails
///
/// Process:
/// 1. Run inference on batch_examples using candidate variant (with internal_dynamic_variant_config)
/// 2. Analyze each inference using tensorzero::optimization::gepa::analyze built-in function
///    (with internal_dynamic_variant_config for the analysis model)
/// 3. Generate mutation using tensorzero::optimization::gepa::mutate built-in function
///    (with internal_dynamic_variant_config for the mutation model)
/// 4. Return new variant config with updated templates
///
/// Built-in functions are defined in `built_in.rs` and their templates/tools are
/// loaded from the embedded config files in `gepa/config/`.
fn mutate(
    client: &TensorzeroHttpClient,
    credentials: &InferenceCredentials,
    clickhouse: &ClickHouseConnectionInfo,
    config: &Config,
    function_name: &str,
    variant_config: &UninitializedVariantInfo,
    batch_examples: &[RenderedSample],
    analysis_model: &str,
    mutation_model: &str,
    job_id: &Uuid,
    max_concurrency: u32,
) -> Option<UninitializedVariantInfo>;
```

### is_improvement

```rust
/// Check if mutation improves over original variant (global Pareto dominance)
/// Returns false if either variant is missing from scores (evaluation failed)
///
/// Uses the evaluation config to determine objective directions (higher_is_better)
fn is_improvement(
    scores: &HashMap<String, EvaluationResults>,
    original_variant: &str,
    mutation_variant: &str,
    config: &Config,
    evaluation_name: &str,
) -> Result<bool, Error>;
```

### convert_variants_to_output

```rust
/// Convert candidate variants to output format
fn convert_variants_to_output(
    variants: &HashMap<String, UninitializedVariantInfo>,
    config: &Config,
    function_name: &str,
) -> Result<HashMap<String, UninitializedChatCompletionConfig>, Error>;
```

## Built-in Functions

GEPA uses two built-in functions that must be defined in `tensorzero-core/src/config/built_in.rs`:

### tensorzero::optimization::gepa::analyze

A Chat function with tools for analyzing inference outputs.

**Configuration:**
- Type: Chat
- Tools: `report_error`, `report_improvement`, `report_optimal`
- Tool definitions loaded from `tensorzero-core/src/optimization/gepa/config/tools/`
- User schema loaded from `tensorzero-core/src/optimization/gepa/config/functions/analyze/user_schema.json`
- Templates loaded from `tensorzero-core/src/optimization/gepa/config/functions/analyze/`

**Usage:**
Called with `internal_dynamic_variant_config` specifying the analysis model (e.g., `openai::gpt-5-mini`).

### tensorzero::optimization::gepa::mutate

A JSON function for generating improved templates based on analysis feedback.

**Configuration:**
- Type: JSON
- Output schema loaded from `tensorzero-core/src/optimization/gepa/config/functions/mutate/output_schema.json`
- User schema loaded from `tensorzero-core/src/optimization/gepa/config/functions/mutate/user_schema.json`
- Templates loaded from `tensorzero-core/src/optimization/gepa/config/functions/mutate/`

**Usage:**
Called with `internal_dynamic_variant_config` specifying the mutation model (e.g., `openai::gpt-5`).

### Implementation Notes

- Both functions are variant-less (empty variants HashMap)
- Templates and schemas must be embedded using `include_str!()` macros
- Tool definitions for analyze function must be parsed from JSON at runtime
- These functions can only be used with `internal_dynamic_variant_config` (they fail without it)

## Modifications to Existing Code

### OptimizerOutput

**Location:** `tensorzero-core/src/optimization/mod.rs`

**Issue:** Current `OptimizerOutput` enum only supports returning a single variant or model

**Proposed Change:** Add new enum variant to support multiple variants

```rust
/// Add Variants to OptimizerOutput enum
pub enum OptimizerOutput {
		Variant(Box<UninitializedVariantConfig>),
		Model(UninitializedModelConfig),
		Variants(HashMap<String, UninitializedVariantConfig>),  // NEW
}
```

**Impact:**

- Backward compatible (existing optimizers continue using `Variant` or `Model`)
- Any code that matches on `OptimizerOutput` must add handling for new `Variants` case
