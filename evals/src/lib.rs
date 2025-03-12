use std::io::Write;
use std::sync::Arc;
use std::{collections::HashMap, path::PathBuf};

use anyhow::{anyhow, bail, Result};
use clap::Parser;
use dataset::query_dataset;
use evaluators::evaluate_inference;
use helpers::{get_tool_params_args, resolved_input_to_input, setup_logging};
use stats::{EvalInfo, EvalStats};
use tensorzero::{
    CacheParamsOptions, Client, ClientBuilder, ClientBuilderMode, ClientInferenceParams,
    DynamicToolParams, FeedbackParams, InferenceOutput, InferenceParams, InferenceResponse,
};
use tensorzero_internal::cache::CacheEnabledMode;
use tensorzero_internal::evals::EvaluatorConfig;
use tensorzero_internal::{
    clickhouse::ClickHouseConnectionInfo, config_parser::Config, endpoints::datasets::Datapoint,
    function::FunctionConfig,
};
use tokio::{sync::Semaphore, task::JoinSet};
use url::Url;
use uuid::Uuid;

pub mod dataset;
pub mod evaluators;
pub mod helpers;
pub mod stats;

const CACHE_OPTIONS: CacheParamsOptions = CacheParamsOptions {
    enabled: CacheEnabledMode::On,
    max_age_s: None,
};

#[derive(clap::ValueEnum, Clone, Debug, Default, PartialEq)]
pub enum OutputFormat {
    Jsonl,
    #[default]
    HumanReadable,
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
    /// Path to tensorzero.toml.
    #[arg(short, long, default_value = "./config/tensorzero.toml")]
    pub config_file: PathBuf,

    /// URL of a running TensorZero HTTP gateway server to use for requests. This runs evals using that gateway.
    #[arg(short, long)]
    pub gateway_url: Option<Url>,

    /// Name of the eval to run.
    #[arg(short, long)]
    pub name: String,

    /// Name of the variant to run.
    #[arg(short, long)]
    pub variant: String,

    /// Number of concurrent requests to make.
    #[arg(short, long, default_value = "1")]
    pub concurrency: usize,

    #[arg(short, long, default_value = "human-readable")]
    pub format: OutputFormat,
}

pub async fn run_eval(args: Args, eval_run_id: Uuid, mut writer: impl Write) -> Result<()> {
    setup_logging(&args)?;
    let semaphore = Semaphore::new(args.concurrency);
    let clickhouse_url = std::env::var("TENSORZERO_CLICKHOUSE_URL")
        .map_err(|_| anyhow!("Missing ClickHouse URL at TENSORZERO_CLICKHOUSE_URL"))?;

    let config = Config::load_and_verify_from_path(&args.config_file).await?;
    let eval_config = config
        .evals
        .get(&args.name)
        .ok_or(anyhow!("Eval not found"))?;
    let function_config = config.get_function(&eval_config.function_name)?;
    #[allow(unused)]
    let tensorzero_client = match args.gateway_url {
        Some(gateway_url) => {
            ClientBuilder::new(ClientBuilderMode::HTTPGateway { url: gateway_url })
        }
        None => ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
            config_file: Some(args.config_file),
            clickhouse_url: Some(clickhouse_url.clone()),
            timeout: None,
        }),
    }
    .build()
    .await
    .map_err(|e| anyhow!("Failed to build client: {}", e))?;
    let tensorzero_client_with_semaphore =
        Arc::new(ThrottledTensorZeroClient::new(tensorzero_client, semaphore));

    let clickhouse_client = ClickHouseConnectionInfo::new(&clickhouse_url).await?;
    let mut join_set = JoinSet::new();

    let dataset = query_dataset(
        &clickhouse_client,
        &eval_config.dataset_name,
        &eval_config.function_name,
        function_config,
    )
    .await?;
    let variant = Arc::new(args.variant);
    let eval_name = Arc::new(args.name);
    let dataset_len = dataset.len();

    // Spawn concurrent tasks for each datapoint
    for datapoint in dataset {
        let client_clone = tensorzero_client_with_semaphore.clone();
        let variant = variant.clone();
        let function_config = function_config.clone();
        let eval_config = eval_config.clone();
        let eval_name = eval_name.clone();
        let eval_run_id_clone = eval_run_id;
        let datapoint = Arc::new(datapoint);

        join_set.spawn(async move {
            let inference_response = Arc::new(
                infer_datapoint(
                    &client_clone,
                    &eval_config.function_name,
                    &variant,
                    eval_run_id_clone,
                    &datapoint,
                    &function_config,
                )
                .await?,
            );

            let eval_result = evaluate_inference(
                inference_response.clone(),
                datapoint.clone(),
                eval_config,
                eval_name,
                client_clone,
                eval_run_id_clone,
            )
            .await?;

            Ok::<(Datapoint, InferenceResponse, evaluators::EvalResult), anyhow::Error>((
                Arc::into_inner(datapoint).ok_or_else(|| anyhow!("Failed to get datapoint for datapoint. This should never happen. Please file a bug report at https://github.com/tensorzero/tensorzero/discussions/categories/bug-reports."))?,
                Arc::into_inner(inference_response).ok_or_else(|| anyhow!("Failed to get inference response for datapoint. This should never happen. Please file a bug report at https://github.com/tensorzero/tensorzero/discussions/categories/bug-reports."))?,
                eval_result,
            ))
        });
    }

    // Collect results
    let mut eval_stats = EvalStats::new(args.format, dataset_len);

    while let Some(result) = join_set.join_next().await {
        match result {
            Ok(Ok((datapoint, inference_response, eval_result))) => {
                eval_stats.push(
                    EvalInfo::new(datapoint, inference_response, eval_result),
                    &mut writer,
                )?;
            }
            Ok(Err(e)) => {
                tracing::warn!("Task error: {}", e);
            }
            Err(e) => {
                tracing::warn!("Failed to join task: {}", e);
            }
        }
    }

    if let Some(progress_bar) = &eval_stats.progress_bar {
        progress_bar.finish_with_message("Done");
    }

    if eval_stats.output_format == OutputFormat::HumanReadable {
        let stats = eval_stats.compute_stats(&eval_config.evaluators);

        // Print all stats
        for (evaluator_name, evaluator_stats) in &stats {
            writeln!(writer, "{}: {}", evaluator_name, evaluator_stats)?;
        }

        // Check cutoffs and handle failures
        let failures = check_evaluator_cutoffs(&stats, &eval_config.evaluators)?;

        // Print failure messages
        for (name, cutoff, actual) in &failures {
            writeln!(
                writer,
                "Failed cutoff for evaluator {} ({}, got {})",
                name, cutoff, actual
            )?;
        }

        // If there are failures, return an error with all failures listed
        if !failures.is_empty() {
            let failure_messages = format_cutoff_failures(&failures);
            bail!("Failed cutoffs for evaluators: {}", failure_messages);
        }
    }

    Ok(())
}

/// Checks if evaluator results meet their cutoff thresholds
///
/// Returns a vector of failures with (evaluator_name, cutoff, actual_value)
pub fn check_evaluator_cutoffs(
    stats: &HashMap<String, stats::EvaluatorStats>,
    evaluator_configs: &HashMap<String, EvaluatorConfig>,
) -> Result<Vec<(String, f32, f32)>> {
    let mut failures = Vec::new();

    for (evaluator_name, evaluator_stats) in stats {
        let evaluator_config = evaluator_configs
            .get(evaluator_name)
            .ok_or_else(|| anyhow!("Evaluator not found for computing stats"))?;

        if let Some(cutoff) = evaluator_config.cutoff() {
            if evaluator_stats.mean < cutoff {
                failures.push((evaluator_name.clone(), cutoff, evaluator_stats.mean));
            }
        }
    }

    Ok(failures)
}

/// Formats a list of cutoff failures into a human-readable string
pub fn format_cutoff_failures(failures: &[(String, f32, f32)]) -> String {
    failures
        .iter()
        .map(|(name, cutoff, actual)| format!("{} (cutoff: {}, got: {})", name, cutoff, actual))
        .collect::<Vec<_>>()
        .join("\n")
}

async fn infer_datapoint(
    tensorzero_client: &ThrottledTensorZeroClient,
    function_name: &str,
    variant_name: &str,
    eval_run_id: Uuid,
    datapoint: &Datapoint,
    function_config: &FunctionConfig,
) -> Result<InferenceResponse> {
    let input = resolved_input_to_input(datapoint.input().clone()).await?;
    let dynamic_tool_params = match datapoint.tool_call_config() {
        Some(tool_params) => get_tool_params_args(tool_params, function_config).await,
        None => DynamicToolParams::default(),
    };
    let output_schema = match (datapoint.output_schema(), function_config) {
        // If the datapoint has an output schema, use it only in the case where it is not the same as the output schema of the function
        (Some(output_schema), FunctionConfig::Json(json_function_config)) => {
            if output_schema == json_function_config.output_schema.value {
                None
            } else {
                Some(output_schema)
            }
        }
        (Some(_), FunctionConfig::Chat(_)) => {
            return Err(anyhow!("Chat function does not support output schema"));
        }
        (None, _) => None,
    };
    let params = ClientInferenceParams {
        function_name: Some(function_name.to_string()),
        variant_name: Some(variant_name.to_string()),
        input,
        tags: HashMap::from([(
            "tensorzero::eval_run_id".to_string(),
            eval_run_id.to_string(),
        )]),
        dynamic_tool_params,
        output_schema: output_schema.cloned(),
        credentials: HashMap::new(),
        cache_options: CACHE_OPTIONS,
        dryrun: Some(false),
        episode_id: None,
        model_name: None,
        stream: Some(false),
        params: InferenceParams::default(),
        include_original_response: false,
        internal: true,
    };
    let inference_result = tensorzero_client.inference(params).await?;
    match inference_result {
        InferenceOutput::NonStreaming(inference_response) => Ok(inference_response),
        InferenceOutput::Streaming(_inference_stream) => {
            bail!("Streaming inference should never happen in evals")
        }
    }
}

struct ThrottledTensorZeroClient {
    client: Client,
    semaphore: Semaphore,
}

impl ThrottledTensorZeroClient {
    fn new(client: Client, semaphore: Semaphore) -> Self {
        Self { client, semaphore }
    }

    async fn inference(&self, params: ClientInferenceParams) -> Result<InferenceOutput> {
        let _permit = self.semaphore.acquire().await;
        let inference_output = self.client.inference(params).await?;
        Ok(inference_output)
    }

    async fn feedback(&self, params: FeedbackParams) -> Result<()> {
        let _permit = self.semaphore.acquire().await;
        self.client.feedback(params).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use tensorzero_internal::evals::ExactMatchConfig;

    use super::*;

    #[test]
    fn test_format_cutoff_failures() {
        let failures = vec![
            ("evaluator1".to_string(), 0.5, 0.4),
            ("evaluator2".to_string(), 0.6, 0.3),
        ];
        let formatted = format_cutoff_failures(&failures);
        assert_eq!(
            formatted,
            "evaluator1 (cutoff: 0.5, got: 0.4)\nevaluator2 (cutoff: 0.6, got: 0.3)"
        );
    }

    #[test]
    fn test_check_evaluator_cutoffs() {
        let stats = {
            let mut stats = HashMap::new();
            stats.insert(
                "evaluator1".to_string(),
                stats::EvaluatorStats {
                    mean: 0.4,
                    stderr: 0.1,
                },
            );
            stats.insert(
                "evaluator2".to_string(),
                stats::EvaluatorStats {
                    mean: 0.3,
                    stderr: 0.1,
                },
            );
            stats
        };
        let evaluators = {
            let mut evaluators = HashMap::new();
            evaluators.insert(
                "evaluator1".to_string(),
                EvaluatorConfig::ExactMatch(ExactMatchConfig { cutoff: Some(0.5) }),
            );
            evaluators.insert(
                "evaluator2".to_string(),
                EvaluatorConfig::ExactMatch(ExactMatchConfig { cutoff: Some(0.6) }),
            );
            evaluators
        };
        let failures = check_evaluator_cutoffs(&stats, &evaluators).unwrap();
        assert_eq!(failures.len(), 2);

        // Check that both expected failures are present, regardless of order
        assert!(failures.contains(&("evaluator1".to_string(), 0.5, 0.4)));
        assert!(failures.contains(&("evaluator2".to_string(), 0.6, 0.3)));
    }
}
