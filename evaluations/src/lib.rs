use std::io::Write;
use std::sync::Arc;
use std::{collections::HashMap, path::PathBuf};

use anyhow::{anyhow, bail, Result};
use clap::Parser;
use dataset::query_dataset;
use evaluators::evaluate_inference;
use helpers::{get_tool_params_args, setup_logging};
use serde::{Deserialize, Serialize};
use stats::{EvaluationError, EvaluationInfo, EvaluationStats, EvaluationUpdate};
use tensorzero::ClientInput;
use tensorzero::{
    input_handling::resolved_input_to_client_input, CacheParamsOptions, Client, ClientBuilder,
    ClientBuilderMode, ClientInferenceParams, DynamicToolParams, FeedbackParams, InferenceOutput,
    InferenceParams, InferenceResponse,
};
use tensorzero_internal::cache::CacheEnabledMode;
use tensorzero_internal::config_parser::MetricConfigOptimize;
use tensorzero_internal::evaluations::{EvaluationConfig, EvaluatorConfig};
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
    #[arg(long, default_value = "./config/tensorzero.toml")]
    pub config_file: PathBuf,

    /// URL of a running TensorZero HTTP gateway server to use for requests. This runs evaluations using that gateway.
    #[arg(long)]
    pub gateway_url: Option<Url>,

    /// Name of the evaluation to run.
    #[arg(short, long)]
    pub evaluation_name: String,

    /// Name of the dataset to run on.
    #[arg(short, long)]
    pub dataset_name: String,

    /// Name of the variant to run.
    #[arg(short, long)]
    pub variant_name: String,

    /// Number of concurrent requests to make.
    #[arg(short, long, default_value = "1")]
    pub concurrency: usize,

    #[arg(short, long, default_value = "human-readable")]
    pub format: OutputFormat,
}

pub async fn run_evaluation(
    args: Args,
    evaluation_run_id: Uuid,
    mut writer: impl Write,
) -> Result<()> {
    setup_logging(&args)?;
    let semaphore = Semaphore::new(args.concurrency);
    let clickhouse_url = std::env::var("TENSORZERO_CLICKHOUSE_URL")
        .map_err(|_| anyhow!("Missing ClickHouse URL at TENSORZERO_CLICKHOUSE_URL"))?;

    // We do not validate credentials here since we just want the evaluator config
    // If we are using an embedded gateway, credentials are validated when that is initialized
    let config =
        Config::load_from_path_optional_verify_credentials(&args.config_file, false).await?;
    let evaluation_config = config
        .evaluations
        .get(&args.evaluation_name)
        .ok_or(anyhow!("evaluation not found"))?
        .clone();
    let EvaluationConfig::Static(static_evaluation_config) = &*evaluation_config;
    let function_config = config.get_function(&static_evaluation_config.function_name)?;
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
        &args.dataset_name,
        &static_evaluation_config.function_name,
        function_config,
    )
    .await?;
    let dataset_name = Arc::new(args.dataset_name);
    let variant_name = Arc::new(args.variant_name);
    let evaluation_name = Arc::new(args.evaluation_name);
    let dataset_len = dataset.len();
    let mut task_id_to_datapoint_id = HashMap::new();

    write_run_info(
        &mut writer,
        &RunInfo {
            evaluation_run_id,
            num_datapoints: dataset_len,
        },
        &args.format,
    )?;

    // Spawn concurrent tasks for each datapoint
    for datapoint in dataset {
        let client_clone = tensorzero_client_with_semaphore.clone();
        let variant_name = variant_name.clone();
        let function_config = function_config.clone();
        let evaluation_config = evaluation_config.clone();
        let dataset_name = dataset_name.clone();
        let function_name = static_evaluation_config.function_name.clone();
        let evaluation_name = evaluation_name.clone();
        let evaluation_run_id_clone = evaluation_run_id;
        let datapoint = Arc::new(datapoint);
        let datapoint_id = datapoint.id();
        let abort_handle = join_set.spawn(async move {
            let input = Arc::new(resolved_input_to_client_input(datapoint.input().clone(), &client_clone.client).await?);
            let inference_response = Arc::new(
                infer_datapoint(InferDatapointParams {
                    tensorzero_client: &client_clone,
                    function_name: &function_name,
                    variant_name: &variant_name,
                    evaluation_run_id: evaluation_run_id_clone,
                    dataset_name: &dataset_name,
                    datapoint: &datapoint,
                    evaluation_name: &evaluation_name,
                    function_config: &function_config,
                    input: &input,
                })
                .await?,
            );

            let evaluation_result = evaluate_inference(
                inference_response.clone(),
                datapoint.clone(),
                input,
                evaluation_config,
                evaluation_name,
                client_clone,
                evaluation_run_id_clone,
            )
            .await?;

            Ok::<(Datapoint, InferenceResponse, evaluators::EvaluationResult), anyhow::Error>((
                Arc::into_inner(datapoint).ok_or_else(|| anyhow!("Failed to get datapoint for datapoint. This should never happen. Please file a bug report at https://github.com/tensorzero/tensorzero/discussions/categories/bug-reports."))?,
                Arc::into_inner(inference_response).ok_or_else(|| anyhow!("Failed to get inference response for datapoint. This should never happen. Please file a bug report at https://github.com/tensorzero/tensorzero/discussions/categories/bug-reports."))?,
                evaluation_result,
            ))
        });
        task_id_to_datapoint_id.insert(abort_handle.id(), datapoint_id);
    }

    // Collect results
    let mut evaluation_stats = EvaluationStats::new(args.format, dataset_len);

    while let Some(result) = join_set.join_next_with_id().await {
        match result {
            Ok((_, Ok((datapoint, inference_response, evaluation_result)))) => {
                evaluation_stats.push(
                    EvaluationUpdate::Success(EvaluationInfo::new(
                        datapoint,
                        inference_response,
                        evaluation_result,
                    )),
                    &mut writer,
                )?;
            }
            Ok((task_id, Err(e))) => {
                tracing::warn!("Task error: {}", e);
                evaluation_stats.push(
                    EvaluationUpdate::Error(EvaluationError {
                        datapoint_id: task_id_to_datapoint_id[&task_id],
                        message: e.to_string(),
                    }),
                    &mut writer,
                )?;
            }
            Err(e) => evaluation_stats.push(
                EvaluationUpdate::Error(EvaluationError {
                    datapoint_id: task_id_to_datapoint_id[&e.id()],
                    message: e.to_string(),
                }),
                &mut writer,
            )?,
        }
    }

    if let Some(progress_bar) = &evaluation_stats.progress_bar {
        progress_bar.finish_with_message("Done");
    }

    if evaluation_stats.output_format == OutputFormat::HumanReadable {
        let stats = evaluation_stats.compute_stats(&static_evaluation_config.evaluators);

        // Print all stats
        for (evaluator_name, evaluator_stats) in &stats {
            writeln!(writer, "{}: {}", evaluator_name, evaluator_stats)?;
        }

        // Check cutoffs and handle failures
        let failures = check_evaluator_cutoffs(&stats, &static_evaluation_config.evaluators)?;

        // Print failure messages
        for (name, cutoff, actual) in &failures {
            writeln!(
                writer,
                "Failed cutoff for evaluator {} ({:.2}, got {:.2})",
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
            match evaluator_config.optimize() {
                MetricConfigOptimize::Max => {
                    if evaluator_stats.mean < cutoff {
                        failures.push((evaluator_name.clone(), cutoff, evaluator_stats.mean));
                    }
                }
                MetricConfigOptimize::Min => {
                    if evaluator_stats.mean > cutoff {
                        failures.push((evaluator_name.clone(), cutoff, evaluator_stats.mean));
                    }
                }
            }
        }
    }

    Ok(failures)
}

/// Formats a list of cutoff failures into a human-readable string
pub fn format_cutoff_failures(failures: &[(String, f32, f32)]) -> String {
    failures
        .iter()
        .map(|(name, cutoff, actual)| {
            format!("{} (cutoff: {:.2}, got: {:.2})", name, cutoff, actual)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

struct InferDatapointParams<'a> {
    tensorzero_client: &'a ThrottledTensorZeroClient,
    function_name: &'a str,
    variant_name: &'a str,
    evaluation_run_id: Uuid,
    dataset_name: &'a str,
    datapoint: &'a Datapoint,
    input: &'a ClientInput,
    evaluation_name: &'a str,
    function_config: &'a FunctionConfig,
}

async fn infer_datapoint(params: InferDatapointParams<'_>) -> Result<InferenceResponse> {
    let InferDatapointParams {
        tensorzero_client,
        function_name,
        variant_name,
        evaluation_run_id,
        dataset_name,
        datapoint,
        evaluation_name,
        function_config,
        input,
    } = params;

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
        input: input.clone(),
        tags: HashMap::from([
            (
                "tensorzero::evaluation_run_id".to_string(),
                evaluation_run_id.to_string(),
            ),
            (
                "tensorzero::datapoint_id".to_string(),
                datapoint.id().to_string(),
            ),
            (
                "tensorzero::evaluation_name".to_string(),
                evaluation_name.to_string(),
            ),
            (
                "tensorzero::dataset_name".to_string(),
                dataset_name.to_string(),
            ),
        ]),
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
        extra_body: Default::default(),
    };
    let inference_result = tensorzero_client.inference(params).await?;
    match inference_result {
        InferenceOutput::NonStreaming(inference_response) => Ok(inference_response),
        InferenceOutput::Streaming(_inference_stream) => {
            bail!("Streaming inference should never happen in evaluations")
        }
    }
}

fn write_run_info(
    writer: &mut impl Write,
    run_info: &RunInfo,
    format: &OutputFormat,
) -> Result<()> {
    match format {
        OutputFormat::Jsonl => {
            writeln!(writer, "{}", serde_json::to_string(run_info)?)?;
        }
        OutputFormat::HumanReadable => {
            writeln!(writer, "Run ID: {}", run_info.evaluation_run_id)?;
            writeln!(writer, "Number of datapoints: {}", run_info.num_datapoints)?;
        }
    }
    Ok(())
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RunInfo {
    pub evaluation_run_id: Uuid,
    pub num_datapoints: usize,
}

pub struct ThrottledTensorZeroClient {
    pub client: Client,
    semaphore: Semaphore,
}

impl ThrottledTensorZeroClient {
    pub fn new(client: Client, semaphore: Semaphore) -> Self {
        Self { client, semaphore }
    }

    async fn inference(&self, params: ClientInferenceParams) -> Result<InferenceOutput> {
        let _permit = self.semaphore.acquire().await;
        let inference_output = self.client.inference(params).await?;
        Ok(inference_output)
    }

    async fn feedback(&self, params: FeedbackParams) -> Result<()> {
        self.client.feedback(params).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use tensorzero_internal::evaluations::ExactMatchConfig;

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
            "evaluator1 (cutoff: 0.50, got: 0.40)\nevaluator2 (cutoff: 0.60, got: 0.30)"
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
            stats.insert(
                "evaluator3".to_string(),
                stats::EvaluatorStats {
                    mean: 0.1,
                    stderr: 0.05,
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
            evaluators.insert(
                "evaluator3".to_string(),
                EvaluatorConfig::ExactMatch(ExactMatchConfig { cutoff: None }),
            );
            evaluators
        };
        let failures = check_evaluator_cutoffs(&stats, &evaluators).unwrap();
        assert_eq!(failures.len(), 2);

        // Check that both expected failures are present, regardless of order
        assert!(failures.contains(&("evaluator1".to_string(), 0.5, 0.4)));
        assert!(failures.contains(&("evaluator2".to_string(), 0.6, 0.3)));

        // Check that evaluator3 is not in the failures list since it has no cutoff
        assert!(!failures.iter().any(|(name, _, _)| name == "evaluator3"));
    }
}
