use std::io::Write;
use std::sync::Arc;
use std::{collections::HashMap, path::PathBuf};

use anyhow::{anyhow, bail, Result};
use clap::Parser;
use dataset::query_dataset;
use evaluators::evaluate_inference;
use helpers::{get_tool_params_args, resolved_input_to_input, setup_logging};
use indicatif::ProgressBar;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tensorzero::{
    CacheParamsOptions, Client, ClientBuilder, ClientBuilderMode, ClientInferenceParams,
    DynamicToolParams, FeedbackParams, InferenceOutput, InferenceParams, InferenceResponse,
};
use tensorzero_internal::cache::CacheEnabledMode;
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

    if let Some(progress_bar) = eval_stats.progress_bar {
        progress_bar.finish_with_message("Done");
    }

    if eval_stats.output_format == OutputFormat::HumanReadable {
        // TODO: Print more human readable stats
        for eval_info in eval_stats.eval_infos {
            writeln!(writer, "{}", serde_json::to_string(&eval_info)?)?;
        }
    }

    Ok(())
}

struct EvalStats {
    output_format: OutputFormat,
    eval_infos: Vec<EvalInfo>,
    progress_bar: Option<ProgressBar>,
}

impl EvalStats {
    fn new(output_format: OutputFormat, dataset_len: usize) -> Self {
        let progress_bar = match output_format {
            OutputFormat::Jsonl => None,
            OutputFormat::HumanReadable => Some(ProgressBar::new(dataset_len as u64)),
        };
        Self {
            output_format,
            eval_infos: Vec::new(),
            progress_bar,
        }
    }

    fn push(&mut self, eval_info: EvalInfo, writer: &mut impl Write) -> Result<()> {
        match self.output_format {
            OutputFormat::Jsonl => {
                writeln!(writer, "{}", serde_json::to_string(&eval_info)?)?;
            }
            OutputFormat::HumanReadable => {
                if let Some(progress_bar) = &mut self.progress_bar {
                    progress_bar.inc(1);
                }
            }
        }
        self.eval_infos.push(eval_info);
        Ok(())
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct EvalInfo {
    pub datapoint: Datapoint,
    pub response: InferenceResponse,
    pub evaluations: HashMap<String, Option<Value>>,
    pub evaluator_errors: HashMap<String, String>,
}

impl EvalInfo {
    fn new(
        datapoint: Datapoint,
        response: InferenceResponse,
        eval_result: evaluators::EvalResult,
    ) -> Self {
        let mut evaluations = HashMap::new();
        let mut evaluator_errors = HashMap::new();
        for (eval_name, eval_result) in eval_result {
            match eval_result {
                Ok(eval_result) => {
                    evaluations.insert(eval_name, eval_result);
                }
                Err(e) => {
                    evaluator_errors.insert(eval_name, e.to_string());
                }
            }
        }
        Self {
            datapoint,
            response,
            evaluations,
            evaluator_errors,
        }
    }
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
