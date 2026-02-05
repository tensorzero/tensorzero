//! Fireworks SFT implementation. The overall flow is:
//! 1. `FireworksSFTConfig.launch` creates and uploads training/validation datasets to Fireworks
//! 2. We kick off a SFT job in Fireworks, and produce a `FireworksSFTJobHandle` with the job ID
//! 3. `FireworksSFTJobHandle.poll` performs the following checks (without maintaining any additional state):
//!    - If the job is still running, we return a `Pending` status
//!    - If the job has failed, we return a `Failed` status
//!    - If the job has completed and deploy_after_training is true, we look for an existing
//!      'default' deployment for the model. If it exists, we return its status - otherwise,
//!      we start a new serverless deployment. When deploy_after_training is false, we skip
//!      deployment and return immediately with the model output.

use std::collections::HashMap;
use std::io::Write;
use std::time::Duration;

use futures::future::try_join_all;
use futures::try_join;
use http::StatusCode;
use reqwest::multipart::{Form, Part};
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::fmt::Display;
use std::sync::Arc;
use url::Url;
use uuid::Uuid;

use tensorzero_core::{
    config::{
        Config, TimeoutsConfig,
        provider_types::{FireworksSFTConfig as FireworksProviderSFTConfig, ProviderTypesConfig},
    },
    db::clickhouse::ClickHouseConnectionInfo,
    endpoints::inference::InferenceCredentials,
    error::{DisplayOrDebugGateway, Error, ErrorDetails, IMPOSSIBLE_ERROR_MESSAGE},
    http::TensorzeroHttpClient,
    inference::types::ContentBlock,
    model::{UninitializedModelConfig, UninitializedModelProvider, UninitializedProviderConfig},
    model_table::{FireworksKind, ProviderKind, ProviderTypeDefaultCredentials},
    optimization::{
        OptimizationJobInfo, OptimizerOutput,
        fireworks_sft::{FireworksSFTConfig, FireworksSFTJobHandle},
    },
    providers::{
        fireworks::{FIREWORKS_API_BASE, FireworksTool, PROVIDER_TYPE, prepare_fireworks_messages},
        helpers::UrlParseErrExt,
        openai::{
            OpenAIMessagesConfig, OpenAIRequestMessage, tensorzero_to_openai_assistant_message,
        },
    },
    stored_inference::{LazyRenderedSample, RenderedSample},
    utils::mock::get_mock_provider_api_base,
};

use crate::{JobHandle, Optimizer};

fn get_sft_config(
    provider_types: &ProviderTypesConfig,
) -> Result<&FireworksProviderSFTConfig, Error> {
    provider_types.fireworks.sft.as_ref().ok_or_else(|| {
        Error::new(ErrorDetails::InvalidRequest {
            message:
                "Fireworks SFT requires `[provider_types.fireworks.sft]` configuration section"
                    .to_string(),
        })
    })
}

impl Optimizer for FireworksSFTConfig {
    type Handle = FireworksSFTJobHandle;

    async fn launch(
        &self,
        client: &TensorzeroHttpClient,
        train_examples: Vec<RenderedSample>,
        val_examples: Option<Vec<RenderedSample>>,
        credentials: &InferenceCredentials,
        _clickhouse_connection_info: &ClickHouseConnectionInfo,
        config: Arc<Config>,
    ) -> Result<Self::Handle, Error> {
        // Get provider-level configuration
        let sft_config = get_sft_config(&config.provider_types)?;

        // Get credentials from provider defaults
        let fireworks_credentials = FireworksKind
            .get_defaulted_credential(None, &config.models.default_credentials)
            .await?;
        let api_key = fireworks_credentials
            .get_api_key(credentials)
            .map_err(|e| e.log())?;

        // Use mock API base for testing if set, otherwise default API base
        let api_base =
            get_mock_provider_api_base("fireworks/").unwrap_or_else(|| FIREWORKS_API_BASE.clone());

        let train_examples = train_examples
            .into_iter()
            .map(RenderedSample::into_lazy_rendered_sample)
            .collect::<Vec<_>>();
        let val_examples = val_examples.map(|examples| {
            examples
                .into_iter()
                .map(RenderedSample::into_lazy_rendered_sample)
                .collect::<Vec<_>>()
        });
        let train_rows: Vec<FireworksSupervisedRow<'_>> = try_join_all(
            train_examples
                .iter()
                .map(FireworksSupervisedRow::from_rendered_sample),
        )
        .await?;

        let val_rows = if let Some(examples) = val_examples.as_ref() {
            Some(
                try_join_all(
                    examples
                        .iter()
                        .map(FireworksSupervisedRow::from_rendered_sample),
                )
                .await?,
            )
        } else {
            None
        };

        // Run these concurrently
        let train_fut = create_and_upload_dataset(
            client,
            api_key,
            &api_base,
            &sft_config.account_id,
            &train_rows,
        );

        let (train_file_id, val_file_id) = if let Some(val_rows) = val_rows.as_ref() {
            let val_fut = create_and_upload_dataset(
                client,
                api_key,
                &api_base,
                &sft_config.account_id,
                val_rows,
            );

            // Run both futures concurrently
            let (train_id, val_id) = try_join!(train_fut, val_fut)?;
            (train_id, Some(val_id))
        } else {
            // Just run the training file upload
            let train_file_id = train_fut.await?;
            (train_file_id, None)
        };

        let body = FireworksFineTuningRequest {
            base_model: self.model.clone(),
            early_stop: self.early_stop,
            epochs: self.epochs,
            learning_rate: self.learning_rate,
            max_context_length: self.max_context_length,
            lora_rank: self.lora_rank,
            batch_size: self.batch_size,
            dataset: train_file_id,
            evaluation_dataset: val_file_id,
            display_name: self.display_name.clone(),
            output_model: self.output_model.clone(),
            warm_start_from: self.warm_start_from.clone(),
            is_turbo: self.is_turbo,
            eval_auto_carveout: self.eval_auto_carveout,
            nodes: self.nodes,
            mtp_enabled: self.mtp_enabled,
            mtp_num_draft_tokens: self.mtp_num_draft_tokens,
            mtp_freeze_base_model: self.mtp_freeze_base_model,
        };

        let request = client
            .post(
                api_base
                    .join(&format!(
                        "v1/accounts/{}/supervisedFineTuningJobs",
                        sft_config.account_id
                    ))
                    .convert_parse_error()?,
            )
            .bearer_auth(api_key.expose_secret())
            .json(&body);
        let res = request.send().await.map_err(|e| {
            Error::new(ErrorDetails::InferenceClient {
                status_code: e.status(),
                message: format!(
                    "Error sending request to Fireworks: {}",
                    DisplayOrDebugGateway::new(e)
                ),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: Some(serde_json::to_string(&body).unwrap_or_default()),
                raw_response: None,
            })
        })?;

        let raw_response = res.text().await.map_err(|e| {
            Error::new(ErrorDetails::InferenceClient {
                status_code: e.status(),
                message: format!(
                    "Error sending request to Fireworks: {}",
                    DisplayOrDebugGateway::new(e)
                ),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: Some(serde_json::to_string(&body).unwrap_or_default()),
                raw_response: None,
            })
        })?;
        let job: FireworksFineTuningJobResponse =
            serde_json::from_str(&raw_response).map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!(
                        "Error parsing JSON response: {}",
                        DisplayOrDebugGateway::new(e)
                    ),
                    raw_request: Some(serde_json::to_string(&body).unwrap_or_default()),
                    raw_response: Some(raw_response.clone()),
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?;

        // Fireworks job names look like 'accounts/{account_id}/supervisedFineTuningJobs/{job_id}'
        // Extract the job id to construct a dashboard URL
        let job_id = job.name.split("/").last().ok_or_else(|| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!("No job ID in job path: {}", job.name),
                raw_request: None,
                raw_response: None,
                provider_type: PROVIDER_TYPE.to_string(),
            })
        })?;

        Ok(FireworksSFTJobHandle {
            job_url: format!("https://app.fireworks.ai/dashboard/fine-tuning/supervised/{job_id}")
                .parse()
                .convert_parse_error()?,
            job_path: job.name,
            deploy_after_training: self.deploy_after_training,
        })
    }
}

impl JobHandle for FireworksSFTJobHandle {
    async fn poll(
        &self,
        client: &TensorzeroHttpClient,
        credentials: &InferenceCredentials,
        default_credentials: &ProviderTypeDefaultCredentials,
        provider_types: &ProviderTypesConfig,
    ) -> Result<OptimizationJobInfo, Error> {
        // Get provider-level configuration
        let sft_config = get_sft_config(provider_types)?;

        // Get credentials from provider defaults
        let fireworks_credentials = FireworksKind
            .get_defaulted_credential(None, default_credentials)
            .await?;
        let api_key = fireworks_credentials
            .get_api_key(credentials)
            .map_err(|e| e.log())?;

        // Use mock API base for testing if set, otherwise default API base
        let api_base =
            get_mock_provider_api_base("fireworks/").unwrap_or_else(|| FIREWORKS_API_BASE.clone());

        let job_status = poll_job(client, api_key, &api_base, &self.job_path).await?;
        if let FireworksFineTuningJobState::JobStateCompleted = job_status.state {
            // Once the job has completed, start polling the model deployment.
            let model_path = job_status.output_model.ok_or_else(|| {
                Error::new(ErrorDetails::InferenceServer {
                    message: "No model path in Fireworks JobStateCompleted response".to_string(),
                    raw_request: None,
                    raw_response: None,
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?;
            // TODO - start using this as the TensorZero model name
            // once the UI has been refactored to allow separate model names and provider model names
            let _model_id = model_path.split("/").last().ok_or_else(|| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("No model ID in model path: {model_path}"),
                    raw_request: None,
                    raw_response: None,
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?;
            let completed_output = OptimizationJobInfo::Completed {
                output: OptimizerOutput::Model(UninitializedModelConfig {
                    routing: vec![model_path.clone().into()],
                    providers: HashMap::from([(
                        model_path.clone().into(),
                        UninitializedModelProvider {
                            config: UninitializedProviderConfig::Fireworks {
                                model_name: model_path.clone(),
                                parse_think_blocks: true,
                                api_key_location: None,
                            },
                            extra_headers: None,
                            extra_body: None,
                            timeouts: TimeoutsConfig::default(),
                            discard_unknown_chunks: false,
                        },
                    )]),
                    timeouts: TimeoutsConfig::default(),
                    skip_relay: None,
                }),
            };
            if !self.deploy_after_training {
                return Ok(completed_output);
            }
            let deployment_state = deploy_or_poll_model(
                client,
                api_key,
                &api_base,
                &sft_config.account_id,
                &model_path,
            )
            .await?;
            match deployment_state {
                FireworksDeploymentState::StateUnspecified
                | FireworksDeploymentState::Deploying
                | FireworksDeploymentState::Undeploying
                | FireworksDeploymentState::Updating => Ok(OptimizationJobInfo::Pending {
                    message: deployment_state.to_string(),
                    estimated_finish: None,
                    trained_tokens: None,
                    error: None,
                }),
                FireworksDeploymentState::Deployed => Ok(completed_output),
            }
        } else {
            convert_to_optimizer_status(job_status)
        }
    }
}

#[derive(Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct FireworksFineTuningRequest {
    pub base_model: String,
    pub dataset: String,
    pub evaluation_dataset: Option<String>,
    pub early_stop: Option<bool>,
    pub epochs: Option<usize>,
    pub learning_rate: Option<f64>,
    pub max_context_length: Option<usize>,
    pub lora_rank: Option<usize>,
    pub batch_size: Option<usize>,
    pub display_name: Option<String>,
    pub output_model: Option<String>,
    pub warm_start_from: Option<String>,
    pub is_turbo: Option<bool>,
    pub eval_auto_carveout: Option<bool>,
    pub nodes: Option<usize>,
    pub mtp_enabled: Option<bool>,
    pub mtp_num_draft_tokens: Option<usize>,
    pub mtp_freeze_base_model: Option<bool>,
}

#[derive(Debug, Serialize)]
struct FireworksSupervisedRow<'a> {
    messages: Vec<OpenAIRequestMessage<'a>>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<FireworksTool<'a>>,
}

impl<'a> FireworksSupervisedRow<'a> {
    pub async fn from_rendered_sample(inference: &'a LazyRenderedSample) -> Result<Self, Error> {
        if inference
            .tool_params
            .parallel_tool_calls
            .unwrap_or_default()
        {
            return Err(Error::new(ErrorDetails::InvalidRenderedStoredInference {
                message: "Parallel tool calls are not supported for Fireworks".to_string(),
            }));
        }
        let tools = inference
            .tool_params
            .additional_tools
            .as_ref()
            .map(|tools| {
                tools
                    .iter()
                    .filter_map(|dt| match &dt {
                        tensorzero_core::tool::Tool::Function(func) => Some(func.into()),
                        tensorzero_core::tool::Tool::OpenAICustom(_) => None, // Skip custom tools for SFT
                    })
                    .collect()
            })
            .unwrap_or_default();
        let mut messages = prepare_fireworks_messages(
            inference.system_input.as_deref(),
            &inference.messages,
            OpenAIMessagesConfig {
                json_mode: None,
                provider_type: PROVIDER_TYPE,
                // For now, this isn't configurable in SFT (we should never need to resolve a file URL here)
                fetch_and_encode_input_files_before_inference: true,
            },
        )
        .await?;

        let Some(output) = &inference.output else {
            return Err(Error::new(ErrorDetails::InvalidRenderedStoredInference {
                message: "No output in inference".to_string(),
            }));
        };
        if output.is_empty() {
            return Err(Error::new(ErrorDetails::InvalidRenderedStoredInference {
                message: "No output in inference".to_string(),
            }));
        }
        let output_content_blocks: Vec<ContentBlock> =
            output.iter().map(|c| c.clone().into()).collect::<Vec<_>>();
        let final_assistant_message = tensorzero_to_openai_assistant_message(
            Cow::Owned(output_content_blocks),
            OpenAIMessagesConfig {
                json_mode: None,
                provider_type: PROVIDER_TYPE,
                // For now, this isn't configurable in SFT (we should never need to resolve a file URL here)
                fetch_and_encode_input_files_before_inference: true,
            },
        )
        .await?;
        messages.push(final_assistant_message);
        Ok(Self { messages, tools })
    }
}

#[derive(Debug, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
struct FireworksDatasetResponse {
    state: String,
}

#[derive(Debug, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FireworksFineTuningJobState {
    JobStateUnspecified,
    JobStateCreating,
    JobStateRunning,
    JobStateCompleted,
    JobStateFailed,
    JobStateCancelled,
    JobStateDeleting,
    JobStateWritingResults,
    JobStateValidating,
    JobStateRollout,
    JobStateEvaluation,
    JobStateFailedCleaningUp,
    JobStateDeletingCleaningUp,
    JobStatePolicyUpdate,
    JobStatePending,
}

impl Display for FireworksFineTuningJobState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.serialize(f)
    }
}

#[derive(Debug, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
struct FireworksFineTuningJobResponse {
    state: FireworksFineTuningJobState,
    status: Option<FireworksFineTuningJobStatus>,
    output_model: Option<String>,
    name: String,
}

#[derive(Debug, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
struct FireworksFineTuningJobStatus {
    message: String,
}

#[derive(Debug, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
struct FireworksDeployedModelResponse {
    state: FireworksDeploymentState,
    status: Option<FireworksDeploymentStatus>,
}

#[derive(Debug, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
struct FireworksDeploymentStatus {
    message: String,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FireworksDeploymentState {
    StateUnspecified,
    Undeploying,
    Deploying,
    Deployed,
    Updating,
}

impl Display for FireworksDeploymentState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.serialize(f)
    }
}

#[derive(Debug, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
struct FireworksModelResponse {
    deployed_model_refs: Vec<FireworksDeployedModelRef>,
}

#[derive(Debug, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
struct FireworksDeployedModelRef {
    name: String,
    default: bool,
    state: FireworksDeploymentState,
}

fn convert_to_optimizer_status(
    job: FireworksFineTuningJobResponse,
) -> Result<OptimizationJobInfo, Error> {
    Ok(match job.state {
        FireworksFineTuningJobState::JobStateCreating
        | FireworksFineTuningJobState::JobStatePending
        | FireworksFineTuningJobState::JobStateRunning
        | FireworksFineTuningJobState::JobStateWritingResults
        | FireworksFineTuningJobState::JobStateValidating
        | FireworksFineTuningJobState::JobStateRollout
        | FireworksFineTuningJobState::JobStateEvaluation
        | FireworksFineTuningJobState::JobStateUnspecified
        | FireworksFineTuningJobState::JobStatePolicyUpdate => OptimizationJobInfo::Pending {
            message: job.state.to_string(),
            estimated_finish: None,
            trained_tokens: None,
            error: None,
        },
        FireworksFineTuningJobState::JobStateFailed
        | FireworksFineTuningJobState::JobStateFailedCleaningUp
        | FireworksFineTuningJobState::JobStateCancelled
        | FireworksFineTuningJobState::JobStateDeleting
        | FireworksFineTuningJobState::JobStateDeletingCleaningUp => OptimizationJobInfo::Failed {
            message: job.state.to_string(),
            error: job.status.map(|s| s.message.into()),
        },
        FireworksFineTuningJobState::JobStateCompleted => {
            return Err(Error::new(ErrorDetails::InternalError {
                message: format!(
                    "JobStateCompleted should have been handled in poll. {IMPOSSIBLE_ERROR_MESSAGE}"
                ),
            }));
        }
    })
}

/// Polls the status of an existing dataset.
/// Returns `true` if the dataset is in the `READY` state.
async fn poll_dataset_read(
    client: &TensorzeroHttpClient,
    api_key: &SecretString,
    api_base: &Url,
    account_id: &str,
    dataset_id: &str,
) -> Result<bool, Error> {
    let res = client
        .get(
            api_base
                .join(&format!("v1/accounts/{account_id}/datasets/{dataset_id}"))
                .convert_parse_error()?,
        )
        .bearer_auth(api_key.expose_secret())
        .send()
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::InferenceClient {
                status_code: e.status(),
                message: format!(
                    "Error checking dataset status: {}",
                    DisplayOrDebugGateway::new(e)
                ),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: None,
                raw_response: None,
            })
        })?;
    let raw_response = res.text().await.map_err(|e| {
        Error::new(ErrorDetails::InferenceClient {
            status_code: e.status(),
            provider_type: PROVIDER_TYPE.to_string(),
            raw_request: None,
            raw_response: None,
            message: format!(
                "Error checking dataset status: {}",
                DisplayOrDebugGateway::new(e)
            ),
        })
    })?;
    let response: FireworksDatasetResponse = serde_json::from_str(&raw_response).map_err(|e| {
        Error::new(ErrorDetails::InferenceServer {
            raw_request: None,
            raw_response: Some(raw_response.clone()),
            message: format!(
                "Error parsing JSON response: {}",
                DisplayOrDebugGateway::new(e)
            ),
            provider_type: PROVIDER_TYPE.to_string(),
        })
    })?;
    Ok(response.state == "READY")
}

/// Produces a new dataset with a randomly-generated id, with the provided 'items' uploaded to it.
/// Returns the Fireworks path to the dataset (e.g. `account/{account_id}/datasets/{dataset_id}`)
async fn create_and_upload_dataset<'a>(
    client: &TensorzeroHttpClient,
    api_key: &SecretString,
    api_base: &Url,
    account_id: &str,
    items: &[FireworksSupervisedRow<'a>],
) -> Result<String, Error> {
    let dataset_id = format!("t0-{}", Uuid::now_v7());
    let dataset_path = format!("accounts/{account_id}/datasets/{dataset_id}");
    let res = client
        .post(
            api_base
                .join(&format!("v1/accounts/{account_id}/datasets"))
                .convert_parse_error()?,
        )
        .bearer_auth(api_key.expose_secret())
        .json(&serde_json::json!({
            "datasetId": dataset_id,
            "dataset": {
                "displayName": dataset_id,
                "format": "CHAT",
                "exampleCount": items.len(),
            },
        }))
        .send()
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::InferenceClient {
                status_code: e.status(),
                message: format!("Error creating dataset: {}", DisplayOrDebugGateway::new(e)),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: None,
                raw_response: None,
            })
        })?;
    let status = res.status();
    let raw_response = res.text().await.map_err(|e| {
        Error::new(ErrorDetails::InferenceClient {
            status_code: e.status(),
            message: format!("Error creating dataset: {}", DisplayOrDebugGateway::new(e)),
            provider_type: PROVIDER_TYPE.to_string(),
            raw_request: None,
            raw_response: None,
        })
    })?;
    if !status.is_success() {
        return Err(Error::new(ErrorDetails::InferenceClient {
            status_code: Some(status),
            message: "Error creating dataset".to_string(),
            provider_type: PROVIDER_TYPE.to_string(),
            raw_request: None,
            raw_response: Some(raw_response),
        }));
    }

    let mut jsonl_data = Vec::new();
    for item in items {
        serde_json::to_writer(&mut jsonl_data, item).map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Error writing to JSONL: {}", DisplayOrDebugGateway::new(e)),
            })
        })?;
        jsonl_data.write_all(b"\n").map_err(|e| {
            Error::new(ErrorDetails::Serialization {
                message: format!("Error writing to JSONL: {}", DisplayOrDebugGateway::new(e)),
            })
        })?;
    }
    let form = Form::new().part(
        "file",
        Part::bytes(jsonl_data)
            .file_name("dataset.jsonl")
            .mime_str("application/jsonl")
            .map_err(|e| {
                Error::new(ErrorDetails::Serialization {
                    message: format!("Error setting MIME type: {}", DisplayOrDebugGateway::new(e)),
                })
            })?,
    );
    let request_url = api_base
        .join(&format!("v1/{dataset_path}:upload"))
        .convert_parse_error()?;
    let request_builder = client
        .post(request_url)
        .bearer_auth(api_key.expose_secret())
        .multipart(form);
    let res = request_builder.send().await.map_err(|e| {
        Error::new(ErrorDetails::InferenceClient {
            status_code: e.status(),
            message: format!(
                "Error sending request to Fireworks: {}",
                DisplayOrDebugGateway::new(e)
            ),
            provider_type: PROVIDER_TYPE.to_string(),
            raw_request: None,
            raw_response: None,
        })
    })?;
    if !res.status().is_success() {
        return Err(Error::new(ErrorDetails::InferenceClient {
            status_code: Some(res.status()),
            message: format!(
                "Unsuccessful status code when uploading dataset to Fireworks: {}",
                res.status()
            ),
            provider_type: PROVIDER_TYPE.to_string(),
            raw_request: None,
            raw_response: res.text().await.ok(),
        }));
    }
    while !poll_dataset_read(client, api_key, api_base, account_id, &dataset_id).await? {
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
    Ok(dataset_path)
}

async fn get_model(
    client: &TensorzeroHttpClient,
    api_key: &SecretString,
    api_base: &Url,
    model_path: &str,
) -> Result<Option<FireworksModelResponse>, Error> {
    let res = client
        .get(
            api_base
                .join(&format!("v1/{model_path}"))
                .convert_parse_error()?,
        )
        .bearer_auth(api_key.expose_secret())
        .send()
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::InferenceClient {
                status_code: e.status(),
                message: format!("Error getting model: {}", DisplayOrDebugGateway::new(e)),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: None,
                raw_response: None,
            })
        })?;
    if res.status() == StatusCode::NOT_FOUND {
        return Ok(None);
    }
    let raw_response = res.text().await.map_err(|e| {
        Error::new(ErrorDetails::InferenceClient {
            status_code: e.status(),
            message: format!("Error getting model: {}", DisplayOrDebugGateway::new(e)),
            provider_type: PROVIDER_TYPE.to_string(),
            raw_request: None,
            raw_response: None,
        })
    })?;
    let model: FireworksModelResponse = serde_json::from_str(&raw_response).map_err(|e| {
        Error::new(ErrorDetails::InferenceServer {
            message: format!(
                "Error parsing JSON response: {}",
                DisplayOrDebugGateway::new(e)
            ),
            raw_request: None,
            raw_response: Some(raw_response.clone()),
            provider_type: PROVIDER_TYPE.to_string(),
        })
    })?;
    Ok(Some(model))
}

async fn deploy_or_poll_model(
    client: &TensorzeroHttpClient,
    api_key: &SecretString,
    api_base: &Url,
    account_id: &str,
    model_path: &str,
) -> Result<FireworksDeploymentState, Error> {
    let model = get_model(client, api_key, api_base, model_path).await?;
    if let Some(model) = model {
        // We created a default deployment, so assume that this is the one that we want
        // It's possible that the user created a different default deployment while we were polling,
        // but that should be fine - we just care that a default deployment succeeded for our model.
        for deployment in model.deployed_model_refs {
            if deployment.default {
                return Ok(deployment.state);
            }
        }
    }

    // If we didn't find a default deployment, kick one off.
    // For now, we only support serverless deployments.
    let res = client
        .post(
            api_base
                .join(&format!("v1/accounts/{account_id}/deployedModels"))
                .convert_parse_error()?,
        )
        .bearer_auth(api_key.expose_secret())
        .json(&serde_json::json!({
            "model": model_path,
            "default": true,
            "serverless": true,
            "public": false,
        }))
        .send()
        .await
        .map_err(|e| {
            Error::new(ErrorDetails::InferenceClient {
                status_code: e.status(),
                message: format!("Error deploying model: {}", DisplayOrDebugGateway::new(e)),
                provider_type: PROVIDER_TYPE.to_string(),
                raw_request: None,
                raw_response: None,
            })
        })?;

    let raw_response = res.text().await.map_err(|e| {
        Error::new(ErrorDetails::InferenceClient {
            status_code: e.status(),
            message: format!("Error deploying model: {}", DisplayOrDebugGateway::new(e)),
            provider_type: PROVIDER_TYPE.to_string(),
            raw_request: None,
            raw_response: None,
        })
    })?;
    let response: FireworksDeployedModelResponse =
        serde_json::from_str(&raw_response).map_err(|e| {
            Error::new(ErrorDetails::InferenceServer {
                message: format!(
                    "Error parsing JSON response: {}",
                    DisplayOrDebugGateway::new(e)
                ),
                raw_request: None,
                raw_response: Some(raw_response.clone()),
                provider_type: PROVIDER_TYPE.to_string(),
            })
        })?;
    Ok(response.state)
}

async fn poll_job(
    client: &TensorzeroHttpClient,
    api_key: &SecretString,
    api_base: &Url,
    job_path: &str,
) -> Result<FireworksFineTuningJobResponse, Error> {
    let request = client
        .get(
            api_base
                .join(&format!("v1/{job_path}"))
                .convert_parse_error()?,
        )
        .bearer_auth(api_key.expose_secret());
    let res = request.send().await.map_err(|e| {
        Error::new(ErrorDetails::InferenceClient {
            status_code: e.status(),
            message: format!(
                "Error sending request to Fireworks: {}",
                DisplayOrDebugGateway::new(e)
            ),
            provider_type: PROVIDER_TYPE.to_string(),
            raw_request: None,
            raw_response: None,
        })
    })?;
    let raw_response = res.text().await.map_err(|e| {
        Error::new(ErrorDetails::InferenceClient {
            status_code: e.status(),
            message: format!(
                "Error sending request to Fireworks: {}",
                DisplayOrDebugGateway::new(e)
            ),
            provider_type: PROVIDER_TYPE.to_string(),
            raw_request: None,
            raw_response: None,
        })
    })?;
    let job: FireworksFineTuningJobResponse = serde_json::from_str(&raw_response).map_err(|e| {
        Error::new(ErrorDetails::InferenceServer {
            message: format!(
                "Error parsing JSON response: {}",
                DisplayOrDebugGateway::new(e)
            ),
            raw_request: None,
            raw_response: Some(raw_response.clone()),
            provider_type: PROVIDER_TYPE.to_string(),
        })
    })?;
    Ok(job)
}
