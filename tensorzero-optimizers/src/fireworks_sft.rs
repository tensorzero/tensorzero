//! Fireworks SFT implementation. The overall flow is:
//! 1. `FireworksSFTConfig.launch` creates and uploads training/validation datasets to Fireworks
//! 2. We kick off a SFT job in Fireworks, and produce a `FireworksSFTJobHandle` with the job ID
//! 3. `FireworksSFTJobHandle.poll` performs the following checks (without maintaining any additional state):
//!    - If the job is still running, we return a `Pending` status
//!    - If the job has failed, we return a `Failed` status
//!    - If the job has completed, we look for an existing 'default' deployment for the model.
//!      If it exists, we return its status - otherwise, we start a new serverless deployment.

use async_trait::async_trait;
use std::collections::HashMap;

use futures::future::try_join_all;
use futures::try_join;
use secrecy::ExposeSecret;
use serde::Serialize;

use tensorzero_core::{
    config::{Config, TimeoutsConfig},
    db::clickhouse::ClickHouseConnectionInfo,
    endpoints::inference::InferenceCredentials,
    error::{DisplayOrDebugGateway, Error, ErrorDetails, IMPOSSIBLE_ERROR_MESSAGE},
    http::TensorzeroHttpClient,
    model::{UninitializedModelConfig, UninitializedModelProvider, UninitializedProviderConfig},
    model_table::{FireworksKind, ProviderKind, ProviderTypeDefaultCredentials},
    optimization::{
        fireworks_sft::{
            FireworksFineTuningJobResponse, FireworksSFTConfig, FireworksSFTJobHandle,
        },
        OptimizationJobInfo, OptimizerOutput,
    },
    providers::{
        fireworks::{FireworksCredentials, PROVIDER_TYPE},
        helpers::UrlParseErrExt,
    },
    stored_inference::RenderedSample,
};

use crate::{JobHandle, Optimizer};

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

#[async_trait]
impl Optimizer for FireworksSFTConfig {
    type Handle = FireworksSFTJobHandle;

    async fn launch(
        &self,
        client: &TensorzeroHttpClient,
        train_examples: Vec<RenderedSample>,
        val_examples: Option<Vec<RenderedSample>>,
        credentials: &InferenceCredentials,
        _clickhouse_connection_info: &ClickHouseConnectionInfo,
        _config: &Config,
    ) -> Result<Self::Handle, Error> {
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
        let train_rows: Vec<
            tensorzero_core::optimization::fireworks_sft::FireworksSupervisedRow<'_>,
        > = try_join_all(
            train_examples.iter().map(
                tensorzero_core::optimization::fireworks_sft::FireworksSupervisedRow::from_rendered_sample,
            ),
        )
        .await?;

        let val_rows = if let Some(examples) = val_examples.as_ref() {
            Some(
                try_join_all(examples.iter().map(
                    tensorzero_core::optimization::fireworks_sft::FireworksSupervisedRow::from_rendered_sample,
                ))
                .await?,
            )
        } else {
            None
        };

        let api_key = self
            .credentials
            .get_api_key(credentials)
            .map_err(|e| e.log())?;

        // Run these concurrently

        let train_fut = self.create_and_upload_dataset(client, api_key, &train_rows);

        let (train_file_id, val_file_id) = if let Some(val_rows) = val_rows.as_ref() {
            let val_fut = self.create_and_upload_dataset(client, api_key, val_rows);

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
                self.api_base
                    .join(&format!(
                        "v1/accounts/{}/supervisedFineTuningJobs",
                        self.account_id
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
            api_base: self.api_base.clone(),
            account_id: self.account_id.clone(),
            job_url: format!("https://app.fireworks.ai/dashboard/fine-tuning/supervised/{job_id}")
                .parse()
                .convert_parse_error()?,
            job_path: job.name,
            credential_location: self.credential_location.clone(),
        })
    }
}

#[async_trait]
impl JobHandle for FireworksSFTJobHandle {
    async fn poll(
        &self,
        client: &TensorzeroHttpClient,
        credentials: &InferenceCredentials,
        default_credentials: &ProviderTypeDefaultCredentials,
    ) -> Result<OptimizationJobInfo, Error> {
        use tensorzero_core::optimization::fireworks_sft::{
            FireworksDeploymentState, FireworksFineTuningJobState,
        };

        let fireworks_credentials: FireworksCredentials = FireworksKind
            .get_defaulted_credential(self.credential_location.as_ref(), default_credentials)
            .await?;
        let api_key = fireworks_credentials
            .get_api_key(credentials)
            .map_err(|e| e.log())?;
        let job_status = self.poll_job(client, api_key).await?;
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
            let deployment_state = self
                .deploy_or_poll_model(client, api_key, &model_path)
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
                FireworksDeploymentState::Deployed => {
                    let model_provider = UninitializedModelProvider {
                        config: UninitializedProviderConfig::Fireworks {
                            model_name: model_path.clone(),
                            parse_think_blocks: true,
                            api_key_location: None,
                        },
                        extra_headers: None,
                        extra_body: None,
                        timeouts: TimeoutsConfig::default(),
                        discard_unknown_chunks: false,
                    };
                    Ok(OptimizationJobInfo::Completed {
                        output: OptimizerOutput::Model(UninitializedModelConfig {
                            routing: vec![model_path.clone().into()],
                            providers: HashMap::from([(model_path.into(), model_provider)]),
                            timeouts: TimeoutsConfig::default(),
                        }),
                    })
                }
            }
        } else {
            convert_to_optimizer_status(job_status)
        }
    }
}

pub fn convert_to_optimizer_status(
    job: tensorzero_core::optimization::fireworks_sft::FireworksFineTuningJobResponse,
) -> Result<OptimizationJobInfo, Error> {
    use tensorzero_core::optimization::fireworks_sft::FireworksFineTuningJobState;

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
            }))
        }
    })
}
