use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{borrow::Cow, collections::HashMap, fmt::Display};

use super::{
    prepare_gcp_vertex_gemini_messages, tensorzero_to_gcp_vertex_gemini_content,
    GCPVertexGeminiFileURI, GCPVertexGeminiSupervisedRow,
};
use crate::{
    config::TimeoutsConfig,
    error::{Error, ErrorDetails},
    inference::types::{ContentBlock, FlattenUnknown},
    model::{
        CredentialLocationWithFallback, UninitializedModelConfig, UninitializedModelProvider,
        UninitializedProviderConfig,
    },
    optimization::{OptimizationJobInfo, OptimizerOutput},
    providers::gcp_vertex_gemini::{
        GCPVertexGeminiContent, GCPVertexGeminiContentPart, GCPVertexGeminiPartData,
        GCPVertexGeminiRole, PROVIDER_TYPE,
    },
    stored_inference::LazyRenderedSample,
    tool::Tool,
};

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct EncryptionSpec {
    pub kms_key_name: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct SupervisedHyperparameters {
    pub epoch_count: Option<usize>,
    pub adapter_size: Option<usize>,
    pub learning_rate_multiplier: Option<f64>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct SupervisedTuningSpec {
    pub training_dataset_uri: GCPVertexGeminiFileURI,
    pub validation_dataset_uri: Option<GCPVertexGeminiFileURI>,
    pub hyper_parameters: Option<SupervisedHyperparameters>,
    #[serde(rename = "export_last_checkpoint_only")]
    pub export_last_checkpoint_only: Option<bool>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct GCPVertexGeminiFineTuningRequest {
    pub base_model: String,
    pub supervised_tuning_spec: SupervisedTuningSpec,
    pub tuned_model_display_name: Option<String>,
    pub service_account: Option<String>,
    pub encryption_spec: Option<EncryptionSpec>,
}

impl<'a> GCPVertexGeminiSupervisedRow<'a> {
    pub async fn from_rendered_sample(inference: &'a LazyRenderedSample) -> Result<Self, Error> {
        let tools = inference
            .tool_params
            .additional_tools
            .as_ref()
            .map(|tools| {
                tools
                    .iter()
                    .filter_map(|dt| match &dt {
                        Tool::Function(func) => Some(func.into()),
                        Tool::OpenAICustom(_) => None, // Skip custom tools for SFT
                    })
                    .collect()
            })
            .unwrap_or_default();
        let mut contents = prepare_gcp_vertex_gemini_messages(&inference.messages).await?;
        let system_instruction =
            inference
                .system_input
                .as_ref()
                .map(|system_instruction| GCPVertexGeminiContent {
                    role: GCPVertexGeminiRole::System,
                    parts: vec![GCPVertexGeminiContentPart {
                        thought: false,
                        thought_signature: None,
                        data: FlattenUnknown::Normal(GCPVertexGeminiPartData::Text {
                            text: Cow::Borrowed(system_instruction),
                        }),
                    }],
                });
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
        let final_model_message = tensorzero_to_gcp_vertex_gemini_content(
            GCPVertexGeminiRole::Model,
            Cow::Owned(output_content_blocks),
            PROVIDER_TYPE,
        )
        .await?;
        contents.push(final_model_message);
        Ok(Self {
            contents,
            system_instruction,
            tools,
        })
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SupervisedTunedModel {
    pub model: String,
    pub endpoint: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct SupervisedTuningDataStatistics {
    pub tuning_dataset_example_count: u64,
    pub total_billable_token_count: u64,
    pub tuning_step_count: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct TuningDataStatistics {
    pub supervised_tuning_data_stats: SupervisedTuningDataStatistics,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GCPVertexGeminiFineTuningJob {
    // There are more fields that echo some of the fields in the GCPVertexGeminiFineTuningRequest
    // but we don't need them for now
    pub error: Option<Value>,
    pub name: String,
    pub create_time: String,      // Unix timestamp in seconds
    pub end_time: Option<String>, // Unix timestamp in seconds
    pub tuned_model: Option<SupervisedTunedModel>,
    pub experiment: Option<String>,
    #[serde(rename = "tuning_data_statistics")]
    pub tuning_data_statistics: Option<TuningDataStatistics>,
    pub state: GCPVertexGeminiFineTuningJobStatus,
}

pub fn convert_to_optimizer_status(
    job: GCPVertexGeminiFineTuningJob,
    location: String,
    project_id: String,
    credential_location: Option<CredentialLocationWithFallback>,
) -> Result<OptimizationJobInfo, Error> {
    let estimated_finish: Option<DateTime<Utc>> = None;

    let trained_tokens: Option<u64> = job.tuning_data_statistics.map(|stats| {
        stats
            .supervised_tuning_data_stats
            .total_billable_token_count
    });
    Ok(match job.state {
        GCPVertexGeminiFineTuningJobStatus::JobStateUnspecified
        | GCPVertexGeminiFineTuningJobStatus::JobStateQueued
        | GCPVertexGeminiFineTuningJobStatus::JobStatePending
        | GCPVertexGeminiFineTuningJobStatus::JobStateRunning
        | GCPVertexGeminiFineTuningJobStatus::JobStatePaused
        | GCPVertexGeminiFineTuningJobStatus::JobStateUpdating => OptimizationJobInfo::Pending {
            message: job.state.to_string(),
            estimated_finish,
            trained_tokens,
            error: job.error,
        },
        GCPVertexGeminiFineTuningJobStatus::JobStateFailed
        | GCPVertexGeminiFineTuningJobStatus::JobStateCancelling
        | GCPVertexGeminiFineTuningJobStatus::JobStateCancelled
        | GCPVertexGeminiFineTuningJobStatus::JobStateExpired
        | GCPVertexGeminiFineTuningJobStatus::JobStatePartiallySucceeded => {
            OptimizationJobInfo::Failed {
                message: job.state.to_string(),
                error: job.error,
            }
        }
        GCPVertexGeminiFineTuningJobStatus::JobStateSucceeded => {
            let tuned_model = job.tuned_model.as_ref().ok_or_else(|| {
                Error::new(ErrorDetails::OptimizationResponse {
                    message: "No tuned_model in Succeeded response".to_string(),
                    provider_type: super::PROVIDER_TYPE.to_string(),
                })
            })?;

            let model_name: String = tuned_model
                .endpoint
                .as_ref()
                .ok_or_else(|| {
                    Error::new(ErrorDetails::OptimizationResponse {
                        message: "Tuned model must have an endpoint".to_string(),
                        provider_type: super::PROVIDER_TYPE.to_string(),
                    })
                })?
                .clone();

            let endpoint_id = tuned_model
                .endpoint
                .as_ref()
                .and_then(|endpoint| endpoint.rsplit('/').next())
                .map(str::to_string);

            let model_provider = UninitializedModelProvider {
                config: UninitializedProviderConfig::GCPVertexGemini {
                    model_id: None,
                    endpoint_id,
                    location,
                    project_id,
                    credential_location,
                },
                extra_headers: None,
                extra_body: None,
                timeouts: TimeoutsConfig::default(),
                discard_unknown_chunks: false,
            };
            OptimizationJobInfo::Completed {
                output: OptimizerOutput::Model(UninitializedModelConfig {
                    routing: vec![model_name.clone().into()],
                    providers: HashMap::from([(model_name.clone().into(), model_provider)]),
                    timeouts: TimeoutsConfig::default(),
                }),
            }
        }
    })
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum GCPVertexGeminiFineTuningJobStatus {
    JobStateUnspecified,
    JobStateQueued,
    JobStatePending,
    JobStateRunning,
    JobStateSucceeded,
    JobStateFailed,
    JobStateCancelling,
    JobStateCancelled,
    JobStatePaused,
    JobStateExpired,
    JobStateUpdating,
    JobStatePartiallySucceeded,
}

// Get the 'SCREAMING_SNAKE_CASE' name for the enum value
impl Display for GCPVertexGeminiFineTuningJobStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.serialize(f)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        inference::types::{
            ContentBlockChatOutput, ModelInput, ResolvedContentBlock, ResolvedRequestMessage, Role,
            StoredInput, StoredInputMessage, StoredInputMessageContent, System, Text,
        },
        stored_inference::{RenderedSample, StoredOutput},
        tool::DynamicToolParams,
    };
    use serde_json::json;

    use super::*;

    #[tokio::test]
    async fn test_convert_to_sft_row() {
        let output = Some(vec![ContentBlockChatOutput::Text(Text {
            text: "The capital of France is Paris.".to_string(),
        })]);
        let inference = RenderedSample {
            function_name: "test".to_string(),
            input: ModelInput {
                system: Some("You are a helpful assistant named Dr. M.M. Patel.".to_string()),
                messages: vec![ResolvedRequestMessage {
                    role: Role::User,
                    content: vec![ResolvedContentBlock::Text(Text {
                        text: "What is the capital of France?".to_string(),
                    })],
                }],
            },
            stored_input: StoredInput {
                system: Some(System::Text(
                    "You are a helpful assistant named Dr. M.M. Patel.".to_string(),
                )),
                messages: vec![StoredInputMessage {
                    role: Role::User,
                    content: vec![StoredInputMessageContent::Text(Text {
                        text: "What is the capital of France?".to_string(),
                    })],
                }],
            },
            output: output.clone(),
            stored_output: output.map(StoredOutput::Chat),
            episode_id: Some(uuid::Uuid::now_v7()),
            inference_id: Some(uuid::Uuid::now_v7()),
            tool_params: DynamicToolParams::default(),
            output_schema: None,
            dispreferred_outputs: vec![],
            tags: HashMap::from([("test_key".to_string(), "test_value".to_string())]),
        };
        let lazy_inference = inference.into_lazy_rendered_sample();
        let row = GCPVertexGeminiSupervisedRow::from_rendered_sample(&lazy_inference)
            .await
            .unwrap();

        // Check that we have the expected number of messages (user + assistant)
        assert_eq!(row.contents.len(), 2);

        // Check system instruction
        let system_instruction = row.system_instruction.as_ref().unwrap();
        assert_eq!(system_instruction.role, GCPVertexGeminiRole::System);

        let system_text = system_instruction
            .parts
            .iter()
            .filter_map(|part| match &part.data {
                FlattenUnknown::Normal(GCPVertexGeminiPartData::Text { text }) => {
                    Some(text.as_ref())
                }
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("");

        assert_eq!(
            system_text,
            "You are a helpful assistant named Dr. M.M. Patel."
        );

        // Check user message
        assert_eq!(row.contents[0].role, GCPVertexGeminiRole::User);
        let user_part = &row.contents[0].parts[0];
        match &user_part.data {
            FlattenUnknown::Normal(GCPVertexGeminiPartData::Text { text }) => {
                assert_eq!(text, "What is the capital of France?");
            }
            _ => panic!("First message should be a text message"),
        }

        // Check assistant message
        assert_eq!(row.contents[1].role, GCPVertexGeminiRole::Model);
        let assistant_part = &row.contents[1].parts[0];
        match &assistant_part.data {
            FlattenUnknown::Normal(GCPVertexGeminiPartData::Text { text }) => {
                assert_eq!(text, "The capital of France is Paris.");
            }
            _ => panic!("Second message should be a text message"),
        }

        // Check tools
        assert_eq!(row.tools.len(), 0);
    }

    #[test]
    fn test_convert_to_optimizer_status() {
        let location = "us-central1".to_string();
        let project_id = "test-project".to_string();

        // Test for "succeeded" status with a model output
        let succeeded_model = json!({
            "name": "projects/test-project/locations/us-central1/tuningJobs/12345",
            "state": "JOB_STATE_SUCCEEDED",
            "createTime": "1620000000",
            "tunedModel": {
                "model": "projects/test-project/locations/us-central1/models/gemini-1.5-flash-001-tuned-12345",
                "endpoint": "projects/test-project/locations/us-central1/endpoints/67890"
            },
            "tuning_data_statistics": {
                "supervisedTuningDataStats": {
                    "tuningDatasetExampleCount": 1000,
                    "totalBillableTokenCount": 50000,
                    "tuningStepCount": 100
                }
            }
        });
        let job = serde_json::from_value::<GCPVertexGeminiFineTuningJob>(succeeded_model).unwrap();
        let status =
            convert_to_optimizer_status(job, location.clone(), project_id.clone(), None).unwrap();
        assert!(matches!(
            status,
            OptimizationJobInfo::Completed {
                output: OptimizerOutput::Model(_),
            }
        ));

        // Test for "succeeded" status with tuned model but no endpoint - should error
        let succeeded_no_endpoint = json!({
            "name": "projects/test-project/locations/us-central1/tuningJobs/12358",
            "state": "JOB_STATE_SUCCEEDED",
            "createTime": "1620000000",
            "tunedModel": {
                "model": "projects/test-project/locations/us-central1/models/gemini-1.5-flash-001-tuned-12345"
                // No endpoint field
            },
            "tuning_data_statistics": {
                "supervisedTuningDataStats": {
                    "tuningDatasetExampleCount": 1000,
                    "totalBillableTokenCount": 50000,
                    "tuningStepCount": 100
                }
            }
        });
        let job =
            serde_json::from_value::<GCPVertexGeminiFineTuningJob>(succeeded_no_endpoint).unwrap();
        let result = convert_to_optimizer_status(job, location.clone(), project_id.clone(), None);

        // Should error when endpoint is missing
        assert!(result.is_err());
        // Optionally, you can also check the error message
        if let Err(error) = result {
            assert!(error
                .to_string()
                .contains("Tuned model must have an endpoint"));
        }

        // Test for "running" status
        let running = json!({
            "name": "projects/test-project/locations/us-central1/tuningJobs/12346",
            "state": "JOB_STATE_RUNNING",
            "createTime": "1620000000",
            "tuning_data_statistics": {
                "supervisedTuningDataStats": {
                    "tuningDatasetExampleCount": 1000,
                    "totalBillableTokenCount": 50000,
                    "tuningStepCount": 100
                }
            }
        });
        let job = serde_json::from_value::<GCPVertexGeminiFineTuningJob>(running).unwrap();
        let status =
            convert_to_optimizer_status(job, location.clone(), project_id.clone(), None).unwrap();
        assert!(matches!(status, OptimizationJobInfo::Pending { .. }));

        // Test for "failed" status
        let failed = json!({
            "name": "projects/test-project/locations/us-central1/tuningJobs/12347",
            "state": "JOB_STATE_FAILED",
            "createTime": "1620000000",
            "tuning_data_statistics": {
                "supervisedTuningDataStats": {
                    "tuningDatasetExampleCount": 1000,
                    "totalBillableTokenCount": 50000,
                    "tuningStepCount": 100
                }
            }
        });
        let job = serde_json::from_value::<GCPVertexGeminiFineTuningJob>(failed).unwrap();
        let status =
            convert_to_optimizer_status(job, location.clone(), project_id.clone(), None).unwrap();
        assert!(matches!(status, OptimizationJobInfo::Failed { .. }));

        // Test for "queued" status
        let queued = json!({
            "name": "projects/test-project/locations/us-central1/tuningJobs/12348",
            "state": "JOB_STATE_QUEUED",
            "createTime": "1620000000",
            "tuning_data_statistics": {
                "supervisedTuningDataStats": {
                    "tuningDatasetExampleCount": 1000,
                    "totalBillableTokenCount": 50000,
                    "tuningStepCount": 100
                }
            }
        });
        let job = serde_json::from_value::<GCPVertexGeminiFineTuningJob>(queued).unwrap();
        let status =
            convert_to_optimizer_status(job, location.clone(), project_id.clone(), None).unwrap();
        assert!(matches!(status, OptimizationJobInfo::Pending { .. }));

        // Test for "pending" status
        let pending = json!({
            "name": "projects/test-project/locations/us-central1/tuningJobs/12349",
            "state": "JOB_STATE_PENDING",
            "createTime": "1620000000",
            "tuning_data_statistics": {
                "supervisedTuningDataStats": {
                    "tuningDatasetExampleCount": 1000,
                    "totalBillableTokenCount": 50000,
                    "tuningStepCount": 100
                }
            }
        });
        let job = serde_json::from_value::<GCPVertexGeminiFineTuningJob>(pending).unwrap();
        let status =
            convert_to_optimizer_status(job, location.clone(), project_id.clone(), None).unwrap();
        assert!(matches!(status, OptimizationJobInfo::Pending { .. }));

        // Test for "pending" status with tuned model but no endpoint
        let pending_with_model_no_endpoint = json!({
            "name": "projects/test-project/locations/us-central1/tuningJobs/12357",
            "state": "JOB_STATE_PENDING",
            "createTime": "1620000000",
            "tunedModel": {
                "model": "projects/test-project/locations/us-central1/models/gemini-1.5-flash-001-tuned-12345"
                // No endpoint field - this is the key part of the test
            },
            "tuning_data_statistics": {
                "supervisedTuningDataStats": {
                    "tuningDatasetExampleCount": 1000,
                    "totalBillableTokenCount": 50000,
                    "tuningStepCount": 100
                }
            }
        });
        let job =
            serde_json::from_value::<GCPVertexGeminiFineTuningJob>(pending_with_model_no_endpoint)
                .unwrap();
        let status =
            convert_to_optimizer_status(job, location.clone(), project_id.clone(), None).unwrap();
        assert!(matches!(status, OptimizationJobInfo::Pending { .. }));

        // Test for "cancelled" status
        let cancelled = json!({
            "name": "projects/test-project/locations/us-central1/tuningJobs/12350",
            "state": "JOB_STATE_CANCELLED",
            "createTime": "1620000000",
            "tuning_data_statistics": {
                "supervisedTuningDataStats": {
                    "tuningDatasetExampleCount": 1000,
                    "totalBillableTokenCount": 50000,
                    "tuningStepCount": 100
                }
            }
        });
        let job = serde_json::from_value::<GCPVertexGeminiFineTuningJob>(cancelled).unwrap();
        let status =
            convert_to_optimizer_status(job, location.clone(), project_id.clone(), None).unwrap();
        assert!(matches!(status, OptimizationJobInfo::Failed { .. }));

        // Test for "paused" status
        let paused = json!({
            "name": "projects/test-project/locations/us-central1/tuningJobs/12351",
            "state": "JOB_STATE_PAUSED",
            "createTime": "1620000000",
            "tuning_data_statistics": {
                "supervisedTuningDataStats": {
                    "tuningDatasetExampleCount": 1000,
                    "totalBillableTokenCount": 50000,
                    "tuningStepCount": 100
                }
            }
        });
        let job = serde_json::from_value::<GCPVertexGeminiFineTuningJob>(paused).unwrap();
        let status =
            convert_to_optimizer_status(job, location.clone(), project_id.clone(), None).unwrap();
        assert!(matches!(status, OptimizationJobInfo::Pending { .. }));

        // Test for "expired" status
        let expired = json!({
            "name": "projects/test-project/locations/us-central1/tuningJobs/12352",
            "state": "JOB_STATE_EXPIRED",
            "createTime": "1620000000",
            "tuning_data_statistics": {
                "supervisedTuningDataStats": {
                    "tuningDatasetExampleCount": 1000,
                    "totalBillableTokenCount": 50000,
                    "tuningStepCount": 100
                }
            }
        });
        let job = serde_json::from_value::<GCPVertexGeminiFineTuningJob>(expired).unwrap();
        let status =
            convert_to_optimizer_status(job, location.clone(), project_id.clone(), None).unwrap();
        assert!(matches!(status, OptimizationJobInfo::Failed { .. }));

        // Test for "updating" status
        let updating = json!({
            "name": "projects/test-project/locations/us-central1/tuningJobs/12353",
            "state": "JOB_STATE_UPDATING",
            "createTime": "1620000000",
            "tuning_data_statistics": {
                "supervisedTuningDataStats": {
                    "tuningDatasetExampleCount": 1000,
                    "totalBillableTokenCount": 50000,
                    "tuningStepCount": 100
                }
            }
        });
        let job = serde_json::from_value::<GCPVertexGeminiFineTuningJob>(updating).unwrap();
        let status =
            convert_to_optimizer_status(job, location.clone(), project_id.clone(), None).unwrap();
        assert!(matches!(status, OptimizationJobInfo::Pending { .. }));

        // Test for "partially succeeded" status
        let partially_succeeded = json!({
            "name": "projects/test-project/locations/us-central1/tuningJobs/12354",
            "state": "JOB_STATE_PARTIALLY_SUCCEEDED",
            "createTime": "1620000000",
            "tuning_data_statistics": {
                "supervisedTuningDataStats": {
                    "tuningDatasetExampleCount": 1000,
                    "totalBillableTokenCount": 50000,
                    "tuningStepCount": 100
                }
            }
        });
        let job =
            serde_json::from_value::<GCPVertexGeminiFineTuningJob>(partially_succeeded).unwrap();
        let status =
            convert_to_optimizer_status(job, location.clone(), project_id.clone(), None).unwrap();
        assert!(matches!(status, OptimizationJobInfo::Failed { .. }));

        // Test for "succeeded" status but missing tuned_model - should error
        let succeeded_missing_model = json!({
            "name": "projects/test-project/locations/us-central1/tuningJobs/12355",
            "state": "JOB_STATE_SUCCEEDED",
            "createTime": "1620000000",
            "tuning_data_statistics": {
                "supervisedTuningDataStats": {
                    "tuningDatasetExampleCount": 1000,
                    "totalBillableTokenCount": 50000,
                    "tuningStepCount": 100
                }
            }
        });
        let job = serde_json::from_value::<GCPVertexGeminiFineTuningJob>(succeeded_missing_model)
            .unwrap();
        let result = convert_to_optimizer_status(job, location.clone(), project_id.clone(), None);
        assert!(result.is_err());

        // Test for missing status field - this would fail deserialization of GCPVertexGeminiFineTuningJob
        let missing_status = json!({
            "name": "projects/test-project/locations/us-central1/tuningJobs/12356",
            "createTime": "1620000000",
            "tuning_data_statistics": {
                "supervisedTuningDataStats": {
                    "tuningDatasetExampleCount": 1000,
                    "totalBillableTokenCount": 50000,
                    "tuningStepCount": 100
                }
            }
        });
        assert!(serde_json::from_value::<GCPVertexGeminiFineTuningJob>(missing_status).is_err());
    }
}
