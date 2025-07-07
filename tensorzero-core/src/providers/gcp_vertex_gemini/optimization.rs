use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{borrow::Cow, collections::HashMap};

use super::{
    prepare_gcp_vertex_gemini_messages, tensorzero_to_gcp_vertex_gemini_model_message,
    tensorzero_to_gcp_vertex_gemini_system_message, GCPVertexGeminiFileURI,
    GCPVertexGeminiSupervisedRow,
};
use crate::{
    config_parser::TimeoutsConfig,
    error::{Error, ErrorDetails},
    inference::types::ContentBlock,
    model::CredentialLocation,
    model::{UninitializedModelConfig, UninitializedModelProvider, UninitializedProviderConfig},
    optimization::{OptimizerOutput, OptimizerStatus},
    providers::gcp_vertex_gemini::PROVIDER_TYPE,
    stored_inference::RenderedSample,
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

impl<'a> TryFrom<&'a RenderedSample> for GCPVertexGeminiSupervisedRow<'a> {
    type Error = Error;
    fn try_from(inference: &'a RenderedSample) -> Result<Self, Self::Error> {
        let tools = vec![]; // TODO: add tools maybe? They do not seem to affect the job and increase the size of the dataset
        let mut contents =
            prepare_gcp_vertex_gemini_messages(&inference.input.messages, PROVIDER_TYPE)?;
        let system_instruction = tensorzero_to_gcp_vertex_gemini_system_message(
            inference.input.system.as_deref(),
            None,
            &contents,
        );
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
        let final_model_message = tensorzero_to_gcp_vertex_gemini_model_message(
            Cow::Owned(output_content_blocks),
            PROVIDER_TYPE,
        )?;
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
    credential_location: CredentialLocation,
) -> Result<OptimizerStatus, Error> {
    let estimated_finish: Option<DateTime<Utc>> = None;

    let trained_tokens: Option<u64> = job.tuning_data_statistics.map(|stats| {
        stats
            .supervised_tuning_data_stats
            .total_billable_token_count
    });
    Ok(match job.state {
        GCPVertexGeminiFineTuningJobStatus::JobStateUnspecified => OptimizerStatus::Pending {
            message: "Job State Unspecified".to_string(),
            estimated_finish,
            trained_tokens,
            error: job.error,
        },
        GCPVertexGeminiFineTuningJobStatus::JobStateQueued => OptimizerStatus::Pending {
            message: "Job State Queued".to_string(),
            estimated_finish,
            trained_tokens,
            error: job.error,
        },
        GCPVertexGeminiFineTuningJobStatus::JobStatePending => OptimizerStatus::Pending {
            message: "Job State Pending".to_string(),
            estimated_finish,
            trained_tokens,
            error: job.error,
        },
        GCPVertexGeminiFineTuningJobStatus::JobStateRunning => OptimizerStatus::Pending {
            message: "Running".to_string(),
            estimated_finish,
            trained_tokens,
            error: job.error,
        },
        GCPVertexGeminiFineTuningJobStatus::JobStateSucceeded => {
            let tuned_model = job.tuned_model.as_ref().ok_or_else(|| {
                Error::new(ErrorDetails::OptimizationResponse {
                    message: "No tuned_model in Succeeded response".to_string(),
                    provider_type: super::PROVIDER_TYPE.to_string(),
                })
            })?;

            // Use endpoint if available, otherwise fall back to model ID
            let model_name: String = tuned_model
                .endpoint
                .as_ref()
                .unwrap_or(&tuned_model.model)
                .clone();

            let model_provider = UninitializedModelProvider {
                config: UninitializedProviderConfig::GCPVertexGemini {
                    model_id: None,
                    endpoint_id: tuned_model.endpoint.clone(), // This is now Option<String>
                    location,
                    project_id,
                    credential_location: Some(credential_location),
                },
                extra_headers: None,
                extra_body: None,
                timeouts: None,
                discard_unknown_chunks: false,
            };
            OptimizerStatus::Completed {
                output: OptimizerOutput::Model(UninitializedModelConfig {
                    routing: vec![model_name.clone().into()],
                    providers: HashMap::from([(model_name.clone().into(), model_provider)]),
                    timeouts: TimeoutsConfig::default(),
                }),
            }
        }
        GCPVertexGeminiFineTuningJobStatus::JobStateFailed => OptimizerStatus::Failed {
            message: "Failed".to_string(),
            error: job.error,
        },
        GCPVertexGeminiFineTuningJobStatus::JobStateCancelling => OptimizerStatus::Failed {
            message: "Cancelling".to_string(),
            error: job.error,
        },
        GCPVertexGeminiFineTuningJobStatus::JobStateCancelled => OptimizerStatus::Failed {
            message: "Cancelled".to_string(),
            error: job.error,
        },
        GCPVertexGeminiFineTuningJobStatus::JobStatePaused => OptimizerStatus::Pending {
            message: "Paused".to_string(),
            estimated_finish,
            trained_tokens,
            error: job.error,
        },
        GCPVertexGeminiFineTuningJobStatus::JobStateExpired => OptimizerStatus::Failed {
            message: "Expired".to_string(),
            error: job.error,
        },
        GCPVertexGeminiFineTuningJobStatus::JobStateUpdating => OptimizerStatus::Pending {
            message: "Updating".to_string(),
            estimated_finish,
            trained_tokens,
            error: job.error,
        },
        GCPVertexGeminiFineTuningJobStatus::JobStatePartiallySucceeded => OptimizerStatus::Failed {
            message: "Partially Succeeded".to_string(),
            error: job.error,
        },
    })
}

#[derive(Debug, Deserialize)]
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

#[cfg(test)]
mod tests {
    use crate::{
        inference::types::{ContentBlockChatOutput, ModelInput, RequestMessage, Role, Text},
        model::CredentialLocation,
        providers::gcp_vertex_gemini::{GCPVertexGeminiContentPart, GCPVertexGeminiRequestMessage},
    };
    use serde_json::json;

    use super::*;

    #[test]
    fn test_convert_to_sft_row() {
        let inference = RenderedSample {
            function_name: "test".to_string(),
            input: ModelInput {
                system: Some("You are a helpful assistant named Dr. M.M. Patel.".to_string()),
                messages: vec![RequestMessage {
                    role: Role::User,
                    content: vec![ContentBlock::Text(Text {
                        text: "What is the capital of France?".to_string(),
                    })],
                }],
            },
            output: Some(vec![ContentBlockChatOutput::Text(Text {
                text: "The capital of France is Paris.".to_string(),
            })]),
            episode_id: Some(uuid::Uuid::now_v7()),
            inference_id: Some(uuid::Uuid::now_v7()),
            tool_params: None,
            output_schema: None,
            dispreferred_outputs: vec![],
        };
        let row = GCPVertexGeminiSupervisedRow::try_from(&inference).unwrap();

        // Check that we have the expected number of messages (user + assistant)
        assert_eq!(row.contents.len(), 2);

        // Check system instruction
        let system_text = if let GCPVertexGeminiRequestMessage::System(system_msg) =
            row.system_instruction.as_ref().unwrap()
        {
            system_msg
                .parts
                .iter()
                .filter_map(|part| {
                    if let GCPVertexGeminiContentPart::Text { text } = part {
                        Some(text.as_ref())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join("")
        } else {
            panic!("Expected system message")
        };

        assert_eq!(
            system_text,
            "You are a helpful assistant named Dr. M.M. Patel."
        );

        // Check user message
        let GCPVertexGeminiRequestMessage::User(user_message) = &row.contents[0] else {
            panic!("First message should be a user message");
        };
        let GCPVertexGeminiContentPart::Text { text } = &user_message.parts[0] else {
            panic!("First message should be a text message");
        };
        assert_eq!(text, "What is the capital of France?");

        // Check assistant message
        let GCPVertexGeminiRequestMessage::Model(assistant_message) = &row.contents[1] else {
            panic!("Second message should be a model/assistant message");
        };
        let GCPVertexGeminiContentPart::Text { text } = &assistant_message.parts[0] else {
            panic!("Second message should be a text message");
        };
        assert_eq!(text, "The capital of France is Paris.");

        // Check tools
        assert_eq!(row.tools.len(), 0);
    }

    #[test]
    fn test_convert_to_optimizer_status() {
        let location = "us-central1".to_string();
        let project_id = "test-project".to_string();
        let credential_location = CredentialLocation::Path("path/to/creds.json".to_string());

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
        let status = convert_to_optimizer_status(
            job,
            location.clone(),
            project_id.clone(),
            credential_location.clone(),
        )
        .unwrap();
        assert!(matches!(
            status,
            OptimizerStatus::Completed {
                output: OptimizerOutput::Model(_),
            }
        ));

        // Test for "succeeded" status with tuned model but no endpoint - should use model ID as fallback
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
        let status = convert_to_optimizer_status(
            job,
            location.clone(),
            project_id.clone(),
            credential_location.clone(),
        )
        .unwrap();
        assert!(matches!(
            status,
            OptimizerStatus::Completed {
                output: OptimizerOutput::Model(_),
            }
        ));

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
        let status = convert_to_optimizer_status(
            job,
            location.clone(),
            project_id.clone(),
            credential_location.clone(),
        )
        .unwrap();
        assert!(matches!(status, OptimizerStatus::Pending { .. }));

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
        let status = convert_to_optimizer_status(
            job,
            location.clone(),
            project_id.clone(),
            credential_location.clone(),
        )
        .unwrap();
        assert!(matches!(status, OptimizerStatus::Failed { .. }));

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
        let status = convert_to_optimizer_status(
            job,
            location.clone(),
            project_id.clone(),
            credential_location.clone(),
        )
        .unwrap();
        assert!(matches!(status, OptimizerStatus::Pending { .. }));

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
        let status = convert_to_optimizer_status(
            job,
            location.clone(),
            project_id.clone(),
            credential_location.clone(),
        )
        .unwrap();
        assert!(matches!(status, OptimizerStatus::Pending { .. }));

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
        let status = convert_to_optimizer_status(
            job,
            location.clone(),
            project_id.clone(),
            credential_location.clone(),
        )
        .unwrap();
        assert!(matches!(status, OptimizerStatus::Pending { .. }));

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
        let status = convert_to_optimizer_status(
            job,
            location.clone(),
            project_id.clone(),
            credential_location.clone(),
        )
        .unwrap();
        assert!(matches!(status, OptimizerStatus::Failed { .. }));

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
        let status = convert_to_optimizer_status(
            job,
            location.clone(),
            project_id.clone(),
            credential_location.clone(),
        )
        .unwrap();
        assert!(matches!(status, OptimizerStatus::Pending { .. }));

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
        let status = convert_to_optimizer_status(
            job,
            location.clone(),
            project_id.clone(),
            credential_location.clone(),
        )
        .unwrap();
        assert!(matches!(status, OptimizerStatus::Failed { .. }));

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
        let status = convert_to_optimizer_status(
            job,
            location.clone(),
            project_id.clone(),
            credential_location.clone(),
        )
        .unwrap();
        assert!(matches!(status, OptimizerStatus::Pending { .. }));

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
        let status = convert_to_optimizer_status(
            job,
            location.clone(),
            project_id.clone(),
            credential_location.clone(),
        )
        .unwrap();
        assert!(matches!(status, OptimizerStatus::Failed { .. }));

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
        let result = convert_to_optimizer_status(
            job,
            location.clone(),
            project_id.clone(),
            credential_location.clone(),
        );
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
