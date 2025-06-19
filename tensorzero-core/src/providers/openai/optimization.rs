use chrono::DateTime;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{borrow::Cow, collections::HashMap};

use super::{
    prepare_openai_messages, tensorzero_to_openai_assistant_message, OpenAICredentials,
    OpenAIFileID, OpenAIProvider, OpenAIRequestMessage, OpenAISFTTool,
};
use crate::{
    config_parser::TimeoutsConfig,
    error::{Error, ErrorDetails},
    inference::types::ContentBlock,
    model::{ModelConfig, ModelProvider, ProviderConfig},
    optimization::{OptimizerOutput, OptimizerStatus},
    stored_inference::RenderedStoredInference,
};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAIFineTuningRequest {
    pub model: String,
    pub training_file: OpenAIFileID,
    pub metadata: Option<HashMap<String, String>>,
    pub method: OpenAIFineTuningMethod,
    pub seed: Option<u64>,
    pub suffix: Option<String>,
    pub validation_file: Option<OpenAIFileID>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpenAIFineTuningMethod {
    #[serde(rename = "dpo")]
    Dpo {
        dpo: Dpo,
    },
    Supervised {
        supervised: Supervised,
    },
    Reinforcement {
        reinforcement: Reinforcement,
    },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Dpo {
    pub hyperparameters: Option<DPOHyperparameters>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Supervised {
    pub hyperparameters: Option<SupervisedHyperparameters>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Reinforcement {
    pub grader: Box<Grader>,
    pub hyperparameters: Option<ReinforcementHyperparameters>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DPOHyperparameters {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub batch_size: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub beta: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub learning_rate_multiplier: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_epochs: Option<usize>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SupervisedHyperparameters {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub batch_size: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub learning_rate_multiplier: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_epochs: Option<usize>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ReinforcementHyperparameters {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub batch_size: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compute_multiplier: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_interval: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_samples: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub learning_rate_multiplier: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_epochs: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Grader {
    StringCheck {
        input: String,
        name: String,
        operation: String,
        reference: String,
    },
    TextSimilarity {
        evaluation_metric: String,
        input: String,
        name: String,
        reference: String,
    },
    // TODO: add an option to load from uninitialized with a Python script
    // We cannot ship Python in TOML here
    Python {
        name: String,
        source: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        image_tag: Option<String>,
    },
    ScoreModel {
        input: Vec<serde_json::Value>,
        model: String,
        name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        range: Option<[f64; 2]>,
        #[serde(skip_serializing_if = "Option::is_none")]
        sampling_params: Option<serde_json::Value>,
    },
    LabelModel {
        input: Vec<serde_json::Value>,
        labels: Vec<String>,
        model: String,
        name: String,
        passing_labels: Vec<String>,
    },
    Multi {
        calculate_output: String,
        graders: HashMap<String, Box<Grader>>,
        name: String,
    },
}

#[derive(Debug, Serialize)]
pub struct OpenAISupervisedRow<'a> {
    messages: Vec<OpenAIRequestMessage<'a>>,
    parallel_tool_calls: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<OpenAISFTTool<'a>>,
}

impl<'a> TryFrom<&'a RenderedStoredInference> for OpenAISupervisedRow<'a> {
    type Error = Error;
    fn try_from(inference: &'a RenderedStoredInference) -> Result<Self, Self::Error> {
        let (parallel_tool_calls, tools) = match &inference.tool_params {
            Some(tool_params) => (
                tool_params.parallel_tool_calls.unwrap_or_default(),
                tool_params
                    .tools_available
                    .iter()
                    .map(|t| t.into())
                    .collect(),
            ),
            None => (false, vec![]),
        };
        let mut messages = prepare_openai_messages(
            inference.input.system.as_deref(),
            &inference.input.messages,
            None,
        )?;
        if inference.output.is_empty() {
            return Err(Error::new(ErrorDetails::InvalidRenderedStoredInference {
                message: "No output in inference".to_string(),
            }));
        }
        let output_content_blocks: Vec<ContentBlock> = inference
            .output
            .iter()
            .map(|c| c.clone().into())
            .collect::<Vec<_>>();
        let final_assistant_message =
            tensorzero_to_openai_assistant_message(Cow::Owned(output_content_blocks))?;
        messages.push(final_assistant_message);
        // TODO: add a test that makes sure the last message is from the assistant
        Ok(Self {
            messages,
            parallel_tool_calls,
            tools,
        })
    }
}

#[derive(Debug, Serialize)]
pub struct OpenAIPreferenceRow<'a> {
    input: OpenAISupervisedRow<'a>,
    // The following two fields need to be of variant OpenAIRequestMessage::Assistant
    // Not sure how to enforce this in Rust, maybe a constructor?
    non_preferred_output: OpenAIRequestMessage<'a>,
    preferred_output: OpenAIRequestMessage<'a>,
}

#[derive(Debug, Serialize)]
pub struct OpenAIReinforcementRow<'a> {
    messages: Vec<OpenAIRequestMessage<'a>>,
}

#[derive(Debug, Deserialize)]
pub struct OpenAIFineTuningJob {
    // There are more fields that echo some of the fields in the OpenAIFineTuningRequest
    // but we don't need them for now
    pub created_at: u64, // Unix timestamp in seconds
    pub error: Option<Value>,
    pub estimated_finish: Option<u64>, // Unix timestamp in seconds
    pub fine_tuned_model: Option<String>,
    pub finished_at: Option<u64>, // Unix timestamp in seconds
    pub id: String,
    pub metadata: Option<HashMap<String, String>>,
    pub trained_tokens: Option<u64>,
    pub status: OpenAIFineTuningJobStatus,
}

pub fn convert_to_optimizer_status(
    job: OpenAIFineTuningJob,
    credentials: &OpenAICredentials,
) -> Result<OptimizerStatus, Error> {
    let estimated_finish = job
        .estimated_finish
        .and_then(|unix_timestamp| DateTime::from_timestamp(unix_timestamp as i64, 0));
    Ok(match job.status {
        OpenAIFineTuningJobStatus::ValidatingFiles => OptimizerStatus::Pending {
            message: "Validating files".to_string(),
            estimated_finish,
            trained_tokens: job.trained_tokens,
            error: job.error,
        },
        OpenAIFineTuningJobStatus::Queued => OptimizerStatus::Pending {
            message: "Queued".to_string(),
            estimated_finish,
            trained_tokens: job.trained_tokens,
            error: job.error,
        },
        OpenAIFineTuningJobStatus::Running => OptimizerStatus::Pending {
            message: "Running".to_string(),
            estimated_finish,
            trained_tokens: job.trained_tokens,
            error: job.error,
        },
        OpenAIFineTuningJobStatus::Succeeded => {
            let model_name = job.fine_tuned_model.ok_or_else(|| {
                Error::new(ErrorDetails::OptimizationResponse {
                    message: "No fine-tuned model name in Succeeded response".to_string(),
                    provider_type: super::PROVIDER_TYPE.to_string(),
                })
            })?;
            let model_provider = ModelProvider {
                name: model_name.clone().into(),
                config: ProviderConfig::OpenAI(OpenAIProvider {
                    model_name: model_name.clone(),
                    api_base: None,
                    credentials: credentials.clone(),
                }),
                extra_headers: None,
                extra_body: None,
                timeouts: None,
                discard_unknown_chunks: false,
            };
            OptimizerStatus::Completed(OptimizerOutput::Model(ModelConfig {
                routing: vec![model_name.clone().into()],
                providers: HashMap::from([(model_name.clone().into(), model_provider)]),
                timeouts: TimeoutsConfig::default(),
            }))
        }
        OpenAIFineTuningJobStatus::Failed => OptimizerStatus::Failed,
        OpenAIFineTuningJobStatus::Cancelled => OptimizerStatus::Failed,
    })
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAIFineTuningJobStatus {
    ValidatingFiles,
    Queued,
    Running,
    Succeeded,
    Failed,
    Cancelled,
}

#[cfg(test)]
mod tests {
    use crate::{
        inference::types::{ContentBlockChatOutput, ModelInput, RequestMessage, Role, Text},
        providers::openai::OpenAIContentBlock,
    };
    use serde_json::json;
    use uuid::Uuid;

    use super::*;

    #[test]
    fn test_convert_to_sft_row() {
        let inference = RenderedStoredInference {
            function_name: "test".to_string(),
            variant_name: "test".to_string(),
            input: ModelInput {
                system: Some("You are a helpful assistant named Dr. M.M. Patel.".to_string()),
                messages: vec![RequestMessage {
                    role: Role::User,
                    content: vec![ContentBlock::Text(Text {
                        text: "What is the capital of France?".to_string(),
                    })],
                }],
            },
            output: vec![ContentBlockChatOutput::Text(Text {
                text: "The capital of France is Paris.".to_string(),
            })],
            episode_id: Uuid::now_v7(),
            inference_id: Uuid::now_v7(),
            tool_params: None,
            output_schema: None,
        };
        let row = OpenAISupervisedRow::try_from(&inference).unwrap();
        assert_eq!(row.messages.len(), 3);
        let OpenAIRequestMessage::System(system_message) = &row.messages[0] else {
            panic!("First message should be a system message");
        };
        assert_eq!(
            system_message.content,
            "You are a helpful assistant named Dr. M.M. Patel."
        );
        let OpenAIRequestMessage::User(user_message) = &row.messages[1] else {
            panic!("First message should be a user message");
        };
        let OpenAIContentBlock::Text { text } = &user_message.content[0] else {
            panic!("First message should be a text message");
        };
        assert_eq!(text, "What is the capital of France?");
        let OpenAIRequestMessage::Assistant(assistant_message) = &row.messages[2] else {
            panic!("Second message should be an assistant message");
        };
        let OpenAIContentBlock::Text { text } =
            assistant_message.content.as_ref().unwrap().first().unwrap()
        else {
            panic!("Second message should be a text message");
        };
        assert_eq!(text, "The capital of France is Paris.");
        assert!(!row.parallel_tool_calls);
        assert_eq!(row.tools.len(), 0);
    }

    #[test]
    fn test_convert_to_optimizer_status() {
        // Test for "succeeded" status with a model output
        let succeeded_model = json!({
            "id": "ftjob-123",
            "status": "succeeded",
            "fine_tuned_model": "ft:gpt-3.5-turbo:my-org:custom_suffix:id",
            "created_at": 1620000000,
            "metadata": {},
        });
        let job = serde_json::from_value::<OpenAIFineTuningJob>(succeeded_model).unwrap();
        let status = convert_to_optimizer_status(job, &OpenAICredentials::None).unwrap();
        assert!(matches!(
            status,
            OptimizerStatus::Completed(OptimizerOutput::Model(_))
        ));

        // Test for "succeeded" status with a file output
        let succeeded_file = json!({
            "id": "ftjob-456",
            "status": "succeeded",
            "result_files": ["file-abc"],
            "fine_tuned_model": "ft:gpt-3.5-turbo:my-org:custom_suffix:id",
            "created_at": 1620000000,
            "metadata": {},
        });
        let job = serde_json::from_value::<OpenAIFineTuningJob>(succeeded_file).unwrap();
        let status = convert_to_optimizer_status(job, &OpenAICredentials::None).unwrap();
        assert!(matches!(
            status,
            OptimizerStatus::Completed(OptimizerOutput::Model(_))
        ));

        // Test for "running" status
        let running = json!({
            "id": "ftjob-789",
            "status": "running",
            "created_at": 1620000000,
            "metadata": {},
        });
        let job = serde_json::from_value::<OpenAIFineTuningJob>(running).unwrap();
        let status = convert_to_optimizer_status(job, &OpenAICredentials::None).unwrap();
        assert!(matches!(status, OptimizerStatus::Pending { .. }));

        // Test for "failed" status
        let failed = json!({
            "id": "ftjob-abc",
            "status": "failed",
            "created_at": 1620000000,
            "metadata": {},
        });
        let job = serde_json::from_value::<OpenAIFineTuningJob>(failed).unwrap();
        let status = convert_to_optimizer_status(job, &OpenAICredentials::None).unwrap();
        assert!(matches!(status, OptimizerStatus::Failed));

        // Test for "validating_files" status
        let validating = json!({
            "id": "ftjob-def",
            "status": "validating_files",
            "created_at": 1620000000,
            "metadata": {},
        });
        let job = serde_json::from_value::<OpenAIFineTuningJob>(validating).unwrap();
        let status = convert_to_optimizer_status(job, &OpenAICredentials::None).unwrap();
        assert!(matches!(status, OptimizerStatus::Pending { .. }));

        // Test for "queued" status
        let queued = json!({
            "id": "ftjob-ghi",
            "status": "queued",
            "created_at": 1620000000,
            "metadata": {},
        });
        let job = serde_json::from_value::<OpenAIFineTuningJob>(queued).unwrap();
        let status = convert_to_optimizer_status(job, &OpenAICredentials::None).unwrap();
        assert!(matches!(status, OptimizerStatus::Pending { .. }));

        // Test for unknown status - this should result in an error from convert_to_optimizer_status
        // as OpenAIFineTuningJobStatus deserialization would fail first if it's truly unknown.
        // If the status is known to OpenAIFineTuningJobStatus but not handled in convert_to_optimizer_status,
        // that would be a different kind of error (panic or unhandled match arm).
        // For this test, let's assume an OpenAIFineTuningJobStatus that convert_to_optimizer_status might not expect.
        // However, the current convert_to_optimizer_status covers all variants of OpenAIFineTuningJobStatus.
        // So, a truly "unknown" status to OpenAIFineTuningJobStatus would fail deserialization.
        // Let's test a valid OpenAIFineTuningJobStatus that might lead to an error in conversion logic,
        // e.g. "succeeded" but missing fine_tuned_model.
        let succeeded_missing_model = json!({
            "id": "ftjob-jkl",
            "status": "succeeded",
            // "fine_tuned_model": null, // This would be an issue
            "created_at": 1620000000,
            "metadata": {},
        });
        let job = serde_json::from_value::<OpenAIFineTuningJob>(succeeded_missing_model).unwrap();
        let result = convert_to_optimizer_status(job, &OpenAICredentials::None);
        assert!(result.is_err());

        // Test for missing status field - this would fail deserialization of OpenAIFineTuningJob
        let missing_status = json!({
            "id": "ftjob-mno",
            // "status": "running", // Status is missing
            "created_at": 1620000000,
            "metadata": {},
        });
        assert!(serde_json::from_value::<OpenAIFineTuningJob>(missing_status).is_err());
    }
}
