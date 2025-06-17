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
