use chrono::DateTime;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{borrow::Cow, collections::HashMap};
use tensorzero_derive::TensorZeroDeserialize;

use tensorzero_core::providers::openai::{
    OpenAIFileID, OpenAIRequestMessage, OpenAISFTTool, SystemOrDeveloper, prepare_openai_messages,
    tensorzero_to_openai_assistant_message,
};
use tensorzero_core::{
    config::TimeoutsConfig,
    error::{Error, ErrorDetails},
    inference::types::{ContentBlock, ContentBlockChatOutput},
    model::{UninitializedModelConfig, UninitializedModelProvider, UninitializedProviderConfig},
    optimization::{OptimizationJobInfo, OptimizerOutput},
    providers::openai::{
        OpenAIMessagesConfig, OpenAIRequestToolCall, PROVIDER_TYPE, grader::OpenAIGrader,
    },
    stored_inference::LazyRenderedSample,
    tool::ToolCall,
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

#[derive(Clone, Debug, Serialize, TensorZeroDeserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
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
    pub grader: Box<OpenAIGrader>,
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

#[derive(Debug, Serialize)]
pub struct OpenAISupervisedRow<'a> {
    messages: Vec<OpenAIRequestMessage<'a>>,
    parallel_tool_calls: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<OpenAISFTTool<'a>>,
}

impl<'a> OpenAISupervisedRow<'a> {
    pub async fn from_rendered_sample(inference: &'a LazyRenderedSample) -> Result<Self, Error> {
        let parallel_tool_calls = inference
            .tool_params
            .parallel_tool_calls
            .unwrap_or_default();
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
        let mut messages = prepare_openai_messages(
            inference
                .system_input
                .as_deref()
                .map(|m| SystemOrDeveloper::System(Cow::Borrowed(m))),
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
    #[serde(flatten)]
    output: OpenAIReinforcementOutput<'a>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<OpenAISFTTool<'a>>,
    parallel_tool_calls: bool,
}

impl<'a> OpenAIReinforcementRow<'a> {
    pub async fn from_rendered_sample(inference: &'a LazyRenderedSample) -> Result<Self, Error> {
        let parallel_tool_calls = inference
            .tool_params
            .parallel_tool_calls
            .unwrap_or_default();
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
        let messages = prepare_openai_messages(
            inference
                .system_input
                .as_deref()
                .map(|m| SystemOrDeveloper::Developer(Cow::Borrowed(m))),
            &inference.messages,
            OpenAIMessagesConfig {
                json_mode: None,
                provider_type: PROVIDER_TYPE,
                // For now, this isn't configurable in RFT (we should never need to resolve a file URL here)
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
        let openai_output = output.try_into()?;
        Ok(Self {
            messages,
            output: openai_output,
            tools,
            parallel_tool_calls,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OpenAIReinforcementOutput<'a> {
    /// Combined text from all Text content blocks
    pub reference_text: Option<String>,
    /// All tool calls from the output
    pub reference_tools: Option<Vec<OpenAIRequestToolCall<'a>>>,
}

impl<'a> TryFrom<&'a Vec<ContentBlockChatOutput>> for OpenAIReinforcementOutput<'a> {
    type Error = Error;

    fn try_from(blocks: &'a Vec<ContentBlockChatOutput>) -> Result<Self, Self::Error> {
        if blocks.is_empty() {
            return Err(Error::new(ErrorDetails::InvalidRenderedStoredInference {
                message: "Output content blocks is empty".to_string(),
            }));
        }

        let mut text_parts = Vec::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();

        for block in blocks {
            match block {
                ContentBlockChatOutput::Text(text) => {
                    text_parts.push(text.text.clone());
                }
                ContentBlockChatOutput::ToolCall(inference_response_tool_call) => {
                    // Convert InferenceResponseToolCall to ToolCall using raw values
                    let tool_call: ToolCall = inference_response_tool_call.clone().into_tool_call();
                    tool_calls.push(tool_call);
                }
                ContentBlockChatOutput::Thought(_) => {
                    // Thoughts are not included in OpenAI reinforcement output
                }
                ContentBlockChatOutput::Unknown(_) => {
                    // Unknown blocks are skipped
                }
            }
        }

        let reference_text = if text_parts.is_empty() {
            None
        } else {
            Some(text_parts.join(""))
        };

        // Convert ToolCall references to OpenAIRequestToolCall using the From impl
        let reference_tools = if tool_calls.is_empty() {
            None
        } else {
            Some(tool_calls.into_iter().map(Into::into).collect())
        };

        Ok(Self {
            reference_text,
            reference_tools,
        })
    }
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

pub fn convert_to_optimizer_status(job: OpenAIFineTuningJob) -> Result<OptimizationJobInfo, Error> {
    let estimated_finish = job
        .estimated_finish
        .and_then(|unix_timestamp| DateTime::from_timestamp(unix_timestamp as i64, 0));
    Ok(match job.status {
        OpenAIFineTuningJobStatus::ValidatingFiles => OptimizationJobInfo::Pending {
            message: "Validating files".to_string(),
            estimated_finish,
            trained_tokens: job.trained_tokens,
            error: job.error,
        },
        OpenAIFineTuningJobStatus::Queued => OptimizationJobInfo::Pending {
            message: "Queued".to_string(),
            estimated_finish,
            trained_tokens: job.trained_tokens,
            error: job.error,
        },
        OpenAIFineTuningJobStatus::Running => OptimizationJobInfo::Pending {
            message: "Running".to_string(),
            estimated_finish,
            trained_tokens: job.trained_tokens,
            error: job.error,
        },
        OpenAIFineTuningJobStatus::Succeeded => {
            let model_name = job.fine_tuned_model.ok_or_else(|| {
                Error::new(ErrorDetails::OptimizationResponse {
                    message: "No fine-tuned model name in Succeeded response".to_string(),
                    provider_type: PROVIDER_TYPE.to_string(),
                })
            })?;
            let model_provider = UninitializedModelProvider {
                config: UninitializedProviderConfig::OpenAI {
                    model_name: model_name.clone(),
                    api_base: None,
                    api_key_location: None,
                    api_type: Default::default(),
                    include_encrypted_reasoning: false,
                    provider_tools: Vec::new(),
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
                    skip_relay: None,
                }),
            }
        }
        OpenAIFineTuningJobStatus::Failed => OptimizationJobInfo::Failed {
            message: "Failed".to_string(),
            error: job.error,
        },
        OpenAIFineTuningJobStatus::Cancelled => OptimizationJobInfo::Failed {
            message: "Cancelled".to_string(),
            error: job.error,
        },
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
    use super::*;
    use serde_json::json;

    use tensorzero_core::{
        inference::types::{
            ContentBlockChatOutput, ModelInput, ResolvedContentBlock, ResolvedRequestMessage, Role,
            StoredInput, StoredInputMessage, StoredInputMessageContent, System, Text,
        },
        providers::openai::OpenAIContentBlock,
        stored_inference::{RenderedSample, StoredOutput},
        tool::{DynamicToolParams, InferenceResponseToolCall},
    };

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
            tags: HashMap::new(),
        };
        let lazy_inference = inference.into_lazy_rendered_sample();
        let row = OpenAISupervisedRow::from_rendered_sample(&lazy_inference)
            .await
            .unwrap();
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

    #[tokio::test]
    async fn test_convert_to_rft_row() {
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
            output: Some(vec![ContentBlockChatOutput::Text(Text {
                text: "The capital of France is Paris.".to_string(),
            })]),
            stored_output: Some(StoredOutput::Chat(vec![ContentBlockChatOutput::Text(
                Text {
                    text: "The capital of France is Paris.".to_string(),
                },
            )])),
            episode_id: Some(uuid::Uuid::now_v7()),
            inference_id: Some(uuid::Uuid::now_v7()),
            tool_params: DynamicToolParams::default(),
            output_schema: None,
            dispreferred_outputs: vec![],
            tags: HashMap::new(),
        };
        let lazy_inference = inference.into_lazy_rendered_sample();
        let row = OpenAIReinforcementRow::from_rendered_sample(&lazy_inference)
            .await
            .unwrap();
        assert_eq!(row.messages.len(), 2); // System and User messages (no assistant message added)
        let OpenAIRequestMessage::Developer(system_message) = &row.messages[0] else {
            panic!("First message should be a developer message");
        };
        assert_eq!(
            system_message.content,
            "You are a helpful assistant named Dr. M.M. Patel."
        );
        let OpenAIRequestMessage::User(user_message) = &row.messages[1] else {
            panic!("Second message should be a user message");
        };
        let OpenAIContentBlock::Text { text } = &user_message.content[0] else {
            panic!("User message should be a text message");
        };
        assert_eq!(text, "What is the capital of France?");

        // Check the output structure
        assert_eq!(
            row.output.reference_text,
            Some("The capital of France is Paris.".to_string())
        );
        assert!(row.output.reference_tools.is_none());
        assert!(!row.parallel_tool_calls);
        assert_eq!(row.tools.len(), 0);
    }

    #[tokio::test]
    async fn test_convert_to_rft_row_with_tool_calls() {
        let inference = RenderedSample {
            function_name: "test".to_string(),
            input: ModelInput {
                system: Some("You are a helpful assistant.".to_string()),
                messages: vec![ResolvedRequestMessage {
                    role: Role::User,
                    content: vec![ResolvedContentBlock::Text(Text {
                        text: "What's the weather like?".to_string(),
                    })],
                }],
            },
            stored_input: StoredInput {
                system: Some(System::Text("You are a helpful assistant.".to_string())),
                messages: vec![StoredInputMessage {
                    role: Role::User,
                    content: vec![StoredInputMessageContent::Text(Text {
                        text: "What's the weather like?".to_string(),
                    })],
                }],
            },
            output: Some(vec![
                ContentBlockChatOutput::Text(Text {
                    text: "I'll check the weather for you.".to_string(),
                }),
                ContentBlockChatOutput::ToolCall(InferenceResponseToolCall {
                    id: "call_123".to_string(),
                    name: Some("get_weather".to_string()),
                    raw_name: "get_weather".to_string(),
                    raw_arguments: r#"{"location": "New York"}"#.to_string(),
                    arguments: Some(serde_json::json!({"location": "New York"})),
                }),
            ]),
            stored_output: Some(StoredOutput::Chat(vec![
                ContentBlockChatOutput::Text(Text {
                    text: "I'll check the weather for you.".to_string(),
                }),
                ContentBlockChatOutput::ToolCall(InferenceResponseToolCall {
                    id: "call_123".to_string(),
                    name: Some("get_weather".to_string()),
                    raw_name: "get_weather".to_string(),
                    raw_arguments: r#"{"location": "New York"}"#.to_string(),
                    arguments: Some(serde_json::json!({"location": "New York"})),
                }),
            ])),
            episode_id: Some(uuid::Uuid::now_v7()),
            inference_id: Some(uuid::Uuid::now_v7()),
            tool_params: DynamicToolParams::default(),
            output_schema: None,
            dispreferred_outputs: vec![],
            tags: HashMap::new(),
        };
        let lazy_inference = inference.into_lazy_rendered_sample();
        let row = OpenAIReinforcementRow::from_rendered_sample(&lazy_inference)
            .await
            .unwrap();

        // Check the output structure
        assert_eq!(
            row.output.reference_text,
            Some("I'll check the weather for you.".to_string())
        );
        assert!(row.output.reference_tools.is_some());
        let tool_calls = row.output.reference_tools.unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "call_123");
        assert_eq!(tool_calls[0].function.name, "get_weather");
        assert_eq!(
            tool_calls[0].function.arguments,
            r#"{"location": "New York"}"#
        );
        assert!(!row.parallel_tool_calls);
        assert_eq!(row.tools.len(), 0);
    }

    #[test]
    fn test_convert_to_optimizer_status() {
        // Test for "succeeded" status with a model output
        let succeeded_model = json!({
            "id": "ftjob-123",
            "status": "succeeded",
            "fine_tuned_model": "ft:gpt-4.1-mini:my-org:custom_suffix:id",
            "created_at": 1620000000,
            "metadata": {},
        });
        let job = serde_json::from_value::<OpenAIFineTuningJob>(succeeded_model).unwrap();
        let status = convert_to_optimizer_status(job).unwrap();
        assert!(matches!(
            status,
            OptimizationJobInfo::Completed {
                output: OptimizerOutput::Model(_),
            }
        ));

        // Test for "succeeded" status with a file output
        let succeeded_file = json!({
            "id": "ftjob-456",
            "status": "succeeded",
            "result_files": ["file-abc"],
            "fine_tuned_model": "ft:gpt-4.1-mini:my-org:custom_suffix:id",
            "created_at": 1620000000,
            "metadata": {},
        });
        let job = serde_json::from_value::<OpenAIFineTuningJob>(succeeded_file).unwrap();
        let status = convert_to_optimizer_status(job).unwrap();
        assert!(matches!(
            status,
            OptimizationJobInfo::Completed {
                output: OptimizerOutput::Model(_),
            }
        ));

        // Test for "running" status
        let running = json!({
            "id": "ftjob-789",
            "status": "running",
            "created_at": 1620000000,
            "metadata": {},
        });
        let job = serde_json::from_value::<OpenAIFineTuningJob>(running).unwrap();
        let status = convert_to_optimizer_status(job).unwrap();
        assert!(matches!(status, OptimizationJobInfo::Pending { .. }));

        // Test for "failed" status
        let failed = json!({
            "id": "ftjob-abc",
            "status": "failed",
            "created_at": 1620000000,
            "metadata": {},
        });
        let job = serde_json::from_value::<OpenAIFineTuningJob>(failed).unwrap();
        let status = convert_to_optimizer_status(job).unwrap();
        assert!(matches!(status, OptimizationJobInfo::Failed { .. }));

        // Test for "validating_files" status
        let validating = json!({
            "id": "ftjob-def",
            "status": "validating_files",
            "created_at": 1620000000,
            "metadata": {},
        });
        let job = serde_json::from_value::<OpenAIFineTuningJob>(validating).unwrap();
        let status = convert_to_optimizer_status(job).unwrap();
        assert!(matches!(status, OptimizationJobInfo::Pending { .. }));

        // Test for "queued" status
        let queued = json!({
            "id": "ftjob-ghi",
            "status": "queued",
            "created_at": 1620000000,
            "metadata": {},
        });
        let job = serde_json::from_value::<OpenAIFineTuningJob>(queued).unwrap();
        let status = convert_to_optimizer_status(job).unwrap();
        assert!(matches!(status, OptimizationJobInfo::Pending { .. }));

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
        let result = convert_to_optimizer_status(job);
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
