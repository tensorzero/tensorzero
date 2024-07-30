use reqwest::Client;
use serde::Deserialize;
use std::time::{SystemTime, UNIX_EPOCH};
use std::{collections::HashMap, path::PathBuf};
use uuid::Uuid;

use crate::error::Error;
use crate::inference::types::{
    AssistantInferenceRequestMessage, ChatInferenceResponse, FunctionType, InferenceRequestMessage,
    InputMessageRole, ModelInferenceRequest, ModelInferenceResponseChunk,
    SystemInferenceRequestMessage, UserInferenceRequestMessage,
};
use crate::jsonschema_util::JSONSchemaFromPath;
use crate::minijinja_util::template_message;
use crate::{
    inference::types::{InferenceResponse, InferenceResponseStream, InputMessage},
    model::ModelConfig,
};

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
#[serde(deny_unknown_fields)]
pub enum VariantConfig {
    ChatCompletion(ChatCompletionConfig),
}

pub trait Variant {
    async fn infer(
        &self,
        messages: &[InputMessage],
        models: &HashMap<String, ModelConfig>,
        output_schema: Option<&JSONSchemaFromPath>,
        client: &Client,
    ) -> Result<InferenceResponse, Error>;

    async fn infer_stream(
        &self,
        messages: &[InputMessage],
        models: &HashMap<String, ModelConfig>,
        output_schema: Option<&JSONSchemaFromPath>,
        client: &Client,
    ) -> Result<(ModelInferenceResponseChunk, InferenceResponseStream), Error>;
}

impl VariantConfig {
    pub fn weight(&self) -> f64 {
        match self {
            VariantConfig::ChatCompletion(params) => params.weight,
        }
    }
    pub fn system_template(&self) -> Option<&PathBuf> {
        match self {
            VariantConfig::ChatCompletion(params) => params.system_template.as_ref(),
        }
    }

    pub fn user_template(&self) -> Option<&PathBuf> {
        match self {
            VariantConfig::ChatCompletion(params) => params.user_template.as_ref(),
        }
    }

    pub fn assistant_template(&self) -> Option<&PathBuf> {
        match self {
            VariantConfig::ChatCompletion(params) => params.assistant_template.as_ref(),
        }
    }
}

impl Variant for VariantConfig {
    async fn infer(
        &self,
        messages: &[InputMessage],
        models: &HashMap<String, ModelConfig>,
        output_schema: Option<&JSONSchemaFromPath>,
        client: &Client,
    ) -> Result<InferenceResponse, Error> {
        match self {
            VariantConfig::ChatCompletion(params) => {
                params.infer(messages, models, output_schema, client).await
            }
        }
    }

    async fn infer_stream(
        &self,
        messages: &[InputMessage],
        models: &HashMap<String, ModelConfig>,
        output_schema: Option<&JSONSchemaFromPath>,
        client: &Client,
    ) -> Result<(ModelInferenceResponseChunk, InferenceResponseStream), Error> {
        match self {
            VariantConfig::ChatCompletion(params) => {
                params
                    .infer_stream(messages, models, output_schema, client)
                    .await
            }
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatCompletionConfig {
    pub weight: f64,
    pub model: String, // TODO: validate that the model is valid given the rest of the config
    pub system_template: Option<PathBuf>,
    pub user_template: Option<PathBuf>,
    pub assistant_template: Option<PathBuf>,
}

impl ChatCompletionConfig {
    fn prepare_request_message(
        &self,
        message: &InputMessage,
    ) -> Result<InferenceRequestMessage, Error> {
        let template_path = match message.role {
            InputMessageRole::System => self.system_template.as_ref(),
            InputMessageRole::User => self.user_template.as_ref(),
            InputMessageRole::Assistant => self.assistant_template.as_ref(),
        };
        let content = match template_path {
            Some(template_path) => template_message(
                template_path.to_str().ok_or(Error::InvalidTemplatePath)?,
                &message.content,
            )?,
            None => {
                // If there is no template, we assume the `content` is a raw string
                message.content.as_str().ok_or(Error::InvalidMessage{ message: format!("Request message content {} is not a string but there is no variant template for Role {}", message.content, message.role) })?.to_string()
            }
        };
        Ok(match message.role {
            InputMessageRole::System => {
                InferenceRequestMessage::System(SystemInferenceRequestMessage { content })
            }
            InputMessageRole::User => {
                InferenceRequestMessage::User(UserInferenceRequestMessage { content })
            }
            InputMessageRole::Assistant => {
                InferenceRequestMessage::Assistant(AssistantInferenceRequestMessage {
                    content: Some(content),
                    tool_calls: None,
                })
            }
        })
    }
}

impl Variant for ChatCompletionConfig {
    async fn infer(
        &self,
        messages: &[InputMessage],
        models: &HashMap<String, ModelConfig>,
        output_schema: Option<&JSONSchemaFromPath>,
        client: &Client,
    ) -> Result<InferenceResponse, Error> {
        let messages = messages
            .iter()
            .map(|message| self.prepare_request_message(message))
            .collect::<Result<Vec<_>, _>>()?;
        let request = ModelInferenceRequest {
            messages,
            tools_available: None,
            tool_choice: None,
            parallel_tool_calls: None,
            temperature: None,
            max_tokens: None,
            stream: false,
            json_mode: false,
            function_type: FunctionType::Chat,
            output_schema: output_schema.as_ref().map(|s| s.value),
        };
        let model_config = models.get(&self.model).ok_or(Error::ModelNotFound {
            model: self.model.clone(),
        })?;
        let model_inference_response = model_config.infer(&request, client).await?;

        let inference_id = Uuid::now_v7();
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_secs();
        let raw_content = model_inference_response.content.clone();
        let tool_calls = model_inference_response.tool_calls.clone();
        let usage = model_inference_response.usage.clone();
        let model_inference_responses = vec![model_inference_response];

        Ok(InferenceResponse::Chat(ChatInferenceResponse::new(
            inference_id,
            created,
            raw_content,
            tool_calls,
            usage,
            model_inference_responses,
            output_schema,
        )))
    }

    async fn infer_stream(
        &self,
        messages: &[InputMessage],
        models: &HashMap<String, ModelConfig>,
        output_schema: Option<&JSONSchemaFromPath>,
        client: &Client,
    ) -> Result<(ModelInferenceResponseChunk, InferenceResponseStream), Error> {
        let messages = messages
            .iter()
            .map(|message| self.prepare_request_message(message))
            .collect::<Result<Vec<_>, _>>()?;
        let request = ModelInferenceRequest {
            messages,
            tools_available: None,
            tool_choice: None,
            parallel_tool_calls: None,
            temperature: None,
            max_tokens: None,
            stream: true,
            json_mode: false,
            function_type: FunctionType::Chat,
            output_schema: output_schema.as_ref().map(|s| s.value),
        };
        let model_config = models.get(&self.model).ok_or(Error::ModelNotFound {
            model: self.model.clone(),
        })?;
        model_config.infer_stream(&request, client).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{error::Error, minijinja_util::idempotent_initialize_test_templates};
    use serde_json::{json, Value};

    #[test]
    fn test_prepare_request_message() {
        // Part 1: test without templates
        let chat_completion_config = ChatCompletionConfig {
            model: "dummy".to_string(),
            weight: 1.0,
            system_template: None,
            user_template: None,
            assistant_template: None,
        };

        // Test case 1: Regular user message
        let input_message = InputMessage {
            role: InputMessageRole::User,
            content: Value::String("Hello, how are you?".to_string()),
        };
        let result = chat_completion_config.prepare_request_message(&input_message);
        assert!(result.is_ok());
        let prepared_message = result.unwrap();
        match prepared_message {
            InferenceRequestMessage::User(user_message) => {
                assert_eq!(user_message.content, "Hello, how are you?");
            }
            _ => panic!("Expected User message"),
        }

        // Test case 2: System message
        let input_message = InputMessage {
            role: InputMessageRole::System,
            content: Value::String("You are a helpful assistant.".to_string()),
        };
        let result = chat_completion_config.prepare_request_message(&input_message);
        assert!(result.is_ok());
        let prepared_message = result.unwrap();
        match prepared_message {
            InferenceRequestMessage::System(system_message) => {
                assert_eq!(system_message.content, "You are a helpful assistant.");
            }
            _ => panic!("Expected System message"),
        }

        // Test case 3: Assistant message
        let input_message = InputMessage {
            role: InputMessageRole::Assistant,
            content: Value::String("I'm doing well, thank you!".to_string()),
        };
        let result = chat_completion_config.prepare_request_message(&input_message);
        assert!(result.is_ok());
        let prepared_message = result.unwrap();
        match prepared_message {
            InferenceRequestMessage::Assistant(assistant_message) => {
                assert_eq!(
                    assistant_message.content,
                    Some("I'm doing well, thank you!".to_string())
                );
            }
            _ => panic!("Expected Assistant message"),
        }
        // Test case 4: Invalid JSON input
        let input_message = InputMessage {
            role: InputMessageRole::User,
            content: serde_json::json!({"invalid": "json"}),
        };
        let result = chat_completion_config
            .prepare_request_message(&input_message)
            .unwrap_err();
        assert_eq!(result, Error::InvalidMessage { message: "Request message content {\"invalid\":\"json\"} is not a string but there is no variant template for Role \"user\"".to_string()});
        // Part 2: test with templates
        let templates = idempotent_initialize_test_templates();
        let system_template = templates.get("system").unwrap();
        let user_template = templates.get("greeting_with_age").unwrap();
        let assistant_template = templates.get("assistant").unwrap();

        let chat_completion_config = ChatCompletionConfig {
            model: "dummy".to_string(),
            weight: 1.0,
            system_template: Some(system_template.to_path_buf()),
            user_template: Some(user_template.to_path_buf()),
            assistant_template: Some(assistant_template.to_path_buf()),
        };

        // Test case 4: System message with template
        let input_message = InputMessage {
            role: InputMessageRole::System,
            content: serde_json::json!({"assistant_name": "ChatGPT"}),
        };
        let result = chat_completion_config.prepare_request_message(&input_message);
        assert!(result.is_ok());
        let prepared_message = result.unwrap();
        match prepared_message {
            InferenceRequestMessage::System(system_message) => {
                assert_eq!(
                    system_message.content,
                    "You are a helpful and friendly assistant namedd ChatGPT"
                );
            }
            _ => panic!("Expected System message"),
        }

        // Test case 5: Assistant message with template
        let input_message = InputMessage {
            role: InputMessageRole::Assistant,
            content: serde_json::json!({"reason": "it's against my ethical guidelines"}),
        };
        let result = chat_completion_config.prepare_request_message(&input_message);
        assert!(result.is_ok());
        let prepared_message = result.unwrap();
        match prepared_message {
            InferenceRequestMessage::Assistant(assistant_message) => {
                assert_eq!(
                    assistant_message.content,
                    Some("I'm sorry but I can't help you with that because of it's against my ethical guidelines".to_string())
                );
            }
            _ => panic!("Expected Assistant message"),
        }

        // Test case 6: User message with template
        let input_message = InputMessage {
            role: InputMessageRole::User,
            content: json!({"name": "John", "age": 30}),
        };
        let result = chat_completion_config.prepare_request_message(&input_message);
        assert!(result.is_ok());
        let prepared_message = result.unwrap();
        match prepared_message {
            InferenceRequestMessage::User(user_message) => {
                assert_eq!(user_message.content, "Hello, John! You are 30 years old.");
            }
            _ => panic!("Expected User message"),
        }

        // Test case 7: User message with bad input (missing required field)
        let input_message = InputMessage {
            role: InputMessageRole::User,
            content: json!({"name": "Alice"}), // Missing "age" field
        };
        let result = chat_completion_config.prepare_request_message(&input_message);
        assert!(result.is_err());
        match result {
            Err(Error::MiniJinjaTemplateRender { message, .. }) => {
                // assert_eq!(template_name, "greeting_with_age");
                println!("{}", message);
                assert!(message.contains("undefined value"));
            }
            _ => panic!("Expected MiniJinjaTemplateRender error"),
        }
    }
}
