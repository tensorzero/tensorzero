use serde::Deserialize;
use std::{collections::HashMap, path::PathBuf};

use crate::error::Error;
use crate::inference::types::{
    AssistantInferenceRequestMessage, FunctionType, InferenceRequestMessage, InputMessageRole,
    ModelInferenceRequest, SystemInferenceRequestMessage, UserInferenceRequestMessage,
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
        output_schema: &Option<JSONSchemaFromPath>,
    ) -> Result<InferenceResponse, Error>;

    async fn infer_stream(
        &self,
        messages: &[InputMessage],
        models: &HashMap<String, ModelConfig>,
        output_schema: &Option<JSONSchemaFromPath>,
    ) -> Result<InferenceResponseStream, Error>;
}

impl VariantConfig {
    pub fn weight(&self) -> f64 {
        match self {
            VariantConfig::ChatCompletion(params) => params.weight,
        }
    }
}

impl Variant for VariantConfig {
    async fn infer(
        &self,
        messages: &[InputMessage],
        models: &HashMap<String, ModelConfig>,
        output_schema: &Option<JSONSchemaFromPath>,
    ) -> Result<InferenceResponse, Error> {
        match self {
            VariantConfig::ChatCompletion(params) => {
                params.infer(messages, models, output_schema).await
            }
        }
    }

    async fn infer_stream(
        &self,
        messages: &[InputMessage],
        models: &HashMap<String, ModelConfig>,
        output_schema: &Option<JSONSchemaFromPath>,
    ) -> Result<InferenceResponseStream, Error> {
        match self {
            VariantConfig::ChatCompletion(params) => {
                params.infer_stream(messages, models, output_schema).await
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
        output_schema: &Option<JSONSchemaFromPath>,
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
            output_schema: output_schema.as_ref().map(|s| s.value()),
        };
        todo!()
    }

    async fn infer_stream(
        &self,
        messages: &[InputMessage],
        models: &HashMap<String, ModelConfig>,
        output_schema: &Option<JSONSchemaFromPath>,
    ) -> Result<InferenceResponseStream, Error> {
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
            output_schema: output_schema.as_ref().map(|s| s.value()),
        };
        todo!()
    }
}
