use reqwest::Client;
use serde::Deserialize;
use serde_json::Value;
use std::{collections::HashMap, path::PathBuf};
use uuid::Uuid;

use crate::error::Error;
use crate::inference::types::{
    ChatInferenceResponse, FunctionType, Input, JSONMode, ModelInferenceRequest,
    ModelInferenceResponseChunk, RequestMessage, Role, ToolChoice,
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
        input: &Input,
        models: &HashMap<String, ModelConfig>,
        output_schema: Option<&JSONSchemaFromPath>,
        client: &Client,
    ) -> Result<InferenceResponse, Error>;

    async fn infer_stream(
        &self,
        input: &Input,
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
        input: &Input,
        models: &HashMap<String, ModelConfig>,
        output_schema: Option<&JSONSchemaFromPath>,
        client: &Client,
    ) -> Result<InferenceResponse, Error> {
        match self {
            VariantConfig::ChatCompletion(params) => {
                params.infer(input, models, output_schema, client).await
            }
        }
    }

    async fn infer_stream(
        &self,
        input: &Input,
        models: &HashMap<String, ModelConfig>,
        output_schema: Option<&JSONSchemaFromPath>,
        client: &Client,
    ) -> Result<(ModelInferenceResponseChunk, InferenceResponseStream), Error> {
        match self {
            VariantConfig::ChatCompletion(params) => {
                params
                    .infer_stream(input, models, output_schema, client)
                    .await
            }
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatCompletionConfig {
    pub weight: f64,
    pub model: String, // TODO (#85): validate that this model exists in the model config
    pub system_template: Option<PathBuf>,
    pub user_template: Option<PathBuf>,
    pub assistant_template: Option<PathBuf>,
}

impl ChatCompletionConfig {
    fn prepare_request_message(&self, message: &InputMessage) -> Result<RequestMessage, Error> {
        let template_path = match message.role {
            Role::User => self.user_template.as_ref(),
            Role::Assistant => self.assistant_template.as_ref(),
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
        Ok(RequestMessage {
            role: message.role,
            content: vec![content.into()],
        })
    }

    fn prepare_system_message(&self, system: &Value) -> Result<String, Error> {
        Ok(match &self.system_template {
            Some(template_path) => template_message(
                template_path.to_str().ok_or(Error::InvalidTemplatePath)?,
                system,
            )?,
            None => system
                .as_str()
                .ok_or(Error::InvalidMessage {
                    message:
                        format!("System message content {} is not a string but there is no variant template", system)
                            .to_string(),
                })?
                .to_string(),
        })
    }
}

impl Variant for ChatCompletionConfig {
    async fn infer(
        &self,
        input: &Input,
        models: &HashMap<String, ModelConfig>,
        output_schema: Option<&JSONSchemaFromPath>,
        client: &Client,
    ) -> Result<InferenceResponse, Error> {
        let messages = input
            .messages
            .iter()
            .map(|message| self.prepare_request_message(message))
            .collect::<Result<Vec<_>, _>>()?;
        let system = input
            .system
            .as_ref()
            .map(|system| self.prepare_system_message(system))
            .transpose()?;
        let output_schema_value = match output_schema {
            // We want this block to throw an error if somehow the jsonschema is missing
            // but return None if the output schema is not provided.
            Some(s) => Some(s.value()?),
            None => None,
        };
        let tool_choice = ToolChoice::None;
        let request = ModelInferenceRequest {
            messages,
            system,
            tools_available: None,
            tool_choice: tool_choice.clone(),
            parallel_tool_calls: None,
            temperature: None,
            max_tokens: None,
            stream: false,
            // TODO (#99): add support for JSON mode
            json_mode: JSONMode::Off,
            function_type: FunctionType::Chat,
            output_schema: output_schema_value,
        };
        let model_config = models.get(&self.model).ok_or(Error::ModelNotFound {
            model: self.model.clone(),
        })?;
        let model_inference_response = model_config.infer(&request, client).await?;

        let inference_id = Uuid::now_v7();

        let raw_content = model_inference_response.content.clone();
        let usage = model_inference_response.usage.clone();
        let model_inference_responses = vec![model_inference_response];
        Ok(InferenceResponse::Chat(ChatInferenceResponse::new(
            inference_id,
            raw_content,
            usage,
            model_inference_responses,
            output_schema,
            tool_choice,
        )))
    }

    async fn infer_stream(
        &self,
        input: &Input,
        models: &HashMap<String, ModelConfig>,
        output_schema: Option<&JSONSchemaFromPath>,
        client: &Client,
    ) -> Result<(ModelInferenceResponseChunk, InferenceResponseStream), Error> {
        let messages = input
            .messages
            .iter()
            .map(|message| self.prepare_request_message(message))
            .collect::<Result<Vec<_>, _>>()?;
        let system = input
            .system
            .as_ref()
            .map(|system| self.prepare_system_message(system))
            .transpose()?;
        let output_schema_value = match output_schema {
            // As above, we want this block to throw an error if somehow the jsonschema is missing
            // but return None if the output schema is not provided
            Some(s) => Some(s.value()?),
            None => None,
        };
        let request = ModelInferenceRequest {
            messages,
            system,
            tools_available: None,
            tool_choice: ToolChoice::None,
            parallel_tool_calls: None,
            temperature: None,
            max_tokens: None,
            stream: true,
            json_mode: JSONMode::Off,
            function_type: FunctionType::Chat,
            output_schema: output_schema_value,
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

    use futures::StreamExt;
    use serde_json::{json, Value};

    use crate::inference::providers::dummy::DummyProvider;
    use crate::inference::types::Usage;
    use crate::minijinja_util::tests::idempotent_initialize_test_templates;
    use crate::model::ProviderConfig;
    use crate::{
        error::Error,
        inference::{
            providers::dummy::{
                DUMMY_INFER_RESPONSE_CONTENT, DUMMY_JSON_RESPONSE_RAW, DUMMY_STREAMING_RESPONSE,
            },
            types::{ContentBlockChunk, Role, TextChunk},
        },
    };

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
            role: Role::User,
            content: Value::String("Hello, how are you?".to_string()),
        };
        let result = chat_completion_config.prepare_request_message(&input_message);
        assert!(result.is_ok());
        let prepared_message = result.unwrap();
        match prepared_message {
            RequestMessage {
                role: Role::User,
                content: user_message,
            } => {
                assert_eq!(user_message, vec!["Hello, how are you?".to_string().into()]);
            }
            _ => unreachable!("Expected User message"),
        }

        // Test case 2: Assistant message
        let input_message = InputMessage {
            role: Role::Assistant,
            content: Value::String("I'm doing well, thank you!".to_string()),
        };
        let result = chat_completion_config.prepare_request_message(&input_message);
        assert!(result.is_ok());
        let prepared_message = result.unwrap();
        match prepared_message {
            RequestMessage {
                role: Role::Assistant,
                content: assistant_message,
            } => {
                assert_eq!(
                    assistant_message,
                    vec!["I'm doing well, thank you!".to_string().into()]
                );
            }
            _ => unreachable!("Expected Assistant message"),
        }
        // Test case 3: Invalid JSON input
        let input_message = InputMessage {
            role: Role::User,
            content: serde_json::json!({"invalid": "json"}),
        };
        let result = chat_completion_config
            .prepare_request_message(&input_message)
            .unwrap_err();
        assert_eq!(result, Error::InvalidMessage { message: "Request message content {\"invalid\":\"json\"} is not a string but there is no variant template for Role user".to_string()});

        // Part 2: test with templates
        idempotent_initialize_test_templates();
        let system_template_name = "system";
        let user_template_name = "greeting_with_age";
        let assistant_template_name = "assistant";

        let chat_completion_config = ChatCompletionConfig {
            model: "dummy".to_string(),
            weight: 1.0,
            system_template: Some(system_template_name.into()),
            user_template: Some(user_template_name.into()),
            assistant_template: Some(assistant_template_name.into()),
        };

        // Test case 4: Assistant message with template
        let input_message = InputMessage {
            role: Role::Assistant,
            content: serde_json::json!({"reason": "it's against my ethical guidelines"}),
        };
        let result = chat_completion_config.prepare_request_message(&input_message);
        assert!(result.is_ok());
        let prepared_message = result.unwrap();
        match prepared_message {
            RequestMessage {
                role: Role::Assistant,
                content: assistant_message,
            } => {
                assert_eq!(
                    assistant_message,
                    vec!["I'm sorry but I can't help you with that because of it's against my ethical guidelines".to_string().into()]
                );
            }
            _ => unreachable!("Expected Assistant message"),
        }

        // Test case 5: User message with template
        let input_message = InputMessage {
            role: Role::User,
            content: json!({"name": "John", "age": 30}),
        };
        let result = chat_completion_config.prepare_request_message(&input_message);
        assert!(result.is_ok());
        let prepared_message = result.unwrap();
        match prepared_message {
            RequestMessage {
                role: Role::User,
                content: user_message,
            } => {
                assert_eq!(
                    user_message,
                    vec!["Hello, John! You are 30 years old.".to_string().into()]
                );
            }
            _ => unreachable!("Expected User message"),
        }

        // Test case 6: User message with bad input (missing required field)
        let input_message = InputMessage {
            role: Role::User,
            content: json!({"name": "Alice"}), // Missing "age" field
        };
        let result = chat_completion_config.prepare_request_message(&input_message);
        assert!(result.is_err());
        match result {
            Err(Error::MiniJinjaTemplateRender { message, .. }) => {
                assert!(message.contains("undefined value"));
            }
            _ => unreachable!("Expected MiniJinjaTemplateRender error"),
        }
        // Test case 7: User message with string content when template is provided
        let input_message = InputMessage {
            role: Role::User,
            content: Value::String("This is a plain string".to_string()),
        };
        let result = chat_completion_config.prepare_request_message(&input_message);
        assert!(result.is_err());
        match result {
            Err(Error::MiniJinjaTemplateRender { message, .. }) => {
                assert!(message.contains("undefined value"), "{}", message);
            }
            _ => unreachable!("Expected MiniJinjaTemplateRender error"),
        }
    }

    #[test]
    fn test_prepare_system_message() {
        // Test without templates
        let chat_completion_config = ChatCompletionConfig {
            model: "dummy".to_string(),
            weight: 1.0,
            system_template: None,
            user_template: None,
            assistant_template: None,
        };
        let input_message = Value::String("You are a helpful assistant.".to_string());
        let result = chat_completion_config.prepare_system_message(&input_message);
        assert!(result.is_ok());
        let prepared_message = result.unwrap();
        assert_eq!(prepared_message, "You are a helpful assistant.".to_string());

        // Test with templates
        idempotent_initialize_test_templates();
        let system_template_name = "system";

        let chat_completion_config = ChatCompletionConfig {
            model: "dummy".to_string(),
            weight: 1.0,
            system_template: Some(system_template_name.into()),
            user_template: None,
            assistant_template: None,
        };

        let input_message = serde_json::json!({"assistant_name": "ChatGPT"});
        let result = chat_completion_config.prepare_system_message(&input_message);
        assert!(result.is_ok());
        let prepared_message = result.unwrap();
        assert_eq!(
            prepared_message,
            "You are a helpful and friendly assistant named ChatGPT".to_string()
        );
    }

    #[tokio::test]
    async fn test_infer_chat_completion() {
        let client = Client::new();
        idempotent_initialize_test_templates();
        let system_template_name = "system";
        let user_template_name = "greeting_with_age";
        let chat_completion_config = ChatCompletionConfig {
            model: "good".to_string(),
            weight: 1.0,
            system_template: Some(system_template_name.into()),
            user_template: Some(user_template_name.into()),
            assistant_template: None,
        };
        let good_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "good".to_string(),
        });
        let error_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "error".to_string(),
        });
        let json_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "json".to_string(),
        });
        let text_model_config = ModelConfig {
            routing: vec!["good".to_string()],
            providers: HashMap::from([("good".to_string(), good_provider_config)]),
        };
        let json_model_config = ModelConfig {
            routing: vec!["json".to_string()],
            providers: HashMap::from([("json".to_string(), json_provider_config)]),
        };
        let error_model_config = ModelConfig {
            routing: vec!["error".to_string()],
            providers: HashMap::from([("error".to_string(), error_provider_config)]),
        };
        // Test case 1: invalid message (String passed when template required)
        let messages = vec![InputMessage {
            role: Role::User,
            content: Value::String("Hello".to_string()),
        }];
        let input = Input {
            system: Some(Value::String("Hello".to_string())),
            messages,
        };
        let result = chat_completion_config
            .infer(&input, &HashMap::new(), None, &client)
            .await
            .unwrap_err();
        match result {
            Error::MiniJinjaTemplateRender { message, .. } => {
                // template_name is a test filename
                assert!(message.contains("undefined value"));
            }
            _ => unreachable!("Expected MiniJinjaTemplateRender error"),
        }

        // Test case 2: invalid model in request
        let messages = vec![InputMessage {
            role: Role::User,
            content: json!({"name": "Luke", "age": 20}),
        }];
        let input = Input {
            system: Some(json!({"assistant_name": "R2-D2"})),
            messages,
        };
        let models = HashMap::from([("invalid_model".to_string(), text_model_config.clone())]);
        let result = chat_completion_config
            .infer(&input, &models, None, &client)
            .await
            .unwrap_err();
        assert!(matches!(result, Error::ModelNotFound { .. }), "{}", result);
        // Test case 3: Model inference fails because of model issues

        let chat_completion_config = ChatCompletionConfig {
            model: "error".to_string(),
            weight: 1.0,
            system_template: Some(system_template_name.into()),
            user_template: Some(user_template_name.into()),
            assistant_template: None,
        };
        let models = HashMap::from([("error".to_string(), error_model_config.clone())]);
        let result = chat_completion_config
            .infer(&input, &models, None, &client)
            .await
            .unwrap_err();
        assert_eq!(
            result,
            Error::ModelProvidersExhausted {
                provider_errors: vec![Error::InferenceClient {
                    message: "Error sending request to Dummy provider.".to_string()
                }]
            }
        );

        // Test case 4: Model inference succeeds
        let chat_completion_config = ChatCompletionConfig {
            model: "good".to_string(),
            weight: 1.0,
            system_template: Some(system_template_name.into()),
            user_template: Some(user_template_name.into()),
            assistant_template: None,
        };
        let models = HashMap::from([("good".to_string(), text_model_config.clone())]);
        let result = chat_completion_config
            .infer(&input, &models, None, &client)
            .await
            .unwrap();
        assert!(matches!(result, InferenceResponse::Chat(_)));
        match result {
            InferenceResponse::Chat(chat_response) => {
                // No output schema means parsed_output is None
                assert_eq!(chat_response.parsed_output, None,);
                assert_eq!(
                    chat_response.content_blocks,
                    vec![DUMMY_INFER_RESPONSE_CONTENT.to_string().into()]
                );
                assert_eq!(
                    chat_response.usage,
                    Usage {
                        prompt_tokens: 10,
                        completion_tokens: 10,
                    }
                );
                assert_eq!(chat_response.model_inference_responses.len(), 1);
                assert_eq!(
                    chat_response.model_inference_responses[0].content,
                    vec![DUMMY_INFER_RESPONSE_CONTENT.to_string().into()]
                );
            }
        }
        // Test case 5: JSON output was supposed to happen but it did not
        let output_schema = serde_json::json!({
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string"
                }
            },
            "required": ["answer"],
            "additionalProperties": false
        });
        let output_schema = JSONSchemaFromPath::from_value(&output_schema);
        let result = chat_completion_config
            .infer(&input, &models, Some(&output_schema), &client)
            .await
            .unwrap();
        assert!(matches!(result, InferenceResponse::Chat(_)));
        match result {
            InferenceResponse::Chat(chat_response) => {
                assert_eq!(
                    chat_response.content_blocks,
                    vec![DUMMY_INFER_RESPONSE_CONTENT.to_string().into()]
                );
                assert_eq!(chat_response.parsed_output, None,);
            }
        }

        // Test case 6: JSON output was supposed to happen and it did
        let models = HashMap::from([("json".to_string(), json_model_config.clone())]);
        let chat_completion_config = ChatCompletionConfig {
            model: "json".to_string(),
            weight: 1.0,
            system_template: Some(system_template_name.into()),
            user_template: Some(user_template_name.into()),
            assistant_template: None,
        };
        let result = chat_completion_config
            .infer(&input, &models, Some(&output_schema), &client)
            .await
            .unwrap();
        assert!(matches!(result, InferenceResponse::Chat(_)));
        match result {
            InferenceResponse::Chat(chat_response) => {
                assert_eq!(
                    chat_response.parsed_output,
                    Some(json!({"answer": "Hello"}))
                );
                assert_eq!(
                    chat_response.content_blocks,
                    vec![DUMMY_JSON_RESPONSE_RAW.to_string().into()]
                );
            }
        }
    }

    #[tokio::test]
    async fn test_infer_chat_completion_stream() {
        let client = Client::new();
        idempotent_initialize_test_templates();
        let system_template_name = "system";
        let user_template_name = "greeting_with_age";
        let good_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "good".to_string(),
        });
        let error_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "error".to_string(),
        });
        let text_model_config = ModelConfig {
            routing: vec!["good".to_string()],
            providers: HashMap::from([("good".to_string(), good_provider_config)]),
        };
        let error_model_config = ModelConfig {
            routing: vec!["error".to_string()],
            providers: HashMap::from([("error".to_string(), error_provider_config)]),
        };
        // Test case 1: Model inference fails because of model issues
        let messages = vec![InputMessage {
            role: Role::User,
            content: json!({"name": "Luke", "age": 20}),
        }];
        let input = Input {
            system: Some(json!({"assistant_name": "R2-D2"})),
            messages,
        };
        let chat_completion_config = ChatCompletionConfig {
            model: "error".to_string(),
            weight: 1.0,
            system_template: Some(system_template_name.into()),
            user_template: Some(user_template_name.into()),
            assistant_template: None,
        };
        let models = HashMap::from([("error".to_string(), error_model_config.clone())]);
        let result = chat_completion_config
            .infer_stream(&input, &models, None, &client)
            .await;
        match result {
            Err(Error::ModelProvidersExhausted {
                provider_errors, ..
            }) => {
                assert_eq!(provider_errors.len(), 1);
                assert!(matches!(provider_errors[0], Error::InferenceClient { .. }));
            }
            _ => unreachable!("Expected ModelProvidersExhausted error"),
        }

        // Test case 2: Model inference succeeds
        let chat_completion_config = ChatCompletionConfig {
            model: "good".to_string(),
            weight: 1.0,
            system_template: Some(system_template_name.into()),
            user_template: Some(user_template_name.into()),
            assistant_template: None,
        };
        let models = HashMap::from([("good".to_string(), text_model_config.clone())]);
        let (first_chunk, mut stream) = chat_completion_config
            .infer_stream(&input, &models, None, &client)
            .await
            .unwrap();
        assert_eq!(
            first_chunk.content,
            vec![ContentBlockChunk::Text(TextChunk {
                text: DUMMY_STREAMING_RESPONSE[0].to_string(),
                id: "0".to_string()
            })]
        );
        let mut i = 1;
        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.unwrap();
            if i == 16 {
                // max length of text, but we have a usage chunk left
                assert_eq!(
                    chunk.usage,
                    Some(Usage {
                        prompt_tokens: 10,
                        completion_tokens: 16
                    })
                );
                break;
            }
            assert_eq!(
                chunk.content,
                vec![ContentBlockChunk::Text(TextChunk {
                    text: DUMMY_STREAMING_RESPONSE[i].to_string(),
                    id: "0".to_string(),
                })]
            );
            i += 1;
        }

        // Since we don't handle streaming JSONs at this level (need to catch the entire response)
        // we don't test it here.
    }
}
