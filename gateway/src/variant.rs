use reqwest::Client;
use serde::Deserialize;
use serde_json::Value;
use std::{collections::HashMap, path::PathBuf};
use uuid::Uuid;

use crate::endpoints::inference::InferenceParams;
use crate::error::Error;
use crate::function::FunctionConfig;
use crate::inference::types::{
    ContentBlock, FunctionType, Input, InputMessageContent, JSONMode, ModelInferenceRequest,
    ModelInferenceResponseChunk, RequestMessage, Role,
};
use crate::minijinja_util::TemplateConfig;
use crate::tool::ToolCallConfig;
use crate::{
    inference::types::{InferenceResult, InputMessage, ModelInferenceResponseStream},
    model::ModelConfig,
};

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
#[serde(deny_unknown_fields)]
pub enum VariantConfig {
    ChatCompletion(ChatCompletionConfig),
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatCompletionConfig {
    pub weight: f64,
    pub model: String,
    pub system_template: Option<PathBuf>,
    pub user_template: Option<PathBuf>,
    pub assistant_template: Option<PathBuf>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub seed: Option<u32>,
    #[serde(default)]
    pub json_mode: JsonEnforcement, // Only for JSON functions, not for chat functions
}

/// This type is used to determine how to enforce JSON mode for a given variant.
/// Variants represent JSON mode in a slightly more abstract sense than ModelInferenceRequests, as
/// we support coercing tool calls into JSON mode.
/// This is represented as a tool config in the
#[derive(Debug, Default, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum JsonEnforcement {
    #[default]
    Default,
    Strict,
    ImplicitTool,
    Off,
}

/// Maps to the subset of Config that applies to the current inference request.
/// It doesn't take into account inference-time overrides (e.g. dynamic tools).
pub struct InferenceConfig<'a> {
    pub models: &'a HashMap<String, ModelConfig>,
    pub function: &'a FunctionConfig,
    pub tool_config: Option<&'a ToolCallConfig>,
    pub templates: &'a TemplateConfig<'a>,
}

pub trait Variant {
    async fn infer(
        &self,
        input: &Input,
        inference_config: &InferenceConfig,
        client: &Client,
        inference_params: &mut InferenceParams,
    ) -> Result<InferenceResult, Error>;

    async fn infer_stream(
        &self,
        input: &Input,
        inference_config: &InferenceConfig,
        client: &Client,
        inference_params: &mut InferenceParams,
    ) -> Result<(ModelInferenceResponseChunk, ModelInferenceResponseStream), Error>;
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
        inference_config: &InferenceConfig<'_>,
        client: &Client,
        inference_params: &mut InferenceParams,
    ) -> Result<InferenceResult, Error> {
        match self {
            VariantConfig::ChatCompletion(params) => {
                params
                    .infer(input, inference_config, client, inference_params)
                    .await
            }
        }
    }

    async fn infer_stream(
        &self,
        input: &Input,
        inference_config: &InferenceConfig<'_>,
        client: &Client,
        inference_params: &mut InferenceParams,
    ) -> Result<(ModelInferenceResponseChunk, ModelInferenceResponseStream), Error> {
        match self {
            VariantConfig::ChatCompletion(params) => {
                params
                    .infer_stream(input, inference_config, client, inference_params)
                    .await
            }
        }
    }
}

impl ChatCompletionConfig {
    fn prepare_request_message(
        &self,
        templates: &TemplateConfig,
        message: &InputMessage,
    ) -> Result<RequestMessage, Error> {
        let template_path = match message.role {
            Role::User => self.user_template.as_ref(),
            Role::Assistant => self.assistant_template.as_ref(),
        };
        let mut content = Vec::new();
        for block in message.content.iter() {
            match block {
                InputMessageContent::Text { value: text } => {
                    let text_content= match template_path {
                        Some(template_path) => templates.template_message(
                            template_path.to_str().ok_or(Error::InvalidTemplatePath)?,
                            text,
                        )?,
                        None => text.as_str().ok_or(Error::InvalidMessage { message: format!("Request message content {} is not a string but there is no variant template for Role {}", text, message.role) })?.to_string(),
                    };
                    content.push(text_content.into());
                }
                // The following two clones are probably removable.
                // We will need to implement a ToolCallRef type or something so that we can avoid cloning the ToolCall and ToolResult.
                InputMessageContent::ToolCall(tool_call) => {
                    content.push(ContentBlock::ToolCall(tool_call.clone()));
                }
                InputMessageContent::ToolResult(tool_result) => {
                    content.push(ContentBlock::ToolResult(tool_result.clone()));
                }
            }
        }
        Ok(RequestMessage {
            role: message.role,
            content,
        })
    }

    fn prepare_system_message(
        &self,
        templates: &TemplateConfig,
        system: &Value,
    ) -> Result<String, Error> {
        Ok(match &self.system_template {
            Some(template_path) => templates.template_message(
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

    fn prepare_request<'a>(
        &self,
        input: &Input,
        function: &'a FunctionConfig,
        templates: &TemplateConfig,
        tool_config: Option<&'a ToolCallConfig>,
        stream: bool,
        inference_params: &mut InferenceParams,
    ) -> Result<ModelInferenceRequest<'a>, Error> {
        let messages = input
            .messages
            .iter()
            .map(|message| self.prepare_request_message(templates, message))
            .collect::<Result<Vec<_>, _>>()?;
        let system = input
            .system
            .as_ref()
            .map(|system| self.prepare_system_message(templates, system))
            .transpose()?;
        // NOTE: line below mutates the inference_params
        inference_params
            .chat_completion
            .backfill_with_variant_params(self.temperature, self.max_tokens, self.seed);
        Ok(match function {
            FunctionConfig::Chat(_) => ModelInferenceRequest {
                messages,
                system,
                tool_config,
                temperature: inference_params.chat_completion.temperature,
                max_tokens: inference_params.chat_completion.max_tokens,
                seed: inference_params.chat_completion.seed,
                stream,
                json_mode: JSONMode::Off,
                function_type: FunctionType::Chat,
                output_schema: None,
            },
            FunctionConfig::Json(json_config) => {
                let tool_config = match self.json_mode {
                    JsonEnforcement::ImplicitTool => Some(&json_config.implicit_tool_call_config),
                    _ => None,
                };
                ModelInferenceRequest {
                    messages,
                    system,
                    tool_config,
                    temperature: inference_params.chat_completion.temperature,
                    max_tokens: inference_params.chat_completion.max_tokens,
                    seed: inference_params.chat_completion.seed,
                    stream,
                    json_mode: (&self.json_mode).into(),
                    function_type: FunctionType::Json,
                    output_schema: Some(json_config.output_schema.value),
                }
            }
        })
    }
}

impl Variant for ChatCompletionConfig {
    async fn infer(
        &self,
        input: &Input,
        inference_config: &InferenceConfig<'_>,
        client: &Client,
        inference_params: &mut InferenceParams,
    ) -> Result<InferenceResult, Error> {
        let request = self.prepare_request(
            input,
            inference_config.function,
            inference_config.templates,
            inference_config.tool_config,
            false,
            inference_params,
        )?;
        let model_config = inference_config
            .models
            .get(&self.model)
            .ok_or(Error::UnknownModel {
                name: self.model.clone(),
            })?;
        let model_inference_response = model_config.infer(&request, client).await?;

        let inference_id = Uuid::now_v7();

        let raw_content = model_inference_response.content.clone();
        let usage = model_inference_response.usage.clone();
        let model_inference_responses = vec![model_inference_response];
        inference_config
            .function
            .prepare_response(
                inference_id,
                raw_content,
                usage,
                model_inference_responses,
                inference_config.tool_config,
            )
            .await
    }

    async fn infer_stream(
        &self,
        input: &Input,
        inference_config: &InferenceConfig<'_>,
        client: &Client,
        inference_params: &mut InferenceParams,
    ) -> Result<(ModelInferenceResponseChunk, ModelInferenceResponseStream), Error> {
        let request = self.prepare_request(
            input,
            inference_config.function,
            inference_config.templates,
            inference_config.tool_config,
            true,
            inference_params,
        )?;
        let model_config = inference_config
            .models
            .get(&self.model)
            .ok_or(Error::UnknownModel {
                name: self.model.clone(),
            })?;
        model_config.infer_stream(&request, client).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use futures::StreamExt;
    use serde_json::{json, Value};

    use crate::endpoints::inference::ChatCompletionInferenceParams;
    use crate::function::{FunctionConfigChat, FunctionConfigJson};
    use crate::inference::providers::common::get_weather_tool_config;
    use crate::inference::providers::dummy::{DummyProvider, DUMMY_JSON_RESPONSE_RAW};
    use crate::inference::types::{ContentBlockOutput, Usage};
    use crate::jsonschema_util::JSONSchemaFromPath;
    use crate::minijinja_util::tests::get_test_template_config;
    use crate::model::ProviderConfig;
    use crate::tool::ToolChoice;
    use crate::{
        error::Error,
        inference::{
            providers::dummy::{DUMMY_INFER_RESPONSE_CONTENT, DUMMY_STREAMING_RESPONSE},
            types::{ContentBlockChunk, Role, TextChunk},
        },
    };

    #[test]
    fn test_prepare_request_message() {
        let templates = get_test_template_config();
        // Part 1: test without templates
        let chat_completion_config = ChatCompletionConfig {
            model: "dummy".to_string(),
            weight: 1.0,
            system_template: None,
            user_template: None,
            assistant_template: None,
            json_mode: JsonEnforcement::Default,
            temperature: None,
            max_tokens: None,
            seed: None,
        };

        // Test case 1: Regular user message
        let input_message = InputMessage {
            role: Role::User,
            content: vec!["Hello, how are you?".to_string().into()],
        };
        let result = chat_completion_config.prepare_request_message(&templates, &input_message);
        assert!(result.is_ok());
        let prepared_message = result.unwrap();
        match prepared_message {
            RequestMessage {
                role: Role::User,
                content: user_message,
            } => {
                assert_eq!(user_message, vec!["Hello, how are you?".to_string().into()]);
            }
            _ => panic!("Expected User message"),
        }

        // Test case 2: Assistant message
        let input_message = InputMessage {
            role: Role::Assistant,
            content: vec!["I'm doing well, thank you!".to_string().into()],
        };
        let result = chat_completion_config.prepare_request_message(&templates, &input_message);
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
            _ => panic!("Expected Assistant message"),
        }
        // Test case 3: Invalid JSON input
        let input_message = InputMessage {
            role: Role::User,
            content: vec![json!({"invalid": "json"}).into()],
        };
        let result = chat_completion_config
            .prepare_request_message(&templates, &input_message)
            .unwrap_err();
        assert_eq!(result, Error::InvalidMessage { message: "Request message content {\"invalid\":\"json\"} is not a string but there is no variant template for Role user".to_string()});

        // Part 2: test with templates
        let system_template_name = "system";
        let user_template_name = "greeting_with_age";
        let assistant_template_name = "assistant";

        let chat_completion_config = ChatCompletionConfig {
            model: "dummy".to_string(),
            weight: 1.0,
            system_template: Some(system_template_name.into()),
            user_template: Some(user_template_name.into()),
            assistant_template: Some(assistant_template_name.into()),
            json_mode: JsonEnforcement::Default,
            ..Default::default()
        };

        // Test case 4: Assistant message with template
        let input_message = InputMessage {
            role: Role::Assistant,
            content: vec![json!({"reason": "it's against my ethical guidelines"}).into()],
        };
        let result = chat_completion_config.prepare_request_message(&templates, &input_message);
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
            _ => panic!("Expected Assistant message"),
        }

        // Test case 5: User message with template
        let input_message = InputMessage {
            role: Role::User,
            content: vec![json!({"name": "John", "age": 30}).into()],
        };
        let result = chat_completion_config.prepare_request_message(&templates, &input_message);
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
            _ => panic!("Expected User message"),
        }

        // Test case 6: User message with bad input (missing required field)
        let input_message = InputMessage {
            role: Role::User,
            content: vec![json!({"name": "Alice"}).into()], // Missing "age" field
        };
        let result = chat_completion_config.prepare_request_message(&templates, &input_message);
        assert!(result.is_err());
        match result {
            Err(Error::MiniJinjaTemplateRender { message, .. }) => {
                assert!(message.contains("undefined value"));
            }
            _ => panic!("Expected MiniJinjaTemplateRender error"),
        }
        // Test case 7: User message with string content when template is provided
        let input_message = InputMessage {
            role: Role::User,
            content: vec!["This is a plain string".to_string().into()],
        };
        let result = chat_completion_config.prepare_request_message(&templates, &input_message);
        assert!(result.is_err());
        match result {
            Err(Error::MiniJinjaTemplateRender { message, .. }) => {
                assert!(message.contains("undefined value"), "{}", message);
            }
            _ => panic!("Expected MiniJinjaTemplateRender error"),
        }
    }

    #[test]
    fn test_prepare_system_message() {
        let templates = get_test_template_config();

        // Test without templates
        let chat_completion_config = ChatCompletionConfig {
            model: "dummy".to_string(),
            weight: 1.0,
            ..Default::default()
        };
        let input_message = Value::String("You are a helpful assistant.".to_string());
        let result = chat_completion_config.prepare_system_message(&templates, &input_message);
        assert!(result.is_ok());
        let prepared_message = result.unwrap();
        assert_eq!(prepared_message, "You are a helpful assistant.".to_string());

        // Test with templates
        let system_template_name = "system";

        let chat_completion_config = ChatCompletionConfig {
            model: "dummy".to_string(),
            weight: 1.0,
            system_template: Some(system_template_name.into()),
            ..Default::default()
        };

        let input_message = serde_json::json!({"assistant_name": "ChatGPT"});
        let result = chat_completion_config.prepare_system_message(&templates, &input_message);
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
        let templates = get_test_template_config();
        let system_template_name = "system";
        let user_template_name = "greeting_with_age";
        let chat_completion_config = ChatCompletionConfig {
            model: "good".to_string(),
            weight: 1.0,
            system_template: Some(system_template_name.into()),
            user_template: Some(user_template_name.into()),
            ..Default::default()
        };
        let function_config = FunctionConfig::Chat(FunctionConfigChat {
            variants: HashMap::new(),
            system_schema: None,
            user_schema: None,
            assistant_schema: None,
            tools: vec![],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: false,
        });
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
        let tool_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "tool".to_string(),
        });
        let tool_model_config = ModelConfig {
            routing: vec!["tool".to_string()],
            providers: HashMap::from([("tool".to_string(), tool_provider_config)]),
        };
        let error_model_config = ModelConfig {
            routing: vec!["error".to_string()],
            providers: HashMap::from([("error".to_string(), error_provider_config)]),
        };
        // Test case 1: invalid message (String passed when template required)
        let messages = vec![InputMessage {
            role: Role::User,
            content: vec!["Hello".to_string().into()],
        }];
        let input = Input {
            system: Some(Value::String("Hello".to_string())),
            messages,
        };
        let mut inference_params = InferenceParams::default();
        let inference_config = InferenceConfig {
            models: &HashMap::new(),
            function: &function_config,
            templates: &templates,
            tool_config: None,
        };
        let result = chat_completion_config
            .infer(&input, &inference_config, &client, &mut inference_params)
            .await
            .unwrap_err();
        match result {
            Error::MiniJinjaTemplateRender { message, .. } => {
                // template_name is a test filename
                assert!(message.contains("undefined value"));
            }
            _ => panic!("Expected MiniJinjaTemplateRender error"),
        }

        // Test case 2: invalid model in request
        let mut inference_params = InferenceParams::default();
        let messages = vec![InputMessage {
            role: Role::User,
            content: vec![json!({"name": "Luke", "age": 20}).into()],
        }];
        let input = Input {
            system: Some(json!({"assistant_name": "R2-D2"})),
            messages,
        };
        let models = HashMap::from([("invalid_model".to_string(), text_model_config)]);
        let inference_config = InferenceConfig {
            models: &models,
            function: &function_config,
            templates: &templates,
            tool_config: None,
        };
        let result = chat_completion_config
            .infer(&input, &inference_config, &client, &mut inference_params)
            .await
            .unwrap_err();
        assert!(matches!(result, Error::UnknownModel { .. }), "{}", result);
        // Test case 3: Model inference fails because of model issues

        let chat_completion_config = ChatCompletionConfig {
            model: "error".to_string(),
            weight: 1.0,
            system_template: Some(system_template_name.into()),
            user_template: Some(user_template_name.into()),
            ..Default::default()
        };
        let mut inference_params = InferenceParams::default();
        let models = HashMap::from([("error".to_string(), error_model_config)]);
        let inference_config = InferenceConfig {
            models: &models,
            function: &function_config,
            templates: &templates,
            tool_config: None,
        };
        let result = chat_completion_config
            .infer(&input, &inference_config, &client, &mut inference_params)
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
        let mut inference_params = InferenceParams::default();
        let chat_completion_config = ChatCompletionConfig {
            model: "good".to_string(),
            weight: 1.0,
            system_template: Some(system_template_name.into()),
            user_template: Some(user_template_name.into()),
            ..Default::default()
        };
        let good_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "good".to_string(),
        });
        let text_model_config = ModelConfig {
            routing: vec!["good".to_string()],
            providers: HashMap::from([("good".to_string(), good_provider_config)]),
        };
        let models = HashMap::from([("good".to_string(), text_model_config)]);
        let inference_config = InferenceConfig {
            models: &models,
            function: &function_config,
            templates: &templates,
            tool_config: None,
        };
        let result = chat_completion_config
            .infer(&input, &inference_config, &client, &mut inference_params)
            .await
            .unwrap();
        assert!(matches!(result, InferenceResult::Chat(_)));
        match result {
            InferenceResult::Chat(chat_response) => {
                assert_eq!(
                    chat_response.output,
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
            _ => panic!("Expected Chat inference response"),
        }

        // Test case 5: tool call
        let mut inference_params = InferenceParams::default();
        let chat_completion_config = ChatCompletionConfig {
            model: "tool".to_string(),
            weight: 1.0,
            ..Default::default()
        };
        let input = Input {
            system: None,
            messages: vec![InputMessage {
                role: Role::User,
                content: vec!["What is the weather in Brooklyn?".to_string().into()],
            }],
        };
        let models = HashMap::from([("tool".to_string(), tool_model_config)]);
        let weather_tool_config = get_weather_tool_config();
        let inference_config = InferenceConfig {
            models: &models,
            function: &function_config,
            templates: &templates,
            tool_config: Some(&weather_tool_config),
        };
        let result = chat_completion_config
            .infer(&input, &inference_config, &client, &mut inference_params)
            .await
            .unwrap();
        assert!(matches!(result, InferenceResult::Chat(_)));
        match result {
            InferenceResult::Chat(chat_response) => {
                assert_eq!(chat_response.output.len(), 1);
                let tool_call = &chat_response.output[0];
                match tool_call {
                    ContentBlockOutput::ToolCall(tool_call) => {
                        assert_eq!(tool_call.name, "get_weather");
                        assert_eq!(
                            tool_call.arguments,
                            r#"{"location":"Brooklyn","units":"celsius"}"#
                        );
                        assert_eq!(tool_call.parsed_name, Some("get_weather".to_string()));
                        assert_eq!(
                            tool_call.parsed_arguments,
                            Some(json!({"location": "Brooklyn", "units": "celsius"}))
                        );
                    }
                    _ => panic!("Expected tool call"),
                }
                assert_eq!(
                    chat_response.usage,
                    Usage {
                        prompt_tokens: 10,
                        completion_tokens: 10,
                    }
                );
            }
            _ => panic!("Expected Chat inference response"),
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
        let implicit_tool_call_config = ToolCallConfig::implicit_from_value(&output_schema);
        let output_schema = JSONSchemaFromPath::from_value(&output_schema);
        let json_function_config = FunctionConfig::Json(FunctionConfigJson {
            variants: HashMap::new(),
            assistant_schema: None,
            system_schema: None,
            user_schema: None,
            output_schema,
            implicit_tool_call_config,
        });
        let inference_config = InferenceConfig {
            models: &models,
            function: &json_function_config,
            templates: &templates,
            tool_config: None,
        };
        let mut inference_params = InferenceParams::default();
        let result = chat_completion_config
            .infer(&input, &inference_config, &client, &mut inference_params)
            .await
            .unwrap();
        match result {
            InferenceResult::Json(json_result) => {
                assert!(json_result.output.parsed.is_none());
                assert_eq!(
                    json_result.output.raw,
                    r#"{"location":"Brooklyn","units":"celsius"}"#.to_string()
                );
                assert_eq!(
                    json_result.usage,
                    Usage {
                        prompt_tokens: 10,
                        completion_tokens: 10,
                    }
                );
                assert_eq!(json_result.model_inference_responses.len(), 1);
            }
            _ => panic!("Expected Json inference response"),
        }
        let messages = vec![InputMessage {
            role: Role::User,
            content: vec![json!({"name": "Luke", "age": 20}).into()],
        }];
        let input = Input {
            system: Some(json!({"assistant_name": "R2-D2"})),
            messages,
        };
        // Test case 6: JSON output was supposed to happen and it did
        let mut inference_params = InferenceParams::default();
        let models = HashMap::from([("json".to_string(), json_model_config)]);
        let inference_config = InferenceConfig {
            models: &models,
            function: &json_function_config,
            templates: &templates,
            tool_config: None,
        };
        let chat_completion_config = ChatCompletionConfig {
            model: "json".to_string(),
            weight: 1.0,
            system_template: Some(system_template_name.into()),
            user_template: Some(user_template_name.into()),
            ..Default::default()
        };
        let result = chat_completion_config
            .infer(&input, &inference_config, &client, &mut inference_params)
            .await
            .unwrap();
        match result {
            InferenceResult::Json(json_result) => {
                assert_eq!(json_result.output.parsed, Some(json!({"answer": "Hello"})));
                assert_eq!(json_result.output.raw, DUMMY_JSON_RESPONSE_RAW.to_string());
                assert_eq!(
                    json_result.usage,
                    Usage {
                        prompt_tokens: 10,
                        completion_tokens: 10,
                    }
                );
                assert_eq!(json_result.model_inference_responses.len(), 1);
            }
            _ => panic!("Expected Json inference response"),
        }
    }

    #[tokio::test]
    async fn test_infer_chat_completion_stream() {
        let client = Client::new();
        let templates = get_test_template_config();
        let function_config = FunctionConfig::Chat(FunctionConfigChat {
            variants: HashMap::new(),
            system_schema: None,
            user_schema: None,
            assistant_schema: None,
            tools: vec![],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: false,
        });
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
        let mut inference_params = InferenceParams::default();
        let messages = vec![InputMessage {
            role: Role::User,
            content: vec![json!({"name": "Luke", "age": 20}).into()],
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
            ..Default::default()
        };
        let models = HashMap::from([("error".to_string(), error_model_config)]);
        let inference_config = InferenceConfig {
            models: &models,
            function: &function_config,
            templates: &templates,
            tool_config: None,
        };
        let result = chat_completion_config
            .infer_stream(&input, &inference_config, &client, &mut inference_params)
            .await;
        match result {
            Err(Error::ModelProvidersExhausted {
                provider_errors, ..
            }) => {
                assert_eq!(provider_errors.len(), 1);
                assert!(matches!(provider_errors[0], Error::InferenceClient { .. }));
            }
            _ => panic!("Expected ModelProvidersExhausted error"),
        }

        // Test case 2: Model inference succeeds
        let mut inference_params = InferenceParams::default();
        let chat_completion_config = ChatCompletionConfig {
            model: "good".to_string(),
            weight: 1.0,
            system_template: Some(system_template_name.into()),
            user_template: Some(user_template_name.into()),
            ..Default::default()
        };
        let models = HashMap::from([("good".to_string(), text_model_config)]);
        let inference_config = InferenceConfig {
            models: &models,
            function: &function_config,
            templates: &templates,
            tool_config: None,
        };
        let (first_chunk, mut stream) = chat_completion_config
            .infer_stream(&input, &inference_config, &client, &mut inference_params)
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

    /// Since we test the preparation of messages extensively above, we focus here on testing the request parameter setting.
    #[test]
    fn test_prepare_request_params() {
        // We won't vary these parameters in this test
        let input = Input {
            system: None,
            messages: vec![],
        };
        let templates = get_test_template_config();
        let tool_config = None;
        let stream = false;
        // We will vary temperature, max_tokens, and seed
        let chat_completion_config = ChatCompletionConfig {
            temperature: Some(0.5),
            max_tokens: Some(100),
            seed: Some(69),
            ..Default::default()
        };
        // We will do Chat and Json separately
        let function_config = FunctionConfig::Chat(FunctionConfigChat {
            variants: HashMap::new(),
            system_schema: None,
            user_schema: None,
            assistant_schema: None,
            tools: vec![],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: false,
        });
        let mut inference_params = InferenceParams::default();
        let model_request = chat_completion_config
            .prepare_request(
                &input,
                &function_config,
                &templates,
                tool_config.as_ref(),
                stream,
                &mut inference_params,
            )
            .unwrap();
        assert_eq!(model_request.temperature, Some(0.5));
        assert_eq!(model_request.max_tokens, Some(100));
        assert_eq!(model_request.seed, Some(69));
        assert_eq!(inference_params.chat_completion.temperature, Some(0.5));
        assert_eq!(inference_params.chat_completion.max_tokens, Some(100));
        assert_eq!(inference_params.chat_completion.seed, Some(69));

        let mut inference_params = InferenceParams {
            chat_completion: ChatCompletionInferenceParams {
                temperature: Some(1.),
                max_tokens: Some(200),
                seed: Some(420),
            },
        };
        let model_request = chat_completion_config
            .prepare_request(
                &input,
                &function_config,
                &templates,
                tool_config.as_ref(),
                stream,
                &mut inference_params,
            )
            .unwrap();
        assert_eq!(model_request.temperature, Some(1.));
        assert_eq!(model_request.max_tokens, Some(200));
        assert_eq!(model_request.seed, Some(420));
        assert_eq!(inference_params.chat_completion.temperature, Some(1.));
        assert_eq!(inference_params.chat_completion.max_tokens, Some(200));
        assert_eq!(inference_params.chat_completion.seed, Some(420));
        // We will vary temperature, max_tokens, and seed
        let chat_completion_config = ChatCompletionConfig::default();
        let mut inference_params = InferenceParams {
            chat_completion: ChatCompletionInferenceParams {
                temperature: Some(0.9),
                ..Default::default()
            },
        };
        let model_request = chat_completion_config
            .prepare_request(
                &input,
                &function_config,
                &templates,
                tool_config.as_ref(),
                stream,
                &mut inference_params,
            )
            .unwrap();
        assert_eq!(model_request.temperature, Some(0.9));
        assert_eq!(model_request.max_tokens, None);
        assert_eq!(model_request.seed, None);
        assert_eq!(inference_params.chat_completion.temperature, Some(0.9));
        assert_eq!(inference_params.chat_completion.max_tokens, None);
        assert_eq!(inference_params.chat_completion.seed, None);

        // We will vary temperature, max_tokens, and seed
        let chat_completion_config = ChatCompletionConfig {
            temperature: Some(0.5),
            max_tokens: Some(100),
            seed: Some(69),
            ..Default::default()
        };
        // Do a JSON function
        let output_schema_value = json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string"
                },
                "age": {
                    "type": "integer",
                    "minimum": 0
                },
                "email": {
                    "type": "string",
                    "format": "email"
                }
            },
            "required": ["name", "age"]
        });
        let function_config = FunctionConfig::Json(FunctionConfigJson {
            variants: HashMap::new(),
            system_schema: None,
            user_schema: None,
            assistant_schema: None,
            output_schema: JSONSchemaFromPath::from_value(&output_schema_value),
            implicit_tool_call_config: ToolCallConfig {
                tools_available: vec![],
                tool_choice: ToolChoice::Auto,
                parallel_tool_calls: false,
            },
        });
        let mut inference_params = InferenceParams::default();
        let model_request = chat_completion_config
            .prepare_request(
                &input,
                &function_config,
                &templates,
                tool_config.as_ref(),
                stream,
                &mut inference_params,
            )
            .unwrap();
        assert_eq!(model_request.temperature, Some(0.5));
        assert_eq!(model_request.max_tokens, Some(100));
        assert_eq!(model_request.seed, Some(69));
        assert_eq!(model_request.json_mode, JSONMode::On);
        assert_eq!(model_request.output_schema, Some(&output_schema_value));
        assert_eq!(inference_params.chat_completion.temperature, Some(0.5));
        assert_eq!(inference_params.chat_completion.max_tokens, Some(100));
        assert_eq!(inference_params.chat_completion.seed, Some(69));

        let mut inference_params = InferenceParams {
            chat_completion: ChatCompletionInferenceParams {
                temperature: Some(1.),
                max_tokens: Some(200),
                seed: Some(420),
            },
        };
        let model_request = chat_completion_config
            .prepare_request(
                &input,
                &function_config,
                &templates,
                tool_config.as_ref(),
                stream,
                &mut inference_params,
            )
            .unwrap();
        assert_eq!(model_request.temperature, Some(1.));
        assert_eq!(model_request.max_tokens, Some(200));
        assert_eq!(model_request.seed, Some(420));
        assert_eq!(model_request.json_mode, JSONMode::On);
        assert_eq!(model_request.output_schema, Some(&output_schema_value));
        assert_eq!(inference_params.chat_completion.temperature, Some(1.));
        assert_eq!(inference_params.chat_completion.max_tokens, Some(200));
        assert_eq!(inference_params.chat_completion.seed, Some(420));
        // We will vary temperature, max_tokens, and seed
        let chat_completion_config = ChatCompletionConfig::default();
        let mut inference_params = InferenceParams {
            chat_completion: ChatCompletionInferenceParams {
                temperature: Some(0.9),
                ..Default::default()
            },
        };
        let model_request = chat_completion_config
            .prepare_request(
                &input,
                &function_config,
                &templates,
                tool_config.as_ref(),
                stream,
                &mut inference_params,
            )
            .unwrap();
        assert_eq!(model_request.temperature, Some(0.9));
        assert_eq!(model_request.max_tokens, None);
        assert_eq!(model_request.seed, None);
        assert_eq!(inference_params.chat_completion.temperature, Some(0.9));
        assert_eq!(inference_params.chat_completion.max_tokens, None);
        assert_eq!(inference_params.chat_completion.seed, None);
    }
}
