use futures::StreamExt;
use reqwest::Client;
use serde::Deserialize;
use serde_json::Value;
use std::borrow::Cow;
use std::{collections::HashMap, path::PathBuf};
use uuid::Uuid;

use crate::endpoints::inference::InferenceParams;
use crate::error::Error;
use crate::function::FunctionConfig;
use crate::inference::types::{
    ContentBlock, FunctionType, InferenceResultChunk, InferenceResultStream, Input,
    InputMessageContent, ModelInferenceRequest, ModelInferenceRequestJsonMode,
    ModelInferenceResponseWithMetadata, RequestMessage, Role,
};
use crate::jsonschema_util::JSONSchemaFromPath;
use crate::minijinja_util::TemplateConfig;
use crate::tool::create_dynamic_implicit_tool_config;
use crate::variant::JsonMode;
use crate::{
    inference::types::{InferenceResult, InputMessage},
    model::ModelConfig,
};

use super::{InferenceConfig, ModelUsedInfo, Variant};

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
    pub json_mode: JsonMode, // Only for JSON functions, not for chat functions
}

impl ChatCompletionConfig {
    pub fn prepare_request_message(
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

    pub fn prepare_system_message(
        &self,
        templates: &TemplateConfig,
        system: Option<&Value>,
    ) -> Result<Option<String>, Error> {
        Ok(match &self.system_template {
            Some(template_path) => Some(templates.template_message(
                template_path.to_str().ok_or(Error::InvalidTemplatePath)?,
                system.unwrap_or(&Value::Null),
            )?),
            None => {
                match system {
                    None => None,
                    Some(system) =>
                Some(system
                .as_str()
                .ok_or(Error::InvalidMessage {
                    message:
                        format!("System message content {} is not a string but there is no variant template", system)
                            .to_string(),
                })?
                .to_string()),
            }
        }})
    }

    fn prepare_request<'a>(
        &self,
        input: &Input,
        function: &'a FunctionConfig,
        inference_config: &'a InferenceConfig<'a>,
        stream: bool,
        inference_params: &mut InferenceParams,
    ) -> Result<ModelInferenceRequest<'a>, Error> {
        let messages = input
            .messages
            .iter()
            .map(|message| self.prepare_request_message(inference_config.templates, message))
            .collect::<Result<Vec<_>, _>>()?;
        let system =
            self.prepare_system_message(inference_config.templates, input.system.as_ref())?;

        inference_params
            .chat_completion
            .backfill_with_variant_params(self.temperature, self.max_tokens, self.seed);
        Ok(match function {
            FunctionConfig::Chat(_) => ModelInferenceRequest {
                messages,
                system,
                tool_config: inference_config.tool_config.as_ref().map(Cow::Borrowed),
                temperature: inference_params.chat_completion.temperature,
                max_tokens: inference_params.chat_completion.max_tokens,
                seed: inference_params.chat_completion.seed,
                stream,
                json_mode: ModelInferenceRequestJsonMode::Off,
                function_type: FunctionType::Chat,
                output_schema: None,
            },
            FunctionConfig::Json(json_config) => {
                let tool_config = match self.json_mode {
                    JsonMode::ImplicitTool => match &inference_config.dynamic_output_schema {
                        Some(schema) => Some(Cow::Owned(create_dynamic_implicit_tool_config(
                            schema.value.clone(),
                        ))),
                        None => Some(Cow::Borrowed(&json_config.implicit_tool_call_config)),
                    },
                    _ => None,
                };
                let output_schema = match &inference_config.dynamic_output_schema {
                    Some(schema) => Some(&schema.value),
                    None => Some(json_config.output_schema.value),
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
                    output_schema,
                }
            }
        })
    }
}

impl Variant for ChatCompletionConfig {
    async fn infer<'a, 'request>(
        &'a self,
        input: &Input,
        models: &'a HashMap<String, ModelConfig>,
        function: &'a FunctionConfig,
        inference_config: &'request InferenceConfig<'request>,
        client: &'request Client,
        inference_params: &mut InferenceParams,
    ) -> Result<InferenceResult<'a>, Error> {
        let request =
            self.prepare_request(input, function, inference_config, false, inference_params)?;
        let model_config = models.get(&self.model).ok_or(Error::UnknownModel {
            name: self.model.clone(),
        })?;
        let model_inference_response = model_config.infer(&request, client).await?;
        let model_inference_result =
            ModelInferenceResponseWithMetadata::new(model_inference_response, &self.model);

        let inference_id = Uuid::now_v7();

        let raw_content = model_inference_result.content.clone();
        let usage = model_inference_result.usage.clone();
        let model_inference_results = vec![model_inference_result];
        function
            .prepare_response(
                inference_id,
                raw_content,
                usage,
                model_inference_results,
                inference_config,
            )
            .await
    }

    async fn infer_stream<'request>(
        &'static self,
        input: &Input,
        models: &'static HashMap<String, ModelConfig>,
        function: &'static FunctionConfig,
        inference_config: &'request InferenceConfig<'request>,
        client: &'request Client,
        inference_params: &mut InferenceParams,
    ) -> Result<
        (
            InferenceResultChunk,
            InferenceResultStream,
            ModelUsedInfo<'static>,
        ),
        Error,
    > {
        let request =
            self.prepare_request(input, function, inference_config, true, inference_params)?;
        let model_config = models.get(&self.model).ok_or(Error::UnknownModel {
            name: self.model.clone(),
        })?;
        let (first_chunk, stream, raw_request, model_provider_name) =
            model_config.infer_stream(&request, client).await?;
        let model_used_info = ModelUsedInfo {
            model_name: &self.model,
            model_provider_name,
            raw_request,
        };
        let first_chunk = InferenceResultChunk::new(first_chunk, function);
        let stream =
            stream.map(move |chunk| chunk.map(|chunk| InferenceResultChunk::new(chunk, function)));
        Ok((first_chunk, Box::pin(stream), model_used_info))
    }

    /// This function validates that the chat completion variant is correctly configured for the given function config.
    /// In order to do this, we need to check:
    ///  - For system, user, and assistant message types:
    ///    - That the template here is provided if the schema is provided in the function.
    ///    - If the template requires variables, the schema is provided.
    ///  - That the model name is a valid model
    ///  - That the weight is non-negative
    fn validate(
        &self,
        function: &FunctionConfig,
        models: &HashMap<String, ModelConfig>,
        templates: &TemplateConfig,
        function_name: &str,
        variant_name: &str,
    ) -> Result<(), Error> {
        // Validate that weight is non-negative
        if self.weight < 0.0 {
            return Err(Error::Config {
                message: format!(
                    "`functions.{function_name}.variants.{variant_name}`: `weight` must be non-negative"
                ),
            });
        }
        let model = models.get(&self.model).ok_or_else(|| Error::Config {
            message: format!("`functions.{function_name}.variants.{variant_name}`: `model` must be a valid model name"),
        })?;

        // If the variant has weight > 0.0, then we need to validate that the model is correctly configured
        if self.weight > 0.0 {
            model.validate().map_err(|e| Error::Config {
                message: format!(
                    "`functions.{function_name}.variants.{variant_name}` and model `{}`: {e}",
                    self.model
                ),
            })?;
        }

        // Validate the system template matches the system schema (best effort, we cannot check the variables comprehensively)
        validate_template_and_schema(
            function.system_schema(),
            self.system_template.as_ref(),
            templates,
        )
        .map_err(|e| Error::Config {
            message: format!(
                "`functions.{function_name}.variants.{variant_name}.system_template`: {e}"
            ),
        })?;

        // Validate the user template matches the user schema (best effort, we cannot check the variables comprehensively)
        validate_template_and_schema(
            function.user_schema(),
            self.user_template.as_ref(),
            templates,
        )
        .map_err(|e| Error::Config {
            message: format!(
                "`functions.{function_name}.variants.{variant_name}.user_template`: {e}"
            ),
        })?;

        // Validate the assistant template matches the assistant schema (best effort, we cannot check the variables comprehensively)
        validate_template_and_schema(
            function.assistant_schema(),
            self.assistant_template.as_ref(),
            templates,
        )
        .map_err(|e| Error::Config {
            message: format!(
                "`functions.{function_name}.variants.{variant_name}.assistant_template`: {e}"
            ),
        })?;
        Ok(())
    }

    fn get_all_template_paths(&self) -> Vec<&PathBuf> {
        let mut templates = Vec::new();
        if let Some(system_template) = &self.system_template {
            templates.push(system_template);
        }
        if let Some(user_template) = &self.user_template {
            templates.push(user_template);
        }
        if let Some(assistant_template) = &self.assistant_template {
            templates.push(assistant_template);
        }
        templates
    }
}

pub fn validate_template_and_schema(
    schema: Option<&JSONSchemaFromPath>,
    template: Option<&PathBuf>,
    templates: &TemplateConfig,
) -> Result<(), Error> {
    match (schema, template) {
        (None, Some(template)) => {
            let template_name = template.to_str().ok_or(Error::InvalidTemplatePath)?;
            if templates.template_needs_variables(template_name)? {
                return Err(Error::Config {
                    message: "schema is required when template is specified and needs variables"
                        .to_string(),
                });
            }
        }
        (Some(_), None) => {
            return Err(Error::Config {
                message: "template is required when schema is specified".to_string(),
            });
        }
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use futures::StreamExt;
    use serde_json::{json, Value};

    use crate::endpoints::inference::ChatCompletionInferenceParams;
    use crate::function::{FunctionConfigChat, FunctionConfigJson};
    use crate::inference::providers::common::get_temperature_tool_config;
    use crate::inference::providers::dummy::{DummyProvider, DUMMY_JSON_RESPONSE_RAW};
    use crate::inference::types::{ContentBlockOutput, Usage};
    use crate::jsonschema_util::{DynamicJSONSchema, JSONSchemaFromPath};
    use crate::minijinja_util::tests::get_test_template_config;
    use crate::model::ProviderConfig;
    use crate::tool::{ToolCallConfig, ToolChoice};
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
            json_mode: JsonMode::On,
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
            json_mode: JsonMode::On,
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
        // Part 3: test with filled out templates
        let system_template_name = "system";
        let user_template_name = "user_filled";
        let assistant_template_name = "assistant_filled";

        let chat_completion_config = ChatCompletionConfig {
            model: "dummy".to_string(),
            weight: 1.0,
            system_template: Some(system_template_name.into()),
            user_template: Some(user_template_name.into()),
            assistant_template: Some(assistant_template_name.into()),
            json_mode: JsonMode::On,
            ..Default::default()
        };

        // Test case 8: assistant message with null input and filled out template
        let input_message = InputMessage {
            role: Role::Assistant,
            content: vec![Value::Null.into()],
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

        // Test case 9: User message with null input and filled out template
        let input_message = InputMessage {
            role: Role::User,
            content: vec![Value::Null.into()],
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
                    vec!["What's the capital of Japan?".to_string().into()]
                );
            }
            _ => panic!("Expected User message"),
        }
    }

    #[test]
    fn test_prepare_system_message() {
        let templates = get_test_template_config();

        // Test without templates, string message
        let chat_completion_config = ChatCompletionConfig {
            model: "dummy".to_string(),
            weight: 1.0,
            ..Default::default()
        };
        let input_message = Value::String("You are a helpful assistant.".to_string());
        let result =
            chat_completion_config.prepare_system_message(&templates, Some(&input_message));
        assert!(result.is_ok());
        let prepared_message = result.unwrap();
        assert_eq!(
            prepared_message,
            Some("You are a helpful assistant.".to_string())
        );

        // Test without templates, object message
        let chat_completion_config = ChatCompletionConfig {
            model: "dummy".to_string(),
            weight: 1.0,
            ..Default::default()
        };
        let input_message = json!({"message": "You are a helpful assistant."});
        let result =
            chat_completion_config.prepare_system_message(&templates, Some(&input_message));
        assert!(result.is_err());
        let prepared_message = result.unwrap_err();
        assert_eq!(
            prepared_message,
            Error::InvalidMessage { message: "System message content {\"message\":\"You are a helpful assistant.\"} is not a string but there is no variant template".to_string() }
        );

        // Test without templates, no message
        let chat_completion_config = ChatCompletionConfig {
            model: "dummy".to_string(),
            weight: 1.0,
            ..Default::default()
        };
        let result = chat_completion_config.prepare_system_message(&templates, None);
        assert!(result.is_ok());
        let prepared_message = result.unwrap();
        assert_eq!(prepared_message, None);

        // Test with templates that need new info
        let system_template_name = "system";

        let chat_completion_config = ChatCompletionConfig {
            model: "dummy".to_string(),
            weight: 1.0,
            system_template: Some(system_template_name.into()),
            ..Default::default()
        };

        let input_message = serde_json::json!({"assistant_name": "ChatGPT"});
        let result =
            chat_completion_config.prepare_system_message(&templates, Some(&input_message));
        assert!(result.is_ok());
        let prepared_message = result.unwrap();
        assert_eq!(
            prepared_message,
            Some("You are a helpful and friendly assistant named ChatGPT".to_string())
        );

        // Test with template that is complete as is (string)
        let system_template_name = "system_filled";

        let chat_completion_config = ChatCompletionConfig {
            model: "dummy".to_string(),
            weight: 1.0,
            system_template: Some(system_template_name.into()),
            ..Default::default()
        };

        let result = chat_completion_config.prepare_system_message(&templates, None);
        assert!(result.is_ok());
        let prepared_message = result.unwrap();
        assert_eq!(
            prepared_message,
            Some("You are a helpful and friendly assistant named ChatGPT".to_string())
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
            routing: vec!["json_provider".to_string()],
            providers: HashMap::from([("json_provider".to_string(), json_provider_config)]),
        };
        let tool_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "tool".to_string(),
        });
        let tool_model_config = ModelConfig {
            routing: vec!["tool_provider".to_string()],
            providers: HashMap::from([("tool_provider".to_string(), tool_provider_config)]),
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
            templates: &templates,
            tool_config: None,
            dynamic_output_schema: None,
        };
        let models = HashMap::new();
        let result = chat_completion_config
            .infer(
                &input,
                &models,
                &function_config,
                &inference_config,
                &client,
                &mut inference_params,
            )
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
            templates: &templates,
            tool_config: None,
            dynamic_output_schema: None,
        };
        let result = chat_completion_config
            .infer(
                &input,
                &models,
                &function_config,
                &inference_config,
                &client,
                &mut inference_params,
            )
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
            templates: &templates,
            tool_config: None,
            dynamic_output_schema: None,
        };
        let result = chat_completion_config
            .infer(
                &input,
                &models,
                &function_config,
                &inference_config,
                &client,
                &mut inference_params,
            )
            .await
            .unwrap_err();
        assert_eq!(
            result,
            Error::ModelProvidersExhausted {
                provider_errors: HashMap::from([(
                    "error".to_string(),
                    Error::InferenceClient {
                        message: "Error sending request to Dummy provider.".to_string()
                    }
                )])
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
            routing: vec!["good_provider".to_string()],
            providers: HashMap::from([("good_provider".to_string(), good_provider_config)]),
        };
        let models = HashMap::from([("good".to_string(), text_model_config)]);
        let inference_config = InferenceConfig {
            templates: &templates,
            tool_config: None,
            dynamic_output_schema: None,
        };
        let result = chat_completion_config
            .infer(
                &input,
                &models,
                &function_config,
                &inference_config,
                &client,
                &mut inference_params,
            )
            .await
            .unwrap();
        assert!(matches!(result, InferenceResult::Chat(_)));
        match result {
            InferenceResult::Chat(chat_response) => {
                assert_eq!(
                    chat_response.content,
                    vec![DUMMY_INFER_RESPONSE_CONTENT.to_string().into()]
                );
                assert_eq!(
                    chat_response.usage,
                    Usage {
                        input_tokens: 10,
                        output_tokens: 10,
                    }
                );
                assert_eq!(chat_response.model_inference_results.len(), 1);
                assert_eq!(
                    chat_response.model_inference_results[0].content,
                    vec![DUMMY_INFER_RESPONSE_CONTENT.to_string().into()]
                );
                assert_eq!(
                    chat_response.model_inference_results[0].model_name,
                    "good".to_string()
                );
                assert_eq!(
                    chat_response.model_inference_results[0].model_provider_name,
                    "good_provider".to_string()
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
        let weather_tool_config = get_temperature_tool_config();
        let inference_config = InferenceConfig {
            templates: &templates,
            tool_config: Some(weather_tool_config),
            dynamic_output_schema: None,
        };
        let result = chat_completion_config
            .infer(
                &input,
                &models,
                &function_config,
                &inference_config,
                &client,
                &mut inference_params,
            )
            .await
            .unwrap();
        assert!(matches!(result, InferenceResult::Chat(_)));
        match result {
            InferenceResult::Chat(chat_response) => {
                assert_eq!(chat_response.content.len(), 1);
                let tool_call = &chat_response.content[0];
                match tool_call {
                    ContentBlockOutput::ToolCall(tool_call) => {
                        assert_eq!(tool_call.raw_name, "get_temperature");
                        assert_eq!(
                            tool_call.raw_arguments,
                            r#"{"location":"Brooklyn","units":"celsius"}"#
                        );
                        assert_eq!(tool_call.name, Some("get_temperature".to_string()));
                        assert_eq!(
                            tool_call.arguments,
                            Some(json!({"location": "Brooklyn", "units": "celsius"}))
                        );
                    }
                    _ => panic!("Expected tool call"),
                }
                assert_eq!(
                    chat_response.usage,
                    Usage {
                        input_tokens: 10,
                        output_tokens: 10,
                    }
                );
                assert_eq!(chat_response.model_inference_results.len(), 1);
                assert_eq!(
                    chat_response.model_inference_results[0].model_provider_name,
                    "tool_provider".to_string()
                );
                assert_eq!(
                    chat_response.model_inference_results[0].model_name,
                    "tool".to_string()
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
            templates: &templates,
            tool_config: None,
            dynamic_output_schema: None,
        };
        let mut inference_params = InferenceParams::default();
        let result = chat_completion_config
            .infer(
                &input,
                &models,
                &json_function_config,
                &inference_config,
                &client,
                &mut inference_params,
            )
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
                        input_tokens: 10,
                        output_tokens: 10,
                    }
                );
                assert_eq!(json_result.model_inference_results.len(), 1);
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
            templates: &templates,
            tool_config: None,
            dynamic_output_schema: None,
        };
        let chat_completion_config = ChatCompletionConfig {
            model: "json".to_string(),
            weight: 1.0,
            system_template: Some(system_template_name.into()),
            user_template: Some(user_template_name.into()),
            ..Default::default()
        };
        let result = chat_completion_config
            .infer(
                &input,
                &models,
                &json_function_config,
                &inference_config,
                &client,
                &mut inference_params,
            )
            .await
            .unwrap();
        match result {
            InferenceResult::Json(json_result) => {
                assert_eq!(json_result.output.parsed, Some(json!({"answer": "Hello"})));
                assert_eq!(json_result.output.raw, DUMMY_JSON_RESPONSE_RAW.to_string());
                assert_eq!(
                    json_result.usage,
                    Usage {
                        input_tokens: 10,
                        output_tokens: 10,
                    }
                );
                assert_eq!(json_result.model_inference_results.len(), 1);
                assert_eq!(
                    json_result.model_inference_results[0].model_provider_name,
                    "json_provider".to_string()
                );
                assert_eq!(
                    json_result.model_inference_results[0].model_name,
                    "json".to_string()
                );
            }
            _ => panic!("Expected Json inference response"),
        }
        // Test case 7: Dynamic JSON output happens and works
        let hardcoded_output_schema = serde_json::json!({
            "type": "object",
            "properties": {
                "response": {
                    "type": "string"
                }
            },
            "required": ["response"],
            "additionalProperties": false
        });
        let implicit_tool_call_config =
            ToolCallConfig::implicit_from_value(&hardcoded_output_schema);
        let hardcoded_output_schema = JSONSchemaFromPath::from_value(&hardcoded_output_schema);
        let json_function_config = FunctionConfig::Json(FunctionConfigJson {
            variants: HashMap::new(),
            assistant_schema: None,
            system_schema: None,
            user_schema: None,
            output_schema: hardcoded_output_schema,
            implicit_tool_call_config,
        });
        let mut inference_params = InferenceParams::default();
        // Will dynamically set "answer" instead of "response"
        let output_schema = DynamicJSONSchema::new(serde_json::json!({
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string"
                }
            },
            "required": ["answer"]
        }));
        let inference_config = InferenceConfig {
            templates: &templates,
            tool_config: None,
            dynamic_output_schema: Some(output_schema),
        };
        let chat_completion_config = ChatCompletionConfig {
            model: "json".to_string(),
            weight: 1.0,
            system_template: Some(system_template_name.into()),
            user_template: Some(user_template_name.into()),
            ..Default::default()
        };
        let result = chat_completion_config
            .infer(
                &input,
                &models,
                &json_function_config,
                &inference_config,
                &client,
                &mut inference_params,
            )
            .await
            .unwrap();
        match result {
            InferenceResult::Json(json_result) => {
                assert_eq!(json_result.output.parsed, Some(json!({"answer": "Hello"})));
                assert_eq!(json_result.output.raw, DUMMY_JSON_RESPONSE_RAW.to_string());
                assert_eq!(
                    json_result.usage,
                    Usage {
                        input_tokens: 10,
                        output_tokens: 10,
                    }
                );
                assert_eq!(json_result.model_inference_results.len(), 1);
                assert_eq!(
                    json_result.model_inference_results[0].model_provider_name,
                    "json_provider".to_string()
                );
                assert_eq!(
                    json_result.model_inference_results[0].model_name,
                    "json".to_string()
                );
            }
            _ => panic!("Expected Json inference response"),
        }
        // Test case 8: Dynamic JSON output fails
        let hardcoded_output_schema = serde_json::json!({
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string"
                }
            },
            "required": ["answer"],
            "additionalProperties": false
        });
        let implicit_tool_call_config =
            ToolCallConfig::implicit_from_value(&hardcoded_output_schema);
        let hardcoded_output_schema = JSONSchemaFromPath::from_value(&hardcoded_output_schema);
        let json_function_config = FunctionConfig::Json(FunctionConfigJson {
            variants: HashMap::new(),
            assistant_schema: None,
            system_schema: None,
            user_schema: None,
            output_schema: hardcoded_output_schema,
            implicit_tool_call_config,
        });
        let mut inference_params = InferenceParams::default();
        // Will dynamically set "response" instead of "answer"
        let output_schema = DynamicJSONSchema::new(serde_json::json!({
            "type": "object",
            "properties": {
                "response": {
                    "type": "string"
                }
            },
            "required": ["response"]
        }));
        let inference_config = InferenceConfig {
            templates: &templates,
            tool_config: None,
            dynamic_output_schema: Some(output_schema),
        };
        let chat_completion_config = ChatCompletionConfig {
            model: "json".to_string(),
            weight: 1.0,
            system_template: Some(system_template_name.into()),
            user_template: Some(user_template_name.into()),
            ..Default::default()
        };
        let result = chat_completion_config
            .infer(
                &input,
                &models,
                &json_function_config,
                &inference_config,
                &client,
                &mut inference_params,
            )
            .await
            .unwrap();
        match result {
            InferenceResult::Json(json_result) => {
                assert_eq!(json_result.output.parsed, None);
                assert_eq!(json_result.output.raw, DUMMY_JSON_RESPONSE_RAW.to_string());
                assert_eq!(
                    json_result.usage,
                    Usage {
                        input_tokens: 10,
                        output_tokens: 10,
                    }
                );
                assert_eq!(json_result.model_inference_results.len(), 1);
                assert_eq!(
                    json_result.model_inference_results[0].model_provider_name,
                    "json_provider".to_string()
                );
                assert_eq!(
                    json_result.model_inference_results[0].model_name,
                    "json".to_string()
                );
            }
            _ => panic!("Expected Json inference response"),
        }
    }

    #[tokio::test]
    async fn test_infer_chat_completion_stream() {
        let client = Client::new();
        let templates = get_test_template_config();
        let function_config = Box::leak(Box::new(FunctionConfig::Chat(FunctionConfigChat {
            variants: HashMap::new(),
            system_schema: None,
            user_schema: None,
            assistant_schema: None,
            tools: vec![],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: false,
        })));
        let system_template_name = "system";
        let user_template_name = "greeting_with_age";
        let good_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "good".to_string(),
        });
        let error_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "error".to_string(),
        });
        let text_model_config = ModelConfig {
            routing: vec!["good_provider".to_string()],
            providers: HashMap::from([("good_provider".to_string(), good_provider_config)]),
        };
        let error_model_config = ModelConfig {
            routing: vec!["error_provider".to_string()],
            providers: HashMap::from([("error_provider".to_string(), error_provider_config)]),
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
        let chat_completion_config = Box::leak(Box::new(ChatCompletionConfig {
            model: "error".to_string(),
            weight: 1.0,
            system_template: Some(system_template_name.into()),
            user_template: Some(user_template_name.into()),
            ..Default::default()
        }));
        let models = Box::leak(Box::new(HashMap::from([(
            "error".to_string(),
            error_model_config,
        )])));
        let inference_config = InferenceConfig {
            templates: &templates,
            tool_config: None,
            dynamic_output_schema: None,
        };
        let result = chat_completion_config
            .infer_stream(
                &input,
                models,
                function_config,
                &inference_config,
                &client,
                &mut inference_params,
            )
            .await;
        match result {
            Err(Error::ModelProvidersExhausted {
                provider_errors, ..
            }) => {
                assert_eq!(provider_errors.len(), 1);
                assert!(matches!(
                    provider_errors["error_provider"],
                    Error::InferenceClient { .. }
                ));
            }
            _ => panic!("Expected ModelProvidersExhausted error"),
        }

        // Test case 2: Model inference succeeds
        let mut inference_params = InferenceParams::default();
        let chat_completion_config = Box::leak(Box::new(ChatCompletionConfig {
            model: "good".to_string(),
            weight: 1.0,
            system_template: Some(system_template_name.into()),
            user_template: Some(user_template_name.into()),
            ..Default::default()
        }));
        let models = Box::leak(Box::new(HashMap::from([(
            "good".to_string(),
            text_model_config,
        )])));
        let inference_config = InferenceConfig {
            templates: &templates,
            tool_config: None,
            dynamic_output_schema: None,
        };
        let (first_chunk, mut stream, models_used) = chat_completion_config
            .infer_stream(
                &input,
                models,
                function_config,
                &inference_config,
                &client,
                &mut inference_params,
            )
            .await
            .unwrap();
        let first_chunk = match first_chunk {
            InferenceResultChunk::Chat(chunk) => chunk,
            _ => panic!("Expected Chat inference response"),
        };
        assert_eq!(
            first_chunk.content,
            vec![ContentBlockChunk::Text(TextChunk {
                text: DUMMY_STREAMING_RESPONSE[0].to_string(),
                id: "0".to_string()
            })]
        );
        assert_eq!(models_used.model_name, "good".to_string());
        assert_eq!(models_used.model_provider_name, "good_provider".to_string());
        let mut i = 1;
        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.unwrap();
            if i == 16 {
                // max length of text, but we have a usage chunk left
                assert_eq!(
                    chunk.usage(),
                    Some(&Usage {
                        input_tokens: 10,
                        output_tokens: 16
                    })
                );
                break;
            }
            let chunk = match chunk {
                InferenceResultChunk::Chat(chunk) => chunk,
                _ => panic!("Expected Chat inference response"),
            };
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
    /// We also test that the request correctly carries dynamic output schemas if set
    #[tokio::test]
    async fn test_prepare_request_params() {
        // We won't vary these parameters in this test
        let input = Input {
            system: None,
            messages: vec![],
        };
        let templates = get_test_template_config();
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
        let inference_config = InferenceConfig {
            templates: &templates,
            tool_config: None,
            dynamic_output_schema: None,
        };
        let model_request = chat_completion_config
            .prepare_request(
                &input,
                &function_config,
                &inference_config,
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
                &inference_config,
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
                &inference_config,
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
        let inference_config = InferenceConfig {
            templates: &templates,
            tool_config: None,
            dynamic_output_schema: None,
        };
        let mut inference_params = InferenceParams::default();
        let model_request = chat_completion_config
            .prepare_request(
                &input,
                &function_config,
                &inference_config,
                stream,
                &mut inference_params,
            )
            .unwrap();
        assert_eq!(model_request.temperature, Some(0.5));
        assert_eq!(model_request.max_tokens, Some(100));
        assert_eq!(model_request.seed, Some(69));
        assert_eq!(model_request.json_mode, ModelInferenceRequestJsonMode::On);
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
                &inference_config,
                stream,
                &mut inference_params,
            )
            .unwrap();
        assert_eq!(model_request.temperature, Some(1.));
        assert_eq!(model_request.max_tokens, Some(200));
        assert_eq!(model_request.seed, Some(420));
        assert_eq!(model_request.json_mode, ModelInferenceRequestJsonMode::On);
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
                &inference_config,
                stream,
                &mut inference_params,
            )
            .unwrap();
        assert_eq!(model_request.temperature, Some(0.9));
        assert_eq!(model_request.max_tokens, None);
        assert_eq!(model_request.seed, None);
        assert_eq!(model_request.output_schema, Some(&output_schema_value));
        assert_eq!(inference_params.chat_completion.temperature, Some(0.9));
        assert_eq!(inference_params.chat_completion.max_tokens, None);
        assert_eq!(inference_params.chat_completion.seed, None);

        let dynamic_output_schema = DynamicJSONSchema::new(serde_json::json!({
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string"
                }
            },
            "required": ["answer"]
        }));
        let dynamic_output_schema_value = dynamic_output_schema.value.clone();
        let inference_config = InferenceConfig {
            templates: &templates,
            tool_config: None,
            dynamic_output_schema: Some(dynamic_output_schema),
        };
        let model_request = chat_completion_config
            .prepare_request(
                &input,
                &function_config,
                &inference_config,
                stream,
                &mut inference_params,
            )
            .unwrap();
        assert_eq!(
            model_request.output_schema,
            Some(&dynamic_output_schema_value)
        );
    }

    #[test]
    fn test_validate_template_and_schema_both_none() {
        let templates = get_test_template_config();
        let result = validate_template_and_schema(None, None, &templates);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_template_and_schema_both_some() {
        let templates = get_test_template_config();
        let schema = JSONSchemaFromPath::new(
            PathBuf::from("fixtures/config/functions/templates_with_variables/system_schema.json"),
            PathBuf::new(),
        )
        .unwrap();
        let template = PathBuf::from("test_validate_template_and_schema_both_some");
        let result = validate_template_and_schema(Some(&schema), Some(&template), &templates);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_template_and_schema_template_no_needs_variables() {
        let templates = get_test_template_config();
        let template = PathBuf::from("system_filled");
        let result = validate_template_and_schema(None, Some(&template), &templates);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_template_and_schema_template_needs_variables() {
        let templates = get_test_template_config(); // Template needing variables
        let template = PathBuf::from("greeting");
        let result = validate_template_and_schema(None, Some(&template), &templates);
        assert!(result.is_err());

        if let Err(Error::Config { message }) = result {
            assert_eq!(
                message,
                "schema is required when template is specified and needs variables".to_string()
            );
        } else {
            panic!("Expected Error::Config");
        }
    }

    #[test]
    fn test_validate_template_and_schema_schema_some_template_none() {
        let templates = get_test_template_config(); // Default TemplateConfig
        let schema = JSONSchemaFromPath::new(
            PathBuf::from("fixtures/config/functions/templates_with_variables/system_schema.json"),
            PathBuf::new(),
        )
        .unwrap();
        let result = validate_template_and_schema(Some(&schema), None, &templates);
        assert!(result.is_err());

        if let Err(Error::Config { message }) = result {
            assert_eq!(
                message,
                "template is required when schema is specified".to_string()
            );
        } else {
            panic!("Expected Error::Config");
        }
    }
}
