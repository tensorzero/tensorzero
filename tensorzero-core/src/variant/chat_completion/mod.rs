use chrono::Duration;
use futures::future::try_join_all;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::config::path::ResolvedTomlPathData;
use crate::config::{ErrorContext, PathWithContents, SchemaData};
use crate::embeddings::EmbeddingModelTable;
use crate::endpoints::inference::{InferenceClients, InferenceModels, InferenceParams};
use crate::error::{Error, ErrorDetails};
use crate::function::FunctionConfig;
use crate::inference::types::extra_body::{ExtraBodyConfig, FullExtraBodyConfig};
use crate::inference::types::extra_headers::{ExtraHeadersConfig, FullExtraHeadersConfig};
use crate::inference::types::resolved_input::{
    LazyResolvedInput, LazyResolvedInputMessage, LazyResolvedInputMessageContent,
};
use crate::utils::retries::RetryConfig;

use crate::inference::types::{
    batch::StartBatchModelInferenceWithMetadata,
    chat_completion_inference_params::{ChatCompletionInferenceParamsV2, ServiceTier},
    ContentBlock, InferenceResultStream, ModelInferenceRequest, RequestMessage, Role, System, Text,
    Unknown,
};
use crate::inference::types::{InferenceResult, ModelInput, ResolvedInputMessage};
use crate::jsonschema_util::StaticJSONSchema;
use crate::minijinja_util::TemplateConfig;
use crate::model::ModelTable;
use crate::variant::JsonMode;

mod templates;
pub use templates::ChatTemplates;

use super::{
    infer_model_request, infer_model_request_stream, prepare_model_inference_request,
    InferModelRequestArgs, InferenceConfig, ModelUsedInfo, Variant,
};

/// If we have a schema, then we forward the 'arguments' object as-is to the template.
/// If we don't have a schema, then we create a single variable corresponding to the template
/// kind (e.g. `SYSTEM_TEXT_TEMPLATE_VAR` for a system template), and set this variable the
/// string contents of the input block.
#[derive(Debug, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct TemplateWithSchema {
    pub template: PathWithContents,
    pub schema: Option<StaticJSONSchema>,
    // If true, this is a template declared with the legacy `user_template`/`assistant_template`/`system_template`
    // or `input_wrappers.user`/`input_wrappers.assistant`/`input_wrappers.system` fields.
    // We allow using these templates without a schema, in which case we inject the special variable
    // `{user_text}`/`{assistant_text}`/`{system_text}` based on the role.
    // New-style template definitions (using `templates.<name>`) will have this set to `false`.
    // Eventually, this field will be removed entirely.
    pub legacy_definition: bool,
}

#[derive(Debug, Default, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct ChatCompletionConfig {
    weight: Option<f64>,
    model: Arc<str>,
    templates: ChatTemplates,
    temperature: Option<f32>,
    top_p: Option<f32>,
    max_tokens: Option<u32>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    seed: Option<u32>,
    stop_sequences: Option<Vec<String>>,
    #[serde(flatten)]
    pub(crate) inference_params_v2: ChatCompletionInferenceParamsV2,
    json_mode: Option<JsonMode>, // Only for JSON functions, not for chat functions
    retries: RetryConfig,
    #[cfg_attr(test, ts(skip))]
    extra_body: Option<ExtraBodyConfig>,
    #[cfg_attr(test, ts(skip))]
    extra_headers: Option<ExtraHeadersConfig>,
    #[serde(skip)]
    _private: (),
}

impl ChatCompletionConfig {
    pub fn weight(&self) -> Option<f64> {
        self.weight
    }

    pub fn set_weight(&mut self, weight: Option<f64>) {
        self.weight = weight;
    }

    pub fn model(&self) -> &Arc<str> {
        &self.model
    }

    pub fn templates(&self) -> &ChatTemplates {
        &self.templates
    }

    pub fn temperature(&self) -> Option<f32> {
        self.temperature
    }

    pub fn top_p(&self) -> Option<f32> {
        self.top_p
    }

    pub fn max_tokens(&self) -> Option<u32> {
        self.max_tokens
    }

    pub fn presence_penalty(&self) -> Option<f32> {
        self.presence_penalty
    }

    pub fn frequency_penalty(&self) -> Option<f32> {
        self.frequency_penalty
    }

    pub fn seed(&self) -> Option<u32> {
        self.seed
    }

    pub fn stop_sequences(&self) -> Option<&Vec<String>> {
        self.stop_sequences.as_ref()
    }

    pub fn reasoning_effort(&self) -> Option<&String> {
        self.inference_params_v2.reasoning_effort.as_ref()
    }

    pub fn thinking_budget_tokens(&self) -> Option<i32> {
        self.inference_params_v2.thinking_budget_tokens
    }

    pub fn service_tier(&self) -> Option<&ServiceTier> {
        self.inference_params_v2.service_tier.as_ref()
    }

    pub fn verbosity(&self) -> Option<&String> {
        self.inference_params_v2.verbosity.as_ref()
    }

    pub fn json_mode(&self) -> Option<&JsonMode> {
        self.json_mode.as_ref()
    }

    pub fn retries(&self) -> &RetryConfig {
        &self.retries
    }

    pub fn extra_body(&self) -> Option<&ExtraBodyConfig> {
        self.extra_body.as_ref()
    }

    pub fn extra_headers(&self) -> Option<&ExtraHeadersConfig> {
        self.extra_headers.as_ref()
    }

    /// Converts this initialized config back to its uninitialized form.
    /// Note: Schema associations and original file paths are not preserved.
    pub fn as_uninitialized(&self) -> UninitializedChatCompletionConfig {
        let mut system_template = None;
        let mut user_template = None;
        let mut assistant_template = None;
        let mut templates_map = HashMap::new();

        // Extract templates from ChatTemplates
        for (name, template_with_schema) in self.templates.iter_templates() {
            let path = template_with_schema.template.path.clone();

            // If this is a legacy template with a known name, put it in the legacy field
            if template_with_schema.legacy_definition {
                match name.as_str() {
                    "system" => {
                        system_template = Some(path);
                        continue;
                    }
                    "user" => {
                        user_template = Some(path);
                        continue;
                    }
                    "assistant" => {
                        assistant_template = Some(path);
                        continue;
                    }
                    _ => {}
                }
            }

            // Otherwise, put it in the new-style templates map
            templates_map.insert(name.clone(), UninitializedChatTemplate { path });
        }

        UninitializedChatCompletionConfig {
            weight: self.weight,
            model: Arc::clone(&self.model),
            system_template,
            user_template,
            assistant_template,
            input_wrappers: None, // input_wrappers are deprecated and converted to templates
            templates: UninitializedChatTemplates {
                inner: templates_map,
            },
            temperature: self.temperature,
            top_p: self.top_p,
            max_tokens: self.max_tokens,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            seed: self.seed,
            stop_sequences: self.stop_sequences.clone(),
            reasoning_effort: self.inference_params_v2.reasoning_effort.clone(),
            service_tier: self.inference_params_v2.service_tier.clone(),
            thinking_budget_tokens: self.inference_params_v2.thinking_budget_tokens,
            verbosity: self.inference_params_v2.verbosity.clone(),
            json_mode: self.json_mode,
            retries: self.retries,
            extra_body: self.extra_body.clone(),
            extra_headers: self.extra_headers.clone(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
#[serde(deny_unknown_fields)]
pub struct UninitializedInputWrappers {
    user: Option<ResolvedTomlPathData>,
    assistant: Option<ResolvedTomlPathData>,
    system: Option<ResolvedTomlPathData>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
#[serde(deny_unknown_fields)]
pub struct UninitializedChatTemplate {
    pub path: ResolvedTomlPathData,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
pub struct UninitializedChatTemplates {
    #[serde(flatten)]
    /// Internal map of chat templates, made public for GEPA optimizer integration.
    /// External users should use provided methods rather than accessing directly.
    pub inner: HashMap<String, UninitializedChatTemplate>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, ts_rs::TS)]
#[ts(export)]
#[serde(deny_unknown_fields)]
pub struct UninitializedChatCompletionConfig {
    #[serde(default)]
    pub weight: Option<f64>,
    pub model: Arc<str>,
    pub system_template: Option<ResolvedTomlPathData>,
    pub user_template: Option<ResolvedTomlPathData>,
    pub assistant_template: Option<ResolvedTomlPathData>,
    pub input_wrappers: Option<UninitializedInputWrappers>,
    #[serde(default)]
    pub templates: UninitializedChatTemplates,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<u32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub seed: Option<u32>,
    pub stop_sequences: Option<Vec<String>>,
    #[cfg_attr(test, ts(optional))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[cfg_attr(test, ts(optional))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTier>,
    #[cfg_attr(test, ts(optional))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_budget_tokens: Option<i32>,
    #[cfg_attr(test, ts(optional))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verbosity: Option<String>,
    #[serde(default)]
    pub json_mode: Option<JsonMode>, // Only for JSON functions, not for chat functions
    #[serde(default)]
    pub retries: RetryConfig,
    #[serde(default)]
    #[ts(skip)]
    pub extra_body: Option<ExtraBodyConfig>,
    #[serde(default)]
    #[ts(skip)]
    pub extra_headers: Option<ExtraHeadersConfig>,
}

impl UninitializedChatCompletionConfig {
    pub fn load(
        self,
        schemas: &SchemaData,
        error_context: &ErrorContext,
    ) -> Result<ChatCompletionConfig, Error> {
        let templates = ChatTemplates::build(&self, schemas, error_context)?;
        Ok(ChatCompletionConfig {
            weight: self.weight,
            model: self.model,
            templates,
            temperature: self.temperature,
            top_p: self.top_p,
            max_tokens: self.max_tokens,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            seed: self.seed,
            stop_sequences: self.stop_sequences,
            inference_params_v2: ChatCompletionInferenceParamsV2 {
                reasoning_effort: self.reasoning_effort,
                service_tier: self.service_tier,
                thinking_budget_tokens: self.thinking_budget_tokens,
                verbosity: self.verbosity,
            },
            json_mode: self.json_mode,
            retries: self.retries,
            extra_body: self.extra_body,
            extra_headers: self.extra_headers,
            _private: (),
        })
    }
}

impl ChatCompletionConfig {
    // NOTE - this method can become synchronous again once
    // we add a `LazyRequestMessage` type
    pub async fn prepare_request_message(
        &self,
        template_config: &TemplateConfig<'_>,
        message: &LazyResolvedInputMessage,
    ) -> Result<RequestMessage, Error> {
        prepare_request_message(message, template_config, &self.templates).await
    }

    pub fn prepare_system_message(
        &self,
        templates: &TemplateConfig,
        system: Option<&System>,
    ) -> Result<Option<String>, Error> {
        prepare_system_message(
            system,
            templates,
            self.templates
                .get_implicit_system_template()
                .map(std::convert::AsRef::as_ref),
        )
    }

    async fn prepare_request<'request>(
        &self,
        input: &LazyResolvedInput,
        function: &'request FunctionConfig,
        inference_config: &'request InferenceConfig,
        stream: bool,
        inference_params: &mut InferenceParams,
    ) -> Result<ModelInferenceRequest<'request>, Error> {
        let messages = try_join_all(
            input
                .messages
                .iter()
                .map(|message| self.prepare_request_message(&inference_config.templates, message)),
        )
        .await?;
        let system =
            self.prepare_system_message(&inference_config.templates, input.system.as_ref())?;

        inference_params
            .chat_completion
            .backfill_with_variant_params(
                self.temperature,
                self.max_tokens,
                self.seed,
                self.top_p,
                self.presence_penalty,
                self.frequency_penalty,
                self.stop_sequences.clone(),
                self.inference_params_v2.clone(),
            );

        let extra_body = FullExtraBodyConfig {
            extra_body: self.extra_body.clone(),
            inference_extra_body: inference_config
                .extra_body
                .clone()
                .filter(&inference_config.variant_name),
        };

        let extra_headers = FullExtraHeadersConfig {
            variant_extra_headers: self.extra_headers.clone(),
            inference_extra_headers: inference_config
                .extra_headers
                .clone()
                .filter(&inference_config.variant_name),
        };

        prepare_model_inference_request(
            messages,
            system,
            function,
            inference_config,
            stream,
            inference_params,
            self.json_mode,
            extra_body,
            extra_headers,
        )
    }
}

/// Prepare a ModelInput using the same machinery as is used by core TensorZero to prepare
/// chat completions requests.
pub async fn prepare_model_input(
    system: Option<&System>,
    messages: &[ResolvedInputMessage],
    templates_config: &TemplateConfig<'_>,
    chat_templates: &ChatTemplates,
) -> Result<ModelInput, Error> {
    let system = prepare_system_message(
        system,
        templates_config,
        chat_templates
            .get_implicit_system_template()
            .map(std::convert::AsRef::as_ref),
    )?;
    let mut templated_messages = Vec::with_capacity(messages.len());
    for message in messages {
        let lazy_message = message.clone().into_lazy_resolved_input_message();
        templated_messages
            .push(prepare_request_message(&lazy_message, templates_config, chat_templates).await?);
    }
    Ok(ModelInput {
        system,
        messages: try_join_all(
            templated_messages
                .into_iter()
                .map(RequestMessage::into_resolved_message),
        )
        .await?,
    })
}

pub fn prepare_system_message(
    system: Option<&System>,
    templates: &TemplateConfig,
    template: Option<&TemplateWithSchema>,
) -> Result<Option<String>, Error> {
    Ok(match template {
        Some(template) => {
            // If we have a no-schema template declared using the legacy syntax
            // (something other than `templates.<template_name>`), then we're going to inject
            // a `system_text` variable.
            let context = if template.schema.is_none() && template.legacy_definition {
                match system {
                    Some(System::Text(_)) | None => {}
                    Some(System::Template(_)) => {
                        return Err(Error::new(ErrorDetails::InvalidMessage {
                            message: "System message content is a template but `input_wrappers.system` is set in the variant config".to_string()
                        }));
                    }
                }
                let system_text = match system {
                    Some(System::Text(text)) => Value::String(text.clone()),
                    _ => Value::Null,
                };
                Cow::<Value>::Owned(serde_json::json!({
                    SYSTEM_TEXT_TEMPLATE_VAR: system_text
                }))
            } else {
                // Otherwise, we use the system message as-is.
                let system_value = match system {
                    Some(System::Text(text)) => Cow::Owned(Value::String(text.clone())),
                    Some(System::Template(arguments)) => {
                        Cow::Owned(Value::Object(arguments.0.clone()))
                    }
                    None => Cow::Owned(Value::Null),
                };
                system_value
            };
            Some(templates.template_message(&template.template.path.get_template_key(), &context)?)
        }
        None => match system {
            None => None,
            Some(System::Text(text)) => Some(text.clone()),
            Some(System::Template(_)) => {
                return Err(Error::new(ErrorDetails::InvalidMessage {
                    message:
                        "System message content is a template but there is no variant template"
                            .to_string(),
                }));
            }
        },
    })
}

pub async fn prepare_request_message(
    message: &LazyResolvedInputMessage,
    templates_config: &TemplateConfig<'_>,
    chat_templates: &ChatTemplates,
) -> Result<RequestMessage, Error> {
    let mut content = Vec::new();
    for block in &message.content {
        match block {
            LazyResolvedInputMessageContent::Text(text) => {
                let template = chat_templates.get_implicit_template(message.role);
                let text_content = match template {
                    Some(template) if template.legacy_definition => {
                        let context = serde_json::json!({
                            message.role.implicit_template_var().to_string(): text.text
                        });
                        templates_config.template_message(
                            &template.template.path.get_template_key(),
                            &context,
                        )?
                    }
                    _ => text.text.clone(),
                };
                content.push(text_content.into());
            }
            LazyResolvedInputMessageContent::Template(template_input) => {
                let template = chat_templates
                    .get_named_template(&template_input.name)
                    .ok_or_else(|| {
                        Error::new(ErrorDetails::InvalidMessage {
                            message: format!("Template `{}` not found", template_input.name),
                        })
                    })?;
                if template.schema.is_none() && template.legacy_definition {
                    return Err(Error::new(ErrorDetails::InvalidMessage {
                        message: format!("Request message content {} is not a string but `input_wrappers.{}` is set in the variant config", serde_json::to_string(&template_input.arguments).unwrap_or_default(), message.role)
                    }));
                }
                let text_content = templates_config.template_message(
                    &template.template.path.get_template_key(),
                    &template_input.arguments,
                )?;
                content.push(text_content.into());
            }
            LazyResolvedInputMessageContent::RawText(raw_text) => {
                content.push(ContentBlock::Text(Text {
                    text: raw_text.value.clone(),
                }));
            }
            // The following two clones are probably removable.
            // We will need to implement a ToolCallRef type or something so that we can avoid cloning the ToolCall and ToolResult.
            LazyResolvedInputMessageContent::ToolCall(tool_call) => {
                content.push(ContentBlock::ToolCall(tool_call.clone()));
            }
            LazyResolvedInputMessageContent::ToolResult(tool_result) => {
                content.push(ContentBlock::ToolResult(tool_result.clone()));
            }
            LazyResolvedInputMessageContent::File(file) => {
                content.push(ContentBlock::File(file.clone()));
            }
            LazyResolvedInputMessageContent::Thought(thought) => {
                content.push(ContentBlock::Thought(thought.clone()));
            }
            LazyResolvedInputMessageContent::Unknown(unknown) => {
                content.push(ContentBlock::Unknown(Unknown {
                    data: unknown.data.clone(),
                    model_name: unknown.model_name.clone(),
                    provider_name: unknown.provider_name.clone(),
                }));
            }
        }
    }

    Ok(RequestMessage {
        role: message.role,
        content,
    })
}

impl Variant for ChatCompletionConfig {
    async fn infer(
        &self,
        input: Arc<LazyResolvedInput>,
        models: InferenceModels,
        function: Arc<FunctionConfig>,
        inference_config: Arc<InferenceConfig>,
        clients: InferenceClients,
        inference_params: InferenceParams,
    ) -> Result<InferenceResult, Error> {
        let inference_config_clone = Arc::clone(&inference_config);
        let mut inference_params = inference_params;
        let request = self
            .prepare_request(
                &input,
                &function,
                &inference_config,
                false,
                &mut inference_params,
            )
            .await?;
        let model_config = models.models.get(&self.model).await?.ok_or_else(|| {
            Error::new(ErrorDetails::UnknownModel {
                name: self.model.to_string(),
            })
        })?;
        let args = InferModelRequestArgs {
            request,
            model_name: self.model.clone(),
            model_config: &model_config,
            function: &function,
            inference_config: inference_config_clone,
            clients,
            inference_params,
            retry_config: &self.retries,
        };
        infer_model_request(args).await
    }

    async fn infer_stream(
        &self,
        input: Arc<LazyResolvedInput>,
        models: InferenceModels,
        function: Arc<FunctionConfig>,
        inference_config: Arc<InferenceConfig>,
        clients: InferenceClients,
        inference_params: InferenceParams,
    ) -> Result<(InferenceResultStream, ModelUsedInfo), Error> {
        let mut inference_params = inference_params;
        let request = self
            .prepare_request(
                &input,
                &function,
                &inference_config,
                true,
                &mut inference_params,
            )
            .await?;
        let model_config = models.models.get(&self.model).await?.ok_or_else(|| {
            Error::new(ErrorDetails::UnknownModel {
                name: self.model.to_string(),
            })
        })?;
        infer_model_request_stream(
            request,
            self.model.clone(),
            &model_config,
            &function,
            clients,
            inference_params,
            self.retries,
        )
        .await
    }

    /// This function validates that the chat completion variant is correctly configured for the given function config.
    /// In order to do this, we need to check:
    ///  - For system, user, and assistant message types:
    ///    - That the template here is provided if the schema is provided in the function.
    ///    - If the template requires variables, the schema is provided.
    ///  - That the model name is a valid model
    ///  - That the weight is non-negative
    async fn validate(
        &self,
        function: Arc<FunctionConfig>,
        models: &ModelTable,
        _embedding_models: &EmbeddingModelTable,
        templates: &TemplateConfig<'_>,
        function_name: &str,
        variant_name: &str,
        _global_outbound_http_timeout: &Duration,
    ) -> Result<(), Error> {
        // Validate that weight is non-negative
        if self.weight.is_some_and(|w| w < 0.0) {
            return Err(ErrorDetails::Config {
                message: format!(
                    "`functions.{function_name}.variants.{variant_name}`: `weight` must be non-negative"
                ),
            }.into());
        }
        models.validate(&self.model)?;

        // Validate the system template matches the system schema (best effort, we cannot check the variables comprehensively)
        validate_legacy_template_and_schema(
            TemplateKind::System,
            function.system_schema(),
            self.templates.get_implicit_system_template().map(|t| &**t),
            templates,
        )
        .map_err(|e| {
            Error::new(ErrorDetails::Config {
                message: format!(
                    "`functions.{function_name}.variants.{variant_name}.system_template`: {e}"
                ),
            })
        })?;

        // Validate the user template matches the user schema (best effort, we cannot check the variables comprehensively)
        validate_legacy_template_and_schema(
            TemplateKind::User,
            function.user_schema(),
            self.templates
                .get_implicit_template(Role::User)
                .map(|t| &**t),
            templates,
        )
        .map_err(|e| {
            Error::new(ErrorDetails::Config {
                message: format!(
                    "`functions.{function_name}.variants.{variant_name}.user_template`: {e}"
                ),
            })
        })?;

        // Validate the assistant template matches the assistant schema (best effort, we cannot check the variables comprehensively)
        validate_legacy_template_and_schema(
            TemplateKind::Assistant,
            function.assistant_schema(),
            self.templates
                .get_implicit_template(Role::Assistant)
                .map(|t| &**t),
            templates,
        )
        .map_err(|e| {
            Error::new(ErrorDetails::Config {
                message: format!(
                    "`functions.{function_name}.variants.{variant_name}.assistant_template`: {e}"
                ),
            })
        })?;

        validate_all_schemas_have_templates(&function, &self.templates).map_err(|e| {
            let schema_name = e.schema_name;
            Error::new(ErrorDetails::Config {
                message: format!(
                    "`functions.{function_name}.variants.{variant_name}.templates.{schema_name}` is required when `functions.{function_name}.schemas.{schema_name}` is specified"
                ),
            })
        })?;

        // Validate that json_mode = "tool" is not used with chat functions that have tools configured
        if let Some(JsonMode::Tool) = self.json_mode {
            if function.tools().next().is_some() {
                return Err(ErrorDetails::Config {
                    message: format!(
                        "`functions.{function_name}.variants.{variant_name}`: Cannot use `json_mode = \"tool\"` with chat functions that have tools configured. Please remove tools from the function or use a JSON function instead."
                    ),
                }
                .into());
            }
        }

        Ok(())
    }

    fn get_all_template_paths(&self) -> Vec<&PathWithContents> {
        self.templates.get_all_template_paths()
    }

    fn get_all_explicit_template_names(&self) -> HashSet<String> {
        self.templates.get_all_explicit_template_names()
    }

    async fn start_batch_inference<'a>(
        &'a self,
        inputs: &[LazyResolvedInput],
        models: InferenceModels,
        function: &'a FunctionConfig,
        inference_configs: &'a [InferenceConfig],
        clients: InferenceClients,
        inference_params: Vec<InferenceParams>,
    ) -> Result<StartBatchModelInferenceWithMetadata<'a>, Error> {
        // First construct all inference configs so they stick around for the duration of this function body
        let mut inference_params = inference_params;

        // Next, prepare all the ModelInferenceRequests
        let mut inference_requests = Vec::new();
        for ((input, inference_param), inference_config) in inputs
            .iter()
            .zip(&mut inference_params)
            .zip(inference_configs)
        {
            let request = self
                .prepare_request(input, function, inference_config, false, inference_param)
                .await?;
            inference_requests.push(request);
        }
        let model_config = models.models.get(&self.model).await?.ok_or_else(|| {
            Error::new(ErrorDetails::UnknownModel {
                name: self.model.to_string(),
            })
        })?;
        let model_inference_response = model_config
            .start_batch_inference(
                &inference_requests,
                &clients.http_client,
                &clients.credentials,
            )
            .await?;
        Ok(StartBatchModelInferenceWithMetadata::new(
            model_inference_response,
            inference_requests,
            &self.model,
            inference_params,
        ))
    }
}

/// The template variable names used when applying a legacy template with no schema
/// Only one of these variables is used per template, based on the `TemplateKind`
pub const SYSTEM_TEXT_TEMPLATE_VAR: &str = "system_text";
pub const USER_TEXT_TEMPLATE_VAR: &str = "user_text";
pub const ASSISTANT_TEXT_TEMPLATE_VAR: &str = "assistant_text";

#[derive(Copy, Clone, Debug)]
pub enum TemplateKind {
    System,
    User,
    Assistant,
}

pub fn validate_legacy_template_and_schema(
    kind: TemplateKind,
    schema: Option<&StaticJSONSchema>,
    template: Option<&TemplateWithSchema>,
    templates: &TemplateConfig,
) -> Result<(), Error> {
    match (schema, template) {
        (None, Some(template)) => {
            let template_name = template.template.path.get_template_key();
            let undeclared_vars = templates.get_undeclared_variables(&template_name)?;
            let allowed_var = match kind {
                TemplateKind::System => SYSTEM_TEXT_TEMPLATE_VAR,
                TemplateKind::User => USER_TEXT_TEMPLATE_VAR,
                TemplateKind::Assistant => ASSISTANT_TEXT_TEMPLATE_VAR,
            };
            // When a legacy template has no schema, the template can have at most one variable.
            // New-style templates (declared with `templates.<name>`) can have any number of variables
            // when no schema is provided - any undefined variables will produce an error when we actually
            // apply the template
            if template.legacy_definition && !undeclared_vars.is_empty() {
                // If the template has any variables, it must be the one allowed variable (e.g. `system_text`)
                // based on the template kind
                let mut undeclared_vars = undeclared_vars.into_iter().collect::<Vec<_>>();
                if undeclared_vars != [allowed_var.to_string()] {
                    // Ensure that the error message is deterministic
                    undeclared_vars.sort();
                    let undeclared_vars_str = format!("[{}]", undeclared_vars.join(", "));
                    return Err(Error::new(ErrorDetails::Config {
                        message:
                            format!("template needs variables: {undeclared_vars_str} but only `{allowed_var}` is allowed when template has no schema")
                                .to_string(),
                    }));
                }
            }
        }
        (Some(_), None) => {
            return Err(Error::new(ErrorDetails::Config {
                message: "template is required when schema is specified".to_string(),
            }));
        }
        _ => {}
    }
    Ok(())
}

pub struct MissingTemplateError {
    pub schema_name: String,
}

/// Checks that all schemas declared by the function have a corresponding template.
pub fn validate_all_schemas_have_templates(
    function: &FunctionConfig,
    templates: &ChatTemplates,
) -> Result<(), MissingTemplateError> {
    for schema_name in function.schemas().inner.keys() {
        if templates.get_named_template(schema_name).is_none() {
            return Err(MissingTemplateError {
                schema_name: schema_name.to_string(),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::rate_limiting::ScopeInfo;
    use indexmap::IndexMap;
    use std::collections::HashMap;
    use std::path::PathBuf;

    use super::*;

    use futures::StreamExt;

    use serde_json::json;
    use uuid::Uuid;

    use crate::cache::{CacheEnabledMode, CacheOptions};
    use crate::config::{provider_types::ProviderTypesConfig, SchemaData, UninitializedSchemas};
    use crate::db::{clickhouse::ClickHouseConnectionInfo, postgres::PostgresConnectionInfo};
    use crate::embeddings::EmbeddingModelTable;
    use crate::endpoints::inference::{
        ChatCompletionInferenceParams, InferenceCredentials, InferenceIds,
    };
    use crate::experimentation::ExperimentationConfig;
    use crate::function::{FunctionConfigChat, FunctionConfigJson};
    use crate::http::TensorzeroHttpClient;
    use crate::inference::types::Template;
    use crate::inference::types::{
        Arguments, ContentBlockChatOutput, InferenceResultChunk, ModelInferenceRequestJsonMode,
        Usage,
    };
    use crate::jsonschema_util::{DynamicJSONSchema, StaticJSONSchema};
    use crate::minijinja_util::tests::{
        get_assistant_template, get_greeting_with_age_template, get_system_filled_template,
        get_system_template, get_test_template_config, test_assistant_template_schema,
        test_system_template_schema, test_user_template_schema,
    };
    use crate::model::{ModelConfig, ModelProvider, ProviderConfig};
    use crate::model_table::ProviderTypeDefaultCredentials;
    use crate::providers::dummy::{DummyProvider, DUMMY_JSON_RESPONSE_RAW};
    use crate::providers::test_helpers::get_temperature_tool_config;
    use crate::tool::{ToolCallConfig, ToolChoice};
    use crate::{
        error::Error,
        inference::types::{ContentBlockChunk, Role, TextChunk},
        providers::dummy::{DUMMY_INFER_RESPONSE_CONTENT, DUMMY_STREAMING_RESPONSE},
    };

    #[tokio::test]
    async fn test_prepare_request_message() {
        let templates = get_test_template_config().await;
        // Part 1: test without templates
        let chat_completion_config = ChatCompletionConfig {
            model: "dummy".into(),
            weight: Some(1.0),
            templates: ChatTemplates::empty(),
            json_mode: Some(JsonMode::On),
            ..Default::default()
        };

        // Test case 1: Regular user message
        let input_message = LazyResolvedInputMessage {
            role: Role::User,
            content: vec!["Hello, how are you?".to_string().into()],
        };
        let prepared_message = chat_completion_config
            .prepare_request_message(&templates, &input_message)
            .await
            .unwrap();
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
        let input_message = LazyResolvedInputMessage {
            role: Role::Assistant,
            content: vec!["I'm doing well, thank you!".to_string().into()],
        };
        let prepared_message = chat_completion_config
            .prepare_request_message(&templates, &input_message)
            .await
            .unwrap();
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
        let input_message = LazyResolvedInputMessage {
            role: Role::User,
            content: vec![LazyResolvedInputMessageContent::Template(Template {
                name: "user".to_string(),
                arguments: Arguments(serde_json::Map::from_iter([(
                    "invalid".to_string(),
                    "json".into(),
                )])),
            })],
        };
        let result = chat_completion_config
            .prepare_request_message(&templates, &input_message)
            .await
            .unwrap_err();
        assert_eq!(
            result,
            ErrorDetails::InvalidMessage {
                message: "Template `user` not found".to_string()
            }
            .into()
        );

        // Part 2: test with templates
        let system_template = get_system_template();
        let user_template = get_greeting_with_age_template();
        let assistant_template = get_assistant_template();

        let chat_completion_config = UninitializedChatCompletionConfig {
            model: "dummy".into(),
            weight: Some(1.0),
            system_template: Some(system_template.clone()),
            user_template: Some(user_template.clone()),
            assistant_template: Some(assistant_template.clone()),
            input_wrappers: None,

            json_mode: Some(JsonMode::On),
            ..Default::default()
        }
        .load(
            &SchemaData::load(
                Some(test_user_template_schema()),
                Some(test_assistant_template_schema()),
                Some(test_system_template_schema()),
                UninitializedSchemas::default(),
                "test",
            )
            .unwrap(),
            &ErrorContext {
                function_name: "test".to_string(),
                variant_name: "test".to_string(),
            },
        )
        .unwrap();

        // Test case 4: Assistant message with template
        let input_message = LazyResolvedInputMessage {
            role: Role::Assistant,
            content: vec![LazyResolvedInputMessageContent::Template(Template {
                name: "assistant".to_string(),
                arguments: Arguments(serde_json::Map::from_iter([(
                    "reason".to_string(),
                    "it's against my ethical guidelines".into(),
                )])),
            })],
        };
        let prepared_message = chat_completion_config
            .prepare_request_message(&templates, &input_message)
            .await
            .unwrap();
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
        let input_message = LazyResolvedInputMessage {
            role: Role::User,
            content: vec![LazyResolvedInputMessageContent::Template(Template {
                name: "user".to_string(),
                arguments: Arguments(serde_json::Map::from_iter([
                    ("name".to_string(), "John".into()),
                    ("age".to_string(), 30.into()),
                ])),
            })],
        };
        let prepared_message = chat_completion_config
            .prepare_request_message(&templates, &input_message)
            .await
            .unwrap();
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
        let input_message = LazyResolvedInputMessage {
            role: Role::User,
            content: vec![LazyResolvedInputMessageContent::Template(Template {
                name: "user".to_string(),
                arguments: Arguments(serde_json::Map::from_iter([(
                    "name".to_string(),
                    "Alice".into(),
                )])), // Missing "age" field
            })],
        };
        let result = chat_completion_config
            .prepare_request_message(&templates, &input_message)
            .await;
        assert!(result.is_err());
        match result.unwrap_err().get_details() {
            ErrorDetails::MiniJinjaTemplateRender { message, .. } => {
                assert!(message.contains("undefined value"));
            }
            _ => panic!("Expected MiniJinjaTemplateRender error"),
        }
        // Test case 7: User message with string content when template is provided.
        // This bypasses the template
        let chat_completion_config_non_legacy = UninitializedChatCompletionConfig {
            model: "dummy".into(),
            weight: Some(1.0),
            templates: UninitializedChatTemplates {
                inner: HashMap::from([(
                    "user".to_string(),
                    UninitializedChatTemplate {
                        path: user_template.clone(),
                    },
                )]),
            },
            input_wrappers: None,
            json_mode: Some(JsonMode::On),
            ..Default::default()
        }
        .load(
            &SchemaData::load(
                Some(test_user_template_schema()),
                Some(test_assistant_template_schema()),
                Some(test_system_template_schema()),
                UninitializedSchemas::default(),
                "test",
            )
            .unwrap(),
            &ErrorContext {
                function_name: "test".to_string(),
                variant_name: "test".to_string(),
            },
        )
        .unwrap();
        let input_message = LazyResolvedInputMessage {
            role: Role::User,
            content: vec!["This is a plain string".to_string().into()],
        };
        let result = chat_completion_config_non_legacy
            .prepare_request_message(&templates, &input_message)
            .await;
        let prepared_message = result.unwrap();
        match prepared_message {
            RequestMessage {
                role: Role::User,
                content: user_message,
            } => {
                assert_eq!(
                    user_message,
                    vec!["This is a plain string".to_string().into()]
                );
            }
            _ => panic!("Expected User message"),
        }
    }

    #[tokio::test]
    async fn test_prepare_system_message() {
        let templates = get_test_template_config().await;

        // Test without templates, string message
        let chat_completion_config = ChatCompletionConfig {
            model: "dummy".into(),
            weight: Some(1.0),
            ..Default::default()
        };
        let input_message = System::Text("You are a helpful assistant.".to_string());
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
            model: "dummy".into(),
            weight: Some(1.0),
            ..Default::default()
        };
        let input_message = System::Template(Arguments(
            json!({"message": "You are a helpful assistant."})
                .as_object()
                .unwrap()
                .clone(),
        ));
        let result =
            chat_completion_config.prepare_system_message(&templates, Some(&input_message));
        assert!(result.is_err());
        let prepared_message = result.unwrap_err();
        assert_eq!(
            prepared_message,
            ErrorDetails::InvalidMessage {
                message: "System message content is a template but there is no variant template"
                    .to_string()
            }
            .into()
        );

        // Test without templates, no message
        let chat_completion_config = ChatCompletionConfig {
            model: "dummy".into(),
            weight: Some(1.0),
            ..Default::default()
        };
        let result = chat_completion_config.prepare_system_message(&templates, None);
        assert!(result.is_ok());
        let prepared_message = result.unwrap();
        assert_eq!(prepared_message, None);

        // Test with templates that need new info
        let system_template = get_system_template();

        let chat_completion_config = UninitializedChatCompletionConfig {
            model: "dummy".into(),
            weight: Some(1.0),
            system_template: Some(system_template),
            user_template: None,
            assistant_template: None,
            input_wrappers: None,
            ..Default::default()
        }
        .load(
            &SchemaData::load(
                None,
                None,
                Some(test_system_template_schema()),
                UninitializedSchemas::default(),
                "test",
            )
            .unwrap(),
            &ErrorContext {
                function_name: "test".to_string(),
                variant_name: "test".to_string(),
            },
        )
        .unwrap();

        let input_message = System::Template(Arguments(
            serde_json::json!({"assistant_name": "ChatGPT"})
                .as_object()
                .unwrap()
                .clone(),
        ));
        let prepared_message = chat_completion_config
            .prepare_system_message(&templates, Some(&input_message))
            .unwrap();
        assert_eq!(
            prepared_message,
            Some("You are a helpful and friendly assistant named ChatGPT".to_string())
        );

        // Test with template that is complete as is (string)
        let system_template = get_system_filled_template();

        let chat_completion_config = UninitializedChatCompletionConfig {
            model: "dummy".into(),
            weight: Some(1.0),
            system_template: Some(system_template),
            ..Default::default()
        }
        .load(
            &SchemaData::default(),
            &ErrorContext {
                function_name: "test".to_string(),
                variant_name: "test".to_string(),
            },
        )
        .unwrap();

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
        let client = TensorzeroHttpClient::new_testing().unwrap();
        let clickhouse_connection_info = ClickHouseConnectionInfo::new_disabled();
        let api_keys = InferenceCredentials::default();
        let clients = InferenceClients {
            http_client: client.clone(),
            clickhouse_connection_info: clickhouse_connection_info.clone(),
            postgres_connection_info: PostgresConnectionInfo::Disabled,
            credentials: Arc::new(api_keys),
            cache_options: CacheOptions {
                max_age_s: None,
                enabled: CacheEnabledMode::WriteOnly,
            },
            tags: Arc::new(Default::default()),
            rate_limiting_config: Arc::new(Default::default()),
            otlp_config: Default::default(),
            deferred_tasks: tokio_util::task::TaskTracker::new(),
            scope_info: ScopeInfo {
                tags: Arc::new(HashMap::new()),
                api_key_public_id: None,
            },
        };
        let templates = Arc::new(get_test_template_config().await);
        let system_template = get_system_template();
        let user_template = get_greeting_with_age_template();
        let chat_completion_config = UninitializedChatCompletionConfig {
            model: "good".into(),
            weight: Some(1.0),
            system_template: Some(system_template.clone()),
            user_template: Some(user_template.clone()),
            assistant_template: None,
            input_wrappers: None,

            ..Default::default()
        }
        .load(
            &SchemaData::load(
                Some(test_user_template_schema()),
                None,
                Some(test_system_template_schema()),
                UninitializedSchemas::default(),
                "test",
            )
            .unwrap(),
            &ErrorContext {
                function_name: "test".to_string(),
                variant_name: "test".to_string(),
            },
        )
        .unwrap();
        let schema_any = StaticJSONSchema::from_value(json!({ "type": "object" })).unwrap();
        let function_config = Arc::new(FunctionConfig::Chat(FunctionConfigChat {
            variants: HashMap::new(),
            schemas: SchemaData::load(
                Some(schema_any.clone()),
                Some(schema_any.clone()),
                Some(schema_any.clone()),
                UninitializedSchemas::default(),
                "test",
            )
            .unwrap(),
            tools: vec![],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: None,
            description: None,
            all_explicit_templates_names: HashSet::new(),
            experimentation: ExperimentationConfig::default(),
        }));
        let good_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "good".into(),
            ..Default::default()
        });
        let error_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "error".into(),
            ..Default::default()
        });
        let json_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "json".into(),
            ..Default::default()
        });
        let text_model_config = ModelConfig {
            routing: vec!["good".into()],
            providers: HashMap::from([(
                "good".into(),
                ModelProvider {
                    name: "good".into(),
                    config: good_provider_config,
                    extra_body: Default::default(),
                    extra_headers: Default::default(),
                    timeouts: Default::default(),
                    discard_unknown_chunks: false,
                },
            )]),
            timeouts: Default::default(),
        };
        let json_model_config = ModelConfig {
            routing: vec!["json_provider".into()],
            providers: HashMap::from([(
                "json_provider".into(),
                ModelProvider {
                    name: "json_provider".into(),
                    config: json_provider_config,
                    extra_body: Default::default(),
                    extra_headers: Default::default(),
                    timeouts: Default::default(),
                    discard_unknown_chunks: false,
                },
            )]),
            timeouts: Default::default(),
        };
        let tool_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "tool".into(),
            ..Default::default()
        });
        let tool_model_config = ModelConfig {
            routing: vec!["tool_provider".into()],
            providers: HashMap::from([(
                "tool_provider".into(),
                ModelProvider {
                    name: "tool_provider".into(),
                    config: tool_provider_config,
                    extra_body: Default::default(),
                    extra_headers: Default::default(),
                    timeouts: Default::default(),
                    discard_unknown_chunks: false,
                },
            )]),
            timeouts: Default::default(),
        };
        let error_model_config = ModelConfig {
            routing: vec!["error".into()],
            providers: HashMap::from([(
                "error".into(),
                ModelProvider {
                    name: "error".into(),
                    config: error_provider_config,
                    extra_body: Default::default(),
                    extra_headers: Default::default(),
                    timeouts: Default::default(),
                    discard_unknown_chunks: false,
                },
            )]),
            timeouts: Default::default(),
        };
        // Test case 1: invalid message (String passed when template required)
        let messages = vec![LazyResolvedInputMessage {
            role: Role::User,
            content: vec!["Hello".to_string().into()],
        }];
        let input = LazyResolvedInput {
            system: Some(System::Text("Hello".to_string())),
            messages,
        };
        let inference_params = InferenceParams::default();
        let inference_config = InferenceConfig {
            templates: templates.clone(),
            tool_config: None,
            function_name: "".into(),
            variant_name: "".into(),
            dynamic_output_schema: None,
            ids: InferenceIds {
                inference_id: Uuid::now_v7(),
                episode_id: Uuid::now_v7(),
            },
            fetch_and_encode_input_files_before_inference: false,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            extra_cache_key: None,
        };
        let models = ModelTable::default();
        let inference_models = InferenceModels {
            models: Arc::new(models),
            embedding_models: Arc::new(EmbeddingModelTable::default()),
        };
        let result = chat_completion_config
            .infer(
                Arc::new(input.clone()),
                inference_models.clone(),
                Arc::clone(&function_config),
                Arc::new(inference_config.clone()),
                clients.clone(),
                inference_params,
            )
            .await
            .unwrap_err();
        match result.get_details() {
            ErrorDetails::MiniJinjaTemplateRender { message, .. } => {
                // template_name is a test filename
                assert!(message.contains("undefined value"));
            }
            _ => panic!("Expected MiniJinjaTemplateRender error"),
        }

        // Test case 2: invalid model in request
        let inference_params = InferenceParams::default();
        let messages = vec![LazyResolvedInputMessage {
            role: Role::User,
            content: vec![],
        }];
        let input = LazyResolvedInput {
            system: Some(System::Template(Arguments(
                json!({"assistant_name": "R2-D2"})
                    .as_object()
                    .unwrap()
                    .clone(),
            ))),
            messages,
        };
        let provider_types = ProviderTypesConfig::default();
        let models = ModelTable::new(
            HashMap::from([("invalid_model".into(), text_model_config)]),
            ProviderTypeDefaultCredentials::new(&provider_types).into(),
            crate::http::DEFAULT_HTTP_CLIENT_TIMEOUT,
        )
        .unwrap();
        let inference_models = InferenceModels {
            models: Arc::new(models),
            embedding_models: Arc::new(EmbeddingModelTable::default()),
        };
        let inference_config = InferenceConfig {
            templates: templates.clone(),
            tool_config: None,
            function_name: "".into(),
            variant_name: "".into(),
            dynamic_output_schema: None,
            fetch_and_encode_input_files_before_inference: false,
            ids: InferenceIds {
                inference_id: Uuid::now_v7(),
                episode_id: Uuid::now_v7(),
            },
            extra_body: Default::default(),
            extra_headers: Default::default(),
            extra_cache_key: None,
        };
        let result = chat_completion_config
            .infer(
                Arc::new(input.clone()),
                inference_models.clone(),
                Arc::clone(&function_config),
                Arc::new(inference_config.clone()),
                clients.clone(),
                inference_params,
            )
            .await
            .unwrap_err();
        assert!(
            matches!(result.get_details(), ErrorDetails::UnknownModel { .. }),
            "{}",
            result
        );
        // Test case 3: Model inference fails because of model issues

        let chat_completion_config = UninitializedChatCompletionConfig {
            model: "error".into(),
            weight: Some(1.0),
            system_template: Some(system_template.clone()),
            user_template: Some(user_template.clone()),
            assistant_template: None,
            input_wrappers: None,
            ..Default::default()
        }
        .load(
            &SchemaData::load(
                Some(test_user_template_schema()),
                None,
                Some(test_system_template_schema()),
                UninitializedSchemas::default(),
                "test",
            )
            .unwrap(),
            &ErrorContext {
                function_name: "test".to_string(),
                variant_name: "test".to_string(),
            },
        )
        .unwrap();
        let inference_params = InferenceParams::default();
        let models = HashMap::from([("error".into(), error_model_config)]);
        let provider_types = ProviderTypesConfig::default();
        let models = ModelTable::new(
            models,
            ProviderTypeDefaultCredentials::new(&provider_types).into(),
            crate::http::DEFAULT_HTTP_CLIENT_TIMEOUT,
        )
        .unwrap();
        let inference_models = InferenceModels {
            models: Arc::new(models),
            embedding_models: Arc::new(
                EmbeddingModelTable::new(
                    HashMap::new(),
                    ProviderTypeDefaultCredentials::new(&provider_types).into(),
                    crate::http::DEFAULT_HTTP_CLIENT_TIMEOUT,
                )
                .unwrap(),
            ),
        };
        let inference_config = InferenceConfig {
            templates: templates.clone(),
            tool_config: None,
            function_name: "".into(),
            variant_name: "".into(),
            dynamic_output_schema: None,
            fetch_and_encode_input_files_before_inference: false,
            ids: InferenceIds {
                inference_id: Uuid::now_v7(),
                episode_id: Uuid::now_v7(),
            },
            extra_body: Default::default(),
            extra_headers: Default::default(),
            extra_cache_key: None,
        };
        let err = chat_completion_config
            .infer(
                Arc::new(input.clone()),
                inference_models.clone(),
                Arc::clone(&function_config),
                Arc::new(inference_config.clone()),
                clients.clone(),
                inference_params,
            )
            .await
            .unwrap_err();
        let details = err.get_details();
        assert_eq!(
            *details,
            ErrorDetails::ModelProvidersExhausted {
                provider_errors: IndexMap::from([(
                    "error".to_string(),
                    Error::new(ErrorDetails::InferenceClient {
                        message: "Error sending request to Dummy provider for model 'error'."
                            .to_string(),
                        status_code: None,
                        raw_request: Some("raw request".to_string()),
                        raw_response: None,
                        provider_type: "dummy".to_string(),
                    })
                )])
            }
        );

        // Test case 4: Model inference succeeds
        let inference_params = InferenceParams::default();
        let chat_completion_config = UninitializedChatCompletionConfig {
            model: "good".into(),
            weight: Some(1.0),
            system_template: Some(system_template.clone()),
            user_template: Some(user_template.clone()),
            assistant_template: None,
            input_wrappers: None,
            ..Default::default()
        }
        .load(
            &SchemaData::load(
                Some(test_user_template_schema()),
                None,
                Some(test_system_template_schema()),
                UninitializedSchemas::default(),
                "test",
            )
            .unwrap(),
            &ErrorContext {
                function_name: "test".to_string(),
                variant_name: "test".to_string(),
            },
        )
        .unwrap();
        let good_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "good".into(),
            ..Default::default()
        });
        let text_model_config = ModelConfig {
            routing: vec!["good_provider".into()],
            providers: HashMap::from([(
                "good_provider".into(),
                ModelProvider {
                    name: "good_provider".into(),
                    config: good_provider_config,
                    extra_body: Default::default(),
                    extra_headers: Default::default(),
                    timeouts: Default::default(),
                    discard_unknown_chunks: false,
                },
            )]),
            timeouts: Default::default(),
        };
        let provider_types = ProviderTypesConfig::default();
        let models = ModelTable::new(
            HashMap::from([("good".into(), text_model_config)]),
            ProviderTypeDefaultCredentials::new(&provider_types).into(),
            crate::http::DEFAULT_HTTP_CLIENT_TIMEOUT,
        )
        .unwrap();
        let inference_models = InferenceModels {
            models: Arc::new(models),
            embedding_models: Arc::new(EmbeddingModelTable::default()),
        };
        let inference_config = InferenceConfig {
            templates: templates.clone(),
            tool_config: None,
            function_name: "".into(),
            variant_name: "".into(),
            dynamic_output_schema: None,
            fetch_and_encode_input_files_before_inference: false,
            ids: InferenceIds {
                inference_id: Uuid::now_v7(),
                episode_id: Uuid::now_v7(),
            },
            extra_body: Default::default(),
            extra_headers: Default::default(),
            extra_cache_key: None,
        };
        let result = chat_completion_config
            .infer(
                Arc::new(input.clone()),
                inference_models.clone(),
                Arc::clone(&function_config),
                Arc::new(inference_config.clone()),
                clients.clone(),
                inference_params.clone(),
            )
            .await
            .unwrap();
        assert!(matches!(result, InferenceResult::Chat(_)));
        assert_eq!(
            result.usage_considering_cached(),
            Usage {
                input_tokens: Some(10),
                output_tokens: Some(1),
            }
        );
        match result {
            InferenceResult::Chat(chat_response) => {
                assert_eq!(
                    chat_response.content,
                    vec![DUMMY_INFER_RESPONSE_CONTENT.to_string().into()]
                );
                assert_eq!(chat_response.model_inference_results.len(), 1);
                assert_eq!(
                    chat_response.model_inference_results[0].output,
                    vec![DUMMY_INFER_RESPONSE_CONTENT.to_string().into()]
                );
                assert_eq!(
                    &*chat_response.model_inference_results[0].model_name,
                    "good".to_string()
                );
                assert_eq!(
                    &*chat_response.model_inference_results[0].model_provider_name,
                    "good_provider".to_string()
                );
                assert_eq!(chat_response.inference_params, inference_params);
            }
            InferenceResult::Json(_) => panic!("Expected Chat inference response"),
        }

        // Test case 5: tool call
        let inference_params = InferenceParams::default();
        let chat_completion_config = ChatCompletionConfig {
            model: "tool".into(),
            weight: Some(1.0),
            ..Default::default()
        };
        let input = LazyResolvedInput {
            system: None,
            messages: vec![LazyResolvedInputMessage {
                role: Role::User,
                content: vec!["What is the weather in Brooklyn?".to_string().into()],
            }],
        };
        let provider_types = ProviderTypesConfig::default();
        let models = ModelTable::new(
            HashMap::from([("tool".into(), tool_model_config)]),
            ProviderTypeDefaultCredentials::new(&provider_types).into(),
            crate::http::DEFAULT_HTTP_CLIENT_TIMEOUT,
        )
        .unwrap();
        let inference_models = InferenceModels {
            models: Arc::new(models),
            embedding_models: Arc::new(EmbeddingModelTable::default()),
        };
        let weather_tool_config = get_temperature_tool_config();
        let inference_config = InferenceConfig {
            templates: templates.clone(),
            tool_config: Some(Arc::new(weather_tool_config)),
            function_name: "".into(),
            variant_name: "".into(),
            dynamic_output_schema: None,
            fetch_and_encode_input_files_before_inference: false,
            ids: InferenceIds {
                inference_id: Uuid::now_v7(),
                episode_id: Uuid::now_v7(),
            },
            extra_body: Default::default(),
            extra_headers: Default::default(),
            extra_cache_key: None,
        };
        let result = chat_completion_config
            .infer(
                Arc::new(input.clone()),
                inference_models.clone(),
                Arc::clone(&function_config),
                Arc::new(inference_config.clone()),
                clients.clone(),
                inference_params.clone(),
            )
            .await
            .unwrap();
        assert!(matches!(result, InferenceResult::Chat(_)));
        assert_eq!(
            result.usage_considering_cached(),
            Usage {
                input_tokens: Some(10),
                output_tokens: Some(1),
            }
        );
        match result {
            InferenceResult::Chat(chat_response) => {
                assert_eq!(chat_response.content.len(), 1);
                let tool_call = &chat_response.content[0];
                match tool_call {
                    ContentBlockChatOutput::ToolCall(tool_call) => {
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
                assert_eq!(chat_response.model_inference_results.len(), 1);
                assert_eq!(
                    &*chat_response.model_inference_results[0].model_provider_name,
                    "tool_provider".to_string()
                );
                assert_eq!(
                    &*chat_response.model_inference_results[0].model_name,
                    "tool".to_string()
                );
                assert_eq!(chat_response.inference_params, inference_params);
            }
            InferenceResult::Json(_) => panic!("Expected Chat inference response"),
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
        let json_mode_tool_call_config = ToolCallConfig::implicit_from_value(&output_schema);
        let output_schema = StaticJSONSchema::from_value(output_schema).unwrap();
        let schema_any = StaticJSONSchema::from_value(json!({ "type": "object" })).unwrap();
        let json_function_config = Arc::new(FunctionConfig::Json(FunctionConfigJson {
            variants: HashMap::new(),
            schemas: SchemaData::load(
                Some(schema_any.clone()),
                Some(schema_any.clone()),
                Some(schema_any.clone()),
                UninitializedSchemas::default(),
                "test",
            )
            .unwrap(),
            output_schema,
            json_mode_tool_call_config,
            description: None,
            all_explicit_template_names: HashSet::new(),
            experimentation: ExperimentationConfig::default(),
        }));
        let inference_config = InferenceConfig {
            templates: templates.clone(),
            tool_config: None,
            function_name: "".into(),
            variant_name: "".into(),
            dynamic_output_schema: None,
            fetch_and_encode_input_files_before_inference: false,
            ids: InferenceIds {
                inference_id: Uuid::now_v7(),
                episode_id: Uuid::now_v7(),
            },
            extra_body: Default::default(),
            extra_headers: Default::default(),
            extra_cache_key: None,
        };
        let inference_params = InferenceParams::default();
        let result = chat_completion_config
            .infer(
                Arc::new(input.clone()),
                inference_models.clone(),
                Arc::clone(&json_function_config),
                Arc::new(inference_config.clone()),
                clients.clone(),
                inference_params.clone(),
            )
            .await
            .unwrap();
        assert_eq!(
            result.usage_considering_cached(),
            Usage {
                input_tokens: Some(10),
                output_tokens: Some(1),
            }
        );
        match result {
            InferenceResult::Json(json_result) => {
                assert!(json_result.output.parsed.is_none());
                assert_eq!(
                    json_result.output.raw,
                    Some(r#"{"location":"Brooklyn","units":"celsius"}"#.to_string())
                );
                assert_eq!(json_result.model_inference_results.len(), 1);
                assert_eq!(json_result.inference_params, inference_params);
            }
            InferenceResult::Chat(_) => panic!("Expected Json inference response"),
        }
        let messages = vec![LazyResolvedInputMessage {
            role: Role::User,
            content: vec![LazyResolvedInputMessageContent::Template(Template {
                name: "user".to_string(),
                arguments: Arguments(
                    json!({"name": "Luke", "age": 20})
                        .as_object()
                        .unwrap()
                        .clone(),
                ),
            })],
        }];
        let input = LazyResolvedInput {
            system: Some(System::Template(Arguments(
                json!({"assistant_name": "R2-D2"})
                    .as_object()
                    .unwrap()
                    .clone(),
            ))),
            messages,
        };
        // Test case 6: JSON output was supposed to happen and it did
        let inference_params = InferenceParams::default();
        let provider_types = ProviderTypesConfig::default();
        let models = ModelTable::new(
            HashMap::from([("json".into(), json_model_config)]),
            ProviderTypeDefaultCredentials::new(&provider_types).into(),
            crate::http::DEFAULT_HTTP_CLIENT_TIMEOUT,
        )
        .unwrap();
        let inference_models = InferenceModels {
            models: Arc::new(models),
            embedding_models: Arc::new(EmbeddingModelTable::default()),
        };
        let inference_config = InferenceConfig {
            ids: InferenceIds {
                inference_id: Uuid::now_v7(),
                episode_id: Uuid::now_v7(),
            },
            templates: templates.clone(),
            tool_config: None,
            function_name: "".into(),
            variant_name: "".into(),
            dynamic_output_schema: None,
            fetch_and_encode_input_files_before_inference: false,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            extra_cache_key: None,
        };
        let chat_completion_config = UninitializedChatCompletionConfig {
            model: "json".into(),
            weight: Some(1.0),
            system_template: Some(system_template.clone()),
            user_template: Some(user_template.clone()),
            assistant_template: None,
            input_wrappers: None,
            extra_body: Default::default(),
            ..Default::default()
        }
        .load(
            &SchemaData::load(
                Some(test_user_template_schema()),
                None,
                Some(test_system_template_schema()),
                UninitializedSchemas::default(),
                "test",
            )
            .unwrap(),
            &ErrorContext {
                function_name: "test".to_string(),
                variant_name: "test".to_string(),
            },
        )
        .unwrap();
        let result = chat_completion_config
            .infer(
                Arc::new(input.clone()),
                inference_models.clone(),
                Arc::clone(&json_function_config),
                Arc::new(inference_config.clone()),
                clients.clone(),
                inference_params.clone(),
            )
            .await
            .unwrap();
        assert_eq!(
            result.usage_considering_cached(),
            Usage {
                input_tokens: Some(10),
                output_tokens: Some(1),
            }
        );
        match result {
            InferenceResult::Json(json_result) => {
                assert_eq!(json_result.output.parsed, Some(json!({"answer": "Hello"})));
                assert_eq!(
                    json_result.output.raw,
                    Some(DUMMY_JSON_RESPONSE_RAW.to_string())
                );
                assert_eq!(json_result.model_inference_results.len(), 1);
                assert_eq!(
                    json_result.model_inference_results[0].model_provider_name,
                    "json_provider".into()
                );
                assert_eq!(
                    json_result.model_inference_results[0].model_name,
                    "json".into()
                );
                assert_eq!(json_result.inference_params, inference_params);
            }
            InferenceResult::Chat(_) => panic!("Expected Json inference response"),
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
        let json_mode_tool_call_config =
            ToolCallConfig::implicit_from_value(&hardcoded_output_schema);
        let hardcoded_output_schema =
            StaticJSONSchema::from_value(hardcoded_output_schema).unwrap();
        let schema_any = StaticJSONSchema::from_value(json!({ "type": "object" })).unwrap();
        let json_function_config = Arc::new(FunctionConfig::Json(FunctionConfigJson {
            variants: HashMap::new(),
            schemas: SchemaData::load(
                Some(schema_any.clone()),
                Some(schema_any.clone()),
                Some(schema_any.clone()),
                UninitializedSchemas::default(),
                "test",
            )
            .unwrap(),
            output_schema: hardcoded_output_schema,
            json_mode_tool_call_config,
            description: None,
            all_explicit_template_names: HashSet::new(),
            experimentation: ExperimentationConfig::default(),
        }));
        let inference_params = InferenceParams {
            chat_completion: ChatCompletionInferenceParams {
                temperature: Some(0.5),
                max_tokens: Some(100),
                seed: Some(42),
                top_p: Some(0.9),
                presence_penalty: Some(0.1),
                frequency_penalty: Some(0.2),
                json_mode: None,
                stop_sequences: None,
                ..Default::default()
            },
        };
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
            ids: InferenceIds {
                inference_id: Uuid::now_v7(),
                episode_id: Uuid::now_v7(),
            },
            templates: templates.clone(),
            tool_config: None,
            function_name: "".into(),
            variant_name: "".into(),
            dynamic_output_schema: Some(Arc::new(output_schema)),
            fetch_and_encode_input_files_before_inference: false,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            extra_cache_key: None,
        };
        let chat_completion_config = UninitializedChatCompletionConfig {
            model: "json".into(),
            weight: Some(1.0),
            system_template: Some(system_template.clone()),
            user_template: Some(user_template.clone()),
            assistant_template: None,
            input_wrappers: None,

            ..Default::default()
        }
        .load(
            &SchemaData::load(
                Some(test_user_template_schema()),
                None,
                Some(test_system_template_schema()),
                UninitializedSchemas::default(),
                "test",
            )
            .unwrap(),
            &ErrorContext {
                function_name: "test".to_string(),
                variant_name: "test".to_string(),
            },
        )
        .unwrap();
        let result = chat_completion_config
            .infer(
                Arc::new(input.clone()),
                inference_models.clone(),
                Arc::clone(&json_function_config),
                Arc::new(inference_config.clone()),
                clients.clone(),
                inference_params.clone(),
            )
            .await
            .unwrap();
        assert_eq!(
            result.usage_considering_cached(),
            Usage {
                input_tokens: Some(10),
                output_tokens: Some(1),
            }
        );
        match result {
            InferenceResult::Json(json_result) => {
                assert_eq!(json_result.output.parsed, Some(json!({"answer": "Hello"})));
                assert_eq!(
                    json_result.output.raw,
                    Some(DUMMY_JSON_RESPONSE_RAW.to_string())
                );
                assert_eq!(json_result.model_inference_results.len(), 1);
                assert_eq!(
                    json_result.model_inference_results[0].model_provider_name,
                    "json_provider".into()
                );
                assert_eq!(
                    json_result.model_inference_results[0].model_name,
                    "json".into()
                );
                assert_eq!(json_result.inference_params, inference_params);
            }
            InferenceResult::Chat(_) => panic!("Expected Json inference response"),
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
        let json_mode_tool_call_config =
            ToolCallConfig::implicit_from_value(&hardcoded_output_schema);
        let hardcoded_output_schema =
            StaticJSONSchema::from_value(hardcoded_output_schema).unwrap();
        let schema_any = StaticJSONSchema::from_value(json!({ "type": "object" })).unwrap();
        let json_function_config = Arc::new(FunctionConfig::Json(FunctionConfigJson {
            variants: HashMap::new(),
            schemas: SchemaData::load(
                Some(schema_any.clone()),
                Some(schema_any.clone()),
                Some(schema_any.clone()),
                UninitializedSchemas::default(),
                "test",
            )
            .unwrap(),
            output_schema: hardcoded_output_schema,
            json_mode_tool_call_config,
            description: None,
            all_explicit_template_names: HashSet::new(),
            experimentation: ExperimentationConfig::default(),
        }));
        let inference_params = InferenceParams::default();
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
            ids: InferenceIds {
                inference_id: Uuid::now_v7(),
                episode_id: Uuid::now_v7(),
            },
            templates: templates.clone(),
            tool_config: None,
            function_name: "".into(),
            variant_name: "".into(),
            dynamic_output_schema: Some(Arc::new(output_schema)),
            fetch_and_encode_input_files_before_inference: false,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            extra_cache_key: None,
        };
        let chat_completion_config = UninitializedChatCompletionConfig {
            model: "json".into(),
            weight: Some(1.0),
            system_template: Some(system_template),
            user_template: Some(user_template),
            assistant_template: None,
            input_wrappers: None,
            temperature: Some(0.5),
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.2),
            max_tokens: Some(100),
            seed: Some(42),
            ..Default::default()
        }
        .load(
            &SchemaData::load(
                Some(test_user_template_schema()),
                None,
                Some(test_system_template_schema()),
                UninitializedSchemas::default(),
                "test",
            )
            .unwrap(),
            &ErrorContext {
                function_name: "test".to_string(),
                variant_name: "test".to_string(),
            },
        )
        .unwrap();
        let result = chat_completion_config
            .infer(
                Arc::new(input.clone()),
                inference_models.clone(),
                Arc::clone(&json_function_config),
                Arc::new(inference_config.clone()),
                clients.clone(),
                inference_params.clone(),
            )
            .await
            .unwrap();
        assert_eq!(
            result.usage_considering_cached(),
            Usage {
                input_tokens: Some(10),
                output_tokens: Some(1),
            }
        );
        match result {
            InferenceResult::Json(json_result) => {
                assert_eq!(json_result.output.parsed, None);
                assert_eq!(
                    json_result.output.raw,
                    Some(DUMMY_JSON_RESPONSE_RAW.to_string())
                );
                assert_eq!(json_result.model_inference_results.len(), 1);
                assert_eq!(
                    json_result.model_inference_results[0].model_provider_name,
                    "json_provider".into()
                );
                assert_eq!(
                    json_result.model_inference_results[0].model_name,
                    "json".into()
                );
                let expected_inference_params = InferenceParams {
                    chat_completion: ChatCompletionInferenceParams {
                        temperature: Some(0.5),
                        max_tokens: Some(100),
                        seed: Some(42),
                        top_p: Some(0.9),
                        presence_penalty: Some(0.1),
                        frequency_penalty: Some(0.2),
                        json_mode: None,
                        stop_sequences: None,
                        ..Default::default()
                    },
                };
                assert_eq!(json_result.inference_params, expected_inference_params);
            }
            InferenceResult::Chat(_) => panic!("Expected Json inference response"),
        }
    }

    #[tokio::test]
    async fn test_infer_chat_completion_stream() {
        let client = TensorzeroHttpClient::new_testing().unwrap();
        let clickhouse_connection_info = ClickHouseConnectionInfo::new_disabled();
        let api_keys = InferenceCredentials::default();
        let clients = InferenceClients {
            http_client: client.clone(),
            clickhouse_connection_info: clickhouse_connection_info.clone(),
            postgres_connection_info: PostgresConnectionInfo::Disabled,
            credentials: Arc::new(api_keys),
            cache_options: CacheOptions {
                max_age_s: None,
                enabled: CacheEnabledMode::WriteOnly,
            },
            tags: Arc::new(Default::default()),
            rate_limiting_config: Arc::new(Default::default()),
            otlp_config: Default::default(),
            deferred_tasks: tokio_util::task::TaskTracker::new(),
            scope_info: ScopeInfo {
                tags: Arc::new(HashMap::new()),
                api_key_public_id: None,
            },
        };
        let templates = Box::leak(Box::new(get_test_template_config().await));
        let schema_any = StaticJSONSchema::from_value(json!({ "type": "object" })).unwrap();
        let function_config = Arc::new(FunctionConfig::Chat(FunctionConfigChat {
            variants: HashMap::new(),
            schemas: SchemaData::load(
                Some(schema_any.clone()),
                Some(schema_any.clone()),
                Some(schema_any.clone()),
                UninitializedSchemas::default(),
                "test",
            )
            .unwrap(),
            tools: vec![],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: None,
            description: None,
            all_explicit_templates_names: HashSet::new(),
            experimentation: ExperimentationConfig::default(),
        }));

        let system_template = get_system_template();
        let user_template = get_greeting_with_age_template();
        let good_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "good".into(),
            ..Default::default()
        });
        let error_provider_config = ProviderConfig::Dummy(DummyProvider {
            model_name: "error".into(),
            ..Default::default()
        });
        let text_model_config = ModelConfig {
            routing: vec!["good_provider".into()],
            providers: HashMap::from([(
                "good_provider".into(),
                ModelProvider {
                    name: "good_provider".into(),
                    config: good_provider_config,
                    extra_body: Default::default(),
                    extra_headers: Default::default(),
                    timeouts: Default::default(),
                    discard_unknown_chunks: false,
                },
            )]),
            timeouts: Default::default(),
        };
        let error_model_config = ModelConfig {
            routing: vec!["error_provider".into()],
            providers: HashMap::from([(
                "error_provider".into(),
                ModelProvider {
                    name: "error_provider".into(),
                    config: error_provider_config,
                    extra_body: Default::default(),
                    extra_headers: Default::default(),
                    timeouts: Default::default(),
                    discard_unknown_chunks: false,
                },
            )]),
            timeouts: Default::default(),
        };
        // Test case 1: Model inference fails because of model issues
        let inference_params = InferenceParams::default();
        let messages = vec![LazyResolvedInputMessage {
            role: Role::User,
            content: vec![LazyResolvedInputMessageContent::Template(Template {
                name: "user".to_string(),
                arguments: Arguments(
                    json!({"name": "Luke", "age": 20})
                        .as_object()
                        .unwrap()
                        .clone(),
                ),
            })],
        }];
        let input = LazyResolvedInput {
            system: Some(System::Template(Arguments(
                json!({"assistant_name": "R2-D2"})
                    .as_object()
                    .unwrap()
                    .clone(),
            ))),
            messages,
        };
        let chat_completion_config = Box::leak(Box::new(
            UninitializedChatCompletionConfig {
                model: "error".into(),
                weight: Some(1.0),
                system_template: Some(system_template.clone()),
                user_template: Some(user_template.clone()),
                assistant_template: None,
                input_wrappers: None,

                ..Default::default()
            }
            .load(
                &SchemaData::load(
                    Some(test_user_template_schema()),
                    None,
                    Some(test_system_template_schema()),
                    UninitializedSchemas::default(),
                    "test",
                )
                .unwrap(),
                &ErrorContext {
                    function_name: "test".to_string(),
                    variant_name: "test".to_string(),
                },
            )
            .unwrap(),
        ));
        let provider_types = Box::leak(Box::new(ProviderTypesConfig::default()));
        let models = Arc::new(
            ModelTable::new(
                HashMap::from([("error".into(), error_model_config)]),
                ProviderTypeDefaultCredentials::new(provider_types).into(),
                chrono::Duration::seconds(120),
            )
            .unwrap(),
        );
        let embedding_models = Arc::new(
            EmbeddingModelTable::new(
                HashMap::new(),
                ProviderTypeDefaultCredentials::new(provider_types).into(),
                chrono::Duration::seconds(120),
            )
            .unwrap(),
        );
        let inference_models = InferenceModels {
            models,
            embedding_models: embedding_models.clone(),
        };
        let inference_config = InferenceConfig {
            ids: InferenceIds {
                inference_id: Uuid::now_v7(),
                episode_id: Uuid::now_v7(),
            },
            templates: Arc::new(templates.clone()),
            tool_config: None,
            dynamic_output_schema: None,
            function_name: "".into(),
            variant_name: "".into(),
            fetch_and_encode_input_files_before_inference: false,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            extra_cache_key: None,
        };
        let result = chat_completion_config
            .infer_stream(
                Arc::new(input.clone()),
                inference_models.clone(),
                Arc::clone(&function_config),
                Arc::new(inference_config.clone()),
                clients.clone(),
                inference_params.clone(),
            )
            .await;
        let err = match result {
            Ok(_) => panic!("Expected error"),
            Err(e) => e,
        };
        match err.get_details() {
            ErrorDetails::ModelProvidersExhausted {
                provider_errors, ..
            } => {
                assert_eq!(provider_errors.len(), 1);
                assert!(matches!(
                    provider_errors["error_provider"].get_details(),
                    ErrorDetails::InferenceClient { .. }
                ));
            }
            _ => panic!("Expected ModelProvidersExhausted error, got {err:?}"),
        }

        // Test case 2: Model inference succeeds
        let inference_params = InferenceParams::default();
        let chat_completion_config = UninitializedChatCompletionConfig {
            model: "good".into(),
            weight: Some(1.0),
            system_template: Some(system_template),
            user_template: Some(user_template),
            assistant_template: None,
            input_wrappers: None,

            ..Default::default()
        }
        .load(
            &SchemaData::load(
                Some(test_user_template_schema()),
                None,
                Some(test_system_template_schema()),
                UninitializedSchemas::default(),
                "test",
            )
            .unwrap(),
            &ErrorContext {
                function_name: "test".to_string(),
                variant_name: "test".to_string(),
            },
        )
        .unwrap();
        let provider_types = Box::leak(Box::new(ProviderTypesConfig::default()));
        let models = Arc::new(
            ModelTable::new(
                HashMap::from([("good".into(), text_model_config)]),
                ProviderTypeDefaultCredentials::new(provider_types).into(),
                chrono::Duration::seconds(120),
            )
            .unwrap(),
        );
        let inference_models = InferenceModels {
            models,
            embedding_models: embedding_models.clone(),
        };
        let inference_config = InferenceConfig {
            ids: InferenceIds {
                inference_id: Uuid::now_v7(),
                episode_id: Uuid::now_v7(),
            },
            templates: Arc::new(templates.clone()),
            tool_config: None,
            function_name: "".into(),
            variant_name: "".into(),
            dynamic_output_schema: None,
            fetch_and_encode_input_files_before_inference: false,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            extra_cache_key: None,
        };
        let (mut stream, models_used) = chat_completion_config
            .infer_stream(
                Arc::new(input.clone()),
                inference_models.clone(),
                Arc::clone(&function_config),
                Arc::new(inference_config.clone()),
                clients.clone(),
                inference_params.clone(),
            )
            .await
            .unwrap();
        let first_chunk = match stream.next().await.unwrap().unwrap() {
            InferenceResultChunk::Chat(chunk) => chunk,
            InferenceResultChunk::Json(_) => panic!("Expected Chat inference response"),
        };
        assert_eq!(
            first_chunk.content,
            vec![ContentBlockChunk::Text(TextChunk {
                text: DUMMY_STREAMING_RESPONSE[0].to_string(),
                id: "0".to_string()
            })]
        );
        assert_eq!(&*models_used.model_name, "good".to_string());
        assert_eq!(
            &*models_used.model_provider_name,
            "good_provider".to_string()
        );
        let mut i = 1;
        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.unwrap();
            if i == 16 {
                // max length of text, but we have a usage chunk left
                assert_eq!(
                    chunk.usage(),
                    Some(&Usage {
                        input_tokens: Some(10),
                        output_tokens: Some(16),
                    })
                );
                break;
            }
            let chunk = match chunk {
                InferenceResultChunk::Chat(chunk) => chunk,
                InferenceResultChunk::Json(_) => panic!("Expected Chat inference response"),
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
        let input = LazyResolvedInput {
            system: None,
            messages: vec![],
        };
        let templates = Box::leak(Box::new(get_test_template_config().await));
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
            schemas: SchemaData::load(None, None, None, UninitializedSchemas::default(), "test")
                .unwrap(),
            tools: vec![],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: None,
            description: None,
            all_explicit_templates_names: HashSet::new(),
            experimentation: ExperimentationConfig::default(),
        });
        let mut inference_params = InferenceParams::default();
        let inference_config = InferenceConfig {
            ids: InferenceIds {
                inference_id: Uuid::now_v7(),
                episode_id: Uuid::now_v7(),
            },
            templates: Arc::new(templates.clone()),
            tool_config: None,
            function_name: "".into(),
            variant_name: "".into(),
            dynamic_output_schema: None,
            fetch_and_encode_input_files_before_inference: false,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            extra_cache_key: None,
        };
        let inference_config_arc = Arc::new(inference_config);
        let model_request = chat_completion_config
            .prepare_request(
                &input,
                &function_config,
                &inference_config_arc,
                stream,
                &mut inference_params,
            )
            .await
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
                top_p: Some(0.9),
                presence_penalty: Some(0.1),
                frequency_penalty: Some(0.2),
                json_mode: None,
                stop_sequences: None,
                ..Default::default()
            },
        };
        let model_request = chat_completion_config
            .prepare_request(
                &input,
                &function_config,
                &inference_config_arc,
                stream,
                &mut inference_params,
            )
            .await
            .unwrap();
        assert_eq!(model_request.temperature, Some(1.));
        assert_eq!(model_request.max_tokens, Some(200));
        assert_eq!(model_request.seed, Some(420));
        assert_eq!(model_request.top_p, Some(0.9));
        assert_eq!(model_request.presence_penalty, Some(0.1));
        assert_eq!(model_request.frequency_penalty, Some(0.2));
        assert_eq!(inference_params.chat_completion.temperature, Some(1.));
        assert_eq!(inference_params.chat_completion.max_tokens, Some(200));
        assert_eq!(inference_params.chat_completion.seed, Some(420));
        assert_eq!(inference_params.chat_completion.top_p, Some(0.9));
        assert_eq!(inference_params.chat_completion.presence_penalty, Some(0.1));
        assert_eq!(
            inference_params.chat_completion.frequency_penalty,
            Some(0.2)
        );

        // We will vary temperature, max_tokens, and seed
        let chat_completion_config = ChatCompletionConfig {
            temperature: Some(0.5),
            max_tokens: Some(100),
            seed: Some(69),
            top_p: Some(0.9),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.2),
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
            schemas: SchemaData::load(None, None, None, UninitializedSchemas::default(), "test")
                .unwrap(),
            output_schema: StaticJSONSchema::from_value(output_schema_value.clone()).unwrap(),
            ..Default::default()
        });
        let inference_config = InferenceConfig {
            ids: InferenceIds {
                inference_id: Uuid::now_v7(),
                episode_id: Uuid::now_v7(),
            },
            templates: Arc::new(templates.clone()),
            tool_config: None,
            dynamic_output_schema: None,
            function_name: "".into(),
            variant_name: "".into(),
            fetch_and_encode_input_files_before_inference: false,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            extra_cache_key: None,
        };
        let inference_config_arc = Arc::new(inference_config);
        let mut inference_params = InferenceParams::default();
        let model_request = chat_completion_config
            .prepare_request(
                &input,
                &function_config,
                &inference_config_arc,
                stream,
                &mut inference_params,
            )
            .await
            .unwrap();
        assert_eq!(model_request.temperature, Some(0.5));
        assert_eq!(model_request.max_tokens, Some(100));
        assert_eq!(model_request.seed, Some(69));
        assert_eq!(model_request.top_p, Some(0.9));
        assert_eq!(model_request.presence_penalty, Some(0.1));
        assert_eq!(model_request.frequency_penalty, Some(0.2));
        assert_eq!(
            model_request.json_mode,
            ModelInferenceRequestJsonMode::Strict
        );
        assert_eq!(model_request.output_schema, Some(&output_schema_value));
        assert_eq!(inference_params.chat_completion.temperature, Some(0.5));
        assert_eq!(inference_params.chat_completion.max_tokens, Some(100));
        assert_eq!(inference_params.chat_completion.seed, Some(69));
        assert_eq!(inference_params.chat_completion.top_p, Some(0.9));
        assert_eq!(inference_params.chat_completion.presence_penalty, Some(0.1));
        assert_eq!(
            inference_params.chat_completion.frequency_penalty,
            Some(0.2)
        );

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
                &inference_config_arc,
                stream,
                &mut inference_params,
            )
            .await
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
            templates: Arc::new(templates.clone()),
            tool_config: None,
            dynamic_output_schema: Some(Arc::new(dynamic_output_schema)),
            function_name: "".into(),
            variant_name: "".into(),
            ids: InferenceIds {
                inference_id: Uuid::now_v7(),
                episode_id: Uuid::now_v7(),
            },
            fetch_and_encode_input_files_before_inference: false,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            extra_cache_key: None,
        };
        let inference_config_arc = Arc::new(inference_config);
        let model_request = chat_completion_config
            .prepare_request(
                &input,
                &function_config,
                &inference_config_arc,
                stream,
                &mut inference_params,
            )
            .await
            .unwrap();
        assert_eq!(
            model_request.output_schema,
            Some(&dynamic_output_schema_value)
        );
    }

    #[tokio::test]
    async fn test_validate_template_and_schema_both_none() {
        let templates = get_test_template_config().await;
        let result =
            validate_legacy_template_and_schema(TemplateKind::System, None, None, &templates);
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_validate_template_and_schema_both_some() {
        let templates = get_test_template_config().await;
        let schema = StaticJSONSchema::from_path(ResolvedTomlPathData::new_for_tests(
            "fixtures/config/functions/templates_with_variables/system_schema.json".into(),
            None,
        ))
        .unwrap();
        let template = PathBuf::from("test_validate_template_and_schema_both_some");
        validate_legacy_template_and_schema(
            TemplateKind::System,
            Some(&schema),
            Some(&TemplateWithSchema {
                template: PathWithContents::from_path(ResolvedTomlPathData::new_for_tests(
                    template,
                    Some("fake_data".to_string()),
                ))
                .unwrap(),
                schema: Some(schema.clone()),
                legacy_definition: false,
            }),
            &templates,
        )
        .unwrap();
    }

    #[tokio::test]
    async fn test_validate_template_and_schema_template_no_needs_variables() {
        let templates = get_test_template_config().await;
        let template = PathBuf::from("system_filled");
        let result = validate_legacy_template_and_schema(
            TemplateKind::System,
            None,
            Some(&TemplateWithSchema {
                template: PathWithContents::from_path(ResolvedTomlPathData::new_for_tests(
                    template,
                    Some("fake_data".to_string()),
                ))
                .unwrap(),
                schema: None,
                legacy_definition: false,
            }),
            &templates,
        );
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_validate_template_and_schema_template_needs_variables() {
        let templates = get_test_template_config().await; // Template needing variables
        let template = PathBuf::from("greeting");
        let err = validate_legacy_template_and_schema(
            TemplateKind::System,
            None,
            Some(&TemplateWithSchema {
                template: PathWithContents::from_path(ResolvedTomlPathData::new_for_tests(
                    template,
                    Some("fake_data".to_string()),
                ))
                .unwrap(),
                schema: None,
                legacy_definition: true,
            }),
            &templates,
        )
        .unwrap_err();
        let details = err.get_details();

        if let ErrorDetails::Config { message } = details {
            assert_eq!(
                *message,
                "template needs variables: [name] but only `system_text` is allowed when template has no schema".to_string()
            );
        } else {
            panic!("Expected Error::Config");
        }
    }

    #[tokio::test]
    async fn test_validate_template_and_schema_schema_some_template_none() {
        let templates = get_test_template_config().await; // Default TemplateConfig
        let schema = StaticJSONSchema::from_path(ResolvedTomlPathData::new_for_tests(
            "fixtures/config/functions/templates_with_variables/system_schema.json".into(),
            None,
        ))
        .unwrap();
        let err = validate_legacy_template_and_schema(
            TemplateKind::System,
            Some(&schema),
            None,
            &templates,
        )
        .unwrap_err();
        let details = err.get_details();

        if let ErrorDetails::Config { message } = details {
            assert_eq!(
                *message,
                "template is required when schema is specified".to_string()
            );
        } else {
            panic!("Expected Error::Config");
        }
    }

    #[test]
    fn test_as_uninitialized_preserves_basic_fields() {
        let uninitialized = UninitializedChatCompletionConfig {
            model: "gpt-4".into(),
            weight: Some(0.8),
            temperature: Some(0.7),
            top_p: Some(0.9),
            max_tokens: Some(150),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.2),
            seed: Some(42),
            stop_sequences: Some(vec!["STOP".to_string(), "END".to_string()]),
            ..Default::default()
        };

        let config = uninitialized
            .load(&SchemaData::default(), &ErrorContext::new_test())
            .unwrap();

        let exported = config.as_uninitialized();

        assert_eq!(exported.model, "gpt-4".into());
        assert_eq!(exported.weight, Some(0.8));
        assert_eq!(exported.temperature, Some(0.7));
        assert_eq!(exported.top_p, Some(0.9));
        assert_eq!(exported.max_tokens, Some(150));
        assert_eq!(exported.presence_penalty, Some(0.1));
        assert_eq!(exported.frequency_penalty, Some(0.2));
        assert_eq!(exported.seed, Some(42));
        assert_eq!(
            exported.stop_sequences,
            Some(vec!["STOP".to_string(), "END".to_string()])
        );
    }

    #[test]
    fn test_as_uninitialized_preserves_inference_params_v2() {
        let uninitialized = UninitializedChatCompletionConfig {
            model: "gpt-4".into(),
            reasoning_effort: Some("high".to_string()),
            service_tier: Some(ServiceTier::Auto),
            thinking_budget_tokens: Some(1000),
            verbosity: Some("verbose".to_string()),
            ..Default::default()
        };

        let config = uninitialized
            .load(&SchemaData::default(), &ErrorContext::new_test())
            .unwrap();

        let exported = config.as_uninitialized();

        assert_eq!(exported.reasoning_effort, Some("high".to_string()));
        assert_eq!(exported.service_tier, Some(ServiceTier::Auto));
        assert_eq!(exported.thinking_budget_tokens, Some(1000));
        assert_eq!(exported.verbosity, Some("verbose".to_string()));
    }

    #[test]
    fn test_as_uninitialized_preserves_none_values() {
        let uninitialized = UninitializedChatCompletionConfig {
            model: "gpt-4".into(),
            weight: None,
            temperature: None,
            top_p: None,
            max_tokens: None,
            stop_sequences: None,
            reasoning_effort: None,
            service_tier: None,
            thinking_budget_tokens: None,
            verbosity: None,
            ..Default::default()
        };

        let config = uninitialized
            .load(&SchemaData::default(), &ErrorContext::new_test())
            .unwrap();

        let exported = config.as_uninitialized();

        assert_eq!(exported.weight, None);
        assert_eq!(exported.temperature, None);
        assert_eq!(exported.top_p, None);
        assert_eq!(exported.max_tokens, None);
        assert_eq!(exported.stop_sequences, None);
        assert_eq!(exported.reasoning_effort, None);
        assert_eq!(exported.service_tier, None);
        assert_eq!(exported.thinking_budget_tokens, None);
        assert_eq!(exported.verbosity, None);
    }

    #[test]
    fn test_as_uninitialized_serialization_round_trip() {
        let original = UninitializedChatCompletionConfig {
            model: "gpt-4".into(),
            weight: Some(0.5),
            temperature: Some(0.9),
            max_tokens: Some(100),
            seed: Some(123),
            ..Default::default()
        };

        let config = original
            .clone()
            .load(&SchemaData::default(), &ErrorContext::new_test())
            .unwrap();

        let exported = config.as_uninitialized();

        // Serialize and deserialize
        let json = serde_json::to_string(&exported).unwrap();
        let deserialized: UninitializedChatCompletionConfig = serde_json::from_str(&json).unwrap();

        // Should be able to load again
        let reloaded = deserialized
            .load(&SchemaData::default(), &ErrorContext::new_test())
            .unwrap();

        assert_eq!(reloaded.model(), &Arc::from("gpt-4"));
        assert_eq!(reloaded.weight(), Some(0.5));
        assert_eq!(reloaded.temperature(), Some(0.9));
        assert_eq!(reloaded.max_tokens(), Some(100));
        assert_eq!(reloaded.seed(), Some(123));
    }
}
