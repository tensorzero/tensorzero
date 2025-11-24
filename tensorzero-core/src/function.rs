use crate::config::SchemaData;
#[cfg(feature = "pyo3")]
use crate::error::IMPOSSIBLE_ERROR_MESSAGE;
use crate::experimentation::ExperimentationConfig;
#[cfg(feature = "pyo3")]
use crate::inference::types::pyo3_helpers::serialize_to_dict;
#[cfg(feature = "pyo3")]
use crate::variant::{
    BestOfNSamplingConfigPyClass, ChainOfThoughtConfigPyClass, ChatCompletionConfigPyClass,
    DiclConfigPyClass, MixtureOfNConfigPyClass, VariantConfig,
};
use chrono::Duration;
#[cfg(feature = "pyo3")]
use pyo3::exceptions::{PyKeyError, PyValueError};
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
#[cfg(feature = "pyo3")]
use pyo3::IntoPyObjectExt;
use serde::Serialize;
use serde_json::Value;
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tracing::instrument;
use uuid::Uuid;

use crate::embeddings::EmbeddingModelTable;
use crate::endpoints::inference::InferenceParams;
use crate::error::{Error, ErrorDetails};
use crate::inference::types::{
    ChatInferenceResult, ContentBlockOutput, InferenceResult, Input, InputMessageContent,
    JsonInferenceResult, ModelInferenceResponseWithMetadata, Role, System,
};
use crate::jsonschema_util::{JsonSchemaRef, StaticJSONSchema};
use crate::minijinja_util::TemplateConfig;
use crate::model::ModelTable;
use crate::tool::{
    DynamicToolParams, StaticToolConfig, ToolCallConfig, ToolCallConfigConstructorArgs,
    ToolCallConfigDatabaseInsert, ToolChoice,
};
use crate::variant::{InferenceConfig, JsonMode, Variant, VariantInfo};

#[derive(Debug, Serialize, ts_rs::TS)]
#[ts(export)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum FunctionConfig {
    Chat(FunctionConfigChat),
    Json(FunctionConfigJson),
}

#[cfg(feature = "pyo3")]
#[pyclass(str, name = "FunctionConfigChat")]
pub struct FunctionConfigChatPyClass {
    pub inner: Arc<FunctionConfig>,
}

#[cfg(feature = "pyo3")]
impl std::fmt::Display for FunctionConfigChatPyClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(&self.inner).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

#[cfg(feature = "pyo3")]
#[pyclass(str, name = "FunctionConfigJson")]
pub struct FunctionConfigJsonPyClass {
    pub inner: Arc<FunctionConfig>,
}

#[cfg(feature = "pyo3")]
impl std::fmt::Display for FunctionConfigJsonPyClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(&self.inner).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "pyo3", pyclass)]
pub enum FunctionConfigType {
    Chat,
    Json,
}

impl FunctionConfig {
    pub fn config_type(&self) -> FunctionConfigType {
        match self {
            FunctionConfig::Chat(_) => FunctionConfigType::Chat,
            FunctionConfig::Json(_) => FunctionConfigType::Json,
        }
    }

    pub fn table_name(&self) -> &str {
        match self {
            FunctionConfig::Chat(_) => "ChatInference",
            FunctionConfig::Json(_) => "JsonInference",
        }
    }

    pub fn experimentation(&self) -> &ExperimentationConfig {
        match self {
            FunctionConfig::Chat(config) => &config.experimentation,
            FunctionConfig::Json(config) => &config.experimentation,
        }
    }

    pub fn tools(&self) -> Box<dyn Iterator<Item = &str> + '_> {
        match self {
            FunctionConfig::Chat(config) => Box::new(config.tools.iter().map(String::as_str)),
            FunctionConfig::Json(_config) => Box::new(std::iter::empty()),
        }
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl FunctionConfigChatPyClass {
    #[getter]
    fn get_type(&self) -> FunctionConfigType {
        self.inner.config_type()
    }

    #[getter]
    fn get_variants(&self) -> VariantsConfigPyClass {
        VariantsConfigPyClass {
            inner: self.inner.variants().clone(),
        }
    }

    #[getter]
    fn get_system_schema(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.inner
            .system_schema()
            .map(|s| serialize_to_dict(py, &s.value))
            .transpose()?
            .into_py_any(py)
    }
    #[getter]
    fn get_user_schema(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.inner
            .user_schema()
            .map(|s| serialize_to_dict(py, &s.value))
            .transpose()?
            .into_py_any(py)
    }

    #[getter]
    fn get_assistant_schema(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.inner
            .assistant_schema()
            .map(|s| serialize_to_dict(py, &s.value))
            .transpose()?
            .into_py_any(py)
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl FunctionConfigJsonPyClass {
    #[getter]
    fn get_type(&self) -> FunctionConfigType {
        self.inner.config_type()
    }

    #[getter]
    fn get_variants(&self) -> VariantsConfigPyClass {
        VariantsConfigPyClass {
            inner: self.inner.variants().clone(),
        }
    }

    #[getter]
    fn get_system_schema(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.inner
            .system_schema()
            .map(|s| serialize_to_dict(py, &s.value))
            .transpose()?
            .into_py_any(py)
    }

    #[getter]
    fn get_user_schema(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.inner
            .user_schema()
            .map(|s| serialize_to_dict(py, &s.value))
            .transpose()?
            .into_py_any(py)
    }

    #[getter]
    fn get_assistant_schema(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.inner
            .assistant_schema()
            .map(|s| serialize_to_dict(py, &s.value))
            .transpose()?
            .into_py_any(py)
    }

    #[getter]
    fn get_output_schema(&self, py: Python) -> PyResult<Py<PyAny>> {
        let FunctionConfig::Json(params) = &*self.inner else {
            return Err(PyValueError::new_err(format!(
                "FunctionConfig is not a JSON function: {IMPOSSIBLE_ERROR_MESSAGE}"
            )));
        };
        serialize_to_dict(py, &params.output_schema.value)
    }
}

#[cfg(feature = "pyo3")]
#[pyclass(mapping, name = "VariantsConfig")]
pub struct VariantsConfigPyClass {
    pub inner: HashMap<String, Arc<VariantInfo>>,
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl VariantsConfigPyClass {
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __getitem__<'py>(&self, py: Python<'py>, key: &str) -> PyResult<Bound<'py, PyAny>> {
        let v = self
            .inner
            .get(key)
            .cloned()
            .ok_or_else(|| PyKeyError::new_err(key.to_string()))?;
        match &v.inner {
            VariantConfig::ChatCompletion(_) => {
                ChatCompletionConfigPyClass { inner: v }.into_bound_py_any(py)
            }
            VariantConfig::BestOfNSampling(_) => {
                BestOfNSamplingConfigPyClass { inner: v }.into_bound_py_any(py)
            }
            VariantConfig::Dicl(_) => DiclConfigPyClass { inner: v }.into_bound_py_any(py),
            VariantConfig::MixtureOfN(_) => {
                MixtureOfNConfigPyClass { inner: v }.into_bound_py_any(py)
            }
            VariantConfig::ChainOfThought(_) => {
                ChainOfThoughtConfigPyClass { inner: v }.into_bound_py_any(py)
            }
        }
    }
}

#[derive(Debug, Default, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct FunctionConfigChat {
    pub variants: HashMap<String, Arc<VariantInfo>>, // variant name => variant config
    pub schemas: SchemaData,
    pub tools: Vec<String>, // tool names
    pub tool_choice: ToolChoice,
    pub parallel_tool_calls: Option<bool>,
    pub description: Option<String>,
    pub experimentation: ExperimentationConfig,
    // Holds all template names (e.g. 'user', 'my_custom_template'
    // which can be invoked through a `{"type": "template", "name": "..."}` input block)
    // This is used to perform early rejection of a template invocation,
    // in the case where all variants either:
    // * do not have the template defined at all, or
    // * define the template as an old-style input wrapper
    //   (which can only be invoked by a {`"type": "text", "text": "..."`} input block)
    //
    // If it least one variant defines the template as a named template (non legacy-input-wrapper),
    // then its name will be included in this set, and we'll let the request go through.
    // The early rejection logic improves error messages in the case where every variant invocation
    // is guaranteed to fail - we avoid showing an 'All variants failed' error message with
    // the same template error for every variant.
    #[serde(skip)]
    pub all_explicit_templates_names: HashSet<String>,
}

#[derive(Debug, Default, Serialize, ts_rs::TS)]
#[ts(export)]
pub struct FunctionConfigJson {
    pub variants: HashMap<String, Arc<VariantInfo>>, // variant name => variant config
    pub schemas: SchemaData,
    pub output_schema: StaticJSONSchema, // schema is mandatory for JSON functions
    pub json_mode_tool_call_config: ToolCallConfig,
    pub description: Option<String>,
    pub experimentation: ExperimentationConfig,
    // See `FunctionConfigChat.all_explicit_template_names`.
    #[serde(skip)]
    pub all_explicit_template_names: HashSet<String>,
}

impl FunctionConfig {
    pub fn variants(&self) -> &HashMap<String, Arc<VariantInfo>> {
        match self {
            FunctionConfig::Chat(params) => &params.variants,
            FunctionConfig::Json(params) => &params.variants,
        }
    }

    pub fn validate_inference_params(
        &self,
        params: &crate::endpoints::inference::Params,
    ) -> Result<(), Error> {
        if let FunctionConfig::Chat(chat_config) = self {
            if let Some(JsonMode::Tool) = &params.params.chat_completion.json_mode {
                // Check if the chat function has tools configured
                if !chat_config.tools.is_empty() {
                    return Err(ErrorDetails::InvalidRequest {
                        message: "JSON mode `tool` is not supported with other tools configured."
                            .to_string(),
                    }
                    .into());
                }

                // Check if the chat function has tool_choice configured (not Auto, which is the default)
                if !matches!(chat_config.tool_choice, ToolChoice::Auto) {
                    return Err(ErrorDetails::InvalidRequest {
                        message: "JSON mode `tool` is not supported with `tool_choice` configured in the function.".to_string(),
                    }
                    .into());
                }

                // Check if the chat function has parallel_tool_calls configured
                if chat_config.parallel_tool_calls.is_some() {
                    return Err(ErrorDetails::InvalidRequest {
                        message: "JSON mode `tool` is not supported with `parallel_tool_calls` configured in the function.".to_string(),
                    }
                    .into());
                }

                // Require output_schema when using `json_mode="tool"`
                if params.output_schema.is_none() {
                    return Err(ErrorDetails::InvalidRequest {
                        message: "JSON mode `tool` requires `output_schema`.".to_string(),
                    }
                    .into());
                }

                // Reject dynamic tool params when using `json_mode="tool"` (similar to JSON functions)
                let DynamicToolParams {
                    ref allowed_tools,
                    ref additional_tools,
                    ref parallel_tool_calls,
                    ref provider_tools,
                    ref tool_choice,
                } = params.dynamic_tool_params;

                if allowed_tools.is_some() {
                    return Err(ErrorDetails::InvalidRequest {
                        message: "Cannot pass `allowed_tools` when using JSON mode `tool`."
                            .to_string(),
                    }
                    .into());
                }
                if additional_tools.is_some() {
                    return Err(ErrorDetails::InvalidRequest {
                        message: "Cannot pass `additional_tools` when using JSON mode `tool`."
                            .to_string(),
                    }
                    .into());
                }
                if parallel_tool_calls.is_some() {
                    return Err(ErrorDetails::InvalidRequest {
                        message: "Cannot pass `parallel_tool_calls` when using JSON mode `tool`."
                            .to_string(),
                    }
                    .into());
                }
                if !provider_tools.is_empty() {
                    return Err(ErrorDetails::InvalidRequest {
                        message: "Cannot pass `provider_tools` when using JSON mode `tool`."
                            .to_string(),
                    }
                    .into());
                }
                if tool_choice.is_some() {
                    return Err(ErrorDetails::InvalidRequest {
                        message: "Cannot pass `tool_choice` when using JSON mode `tool`."
                            .to_string(),
                    }
                    .into());
                }
            }
        }
        self.validate_input(&params.input)
    }
    /// Validate the input against the function's input schemas.
    /// The validation is done based on the function's type:
    /// - For a chat function, the input is validated against the system, user, and assistant schemas.
    /// - For a JSON function, the input is validated against the system, user, and assistant schemas.
    ///
    /// We do not validate ContentBlocks that are not text (tool calls and tool responses).
    pub fn validate_input(&self, input: &Input) -> Result<(), Error> {
        match &self {
            FunctionConfig::Chat(params) => {
                validate_all_text_input(
                    &params.schemas,
                    input,
                    &params.all_explicit_templates_names,
                )?;
            }
            FunctionConfig::Json(params) => {
                validate_all_text_input(
                    &params.schemas,
                    input,
                    &params.all_explicit_template_names,
                )?;
            }
        }
        Ok(())
    }

    /// Prepare the tool config for the function.
    /// For a Chat function, this will incorporate the tool information configured in the function as
    /// well as the dynamic tool calling information passed in `dynamic_tool_params`.
    /// JSON functions do not get tool_configs even if they end up using tools under the hood.
    pub fn prepare_tool_config(
        &self,
        dynamic_tool_params: DynamicToolParams,
        static_tools: &HashMap<String, Arc<StaticToolConfig>>,
    ) -> Result<Option<ToolCallConfig>, Error> {
        match self {
            FunctionConfig::Chat(params) => {
                let DynamicToolParams {
                    allowed_tools,
                    additional_tools,
                    parallel_tool_calls,
                    provider_tools,
                    tool_choice,
                } = dynamic_tool_params;
                Ok(ToolCallConfig::new(ToolCallConfigConstructorArgs {
                    function_tools: &params.tools,
                    function_tool_choice: &params.tool_choice,
                    function_parallel_tool_calls: params.parallel_tool_calls,
                    static_tools,
                    dynamic_allowed_tools: allowed_tools,
                    dynamic_additional_tools: additional_tools,
                    dynamic_tool_choice: tool_choice,
                    dynamic_parallel_tool_calls: parallel_tool_calls,
                    dynamic_provider_tools: provider_tools,
                })?)
            }
            FunctionConfig::Json(_) => {
                if dynamic_tool_params.allowed_tools.is_some() {
                    return Err(ErrorDetails::InvalidRequest {
                        message: "Cannot pass `allowed_tools` to a JSON function.".to_string(),
                    }
                    .into());
                }
                if dynamic_tool_params.additional_tools.is_some() {
                    return Err(ErrorDetails::InvalidRequest {
                        message: "Cannot pass `additional_tools` to a JSON function.".to_string(),
                    }
                    .into());
                }
                if dynamic_tool_params.tool_choice.is_some() {
                    return Err(ErrorDetails::InvalidRequest {
                        message: "Cannot pass `tool_choice` to a JSON function".to_string(),
                    }
                    .into());
                }
                if dynamic_tool_params.parallel_tool_calls.is_some() {
                    return Err(ErrorDetails::InvalidRequest {
                        message: "Cannot pass `parallel_tool_calls` to a JSON function".to_string(),
                    }
                    .into());
                }
                Ok(None)
            }
        }
    }

    /// Convert DynamicToolParams to ToolCallConfigDatabaseInsert using prepare_tool_config
    /// This properly merges static and dynamic tools according to function configuration
    pub fn dynamic_tool_params_to_database_insert(
        &self,
        dynamic_params: DynamicToolParams,
        static_tools: &HashMap<String, Arc<StaticToolConfig>>,
    ) -> Result<Option<ToolCallConfigDatabaseInsert>, Error> {
        let tool_config = self.prepare_tool_config(dynamic_params, static_tools)?;
        Ok(tool_config.map(std::convert::Into::into))
    }

    #[instrument(skip_all, fields(inference_id))]
    pub async fn prepare_response(
        &self,
        inference_id: Uuid,
        content_blocks: Vec<ContentBlockOutput>,
        model_inference_results: Vec<ModelInferenceResponseWithMetadata>,
        inference_config: &InferenceConfig,
        inference_params: InferenceParams,
        original_response: Option<String>,
    ) -> Result<InferenceResult, Error> {
        match self {
            FunctionConfig::Chat(..) => {
                // Extract json_mode to pass to ChatInferenceResult
                let json_mode = inference_params.chat_completion.json_mode;
                Ok(InferenceResult::Chat(
                    ChatInferenceResult::new(
                        inference_id,
                        content_blocks,
                        model_inference_results,
                        inference_config.tool_config.as_deref(),
                        inference_params,
                        original_response,
                        json_mode,
                    )
                    .await,
                ))
            }
            FunctionConfig::Json(params) => {
                let (raw_output, auxiliary_content, json_block_index) =
                    get_json_output_from_content_blocks(content_blocks);

                // Try to parse the raw output as JSON.
                //
                // If the raw output is None, parsed output is also None.
                // If the raw output is not a valid JSON string, log an error and set parsed output to None.
                let parsed_output: Option<Value> = raw_output.as_ref().and_then(|raw_output| {
                    serde_json::from_str::<Value>(raw_output)
                        .map_err(|e| {
                            Error::new(ErrorDetails::OutputParsing {
                                message: format!(
                                    "Failed to parse output from JSON function response {e}",
                                ),
                                raw_output: raw_output.to_string(),
                            })
                        })
                        .ok()
                });

                let output_schema = match &inference_config.dynamic_output_schema {
                    Some(schema) => JsonSchemaRef::Dynamic(schema),
                    None => JsonSchemaRef::Static(&params.output_schema),
                };

                // If the parsed output fails validation, we log the error and set `parsed_output` to None
                let parsed_output = match parsed_output {
                    Some(parsed_output) => match output_schema.validate(&parsed_output).await {
                        Ok(()) => Some(parsed_output),
                        Err(_) => None,
                    },
                    None => None,
                };
                Ok(InferenceResult::Json(JsonInferenceResult::new(
                    inference_id,
                    raw_output,
                    parsed_output,
                    json_block_index,
                    auxiliary_content,
                    model_inference_results,
                    output_schema.value().clone(),
                    inference_params,
                    original_response,
                )))
            }
        }
    }

    pub fn schemas(&self) -> &SchemaData {
        match self {
            FunctionConfig::Chat(params) => &params.schemas,
            FunctionConfig::Json(params) => &params.schemas,
        }
    }

    pub fn system_schema(&self) -> Option<&StaticJSONSchema> {
        match self {
            FunctionConfig::Chat(params) => params
                .schemas
                .get_implicit_system_schema()
                .map(|s| &s.schema),
            FunctionConfig::Json(params) => params
                .schemas
                .get_implicit_system_schema()
                .map(|s| &s.schema),
        }
    }

    pub fn user_schema(&self) -> Option<&StaticJSONSchema> {
        match self {
            FunctionConfig::Chat(params) => {
                params.schemas.get_implicit_user_schema().map(|s| &s.schema)
            }
            FunctionConfig::Json(params) => {
                params.schemas.get_implicit_user_schema().map(|s| &s.schema)
            }
        }
    }

    pub fn assistant_schema(&self) -> Option<&StaticJSONSchema> {
        match self {
            FunctionConfig::Chat(params) => params
                .schemas
                .get_implicit_assistant_schema()
                .map(|s| &s.schema),
            FunctionConfig::Json(params) => params
                .schemas
                .get_implicit_assistant_schema()
                .map(|s| &s.schema),
        }
    }

    pub fn description(&self) -> Option<&String> {
        match self {
            FunctionConfig::Chat(params) => params.description.as_ref(),
            FunctionConfig::Json(params) => params.description.as_ref(),
        }
    }

    // This needs to be `async` because we end up validating GCP model providers,
    // which may call an async GCP SDK function to fetch credentials from the environment.
    #[instrument(skip_all, fields(function_name = %function_name))]
    pub async fn validate(
        self: &Arc<Self>,
        static_tools: &HashMap<String, Arc<StaticToolConfig>>,
        models: &ModelTable,
        embedding_models: &EmbeddingModelTable,
        templates: &TemplateConfig<'_>,
        function_name: &str,
        global_outbound_http_timeout: &Duration,
    ) -> Result<(), Error> {
        // Validate each variant
        for (variant_name, variant) in self.variants() {
            if variant_name.starts_with("tensorzero::") {
                return Err(ErrorDetails::Config {
                    message: format!(
                        "Variant name cannot start with 'tensorzero::': {variant_name}"
                    ),
                }
                .into());
            }
            variant
                .validate(
                    Arc::clone(self),
                    models,
                    embedding_models,
                    templates,
                    function_name,
                    variant_name,
                    global_outbound_http_timeout,
                )
                .await?;
        }
        match self.as_ref() {
            FunctionConfig::Chat(params) => {
                for tool in &params.tools {
                    static_tools.get(tool).ok_or_else(|| Error::new(ErrorDetails::Config {
                        message: format!("`functions.{function_name}.tools`: tool `{tool}` is not present in the config"),
                    }))?;
                }
                Ok(())
            }
            FunctionConfig::Json(_) => Ok(()),
        }
    }
}

/// Parse the content blocks into a JSON object
/// We assume here that the last content block that's text or a tool call is the JSON object.
/// (this is because we could have used an implicit tool call and there is no other reason for a tool call in a JSON function).
///
/// Sometimes models will return no content blocks (e.g. when instructed to not return anything), so `raw_output` will be `None` then.
///
/// Returns: the raw output, the auxiliary content, and the index of the JSON block in the original content blocks.
fn get_json_output_from_content_blocks(
    mut content_blocks: Vec<ContentBlockOutput>,
) -> (Option<String>, Vec<ContentBlockOutput>, Option<usize>) {
    let raw_output = content_blocks
        .iter()
        .rev()
        .find_map(|content_block| match content_block {
            ContentBlockOutput::Text(text) => Some(text.text.to_string()),
            ContentBlockOutput::ToolCall(tool_call) => Some(tool_call.arguments.to_string()),
            _ => None,
        });
    let maybe_index_from_end = content_blocks.iter().rev().position(|content_block| {
        matches!(
            content_block,
            ContentBlockOutput::Text(_) | ContentBlockOutput::ToolCall(_)
        )
    });
    let json_block_index = match maybe_index_from_end {
        Some(i) => {
            let index_from_start = content_blocks.len() - 1 - i;
            content_blocks.remove(index_from_start);
            Some(index_from_start)
        }
        None => None,
    };
    (raw_output, content_blocks, json_block_index)
}

/// Validate all input messages that contain text (not raw_text).
/// The validation is done based on the input's role and the function's schemas.
/// We first validate the system message (if it exists)
/// Next we validate all messages containing text blocks.
/// When we add support for `{"type": "template"}` input blocks, we'll need to validate those two
fn validate_all_text_input(
    schemas: &SchemaData,
    input: &Input,
    all_templates_names: &HashSet<String>,
) -> Result<(), Error> {
    match (input.system.as_ref(), schemas.get_implicit_system_schema()) {
        // If there is any system message passed we validate it
        (Some(system), _) => {
            let system_value = match system {
                System::Text(text) => Cow::Owned(Value::String(text.clone())),
                System::Template(arguments) => Cow::Owned(Value::Object(arguments.0.clone())),
            };
            validate_single_message(
                &system_value,
                schemas.get_implicit_system_schema().map(|s| &s.schema),
                "system",
                all_templates_names,
                None,
            )
        }
        // If there is no system message and no schema we accept
        (None, None) => Ok(()),
        // If no system message is passed and we have a schema we fail
        (None, Some(_)) => Err(Error::new(ErrorDetails::InvalidMessage {
            message: "`input.system` is empty but a system template is present.".to_string(),
        })),
    }?;
    for (index, message) in input.messages.iter().enumerate() {
        for block in &message.content {
            match block {
                InputMessageContent::Text(text) => {
                    let content = Cow::Owned(Value::String(text.text.clone()));
                    let schema = match &message.role {
                        Role::Assistant => schemas.get_implicit_assistant_schema(),
                        Role::User => schemas.get_implicit_user_schema(),
                    };
                    validate_single_message(
                        &content,
                        schema.map(|s| &s.schema),
                        message.role.implicit_template_name(),
                        all_templates_names,
                        Some(index),
                    )?;
                }
                InputMessageContent::Template(template) => {
                    // TODO: figure out a way to avoid this clone
                    let value = Value::Object(template.arguments.0.clone());
                    validate_single_message(
                        &value,
                        schemas.get_named_schema(&template.name).map(|s| &s.schema),
                        &template.name,
                        all_templates_names,
                        Some(index),
                    )?;
                }
                _ => {}
            }
        }
    }
    Ok(())
}

/// Validates a single message according to the following rules:
/// If there is no schema, we check that at least one
/// variant has a matching template (as determined by `all_templates_names`)
/// Otherwise, the message must contain JSON content that matches the schema
fn validate_single_message(
    content: &Value,
    schema: Option<&StaticJSONSchema>,
    template_name: &str,
    all_templates_names: &HashSet<String>,
    index: Option<usize>,
) -> Result<(), Error> {
    match schema {
        Some(schema) => schema.validate(content)?,
        None => {
            if !content.is_string() && !all_templates_names.contains(template_name) {
                return Err(match index {
                    Some(index) => Error::new(ErrorDetails::InvalidMessage {
                        message: format!("Message at index {index} has non-string content but there is no template `{template_name}` in any variant"),
                    }),
                    None => Error::new(ErrorDetails::InvalidMessage {
                        message: format!("System message has non-string content but there is no template `{template_name}` in any variant"),
                    }),
                });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::path::ResolvedTomlPathData;
    use crate::config::UninitializedSchemas;
    use crate::endpoints::inference::InferenceIds;
    use crate::inference::types::Arguments;
    use crate::inference::types::FinishReason;
    use crate::inference::types::InputMessage;
    use crate::inference::types::Latency;
    use crate::inference::types::RawText;
    use crate::inference::types::RequestMessagesOrBatch;
    use crate::inference::types::Template;
    use crate::inference::types::Text;
    use crate::inference::types::Thought;
    use crate::inference::types::Usage;
    use crate::jsonschema_util::DynamicJSONSchema;
    use crate::minijinja_util::TemplateConfig;
    use crate::tool::ToolCall;
    use serde_json::json;
    use std::io::Write;
    use std::time::Duration;
    use std::time::Instant;
    use tempfile::NamedTempFile;

    fn create_test_schema() -> StaticJSONSchema {
        let schema = r#"
        {
            "type": "object",
            "properties": {
                "name": { "type": "string" }
            },
            "required": ["name"],
            "additionalProperties": false
        }
        "#;

        let mut temp_file = NamedTempFile::new().expect("Failed to create temporary file");
        write!(temp_file, "{schema}").expect("Failed to write schema to temporary file");

        StaticJSONSchema::from_path(ResolvedTomlPathData::new_for_tests(
            temp_file.path().to_owned(),
            None,
        ))
        .expect("Failed to create schema")
    }

    #[test]
    fn test_validate_input_chat_no_schema() {
        let chat_config = FunctionConfigChat {
            variants: HashMap::new(),
            schemas: SchemaData::default(),
            tools: vec![],
            ..Default::default()
        };
        let function_config = FunctionConfig::Chat(chat_config);

        let messages = vec![
            InputMessage {
                role: Role::User,
                content: vec!["user content".to_string().into()],
            },
            InputMessage {
                role: Role::Assistant,
                content: vec!["assistant content".to_string().into()],
            },
        ];

        let input = Input {
            system: Some(System::Text("system content".to_string())),
            messages,
        };

        assert!(function_config.validate_input(&input).is_ok());

        let messages = vec![
            InputMessage {
                role: Role::User,
                content: vec!["user content".to_string().into()],
            },
            InputMessage {
                role: Role::Assistant,
                content: vec![InputMessageContent::Template(Template {
                    name: "assistant".to_string(),
                    arguments: Arguments(
                        json!({ "name": "assistant name" })
                            .as_object()
                            .unwrap()
                            .clone(),
                    ),
                })],
            },
        ];
        let input = Input {
            system: Some(System::Text("system name".to_string())),
            messages,
        };

        let validation_result = function_config.validate_input(&input);
        assert_eq!(
            validation_result.unwrap_err(),
            Error::new(ErrorDetails::InvalidMessage {
                message: "Message at index 1 has non-string content but there is no template `assistant` in any variant".to_string(),
            })
        );

        // Test case for multiple text content blocks in one message
        // This is allowed behavior
        let messages = vec![
            InputMessage {
                role: Role::User,
                content: vec![
                    "first user content".to_string().into(),
                    "second user content".to_string().into(),
                ],
            },
            InputMessage {
                role: Role::Assistant,
                content: vec!["assistant content".to_string().into()],
            },
        ];
        let input = Input {
            system: Some(System::Text("system content".to_string())),
            messages,
        };

        function_config.validate_input(&input).unwrap();
    }

    #[test]
    fn test_validate_input_chat_system_schema() {
        let system_schema = create_test_schema();
        let system_value = system_schema.value.clone();
        let chat_config = FunctionConfigChat {
            variants: HashMap::new(),
            schemas: SchemaData::load(
                None,
                None,
                Some(system_schema),
                UninitializedSchemas::default(),
                "test",
            )
            .unwrap(),
            tools: vec![],
            ..Default::default()
        };
        let function_config = FunctionConfig::Chat(chat_config);

        let messages = vec![
            InputMessage {
                role: Role::User,
                content: vec!["user content".to_string().into()],
            },
            InputMessage {
                role: Role::Assistant,
                content: vec!["assistant content".to_string().into()],
            },
        ];
        let input = Input {
            system: Some(System::Text("system content".to_string())),
            messages,
        };

        let validation_result = function_config.validate_input(&input);
        assert_eq!(
            validation_result.unwrap_err(),
            Error::new(ErrorDetails::JsonSchemaValidation {
                messages: vec!["\"system content\" is not of type \"object\"".to_string()],
                data: Box::new(json!("system content")),
                schema: Box::new(system_value),
            })
        );

        let messages = vec![
            InputMessage {
                role: Role::User,
                content: vec!["user content".to_string().into()],
            },
            InputMessage {
                role: Role::Assistant,
                content: vec!["assistant content".to_string().into()],
            },
        ];
        let input = Input {
            system: Some(System::Template(Arguments(
                json!({ "name": "system name" })
                    .as_object()
                    .unwrap()
                    .clone(),
            ))),
            messages,
        };

        assert!(function_config.validate_input(&input).is_ok());
    }

    #[test]
    fn test_validate_input_chat_user_schema() {
        let user_schema = create_test_schema();
        let user_value = user_schema.value.clone();
        let chat_config = FunctionConfigChat {
            variants: HashMap::new(),
            schemas: SchemaData::load(
                Some(user_schema),
                None,
                None,
                UninitializedSchemas::default(),
                "test",
            )
            .unwrap(),
            tools: vec![],
            ..Default::default()
        };
        let function_config = FunctionConfig::Chat(chat_config);

        let messages = vec![
            InputMessage {
                role: Role::User,
                content: vec!["user content".to_string().into()],
            },
            InputMessage {
                role: Role::Assistant,
                content: vec!["assistant content".to_string().into()],
            },
        ];
        let input = Input {
            system: Some(System::Text("system content".to_string())),
            messages,
        };
        let validation_result = function_config.validate_input(&input);
        assert_eq!(
            validation_result.unwrap_err(),
            ErrorDetails::JsonSchemaValidation {
                messages: vec!["\"user content\" is not of type \"object\"".to_string()],
                data: Box::new(json!("user content")),
                schema: Box::new(user_value),
            }
            .into()
        );

        let messages = vec![
            InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::Template(Template {
                    name: "user".to_string(),
                    arguments: Arguments(serde_json::Map::from_iter([(
                        "name".to_string(),
                        "user name".into(),
                    )])),
                })],
            },
            InputMessage {
                role: Role::Assistant,
                content: vec!["assistant content".to_string().into()],
            },
        ];
        let input = Input {
            system: Some(System::Text("system content".to_string())),
            messages,
        };

        assert!(function_config.validate_input(&input).is_ok());
    }

    #[test]
    fn test_validate_input_chat_assistant_schema() {
        let assistant_schema = create_test_schema();
        let assistant_value = assistant_schema.value.clone();
        let chat_config = FunctionConfigChat {
            variants: HashMap::new(),
            schemas: SchemaData::load(
                None,
                Some(assistant_schema),
                None,
                UninitializedSchemas::default(),
                "test",
            )
            .unwrap(),
            tools: vec![],
            ..Default::default()
        };
        let function_config = FunctionConfig::Chat(chat_config);

        let messages = vec![
            InputMessage {
                role: Role::User,
                content: vec!["user content".to_string().into()],
            },
            InputMessage {
                role: Role::Assistant,
                content: vec!["assistant content".to_string().into()],
            },
        ];
        let input = Input {
            system: Some(System::Text("system content".to_string())),
            messages,
        };
        let validation_result = function_config.validate_input(&input);
        assert_eq!(
            validation_result.unwrap_err(),
            ErrorDetails::JsonSchemaValidation {
                messages: vec!["\"assistant content\" is not of type \"object\"".to_string()],
                data: Box::new(json!("assistant content")),
                schema: Box::new(assistant_value),
            }
            .into()
        );

        let messages = vec![
            InputMessage {
                role: Role::User,
                content: vec!["user content".to_string().into()],
            },
            InputMessage {
                role: Role::Assistant,
                content: vec![InputMessageContent::Template(Template {
                    name: "assistant".to_string(),
                    arguments: Arguments(
                        json!({ "name": "assistant name" })
                            .as_object()
                            .unwrap()
                            .clone(),
                    ),
                })],
            },
        ];
        let input = Input {
            system: Some(System::Text("system content".to_string())),
            messages,
        };

        assert!(function_config.validate_input(&input).is_ok());
    }

    #[test]
    fn test_validate_input_chat_all_schemas() {
        let system_schema = create_test_schema();
        let user_schema = create_test_schema();
        let assistant_schema = create_test_schema();
        let system_value = system_schema.value.clone();
        let chat_config = FunctionConfigChat {
            variants: HashMap::new(),
            schemas: SchemaData::load(
                Some(user_schema),
                Some(assistant_schema),
                Some(system_schema),
                UninitializedSchemas::default(),
                "test",
            )
            .unwrap(),
            tools: vec![],
            ..Default::default()
        };
        let function_config = FunctionConfig::Chat(chat_config);

        let messages = vec![
            InputMessage {
                role: Role::User,
                content: vec!["user content".to_string().into()],
            },
            InputMessage {
                role: Role::Assistant,
                content: vec!["assistant content".to_string().into()],
            },
            InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::RawText(RawText {
                    value: "raw text".to_string(),
                })],
            },
        ];

        let input = Input {
            system: Some(System::Text("system content".to_string())),
            messages,
        };

        let validation_result = function_config.validate_input(&input);
        assert_eq!(
            validation_result.unwrap_err(),
            ErrorDetails::JsonSchemaValidation {
                messages: vec!["\"system content\" is not of type \"object\"".to_string()],
                data: Box::new(json!("system content")),
                schema: Box::new(system_value),
            }
            .into()
        );

        let messages = vec![
            InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::Template(Template {
                    name: "user".to_string(),
                    arguments: Arguments(serde_json::Map::from_iter([(
                        "name".to_string(),
                        "user name".into(),
                    )])),
                })],
            },
            InputMessage {
                role: Role::Assistant,
                content: vec![InputMessageContent::Template(Template {
                    name: "assistant".to_string(),
                    arguments: Arguments(
                        json!({ "name": "assistant name" })
                            .as_object()
                            .unwrap()
                            .clone(),
                    ),
                })],
            },
        ];

        let input = Input {
            system: Some(System::Template(Arguments(
                json!({ "name": "system name" })
                    .as_object()
                    .unwrap()
                    .clone(),
            ))),
            messages,
        };

        assert!(function_config.validate_input(&input).is_ok());
    }

    #[test]
    fn test_validate_input_raw_bypass_schemas() {
        let system_schema = create_test_schema();
        let user_schema = create_test_schema();
        let assistant_schema = create_test_schema();
        let chat_config = FunctionConfigChat {
            variants: HashMap::new(),
            schemas: SchemaData::load(
                Some(user_schema),
                Some(assistant_schema),
                Some(system_schema),
                UninitializedSchemas::default(),
                "test",
            )
            .unwrap(),
            tools: vec![],
            ..Default::default()
        };
        let function_config = FunctionConfig::Chat(chat_config);

        let messages = vec![
            InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::RawText(RawText {
                    value: "user content".to_string(),
                })],
            },
            InputMessage {
                role: Role::Assistant,
                content: vec![InputMessageContent::RawText(RawText {
                    value: "assistant content".to_string(),
                })],
            },
            InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::RawText(RawText {
                    value: "raw text".to_string(),
                })],
            },
        ];

        let input = Input {
            system: Some(System::Template(Arguments(
                json!({ "name": "system name" })
                    .as_object()
                    .unwrap()
                    .clone(),
            ))),
            messages,
        };

        let validation_result = function_config.validate_input(&input);
        assert!(validation_result.is_ok());
    }

    #[test]
    fn test_validate_input_chat_multiple_text_blocks() {
        // We test that we allow multiple text blocks in a message as long as they pass the schema if present
        let chat_config = FunctionConfigChat {
            variants: HashMap::new(),
            schemas: SchemaData::default(),
            tools: vec![],
            ..Default::default()
        };
        let function_config = FunctionConfig::Chat(chat_config);

        let messages = vec![
            InputMessage {
                role: Role::User,
                content: vec![
                    "user content".to_string().into(),
                    "extra content".to_string().into(),
                ],
            },
            InputMessage {
                role: Role::Assistant,
                content: vec!["assistant content".to_string().into()],
            },
            InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::RawText(RawText {
                    value: "raw text".to_string(),
                })],
            },
        ];

        let input = Input {
            system: Some(System::Text("system content".to_string())),
            messages,
        };

        function_config.validate_input(&input).unwrap();
        let user_schema = create_test_schema();
        let assistant_schema = create_test_schema();
        let chat_config = FunctionConfigChat {
            variants: HashMap::new(),
            schemas: SchemaData::load(
                Some(user_schema),
                Some(assistant_schema),
                None,
                UninitializedSchemas::default(),
                "test",
            )
            .unwrap(),
            tools: vec![],
            ..Default::default()
        };
        let function_config = FunctionConfig::Chat(chat_config);

        let messages = vec![
            InputMessage {
                role: Role::User,
                content: vec![
                    InputMessageContent::Template(Template {
                        name: "user".to_string(),
                        arguments: Arguments(serde_json::Map::from_iter([(
                            "name".to_string(),
                            "user name".into(),
                        )])),
                    }),
                    InputMessageContent::Template(Template {
                        name: "user".to_string(),
                        arguments: Arguments(
                            json!({ "name": "extra content" })
                                .as_object()
                                .unwrap()
                                .clone(),
                        ),
                    }),
                ],
            },
            InputMessage {
                role: Role::Assistant,
                content: vec![InputMessageContent::Template(Template {
                    name: "assistant".to_string(),
                    arguments: Arguments(
                        json!({ "name": "assistant name" })
                            .as_object()
                            .unwrap()
                            .clone(),
                    ),
                })],
            },
        ];

        let input = Input {
            system: Some(System::Text("system content".to_string())),
            messages,
        };

        function_config.validate_input(&input).unwrap();
    }

    #[test]
    fn test_validate_input_json_no_schema() {
        let output_schema = json!({});
        let json_mode_tool_call_config = ToolCallConfig::implicit_from_value(&output_schema);
        let tool_config = FunctionConfigJson {
            variants: HashMap::new(),
            schemas: SchemaData::default(),
            output_schema: StaticJSONSchema::from_value(json!({})).unwrap(),
            json_mode_tool_call_config,
            description: None,
            all_explicit_template_names: HashSet::new(),
            experimentation: ExperimentationConfig::default(),
        };
        let function_config = FunctionConfig::Json(tool_config);

        let messages = vec![
            InputMessage {
                role: Role::User,
                content: vec!["user content".to_string().into()],
            },
            InputMessage {
                role: Role::Assistant,
                content: vec!["assistant content".to_string().into()],
            },
            InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::RawText(RawText {
                    value: "raw text".to_string(),
                })],
            },
        ];

        let input = Input {
            system: Some(System::Text("system content".to_string())),
            messages,
        };

        assert!(function_config.validate_input(&input).is_ok());

        let messages = vec![
            InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::Template(Template {
                    name: "user".to_string(),
                    arguments: Arguments(serde_json::Map::from_iter([(
                        "name".to_string(),
                        "user name".into(),
                    )])),
                })],
            },
            InputMessage {
                role: Role::Assistant,
                content: vec![InputMessageContent::Template(Template {
                    name: "assistant".to_string(),
                    arguments: Arguments(
                        json!({ "name": "assistant name" })
                            .as_object()
                            .unwrap()
                            .clone(),
                    ),
                })],
            },
        ];

        let input = Input {
            system: Some(System::Text("system content".to_string())),
            messages,
        };

        let validation_result = function_config.validate_input(&input);
        assert_eq!(
            validation_result.unwrap_err(),
            ErrorDetails::InvalidMessage {
                message: "Message at index 0 has non-string content but there is no template `user` in any variant".to_string()
            }.into()
        );
    }

    #[test]
    fn test_validate_input_json_system_schema() {
        let system_schema = create_test_schema();
        let system_value = system_schema.value.clone();
        let output_schema = json!({});
        let json_mode_tool_call_config = ToolCallConfig::implicit_from_value(&output_schema);
        let tool_config = FunctionConfigJson {
            variants: HashMap::new(),
            schemas: SchemaData::load(
                None,
                None,
                Some(system_schema),
                UninitializedSchemas::default(),
                "test",
            )
            .unwrap(),
            output_schema: StaticJSONSchema::from_value(output_schema).unwrap(),
            json_mode_tool_call_config,
            description: None,
            all_explicit_template_names: HashSet::new(),
            experimentation: ExperimentationConfig::default(),
        };
        let function_config = FunctionConfig::Json(tool_config);

        let messages = vec![
            InputMessage {
                role: Role::User,
                content: vec!["user content".to_string().into()],
            },
            InputMessage {
                role: Role::Assistant,
                content: vec![json!("assistant content").to_string().into()],
            },
        ];

        let input = Input {
            system: Some(System::Text("system content".to_string())),
            messages,
        };

        let validation_result = function_config.validate_input(&input);
        assert_eq!(
            validation_result.unwrap_err(),
            ErrorDetails::JsonSchemaValidation {
                messages: vec!["\"system content\" is not of type \"object\"".to_string()],
                data: Box::new(json!("system content")),
                schema: Box::new(system_value),
            }
            .into()
        );

        let messages = vec![
            InputMessage {
                role: Role::User,
                content: vec!["user content".to_string().into()],
            },
            InputMessage {
                role: Role::Assistant,
                content: vec![json!("assistant content").to_string().into()],
            },
        ];

        let input = Input {
            system: Some(System::Template(Arguments(
                json!({ "name": "system name" })
                    .as_object()
                    .unwrap()
                    .clone(),
            ))),
            messages,
        };

        assert!(function_config.validate_input(&input).is_ok());
    }

    #[test]
    fn test_validate_input_json_user_schema() {
        let user_schema = create_test_schema();
        let user_value = user_schema.value.clone();
        let output_schema = json!({});
        let json_mode_tool_call_config = ToolCallConfig::implicit_from_value(&output_schema);
        let tool_config = FunctionConfigJson {
            variants: HashMap::new(),
            schemas: SchemaData::load(
                Some(user_schema),
                None,
                None,
                UninitializedSchemas::default(),
                "test",
            )
            .unwrap(),
            output_schema: StaticJSONSchema::from_value(output_schema).unwrap(),
            json_mode_tool_call_config,
            description: None,
            all_explicit_template_names: HashSet::new(),
            experimentation: ExperimentationConfig::default(),
        };
        let function_config = FunctionConfig::Json(tool_config);

        let messages = vec![
            InputMessage {
                role: Role::User,
                content: vec!["user content".to_string().into()],
            },
            InputMessage {
                role: Role::Assistant,
                content: vec![json!("assistant content").to_string().into()],
            },
        ];

        let input = Input {
            system: Some(System::Text("system content".to_string())),
            messages,
        };

        let validation_result = function_config.validate_input(&input);
        assert_eq!(
            validation_result.unwrap_err(),
            ErrorDetails::JsonSchemaValidation {
                messages: vec!["\"user content\" is not of type \"object\"".to_string()],
                data: Box::new(json!("user content")),
                schema: Box::new(user_value),
            }
            .into()
        );

        let messages = vec![
            InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::Template(Template {
                    name: "user".to_string(),
                    arguments: Arguments(serde_json::Map::from_iter([(
                        "name".to_string(),
                        "user name".into(),
                    )])),
                })],
            },
            InputMessage {
                role: Role::Assistant,
                content: vec!["assistant content".to_string().into()],
            },
        ];
        let input = Input {
            system: Some(System::Text("system content".to_string())),
            messages,
        };

        assert!(function_config.validate_input(&input).is_ok());
    }

    #[test]
    fn test_validate_input_json_assistant_schema() {
        let assistant_schema = create_test_schema();
        let assistant_value = assistant_schema.value.clone();
        let output_schema = json!({});
        let json_mode_tool_call_config = ToolCallConfig::implicit_from_value(&output_schema);
        let tool_config = FunctionConfigJson {
            variants: HashMap::new(),
            schemas: SchemaData::load(
                None,
                Some(assistant_schema),
                None,
                UninitializedSchemas::default(),
                "test",
            )
            .unwrap(),
            output_schema: StaticJSONSchema::from_value(output_schema).unwrap(),
            json_mode_tool_call_config,
            description: None,
            all_explicit_template_names: HashSet::new(),
            experimentation: ExperimentationConfig::default(),
        };
        let function_config = FunctionConfig::Json(tool_config);

        let messages = vec![
            InputMessage {
                role: Role::User,
                content: vec!["user content".to_string().into()],
            },
            InputMessage {
                role: Role::Assistant,
                content: vec!["assistant content".to_string().into()],
            },
        ];
        let input = Input {
            system: Some(System::Text("system content".to_string())),
            messages,
        };

        let validation_result = function_config.validate_input(&input);
        assert_eq!(
            validation_result.unwrap_err(),
            ErrorDetails::JsonSchemaValidation {
                messages: vec!["\"assistant content\" is not of type \"object\"".to_string()],
                data: Box::new(json!("assistant content")),
                schema: Box::new(assistant_value),
            }
            .into()
        );

        let messages = vec![
            InputMessage {
                role: Role::User,
                content: vec!["user content".to_string().into()],
            },
            InputMessage {
                role: Role::Assistant,
                content: vec![InputMessageContent::Template(Template {
                    name: "assistant".to_string(),
                    arguments: Arguments(
                        json!({ "name": "assistant name" })
                            .as_object()
                            .unwrap()
                            .clone(),
                    ),
                })],
            },
        ];
        let input = Input {
            system: Some(System::Text("system content".to_string())),
            messages,
        };

        assert!(function_config.validate_input(&input).is_ok());
    }

    #[test]
    fn test_validate_input_json_all_schemas() {
        let system_schema = create_test_schema();
        let user_schema = create_test_schema();
        let assistant_schema = create_test_schema();
        let system_value = system_schema.value.clone();
        let output_schema = json!({});
        let json_mode_tool_call_config = ToolCallConfig::implicit_from_value(&output_schema);
        let tool_config = FunctionConfigJson {
            variants: HashMap::new(),
            schemas: SchemaData::load(
                Some(user_schema),
                Some(assistant_schema),
                Some(system_schema),
                UninitializedSchemas::default(),
                "test",
            )
            .unwrap(),
            output_schema: StaticJSONSchema::from_value(output_schema).unwrap(),
            json_mode_tool_call_config,
            description: None,
            all_explicit_template_names: HashSet::new(),
            experimentation: ExperimentationConfig::default(),
        };
        let function_config = FunctionConfig::Json(tool_config);

        let messages = vec![
            InputMessage {
                role: Role::User,
                content: vec!["user content".to_string().into()],
            },
            InputMessage {
                role: Role::Assistant,
                content: vec![json!("assistant content").to_string().into()],
            },
        ];
        let input = Input {
            system: Some(System::Text("system content".to_string())),
            messages,
        };

        let validation_result = function_config.validate_input(&input);
        assert_eq!(
            validation_result.unwrap_err(),
            ErrorDetails::JsonSchemaValidation {
                messages: vec!["\"system content\" is not of type \"object\"".to_string()],
                data: Box::new(json!("system content")),
                schema: Box::new(system_value),
            }
            .into()
        );

        let messages = vec![
            InputMessage {
                role: Role::User,
                content: vec![InputMessageContent::Template(Template {
                    name: "user".to_string(),
                    arguments: Arguments(serde_json::Map::from_iter([(
                        "name".to_string(),
                        "user name".into(),
                    )])),
                })],
            },
            InputMessage {
                role: Role::Assistant,
                content: vec![InputMessageContent::Template(Template {
                    name: "assistant".to_string(),
                    arguments: Arguments(
                        json!({ "name": "assistant name" })
                            .as_object()
                            .unwrap()
                            .clone(),
                    ),
                })],
            },
        ];

        let input = Input {
            system: Some(System::Template(Arguments(
                json!({ "name": "system name" })
                    .as_object()
                    .unwrap()
                    .clone(),
            ))),
            messages,
        };

        assert!(function_config.validate_input(&input).is_ok());
    }

    /// Tests the `sample_variant` function with a variety of test cases through Monte Carlo simulations.
    ///
    /// NOTE: If this test fails, it might be due to sampling. Please run it again to check if the
    ///       issue persists.

    #[test]
    fn test_description_getter() {
        // Test for Chat function with description
        let chat_config = FunctionConfigChat {
            variants: HashMap::new(),
            schemas: SchemaData::default(),
            tools: vec![],
            tool_choice: ToolChoice::None,
            parallel_tool_calls: None,
            description: Some("A chat function description".to_string()),
            all_explicit_templates_names: HashSet::new(),
            experimentation: ExperimentationConfig::default(),
        };
        let function_config = FunctionConfig::Chat(chat_config);
        assert_eq!(
            function_config.description(),
            Some(&"A chat function description".to_string())
        );

        // Test for JSON function with description
        let output_schema = StaticJSONSchema::from_value(json!({})).unwrap();
        let json_mode_tool_call_config = ToolCallConfig::implicit_from_value(&json!({}));
        let json_config = FunctionConfigJson {
            variants: HashMap::new(),
            schemas: SchemaData::default(),
            output_schema,
            json_mode_tool_call_config,
            description: Some("A JSON function description".to_string()),
            all_explicit_template_names: HashSet::new(),
            experimentation: ExperimentationConfig::default(),
        };
        let function_config = FunctionConfig::Json(json_config);
        assert_eq!(
            function_config.description(),
            Some(&"A JSON function description".to_string())
        );

        // Test for None description
        let chat_config = FunctionConfigChat {
            variants: HashMap::new(),
            schemas: SchemaData::default(),
            tools: vec![],
            tool_choice: ToolChoice::None,
            parallel_tool_calls: None,
            description: None,
            all_explicit_templates_names: HashSet::new(),
            experimentation: ExperimentationConfig::default(),
        };
        let function_config = FunctionConfig::Chat(chat_config);
        assert_eq!(function_config.description(), None);
    }

    #[tokio::test]
    async fn test_prepare_response_json() {
        let logs_contain = crate::utils::testing::capture_logs();
        // The Chat stuff is tested in types::test_create_chat_inference_response
        // Here we focus on the JSON stuff
        let output_schema = json!({
          "$schema": "http://json-schema.org/draft-07/schema#",
          "type": "object",
          "properties": {
            "name": {
              "type": "string"
            },
            "age": {
              "type": "integer",
              "minimum": 0
            }
          },
          "required": ["name", "age"],
          "additionalProperties": false
        });
        let json_mode_tool_call_config = ToolCallConfig::implicit_from_value(&output_schema);
        let output_schema = StaticJSONSchema::from_value(output_schema).unwrap();
        let function_config = FunctionConfig::Json(FunctionConfigJson {
            variants: HashMap::new(),
            schemas: SchemaData::default(),
            output_schema,
            json_mode_tool_call_config,
            description: None,
            all_explicit_template_names: HashSet::new(),
            experimentation: ExperimentationConfig::default(),
        });
        let raw_request = "raw_request".to_string();

        // Test with a non-JSON content block
        let inference_id = Uuid::now_v7();
        let content_blocks = vec!["Hello, world!".to_string().into()];
        let usage = Usage {
            input_tokens: Some(10),
            output_tokens: Some(10),
        };
        let latency = Latency::NonStreaming {
            response_time: Duration::from_millis(100),
        };
        let model_response = ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            created: Instant::now().elapsed().as_secs(),
            system: None,
            input_messages: RequestMessagesOrBatch::Message(vec![]),
            output: content_blocks.clone(),
            raw_request: raw_request.clone(),
            raw_response: "content".to_string(),
            usage,
            model_provider_name: "model_provider_name".into(),
            model_name: "model_name".into(),
            finish_reason: Some(FinishReason::Stop),
            latency,
            cached: false,
        };
        let templates = Arc::new(TemplateConfig::default());
        let inference_config = InferenceConfig {
            ids: InferenceIds {
                inference_id: Uuid::now_v7(),
                episode_id: Uuid::now_v7(),
            },
            tool_config: None,
            function_name: "".into(),
            variant_name: "".into(),
            templates: templates.clone(),
            dynamic_output_schema: None,
            extra_body: Default::default(),
            extra_headers: Default::default(),
            fetch_and_encode_input_files_before_inference: false,
            extra_cache_key: None,
        };
        let response = function_config
            .prepare_response(
                inference_id,
                content_blocks,
                vec![model_response.clone()],
                &inference_config,
                InferenceParams::default(),
                None,
            )
            .await
            .unwrap();
        assert!(logs_contain(
            "Failed to parse output from JSON function response"
        ));
        assert_eq!(response.usage_considering_cached(), usage);
        match response {
            InferenceResult::Json(result) => {
                assert_eq!(result.inference_id, inference_id);
                assert!(result.output.parsed.is_none());
                assert_eq!(result.output.raw, Some("Hello, world!".to_string()));
                assert_eq!(result.finish_reason, Some(FinishReason::Stop));
                assert_eq!(result.model_inference_results, vec![model_response]);
            }
            InferenceResult::Chat(_) => panic!("Expected a JSON inference result"),
        }

        // Test with a correct content block
        let inference_id = Uuid::now_v7();
        let content_blocks = vec![r#"{"name": "Jerry", "age": 30}"#.to_string().into()];
        let usage = Usage {
            input_tokens: Some(10),
            output_tokens: Some(10),
        };
        let latency = Latency::NonStreaming {
            response_time: Duration::from_millis(100),
        };
        let model_response = ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            created: Instant::now().elapsed().as_secs(),
            system: None,
            input_messages: RequestMessagesOrBatch::Message(vec![]),
            output: content_blocks.clone(),
            raw_request: raw_request.clone(),
            raw_response: "content".to_string(),
            usage,
            model_provider_name: "model_provider_name".into(),
            model_name: "model_name".into(),
            finish_reason: Some(FinishReason::ToolCall),
            latency,
            cached: false,
        };
        let response = function_config
            .prepare_response(
                inference_id,
                content_blocks,
                vec![model_response.clone()],
                &inference_config,
                InferenceParams::default(),
                None,
            )
            .await
            .unwrap();
        assert_eq!(response.usage_considering_cached(), usage);
        match response {
            InferenceResult::Json(result) => {
                assert_eq!(result.inference_id, inference_id);
                assert_eq!(
                    result.output.parsed.unwrap(),
                    json!({"name": "Jerry", "age": 30}),
                );
                assert_eq!(
                    result.output.raw,
                    Some("{\"name\": \"Jerry\", \"age\": 30}".to_string())
                );
                assert_eq!(result.model_inference_results, vec![model_response]);
            }
            InferenceResult::Chat(_) => panic!("Expected a JSON inference result"),
        }

        // Test with an incorrect JSON content block
        let inference_id = Uuid::now_v7();
        let content_blocks = vec![r#"{"name": "Jerry", "age": "thirty"}"#.to_string().into()];
        let usage = Usage {
            input_tokens: Some(10),
            output_tokens: Some(10),
        };
        let latency = Latency::NonStreaming {
            response_time: Duration::from_millis(100),
        };
        let model_response = ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            created: Instant::now().elapsed().as_secs(),
            system: None,
            input_messages: RequestMessagesOrBatch::Message(vec![]),
            output: content_blocks.clone(),
            raw_request: raw_request.clone(),
            raw_response: "content".to_string(),
            usage,
            model_provider_name: "model_provider_name".into(),
            model_name: "model_name".into(),
            finish_reason: Some(FinishReason::ToolCall),
            latency,
            cached: false,
        };
        let response = function_config
            .prepare_response(
                inference_id,
                content_blocks,
                vec![model_response.clone()],
                &inference_config,
                InferenceParams::default(),
                None,
            )
            .await
            .unwrap();
        assert_eq!(response.usage_considering_cached(), usage);
        match response {
            InferenceResult::Json(result) => {
                assert_eq!(result.inference_id, inference_id);
                assert!(result.output.parsed.is_none());
                assert_eq!(
                    result.output.raw,
                    Some("{\"name\": \"Jerry\", \"age\": \"thirty\"}".to_string())
                );
                assert_eq!(result.model_inference_results, vec![model_response]);
                assert_eq!(result.finish_reason, Some(FinishReason::ToolCall));
            }
            InferenceResult::Chat(_) => panic!("Expected a JSON inference result"),
        }

        // Test with a tool content block with bad output
        let inference_id = Uuid::now_v7();
        let tool_call = ToolCall {
            id: "tool_call_id".to_string(),
            name: "tool_call_name".to_string(),
            arguments: "tool_call_arguments".to_string(),
        };
        let content_blocks = vec![ContentBlockOutput::ToolCall(tool_call)];
        let usage = Usage {
            input_tokens: Some(10),
            output_tokens: Some(10),
        };
        let model_response = ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            created: Instant::now().elapsed().as_secs(),
            system: None,
            input_messages: RequestMessagesOrBatch::Message(vec![]),
            output: content_blocks.clone(),
            raw_request: raw_request.clone(),
            raw_response: "content".to_string(),
            usage,
            model_provider_name: "model_provider_name".into(),
            model_name: "model_name".into(),
            finish_reason: Some(FinishReason::ToolCall),
            latency: Latency::NonStreaming {
                response_time: Duration::from_millis(100),
            },
            cached: false,
        };
        let response = function_config
            .prepare_response(
                inference_id,
                content_blocks,
                vec![model_response.clone()],
                &inference_config,
                InferenceParams::default(),
                None,
            )
            .await
            .unwrap();
        assert!(logs_contain("JSON Schema validation failed"));
        assert_eq!(response.usage_considering_cached(), usage);
        match response {
            InferenceResult::Json(result) => {
                assert_eq!(result.inference_id, inference_id);
                assert!(result.output.parsed.is_none());
                assert_eq!(result.output.raw, Some("tool_call_arguments".to_string()));
                assert_eq!(result.model_inference_results, vec![model_response]);
                assert_eq!(result.finish_reason, Some(FinishReason::ToolCall));
            }
            InferenceResult::Chat(_) => panic!("Expected a JSON inference result"),
        }

        // Test with a tool content block with good output
        let inference_id = Uuid::now_v7();
        let tool_call = ToolCall {
            id: "tool_call_id".to_string(),
            name: "tool_call_name".to_string(),
            arguments: r#"{"name": "Jerry", "age": 30}"#.to_string(),
        };
        let content_blocks = vec![ContentBlockOutput::ToolCall(tool_call)];
        let usage = Usage {
            input_tokens: Some(10),
            output_tokens: Some(10),
        };
        let model_response = ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            created: Instant::now().elapsed().as_secs(),
            system: None,
            input_messages: RequestMessagesOrBatch::Message(vec![]),
            output: content_blocks.clone(),
            raw_request: raw_request.clone(),
            raw_response: "content".to_string(),
            usage,
            model_provider_name: "model_provider_name".into(),
            model_name: "model_name".into(),
            finish_reason: Some(FinishReason::ContentFilter),
            latency: Latency::NonStreaming {
                response_time: Duration::from_millis(100),
            },
            cached: false,
        };
        let response = function_config
            .prepare_response(
                inference_id,
                content_blocks,
                vec![model_response.clone()],
                &inference_config,
                InferenceParams::default(),
                None,
            )
            .await
            .unwrap();
        assert_eq!(response.usage_considering_cached(), usage);
        match response {
            InferenceResult::Json(result) => {
                assert_eq!(result.inference_id, inference_id);
                assert_eq!(
                    result.output.parsed.unwrap(),
                    json!({"name": "Jerry", "age": 30}),
                );
                assert_eq!(
                    result.output.raw,
                    Some(r#"{"name": "Jerry", "age": 30}"#.to_string())
                );
                assert_eq!(result.model_inference_results, vec![model_response]);
                assert_eq!(result.finish_reason, Some(FinishReason::ContentFilter));
            }
            InferenceResult::Chat(_) => panic!("Expected a JSON inference result"),
        }

        // Test with no content blocks
        let inference_id = Uuid::now_v7();
        let content_blocks = Vec::new();
        let usage = Usage {
            input_tokens: Some(10),
            output_tokens: Some(0),
        };
        let model_response = ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            created: Instant::now().elapsed().as_secs(),
            system: None,
            input_messages: RequestMessagesOrBatch::Message(vec![]),
            output: content_blocks.clone(),
            raw_request: raw_request.clone(),
            raw_response: "content".to_string(),
            usage,
            model_provider_name: "model_provider_name".into(),
            model_name: "model_name".into(),
            finish_reason: Some(FinishReason::Stop),
            latency: Latency::NonStreaming {
                response_time: Duration::from_millis(100),
            },
            cached: false,
        };
        let response = function_config
            .prepare_response(
                inference_id,
                content_blocks,
                vec![model_response.clone()],
                &inference_config,
                InferenceParams::default(),
                None,
            )
            .await
            .unwrap();
        assert_eq!(response.usage_considering_cached(), usage);
        match response {
            InferenceResult::Json(result) => {
                assert_eq!(result.inference_id, inference_id);
                assert!(result.output.parsed.is_none());
                assert!(result.output.raw.is_none());
                assert_eq!(result.finish_reason, model_response.finish_reason);
                assert_eq!(result.model_inference_results, vec![model_response]);
            }
            InferenceResult::Chat(_) => panic!("Expected a JSON inference result"),
        }

        let dynamic_output_schema = DynamicJSONSchema::new(serde_json::json!({
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
            tool_config: None,
            function_name: "".into(),
            variant_name: "".into(),
            templates: templates.clone(),
            dynamic_output_schema: Some(Arc::new(dynamic_output_schema)),
            extra_body: Default::default(),
            extra_headers: Default::default(),
            fetch_and_encode_input_files_before_inference: false,
            extra_cache_key: None,
        };
        // Test with a correct content block
        let inference_id = Uuid::now_v7();
        let content_blocks = vec![r#"{"answer": "42"}"#.to_string().into()];
        let usage = Usage {
            input_tokens: Some(10),
            output_tokens: Some(10),
        };
        let latency = Latency::NonStreaming {
            response_time: Duration::from_millis(100),
        };
        let model_response = ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            created: Instant::now().elapsed().as_secs(),
            system: None,
            input_messages: RequestMessagesOrBatch::Message(vec![]),
            output: content_blocks.clone(),
            raw_request: raw_request.clone(),
            raw_response: "content".to_string(),
            usage,
            model_provider_name: "model_provider_name".into(),
            model_name: "model_name".into(),
            finish_reason: Some(FinishReason::Stop),
            latency,
            cached: false,
        };
        let response = function_config
            .prepare_response(
                inference_id,
                content_blocks,
                vec![model_response.clone()],
                &inference_config,
                InferenceParams::default(),
                None,
            )
            .await
            .unwrap();
        assert_eq!(response.usage_considering_cached(), usage);
        match response {
            InferenceResult::Json(result) => {
                assert_eq!(result.inference_id, inference_id);
                assert_eq!(result.output.parsed.unwrap(), json!({"answer": "42"}),);
                assert_eq!(result.output.raw, Some(r#"{"answer": "42"}"#.to_string()));
                assert_eq!(result.model_inference_results, vec![model_response]);
            }
            InferenceResult::Chat(_) => panic!("Expected a JSON inference result"),
        }

        // Test with an incorrect JSON content block
        let inference_id = Uuid::now_v7();
        let content_blocks = vec![r#"{"response": "forty-two"}"#.to_string().into()];
        let usage = Usage {
            input_tokens: Some(10),
            output_tokens: Some(10),
        };
        let latency = Latency::NonStreaming {
            response_time: Duration::from_millis(100),
        };
        let model_response = ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            created: Instant::now().elapsed().as_secs(),
            system: None,
            input_messages: RequestMessagesOrBatch::Message(vec![]),
            output: content_blocks.clone(),
            raw_request: raw_request.clone(),
            raw_response: "content".to_string(),
            usage,
            model_provider_name: "model_provider_name".into(),
            model_name: "model_name".into(),
            finish_reason: None,
            latency,
            cached: false,
        };
        let response = function_config
            .prepare_response(
                inference_id,
                content_blocks,
                vec![model_response.clone()],
                &inference_config,
                InferenceParams::default(),
                None,
            )
            .await
            .unwrap();
        assert_eq!(response.usage_considering_cached(), usage);
        match response {
            InferenceResult::Json(result) => {
                assert_eq!(result.inference_id, inference_id);
                assert!(result.output.parsed.is_none());
                assert_eq!(
                    result.output.raw,
                    Some(r#"{"response": "forty-two"}"#.to_string())
                );
                assert_eq!(result.model_inference_results, vec![model_response]);
            }
            InferenceResult::Chat(_) => panic!("Expected a JSON inference result"),
        }

        // Test with a tool content block with bad output
        let inference_id = Uuid::now_v7();
        let tool_call = ToolCall {
            id: "tool_call_id".to_string(),
            name: "tool_call_name".to_string(),
            arguments: "tool_call_arguments".to_string(),
        };
        let content_blocks = vec![ContentBlockOutput::ToolCall(tool_call)];
        let usage = Usage {
            input_tokens: Some(10),
            output_tokens: Some(10),
        };
        let model_response = ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            created: Instant::now().elapsed().as_secs(),
            system: None,
            input_messages: RequestMessagesOrBatch::Message(vec![]),
            output: content_blocks.clone(),
            raw_request: raw_request.clone(),
            raw_response: "content".to_string(),
            usage,
            model_provider_name: "model_provider_name".into(),
            model_name: "model_name".into(),
            finish_reason: Some(FinishReason::ToolCall),
            latency: Latency::NonStreaming {
                response_time: Duration::from_millis(100),
            },
            cached: false,
        };
        let response = function_config
            .prepare_response(
                inference_id,
                content_blocks,
                vec![model_response.clone()],
                &inference_config,
                InferenceParams::default(),
                None,
            )
            .await
            .unwrap();
        assert!(logs_contain("JSON Schema validation failed"));
        assert_eq!(response.usage_considering_cached(), usage);
        match response {
            InferenceResult::Json(result) => {
                assert_eq!(result.inference_id, inference_id);
                assert!(result.output.parsed.is_none());
                assert_eq!(result.output.raw, Some("tool_call_arguments".to_string()));
                assert_eq!(result.model_inference_results, vec![model_response]);
            }
            InferenceResult::Chat(_) => panic!("Expected a JSON inference result"),
        }

        // Test with a tool content block with good output
        let inference_id = Uuid::now_v7();
        let tool_call = ToolCall {
            id: "tool_call_id".to_string(),
            name: "tool_call_name".to_string(),
            arguments: r#"{"answer": "42"}"#.to_string(),
        };
        let content_blocks = vec![ContentBlockOutput::ToolCall(tool_call)];
        let usage = Usage {
            input_tokens: Some(10),
            output_tokens: Some(10),
        };
        let model_response = ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            created: Instant::now().elapsed().as_secs(),
            system: None,
            input_messages: RequestMessagesOrBatch::Message(vec![]),
            output: content_blocks.clone(),
            raw_request: raw_request.clone(),
            raw_response: "content".to_string(),
            usage,
            model_provider_name: "model_provider_name".into(),
            model_name: "model_name".into(),
            finish_reason: None,
            latency: Latency::NonStreaming {
                response_time: Duration::from_millis(100),
            },
            cached: false,
        };
        let response = function_config
            .prepare_response(
                inference_id,
                content_blocks,
                vec![model_response.clone()],
                &inference_config,
                InferenceParams::default(),
                None,
            )
            .await
            .unwrap();
        assert_eq!(response.usage_considering_cached(), usage);
        match response {
            InferenceResult::Json(result) => {
                assert_eq!(result.inference_id, inference_id);
                assert_eq!(result.output.parsed.unwrap(), json!({"answer": "42"}),);
                assert_eq!(result.output.raw, Some(r#"{"answer": "42"}"#.to_string()));
                assert_eq!(result.model_inference_results, vec![model_response]);
            }
            InferenceResult::Chat(_) => panic!("Expected a JSON inference result"),
        }

        // Test with an empty output schema
        let output_schema = json!({});
        let json_mode_tool_call_config = ToolCallConfig::implicit_from_value(&output_schema);
        let output_schema = StaticJSONSchema::from_value(output_schema).unwrap();
        let function_config = FunctionConfig::Json(FunctionConfigJson {
            variants: HashMap::new(),
            schemas: SchemaData::default(),
            output_schema,
            json_mode_tool_call_config,
            description: None,
            all_explicit_template_names: HashSet::new(),
            experimentation: ExperimentationConfig::default(),
        });
        let inference_id = Uuid::now_v7();
        let content_blocks = vec![r#"{"answer": "42"}"#.to_string().into()];
        let usage = Usage {
            input_tokens: Some(10),
            output_tokens: Some(10),
        };
        let latency = Latency::NonStreaming {
            response_time: Duration::from_millis(100),
        };
        let model_response = ModelInferenceResponseWithMetadata {
            id: Uuid::now_v7(),
            created: Instant::now().elapsed().as_secs(),
            system: None,
            input_messages: RequestMessagesOrBatch::Message(vec![]),
            output: content_blocks.clone(),
            raw_request: raw_request.clone(),
            raw_response: "content".to_string(),
            usage,
            model_provider_name: "model_provider_name".into(),
            model_name: "model_name".into(),
            finish_reason: Some(FinishReason::Stop),
            latency,
            cached: false,
        };
        let response = function_config
            .prepare_response(
                inference_id,
                content_blocks,
                vec![model_response.clone()],
                &inference_config,
                InferenceParams::default(),
                None,
            )
            .await
            .unwrap();
        assert_eq!(response.usage_considering_cached(), usage);
        match response {
            InferenceResult::Json(result) => {
                assert_eq!(result.inference_id, inference_id);
                assert_eq!(result.output.parsed.unwrap(), json!({"answer": "42"}),);
                assert_eq!(result.output.raw, Some(r#"{"answer": "42"}"#.to_string()));
                assert_eq!(result.model_inference_results, vec![model_response]);
                assert_eq!(result.finish_reason, Some(FinishReason::Stop));
            }
            InferenceResult::Chat(_) => panic!("Expected a JSON inference result"),
        }
    }

    #[test]
    fn test_get_json_output_from_content_blocks() {
        // Case 1: Text followed by ToolCall
        let content_blocks = vec![
            ContentBlockOutput::Text(Text {
                text: "Hello".to_string(),
            }),
            ContentBlockOutput::ToolCall(ToolCall {
                id: "tool_call_id".to_string(),
                name: "tool_call_name".to_string(),
                arguments: "tool_call_arguments".to_string(),
            }),
        ];
        let (raw_output, auxiliary_content, json_block_index) =
            get_json_output_from_content_blocks(content_blocks.clone());
        assert_eq!(raw_output, Some("tool_call_arguments".to_string()));
        assert_eq!(auxiliary_content.len(), 1);
        assert_eq!(json_block_index, Some(1));
        match &auxiliary_content[0] {
            ContentBlockOutput::Text(t) => assert_eq!(t.text, "Hello"),
            _ => panic!("Expected Text block"),
        }

        // Case 2: Only Thought blocks
        let content_blocks = vec![
            ContentBlockOutput::Thought(Thought {
                text: Some("thinking...".to_string()),
                signature: None,
                summary: None,
                provider_type: None,
            }),
            ContentBlockOutput::Thought(Thought {
                text: Some("still thinking".to_string()),
                signature: Some("sig".to_string()),
                summary: None,
                provider_type: None,
            }),
        ];
        let (raw_output, auxiliary_content, json_block_index) =
            get_json_output_from_content_blocks(content_blocks.clone());
        assert_eq!(raw_output, None);
        assert_eq!(auxiliary_content, content_blocks);
        assert_eq!(json_block_index, None);

        // Case 3: Mixed Text, Thought, ToolCall
        let content_blocks = vec![
            ContentBlockOutput::Thought(Thought {
                text: Some("first thought".to_string()),
                signature: None,
                summary: None,
                provider_type: None,
            }),
            ContentBlockOutput::Text(Text {
                text: "Some text".to_string(),
            }),
            ContentBlockOutput::Thought(Thought {
                text: Some("second thought".to_string()),
                signature: Some("sig2".to_string()),
                summary: None,
                provider_type: None,
            }),
            ContentBlockOutput::ToolCall(ToolCall {
                id: "id2".to_string(),
                name: "name2".to_string(),
                arguments: "{\"foo\": 1}".to_string(),
            }),
        ];
        let (raw_output, auxiliary_content, json_block_index) =
            get_json_output_from_content_blocks(content_blocks.clone());
        assert_eq!(raw_output, Some("{\"foo\": 1}".to_string()));
        assert_eq!(json_block_index, Some(3));
        // Should exclude the ToolCall block from auxiliary_content
        assert_eq!(auxiliary_content.len(), 3);
        assert!(auxiliary_content
            .iter()
            .any(|b| matches!(b, ContentBlockOutput::Text(_))));
        assert_eq!(
            auxiliary_content
                .iter()
                .filter(|b| matches!(b, ContentBlockOutput::Thought(_)))
                .count(),
            2
        );

        // Case 4: Only Text blocks
        let content_blocks = vec![
            ContentBlockOutput::Text(Text {
                text: "A".to_string(),
            }),
            ContentBlockOutput::Text(Text {
                text: "B".to_string(),
            }),
        ];
        let (raw_output, auxiliary_content, json_block_index) =
            get_json_output_from_content_blocks(content_blocks.clone());
        assert_eq!(raw_output, Some("B".to_string()));
        assert_eq!(auxiliary_content.len(), 1);
        assert_eq!(json_block_index, Some(1));
        match &auxiliary_content[0] {
            ContentBlockOutput::Text(t) => assert_eq!(t.text, "A"),
            _ => panic!("Expected Text block"),
        }

        // Case 5: Thought block at the end
        let content_blocks = vec![
            ContentBlockOutput::Text(Text {
                text: "A".to_string(),
            }),
            ContentBlockOutput::Thought(Thought {
                text: Some("final thought".to_string()),
                signature: None,
                summary: None,
                provider_type: None,
            }),
        ];
        let (raw_output, auxiliary_content, json_block_index) =
            get_json_output_from_content_blocks(content_blocks.clone());
        assert_eq!(raw_output, Some("A".to_string()));
        assert_eq!(auxiliary_content.len(), 1);
        assert_eq!(json_block_index, Some(0));
        match &auxiliary_content[0] {
            ContentBlockOutput::Thought(t) => {
                assert_eq!(t.text, Some("final thought".to_string()));
            }
            _ => panic!("Expected Thought block"),
        }
    }
}
