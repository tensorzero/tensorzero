use std::{collections::HashMap, sync::Arc};

use crate::config::Config;
use crate::db::datasets::{
    ChatInferenceDatapointInsert, DatapointInsert, JsonInferenceDatapointInsert,
};
use crate::endpoints::datasets::v1::types::{
    CreateChatDatapointRequest, CreateDatapointRequest, CreateDatapointsFromInferenceOutputSource,
    CreateJsonDatapointRequest, JsonDatapointOutputUpdate,
};
use crate::error::{Error, ErrorDetails};
use crate::function::FunctionConfig;
#[cfg(feature = "pyo3")]
use crate::inference::types::pyo3_helpers::{
    content_block_chat_output_to_python, serialize_to_dict, uuid_to_python,
};
use crate::inference::types::stored_input::StoredInput;
use crate::inference::types::{
    ContentBlockChatOutput, JsonInferenceOutput, ModelInput, RequestMessage, ResolvedInput,
    ResolvedRequestMessage, Text,
};
use crate::tool::{
    deserialize_tool_info, DynamicToolParams, StaticToolConfig, ToolCallConfigDatabaseInsert,
};
use crate::variant::{chat_completion::prepare_model_input, VariantConfig};
use chrono::{DateTime, Utc};
#[cfg(feature = "pyo3")]
use pyo3::types::PyList;
#[cfg(feature = "pyo3")]
use pyo3::{prelude::*, IntoPyObjectExt};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

/// This trait is used to represent a stored sample of data.
/// It should contain all the methods used by `render_samples`
/// from the stored sample of data so that we can abstract over the
/// different places where we could get training samples from, notably
/// datasets and stored inferences.
pub trait StoredSample {
    fn function_name(&self) -> &str;
    fn into_input(self) -> StoredInput;
    fn input(&self) -> &StoredInput;
    fn input_mut(&mut self) -> &mut StoredInput;
    fn owned_simple_info(self) -> SimpleStoredSampleInfo;
}

/// Utility struct that contains the information needed for a RenderedSample
/// that is just copied over from the StoredSample.
pub struct SimpleStoredSampleInfo {
    pub function_name: String,
    pub input: StoredInput,
    pub episode_id: Option<Uuid>,
    pub inference_id: Option<Uuid>,
    pub output: Option<Vec<ContentBlockChatOutput>>,
    pub stored_output: Option<StoredOutput>,
    pub dispreferred_outputs: Vec<Vec<ContentBlockChatOutput>>,
    pub tool_params: Option<ToolCallConfigDatabaseInsert>,
    pub output_schema: Option<Value>,
    pub tags: HashMap<String, String>,
}

/// Wire variant of StoredInference for API responses with Python/TypeScript bindings
/// This one should be used in all public interfaces
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, JsonSchema, ts_rs::TS)]
#[serde(tag = "type", rename_all = "snake_case")]
#[ts(export)]
pub enum StoredInference {
    #[schemars(title = "StoredInferenceChat")]
    Chat(StoredChatInference),
    #[schemars(title = "StoredInferenceJson")]
    Json(StoredJsonInference),
}

impl std::fmt::Display for StoredInference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

impl StoredInference {
    pub fn id(&self) -> Uuid {
        match self {
            StoredInference::Json(inference) => inference.inference_id,
            StoredInference::Chat(inference) => inference.inference_id,
        }
    }

    /// Convert a StoredInference to a DatapointInsert. Generates a new datapoint ID in the process.
    /// The output_source parameter allows overriding to None even if the inference has an output.
    pub fn into_datapoint_insert(
        self,
        dataset_name: &str,
        output_source: &CreateDatapointsFromInferenceOutputSource,
        config: &Config,
    ) -> Result<DatapointInsert, Error> {
        let datapoint_id = Uuid::now_v7();

        match self {
            StoredInference::Json(inference) => {
                let output = match output_source {
                    CreateDatapointsFromInferenceOutputSource::None => None,
                    CreateDatapointsFromInferenceOutputSource::Inference => Some(inference.output),
                    CreateDatapointsFromInferenceOutputSource::Demonstration => {
                        Some(inference.output)
                    }
                };

                let datapoint = JsonInferenceDatapointInsert {
                    dataset_name: dataset_name.to_string(),
                    function_name: inference.function_name,
                    name: None,
                    id: datapoint_id,
                    episode_id: Some(inference.episode_id),
                    input: inference.input,
                    output,
                    output_schema: inference.output_schema,
                    tags: Some(inference.tags),
                    auxiliary: String::new(),
                    staled_at: None,
                    source_inference_id: Some(inference.inference_id),
                    is_custom: false,
                };

                Ok(DatapointInsert::Json(datapoint))
            }
            StoredInference::Chat(inference) => {
                let output = match output_source {
                    CreateDatapointsFromInferenceOutputSource::None => None,
                    CreateDatapointsFromInferenceOutputSource::Inference => Some(inference.output),
                    CreateDatapointsFromInferenceOutputSource::Demonstration => {
                        Some(inference.output)
                    }
                };

                // Convert DynamicToolParams (wire type) to ToolCallConfigDatabaseInsert (storage type)
                let function_config = config.get_function(&inference.function_name)?;
                let tool_params = function_config
                    .dynamic_tool_params_to_database_insert(inference.tool_params, &config.tools)?
                    .unwrap_or_default();

                let datapoint = ChatInferenceDatapointInsert {
                    dataset_name: dataset_name.to_string(),
                    function_name: inference.function_name,
                    name: None,
                    id: datapoint_id,
                    episode_id: Some(inference.episode_id),
                    input: inference.input,
                    output,
                    tool_params: Some(tool_params),
                    tags: Some(inference.tags),
                    auxiliary: String::new(),
                    staled_at: None,
                    source_inference_id: Some(inference.inference_id),
                    is_custom: false,
                };

                Ok(DatapointInsert::Chat(datapoint))
            }
        }
    }
}

impl StoredInferenceDatabase {
    /// Convert to wire type, properly handling tool params by subtracting static tools
    pub fn into_stored_inference(self) -> Result<StoredInference, Error> {
        match self {
            StoredInferenceDatabase::Chat(chat) => {
                Ok(StoredInference::Chat(chat.into_stored_inference()))
            }
            StoredInferenceDatabase::Json(json) => Ok(StoredInference::Json(json)),
        }
    }
}

impl StoredInference {
    /// Convert to storage type, converting tool params from wire format to storage format
    pub fn to_storage(self, config: &Config) -> Result<StoredInferenceDatabase, Error> {
        match self {
            StoredInference::Chat(chat) => {
                let function_config = config.get_function(&chat.function_name)?;
                Ok(StoredInferenceDatabase::Chat(
                    chat.to_storage(&function_config, &config.tools)?,
                ))
            }
            StoredInference::Json(json) => Ok(StoredInferenceDatabase::Json(json)),
        }
    }
}

impl StoredChatInference {
    /// Convert to storage type, properly handling tool params with function config
    pub fn to_storage(
        self,
        function_config: &FunctionConfig,
        static_tools: &HashMap<String, Arc<StaticToolConfig>>,
    ) -> Result<StoredChatInferenceDatabase, Error> {
        let tool_params = function_config
            .dynamic_tool_params_to_database_insert(self.tool_params, static_tools)?
            .unwrap_or_default();

        Ok(StoredChatInferenceDatabase {
            function_name: self.function_name,
            variant_name: self.variant_name,
            input: self.input,
            output: self.output,
            dispreferred_outputs: self.dispreferred_outputs,
            timestamp: self.timestamp,
            episode_id: self.episode_id,
            inference_id: self.inference_id,
            tool_params,
            tags: self.tags,
        })
    }
}

/// Storage variant of StoredInference for database operations (no Python/TypeScript bindings)
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StoredInferenceDatabase {
    Chat(StoredChatInferenceDatabase),
    Json(StoredJsonInference),
}

impl std::fmt::Display for StoredInferenceDatabase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

impl StoredInferenceDatabase {
    pub fn id(&self) -> Uuid {
        match self {
            StoredInferenceDatabase::Json(inference) => inference.inference_id,
            StoredInferenceDatabase::Chat(inference) => inference.inference_id,
        }
    }
}

/// Wire variant of StoredChatInference for API responses with Python/TypeScript bindings
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, JsonSchema, ts_rs::TS)]
#[ts(export)]
pub struct StoredChatInference {
    pub function_name: String,
    pub variant_name: String,
    pub input: StoredInput,
    pub output: Vec<ContentBlockChatOutput>,
    #[serde(default)]
    pub dispreferred_outputs: Vec<Vec<ContentBlockChatOutput>>,
    #[schemars(with = "String")]
    pub timestamp: DateTime<Utc>,
    pub episode_id: Uuid,
    pub inference_id: Uuid,
    #[serde(flatten)]
    #[serde(default)]
    pub tool_params: DynamicToolParams,
    #[serde(default)]
    pub tags: HashMap<String, String>,
}

impl std::fmt::Display for StoredChatInference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

impl StoredChatInferenceDatabase {
    /// Convert to wire type, converting tool params from storage format to wire format
    pub fn into_stored_inference(self) -> StoredChatInference {
        StoredChatInference {
            function_name: self.function_name,
            variant_name: self.variant_name,
            input: self.input,
            output: self.output,
            dispreferred_outputs: self.dispreferred_outputs,
            timestamp: self.timestamp,
            episode_id: self.episode_id,
            inference_id: self.inference_id,
            tool_params: self.tool_params.into(),
            tags: self.tags,
        }
    }
}

/// Storage variant of StoredChatInference for database operations (no Python/TypeScript bindings)
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct StoredChatInferenceDatabase {
    pub function_name: String,
    pub variant_name: String,
    pub input: StoredInput,
    pub output: Vec<ContentBlockChatOutput>,
    #[serde(default)]
    pub dispreferred_outputs: Vec<Vec<ContentBlockChatOutput>>,
    pub timestamp: DateTime<Utc>,
    pub episode_id: Uuid,
    pub inference_id: Uuid,
    #[serde(flatten, deserialize_with = "deserialize_tool_info")]
    pub tool_params: ToolCallConfigDatabaseInsert,
    #[serde(default)]
    pub tags: HashMap<String, String>,
}

impl std::fmt::Display for StoredChatInferenceDatabase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, JsonSchema, ts_rs::TS)]
#[ts(export)]
pub struct StoredJsonInference {
    pub function_name: String,
    pub variant_name: String,
    pub input: StoredInput,
    pub output: JsonInferenceOutput,
    #[serde(default)]
    pub dispreferred_outputs: Vec<JsonInferenceOutput>,
    #[schemars(with = "String")]
    pub timestamp: DateTime<Utc>,
    pub episode_id: Uuid,
    pub inference_id: Uuid,
    pub output_schema: Value,
    #[serde(default)]
    pub tags: HashMap<String, String>,
}

impl std::fmt::Display for StoredJsonInference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

impl StoredSample for StoredInferenceDatabase {
    fn input_mut(&mut self) -> &mut StoredInput {
        match self {
            StoredInferenceDatabase::Chat(example) => &mut example.input,
            StoredInferenceDatabase::Json(example) => &mut example.input,
        }
    }
    fn input(&self) -> &StoredInput {
        match self {
            StoredInferenceDatabase::Chat(example) => &example.input,
            StoredInferenceDatabase::Json(example) => &example.input,
        }
    }

    fn into_input(self) -> StoredInput {
        match self {
            StoredInferenceDatabase::Chat(example) => example.input,
            StoredInferenceDatabase::Json(example) => example.input,
        }
    }

    fn function_name(&self) -> &str {
        match self {
            StoredInferenceDatabase::Chat(example) => &example.function_name,
            StoredInferenceDatabase::Json(example) => &example.function_name,
        }
    }

    fn owned_simple_info(self) -> SimpleStoredSampleInfo {
        match self {
            StoredInferenceDatabase::Chat(example) => SimpleStoredSampleInfo {
                function_name: example.function_name,
                input: example.input,
                episode_id: Some(example.episode_id),
                inference_id: Some(example.inference_id),
                output: Some(example.output.clone()),
                stored_output: Some(StoredOutput::Chat(example.output)),
                dispreferred_outputs: example.dispreferred_outputs,
                tool_params: Some(example.tool_params),
                output_schema: None,
                tags: example.tags,
            },
            StoredInferenceDatabase::Json(example) => {
                let output = json_output_to_content_block_chat_output(example.output.clone());
                let dispreferred_outputs = example
                    .dispreferred_outputs
                    .into_iter()
                    .map(json_output_to_content_block_chat_output)
                    .collect();
                SimpleStoredSampleInfo {
                    function_name: example.function_name,
                    input: example.input,
                    episode_id: Some(example.episode_id),
                    inference_id: Some(example.inference_id),
                    output: Some(output),
                    stored_output: Some(StoredOutput::Json(example.output)),
                    dispreferred_outputs,
                    tool_params: None,
                    output_schema: Some(example.output_schema),
                    tags: example.tags,
                }
            }
        }
    }
}

fn json_output_to_content_block_chat_output(
    output: JsonInferenceOutput,
) -> Vec<ContentBlockChatOutput> {
    match output.raw {
        Some(raw) => vec![ContentBlockChatOutput::Text(Text { text: raw })],
        None => vec![],
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, ts_rs::TS)]
#[ts(export)]
#[serde(untagged)]
pub enum StoredOutput {
    Chat(Vec<ContentBlockChatOutput>),
    Json(JsonInferenceOutput),
}

/// Represents an inference that has been prepared for fine-tuning.
/// This is constructed by rendering a StoredInference with a variant for messages
/// and by resolving all network resources (e.g. images).
/// This is a wire type - it uses DynamicToolParams and has Python/TypeScript bindings.
#[cfg_attr(feature = "pyo3", pyclass(str))]
#[derive(Clone, Debug, Serialize, Deserialize, ts_rs::TS)]
#[cfg_attr(any(feature = "e2e_tests", test), derive(PartialEq))]
#[ts(export)]
pub struct RenderedSample {
    pub function_name: String,
    pub input: ModelInput,
    pub stored_input: StoredInput,
    pub output: Option<Vec<ContentBlockChatOutput>>,
    pub stored_output: Option<StoredOutput>,
    pub dispreferred_outputs: Vec<Vec<ContentBlockChatOutput>>,
    pub episode_id: Option<Uuid>,
    pub inference_id: Option<Uuid>,
    pub tool_params: DynamicToolParams,
    pub output_schema: Option<Value>,
    pub tags: HashMap<String, String>,
}

impl RenderedSample {
    pub fn into_lazy_rendered_sample(self) -> LazyRenderedSample {
        LazyRenderedSample {
            function_name: self.function_name,
            system_input: self.input.system,
            messages: self
                .input
                .messages
                .into_iter()
                .map(ResolvedRequestMessage::into_request_message)
                .collect(),
            stored_input: self.stored_input,
            output: self.output,
            stored_output: self.stored_output,
            dispreferred_outputs: self.dispreferred_outputs,
            episode_id: self.episode_id,
            inference_id: self.inference_id,
            tool_params: self.tool_params,
            output_schema: self.output_schema,
            tags: self.tags,
        }
    }

    /// Convert this RenderedSample into a CreateDatapointRequest for use with the datasets v1 API.
    ///
    /// This method handles the conversion from RenderedSample (which has StoredInput and StoredOutput)
    /// to CreateDatapointRequest (which expects Input and type-specific output).
    ///
    /// The type discrimination (Chat vs JSON) is based on the stored_output enum variant.
    pub fn into_create_datapoint_request(self) -> Result<CreateDatapointRequest, Error> {
        // Convert StoredInput to Input
        let input = self.stored_input.into_input();

        // Use stored_output to determine whether this is a Chat or JSON datapoint
        match self.stored_output {
            Some(StoredOutput::Json(json_output)) => {
                // JSON function datapoint
                let output = json_output
                    .raw
                    .map(|raw| JsonDatapointOutputUpdate { raw: Some(raw) });

                Ok(CreateDatapointRequest::Json(CreateJsonDatapointRequest {
                    function_name: self.function_name,
                    episode_id: self.episode_id,
                    input,
                    output,
                    output_schema: self.output_schema,
                    tags: Some(self.tags),
                    name: None,
                }))
            }
            Some(StoredOutput::Chat(_)) | None => {
                // Chat function datapoint
                Ok(CreateDatapointRequest::Chat(CreateChatDatapointRequest {
                    function_name: self.function_name,
                    episode_id: self.episode_id,
                    input,
                    output: self.output,
                    dynamic_tool_params: self.tool_params,
                    tags: Some(self.tags),
                    name: None,
                }))
            }
        }
    }
}

/// Like `RenderedSample`, but holds `RequestMessage`s instead of `ResolvedRequestMessage`s
pub struct LazyRenderedSample {
    pub function_name: String,
    pub system_input: Option<String>,
    // This is a a `Vec<ResolvedRequestMessage>` in `RenderedSample`
    pub messages: Vec<RequestMessage>,
    pub stored_input: StoredInput,
    pub output: Option<Vec<ContentBlockChatOutput>>,
    pub stored_output: Option<StoredOutput>,
    pub dispreferred_outputs: Vec<Vec<ContentBlockChatOutput>>,
    pub episode_id: Option<Uuid>,
    pub inference_id: Option<Uuid>,
    pub tool_params: DynamicToolParams,
    pub output_schema: Option<Value>,
    pub tags: HashMap<String, String>,
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl RenderedSample {
    #[getter]
    pub fn get_function_name(&self) -> &str {
        &self.function_name
    }

    #[getter]
    pub fn get_input(&self) -> ModelInput {
        self.input.clone()
    }

    #[getter]
    pub fn get_output<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if let Some(output) = &self.output {
            let output = output
                .iter()
                .map(|x| content_block_chat_output_to_python(py, x.clone()))
                .collect::<PyResult<Vec<_>>>()?;
            PyList::new(py, output).map(Bound::into_any)
        } else {
            Ok(py.None().into_bound(py))
        }
    }

    #[getter]
    pub fn get_stored_output<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if let Some(stored_output) = &self.stored_output {
            match stored_output {
                StoredOutput::Chat(output) => {
                    let output = output
                        .iter()
                        .map(|x| content_block_chat_output_to_python(py, x.clone()))
                        .collect::<PyResult<Vec<_>>>()?;
                    PyList::new(py, output).map(Bound::into_any)
                }
                StoredOutput::Json(output) => Ok(output.clone().into_py_any(py)?.into_bound(py)),
            }
        } else {
            Ok(py.None().into_bound(py))
        }
    }

    #[getter]
    pub fn get_dispreferred_outputs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let dispreferred_outputs = self
            .dispreferred_outputs
            .iter()
            .map(|x| {
                x.iter()
                    .map(|y| content_block_chat_output_to_python(py, y.clone()))
                    .collect::<PyResult<Vec<_>>>()
            })
            .collect::<PyResult<Vec<_>>>()?;
        PyList::new(py, dispreferred_outputs).map(Bound::into_any)
    }

    #[getter]
    pub fn get_output_schema<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        serialize_to_dict(py, self.output_schema.clone()).map(|x| x.into_bound(py))
    }

    #[getter]
    pub fn get_allowed_tools(&self) -> Option<Vec<String>> {
        self.tool_params.allowed_tools.clone()
    }

    #[getter]
    pub fn get_additional_tools<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.tool_params
            .additional_tools
            .clone()
            .into_bound_py_any(py)
    }

    // Note: We're intentionally skipping tool_choice as it's not exposed in the Python API

    #[getter]
    pub fn get_parallel_tool_calls(&self) -> Option<bool> {
        self.tool_params.parallel_tool_calls
    }

    #[getter]
    pub fn get_provider_tools<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.tool_params
            .provider_tools
            .clone()
            .into_bound_py_any(py)
    }

    #[getter]
    pub fn get_episode_id<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self.episode_id {
            Some(id) => uuid_to_python(py, id),
            None => Ok(py.None().into_bound(py)),
        }
    }

    #[getter]
    pub fn get_inference_id<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self.inference_id {
            Some(id) => uuid_to_python(py, id),
            None => Ok(py.None().into_bound(py)),
        }
    }

    pub fn __repr__(&self) -> String {
        self.to_string()
    }

    #[getter]
    pub fn get_tags(&self) -> HashMap<String, String> {
        self.tags.clone()
    }

    #[getter]
    pub fn get_stored_input(&self) -> StoredInput {
        self.stored_input.clone()
    }
}

impl std::fmt::Display for RenderedSample {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Serialize the rendered inference to pretty-printed JSON
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

/// Convert a StoredInference's input to a ModelInput.
/// `variants` should be a map from function name to variant name, i.e. what variant to use for a particular function
/// as the stored inference is being rendered.
/// This does not handle resolving network resources (e.g. images).
async fn render_model_input(
    resolved_input: &ResolvedInput,
    function_name: &str,
    config: &Config,
    variants: &HashMap<String, String>,
) -> Result<ModelInput, Error> {
    let variant_name = variants.get(function_name).ok_or_else(|| {
        Error::new(ErrorDetails::MissingFunctionInVariants {
            function_name: function_name.to_string(),
        })
    })?;
    let function_config = config.get_function(function_name)?;
    let variant_config = function_config
        .variants()
        .get(variant_name)
        .ok_or_else(|| {
            Error::new(ErrorDetails::UnknownVariant {
                name: variant_name.clone(),
            })
        })?;
    let VariantConfig::ChatCompletion(chat_completion_config) = &variant_config.inner else {
        return Err(Error::new(ErrorDetails::InvalidVariantForOptimization {
            function_name: function_name.to_string(),
            variant_name: variant_name.clone(),
        }));
    };
    prepare_model_input(
        resolved_input.system.as_ref(),
        &resolved_input.messages,
        &config.templates,
        chat_completion_config.templates(),
    )
    .await
}

/// Render an impl StoredSample to a RenderedStoredInference.
/// `variants` should be a map from function name to variant name, i.e. what variant to use for a particular function
/// as the inference example is being rendered.
///
/// This does not handle resolving network resources (e.g. images).
pub async fn render_stored_sample<T: StoredSample>(
    stored_sample: T,
    resolved_input: ResolvedInput,
    config: &Config,
    variants: &HashMap<String, String>,
) -> Result<RenderedSample, Error> {
    let SimpleStoredSampleInfo {
        function_name,
        input: _,
        output,
        stored_output,
        dispreferred_outputs,
        tool_params,
        output_schema,
        episode_id,
        inference_id,
        tags,
    } = stored_sample.owned_simple_info();
    let model_input = render_model_input(&resolved_input, &function_name, config, variants).await?;

    // Convert tool_params from storage format to wire format
    let dynamic_tool_params = tool_params
        .map(|tp| tp.into())
        // should default for JSON functions or functions with no tools to a default DynamicToolParams
        // where everything is empty
        .unwrap_or_default();

    Ok(RenderedSample {
        function_name,
        episode_id,
        inference_id,
        input: model_input,
        stored_input: resolved_input.into_stored_input(),
        output,
        stored_output,
        dispreferred_outputs,
        tool_params: dynamic_tool_params,
        output_schema,
        tags,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, SchemaData};
    use crate::db::datasets::DatapointInsert;
    use crate::endpoints::datasets::v1::types::CreateDatapointsFromInferenceOutputSource;
    use crate::experimentation::ExperimentationConfig;
    use crate::function::{FunctionConfig, FunctionConfigChat, FunctionConfigJson};
    use crate::inference::types::System;
    use crate::inference::types::{ContentBlockChatOutput, JsonInferenceOutput, Text};
    use crate::jsonschema_util::StaticJSONSchema;
    use crate::tool::{DynamicToolParams, ToolCallConfig, ToolChoice};
    use std::sync::Arc;

    /// Helper to create a test config with the functions registered
    fn create_test_config() -> Config {
        let mut config = Config::default();

        // Add the test_function (Chat function)
        config.functions.insert(
            "test_function".to_string(),
            Arc::new(FunctionConfig::Chat(FunctionConfigChat {
                variants: Default::default(),
                schemas: SchemaData::default(),
                tools: vec![],
                tool_choice: ToolChoice::Auto,
                parallel_tool_calls: None,
                description: None,
                experimentation: ExperimentationConfig::default(),
                all_explicit_templates_names: Default::default(),
            })),
        );

        // Add the json_function (Json function)
        config.functions.insert(
            "json_function".to_string(),
            Arc::new(FunctionConfig::Json(FunctionConfigJson {
                variants: Default::default(),
                schemas: SchemaData::default(),
                output_schema: StaticJSONSchema::default(),
                json_mode_tool_call_config: ToolCallConfig::default(),
                description: None,
                experimentation: ExperimentationConfig::default(),
                all_explicit_template_names: Default::default(),
            })),
        );

        config
    }

    /// Helper to create a test StoredChatInference with all fields populated
    fn create_test_chat_inference() -> StoredChatInference {
        let inference_id = Uuid::now_v7();
        let episode_id = Uuid::now_v7();

        StoredChatInference {
            function_name: "test_function".to_string(),
            variant_name: "test_variant".to_string(),
            input: StoredInput {
                system: Some(System::Text("Test system prompt".to_string())),
                messages: vec![],
            },
            output: vec![
                ContentBlockChatOutput::Text(Text {
                    text: "Test output 1".to_string(),
                }),
                ContentBlockChatOutput::Text(Text {
                    text: "Test output 2".to_string(),
                }),
            ],
            dispreferred_outputs: vec![],
            timestamp: DateTime::parse_from_rfc3339("2024-01-01T00:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
            episode_id,
            inference_id,
            tool_params: DynamicToolParams::default(),
            tags: {
                let mut tags = HashMap::new();
                tags.insert("key1".to_string(), "value1".to_string());
                tags.insert("key2".to_string(), "value2".to_string());
                tags
            },
        }
    }

    /// Helper to create a test StoredJsonInference with all fields populated
    fn create_test_json_inference() -> StoredJsonInference {
        let inference_id = Uuid::now_v7();
        let episode_id = Uuid::now_v7();

        StoredJsonInference {
            function_name: "json_function".to_string(),
            variant_name: "json_variant".to_string(),
            input: StoredInput {
                system: Some(System::Text("JSON system prompt".to_string())),
                messages: vec![],
            },
            output: JsonInferenceOutput {
                raw: Some(r#"{"result": "test"}"#.to_string()),
                parsed: Some(serde_json::json!({"result": "test"})),
            },
            dispreferred_outputs: vec![],
            timestamp: DateTime::parse_from_rfc3339("2024-01-01T00:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
            episode_id,
            inference_id,
            output_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "result": {"type": "string"}
                }
            }),
            tags: {
                let mut tags = HashMap::new();
                tags.insert("json_key".to_string(), "json_value".to_string());
                tags
            },
        }
    }

    /// Helper to create a test RenderedSample for Chat function
    fn create_test_chat_rendered_sample() -> RenderedSample {
        let inference_id = Uuid::now_v7();
        let episode_id = Uuid::now_v7();

        RenderedSample {
            function_name: "test_function".to_string(),
            input: ModelInput {
                system: Some("Test system prompt".to_string()),
                messages: vec![],
            },
            stored_input: StoredInput {
                system: Some(System::Text("Test system prompt".to_string())),
                messages: vec![],
            },
            output: Some(vec![
                ContentBlockChatOutput::Text(Text {
                    text: "Test output 1".to_string(),
                }),
                ContentBlockChatOutput::Text(Text {
                    text: "Test output 2".to_string(),
                }),
            ]),
            stored_output: Some(StoredOutput::Chat(vec![
                ContentBlockChatOutput::Text(Text {
                    text: "Test output 1".to_string(),
                }),
                ContentBlockChatOutput::Text(Text {
                    text: "Test output 2".to_string(),
                }),
            ])),
            dispreferred_outputs: vec![],
            episode_id: Some(episode_id),
            inference_id: Some(inference_id),
            tool_params: DynamicToolParams::default(),
            output_schema: None,
            tags: {
                let mut tags = HashMap::new();
                tags.insert("key1".to_string(), "value1".to_string());
                tags.insert("key2".to_string(), "value2".to_string());
                tags
            },
        }
    }

    /// Helper to create a test RenderedSample for JSON function
    fn create_test_json_rendered_sample() -> RenderedSample {
        let inference_id = Uuid::now_v7();
        let episode_id = Uuid::now_v7();

        RenderedSample {
            function_name: "json_function".to_string(),
            input: ModelInput {
                system: Some("JSON system prompt".to_string()),
                messages: vec![],
            },
            stored_input: StoredInput {
                system: Some(System::Text("JSON system prompt".to_string())),
                messages: vec![],
            },
            output: None, // JSON functions don't have chat output
            stored_output: Some(StoredOutput::Json(JsonInferenceOutput {
                raw: Some(r#"{"result": "test"}"#.to_string()),
                parsed: Some(serde_json::json!({"result": "test"})),
            })),
            dispreferred_outputs: vec![],
            episode_id: Some(episode_id),
            inference_id: Some(inference_id),
            tool_params: DynamicToolParams::default(),
            output_schema: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "result": {"type": "string"}
                }
            })),
            tags: {
                let mut tags = HashMap::new();
                tags.insert("json_key".to_string(), "json_value".to_string());
                tags
            },
        }
    }

    #[test]
    fn test_chat_inference_to_datapoint_with_inference_output() {
        let chat_inference = create_test_chat_inference();
        let dataset_name = "test_dataset";
        let output_source = CreateDatapointsFromInferenceOutputSource::Inference;
        let config = create_test_config();

        let original_inference_id = chat_inference.inference_id;
        let original_episode_id = chat_inference.episode_id;
        let original_function_name = chat_inference.function_name.clone();
        let original_input = chat_inference.input.clone();
        let original_output = chat_inference.output.clone();
        let original_tags = chat_inference.tags.clone();

        let inference = StoredInference::Chat(chat_inference);
        let datapoint = inference
            .into_datapoint_insert(dataset_name, &output_source, &config)
            .unwrap();

        match datapoint {
            DatapointInsert::Chat(dp) => {
                assert_eq!(dp.dataset_name, dataset_name);
                assert_eq!(dp.function_name, original_function_name);
                assert_eq!(dp.name, None);
                assert_ne!(dp.id, Uuid::nil());
                assert_eq!(dp.episode_id, Some(original_episode_id));
                assert_eq!(dp.input, original_input);
                assert_eq!(dp.output, Some(original_output));
                // tool_params are converted from DynamicToolParams to ToolCallConfigDatabaseInsert
                // Since we used default DynamicToolParams, we should get default ToolCallConfigDatabaseInsert
                assert!(dp.tool_params.is_some());
                assert_eq!(dp.tags, Some(original_tags));
                assert_eq!(dp.staled_at, None);
                assert_eq!(dp.source_inference_id, Some(original_inference_id));
                assert!(!dp.is_custom);
            }
            DatapointInsert::Json(_) => panic!("Expected Chat datapoint, got Json"),
        }
    }

    #[test]
    fn test_chat_inference_to_datapoint_with_none_output() {
        let chat_inference = create_test_chat_inference();
        let dataset_name = "test_dataset";
        let output_source = CreateDatapointsFromInferenceOutputSource::None;
        let config = create_test_config();

        let inference = StoredInference::Chat(chat_inference);
        let datapoint = inference
            .into_datapoint_insert(dataset_name, &output_source, &config)
            .unwrap();

        match datapoint {
            DatapointInsert::Chat(dp) => {
                // When output_source is None, output should be None
                assert_eq!(dp.output, None);

                // All other fields should still be preserved correctly
                assert_eq!(dp.dataset_name, dataset_name);
                assert!(!dp.is_custom);
            }
            DatapointInsert::Json(_) => panic!("Expected Chat datapoint, got Json"),
        }
    }

    #[test]
    fn test_chat_inference_to_datapoint_with_demonstration_output() {
        let chat_inference = create_test_chat_inference();
        let dataset_name = "test_dataset";
        let output_source = CreateDatapointsFromInferenceOutputSource::Demonstration;
        let config = create_test_config();

        let original_output = chat_inference.output.clone();
        let inference = StoredInference::Chat(chat_inference);
        let datapoint = inference
            .into_datapoint_insert(dataset_name, &output_source, &config)
            .unwrap();

        match datapoint {
            DatapointInsert::Chat(dp) => {
                // Demonstration output is joined during the query; we just make sure it's present.
                assert_eq!(dp.output, Some(original_output));
            }
            DatapointInsert::Json(_) => panic!("Expected Chat datapoint, got Json"),
        }
    }

    #[test]
    fn test_json_inference_to_datapoint_with_inference_output() {
        let json_inference = create_test_json_inference();
        let dataset_name = "json_dataset";
        let output_source = CreateDatapointsFromInferenceOutputSource::Inference;
        let config = create_test_config();

        let original_inference_id = json_inference.inference_id;
        let original_episode_id = json_inference.episode_id;
        let original_function_name = json_inference.function_name.clone();
        let original_input = json_inference.input.clone();
        let original_output = json_inference.output.clone();
        let original_output_schema = json_inference.output_schema.clone();
        let original_tags = json_inference.tags.clone();

        let inference = StoredInference::Json(json_inference);
        let datapoint = inference
            .into_datapoint_insert(dataset_name, &output_source, &config)
            .unwrap();

        match datapoint {
            DatapointInsert::Json(dp) => {
                assert_eq!(dp.dataset_name, dataset_name);
                assert_eq!(dp.function_name, original_function_name);
                assert_eq!(dp.name, None);
                assert_ne!(dp.id, Uuid::nil());
                assert_eq!(dp.episode_id, Some(original_episode_id));
                assert_eq!(dp.input, original_input);
                assert_eq!(dp.output, Some(original_output));
                assert_eq!(dp.output_schema, original_output_schema);
                assert_eq!(dp.tags, Some(original_tags));
                assert_eq!(dp.staled_at, None);
                assert_eq!(dp.source_inference_id, Some(original_inference_id));
                assert!(!dp.is_custom);
            }
            DatapointInsert::Chat(_) => panic!("Expected Json datapoint, got Chat"),
        }
    }

    #[test]
    fn test_json_inference_to_datapoint_with_none_output() {
        let json_inference = create_test_json_inference();
        let dataset_name = "json_dataset";
        let output_source = CreateDatapointsFromInferenceOutputSource::None;
        let config = create_test_config();

        let inference = StoredInference::Json(json_inference);
        let datapoint = inference
            .into_datapoint_insert(dataset_name, &output_source, &config)
            .unwrap();

        match datapoint {
            DatapointInsert::Json(dp) => {
                // When output_source is None, output should be None
                assert_eq!(dp.output, None);

                // All other fields should still be preserved correctly
                assert_eq!(dp.dataset_name, dataset_name);
                assert!(!dp.is_custom);
            }
            DatapointInsert::Chat(_) => panic!("Expected Json datapoint, got Chat"),
        }
    }

    #[test]
    fn test_json_inference_to_datapoint_with_demonstration_output() {
        let json_inference = create_test_json_inference();
        let dataset_name = "json_dataset";
        let output_source = CreateDatapointsFromInferenceOutputSource::Demonstration;
        let config = create_test_config();

        let original_output = json_inference.output.clone();
        let inference = StoredInference::Json(json_inference);
        let datapoint = inference
            .into_datapoint_insert(dataset_name, &output_source, &config)
            .unwrap();

        match datapoint {
            DatapointInsert::Json(dp) => {
                // Demonstration output is joined during the query; we just make sure it's present.
                assert_eq!(dp.output, Some(original_output));
            }
            DatapointInsert::Chat(_) => panic!("Expected Json datapoint, got Chat"),
        }
    }

    #[test]
    fn test_new_datapoint_id_is_generated_for_each_conversion() {
        let chat_inference = create_test_chat_inference();
        let dataset_name = "test_dataset";
        let output_source = CreateDatapointsFromInferenceOutputSource::Inference;
        let config = create_test_config();

        // Convert the same inference twice
        let inference1 = StoredInference::Chat(chat_inference.clone());
        let inference2 = StoredInference::Chat(chat_inference);
        let datapoint1 = inference1
            .into_datapoint_insert(dataset_name, &output_source, &config)
            .unwrap();
        let datapoint2 = inference2
            .into_datapoint_insert(dataset_name, &output_source, &config)
            .unwrap();

        // Extract IDs
        let id1 = match datapoint1 {
            DatapointInsert::Chat(dp) => dp.id,
            DatapointInsert::Json(_) => panic!("Expected Chat"),
        };

        let id2 = match datapoint2 {
            DatapointInsert::Chat(dp) => dp.id,
            DatapointInsert::Json(_) => panic!("Expected Chat"),
        };

        // IDs should be different (each conversion generates a new UUID)
        assert_ne!(
            id1, id2,
            "Datapoint IDs should be unique for each conversion"
        );
    }

    #[test]
    fn test_chat_inference_with_empty_tags() {
        let mut chat_inference = create_test_chat_inference();
        chat_inference.tags = HashMap::new();

        let dataset_name = "test_dataset";
        let output_source = CreateDatapointsFromInferenceOutputSource::Inference;
        let config = create_test_config();

        let inference = StoredInference::Chat(chat_inference);
        let datapoint = inference
            .into_datapoint_insert(dataset_name, &output_source, &config)
            .unwrap();

        match datapoint {
            DatapointInsert::Chat(dp) => {
                // Empty HashMap should be converted to Some(empty HashMap)
                assert_eq!(dp.tags, Some(HashMap::new()));
            }
            DatapointInsert::Json(_) => panic!("Expected Chat datapoint"),
        }
    }

    #[test]
    fn test_json_inference_with_empty_tags() {
        let mut json_inference = create_test_json_inference();
        json_inference.tags = HashMap::new();

        let dataset_name = "test_dataset";
        let output_source = CreateDatapointsFromInferenceOutputSource::Inference;
        let config = create_test_config();

        let inference = StoredInference::Json(json_inference);
        let datapoint = inference
            .into_datapoint_insert(dataset_name, &output_source, &config)
            .unwrap();

        match datapoint {
            DatapointInsert::Json(dp) => {
                // Empty HashMap should be converted to Some(empty HashMap)
                assert_eq!(dp.tags, Some(HashMap::new()));
            }
            DatapointInsert::Chat(_) => panic!("Expected Json datapoint"),
        }
    }

    #[test]
    fn test_stored_inference_id() {
        let chat_inference = create_test_chat_inference();
        let json_inference = create_test_json_inference();

        let chat_id = StoredInference::Chat(chat_inference.clone()).id();
        let json_id = StoredInference::Json(json_inference.clone()).id();

        assert_eq!(chat_id, chat_inference.inference_id);
        assert_eq!(json_id, json_inference.inference_id);
    }

    // Tests for RenderedSample::into_create_datapoint_request()

    #[test]
    fn test_chat_rendered_sample_to_create_datapoint_request_with_output() {
        let sample = create_test_chat_rendered_sample();

        let original_function_name = sample.function_name.clone();
        let original_episode_id = sample.episode_id;
        let original_output = sample.output.clone();
        let original_tool_params = sample.tool_params.clone();
        let original_tags = sample.tags.clone();

        let result = sample.into_create_datapoint_request().unwrap();

        match result {
            CreateDatapointRequest::Chat(req) => {
                assert_eq!(req.function_name, original_function_name);
                assert_eq!(req.episode_id, original_episode_id);
                assert_eq!(req.output, original_output);
                assert_eq!(req.dynamic_tool_params, original_tool_params);
                assert_eq!(req.tags, Some(original_tags));
                assert_eq!(req.name, None);

                // Verify input conversion worked (system should be preserved)
                match &req.input.system {
                    Some(crate::inference::types::System::Text(text)) => {
                        assert_eq!(text, "Test system prompt");
                    }
                    _ => panic!("Expected Text system"),
                }
                assert_eq!(req.input.messages.len(), 0);
            }
            CreateDatapointRequest::Json(_) => panic!("Expected Chat datapoint, got Json"),
        }
    }

    #[test]
    fn test_chat_rendered_sample_to_create_datapoint_request_without_output() {
        let mut sample = create_test_chat_rendered_sample();
        sample.stored_output = None;
        sample.output = None;

        let result = sample.into_create_datapoint_request().unwrap();

        match result {
            CreateDatapointRequest::Chat(req) => {
                // When stored_output is None, it should still create a Chat variant
                assert_eq!(req.output, None);
                assert_eq!(req.function_name, "test_function");
            }
            CreateDatapointRequest::Json(_) => panic!("Expected Chat datapoint, got Json"),
        }
    }

    #[test]
    fn test_json_rendered_sample_to_create_datapoint_request_with_output() {
        let sample = create_test_json_rendered_sample();

        let original_function_name = sample.function_name.clone();
        let original_episode_id = sample.episode_id;
        let original_output_schema = sample.output_schema.clone();
        let original_tags = sample.tags.clone();

        let result = sample.into_create_datapoint_request().unwrap();

        match result {
            CreateDatapointRequest::Json(req) => {
                assert_eq!(req.function_name, original_function_name);
                assert_eq!(req.episode_id, original_episode_id);
                assert_eq!(req.output_schema, original_output_schema);
                assert_eq!(req.tags, Some(original_tags));
                assert_eq!(req.name, None);

                // Verify output was extracted correctly
                assert!(req.output.is_some());
                let output = req.output.unwrap();
                assert_eq!(output.raw.unwrap(), r#"{"result": "test"}"#);

                // Verify input conversion worked
                match &req.input.system {
                    Some(crate::inference::types::System::Text(text)) => {
                        assert_eq!(text, "JSON system prompt");
                    }
                    _ => panic!("Expected Text system"),
                }
            }
            CreateDatapointRequest::Chat(_) => panic!("Expected Json datapoint, got Chat"),
        }
    }

    #[test]
    fn test_json_rendered_sample_to_create_datapoint_request_without_output() {
        let mut sample = create_test_json_rendered_sample();
        sample.stored_output = Some(StoredOutput::Json(JsonInferenceOutput {
            raw: None,
            parsed: None,
        }));

        let result = sample.into_create_datapoint_request().unwrap();

        match result {
            CreateDatapointRequest::Json(req) => {
                // When raw is None, output should be None
                assert!(req.output.is_none());
                assert_eq!(req.function_name, "json_function");
            }
            CreateDatapointRequest::Chat(_) => panic!("Expected Json datapoint, got Chat"),
        }
    }

    #[test]
    fn test_chat_rendered_sample_with_empty_tags() {
        let mut sample = create_test_chat_rendered_sample();
        sample.tags = HashMap::new();

        let result = sample.into_create_datapoint_request().unwrap();

        match result {
            CreateDatapointRequest::Chat(req) => {
                // Empty HashMap should be converted to Some(empty HashMap)
                assert_eq!(req.tags, Some(HashMap::new()));
            }
            CreateDatapointRequest::Json(_) => panic!("Expected Chat datapoint"),
        }
    }

    #[test]
    fn test_json_rendered_sample_with_empty_tags() {
        let mut sample = create_test_json_rendered_sample();
        sample.tags = HashMap::new();

        let result = sample.into_create_datapoint_request().unwrap();

        match result {
            CreateDatapointRequest::Json(req) => {
                // Empty HashMap should be converted to Some(empty HashMap)
                assert_eq!(req.tags, Some(HashMap::new()));
            }
            CreateDatapointRequest::Chat(_) => panic!("Expected Json datapoint"),
        }
    }

    #[test]
    fn test_stored_input_to_input_conversion() {
        let sample = create_test_chat_rendered_sample();

        // Verify the StoredInput → Input conversion works correctly
        let result = sample.into_create_datapoint_request().unwrap();

        match result {
            CreateDatapointRequest::Chat(req) => {
                // The input should have been successfully converted
                // System should be preserved
                assert!(req.input.system.is_some());
                // Messages should be preserved (empty in this case)
                assert_eq!(req.input.messages.len(), 0);
            }
            CreateDatapointRequest::Json(_) => panic!("Expected Chat datapoint"),
        }
    }
}
