use std::{collections::HashMap, sync::Arc};

use crate::db::datasets::{
    ChatInferenceDatapointInsert, DatapointInsert, JsonInferenceDatapointInsert,
};
use crate::endpoints::datasets::v1::types::CreateDatapointsFromInferenceOutputSource;
use crate::function::FunctionConfig;
#[cfg(feature = "pyo3")]
use crate::inference::types::pyo3_helpers::{
    content_block_chat_output_to_python, serialize_to_dict, uuid_to_python,
};
use crate::inference::types::stored_input::StoredInput;
use crate::inference::types::{RequestMessage, ResolvedRequestMessage, Text};
use crate::tool::{DynamicToolParams, StaticToolConfig};
#[cfg(feature = "pyo3")]
use crate::tool::{ProviderTool, Tool, ToolChoice};
use crate::{
    config::Config,
    error::{Error, ErrorDetails},
    inference::types::{ContentBlockChatOutput, JsonInferenceOutput, ModelInput, ResolvedInput},
    tool::ToolCallConfigDatabaseInsert,
    variant::{chat_completion::prepare_model_input, VariantConfig},
};
use chrono::{DateTime, Utc};
#[cfg(feature = "pyo3")]
use pyo3::types::PyList;
#[cfg(feature = "pyo3")]
use pyo3::{exceptions::PyValueError, prelude::*, IntoPyObjectExt};
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
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, ts_rs::TS)]
#[serde(tag = "type", rename_all = "snake_case")]
#[cfg_attr(feature = "pyo3", pyclass(str, name = "StoredInference"))]
#[ts(export)]
pub enum StoredInference {
    Chat(StoredChatInference),
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

#[cfg(feature = "pyo3")]
#[pymethods]
impl StoredInference {
    #[new]
    #[pyo3(signature = (r#type, function_name, variant_name, input, output, episode_id, inference_id, timestamp, output_schema=None, dispreferred_outputs=None, tags=None, allowed_tools=None, additional_tools=None, tool_choice=None, parallel_tool_calls=None, provider_tools=None))]
    #[expect(clippy::too_many_arguments)]
    pub fn new(
        py: Python<'_>,
        r#type: String,
        function_name: String,
        variant_name: String,
        input: Bound<'_, PyAny>,
        output: Bound<'_, PyAny>,
        episode_id: Bound<'_, PyAny>,
        inference_id: Bound<'_, PyAny>,
        timestamp: Bound<'_, PyAny>,
        output_schema: Option<Bound<'_, PyAny>>,
        dispreferred_outputs: Option<Bound<'_, PyAny>>,
        tags: Option<Bound<'_, PyAny>>,
        // Flattened DynamicToolParams fields
        allowed_tools: Option<Vec<String>>,
        additional_tools: Option<Bound<'_, PyAny>>,
        tool_choice: Option<Bound<'_, PyAny>>,
        parallel_tool_calls: Option<bool>,
        provider_tools: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        use crate::inference::types::pyo3_helpers::deserialize_from_pyobj;

        // Deserialize common fields
        let input: StoredInput = deserialize_from_pyobj(py, &input)?;
        let episode_id: Uuid = deserialize_from_pyobj(py, &episode_id)?;
        let inference_id: Uuid = deserialize_from_pyobj(py, &inference_id)?;
        let timestamp: DateTime<Utc> = deserialize_from_pyobj(py, &timestamp)?;
        let tags: HashMap<String, String> = tags
            .as_ref()
            .map(|x| deserialize_from_pyobj(py, x))
            .transpose()?
            .unwrap_or_default();

        match r#type.as_str() {
            "chat" => {
                let output: Vec<ContentBlockChatOutput> = deserialize_from_pyobj(py, &output)?;
                let dispreferred_outputs: Option<Vec<Vec<ContentBlockChatOutput>>> =
                    dispreferred_outputs
                        .as_ref()
                        .map(|x| deserialize_from_pyobj(py, x))
                        .transpose()?;

                // Build DynamicToolParams from flattened fields
                let additional_tools: Option<Vec<Tool>> = additional_tools
                    .as_ref()
                    .map(|x| deserialize_from_pyobj(py, x))
                    .transpose()?;
                let tool_choice: Option<ToolChoice> = tool_choice
                    .as_ref()
                    .map(|x| deserialize_from_pyobj(py, x))
                    .transpose()?;
                let provider_tools: Option<Vec<ProviderTool>> = provider_tools
                    .as_ref()
                    .map(|x| deserialize_from_pyobj(py, x))
                    .transpose()?;

                let tool_params = DynamicToolParams {
                    allowed_tools,
                    additional_tools,
                    tool_choice,
                    parallel_tool_calls,
                    provider_tools,
                };

                Ok(Self::Chat(StoredChatInference {
                    function_name,
                    variant_name,
                    input,
                    output,
                    dispreferred_outputs: dispreferred_outputs.unwrap_or_default(),
                    episode_id,
                    inference_id,
                    tool_params,
                    tags,
                    timestamp,
                }))
            }
            "json" => {
                let output: JsonInferenceOutput = deserialize_from_pyobj(py, &output)?;
                let dispreferred_outputs: Option<Vec<JsonInferenceOutput>> = dispreferred_outputs
                    .as_ref()
                    .map(|x| deserialize_from_pyobj(py, x))
                    .transpose()?;
                let Some(output_schema) = output_schema
                    .as_ref()
                    .map(|x| deserialize_from_pyobj(py, x))
                else {
                    return Err(PyValueError::new_err(
                        "output_schema is required for json inferences",
                    ));
                };
                let output_schema: Value = output_schema?;
                Ok(Self::Json(StoredJsonInference {
                    function_name,
                    variant_name,
                    input,
                    output,
                    dispreferred_outputs: dispreferred_outputs.unwrap_or_default(),
                    episode_id,
                    inference_id,
                    output_schema,
                    tags,
                    timestamp,
                }))
            }
            _ => Err(PyValueError::new_err(format!(
                "Invalid inference type: {type}. Must be 'chat' or 'json'",
            ))),
        }
    }

    pub fn __repr__(&self) -> String {
        self.to_string()
    }

    #[getter]
    pub fn get_function_name(&self) -> String {
        match self {
            StoredInference::Chat(example) => example.function_name.clone(),
            StoredInference::Json(example) => example.function_name.clone(),
        }
    }

    #[getter]
    pub fn get_variant_name(&self) -> String {
        match self {
            StoredInference::Chat(example) => example.variant_name.clone(),
            StoredInference::Json(example) => example.variant_name.clone(),
        }
    }

    #[getter]
    pub fn get_input(&self) -> StoredInput {
        match self {
            StoredInference::Chat(example) => example.input.clone(),
            StoredInference::Json(example) => example.input.clone(),
        }
    }
    /// Returns the output of the inference as PyO3 classes.
    /// This is actually a List of ContentBlockChatOutputs for StoredChatInference
    /// and a JsonInferenceOutput for StoredJsonInference.
    #[getter]
    pub fn get_output<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(match self {
            StoredInference::Chat(example) => example
                .output
                .iter()
                .map(|x| content_block_chat_output_to_python(py, x.clone()))
                .collect::<PyResult<Vec<_>>>()?
                .into_bound_py_any(py)?,
            StoredInference::Json(example) => example.output.clone().into_bound_py_any(py)?,
        })
    }

    #[getter]
    pub fn get_dispreferred_outputs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(match self {
            StoredInference::Chat(example) => example
                .dispreferred_outputs
                .iter()
                .map(|x| {
                    x.iter()
                        .map(|y| content_block_chat_output_to_python(py, y.clone()))
                        .collect::<PyResult<Vec<_>>>()
                })
                .collect::<PyResult<Vec<Vec<_>>>>()?
                .into_bound_py_any(py)?,
            StoredInference::Json(example) => {
                example.dispreferred_outputs.clone().into_bound_py_any(py)?
            }
        })
    }

    #[getter]
    pub fn get_episode_id<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self {
            StoredInference::Chat(example) => uuid_to_python(py, example.episode_id),
            StoredInference::Json(example) => uuid_to_python(py, example.episode_id),
        }
    }

    #[getter]
    pub fn get_inference_id<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self {
            StoredInference::Chat(example) => uuid_to_python(py, example.inference_id),
            StoredInference::Json(example) => uuid_to_python(py, example.inference_id),
        }
    }

    #[getter]
    pub fn get_output_schema<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(match self {
            StoredInference::Chat(_) => py.None().into_bound(py),
            StoredInference::Json(example) => {
                serialize_to_dict(py, example.output_schema.clone())?.into_bound(py)
            }
        })
    }

    #[getter]
    pub fn get_type(&self) -> String {
        match self {
            StoredInference::Chat(_) => "chat".to_string(),
            StoredInference::Json(_) => "json".to_string(),
        }
    }

    #[getter]
    pub fn get_tags(&self) -> HashMap<String, String> {
        match self {
            StoredInference::Chat(example) => example.tags.clone(),
            StoredInference::Json(example) => example.tags.clone(),
        }
    }

    #[getter]
    pub fn get_timestamp(&self) -> String {
        match self {
            StoredInference::Chat(example) => example.timestamp.to_rfc3339(),
            StoredInference::Json(example) => example.timestamp.to_rfc3339(),
        }
    }

    #[getter]
    pub fn get_allowed_tools(&self) -> Option<Vec<String>> {
        match self {
            StoredInference::Chat(example) => example.tool_params.allowed_tools.clone(),
            StoredInference::Json(_) => None,
        }
    }

    #[getter]
    pub fn get_additional_tools<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self {
            StoredInference::Chat(example) => example
                .tool_params
                .additional_tools
                .clone()
                .into_bound_py_any(py),
            StoredInference::Json(_) => Ok(py.None().into_bound(py)),
        }
    }

    // Note: We're intentionally skipping tool_choice as it's not exposed in the Python API

    #[getter]
    pub fn get_parallel_tool_calls(&self) -> Option<bool> {
        match self {
            StoredInference::Chat(example) => example.tool_params.parallel_tool_calls,
            StoredInference::Json(_) => None,
        }
    }

    #[getter]
    pub fn get_provider_tools<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self {
            StoredInference::Chat(example) => example
                .tool_params
                .provider_tools
                .clone()
                .into_bound_py_any(py),
            StoredInference::Json(_) => Ok(py.None().into_bound(py)),
        }
    }
}

impl StoredInferenceDatabase {
    /// Convert to wire type, properly handling tool params by subtracting static tools
    pub fn into_stored_inference(self, config: &Config) -> Result<StoredInference, Error> {
        match self {
            StoredInferenceDatabase::Chat(chat) => {
                let function_config = config.get_function(&chat.function_name)?;
                Ok(StoredInference::Chat(
                    chat.into_stored_inference(&function_config),
                ))
            }
            StoredInferenceDatabase::Json(json) => Ok(StoredInference::Json(json)),
        }
    }
}

impl StoredInference {
    /// Convert to storage type, properly handling tool params with function config
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
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, ts_rs::TS)]
#[cfg_attr(feature = "pyo3", pyclass(str))]
#[ts(export)]
pub struct StoredChatInference {
    pub function_name: String,
    pub variant_name: String,
    pub input: StoredInput,
    pub output: Vec<ContentBlockChatOutput>,
    #[serde(default)]
    pub dispreferred_outputs: Vec<Vec<ContentBlockChatOutput>>,
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

#[cfg(feature = "pyo3")]
#[pymethods]
impl StoredChatInference {
    pub fn __repr__(&self) -> String {
        self.to_string()
    }
}

impl StoredChatInferenceDatabase {
    /// Convert to wire type, properly handling tool params by subtracting static tools
    pub fn into_stored_inference(self, function_config: &FunctionConfig) -> StoredChatInference {
        let tool_params = function_config.database_insert_to_dynamic_tool_params(self.tool_params);

        StoredChatInference {
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
    #[serde(default)]
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

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, ts_rs::TS)]
#[cfg_attr(feature = "pyo3", pyclass(str))]
#[ts(export)]
pub struct StoredJsonInference {
    pub function_name: String,
    pub variant_name: String,
    pub input: StoredInput,
    pub output: JsonInferenceOutput,
    #[serde(default)]
    pub dispreferred_outputs: Vec<JsonInferenceOutput>,
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

#[cfg(feature = "pyo3")]
#[pymethods]
impl StoredJsonInference {
    pub fn __repr__(&self) -> String {
        self.to_string()
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

    // Convert tool_params from storage format to wire format, subtracting static tools
    let function_config = config.get_function(&function_name)?;
    let dynamic_tool_params = tool_params
        .map(|tp| function_config.database_insert_to_dynamic_tool_params(tp))
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
                implicit_tool_call_config: ToolCallConfig::default(),
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
}
