#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
#[cfg(feature = "pyo3")]
use pyo3::{
    IntoPyObjectExt,
    types::{PyDict, PyModule},
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};
use tensorzero_derive::TensorZeroDeserialize;
use tensorzero_derive::export_schema;
use uuid::Uuid;

use crate::config::snapshot::SnapshotHash;
use crate::db::clickhouse::TableName;
use crate::db::stored_datapoint::{StoredChatInferenceDatapoint, StoredJsonInferenceDatapoint};
use crate::error::{Error, ErrorDetails};
use crate::function::FunctionConfig;
use crate::inference::types::{
    ContentBlockChatOutput, FetchContext, Input, InputExt, JsonInferenceOutput,
};
use crate::serde_util::{
    deserialize_optional_string_or_parsed_json, deserialize_string_or_parsed_json,
};
use crate::tool::{DynamicToolParams, StaticToolConfig};

#[cfg(feature = "pyo3")]
use crate::inference::types::pyo3_helpers::{
    content_block_chat_output_to_python, serialize_to_dict, uuid_to_python,
};

pub const CLICKHOUSE_DATETIME_FORMAT: &str = "%Y-%m-%d %H:%M:%S%.6f";

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "ts-bindings", ts(export))]
pub enum DatapointKind {
    Chat,
    Json,
}

impl DatapointKind {
    pub fn table_name(&self) -> TableName {
        match self {
            DatapointKind::Chat => TableName::ChatInferenceDatapoint,
            DatapointKind::Json => TableName::JsonInferenceDatapoint,
        }
    }
}

/// Wire variant of Datapoint enum for API responses with Python/TypeScript bindings
/// This one should be used in all public interfaces.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, JsonSchema, Serialize, TensorZeroDeserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "pyo3", pyclass(str, name = "LegacyDatapoint"))]
#[cfg_attr(feature = "ts-bindings", ts(export))]
#[export_schema]
pub enum Datapoint {
    #[schemars(title = "DatapointChat")]
    Chat(ChatInferenceDatapoint),
    #[schemars(title = "DatapointJson")]
    Json(JsonInferenceDatapoint),
}

impl std::fmt::Display for Datapoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

impl Datapoint {
    pub fn dataset_name(&self) -> &str {
        match self {
            Datapoint::Chat(datapoint) => &datapoint.dataset_name,
            Datapoint::Json(datapoint) => &datapoint.dataset_name,
        }
    }

    pub fn function_name(&self) -> &str {
        match self {
            Datapoint::Chat(datapoint) => &datapoint.function_name,
            Datapoint::Json(datapoint) => &datapoint.function_name,
        }
    }

    pub fn id(&self) -> Uuid {
        match self {
            Datapoint::Chat(datapoint) => datapoint.id,
            Datapoint::Json(datapoint) => datapoint.id,
        }
    }

    pub fn input(&self) -> &Input {
        match self {
            Datapoint::Chat(datapoint) => &datapoint.input,
            Datapoint::Json(datapoint) => &datapoint.input,
        }
    }

    pub fn tool_call_config(&self) -> Option<&DynamicToolParams> {
        match self {
            Datapoint::Chat(datapoint) => Some(&datapoint.tool_params),
            Datapoint::Json(_) => None,
        }
    }

    pub fn output_schema(&self) -> Option<&serde_json::Value> {
        match self {
            Datapoint::Chat(_datapoint) => None,
            Datapoint::Json(datapoint) => Some(&datapoint.output_schema),
        }
    }
}

impl ChatInferenceDatapoint {
    /// Convert to storage type, properly handling tool params with function config.
    /// If `fetch_context` is provided, any external URLs or Base64 files will be properly resolved and stored.
    /// If `fetch_context` is not provided, we will return an error if the input contains any external URLs or Base64 files,
    /// because we cannot represent them as the database type.
    pub async fn into_storage(
        self,
        function_config: &FunctionConfig,
        static_tools: &HashMap<String, Arc<StaticToolConfig>>,
        fetch_context: &FetchContext<'_>,
        snapshot_hash: SnapshotHash,
    ) -> Result<StoredChatInferenceDatapoint, Error> {
        let tool_params = function_config
            .dynamic_tool_params_to_database_insert(self.tool_params, static_tools)?;
        let stored_input = self
            .input
            .into_lazy_resolved_input(fetch_context)?
            .into_stored_input(fetch_context.object_store_info)
            .await?;

        Ok(StoredChatInferenceDatapoint {
            dataset_name: self.dataset_name,
            function_name: self.function_name,
            id: self.id,
            episode_id: self.episode_id,
            input: stored_input,
            output: self.output,
            tool_params,
            tags: self.tags,
            auxiliary: self.auxiliary,
            is_deleted: self.is_deleted,
            is_custom: self.is_custom,
            source_inference_id: self.source_inference_id,
            staled_at: self.staled_at,
            updated_at: self.updated_at,
            name: self.name,
            snapshot_hash: Some(snapshot_hash),
        })
    }

    /// Convert to storage type, without resolving network resources for files.
    /// This is used in PyO3 where we do not have a fetch context available.
    /// Returns an error if the input contains any external URLs or Base64 files.
    pub fn into_storage_without_file_handling(
        self,
        function_config: &FunctionConfig,
        static_tools: &HashMap<String, Arc<StaticToolConfig>>,
        snapshot_hash: &SnapshotHash,
    ) -> Result<StoredChatInferenceDatapoint, Error> {
        let tool_params = function_config
            .dynamic_tool_params_to_database_insert(self.tool_params, static_tools)?;

        let stored_input = self.input.into_stored_input_without_file_handling()?;

        Ok(StoredChatInferenceDatapoint {
            dataset_name: self.dataset_name,
            function_name: self.function_name,
            id: self.id,
            episode_id: self.episode_id,
            input: stored_input,
            output: self.output,
            tool_params,
            tags: self.tags,
            auxiliary: self.auxiliary,
            is_deleted: self.is_deleted,
            is_custom: self.is_custom,
            source_inference_id: self.source_inference_id,
            staled_at: self.staled_at,
            updated_at: self.updated_at,
            name: self.name,
            snapshot_hash: Some(snapshot_hash.clone()),
        })
    }
}

impl JsonInferenceDatapoint {
    /// Convert to storage type, possibly handling input file storage.
    /// If `fetch_context` is provided, any external URLs or Base64 files will be properly resolved and stored.
    /// If `fetch_context` is not provided, we will return an error if the input contains any external URLs or Base64 files,
    /// because we cannot represent them as the database type.
    pub async fn into_storage(
        self,
        fetch_context: Option<&FetchContext<'_>>,
        snapshot_hash: SnapshotHash,
    ) -> Result<StoredJsonInferenceDatapoint, Error> {
        let stored_input = match fetch_context {
            Some(fetch_context) => {
                self.input
                    .into_lazy_resolved_input(fetch_context)?
                    .into_stored_input(fetch_context.object_store_info)
                    .await?
            }
            None => self.input.into_stored_input_without_file_handling()?,
        };

        Ok(StoredJsonInferenceDatapoint {
            dataset_name: self.dataset_name,
            function_name: self.function_name,
            id: self.id,
            episode_id: self.episode_id,
            input: stored_input,
            output: self.output,
            output_schema: self.output_schema,
            tags: self.tags,
            auxiliary: self.auxiliary,
            is_deleted: self.is_deleted,
            is_custom: self.is_custom,
            source_inference_id: self.source_inference_id,
            staled_at: self.staled_at,
            updated_at: self.updated_at,
            name: self.name,
            snapshot_hash: Some(snapshot_hash),
        })
    }

    /// Convert to storage type, without resolving network resources for files.
    /// This is used in PyO3 where we do not have a fetch context available.
    /// Returns an error if the input contains any external URLs or Base64 files.
    pub fn into_storage_without_file_handling(
        self,
        snapshot_hash: SnapshotHash,
    ) -> Result<StoredJsonInferenceDatapoint, Error> {
        let stored_input = self.input.into_stored_input_without_file_handling()?;

        Ok(StoredJsonInferenceDatapoint {
            dataset_name: self.dataset_name,
            function_name: self.function_name,
            id: self.id,
            episode_id: self.episode_id,
            input: stored_input,
            output: self.output,
            output_schema: self.output_schema,
            tags: self.tags,
            auxiliary: self.auxiliary,
            is_deleted: self.is_deleted,
            is_custom: self.is_custom,
            source_inference_id: self.source_inference_id,
            staled_at: self.staled_at,
            updated_at: self.updated_at,
            name: self.name,
            snapshot_hash: Some(snapshot_hash),
        })
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl Datapoint {
    pub fn __repr__(&self) -> String {
        self.to_string()
    }

    #[getter]
    pub fn get_id<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        uuid_to_python(py, self.id())
    }

    #[getter]
    pub fn get_input<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        // This is python_helpers.rs convert_response_to_python_dataclass, but we can't import it across crates.
        // We will remove the whole Datapoint type and replace with generated types soon.

        // Serialize Rust response to JSON dict

        let dict = serialize_to_dict(py, self.input().clone())?;

        // Import the target dataclass
        let module = PyModule::import(py, "tensorzero")?;
        let data_class = module.getattr("Input")?;

        // Use dacite.from_dict to construct the dataclass, so that it can handle nested dataclass construction.
        let dacite = PyModule::import(py, "dacite")?;
        let from_dict = dacite.getattr("from_dict")?;

        // Call dacite.from_dict(data_class=TargetClass, data=dict)
        let kwargs = PyDict::new(py);
        kwargs.set_item("data_class", data_class)?;
        kwargs.set_item("data", dict)?;

        from_dict.call((), Some(&kwargs))
    }

    #[getter]
    pub fn get_output<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(match self {
            Datapoint::Chat(datapoint) => match &datapoint.output {
                Some(output) => output
                    .iter()
                    .map(|x| content_block_chat_output_to_python(py, x.clone()))
                    .collect::<PyResult<Vec<_>>>()?
                    .into_bound_py_any(py)?,
                None => py.None().into_bound(py),
            },
            Datapoint::Json(datapoint) => datapoint.output.clone().into_bound_py_any(py)?,
        })
    }

    #[getter]
    pub fn get_dataset_name(&self) -> String {
        self.dataset_name().to_string()
    }

    #[getter]
    pub fn get_function_name(&self) -> String {
        self.function_name().to_string()
    }

    #[getter]
    pub fn get_allowed_tools(&self) -> Option<Vec<String>> {
        match self {
            Datapoint::Chat(datapoint) => datapoint.tool_params.allowed_tools.clone(),
            Datapoint::Json(_) => None,
        }
    }

    #[getter]
    pub fn get_additional_tools<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self {
            Datapoint::Chat(datapoint) => datapoint
                .tool_params
                .additional_tools
                .clone()
                .into_bound_py_any(py),
            Datapoint::Json(_) => Ok(py.None().into_bound(py)),
        }
    }

    // Note: We're intentionally skipping tool_choice as it's not exposed in the Python API

    #[getter]
    pub fn get_parallel_tool_calls(&self) -> Option<bool> {
        match self {
            Datapoint::Chat(datapoint) => datapoint.tool_params.parallel_tool_calls,
            Datapoint::Json(_) => None,
        }
    }

    #[getter]
    pub fn get_provider_tools<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self {
            Datapoint::Chat(datapoint) => datapoint
                .tool_params
                .provider_tools
                .clone()
                .into_bound_py_any(py),
            Datapoint::Json(_) => Ok(py.None().into_bound(py)),
        }
    }

    #[getter]
    pub fn get_output_schema<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(match self {
            Datapoint::Chat(_) => py.None().into_bound(py),
            Datapoint::Json(datapoint) => {
                serialize_to_dict(py, &datapoint.output_schema)?.into_bound(py)
            }
        })
    }

    #[getter]
    pub fn get_is_custom(&self) -> bool {
        match self {
            Datapoint::Chat(datapoint) => datapoint.is_custom,
            Datapoint::Json(datapoint) => datapoint.is_custom,
        }
    }

    #[getter]
    pub fn get_name(&self) -> Option<String> {
        match self {
            Datapoint::Chat(datapoint) => datapoint.name.clone(),
            Datapoint::Json(datapoint) => datapoint.name.clone(),
        }
    }
}

/// Wire variant of ChatInferenceDatapoint for API responses with Python/TypeScript bindings
/// This one should be used in all public interfaces.
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, JsonSchema)]
#[cfg_attr(feature = "pyo3", pyclass(str))]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
#[export_schema]
pub struct ChatInferenceDatapoint {
    pub dataset_name: String,
    pub function_name: String,
    pub id: Uuid,
    pub episode_id: Option<Uuid>,
    #[serde(deserialize_with = "deserialize_string_or_parsed_json")]
    pub input: Input,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    #[serde(deserialize_with = "deserialize_optional_string_or_parsed_json")]
    pub output: Option<Vec<ContentBlockChatOutput>>,
    // `tool_params` are always flattened to match the convention of LLM APIs
    #[serde(flatten)]
    #[serde(default)]
    pub tool_params: DynamicToolParams,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    #[cfg_attr(feature = "ts-bindings", ts(type = "Record<string, string>"))]
    pub tags: Option<HashMap<String, String>>,
    #[serde(
        skip_serializing,
        default,
        deserialize_with = "crate::serde_util::deserialize_null_default"
    )]
    pub auxiliary: String,
    pub is_deleted: bool,
    #[serde(default)]
    pub is_custom: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub source_inference_id: Option<Uuid>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub staled_at: Option<String>,
    pub updated_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub name: Option<String>,
}

impl std::fmt::Display for ChatInferenceDatapoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS))]
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, JsonSchema)]
#[cfg_attr(feature = "pyo3", pyclass(str))]
#[export_schema]
#[cfg_attr(feature = "ts-bindings", ts(export, optional_fields))]
pub struct JsonInferenceDatapoint {
    pub dataset_name: String,
    pub function_name: String,
    pub id: Uuid,
    pub episode_id: Option<Uuid>,
    #[serde(deserialize_with = "deserialize_string_or_parsed_json")]
    pub input: Input,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    #[serde(deserialize_with = "deserialize_optional_string_or_parsed_json")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub output: Option<JsonInferenceOutput>,
    #[serde(deserialize_with = "deserialize_string_or_parsed_json")]
    pub output_schema: serde_json::Value,

    // By default, ts_rs generates { [key in string]?: string } | undefined, which means values are string | undefined which isn't what we want.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    #[cfg_attr(feature = "ts-bindings", ts(type = "Record<string, string>"))]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub tags: Option<HashMap<String, String>>,
    #[serde(
        skip_serializing,
        default,
        deserialize_with = "crate::serde_util::deserialize_null_default"
    )] // this will become an object
    pub auxiliary: String,
    pub is_deleted: bool,
    #[serde(default)]
    pub is_custom: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub source_inference_id: Option<Uuid>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub staled_at: Option<String>,
    pub updated_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub name: Option<String>,
}

impl std::fmt::Display for JsonInferenceDatapoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

impl StoredChatInferenceDatapoint {
    /// Convert to wire type, converting tool params from storage format to wire format using From<> trait
    pub fn into_datapoint(self) -> ChatInferenceDatapoint {
        let tool_params = self.tool_params.map(|tp| tp.into()).unwrap_or_default();

        ChatInferenceDatapoint {
            dataset_name: self.dataset_name,
            function_name: self.function_name,
            id: self.id,
            episode_id: self.episode_id,
            input: self.input.into_input(),
            output: self.output,
            tool_params,
            tags: self.tags,
            auxiliary: self.auxiliary,
            is_deleted: self.is_deleted,
            is_custom: self.is_custom,
            source_inference_id: self.source_inference_id,
            staled_at: self.staled_at,
            updated_at: self.updated_at,
            name: self.name,
        }
    }
}

impl StoredJsonInferenceDatapoint {
    pub fn into_datapoint(self) -> JsonInferenceDatapoint {
        JsonInferenceDatapoint {
            dataset_name: self.dataset_name,
            function_name: self.function_name,
            id: self.id,
            episode_id: self.episode_id,
            input: self.input.into_input(),
            output: self.output,
            output_schema: self.output_schema,
            tags: self.tags,
            auxiliary: self.auxiliary,
            is_deleted: self.is_deleted,
            is_custom: self.is_custom,
            source_inference_id: self.source_inference_id,
            staled_at: self.staled_at,
            updated_at: self.updated_at,
            name: self.name,
        }
    }
}

pub(crate) fn validate_dataset_name(dataset_name: &str) -> Result<(), Error> {
    if dataset_name == "builder" || dataset_name.starts_with("tensorzero::") {
        Err(Error::new(ErrorDetails::InvalidDatasetName {
            dataset_name: dataset_name.to_string(),
        }))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_validate_dataset_name_builder() {
        let err = validate_dataset_name("builder").unwrap_err();
        assert_eq!(
            err.to_string(),
            "Invalid dataset name: builder. Datasets cannot be named \"builder\" or begin with \"tensorzero::\""
        );
    }

    #[test]
    fn test_validate_dataset_name_tensorzero_prefix() {
        let err = validate_dataset_name("tensorzero::test").unwrap_err();
        assert_eq!(
            err.to_string(),
            "Invalid dataset name: tensorzero::test. Datasets cannot be named \"builder\" or begin with \"tensorzero::\""
        );
    }

    #[test]
    fn test_validate_dataset_name_valid() {
        validate_dataset_name("test").unwrap();
    }
}
