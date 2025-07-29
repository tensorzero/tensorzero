use std::borrow::Cow;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::types::{IntoPyDict, PyDict};
use pyo3::{intern, prelude::*};
use pyo3::{sync::GILOnceCell, types::PyModule, Bound, Py, PyAny, PyErr, PyResult, Python};
use serde::Deserialize;
use serde_json::Value;
use uuid::Uuid;

use crate::endpoints::datasets::Datapoint;
use crate::inference::types::{ContentBlockChatOutput, ResolvedInput, ResolvedInputMessageContent};
use crate::optimization::fireworks_sft::UninitializedFireworksSFTConfig;
use crate::optimization::openai_sft::UninitializedOpenAISFTConfig;
use crate::optimization::UninitializedOptimizerConfig;
use crate::stored_inference::{
    RenderedSample, SimpleStoredSampleInfo, StoredInference, StoredSample,
};

use super::ContentBlock;

pub static JSON_LOADS: GILOnceCell<Py<PyAny>> = GILOnceCell::new();
pub static JSON_DUMPS: GILOnceCell<Py<PyAny>> = GILOnceCell::new();
pub static UUID_UUID: GILOnceCell<Py<PyAny>> = GILOnceCell::new();
static TENSORZERO_INTERNAL_ERROR: GILOnceCell<Py<PyAny>> = GILOnceCell::new();
static TENSORZERO_ERROR: GILOnceCell<Py<PyAny>> = GILOnceCell::new();

pub fn uuid_to_python(py: Python<'_>, uuid: Uuid) -> PyResult<Bound<'_, PyAny>> {
    let uuid_class = UUID_UUID.get_or_try_init::<_, PyErr>(py, || {
        let self_module = PyModule::import(py, "uuid")?;
        Ok(self_module.getattr("UUID")?.unbind())
    })?;
    let kwargs = [(intern!(py, "bytes"), uuid.as_bytes())].into_py_dict(py)?;
    let uuid_obj = uuid_class.call(py, (), Some(&kwargs))?;
    Ok(uuid_obj.into_bound(py))
}

fn import_text_content_block(py: Python<'_>) -> PyResult<&Py<PyAny>> {
    // NOTE: we are reusing the type as is used in our output.
    // We may want to consider not doing this so that we don't have these tied together in our interface.
    // However, they are currently nearly identical so this would be duplicated code for now and
    // not intutitive for users
    static TEXT_CONTENT_BLOCK: GILOnceCell<Py<PyAny>> = GILOnceCell::new();
    TEXT_CONTENT_BLOCK.get_or_try_init::<_, PyErr>(py, || {
        let self_module = PyModule::import(py, "tensorzero.types")?;
        Ok(self_module.getattr("Text")?.unbind())
    })
}

fn import_raw_text_content_block(py: Python<'_>) -> PyResult<&Py<PyAny>> {
    static RAW_TEXT_CONTENT_BLOCK: GILOnceCell<Py<PyAny>> = GILOnceCell::new();
    RAW_TEXT_CONTENT_BLOCK.get_or_try_init::<_, PyErr>(py, || {
        let self_module = PyModule::import(py, "tensorzero.types")?;
        Ok(self_module.getattr("RawText")?.unbind())
    })
}

fn import_file_content_block(py: Python<'_>) -> PyResult<&Py<PyAny>> {
    static FILE_CONTENT_BLOCK: GILOnceCell<Py<PyAny>> = GILOnceCell::new();
    FILE_CONTENT_BLOCK.get_or_try_init::<_, PyErr>(py, || {
        let self_module = PyModule::import(py, "tensorzero.types")?;
        Ok(self_module.getattr("FileBase64")?.unbind())
    })
}

fn import_tool_call_content_block(py: Python<'_>) -> PyResult<&Py<PyAny>> {
    static TOOL_CALL_CONTENT_BLOCK: GILOnceCell<Py<PyAny>> = GILOnceCell::new();
    TOOL_CALL_CONTENT_BLOCK.get_or_try_init::<_, PyErr>(py, || {
        let self_module = PyModule::import(py, "tensorzero.types")?;
        Ok(self_module.getattr("ToolCall")?.unbind())
    })
}

fn import_thought_content_block(py: Python<'_>) -> PyResult<&Py<PyAny>> {
    static THOUGHT_CONTENT_BLOCK: GILOnceCell<Py<PyAny>> = GILOnceCell::new();
    THOUGHT_CONTENT_BLOCK.get_or_try_init::<_, PyErr>(py, || {
        let self_module = PyModule::import(py, "tensorzero.types")?;
        Ok(self_module.getattr("Thought")?.unbind())
    })
}

fn import_tool_result_content_block(py: Python<'_>) -> PyResult<&Py<PyAny>> {
    static TOOL_RESULT_CONTENT_BLOCK: GILOnceCell<Py<PyAny>> = GILOnceCell::new();
    TOOL_RESULT_CONTENT_BLOCK.get_or_try_init::<_, PyErr>(py, || {
        let self_module = PyModule::import(py, "tensorzero.types")?;
        Ok(self_module.getattr("ToolResult")?.unbind())
    })
}

fn import_unknown_content_block(py: Python<'_>) -> PyResult<&Py<PyAny>> {
    static UNKNOWN_CONTENT_BLOCK: GILOnceCell<Py<PyAny>> = GILOnceCell::new();
    UNKNOWN_CONTENT_BLOCK.get_or_try_init::<_, PyErr>(py, || {
        let self_module = PyModule::import(py, "tensorzero.types")?;
        Ok(self_module.getattr("UnknownContentBlock")?.unbind())
    })
}

pub fn content_block_to_python(
    py: Python<'_>,
    content_block: &ContentBlock,
) -> PyResult<Py<PyAny>> {
    match content_block {
        ContentBlock::Text(text) => {
            let text_content_block = import_text_content_block(py)?;
            text_content_block.call1(py, (text.text.clone(),))
        }
        ContentBlock::File(file) => {
            let file_content_block = import_file_content_block(py)?;
            file_content_block.call1(
                py,
                (
                    file.file.data.clone().unwrap_or(String::new()),
                    file.file.mime_type.to_string(),
                ),
            )
        }
        ContentBlock::ToolCall(tool_call) => {
            let tool_call_content_block = import_tool_call_content_block(py)?;
            tool_call_content_block.call1(
                py,
                (
                    tool_call.id.clone(),
                    tool_call.arguments.clone(),
                    tool_call.name.clone(),
                    tool_call.arguments.clone(),
                    tool_call.name.clone(),
                ),
            )
        }
        ContentBlock::Thought(thought) => {
            let thought_content_block = import_thought_content_block(py)?;
            thought_content_block.call1(py, (thought.text.clone(),))
        }
        ContentBlock::ToolResult(tool_result) => {
            let tool_result_content_block = import_tool_result_content_block(py)?;
            tool_result_content_block.call1(
                py,
                (
                    tool_result.name.clone(),
                    tool_result.result.clone(),
                    tool_result.id.clone(),
                ),
            )
        }
        ContentBlock::Unknown {
            data,
            model_provider_name,
        } => {
            let unknown_content_block = import_unknown_content_block(py)?;
            let serialized_data = serialize_to_dict(py, data)?;
            unknown_content_block.call1(py, (serialized_data, model_provider_name))
        }
    }
}

pub fn content_block_chat_output_to_python(
    py: Python<'_>,
    content_block_chat_output: ContentBlockChatOutput,
) -> PyResult<Py<PyAny>> {
    match content_block_chat_output {
        ContentBlockChatOutput::Text(text) => {
            let text_content_block = import_text_content_block(py)?;
            text_content_block.call1(py, (text.text,))
        }
        ContentBlockChatOutput::ToolCall(tool_call) => {
            let tool_call_content_block = import_tool_call_content_block(py)?;
            tool_call_content_block.call1(
                py,
                (
                    tool_call.id,
                    tool_call.raw_arguments,
                    tool_call.raw_name,
                    serialize_to_dict(py, tool_call.arguments)?,
                    tool_call.name,
                ),
            )
        }
        ContentBlockChatOutput::Thought(thought) => {
            let thought_content_block = import_thought_content_block(py)?;
            thought_content_block.call1(py, (thought.text,))
        }
        ContentBlockChatOutput::Unknown {
            data,
            model_provider_name,
        } => {
            let unknown_content_block = import_unknown_content_block(py)?;
            let serialized_data = serialize_to_dict(py, data)?;
            unknown_content_block.call1(py, (serialized_data, model_provider_name))
        }
    }
}

pub fn resolved_input_message_content_to_python(
    py: Python<'_>,
    content: ResolvedInputMessageContent,
) -> PyResult<Py<PyAny>> {
    match content {
        ResolvedInputMessageContent::Text { value } => {
            let text_content_block = import_text_content_block(py)?;
            match value {
                Value::String(s) => {
                    let kwargs = [(intern!(py, "text"), s)].into_py_dict(py)?;
                    text_content_block.call(py, (), Some(&kwargs))
                }
                _ => {
                    let value = serialize_to_dict(py, value)?;
                    let kwargs = [(intern!(py, "arguments"), value)].into_py_dict(py)?;
                    text_content_block.call(py, (), Some(&kwargs))
                }
            }
        }
        ResolvedInputMessageContent::ToolCall(tool_call) => {
            let tool_call_content_block = import_tool_call_content_block(py)?;
            let parsed_arguments_py = JSON_LOADS
                .get(py)
                .ok_or_else(|| {
                    PyRuntimeError::new_err(
                        "TensorZero: JSON_LOADS was not initialized. This should never happen",
                    )
                })?
                .call1(py, (tool_call.arguments.clone().into_pyobject(py)?,))
                .ok();
            tool_call_content_block.call1(
                py,
                (
                    tool_call.id,
                    tool_call.arguments,
                    tool_call.name.clone(),
                    parsed_arguments_py,
                    tool_call.name,
                ),
            )
        }
        ResolvedInputMessageContent::ToolResult(tool_result) => {
            let tool_result_content_block = import_tool_result_content_block(py)?;
            tool_result_content_block
                .call1(py, (tool_result.name, tool_result.result, tool_result.id))
        }
        ResolvedInputMessageContent::Thought(thought) => {
            let thought_content_block = import_thought_content_block(py)?;
            thought_content_block.call1(py, (thought.text,))
        }
        ResolvedInputMessageContent::RawText { value } => {
            let raw_text_content_block = import_raw_text_content_block(py)?;
            raw_text_content_block.call1(py, (value,))
        }
        ResolvedInputMessageContent::File(file) => {
            let file_content_block = import_file_content_block(py)?;
            file_content_block.call1(
                py,
                (
                    file.file.data.clone().unwrap_or(String::new()),
                    file.file.mime_type.to_string(),
                ),
            )
        }
        ResolvedInputMessageContent::Unknown {
            data,
            model_provider_name,
        } => {
            let unknown_content_block = import_unknown_content_block(py)?;
            let serialized_data = serialize_to_dict(py, data)?;
            unknown_content_block.call1(py, (serialized_data, model_provider_name))
        }
    }
}

/// Serializes a Rust type to JSON via serde_json, then converts to a Python dictionary
/// using `json.loads`
pub fn serialize_to_dict<T: serde::ser::Serialize>(py: Python<'_>, val: T) -> PyResult<Py<PyAny>> {
    let json_str = serde_json::to_string(&val)
        .map_err(|e| PyValueError::new_err(format!("Failed to serialize to JSON: {e:?}")))?;
    JSON_LOADS
        .get(py)
        .ok_or_else(|| {
            PyRuntimeError::new_err(
                "TensorZero: JSON_LOADS was not initialized. This should never happen",
            )
        })?
        .call1(py, (json_str.into_pyobject(py)?,))
}

/// In the `render_samples` function, we need to be able to accept both
/// impl StoredSample objects passed in from the output of the `list_inferences` and `list_datapoints` functions
/// and arbitrary Python objects that match the serialization pattern of the `StoredSample`
/// type.
/// This is necessary since developers might construct data for rendering by hand.
/// In order to support this, we first check if the object is a `StoredSample` object.
/// If it is, we return it directly.
/// If it is not, we assume it is a Python object that matches the serialization pattern of the
/// `StoredSample` type and deserialize it (and throw an error if it doesn't match).
pub fn deserialize_from_stored_sample<'a>(
    py: Python<'a>,
    obj: &Bound<'a, PyAny>,
) -> PyResult<StoredSampleItem> {
    if obj.is_instance_of::<StoredInference>() {
        Ok(StoredSampleItem::StoredInference(obj.extract()?))
    } else if obj.is_instance_of::<Datapoint>() {
        Ok(StoredSampleItem::Datapoint(obj.extract()?))
    } else {
        deserialize_from_pyobj(py, obj)
    }
}

/// In the `experimental_launch_optimization` function, we need to be able to accept
/// either an arbitrary Python object that matches the serialization pattern of the
/// `RenderedSample` type or a `RenderedSample` object.
pub fn deserialize_from_rendered_sample<'a>(
    py: Python<'a>,
    obj: &Bound<'a, PyAny>,
) -> PyResult<RenderedSample> {
    if obj.is_instance_of::<RenderedSample>() {
        Ok(obj.extract()?)
    } else {
        deserialize_from_pyobj(py, obj)
    }
}

pub fn deserialize_optimization_config(
    obj: &Bound<'_, PyAny>,
) -> PyResult<UninitializedOptimizerConfig> {
    if obj.is_instance_of::<UninitializedOpenAISFTConfig>() {
        Ok(UninitializedOptimizerConfig::OpenAISFT(obj.extract()?))
    } else if obj.is_instance_of::<UninitializedFireworksSFTConfig>() {
        Ok(UninitializedOptimizerConfig::FireworksSFT(obj.extract()?))
    } else {
        Err(PyValueError::new_err(
            "Invalid optimization config. Expected OpenAISFTConfig or FireworksSFTConfig",
        ))
    }
}

#[derive(Clone, Debug, Deserialize)]
pub enum StoredSampleItem {
    StoredInference(StoredInference),
    Datapoint(Datapoint),
}

impl StoredSample for StoredSampleItem {
    fn function_name(&self) -> &str {
        match self {
            StoredSampleItem::StoredInference(inference) => inference.function_name(),
            StoredSampleItem::Datapoint(datapoint) => datapoint.function_name(),
        }
    }

    fn input(&self) -> &ResolvedInput {
        match self {
            StoredSampleItem::StoredInference(inference) => inference.input(),
            StoredSampleItem::Datapoint(datapoint) => datapoint.input(),
        }
    }

    fn input_mut(&mut self) -> &mut ResolvedInput {
        match self {
            StoredSampleItem::StoredInference(inference) => inference.input_mut(),
            StoredSampleItem::Datapoint(datapoint) => datapoint.input_mut(),
        }
    }

    fn owned_simple_info(self) -> SimpleStoredSampleInfo {
        match self {
            StoredSampleItem::StoredInference(inference) => inference.owned_simple_info(),
            StoredSampleItem::Datapoint(datapoint) => datapoint.owned_simple_info(),
        }
    }
}

/// Converts a Python dictionary/list to json with `json.dumps`,
/// then deserializes to a Rust type via serde
pub fn deserialize_from_pyobj<'a, T: serde::de::DeserializeOwned>(
    py: Python<'a>,
    obj: &Bound<'a, PyAny>,
) -> PyResult<T> {
    let self_module = PyModule::import(py, "tensorzero.types")?;
    let to_dict_encoder: Bound<'_, PyAny> = self_module.getattr("TensorZeroTypeEncoder")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "cls"), to_dict_encoder)?;

    let json_str_obj = JSON_DUMPS
        .get(py)
        .ok_or_else(|| {
            PyRuntimeError::new_err(
                "TensorZero: JSON_DUMPS was not initialized. This should never happen",
            )
        })?
        .call(py, (obj,), Some(&kwargs))?;
    let json_str: Cow<'_, str> = json_str_obj.extract(py)?;
    let mut deserializer = serde_json::Deserializer::from_str(json_str.as_ref());
    let val: Result<T, _> = serde_path_to_error::deserialize(&mut deserializer);
    match val {
        Ok(val) => Ok(val),
        Err(e) => Err(tensorzero_core_error(
            py,
            &format!(
                "Failed to deserialize JSON to {}: {}",
                std::any::type_name::<T>(),
                e
            ),
        )?),
    }
}

pub fn tensorzero_error_class(py: Python<'_>) -> PyResult<&Py<PyAny>> {
    TENSORZERO_ERROR.get_or_try_init::<_, PyErr>(py, || {
        let self_module = PyModule::import(py, "tensorzero.types")?;
        let err: Bound<'_, PyAny> = self_module.getattr("TensorZeroError")?;
        Ok(err.unbind())
    })
}

pub fn tensorzero_core_error_class(py: Python<'_>) -> PyResult<&Py<PyAny>> {
    TENSORZERO_INTERNAL_ERROR.get_or_try_init::<_, PyErr>(py, || {
        let self_module = PyModule::import(py, "tensorzero.types")?;
        let err: Bound<'_, PyAny> = self_module.getattr("TensorZeroInternalError")?;
        Ok(err.unbind())
    })
}

pub fn tensorzero_core_error(py: Python<'_>, msg: &str) -> PyResult<PyErr> {
    Ok(PyErr::from_value(
        tensorzero_core_error_class(py)?.bind(py).call1((msg,))?,
    ))
}
