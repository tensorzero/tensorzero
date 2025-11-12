//! This module defines several serialization/deserialization helpers that we use to convert
//! between Python classes and the corresponding Rust types in the Rust `tensorzero` client.

use std::collections::HashMap;

use pyo3::{exceptions::PyValueError, prelude::*, sync::PyOnceLock, types::PyDict};
use tensorzero_core::endpoints::workflow_evaluation_run::WorkflowEvaluationRunEpisodeResponse;
use tensorzero_core::inference::types::pyo3_helpers::{deserialize_from_pyobj, serialize_to_dict};
use tensorzero_rust::{
    FeedbackResponse, InferenceResponse, InferenceResponseChunk, Tool,
    WorkflowEvaluationRunResponse,
};
use uuid::Uuid;

// We lazily lookup the corresponding Python class/method for all of these helpers,
// since they're not available during module initialization.
pub fn parse_feedback_response(py: Python<'_>, data: FeedbackResponse) -> PyResult<Py<PyAny>> {
    static PARSE_FEEDBACK_RESPONSE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    static UUID: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    // This should never actually fail, since we're just importing code defined in our own Python
    // package. However, we still produce a Python error if it fails, rather than panicking
    // and bringing down the entire Python process.
    let parse_feedback_response =
        PARSE_FEEDBACK_RESPONSE.get_or_try_init::<_, PyErr>(py, || {
            let self_module = PyModule::import(py, "tensorzero.types")?;
            Ok(self_module.getattr("FeedbackResponse")?.unbind())
        })?;
    let uuid = UUID.get_or_try_init::<_, PyErr>(py, || {
        let self_module = PyModule::import(py, "uuid")?;
        Ok(self_module.getattr("UUID")?.unbind())
    })?;
    let uuid_str = data.feedback_id.to_string();
    let python_uuid = uuid.call1(py, (uuid_str,))?;
    let python_parsed = parse_feedback_response.call1(py, (python_uuid,))?;
    Ok(python_parsed.into_any())
}

pub fn parse_inference_response(py: Python<'_>, data: InferenceResponse) -> PyResult<Py<PyAny>> {
    convert_response_to_python(py, data, "tensorzero.types", "InferenceResponse")
}

pub fn parse_inference_chunk(py: Python<'_>, chunk: InferenceResponseChunk) -> PyResult<Py<PyAny>> {
    convert_response_to_python(py, chunk, "tensorzero.types", "InferenceChunk")
}

pub fn parse_workflow_evaluation_run_response(
    py: Python<'_>,
    data: WorkflowEvaluationRunResponse,
) -> PyResult<Py<PyAny>> {
    convert_response_to_python(py, data, "tensorzero.types", "WorkflowEvaluationRunResponse")
}

pub fn parse_workflow_evaluation_run_episode_response(
    py: Python<'_>,
    data: WorkflowEvaluationRunEpisodeResponse,
) -> PyResult<Py<PyAny>> {
    convert_response_to_python(
        py,
        data,
        "tensorzero.types",
        "WorkflowEvaluationRunEpisodeResponse",
    )
}

pub fn python_uuid_to_uuid(param_name: &str, val: Bound<'_, PyAny>) -> PyResult<Uuid> {
    // We could try to be more clever and extract the UUID bytes from Python, but for now
    // we just stringify and re-parse.
    let val_str = val.str()?;
    let val_cow = val_str.to_cow()?;
    Uuid::parse_str(&val_cow)
        .map_err(|e| PyValueError::new_err(format!("Failed to parse {param_name} as UUID: {e:?}")))
}

pub fn parse_tool(py: Python<'_>, key_vals: HashMap<String, Bound<'_, PyAny>>) -> PyResult<Tool> {
    let name = key_vals.get("name").ok_or_else(|| {
        PyValueError::new_err(format!("Missing 'name' in additional tool: {key_vals:?}"))
    })?;
    let description = key_vals.get("description").ok_or_else(|| {
        PyValueError::new_err(format!(
            "Missing 'description' in additional tool: {key_vals:?}"
        ))
    })?;
    let params = key_vals.get("parameters").ok_or_else(|| {
        PyValueError::new_err(format!(
            "Missing 'parameters' in additional tool: {key_vals:?}"
        ))
    })?;
    let strict = if let Some(val) = key_vals.get("strict") {
        val.extract::<bool>()?
    } else {
        false
    };
    let tool_params: serde_json::Value = deserialize_from_pyobj(py, params)?;
    Ok(Tool {
        name: name.extract()?,
        description: description.extract()?,
        parameters: tool_params,
        strict,
    })
}

/// Converts a serializable Rust response type to a Python dataclass
/// by serializing to JSON dict and using dacite.from_dict to construct
/// the dataclass with proper nested type handling.
///
/// This helper automatically handles type conversions (e.g., UUID → string)
/// via serde serialization, and dacite handles nested dataclass construction.
/// It configures dacite with type_hooks to handle UUID conversion from strings.
///
/// # Arguments
/// * `py` - Python interpreter
/// * `response` - Rust response object that implements Serialize
/// * `python_module` - Python module path (e.g., "tensorzero")
/// * `python_class` - Python class name (e.g., "UpdateDatapointsResponse")
///
/// # Example
/// ```ignore
/// convert_response_to_python_dataclass(
///     py,
///     update_response,
///     "tensorzero",
///     "UpdateDatapointsResponse"
/// )
/// ```
pub fn convert_response_to_python_dataclass<T: serde::Serialize>(
    py: Python<'_>,
    response: T,
    python_module: &str,
    python_class: &str,
) -> PyResult<Py<PyAny>> {
    // Serialize Rust response to JSON dict
    let dict = serialize_to_dict(py, &response)?;

    // Import the target dataclass
    let module = PyModule::import(py, python_module)?;
    let data_class = module.getattr(python_class)?;

    // Import UUID class for type_hooks
    let uuid_module = PyModule::import(py, "uuid")?;
    let uuid_class = uuid_module.getattr("UUID")?;

    // Use dacite.from_dict to construct the dataclass, so that it can handle nested dataclass construction.
    let dacite = PyModule::import(py, "dacite")?;
    let from_dict = dacite.getattr("from_dict")?;
    let config_class = dacite.getattr("Config")?;

    // Create type_hooks dict: {UUID: UUID} to convert str -> UUID
    let type_hooks = PyDict::new(py);
    type_hooks.set_item(&uuid_class, &uuid_class)?;

    // Create dacite.Config with type_hooks
    let config_kwargs = PyDict::new(py);
    config_kwargs.set_item("type_hooks", type_hooks)?;
    let config = config_class.call((), Some(&config_kwargs))?;

    // Call dacite.from_dict(data_class=TargetClass, data=dict, config=config)
    let kwargs = PyDict::new(py);
    kwargs.set_item("data_class", data_class)?;
    kwargs.set_item("data", dict)?;
    kwargs.set_item("config", config)?;

    from_dict.call((), Some(&kwargs)).map(Bound::unbind)
}
