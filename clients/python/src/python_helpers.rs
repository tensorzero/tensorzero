//! This module defines several serialization/deserialization helpers that we use to convert
//! between Python classes and the corresponding Rust types in the Rust `tensorzero` client.

use std::collections::HashMap;

use pyo3::{exceptions::PyValueError, prelude::*, sync::PyOnceLock, types::PyDict};
use tensorzero_core::endpoints::workflow_evaluation_run::WorkflowEvaluationRunEpisodeResponse;
use tensorzero_core::inference::types::pyo3_helpers::{deserialize_from_pyobj, serialize_to_dict};
use tensorzero_rust::{
    FeedbackResponse, FunctionTool, InferenceResponse, InferenceResponseChunk,
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
    let json_data = serialize_to_dict(py, data)?;
    static PARSE_INFERENCE_RESPONSE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    // This should never actually fail, since we're just importing code defined in our own Python
    // package. However, we still produce a Python error if it fails, rather than panicking
    // and bringing down the entire Python process.
    let parse_inference_response =
        PARSE_INFERENCE_RESPONSE.get_or_try_init::<_, PyErr>(py, || {
            let self_module = PyModule::import(py, "tensorzero.types")?;
            Ok(self_module.getattr("parse_inference_response")?.unbind())
        })?;
    let python_parsed = parse_inference_response.call1(py, (json_data,))?;
    Ok(python_parsed.into_any())
}

pub fn parse_inference_chunk(py: Python<'_>, chunk: InferenceResponseChunk) -> PyResult<Py<PyAny>> {
    static PARSE_INFERENCE_CHUNK: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    // This should never actually fail, since we're just importing code defined in our own Python
    // package. However, we still produce a Python error if it fails, rather than panicking
    // and bringing down the entire Python process.
    let parse_inference_chunk = PARSE_INFERENCE_CHUNK.get_or_try_init::<_, PyErr>(py, || {
        let self_module = PyModule::import(py, "tensorzero.types")?;
        Ok(self_module.getattr("parse_inference_chunk")?.unbind())
    })?;

    let json_data = serialize_to_dict(py, chunk)?;
    let python_parsed = parse_inference_chunk.call1(py, (json_data,))?;
    Ok(python_parsed.into_any())
}

pub fn parse_workflow_evaluation_run_response(
    py: Python<'_>,
    data: WorkflowEvaluationRunResponse,
) -> PyResult<Py<PyAny>> {
    static PARSE_WORKFLOW_EVALUATION_RUN_RESPONSE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    // This should never actually fail, since we're just importing code defined in our own Python
    // package. However, we still produce a Python error if it fails, rather than panicking
    // and bringing down the entire Python process.
    let parse_workflow_evaluation_run_response = PARSE_WORKFLOW_EVALUATION_RUN_RESPONSE
        .get_or_try_init::<_, PyErr>(py, || {
            let self_module = PyModule::import(py, "tensorzero.types")?;
            Ok(self_module
                .getattr("parse_workflow_evaluation_run_response")?
                .unbind())
        })?;
    let json_data = serialize_to_dict(py, data)?;
    let python_parsed = parse_workflow_evaluation_run_response.call1(py, (json_data,))?;
    Ok(python_parsed.into_any())
}

pub fn parse_workflow_evaluation_run_episode_response(
    py: Python<'_>,
    data: WorkflowEvaluationRunEpisodeResponse,
) -> PyResult<Py<PyAny>> {
    static PARSE_WORKFLOW_EVALUATION_RUN_EPISODE_RESPONSE: PyOnceLock<Py<PyAny>> =
        PyOnceLock::new();
    // This should never actually fail, since we're just importing code defined in our own Python
    // package. However, we still produce a Python error if it fails, rather than panicking
    // and bringing down the entire Python process.
    let parse_workflow_evaluation_run_episode_response =
        PARSE_WORKFLOW_EVALUATION_RUN_EPISODE_RESPONSE.get_or_try_init::<_, PyErr>(py, || {
            let self_module = PyModule::import(py, "tensorzero.types")?;
            Ok(self_module
                .getattr("parse_workflow_evaluation_run_episode_response")?
                .unbind())
        })?;
    let json_data = serialize_to_dict(py, data)?;
    let python_parsed = parse_workflow_evaluation_run_episode_response.call1(py, (json_data,))?;
    Ok(python_parsed.into_any())
}

pub fn python_uuid_to_uuid(param_name: &str, val: Bound<'_, PyAny>) -> PyResult<Uuid> {
    // We could try to be more clever and extract the UUID bytes from Python, but for now
    // we just stringify and re-parse.
    let val_str = val.str()?;
    let val_cow = val_str.to_cow()?;
    Uuid::parse_str(&val_cow)
        .map_err(|e| PyValueError::new_err(format!("Failed to parse {param_name} as UUID: {e:?}")))
}

pub fn parse_tool(
    py: Python<'_>,
    key_vals: HashMap<String, Bound<'_, PyAny>>,
) -> PyResult<FunctionTool> {
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
    Ok(FunctionTool {
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
    response: &T,
    python_module: &str,
    python_class: &str,
) -> PyResult<Py<PyAny>> {
    // Serialize Rust response to JSON dict
    let dict = serialize_to_dict(py, response)?;

    // Import the target dataclass
    let module = PyModule::import(py, python_module)?;
    let data_class = module.getattr(python_class)?;

    // Use dacite.from_dict to construct the dataclass, so that it can handle nested dataclass construction.
    let dacite = PyModule::import(py, "dacite")?;
    let from_dict = dacite.getattr("from_dict")?;

    // Call dacite.from_dict(data_class=TargetClass, data=dict)
    let kwargs = PyDict::new(py);
    kwargs.set_item("data_class", data_class)?;
    kwargs.set_item("data", dict)?;

    from_dict.call((), Some(&kwargs)).map(Bound::unbind)
}
