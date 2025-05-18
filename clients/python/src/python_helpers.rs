//! This module defines several serialization/deserialization helpers that we use to convert
//! between Python classes and the corresponding Rust types in the Rust `tensorzero` client.

use std::{borrow::Cow, collections::HashMap};

use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    intern,
    prelude::*,
    sync::GILOnceCell,
    types::{PyAny, PyDict},
};
use tensorzero_internal::endpoints::dynamic_evaluation_run::DynamicEvaluationRunEpisodeResponse;
use tensorzero_rust::{
    Datapoint, DynamicEvaluationRunResponse, FeedbackResponse, InferenceResponse,
    InferenceResponseChunk, Tool,
};
use uuid::Uuid;

use crate::{tensorzero_internal_error, JSON_DUMPS, JSON_LOADS};

// We lazily lookup the corresponding Python class/method for all of these helpers,
// since they're not available during module initialization.
pub fn parse_feedback_response(py: Python<'_>, data: FeedbackResponse) -> PyResult<Py<PyAny>> {
    static PARSE_FEEDBACK_RESPONSE: GILOnceCell<Py<PyAny>> = GILOnceCell::new();
    // This should never actually fail, since we're just importing code defined in our own Python
    // package. However, we still produce a Python error if it fails, rather than panicking
    // and bringing down the entire Python process.
    let parse_feedback_response =
        PARSE_FEEDBACK_RESPONSE.get_or_try_init::<_, PyErr>(py, || {
            let self_module = PyModule::import(py, "tensorzero.types")?;
            Ok(self_module.getattr("FeedbackResponse")?.unbind())
        })?;
    let json_data = serialize_to_dict(py, data)?;
    let python_parsed = parse_feedback_response.call1(py, (json_data,))?;
    Ok(python_parsed.into_any())
}

pub fn parse_inference_response(py: Python<'_>, data: InferenceResponse) -> PyResult<Py<PyAny>> {
    let json_data = serialize_to_dict(py, data)?;
    static PARSE_INFERENCE_RESPONSE: GILOnceCell<Py<PyAny>> = GILOnceCell::new();
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

pub fn parse_datapoint(py: Python<'_>, data: Datapoint) -> PyResult<Py<PyAny>> {
    let json_datapoint = serialize_to_dict(py, data)?;
    static PARSE_DATAPOINT: GILOnceCell<Py<PyAny>> = GILOnceCell::new();
    // This should never actually fail, since we're just importing code defined in our own Python
    // package. However, we still produce a Python error if it fails, rather than panicking
    // and bringing down the entire Python process.
    let parse_datapoint = PARSE_DATAPOINT.get_or_try_init::<_, PyErr>(py, || {
        let self_module = PyModule::import(py, "tensorzero.types")?;
        Ok(self_module.getattr("parse_datapoint")?.unbind())
    })?;
    let python_parsed = parse_datapoint.call1(py, (json_datapoint,))?;
    Ok(python_parsed.into_any())
}

pub fn parse_inference_chunk(py: Python<'_>, chunk: InferenceResponseChunk) -> PyResult<Py<PyAny>> {
    static PARSE_INFERENCE_CHUNK: GILOnceCell<Py<PyAny>> = GILOnceCell::new();
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

pub fn parse_dynamic_evaluation_run_response(
    py: Python<'_>,
    data: DynamicEvaluationRunResponse,
) -> PyResult<Py<PyAny>> {
    static PARSE_DYNAMIC_EVALUATION_RUN_RESPONSE: GILOnceCell<Py<PyAny>> = GILOnceCell::new();
    // This should never actually fail, since we're just importing code defined in our own Python
    // package. However, we still produce a Python error if it fails, rather than panicking
    // and bringing down the entire Python process.
    let parse_dynamic_evaluation_run_response = PARSE_DYNAMIC_EVALUATION_RUN_RESPONSE
        .get_or_try_init::<_, PyErr>(py, || {
            let self_module = PyModule::import(py, "tensorzero.types")?;
            Ok(self_module
                .getattr("parse_dynamic_evaluation_run_response")?
                .unbind())
        })?;
    let json_data = serialize_to_dict(py, data)?;
    let python_parsed = parse_dynamic_evaluation_run_response.call1(py, (json_data,))?;
    Ok(python_parsed.into_any())
}

pub fn parse_dynamic_evaluation_run_episode_response(
    py: Python<'_>,
    data: DynamicEvaluationRunEpisodeResponse,
) -> PyResult<Py<PyAny>> {
    static PARSE_DYNAMIC_EVALUATION_RUN_EPISODE_RESPONSE: GILOnceCell<Py<PyAny>> =
        GILOnceCell::new();
    // This should never actually fail, since we're just importing code defined in our own Python
    // package. However, we still produce a Python error if it fails, rather than panicking
    // and bringing down the entire Python process.
    let parse_dynamic_evaluation_run_episode_response =
        PARSE_DYNAMIC_EVALUATION_RUN_EPISODE_RESPONSE.get_or_try_init::<_, PyErr>(py, || {
            let self_module = PyModule::import(py, "tensorzero.types")?;
            Ok(self_module
                .getattr("parse_dynamic_evaluation_run_episode_response")?
                .unbind())
        })?;
    let json_data = serialize_to_dict(py, data)?;
    let python_parsed = parse_dynamic_evaluation_run_episode_response.call1(py, (json_data,))?;
    Ok(python_parsed.into_any())
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

/// Checks if a type is NotRequired (from typing_extensions or Python 3.11+)
pub fn is_not_required(py: Python<'_>, obj: &PyAny) -> bool {
    // Try typing_extensions first, then typing (Python 3.11+)
    let typing_ext = py.import("typing_extensions").ok();
    let typing = py.import("typing").ok();
    if let Some(typing_ext) = typing_ext {
        if let Ok(not_required) = typing_ext.getattr("NotRequired") {
            if obj.is(not_required) {
                return true;
            }
        }
    }
    if let Some(typing) = typing {
        if let Ok(not_required) = typing.getattr("NotRequired") {
            if obj.is(not_required) {
                return true;
            }
        }
    }
    false
}

pub fn deserialize_from_pyobj<T: serde::de::DeserializeOwned>(
    py: Python<'_>,
    obj: &PyAny,
) -> PyResult<T> {
    // If the object is NotRequired, raise an error or handle as needed
    if is_not_required(py, obj) {
        return Err(PyValueError::new_err(
            "NotRequired fields should not be treated as Optional. Please provide a value or omit the field.",
        ));
    }
    // your existing deserialization logic
    let json_str = crate::python_helpers::to_json_string(py, obj)?;
    serde_json::from_str(&json_str).map_err(|e| PyValueError::new_err(format!("Failed to deserialize: {e}")))
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
