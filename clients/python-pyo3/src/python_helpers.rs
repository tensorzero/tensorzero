//! This module defines several serialization/deserialization helpers that we use to convert
//! between Python classes and the corresponding Rust types in the Rust `tensorzero` client.

use std::{borrow::Cow, collections::HashMap};

use pyo3::{exceptions::PyValueError, intern, prelude::*, sync::GILOnceCell, types::PyDict};
use tensorzero_rust::{FeedbackResponse, InferenceResponse, InferenceResponseChunk, Tool};
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

/// Serializes a Rust type to JSON via serde_json, then converts to a Python dictionary
/// using `json.loads`
pub fn serialize_to_dict<T: serde::ser::Serialize>(py: Python<'_>, val: T) -> PyResult<Py<PyAny>> {
    let json_str = serde_json::to_string(&val)
        .map_err(|e| PyValueError::new_err(format!("Failed to serialize to JSON: {e:?}")))?;
    JSON_LOADS
        .get(py)
        .expect("JSON_LOADS was not initialized")
        .call1(py, (json_str.into_pyobject(py)?,))
}

// Converts a Python dictionary/list to json with `json.dumps`,
/// then deserializes to a Rust type via serde
pub fn deserialize_from_pyobj<'a, T: serde::de::DeserializeOwned>(
    py: Python<'a>,
    obj: &Bound<'a, PyAny>,
) -> PyResult<T> {
    let self_module = PyModule::import(py, "tensorzero.types")?;
    let to_dict_encoder: Bound<'_, PyAny> = self_module.getattr("ToDictEncoder")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "cls"), to_dict_encoder)?;

    let json_str_obj = JSON_DUMPS
        .get(py)
        .expect("JSON_DUMPS was not initialized")
        .call(py, (obj,), Some(&kwargs))?;
    let json_str: Cow<'_, str> = json_str_obj.extract(py)?;
    let val = serde_json::from_str::<T>(json_str.as_ref());
    match val {
        Ok(val) => Ok(val),
        Err(e) => Err(tensorzero_internal_error(
            py,
            &format!(
                "Failed to deserialize JSON to {}: {:?}",
                std::any::type_name::<T>(),
                e
            ),
        )?),
    }
}

pub fn python_uuid_to_uuid(
    param_name: &str,
    val: Option<Bound<'_, PyAny>>,
) -> PyResult<Option<Uuid>> {
    // We could try to be more clever and extract the UUID bytes from Python, but for now
    // we just stringify and re-parse.
    Ok(if let Some(val) = val {
        Some(Uuid::parse_str(&val.str()?.to_cow()?).map_err(|e| {
            PyValueError::new_err(format!("Failed to parse {param_name} as UUID: {e:?}"))
        })?)
    } else {
        None
    })
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
