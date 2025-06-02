//! This module defines several serialization/deserialization helpers that we use to convert
//! between Python classes and the corresponding Rust types in the Rust `tensorzero` client.

use std::collections::HashMap;

use pyo3::{exceptions::PyValueError, prelude::*, sync::GILOnceCell};
use tensorzero_internal::endpoints::dynamic_evaluation_run::DynamicEvaluationRunEpisodeResponse;
use tensorzero_internal::inference::types::pyo3_helpers::{
    deserialize_from_pyobj, serialize_to_dict,
};
use tensorzero_rust::{
    Datapoint, DynamicEvaluationRunResponse, FeedbackResponse, InferenceResponse,
    InferenceResponseChunk, Tool,
};
use uuid::Uuid;

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
