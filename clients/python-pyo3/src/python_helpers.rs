use pyo3::{exceptions::PyValueError, prelude::*, sync::GILOnceCell};
use tensorzero_rust::{FeedbackResponse, InferenceResponse, InferenceResponseChunk};

use crate::JSON_LOADS;

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

// We lazily lookup the corresponding Python class/method for all of these helpers,
// since they're not available during module initialization.
pub fn parse_feedback_response(py: Python<'_>, data: FeedbackResponse) -> PyResult<Py<PyAny>> {
    static PARSE_FEEDBACK_RESPONSE: GILOnceCell<Py<PyAny>> = GILOnceCell::new();
    let parse_feedback_response =
        PARSE_FEEDBACK_RESPONSE.get_or_try_init::<_, PyErr>(py, || {
            let self_module = PyModule::import(py, "tensorzero")?;
            Ok(self_module.getattr("FeedbackResponse")?.unbind())
        })?;
    let json_data = serialize_to_dict(py, data)?;
    let python_parsed = parse_feedback_response.call1(py, (json_data,))?;
    Ok(python_parsed.into_any())
}

pub fn parse_inference_response(py: Python<'_>, data: InferenceResponse) -> PyResult<Py<PyAny>> {
    let json_data = serialize_to_dict(py, data)?;
    static PARSE_INFERENCE_RESPONSE: GILOnceCell<Py<PyAny>> = GILOnceCell::new();
    let parse_inference_response =
        PARSE_INFERENCE_RESPONSE.get_or_try_init::<_, PyErr>(py, || {
            let self_module = PyModule::import(py, "tensorzero")?;
            Ok(self_module.getattr("parse_inference_response")?.unbind())
        })?;
    let python_parsed = parse_inference_response.call1(py, (json_data,))?;
    Ok(python_parsed.into_any())
}

pub fn parse_inference_chunk(py: Python<'_>, chunk: InferenceResponseChunk) -> PyResult<Py<PyAny>> {
    static PARSE_INFERENCE_CHUNK: GILOnceCell<Py<PyAny>> = GILOnceCell::new();
    let parse_inference_chunk = PARSE_INFERENCE_CHUNK.get_or_try_init::<_, PyErr>(py, || {
        let self_module = PyModule::import(py, "tensorzero")?;
        Ok(self_module.getattr("parse_inference_chunk")?.unbind())
    })?;

    let json_data = serialize_to_dict(py, chunk)?;
    let python_parsed = parse_inference_chunk.call1(py, (json_data,))?;
    Ok(python_parsed.into_any())
}
