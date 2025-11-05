use pyo3::prelude::*;
use uuid::Uuid;

use crate::inference::types::pyo3_helpers::{uuid_to_python, deserialize_from_pyobj, serialize_to_dict};
use super::types::*;

// ============================================================================
// Response Types - these just need getters
// ============================================================================

#[cfg(feature = "pyo3")]
#[pymethods]
impl CreateDatapointsResponse {
    #[getter]
    fn ids<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        self.ids
            .iter()
            .map(|id| uuid_to_python(py, *id))
            .collect()
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl UpdateDatapointsResponse {
    #[getter]
    fn ids<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        self.ids
            .iter()
            .map(|id| uuid_to_python(py, *id))
            .collect()
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl GetDatapointsResponse {
    #[getter]
    fn datapoints(&self) -> Vec<crate::endpoints::datasets::Datapoint> {
        self.datapoints.clone()
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl DeleteDatapointsResponse {
    #[getter]
    fn num_deleted_datapoints(&self) -> u64 {
        self.num_deleted_datapoints
    }
}

// ============================================================================
// Simple Value Types - just getters and constructors
// ============================================================================

#[cfg(feature = "pyo3")]
#[pymethods]
impl JsonDatapointOutputUpdate {
    #[new]
    fn new(raw: String) -> Self {
        Self { raw }
    }

    #[getter]
    fn raw(&self) -> String {
        self.raw.clone()
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl DatapointMetadataUpdate {
    #[new]
    #[pyo3(signature = (*, name))]
    fn new(name: Option<Option<String>>) -> Self {
        Self { name }
    }

    #[getter]
    fn name(&self) -> Option<Option<String>> {
        self.name.clone()
    }
}

// ============================================================================
// Update Request Types
// ============================================================================

#[cfg(feature = "pyo3")]
#[pymethods]
impl UpdateChatDatapointRequest {
    #[new]
    #[pyo3(signature = (*, id, input=None, output=None, tool_params, tags=None, metadata=None))]
    fn new(
        py: Python<'_>,
        id: Bound<'_, PyAny>,
        input: Option<Bound<'_, PyAny>>,
        output: Option<Bound<'_, PyAny>>,
        tool_params: Option<Option<crate::tool::DynamicToolParams>>,
        tags: Option<std::collections::HashMap<String, String>>,
        metadata: Option<DatapointMetadataUpdate>,
    ) -> PyResult<Self> {
        let id_str: String = id.extract()?;
        let id = Uuid::parse_str(&id_str)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid UUID: {e}")))?;

        let input = if let Some(input_obj) = input {
            Some(deserialize_from_pyobj(py, &input_obj)?)
        } else {
            None
        };

        let output = if let Some(output_obj) = output {
            Some(deserialize_from_pyobj(py, &output_obj)?)
        } else {
            None
        };

        Ok(Self {
            id,
            input,
            output,
            tool_params,
            tags,
            metadata,
        })
    }

    #[getter]
    fn id<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        uuid_to_python(py, self.id)
    }

    #[getter]
    fn input<'py>(&self, py: Python<'py>) -> PyResult<Option<Py<PyAny>>> {
        self.input.as_ref().map(|i| serialize_to_dict(py, i)).transpose()
    }

    #[getter]
    fn output<'py>(&self, py: Python<'py>) -> PyResult<Option<Py<PyAny>>> {
        self.output.as_ref().map(|o| serialize_to_dict(py, o)).transpose()
    }

    #[getter]
    fn tool_params(&self) -> Option<Option<crate::tool::DynamicToolParams>> {
        self.tool_params.clone()
    }

    #[getter]
    fn tags(&self) -> Option<std::collections::HashMap<String, String>> {
        self.tags.clone()
    }

    #[getter]
    fn metadata(&self) -> Option<DatapointMetadataUpdate> {
        self.metadata.clone()
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl UpdateJsonDatapointRequest {
    #[new]
    #[pyo3(signature = (*, id, input=None, output, output_schema=None, tags=None, metadata=None))]
    fn new(
        py: Python<'_>,
        id: Bound<'_, PyAny>,
        input: Option<Bound<'_, PyAny>>,
        output: Option<Option<JsonDatapointOutputUpdate>>,
        output_schema: Option<Bound<'_, PyAny>>,
        tags: Option<std::collections::HashMap<String, String>>,
        metadata: Option<DatapointMetadataUpdate>,
    ) -> PyResult<Self> {
        let id_str: String = id.extract()?;
        let id = Uuid::parse_str(&id_str)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid UUID: {e}")))?;

        let input = if let Some(input_obj) = input {
            Some(deserialize_from_pyobj(py, &input_obj)?)
        } else {
            None
        };

        let output_schema = if let Some(schema_obj) = output_schema {
            Some(deserialize_from_pyobj(py, &schema_obj)?)
        } else {
            None
        };

        Ok(Self {
            id,
            input,
            output,
            output_schema,
            tags,
            metadata,
        })
    }

    #[getter]
    fn id<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        uuid_to_python(py, self.id)
    }

    #[getter]
    fn input<'py>(&self, py: Python<'py>) -> PyResult<Option<Py<PyAny>>> {
        self.input.as_ref().map(|i| serialize_to_dict(py, i)).transpose()
    }

    #[getter]
    fn output(&self) -> Option<Option<JsonDatapointOutputUpdate>> {
        self.output.clone()
    }

    #[getter]
    fn output_schema<'py>(&self, py: Python<'py>) -> PyResult<Option<Py<PyAny>>> {
        self.output_schema.as_ref().map(|s| serialize_to_dict(py, s)).transpose()
    }

    #[getter]
    fn tags(&self) -> Option<std::collections::HashMap<String, String>> {
        self.tags.clone()
    }

    #[getter]
    fn metadata(&self) -> Option<DatapointMetadataUpdate> {
        self.metadata.clone()
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl UpdateDatapointMetadataRequest {
    #[new]
    fn new(_py: Python<'_>, id: Bound<'_, PyAny>, metadata: DatapointMetadataUpdate) -> PyResult<Self> {
        let id_str: String = id.extract()?;
        let id = Uuid::parse_str(&id_str)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid UUID: {e}")))?;

        Ok(Self { id, metadata })
    }

    #[getter]
    fn id<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        uuid_to_python(py, self.id)
    }

    #[getter]
    fn metadata(&self) -> DatapointMetadataUpdate {
        self.metadata.clone()
    }
}

// ============================================================================
// Create Request Types
// ============================================================================

#[cfg(feature = "pyo3")]
#[pymethods]
impl CreateChatDatapointRequest {
    #[new]
    #[pyo3(signature = (function_name, input, episode_id=None, output=None, dynamic_tool_params=None, tags=None, name=None))]
    fn new(
        py: Python<'_>,
        function_name: String,
        input: Bound<'_, PyAny>,
        episode_id: Option<Bound<'_, PyAny>>,
        output: Option<Bound<'_, PyAny>>,
        dynamic_tool_params: Option<crate::tool::DynamicToolParams>,
        tags: Option<std::collections::HashMap<String, String>>,
        name: Option<String>,
    ) -> PyResult<Self> {
        let input = deserialize_from_pyobj(py, &input)?;

        let output = if let Some(output_obj) = output {
            Some(deserialize_from_pyobj(py, &output_obj)?)
        } else {
            None
        };

        let episode_id = if let Some(ep_id) = episode_id {
            let id_str: String = ep_id.extract()?;
            Some(Uuid::parse_str(&id_str)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid UUID: {e}")))?)
        } else {
            None
        };

        Ok(Self {
            function_name,
            episode_id,
            input,
            output,
            dynamic_tool_params: dynamic_tool_params.unwrap_or_default(),
            tags,
            name,
        })
    }

    #[getter]
    fn function_name(&self) -> String {
        self.function_name.clone()
    }

    #[getter]
    fn episode_id<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        self.episode_id
            .map(|id| uuid_to_python(py, id))
            .transpose()
    }

    #[getter]
    fn input<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        serialize_to_dict(py, &self.input)
    }

    #[getter]
    fn output<'py>(&self, py: Python<'py>) -> PyResult<Option<Py<PyAny>>> {
        self.output.as_ref().map(|o| serialize_to_dict(py, o)).transpose()
    }

    #[getter]
    fn dynamic_tool_params(&self) -> crate::tool::DynamicToolParams {
        self.dynamic_tool_params.clone()
    }

    #[getter]
    fn tags(&self) -> Option<std::collections::HashMap<String, String>> {
        self.tags.clone()
    }

    #[getter]
    fn name(&self) -> Option<String> {
        self.name.clone()
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl CreateJsonDatapointRequest {
    #[new]
    #[pyo3(signature = (function_name, input, episode_id=None, output=None, output_schema=None, tags=None, name=None))]
    fn new(
        py: Python<'_>,
        function_name: String,
        input: Bound<'_, PyAny>,
        episode_id: Option<Bound<'_, PyAny>>,
        output: Option<JsonDatapointOutputUpdate>,
        output_schema: Option<Bound<'_, PyAny>>,
        tags: Option<std::collections::HashMap<String, String>>,
        name: Option<String>,
    ) -> PyResult<Self> {
        let input = deserialize_from_pyobj(py, &input)?;

        let episode_id = if let Some(ep_id) = episode_id {
            let id_str: String = ep_id.extract()?;
            Some(Uuid::parse_str(&id_str)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid UUID: {e}")))?)
        } else {
            None
        };

        let output_schema = if let Some(schema_obj) = output_schema {
            Some(deserialize_from_pyobj(py, &schema_obj)?)
        } else {
            None
        };

        Ok(Self {
            function_name,
            episode_id,
            input,
            output,
            output_schema,
            tags,
            name,
        })
    }

    #[getter]
    fn function_name(&self) -> String {
        self.function_name.clone()
    }

    #[getter]
    fn episode_id<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        self.episode_id
            .map(|id| uuid_to_python(py, id))
            .transpose()
    }

    #[getter]
    fn input<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        serialize_to_dict(py, &self.input)
    }

    #[getter]
    fn output(&self) -> Option<JsonDatapointOutputUpdate> {
        self.output.clone()
    }

    #[getter]
    fn output_schema<'py>(&self, py: Python<'py>) -> PyResult<Option<Py<PyAny>>> {
        self.output_schema.as_ref().map(|s| serialize_to_dict(py, s)).transpose()
    }

    #[getter]
    fn tags(&self) -> Option<std::collections::HashMap<String, String>> {
        self.tags.clone()
    }

    #[getter]
    fn name(&self) -> Option<String> {
        self.name.clone()
    }
}

// ============================================================================
// List/Filter Request Types
// ============================================================================

#[cfg(feature = "pyo3")]
#[pymethods]
impl ListDatapointsRequest {
    #[new]
    #[pyo3(signature = (function_name=None, limit=None, page_size=None, offset=None, filter=None))]
    #[allow(deprecated)]
    fn new(
        py: Python<'_>,
        function_name: Option<String>,
        limit: Option<u32>,
        page_size: Option<u32>,
        offset: Option<u32>,
        filter: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let filter = if let Some(filter_obj) = filter {
            Some(deserialize_from_pyobj(py, &filter_obj)?)
        } else {
            None
        };

        Ok(Self {
            function_name,
            limit,
            page_size,
            offset,
            filter,
        })
    }

    #[getter]
    fn function_name(&self) -> Option<String> {
        self.function_name.clone()
    }

    #[getter]
    fn limit(&self) -> Option<u32> {
        self.limit
    }

    #[getter]
    #[allow(deprecated)]
    fn page_size(&self) -> Option<u32> {
        self.page_size
    }

    #[getter]
    fn offset(&self) -> Option<u32> {
        self.offset
    }

    #[getter]
    fn filter<'py>(&self, py: Python<'py>) -> PyResult<Option<Py<PyAny>>> {
        self.filter.as_ref().map(|f| serialize_to_dict(py, f)).transpose()
    }
}

// ============================================================================
// Enum Types - not exposed to Python directly
// ============================================================================

// UpdateDatapointRequest and CreateDatapointRequest are not exposed directly.
// They are created from dicts when needed in the gateway methods.

#[cfg(feature = "pyo3")]
#[pymethods]
impl CreateDatapointsFromInferenceOutputSource {
    #[new]
    fn new() -> Self {
        CreateDatapointsFromInferenceOutputSource::None
    }
}

// CreateDatapointsFromInferenceRequestParams is not exposed directly to Python
// Instead, it's created from dicts when needed
