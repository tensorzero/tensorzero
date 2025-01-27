use std::{collections::HashMap, path::PathBuf, sync::Arc};

use futures::StreamExt;
use pyo3::{
    exceptions::{PyStopAsyncIteration, PyStopIteration, PyValueError},
    intern,
    prelude::*,
    sync::GILOnceCell,
    types::{PyDict, PyString, PyType},
};
use python_helpers::{
    parse_feedback_response, parse_inference_chunk, parse_inference_response, serialize_to_dict,
};
use tensorzero_rust::{
    err_to_http, Client, ClientBuilder, ClientBuilderMode, ClientInferenceParams,
    ClientSecretString, DynamicToolParams, FeedbackParams, InferenceOutput, InferenceParams,
    InferenceStream, Input, TensorZeroError, Tool,
};
use tokio::sync::Mutex;
use url::Url;
use uuid::Uuid;

mod python_helpers;

// TODO - this should extend the python `ABC` class once pyo3 supports it: https://github.com/PyO3/pyo3/issues/991
#[pyclass(subclass)]
struct BaseTensorZeroGateway {
    client: Arc<Client>,
}

#[pyclass]
struct AsyncStreamWrapper {
    stream: Arc<Mutex<InferenceStream>>,
}

#[pymethods]
impl AsyncStreamWrapper {
    fn __aiter__(this: Py<Self>) -> Py<Self> {
        this
    }

    // This method returns a Python `Future` which will either resolve to a chunk of the stream,
    // or raise a `StopAsyncIteration` exception if the stream is finished.
    fn __anext__<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PyAny>> {
        let stream = self.stream.clone();
        // The Rust future relies on a tokio runtime (via `reqwest` and `hyper`), so
        // we need to use `pyo3_async_runtimes` to convert it into a Python future that runs
        // on Tokio. Inside the `async move` block, we can `.await` on Rust futures just like
        // we would in normal Rust code.
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let chunk = stream.lock().await.next().await;
            let Some(chunk) = chunk else {
                return Err(PyStopAsyncIteration::new_err(()));
            };
            // The overall 'async move' future needs to be 'static, so we cannot capture
            // the `py` parameter from `__anext__`,
            Python::with_gil(|py| {
                let chunk = match chunk {
                    Ok(chunk) => chunk,
                    Err(e) => {
                        return Err(convert_error(py, err_to_http(e))?);
                    }
                };
                parse_inference_chunk(py, chunk)
            })
        })
    }
}

#[pyclass]
struct StreamWrapper {
    stream: Arc<Mutex<InferenceStream>>,
}

#[pymethods]
impl StreamWrapper {
    fn __iter__(this: Py<Self>) -> Py<Self> {
        this
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let stream = self.stream.clone();
        pyo3_async_runtimes::tokio::get_runtime().block_on(async move {
            let chunk = stream.lock().await.next().await;
            let Some(chunk) = chunk else {
                return Err(PyStopIteration::new_err(()));
            };
            let chunk = match chunk {
                Ok(chunk) => chunk,
                Err(e) => {
                    return Err(tensorzero_internal_error(
                        py,
                        &format!("Failed to read streaming chunk: {e:?}"),
                    )?);
                }
            };
            parse_inference_chunk(py, chunk)
        })
    }
}

#[pymethods]
impl BaseTensorZeroGateway {
    #[new]
    fn new(py: Python<'_>, base_url: &str) -> PyResult<Self> {
        let client = ClientBuilder::new(ClientBuilderMode::HTTPGateway {
            url: Url::parse(base_url)
                .map_err(|e| PyValueError::new_err(format!("Failed to parse base_url: {e:?}")))?,
        })
        .build_http();
        let client = match client {
            Ok(client) => client,
            Err(e) => {
                return Err(tensorzero_internal_error(
                    py,
                    &format!("Failed to construct TensorZero client: {e:?}"),
                )?);
            }
        };

        Ok(Self {
            client: Arc::new(client),
        })
    }

    #[pyo3(signature = (*, function_name, input, episode_id=None, stream=None, params=None, variant_name=None, dryrun=None, allowed_tools=None, additional_tools=None, tool_choice=None, parallel_tool_calls=None, tags=None, credentials=None))]
    #[allow(clippy::too_many_arguments)]
    fn _prepare_inference_request(
        this: PyRef<'_, Self>,
        function_name: String,
        input: Bound<'_, PyDict>,
        episode_id: Option<Bound<'_, PyAny>>,
        stream: Option<bool>,
        params: Option<&Bound<'_, PyDict>>,
        variant_name: Option<String>,
        dryrun: Option<bool>,
        allowed_tools: Option<Vec<String>>,
        additional_tools: Option<Vec<HashMap<String, Bound<'_, PyAny>>>>,
        tool_choice: Option<Bound<'_, PyAny>>,
        parallel_tool_calls: Option<bool>,
        tags: Option<HashMap<String, String>>,
        credentials: Option<HashMap<String, ClientSecretString>>,
    ) -> PyResult<Py<PyAny>> {
        let params = BaseTensorZeroGateway::prepare_inference_params(
            this.py(),
            function_name,
            input,
            episode_id,
            stream,
            params,
            variant_name,
            dryrun,
            allowed_tools,
            additional_tools,
            tool_choice,
            parallel_tool_calls,
            tags,
            credentials,
        )?;
        serialize_to_dict(this.py(), params)
    }
}

#[pyclass(extends=BaseTensorZeroGateway)]
struct TensorZeroGateway {}

/// Converts a Python dictionary to json with `json.dumps`,
/// then deserializes to a Rust type via serde
fn deserialize_from_json<'a, T: serde::de::DeserializeOwned>(
    py: Python<'a>,
    dict: &Bound<'a, PyAny>,
) -> PyResult<T> {
    let self_module = PyModule::import(py, "tensorzero")?;
    let to_dict_encoder: Bound<'_, PyAny> = self_module.getattr("ToDictEncoder")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "cls"), to_dict_encoder)?;

    let json_str_obj = JSON_DUMPS
        .get(py)
        .expect("JSON_DUMPS was not initialized")
        .call(py, (dict,), Some(&kwargs))?;
    let json_str: &str = json_str_obj.extract(py)?;
    let val = serde_json::from_str::<T>(json_str);
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

fn to_uuid(param_name: &str, val: Option<Bound<'_, PyAny>>) -> PyResult<Option<Uuid>> {
    Ok(if let Some(val) = val {
        Some(Uuid::parse_str(val.str()?.to_str()?).map_err(|e| {
            PyValueError::new_err(format!("Failed to parse {param_name} as UUID: {e:?}"))
        })?)
    } else {
        None
    })
}

fn parse_tool(py: Python<'_>, key_vals: HashMap<String, Bound<'_, PyAny>>) -> PyResult<Tool> {
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
    let tool_params: serde_json::Value = deserialize_from_json(py, params)?;
    Ok(Tool {
        name: name.extract()?,
        description: description.extract()?,
        parameters: tool_params,
        strict,
    })
}

impl BaseTensorZeroGateway {
    fn prepare_feedback_params(
        py: Python<'_>,
        metric_name: String,
        value: Bound<'_, PyAny>,
        inference_id: Option<Bound<'_, PyAny>>,
        episode_id: Option<Bound<'_, PyAny>>,
        dryrun: Option<bool>,
        tags: Option<HashMap<String, String>>,
    ) -> PyResult<FeedbackParams> {
        Ok(FeedbackParams {
            metric_name,
            value: deserialize_from_json(py, &value)?,
            episode_id: to_uuid("episode_id", episode_id)?,
            inference_id: to_uuid("inference_id", inference_id)?,
            dryrun,
            tags: tags.unwrap_or_default(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_inference_params(
        py: Python<'_>,
        function_name: String,
        input: Bound<'_, PyDict>,
        episode_id: Option<Bound<'_, PyAny>>,
        stream: Option<bool>,
        params: Option<&Bound<'_, PyDict>>,
        variant_name: Option<String>,
        dryrun: Option<bool>,
        allowed_tools: Option<Vec<String>>,
        additional_tools: Option<Vec<HashMap<String, Bound<'_, PyAny>>>>,
        tool_choice: Option<Bound<'_, PyAny>>,
        parallel_tool_calls: Option<bool>,
        tags: Option<HashMap<String, String>>,
        credentials: Option<HashMap<String, ClientSecretString>>,
    ) -> PyResult<ClientInferenceParams> {
        let episode_id = to_uuid("episode_id", episode_id)?;

        let params: Option<InferenceParams> = if let Some(params) = params {
            deserialize_from_json(py, params)?
        } else {
            None
        };

        let additional_tools: Option<Vec<Tool>> = if let Some(tools) = additional_tools {
            Some(
                tools
                    .into_iter()
                    .map(|key_vals| parse_tool(py, key_vals))
                    .collect::<Result<Vec<Tool>, PyErr>>()?,
            )
        } else {
            None
        };

        let tool_choice = if let Some(tool_choice) = tool_choice {
            if tool_choice.is_instance_of::<PyString>() {
                Some(
                    serde_json::from_value(tool_choice.str()?.to_str()?.into()).map_err(|e| {
                        PyValueError::new_err(format!(
                            "Failed to parse tool_choice as valid JSON: {e:?}"
                        ))
                    })?,
                )
            } else {
                Some(deserialize_from_json(py, &tool_choice)?)
            }
        } else {
            None
        };

        let input: Input = deserialize_from_json(py, &input)?;

        Ok(ClientInferenceParams {
            function_name,
            stream,
            episode_id,
            variant_name,
            dryrun,
            tags: tags.unwrap_or_default(),
            params: params.unwrap_or_default(),
            dynamic_tool_params: DynamicToolParams {
                allowed_tools,
                parallel_tool_calls,
                additional_tools,
                tool_choice,
            },
            input,
            credentials: credentials.unwrap_or_default(),
            ..Default::default()
        })
    }
}
#[pymethods]
impl TensorZeroGateway {
    #[new]
    fn new(py: Python<'_>, base_url: &str) -> PyResult<(Self, BaseTensorZeroGateway)> {
        Ok((Self {}, BaseTensorZeroGateway::new(py, base_url)?))
    }

    fn __enter__(this: Py<Self>) -> Py<Self> {
        this
    }

    // TODO - implement closing the 'reqwest' connection pool
    fn __exit__(
        _this: Py<Self>,
        _exc_type: Py<PyAny>,
        _exc_value: Py<PyAny>,
        _traceback: Py<PyAny>,
    ) -> PyResult<()> {
        Ok(())
    }

    #[classmethod]
    #[pyo3(signature = (*, config_path, clickhouse_url=None))]
    fn create_embedded_gateway(
        cls: &Bound<'_, PyType>,
        config_path: &str,
        clickhouse_url: Option<String>,
    ) -> PyResult<Py<TensorZeroGateway>> {
        let client_fut = ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
            config_path: PathBuf::from(config_path),
            clickhouse_url,
        })
        .build();
        pyo3_async_runtimes::tokio::get_runtime().block_on(async move {
            let client = match client_fut.await {
                Ok(client) => client,
                Err(e) => {
                    return Err(tensorzero_internal_error(
                        cls.py(),
                        &format!("Failed to construct TensorZero client: {e:?}"),
                    )?);
                }
            };
            let instance = PyClassInitializer::from(BaseTensorZeroGateway {
                client: Arc::new(client),
            })
            .add_subclass(TensorZeroGateway {});
            Py::new(cls.py(), instance)
        })
    }

    #[pyo3(signature = (*, metric_name, value, inference_id=None, episode_id=None, dryrun=None, tags=None))]
    #[allow(clippy::too_many_arguments)]
    fn feedback(
        this: PyRef<'_, Self>,
        py: Python<'_>,
        metric_name: String,
        value: Bound<'_, PyAny>,
        inference_id: Option<Bound<'_, PyAny>>,
        episode_id: Option<Bound<'_, PyAny>>,
        dryrun: Option<bool>,
        tags: Option<HashMap<String, String>>,
    ) -> PyResult<Py<PyAny>> {
        let fut = this
            .as_super()
            .client
            .feedback(BaseTensorZeroGateway::prepare_feedback_params(
                py,
                metric_name,
                value,
                inference_id,
                episode_id,
                dryrun,
                tags,
            )?);
        match pyo3_async_runtimes::tokio::get_runtime().block_on(fut) {
            Ok(resp) => Ok(parse_feedback_response(py, resp)?.into_any()),
            Err(e) => Err(convert_error(py, e)?),
        }
    }

    #[pyo3(signature = (*, function_name, input, episode_id=None, stream=None, params=None, variant_name=None, dryrun=None, allowed_tools=None, additional_tools=None, tool_choice=None, parallel_tool_calls=None, tags=None, credentials=None))]
    #[allow(clippy::too_many_arguments)]
    fn inference(
        this: PyRef<'_, Self>,
        py: Python<'_>,
        function_name: String,
        input: Bound<'_, PyDict>,
        episode_id: Option<Bound<'_, PyAny>>,
        stream: Option<bool>,
        params: Option<&Bound<'_, PyDict>>,
        variant_name: Option<String>,
        dryrun: Option<bool>,
        allowed_tools: Option<Vec<String>>,
        additional_tools: Option<Vec<HashMap<String, Bound<'_, PyAny>>>>,
        tool_choice: Option<Bound<'_, PyAny>>,
        parallel_tool_calls: Option<bool>,
        tags: Option<HashMap<String, String>>,
        credentials: Option<HashMap<String, ClientSecretString>>,
    ) -> PyResult<Py<PyAny>> {
        let fut =
            this.as_super()
                .client
                .inference(BaseTensorZeroGateway::prepare_inference_params(
                    py,
                    function_name,
                    input,
                    episode_id,
                    stream,
                    params,
                    variant_name,
                    dryrun,
                    allowed_tools,
                    additional_tools,
                    tool_choice,
                    parallel_tool_calls,
                    tags,
                    credentials,
                )?);

        let resp = pyo3_async_runtimes::tokio::get_runtime().block_on(fut);
        match resp {
            Ok(InferenceOutput::NonStreaming(data)) => parse_inference_response(py, data),
            Ok(InferenceOutput::Streaming(stream)) => Ok(StreamWrapper {
                stream: Arc::new(Mutex::new(stream)),
            }
            .into_pyobject(py)?
            .into_any()
            .unbind()),
            Err(e) => Err(convert_error(py, e)?),
        }
    }
}

#[pyclass(extends=BaseTensorZeroGateway)]
struct AsyncTensorZeroGateway {}

#[pymethods]
impl AsyncTensorZeroGateway {
    #[new]
    fn new(py: Python<'_>, base_url: &str) -> PyResult<(Self, BaseTensorZeroGateway)> {
        Ok((Self {}, BaseTensorZeroGateway::new(py, base_url)?))
    }

    async fn __aenter__(this: Py<Self>) -> Py<Self> {
        this
    }

    // TODO - implement closing the 'reqwest' connection pool
    async fn __aexit__(
        this: Py<Self>,
        _exc_type: Py<PyAny>,
        _exc_value: Py<PyAny>,
        _traceback: Py<PyAny>,
    ) -> Py<Self> {
        this
    }

    // We make this a class method rather than adding parameters to the `__init__` method,
    // becaues this needs to return a python `Future` (since we need to connect to ClickHouse
    // and run DB migrations.
    //
    // While we could block in the `__init__` method, this would be very suprising to consumers,
    // as `AsyncTensorZeroGateway` would be completely async *except* for this one method
    // (which potentially takes a very long time due to running DB migrations).
    #[classmethod]
    #[pyo3(signature = (*, config_path, clickhouse_url=None))]
    fn create_embedded_gateway<'a>(
        cls: &Bound<'a, PyType>,
        config_path: &str,
        clickhouse_url: Option<String>,
    ) -> PyResult<Bound<'a, PyAny>> {
        let client_fut = ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
            config_path: PathBuf::from(config_path),
            clickhouse_url,
        })
        .build();

        // See `AsyncStreamWrapper::__anext__` for more details about `future_into_py`
        pyo3_async_runtimes::tokio::future_into_py(cls.py(), async move {
            let client = client_fut.await;
            Python::with_gil(|py| {
                let client = match client {
                    Ok(client) => client,
                    Err(e) => {
                        return Err(tensorzero_internal_error(
                            py,
                            &format!("Failed to construct TensorZero client: {e:?}"),
                        )?);
                    }
                };
                let instance = PyClassInitializer::from(BaseTensorZeroGateway {
                    client: Arc::new(client),
                })
                .add_subclass(AsyncTensorZeroGateway {});
                Py::new(py, instance)
            })
        })
    }

    #[pyo3(signature = (*, function_name, input, episode_id=None, stream=None, params=None, variant_name=None, dryrun=None, allowed_tools=None, additional_tools=None, tool_choice=None, parallel_tool_calls=None, tags=None, credentials=None))]
    #[allow(clippy::too_many_arguments)]
    fn inference<'a>(
        this: PyRef<'_, Self>,
        py: Python<'a>,
        function_name: String,
        input: Bound<'_, PyDict>,
        episode_id: Option<Bound<'_, PyAny>>,
        stream: Option<bool>,
        params: Option<&Bound<'_, PyDict>>,
        variant_name: Option<String>,
        dryrun: Option<bool>,
        allowed_tools: Option<Vec<String>>,
        additional_tools: Option<Vec<HashMap<String, Bound<'_, PyAny>>>>,
        tool_choice: Option<Bound<'_, PyAny>>,
        parallel_tool_calls: Option<bool>,
        tags: Option<HashMap<String, String>>,
        credentials: Option<HashMap<String, ClientSecretString>>,
    ) -> PyResult<Bound<'a, PyAny>> {
        let params = BaseTensorZeroGateway::prepare_inference_params(
            py,
            function_name,
            input,
            episode_id,
            stream,
            params,
            variant_name,
            dryrun,
            allowed_tools,
            additional_tools,
            tool_choice,
            parallel_tool_calls,
            tags,
            credentials,
        )?;
        let client = this.as_super().client.clone();
        // See `AsyncStreamWrapper::__anext__` for more details about `future_into_py`
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let res = client.inference(params).await;
            Python::with_gil(|py| match res {
                Ok(InferenceOutput::NonStreaming(data)) => parse_inference_response(py, data),
                Ok(InferenceOutput::Streaming(stream)) => Ok(AsyncStreamWrapper {
                    stream: Arc::new(Mutex::new(stream)),
                }
                .into_pyobject(py)?
                .into_any()
                .unbind()),
                Err(e) => Err(convert_error(py, e)?),
            })
        })
    }

    #[pyo3(signature = (*, metric_name, value, inference_id=None, episode_id=None, dryrun=None, tags=None))]
    fn feedback<'a>(
        this: PyRef<'a, Self>,
        metric_name: String,
        value: Bound<'_, PyAny>,
        inference_id: Option<Bound<'_, PyAny>>,
        episode_id: Option<Bound<'_, PyAny>>,
        dryrun: Option<bool>,
        tags: Option<HashMap<String, String>>,
    ) -> PyResult<Bound<'a, PyAny>> {
        let client = this.as_super().client.clone();
        let params = BaseTensorZeroGateway::prepare_feedback_params(
            this.py(),
            metric_name,
            value,
            inference_id,
            episode_id,
            dryrun,
            tags,
        )?;
        // See `AsyncStreamWrapper::__anext__` for more details about `future_into_py`
        pyo3_async_runtimes::tokio::future_into_py(this.py(), async move {
            let res = client.feedback(params).await;
            Python::with_gil(|py| match res {
                Ok(resp) => Ok(parse_feedback_response(py, resp)?.into_any()),
                Err(e) => Err(convert_error(py, e)?),
            })
        })
    }
}

#[allow(unknown_lints)]
// This lint currently does nothing on stable, but let's include it
// so that it will start working automatically when it's stabilized
#[deny(non_exhaustive_omitted_patterns)]
fn convert_error(py: Python<'_>, e: TensorZeroError) -> PyResult<PyErr> {
    match e {
        TensorZeroError::Http {
            status_code,
            text,
            source: _,
        } => tensorzero_error(py, status_code, text),
        TensorZeroError::Other { source } => tensorzero_internal_error(py, &source.to_string()),
        // Required due to the `#[non_exhaustive]` attribute on `TensorZeroError` - we want to force
        // downstream consumers to handle all possible error types, but the compiler also requires us
        // to do this (since our python bindings are in a different crates from the Rust client.)
        _ => unreachable!(),
    }
}

fn tensorzero_error(py: Python<'_>, status_code: u16, text: Option<String>) -> PyResult<PyErr> {
    let err = TENSORZERO_HTTP_ERROR.get_or_try_init::<_, PyErr>(py, || {
        let self_module = PyModule::import(py, "tensorzero")?;
        let err: Bound<'_, PyAny> = self_module.getattr("TensorZeroError")?;
        Ok(err.unbind())
    })?;
    Ok(PyErr::from_value(err.bind(py).call1((status_code, text))?))
}

fn tensorzero_internal_error(py: Python<'_>, msg: &str) -> PyResult<PyErr> {
    let err = TENSORZERO_INTERNAL_ERROR.get_or_try_init::<_, PyErr>(py, || {
        let self_module = PyModule::import(py, "tensorzero")?;
        let err: Bound<'_, PyAny> = self_module.getattr("TensorZeroInternalError")?;
        Ok(err.unbind())
    })?;
    Ok(PyErr::from_value(err.bind(py).call1((msg,))?))
}

pub(crate) static JSON_LOADS: GILOnceCell<Py<PyAny>> = GILOnceCell::new();
pub(crate) static JSON_DUMPS: GILOnceCell<Py<PyAny>> = GILOnceCell::new();
pub(crate) static TENSORZERO_HTTP_ERROR: GILOnceCell<Py<PyAny>> = GILOnceCell::new();
pub(crate) static TENSORZERO_INTERNAL_ERROR: GILOnceCell<Py<PyAny>> = GILOnceCell::new();

#[pymodule]
fn tensorzero(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BaseTensorZeroGateway>()?;
    m.add_class::<AsyncTensorZeroGateway>()?;
    m.add_class::<TensorZeroGateway>()?;

    let py_json = PyModule::import(m.py(), "json")?;
    let json_loads = py_json.getattr("loads")?;
    let json_dumps = py_json.getattr("dumps")?;
    JSON_LOADS
        .set(m.py(), json_loads.unbind())
        .expect("Failed to set JSON_LOADS");
    JSON_DUMPS
        .set(m.py(), json_dumps.unbind())
        .expect("Failed to set JSON_DUMPS");
    Ok(())
}
