/// Implements a Python tensorzero client, using `pyo3` to wrap the existing Rust client.
/// Overall structure of the crate:
/// * `src/lib.rs` - the main entrypoint of the Python native module - the `#[pymodule]` function
///   initializes the Python module.
/// * `tensorzero/` - this contains the Python code for the overall `tensorzero` package.
///   This re-exports types from the Rust native module, and also defines several pure-Python
///   classes/functions used by the Rust code.
///
/// This module defines several Python classes (`BaseTensorZeroGateway`, `TensorZeroGateway`, `AsyncTensorZeroGateway`),
/// and defines methods on them.
use std::{collections::HashMap, future::Future, path::PathBuf, sync::Arc, time::Duration};

use futures::StreamExt;
use pyo3::{
    exceptions::{PyStopAsyncIteration, PyStopIteration, PyValueError},
    ffi::c_str,
    marker::Ungil,
    prelude::*,
    sync::GILOnceCell,
    types::{PyDict, PyList, PyString, PyType},
    IntoPyObjectExt,
};
use python_helpers::{
    deserialize_from_pyobj, parse_feedback_response, parse_inference_chunk,
    parse_inference_response, parse_tool, python_uuid_to_uuid, serialize_to_dict,
};
use tensorzero_internal::{
    gateway_util::ShutdownHandle, inference::types::extra_body::UnfilteredInferenceExtraBody,
};
use tensorzero_rust::{
    err_to_http, observability::LogFormat, CacheParamsOptions, Client, ClientBuilder,
    ClientBuilderMode, ClientInferenceParams, ClientInput, ClientSecretString, DynamicToolParams,
    FeedbackParams, InferenceOutput, InferenceParams, InferenceStream, TensorZeroError, Tool,
};
use tokio::sync::Mutex;
use url::Url;

mod python_helpers;

pub(crate) static JSON_LOADS: GILOnceCell<Py<PyAny>> = GILOnceCell::new();
pub(crate) static JSON_DUMPS: GILOnceCell<Py<PyAny>> = GILOnceCell::new();
pub(crate) static TENSORZERO_HTTP_ERROR: GILOnceCell<Py<PyAny>> = GILOnceCell::new();
pub(crate) static TENSORZERO_INTERNAL_ERROR: GILOnceCell<Py<PyAny>> = GILOnceCell::new();

#[pymodule]
fn tensorzero(m: &Bound<'_, PyModule>) -> PyResult<()> {
    tensorzero_rust::observability::setup_logs(false, LogFormat::Pretty);
    m.add_class::<BaseTensorZeroGateway>()?;
    m.add_class::<AsyncTensorZeroGateway>()?;
    m.add_class::<TensorZeroGateway>()?;
    m.add_class::<LocalHttpGateway>()?;

    let py_json = PyModule::import(m.py(), "json")?;
    let json_loads = py_json.getattr("loads")?;
    let json_dumps = py_json.getattr("dumps")?;
    JSON_LOADS
        .set(m.py(), json_loads.unbind())
        .expect("Failed to set JSON_LOADS");
    JSON_DUMPS
        .set(m.py(), json_dumps.unbind())
        .expect("Failed to set JSON_DUMPS");

    m.add_wrapped(wrap_pyfunction!(_start_http_gateway))
        .expect("Failed to add start_http_gateway");

    Ok(())
}

#[pyclass]
struct LocalHttpGateway {
    #[pyo3(get)]
    base_url: String,
    #[allow(dead_code)]
    shutdown_handle: Option<ShutdownHandle>,
}

#[pymethods]
impl LocalHttpGateway {
    fn close(&mut self) {
        self.shutdown_handle.take();
    }
}

#[pyfunction]
#[pyo3(signature = (*, config_file, clickhouse_url, async_setup))]
fn _start_http_gateway(
    py: Python<'_>,
    config_file: Option<String>,
    clickhouse_url: Option<String>,
    async_setup: bool,
) -> PyResult<Bound<'_, PyAny>> {
    warn_no_config(py, config_file.as_deref())?;
    let gateway_fut = async move {
        let (addr, handle) = tensorzero_internal::gateway_util::start_openai_compatible_gateway(
            config_file,
            clickhouse_url,
        )
        .await?;
        Ok(LocalHttpGateway {
            base_url: format!("http://{}/openai/v1", addr),
            shutdown_handle: Some(handle),
        })
    };
    if async_setup {
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            gateway_fut.await.map_err(|e| {
                Python::with_gil(|py| convert_error(py, TensorZeroError::Other { source: e }))
            })
        })
    } else {
        Ok(tokio_block_on_without_gil(py, gateway_fut)
            .map_err(|e| convert_error(py, TensorZeroError::Other { source: e }))?
            .into_bound_py_any(py)?)
    }
}

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
            // the `py` parameter from `__anext__`.
            // We need to interact with Python objects here (to build up a Python `InferenceChunk`),
            // so we need the GIL
            Python::with_gil(|py| {
                let chunk = chunk.map_err(|e| convert_error(py, err_to_http(e)))?;
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
        // The `__next__` method is blocking (`StreamWrapper` comes from the synchronous `TensorZeroGateway`),
        // so we need to block on the Rust future. All of the `.await` calls are confined to the `async move` block,
        let chunk = tokio_block_on_without_gil(py, async move { stream.lock().await.next().await });
        let Some(chunk) = chunk else {
            return Err(PyStopIteration::new_err(()));
        };
        let chunk = chunk.map_err(|e| convert_error(py, err_to_http(e)))?;
        parse_inference_chunk(py, chunk)
    }
}

#[pymethods]
impl BaseTensorZeroGateway {
    #[new]
    #[pyo3(signature = (base_url, *, timeout=None, verbose_errors=false))]
    fn new(
        py: Python<'_>,
        base_url: &str,
        timeout: Option<f64>,
        verbose_errors: bool,
    ) -> PyResult<Self> {
        let mut client_builder = ClientBuilder::new(ClientBuilderMode::HTTPGateway {
            url: Url::parse(base_url)
                .map_err(|e| PyValueError::new_err(format!("Failed to parse base_url: {e:?}")))?,
        })
        .with_verbose_errors(verbose_errors);
        if let Some(timeout) = timeout {
            let http_client = reqwest::Client::builder()
                .timeout(
                    Duration::try_from_secs_f64(timeout)
                        .map_err(|e| PyValueError::new_err(format!("Invalid timeout: {e}")))?,
                )
                .build()
                .map_err(|e| {
                    PyValueError::new_err(format!("Failed to build HTTP client: {e:?}"))
                })?;
            client_builder = client_builder.with_http_client(http_client);
        }
        let client = match client_builder.build_http() {
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

    #[pyo3(signature = (*, input, function_name=None, model_name=None, episode_id=None, stream=None, params=None, variant_name=None, dryrun=None, output_schema=None, allowed_tools=None, additional_tools=None, tool_choice=None, parallel_tool_calls=None, internal=None, tags=None, credentials=None, cache_options=None, extra_body=None))]
    #[allow(clippy::too_many_arguments)]
    fn _prepare_inference_request(
        this: PyRef<'_, Self>,
        input: Bound<'_, PyDict>,
        function_name: Option<String>,
        model_name: Option<String>,
        episode_id: Option<Bound<'_, PyAny>>,
        stream: Option<bool>,
        params: Option<&Bound<'_, PyDict>>,
        variant_name: Option<String>,
        dryrun: Option<bool>,
        output_schema: Option<&Bound<'_, PyDict>>,
        allowed_tools: Option<Vec<String>>,
        additional_tools: Option<Vec<HashMap<String, Bound<'_, PyAny>>>>,
        tool_choice: Option<Bound<'_, PyAny>>,
        parallel_tool_calls: Option<bool>,
        internal: Option<bool>,
        tags: Option<HashMap<String, String>>,
        credentials: Option<HashMap<String, ClientSecretString>>,
        cache_options: Option<&Bound<'_, PyDict>>,
        extra_body: Option<&Bound<'_, PyList>>,
    ) -> PyResult<Py<PyAny>> {
        let params = BaseTensorZeroGateway::prepare_inference_params(
            this.py(),
            input,
            function_name,
            model_name,
            episode_id,
            stream,
            params,
            variant_name,
            dryrun,
            output_schema,
            allowed_tools,
            additional_tools,
            tool_choice,
            parallel_tool_calls,
            internal.unwrap_or(false),
            tags,
            credentials,
            cache_options,
            extra_body,
        )?;
        serialize_to_dict(this.py(), params)
    }
}

#[pyclass(extends=BaseTensorZeroGateway)]
/// A synchronous client for a TensorZero gateway.
///
/// To connect to a running HTTP gateway, call `TensorZeroGateway.build_http(base_url = "http://gateway_url")`
/// To create an embedded gateway, call `TensorZeroGateway.build_embedded(config_file = "/path/to/tensorzero.toml", clickhouse_url = "http://clickhouse_url")`
struct TensorZeroGateway {}

/// Calls `tokio::Runtime::block_on` without holding the Python GIL.
/// This is used when we call into pure-Rust code from the synchronous `TensorZeroGateway`
/// We don't need (or want) to hold the GIL when the Rust client code is running,
/// since it doesn't need to interact with any Python objects.
/// This allows other Python threads to run while the current thread is blocked on the Rust execution.
fn tokio_block_on_without_gil<F: Future + Send>(py: Python<'_>, fut: F) -> F::Output
where
    F::Output: Ungil,
{
    // The Tokio runtime is managed by `pyo3_async_runtimes` - the entrypoint to
    // our crate (`python-pyo3`) is the `pymodule` function, rather than
    // a `#[tokio::main]` function, so we need `pyo3_async_runtimes` to keep track of
    // a Tokio runtime for us.
    py.allow_threads(|| pyo3_async_runtimes::tokio::get_runtime().block_on(fut))
}

impl BaseTensorZeroGateway {
    #[allow(clippy::too_many_arguments)]
    fn prepare_feedback_params(
        py: Python<'_>,
        metric_name: String,
        value: Bound<'_, PyAny>,
        inference_id: Option<Bound<'_, PyAny>>,
        episode_id: Option<Bound<'_, PyAny>>,
        dryrun: Option<bool>,
        internal: bool,
        tags: Option<HashMap<String, String>>,
    ) -> PyResult<FeedbackParams> {
        Ok(FeedbackParams {
            metric_name,
            value: deserialize_from_pyobj(py, &value)?,
            episode_id: python_uuid_to_uuid("episode_id", episode_id)?,
            inference_id: python_uuid_to_uuid("inference_id", inference_id)?,
            dryrun,
            tags: tags.unwrap_or_default(),
            internal,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_inference_params(
        py: Python<'_>,
        input: Bound<'_, PyDict>,
        function_name: Option<String>,
        model_name: Option<String>,
        episode_id: Option<Bound<'_, PyAny>>,
        stream: Option<bool>,
        params: Option<&Bound<'_, PyDict>>,
        variant_name: Option<String>,
        dryrun: Option<bool>,
        output_schema: Option<&Bound<'_, PyDict>>,
        allowed_tools: Option<Vec<String>>,
        additional_tools: Option<Vec<HashMap<String, Bound<'_, PyAny>>>>,
        tool_choice: Option<Bound<'_, PyAny>>,
        parallel_tool_calls: Option<bool>,
        internal: bool,
        tags: Option<HashMap<String, String>>,
        credentials: Option<HashMap<String, ClientSecretString>>,
        cache_options: Option<&Bound<'_, PyDict>>,
        extra_body: Option<&Bound<'_, PyList>>,
    ) -> PyResult<ClientInferenceParams> {
        let episode_id = python_uuid_to_uuid("episode_id", episode_id)?;

        let params: Option<InferenceParams> = if let Some(params) = params {
            deserialize_from_pyobj(py, params)?
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
                    serde_json::from_value(tool_choice.str()?.to_cow()?.into()).map_err(|e| {
                        PyValueError::new_err(format!(
                            "Failed to parse tool_choice as valid JSON: {e:?}"
                        ))
                    })?,
                )
            } else {
                Some(deserialize_from_pyobj(py, &tool_choice)?)
            }
        } else {
            None
        };

        let cache_options: Option<CacheParamsOptions> = if let Some(cache_options) = cache_options {
            Some(deserialize_from_pyobj(py, cache_options)?)
        } else {
            None
        };
        let output_schema: Option<serde_json::Value> = if let Some(output_schema) = output_schema {
            Some(deserialize_from_pyobj(py, output_schema)?)
        } else {
            None
        };

        let extra_body: UnfilteredInferenceExtraBody = if let Some(extra_body) = extra_body {
            deserialize_from_pyobj(py, extra_body)?
        } else {
            Default::default()
        };

        let input: ClientInput = deserialize_from_pyobj(py, &input)?;

        Ok(ClientInferenceParams {
            function_name,
            model_name,
            stream,
            episode_id,
            variant_name,
            dryrun,
            tags: tags.unwrap_or_default(),
            internal,
            params: params.unwrap_or_default(),
            dynamic_tool_params: DynamicToolParams {
                allowed_tools,
                parallel_tool_calls,
                additional_tools,
                tool_choice,
            },
            input,
            credentials: credentials.unwrap_or_default(),
            cache_options: cache_options.unwrap_or_default(),
            output_schema,
            // This is currently unsupported in the Python client
            include_original_response: false,
            extra_body,
        })
    }
}
#[pymethods]
impl TensorZeroGateway {
    #[new]
    #[pyo3(signature = (base_url, *, timeout=None))]
    fn new(
        py: Python<'_>,
        base_url: &str,
        timeout: Option<f64>,
    ) -> PyResult<(Self, BaseTensorZeroGateway)> {
        tracing::warn!("TensorZeroGateway.__init__ is deprecated. Use TensorZeroGateway.build_http or TensorZeroGateway.build_embedded instead.");
        Ok((
            Self {},
            BaseTensorZeroGateway::new(py, base_url, timeout, false)?,
        ))
    }

    #[classmethod]
    #[pyo3(signature = (*, gateway_url, timeout=None, verbose_errors=false))]
    /// Initialize the TensorZero client, using the HTTP gateway.
    /// :param gateway_url: The base URL of the TensorZero gateway. Example: "http://localhost:3000"
    /// :param timeout: The timeout for the HTTP client in seconds. If not provided, no timeout will be set.
    /// :param verbose_errors: If true, the client will increase the detail in errors (increasing the risk of leaking sensitive information).
    /// :return: A `TensorZeroGateway` instance configured to use the HTTP gateway.
    fn build_http(
        cls: &Bound<'_, PyType>,
        gateway_url: &str,
        timeout: Option<f64>,
        verbose_errors: bool,
    ) -> PyResult<Py<TensorZeroGateway>> {
        let mut client_builder = ClientBuilder::new(ClientBuilderMode::HTTPGateway {
            url: Url::parse(gateway_url)
                .map_err(|e| PyValueError::new_err(format!("Invalid gateway URL: {e}")))?,
        })
        .with_verbose_errors(verbose_errors);
        if let Some(timeout) = timeout {
            let http_client = reqwest::Client::builder()
                .timeout(
                    Duration::try_from_secs_f64(timeout)
                        .map_err(|e| PyValueError::new_err(format!("Invalid timeout: {e}")))?,
                )
                .build()
                .map_err(|e| PyValueError::new_err(format!("Failed to build HTTP client: {e}")))?;
            client_builder = client_builder.with_http_client(http_client);
        }
        let client_fut = client_builder.build();
        let client_res = tokio_block_on_without_gil(cls.py(), client_fut);
        let client = match client_res {
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
    }

    /// **Deprecated** (use `build_http` or `build_embedded` instead)
    /// Initialize the TensorZero client.
    ///
    /// :param base_url: The base URL of the TensorZero gateway. Example: "http://localhost:3000"
    /// :param timeout: The timeout for the HTTP client in seconds. If not provided, no timeout will be set.
    #[allow(unused_variables)]
    #[pyo3(signature = (base_url, *, timeout=None))]
    fn __init__(this: Py<Self>, base_url: &str, timeout: Option<f64>) -> Py<Self> {
        // The actual logic is in the 'new' method - this method just exists to generate a docstring
        this
    }

    /// Close the connection to the TensorZero gateway.
    fn close(&self) {
        // TODO - implement closing the 'reqwest' connection pool: https://github.com/tensorzero/tensorzero/issues/857
    }

    fn __enter__(this: Py<Self>) -> Py<Self> {
        this
    }

    // TODO - implement closing the 'reqwest' connection pool: https://github.com/tensorzero/tensorzero/issues/857
    fn __exit__(
        _this: Py<Self>,
        _exc_type: Py<PyAny>,
        _exc_value: Py<PyAny>,
        _traceback: Py<PyAny>,
    ) -> PyResult<()> {
        Ok(())
    }

    #[classmethod]
    #[pyo3(signature = (*, config_file=None, clickhouse_url=None, timeout=None))]
    /// Initialize the TensorZero client, using an embedded gateway.
    /// This connects to ClickHouse (if provided) and runs DB migrations.
    ///
    /// :param config_file: The path to the TensorZero configuration file. Example: "tensorzero.toml"
    /// :param clickhouse_url: The URL of the ClickHouse instance to use for the gateway. If observability is disabled in the config, this can be `None`
    /// :param timeout: The timeout for embedded gateway request processing, in seconds. If this timeout is hit, any in-progress LLM requests may be aborted. If not provided, no timeout will be set.
    /// :return: A `TensorZeroGateway` instance configured to use an embedded gateway.
    fn build_embedded(
        cls: &Bound<'_, PyType>,
        config_file: Option<&str>,
        clickhouse_url: Option<String>,
        timeout: Option<f64>,
    ) -> PyResult<Py<TensorZeroGateway>> {
        warn_no_config(cls.py(), config_file)?;
        let timeout = timeout
            .map(Duration::try_from_secs_f64)
            .transpose()
            .map_err(|e| PyValueError::new_err(format!("Invalid timeout: {e}")))?;
        let client_fut = ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
            config_file: config_file.map(PathBuf::from),
            clickhouse_url,
            timeout,
        })
        .build();
        let client = tokio_block_on_without_gil(cls.py(), client_fut);
        let client = match client {
            Ok(client) => client,
            Err(e) => {
                return Err(tensorzero_internal_error(
                    cls.py(),
                    &format!("Failed to construct TensorZero client: {e:?}"),
                )?);
            }
        };
        // Construct an instance of `TensorZeroGateway` (while providing the fields from the `BaseTensorZeroGateway` superclass).
        let instance = PyClassInitializer::from(BaseTensorZeroGateway {
            client: Arc::new(client),
        })
        .add_subclass(TensorZeroGateway {});
        Py::new(cls.py(), instance)
    }

    #[pyo3(signature = (*, metric_name, value, inference_id=None, episode_id=None, dryrun=None, internal=None, tags=None))]
    /// Make a request to the /feedback endpoint of the gateway
    ///
    /// :param metric_name: The name of the metric to provide feedback for
    /// :param value: The value of the feedback. It should correspond to the metric type.
    /// :param inference_id: The inference ID to assign the feedback to.
    ///                      Only use inference IDs that were returned by the TensorZero gateway.
    ///                      Note: You can assign feedback to either an episode or an inference, but not both.
    /// :param episode_id: The episode ID to use for the request
    ///                    Only use episode IDs that were returned by the TensorZero gateway.
    ///                    Note: You can assign feedback to either an episode or an inference, but not both.
    /// :param dryrun: If true, the feedback request will be executed but won't be stored to the database (i.e. no-op).
    /// :param tags: If set, adds tags to the feedback request.
    /// :return: {"feedback_id": str}
    #[allow(clippy::too_many_arguments)]
    fn feedback(
        this: PyRef<'_, Self>,
        py: Python<'_>,
        metric_name: String,
        value: Bound<'_, PyAny>,
        inference_id: Option<Bound<'_, PyAny>>,
        episode_id: Option<Bound<'_, PyAny>>,
        dryrun: Option<bool>,
        internal: Option<bool>,
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
                internal.unwrap_or(false),
                tags,
            )?);
        // We're in the synchronous `TensorZeroGateway` class, so we need to block on the Rust future,
        // and then return the result to the Python caller directly (not wrapped in a Python `Future`).
        match tokio_block_on_without_gil(py, fut) {
            Ok(resp) => Ok(parse_feedback_response(py, resp)?.into_any()),
            Err(e) => Err(convert_error(py, e)),
        }
    }

    #[pyo3(signature = (*, input, function_name=None, model_name=None, episode_id=None, stream=None, params=None, variant_name=None, dryrun=None, output_schema=None, allowed_tools=None, additional_tools=None, tool_choice=None, parallel_tool_calls=None, internal=None, tags=None, credentials=None, cache_options=None, extra_body=None))]
    #[allow(clippy::too_many_arguments)]
    /// Make a request to the /inference endpoint.
    ///
    /// :param function_name: The name of the function to call
    /// :param input: The input to the function
    ///               Structure: {"system": Optional[str], "messages": List[{"role": "user" | "assistant", "content": Any}]}
    ///               The input will be validated server side against the input schema of the function being called.
    /// :param episode_id: The episode ID to use for the inference.
    ///                    If this is the first inference in an episode, leave this field blank. The TensorZero gateway will generate and return a new episode ID.
    ///                    Note: Only use episode IDs generated by the TensorZero gateway. Don't generate them yourself.
    /// :param stream: If set, the TensorZero gateway will stream partial message deltas (e.g. generated tokens) as it receives them from model providers.
    /// :param params: Override inference-time parameters for a particular variant type. Currently, we support:
    ///                 {"chat_completion": {"temperature": float, "max_tokens": int, "seed": int}}
    /// :param variant_name: If set, pins the inference request to a particular variant.
    ///                      Note: You should generally not do this, and instead let the TensorZero gateway assign a
    ///                      particular variant. This field is primarily used for testing or debugging purposes.
    /// :param dryrun: If true, the request will be executed but won't be stored to the database.
    /// :param output_schema: If set, the JSON schema of a JSON function call will be validated against the given JSON Schema.
    ///                       Overrides the output schema configured for the function.
    /// :param allowed_tools: If set, restricts the tools available during this inference request.
    ///                       The list of names should be a subset of the tools configured for the function.
    ///                       Tools provided at inference time in `additional_tools` (if any) are always available.
    /// :param additional_tools: A list of additional tools to use for the request. Each element should look like {"name": str, "parameters": valid JSON Schema, "description": str}
    /// :param tool_choice: If set, overrides the tool choice strategy for the request.
    ///                     It should be one of: "auto", "required", "off", or {"specific": str}. The last option pins the request to a specific tool name.
    /// :param parallel_tool_calls: If true, the request will allow for multiple tool calls in a single inference request.
    /// :param tags: If set, adds tags to the inference request.
    /// :param cache_options: If set, overrides the cache options for the inference request.
    ///                      Structure: {"max_age_s": Optional[int], "enabled": "on" | "off" | "read_only" | "write_only"}
    /// :param extra_body: If set, injects extra fields into the provider request body.
    /// :return: If stream is false, returns an InferenceResponse.
    ///          If stream is true, returns a geerator that yields InferenceChunks as they come in.
    fn inference(
        this: PyRef<'_, Self>,
        py: Python<'_>,
        input: Bound<'_, PyDict>,
        function_name: Option<String>,
        model_name: Option<String>,
        episode_id: Option<Bound<'_, PyAny>>,
        stream: Option<bool>,
        params: Option<&Bound<'_, PyDict>>,
        variant_name: Option<String>,
        dryrun: Option<bool>,
        output_schema: Option<&Bound<'_, PyDict>>,
        allowed_tools: Option<Vec<String>>,
        additional_tools: Option<Vec<HashMap<String, Bound<'_, PyAny>>>>,
        tool_choice: Option<Bound<'_, PyAny>>,
        parallel_tool_calls: Option<bool>,
        internal: Option<bool>,
        tags: Option<HashMap<String, String>>,
        credentials: Option<HashMap<String, ClientSecretString>>,
        cache_options: Option<&Bound<'_, PyDict>>,
        extra_body: Option<&Bound<'_, PyList>>,
    ) -> PyResult<Py<PyAny>> {
        let fut =
            this.as_super()
                .client
                .inference(BaseTensorZeroGateway::prepare_inference_params(
                    py,
                    input,
                    function_name,
                    model_name,
                    episode_id,
                    stream,
                    params,
                    variant_name,
                    dryrun,
                    output_schema,
                    allowed_tools,
                    additional_tools,
                    tool_choice,
                    parallel_tool_calls,
                    internal.unwrap_or(false),
                    tags,
                    credentials,
                    cache_options,
                    extra_body,
                )?);

        // We're in the synchronous `TensorZeroGateway` class, so we need to block on the Rust future,
        // and then return the result to the Python caller directly (not wrapped in a Python `Future`).
        let resp = tokio_block_on_without_gil(py, fut).map_err(|e| convert_error(py, e))?;
        match resp {
            InferenceOutput::NonStreaming(data) => parse_inference_response(py, data),
            InferenceOutput::Streaming(stream) => Ok(StreamWrapper {
                stream: Arc::new(Mutex::new(stream)),
            }
            .into_pyobject(py)?
            .into_any()
            .unbind()),
        }
    }
}

#[pyclass(extends=BaseTensorZeroGateway)]
/// An async client for a TensorZero gateway.
///
/// To connect to a running HTTP gateway, call `AsyncTensorZeroGateway.build_http(gateway_url="http://gateway_url")`
/// To create an embedded gateway, call `AsyncTensorZeroGateway.build_embedded(config_file="/path/to/tensorzero.toml")`
struct AsyncTensorZeroGateway {}

#[pymethods]
impl AsyncTensorZeroGateway {
    #[new]
    #[pyo3(signature = (base_url, *, timeout=None))]
    fn new(
        py: Python<'_>,
        base_url: &str,
        timeout: Option<f64>,
    ) -> PyResult<(Self, BaseTensorZeroGateway)> {
        tracing::warn!("AsyncTensorZeroGateway.__init__ is deprecated. Use AsyncTensorZeroGateway.build_http or AsyncTensorZeroGateway.build_embedded instead.");
        Ok((
            Self {},
            BaseTensorZeroGateway::new(py, base_url, timeout, false)?,
        ))
    }

    #[classmethod]
    #[pyo3(signature = (*, gateway_url, timeout=None, verbose_errors=false, async_setup=true))]
    /// Initialize the TensorZero client, using the HTTP gateway.
    /// :param gateway_url: The base URL of the TensorZero gateway. Example: "http://localhost:3000"
    /// :param timeout: The timeout for the HTTP client in seconds. If not provided, no timeout will be set.
    /// :param verbose_errors: If true, the client will increase the detail in errors (increasing the risk of leaking sensitive information).
    /// :param async_setup: If true, this method will return a `Future` that resolves to an `AsyncTensorZeroGateway` instance. Otherwise, it will block and construct the `AsyncTensorZeroGateway`
    /// :return: An `AsyncTensorZeroGateway` instance configured to use the HTTP gateway.
    fn build_http(
        cls: &Bound<'_, PyType>,
        gateway_url: &str,
        timeout: Option<f64>,
        verbose_errors: bool,
        async_setup: bool,
    ) -> PyResult<Py<PyAny>> {
        let mut client_builder = ClientBuilder::new(ClientBuilderMode::HTTPGateway {
            url: Url::parse(gateway_url)
                .map_err(|e| PyValueError::new_err(format!("Invalid gateway URL: {e}")))?,
        })
        .with_verbose_errors(verbose_errors);
        if let Some(timeout) = timeout {
            let http_client = reqwest::Client::builder()
                .timeout(Duration::from_secs_f64(timeout))
                .build()
                .map_err(|e| PyValueError::new_err(format!("Failed to build HTTP client: {e}")))?;
            client_builder = client_builder.with_http_client(http_client);
        }
        let client_fut = client_builder.build();
        let build_gateway = async move {
            let client = client_fut.await;
            // We need to interact with Python objects here (to build up a Python `AsyncTensorZeroGateway`),
            // so we need the GIL
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

                // Construct an instance of `AsyncTensorZeroGateway` (while providing the fields from the `BaseTensorZeroGateway` superclass).
                let instance = PyClassInitializer::from(BaseTensorZeroGateway {
                    client: Arc::new(client),
                })
                .add_subclass(AsyncTensorZeroGateway {});
                Py::new(py, instance)
            })
        };
        if async_setup {
            Ok(pyo3_async_runtimes::tokio::future_into_py(cls.py(), build_gateway)?.unbind())
        } else {
            Ok(tokio_block_on_without_gil(cls.py(), build_gateway)?.into_any())
        }
    }

    /// **Deprecated** (use `build_http` or `build_embedded` instead)
    /// Initialize the TensorZero client.
    ///
    /// :param base_url: The base URL of the TensorZero gateway. Example: "http://localhost:3000"
    /// :param timeout: The timeout for the HTTP client in seconds. If not provided, no timeout will be set.
    #[allow(unused_variables)]
    #[pyo3(signature = (base_url, *, timeout=None))]
    fn __init__(this: Py<Self>, base_url: &str, timeout: Option<f64>) -> Py<Self> {
        // The actual logic is in the 'new' method - this method just exists to generate a docstring
        this
    }

    /// Close the connection to the TensorZero gateway.
    async fn close(&self) {
        // TODO - implement closing the 'reqwest' connection pool: https://github.com/tensorzero/tensorzero/issues/857
    }

    async fn __aenter__(this: Py<Self>) -> Py<Self> {
        this
    }

    async fn __aexit__(
        _this: Py<Self>,
        _exc_type: Py<PyAny>,
        _exc_value: Py<PyAny>,
        _traceback: Py<PyAny>,
    ) -> PyResult<()> {
        // TODO - implement closing the 'reqwest' connection pool: https://github.com/tensorzero/tensorzero/issues/857
        Ok(())
    }

    // We make this a class method rather than adding parameters to the `__init__` method,
    // becaues this needs to return a python `Future` (since we need to connect to ClickHouse
    // and run DB migrations.
    //
    // While we could block in the `__init__` method, this would be very suprising to consumers,
    // as `AsyncTensorZeroGateway` would be completely async *except* for this one method
    // (which potentially takes a very long time due to running DB migrations).
    #[classmethod]
    #[pyo3(signature = (*, config_file=None, clickhouse_url=None, timeout=None, async_setup=true))]
    /// Initialize the TensorZero client, using an embedded gateway.
    /// This connects to ClickHouse (if provided) and runs DB migrations.
    ///
    /// :param config_file: The path to the TensorZero configuration file. Example: "tensorzero.toml"
    /// :param clickhouse_url: The URL of the ClickHouse instance to use for the gateway. If observability is disabled in the config, this can be `None`
    /// :param timeout: The timeout for embedded gateway request processing, in seconds. If this timeout is hit, any in-progress LLM requests may be aborted. If not provided, no timeout will be set.
    /// :param async_setup: If true, this method will return a `Future` that resolves to an `AsyncTensorZeroGateway` instance. Otherwise, it will block and construct the `AsyncTensorZeroGateway`
    /// :return: A `Future` that resolves to an `AsyncTensorZeroGateway` instance configured to use an embedded gateway (or an `AsyncTensorZeroGateway` if `async_setup=False`).
    fn build_embedded(
        // This is a classmethod, so it receives the class object as a parameter.
        cls: &Bound<'_, PyType>,
        config_file: Option<&str>,
        clickhouse_url: Option<String>,
        timeout: Option<f64>,
        async_setup: bool,
    ) -> PyResult<Py<PyAny>> {
        warn_no_config(cls.py(), config_file)?;
        let timeout = timeout
            .map(Duration::try_from_secs_f64)
            .transpose()
            .map_err(|e| PyValueError::new_err(format!("Invalid timeout: {e}")))?;
        let client_fut = ClientBuilder::new(ClientBuilderMode::EmbeddedGateway {
            config_file: config_file.map(PathBuf::from),
            clickhouse_url,
            timeout,
        })
        .build();

        let fut = async move {
            let client = client_fut.await;
            // We need to interact with Python objects here (to build up a Python `AsyncTensorZeroGateway`),
            // so we need the GIL
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

                // Construct an instance of `AsyncTensorZeroGateway` (while providing the fields from the `BaseTensorZeroGateway` superclass).
                let instance = PyClassInitializer::from(BaseTensorZeroGateway {
                    client: Arc::new(client),
                })
                .add_subclass(AsyncTensorZeroGateway {});
                Py::new(py, instance)
            })
        };
        if async_setup {
            // See `AsyncStreamWrapper::__anext__` for more details about `future_into_py`
            Ok(pyo3_async_runtimes::tokio::future_into_py(cls.py(), fut)?.unbind())
        } else {
            // If the user doesn't want to use async, we block on the future here.
            // This is useful for testing, or for users who want to use the async client in a synchronous context.
            Ok(tokio_block_on_without_gil(cls.py(), fut)?.into_any())
        }
    }

    #[pyo3(signature = (*, input, function_name=None, model_name=None, episode_id=None, stream=None, params=None, variant_name=None, dryrun=None, output_schema=None, allowed_tools=None, additional_tools=None, tool_choice=None, parallel_tool_calls=None, internal=None,tags=None, credentials=None, cache_options=None, extra_body=None))]
    #[allow(clippy::too_many_arguments)]
    /// Make a request to the /inference endpoint.
    ///
    /// :param function_name: The name of the function to call
    /// :param input: The input to the function
    ///               Structure: {"system": Optional[str], "messages": List[{"role": "user" | "assistant", "content": Any}]}
    ///               The input will be validated server side against the input schema of the function being called.
    /// :param episode_id: The episode ID to use for the inference.
    ///                    If this is the first inference in an episode, leave this field blank. The TensorZero gateway will generate and return a new episode ID.
    ///                    Note: Only use episode IDs generated by the TensorZero gateway. Don't generate them yourself.
    /// :param stream: If set, the TensorZero gateway will stream partial message deltas (e.g. generated tokens) as it receives them from model providers.
    /// :param params: Override inference-time parameters for a particular variant type. Currently, we support:
    ///                 {"chat_completion": {"temperature": float, "max_tokens": int, "seed": int}}
    /// :param variant_name: If set, pins the inference request to a particular variant.
    ///                      Note: You should generally not do this, and instead let the TensorZero gateway assign a
    ///                      particular variant. This field is primarily used for testing or debugging purposes.
    /// :param dryrun: If true, the request will be executed but won't be stored to the database.
    /// :param output_schema: If set, the JSON schema of a JSON function call will be validated against the given JSON Schema.
    ///                       Overrides the output schema configured for the function.
    /// :param allowed_tools: If set, restricts the tools available during this inference request.
    ///                       The list of names should be a subset of the tools configured for the function.
    ///                       Tools provided at inference time in `additional_tools` (if any) are always available.
    /// :param additional_tools: A list of additional tools to use for the request. Each element should look like {"name": str, "parameters": valid JSON Schema, "description": str}
    /// :param tool_choice: If set, overrides the tool choice strategy for the request.
    ///                     It should be one of: "auto", "required", "off", or {"specific": str}. The last option pins the request to a specific tool name.
    /// :param parallel_tool_calls: If true, the request will allow for multiple tool calls in a single inference request.
    /// :param tags: If set, adds tags to the inference request.
    /// :param cache_options: If set, overrides the cache options for the inference request.
    ///                      Structure: {"max_age_s": Optional[int], "enabled": "on" | "off" | "read_only" | "write_only"}
    /// :param extra_body: If set, injects extra fields into the provider request body.
    /// :return: If stream is false, returns an InferenceResponse.
    ///          If stream is true, returns an async generator that yields InferenceChunks as they come in.
    fn inference<'a>(
        this: PyRef<'_, Self>,
        py: Python<'a>,
        input: Bound<'_, PyDict>,
        function_name: Option<String>,
        model_name: Option<String>,
        episode_id: Option<Bound<'_, PyAny>>,
        stream: Option<bool>,
        params: Option<&Bound<'_, PyDict>>,
        variant_name: Option<String>,
        dryrun: Option<bool>,
        output_schema: Option<&Bound<'_, PyDict>>,
        allowed_tools: Option<Vec<String>>,
        additional_tools: Option<Vec<HashMap<String, Bound<'_, PyAny>>>>,
        tool_choice: Option<Bound<'_, PyAny>>,
        parallel_tool_calls: Option<bool>,
        internal: Option<bool>,
        tags: Option<HashMap<String, String>>,
        credentials: Option<HashMap<String, ClientSecretString>>,
        cache_options: Option<&Bound<'_, PyDict>>,
        extra_body: Option<&Bound<'_, PyList>>,
    ) -> PyResult<Bound<'a, PyAny>> {
        let params = BaseTensorZeroGateway::prepare_inference_params(
            py,
            input,
            function_name,
            model_name,
            episode_id,
            stream,
            params,
            variant_name,
            dryrun,
            output_schema,
            allowed_tools,
            additional_tools,
            tool_choice,
            parallel_tool_calls,
            internal.unwrap_or(false),
            tags,
            credentials,
            cache_options,
            extra_body,
        )?;
        let client = this.as_super().client.clone();
        // See `AsyncStreamWrapper::__anext__` for more details about `future_into_py`
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let res = client.inference(params).await;
            // We need to interact with Python objects here (to build up a Python inference response),
            // so we need the GIL
            Python::with_gil(|py| {
                let output = res.map_err(|e| convert_error(py, e))?;
                match output {
                    InferenceOutput::NonStreaming(data) => parse_inference_response(py, data),
                    InferenceOutput::Streaming(stream) => Ok(AsyncStreamWrapper {
                        stream: Arc::new(Mutex::new(stream)),
                    }
                    .into_pyobject(py)?
                    .into_any()
                    .unbind()),
                }
            })
        })
    }

    #[pyo3(signature = (*, metric_name, value, inference_id=None, episode_id=None, dryrun=None, internal=None, tags=None))]
    /// Make a request to the /feedback endpoint.
    ///
    /// :param metric_name: The name of the metric to provide feedback for
    /// :param value: The value of the feedback. It should correspond to the metric type.
    /// :param inference_id: The inference ID to assign the feedback to.
    ///                      Only use inference IDs that were returned by the TensorZero gateway.
    ///                      Note: You can assign feedback to either an episode or an inference, but not both.
    /// :param episode_id: The episode ID to use for the request
    ///                    Only use episode IDs that were returned by the TensorZero gateway.
    ///                    Note: You can assign feedback to either an episode or an inference, but not both.
    /// :param dryrun: If true, the feedback request will be executed but won't be stored to the database (i.e. no-op).
    /// :param tags: If set, adds tags to the feedback request.
    /// :return: {"feedback_id": str}
    #[allow(clippy::too_many_arguments)]
    fn feedback<'a>(
        this: PyRef<'a, Self>,
        metric_name: String,
        value: Bound<'_, PyAny>,
        inference_id: Option<Bound<'_, PyAny>>,
        episode_id: Option<Bound<'_, PyAny>>,
        dryrun: Option<bool>,
        internal: Option<bool>,
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
            internal.unwrap_or(false),
            tags,
        )?;
        // See `AsyncStreamWrapper::__anext__` for more details about `future_into_py`
        pyo3_async_runtimes::tokio::future_into_py(this.py(), async move {
            let res = client.feedback(params).await;
            // We need to interact with Python objects here (to build up a Python feedback response),
            // so we need the GIL
            Python::with_gil(|py| match res {
                Ok(resp) => Ok(parse_feedback_response(py, resp)?.into_any()),
                Err(e) => Err(convert_error(py, e)),
            })
        })
    }
}

#[allow(unknown_lints)]
// This lint currently does nothing on stable, but let's include it
// so that it will start working automatically when it's stabilized
#[deny(non_exhaustive_omitted_patterns)]
fn convert_error(py: Python<'_>, e: TensorZeroError) -> PyErr {
    match e {
        TensorZeroError::Http {
            status_code,
            text,
            source: _,
        } => tensorzero_error(py, status_code, text).unwrap_or_else(|e| e),
        TensorZeroError::Other { source } => {
            tensorzero_internal_error(py, &source.to_string()).unwrap_or_else(|e| e)
        }
        TensorZeroError::RequestTimeout => {
            tensorzero_internal_error(py, &e.to_string()).unwrap_or_else(|e| e)
        }
        // Required due to the `#[non_exhaustive]` attribute on `TensorZeroError` - we want to force
        // downstream consumers to handle all possible error types, but the compiler also requires us
        // to do this (since our python bindings are in a different crate from the Rust client.)
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

fn warn_no_config(py: Python<'_>, config: Option<&str>) -> PyResult<()> {
    if config.is_none() {
        let user_warning = py.get_type::<pyo3::exceptions::PyUserWarning>();
        PyErr::warn(
            py,
            &user_warning,
            c_str!("No config file provided, so only default functions will be available. Use `config_file=\"path/to/tensorzero.toml\"` to specify a config file."), 0
        )?;
    }
    Ok(())
}
