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
use std::{collections::HashMap, path::PathBuf, sync::Arc, time::Duration};

use evaluations::{run_evaluation_core_streaming, EvaluationCoreArgs};
use futures::StreamExt;
use pyo3::{
    exceptions::{PyDeprecationWarning, PyStopAsyncIteration, PyStopIteration, PyValueError},
    ffi::c_str,
    prelude::*,
    types::{PyDict, PyList, PyString, PyType},
    IntoPyObjectExt,
};
use python_helpers::{
    parse_feedback_response, parse_inference_chunk, parse_inference_response, parse_tool,
    parse_workflow_evaluation_run_episode_response, parse_workflow_evaluation_run_response,
    python_uuid_to_uuid,
};
use tensorzero_core::{
    config::{ConfigPyClass, FunctionsConfigPyClass, UninitializedVariantInfo},
    db::clickhouse::query_builder::OrderBy,
    function::{FunctionConfigChatPyClass, FunctionConfigJsonPyClass, VariantsConfigPyClass},
    inference::types::{
        pyo3_helpers::{
            deserialize_from_pyobj, deserialize_from_rendered_sample,
            deserialize_from_stored_sample, deserialize_optimization_config, serialize_to_dict,
            tensorzero_core_error, tensorzero_core_error_class, tensorzero_error_class, JSON_DUMPS,
            JSON_LOADS,
        },
        ResolvedInput, ResolvedInputMessage,
    },
    optimization::{
        dicl::UninitializedDiclOptimizationConfig, fireworks_sft::UninitializedFireworksSFTConfig,
        gcp_vertex_gemini_sft::UninitializedGCPVertexGeminiSFTConfig,
        openai_rft::UninitializedOpenAIRFTConfig, openai_sft::UninitializedOpenAISFTConfig,
        together_sft::UninitializedTogetherSFTConfig, OptimizationJobInfoPyClass,
        OptimizationJobStatus, UninitializedOptimizerInfo,
    },
    tool::ProviderTool,
    variant::{
        BestOfNSamplingConfigPyClass, ChainOfThoughtConfigPyClass, ChatCompletionConfigPyClass,
        DiclConfigPyClass, MixtureOfNConfigPyClass,
    },
};
use tensorzero_core::{
    endpoints::{
        datasets::InsertDatapointParams,
        workflow_evaluation_run::WorkflowEvaluationRunEpisodeParams,
    },
    inference::types::{
        extra_body::UnfilteredInferenceExtraBody, extra_headers::UnfilteredInferenceExtraHeaders,
    },
    utils::gateway::ShutdownHandle,
};
use tensorzero_rust::{
    err_to_http, observability::LogFormat, CacheParamsOptions, Client, ClientBuilder,
    ClientBuilderMode, ClientExt, ClientInferenceParams, ClientInput, ClientSecretString,
    Datapoint, DynamicToolParams, FeedbackParams, InferenceOutput, InferenceParams,
    InferenceStream, LaunchOptimizationParams, ListDatapointsRequest, ListInferencesParams,
    OptimizationJobHandle, RenderedSample, StoredInference, TensorZeroError, Tool,
    WorkflowEvaluationRunParams,
};
use tokio::sync::Mutex;
use url::Url;

mod evaluation_handlers;
mod gil_helpers;
mod python_helpers;

use crate::evaluation_handlers::{AsyncEvaluationJobHandler, EvaluationJobHandler};
use crate::gil_helpers::{tokio_block_on_without_gil, DropInTokio};

#[pymodule]
fn tensorzero(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Make sure that we can load our error classes, so that we don't trigger
    // a nested exception when calling `convert_error` below
    let _ = tensorzero_error_class(m.py())?;
    let _ = tensorzero_core_error_class(m.py())?;
    // Otel is disabled for now in the Python client until we decide how it should be configured
    // We might have produced an error when trying to construct the (not yet enabled) OTEL layer,
    // which will just get ignored here. The HTTP gateway will handle that error, as that's
    // the only place where we actually try to enable OTEL.
    let _delayed_enable = tokio_block_on_without_gil(
        m.py(),
        tensorzero_rust::observability::setup_observability(LogFormat::Pretty),
    )
    .map_err(|e| convert_error(m.py(), TensorZeroError::Other { source: e.into() }))?;
    m.add_class::<BaseTensorZeroGateway>()?;
    m.add_class::<AsyncTensorZeroGateway>()?;
    m.add_class::<TensorZeroGateway>()?;
    m.add_class::<LocalHttpGateway>()?;
    m.add_class::<RenderedSample>()?;
    m.add_class::<StoredInference>()?;
    m.add_class::<EvaluationJobHandler>()?;
    m.add_class::<AsyncEvaluationJobHandler>()?;
    m.add_class::<UninitializedOpenAIRFTConfig>()?;
    m.add_class::<UninitializedOpenAISFTConfig>()?;
    m.add_class::<UninitializedFireworksSFTConfig>()?;
    m.add_class::<UninitializedDiclOptimizationConfig>()?;
    m.add_class::<UninitializedGCPVertexGeminiSFTConfig>()?;
    m.add_class::<UninitializedTogetherSFTConfig>()?;
    m.add_class::<Datapoint>()?;
    m.add_class::<ResolvedInput>()?;
    m.add_class::<ResolvedInputMessage>()?;
    m.add_class::<ConfigPyClass>()?;
    m.add_class::<FunctionsConfigPyClass>()?;
    m.add_class::<FunctionConfigChatPyClass>()?;
    m.add_class::<FunctionConfigJsonPyClass>()?;
    m.add_class::<VariantsConfigPyClass>()?;
    m.add_class::<ChatCompletionConfigPyClass>()?;
    m.add_class::<BestOfNSamplingConfigPyClass>()?;
    m.add_class::<DiclConfigPyClass>()?;
    m.add_class::<MixtureOfNConfigPyClass>()?;
    m.add_class::<ChainOfThoughtConfigPyClass>()?;
    m.add_class::<OptimizationJobHandle>()?;
    m.add_class::<OptimizationJobInfoPyClass>()?;
    m.add_class::<OptimizationJobStatus>()?;

    let py_json = PyModule::import(m.py(), "json")?;
    let json_loads = py_json.getattr("loads")?;
    let json_dumps = py_json.getattr("dumps")?;

    // We don't care if the PyOnceLock was already set
    let _ = JSON_LOADS.set(m.py(), json_loads.unbind());
    let _ = JSON_DUMPS.set(m.py(), json_dumps.unbind());

    m.add_wrapped(wrap_pyfunction!(_start_http_gateway))?;

    Ok(())
}

#[pyclass]
struct LocalHttpGateway {
    #[pyo3(get)]
    base_url: String,
    // We use a double `Option` so that we can implement `LocalHttpGateway.close`
    // by setting it to `None`, without needing to complicate the api of `DropInTokio`
    shutdown_handle: Option<DropInTokio<Option<ShutdownHandle>>>,
}

impl Drop for LocalHttpGateway {
    fn drop(&mut self) {
        self.close();
    }
}

#[pymethods]
impl LocalHttpGateway {
    fn close(&mut self) {
        self.shutdown_handle = None;
    }
}

#[pyfunction]
#[pyo3(signature = (*, config_file, clickhouse_url, postgres_url, async_setup))]
fn _start_http_gateway(
    py: Python<'_>,
    config_file: Option<String>,
    clickhouse_url: Option<String>,
    postgres_url: Option<String>,
    async_setup: bool,
) -> PyResult<Bound<'_, PyAny>> {
    warn_no_config(py, config_file.as_deref())?;
    let gateway_fut = async move {
        let (addr, handle) = tensorzero_core::utils::gateway::start_openai_compatible_gateway(
            config_file,
            clickhouse_url,
            postgres_url,
        )
        .await?;
        Ok(LocalHttpGateway {
            base_url: format!("http://{addr}/openai/v1"),
            shutdown_handle: Some(DropInTokio::new(Some(handle), || None)),
        })
    };
    if async_setup {
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            gateway_fut.await.map_err(|e| {
                Python::attach(|py| convert_error(py, TensorZeroError::Other { source: e }))
            })
        })
    } else {
        Ok(tokio_block_on_without_gil(py, gateway_fut)
            .map_err(|e| convert_error(py, TensorZeroError::Other { source: e }))?
            .into_bound_py_any(py)?)
    }
}

// TODO - this should extend the python `ABC` class once pyo3 supports it: https://github.com/PyO3/pyo3/issues/991
#[pyclass(subclass, frozen)]
struct BaseTensorZeroGateway {
    client: DropInTokio<Client>,
}

#[pyclass(frozen)]
struct AsyncStreamWrapper {
    stream: Arc<Mutex<InferenceStream>>,
    // A handle to the original `AsyncTensorZeroGateway` object.
    // This ensures that Python will only garbage-collect the `AsyncTensorZeroGateway`
    // after all `AsyncStreamWrapper` objects have been garbage collected.
    // This allows us to safely block from within the Drop impl of `AsyncTensorZeroGateway`.
    // knowing that there are no remaining Python objects holding on to a `ClickhouseConnectionInfo`
    _gateway: Py<PyAny>,
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
            Python::attach(|py| {
                let chunk = chunk.map_err(|e| convert_error(py, err_to_http(e)))?;
                parse_inference_chunk(py, chunk)
            })
        })
    }
}

fn check_stream_terminated(stream: Arc<Mutex<InferenceStream>>) {
    pyo3_async_runtimes::tokio::get_runtime().spawn(async move {
        let stream = stream.lock().await;
        if !stream.is_terminated() {
            tracing::warn!("Stream was garbage-collected without being iterated to completion");
        }
    });
}

impl Drop for AsyncStreamWrapper {
    fn drop(&mut self) {
        check_stream_terminated(self.stream.clone());
    }
}

#[pyclass(frozen)]
struct StreamWrapper {
    stream: Arc<Mutex<InferenceStream>>,
    // A handle to the original `TensorZeroGateway` object.
    // This ensures that Python will only garbage-collect the `TensorZeroGateway`
    // after all `StreamWrapper` objects have been garbage collected.
    // This allows us to safely block from within the Drop impl of `TensorZeroGateway`.
    // knowing that there are no remaining Python objects holding on to a `ClickhouseConnectionInfo`
    _gateway: Py<PyAny>,
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

impl Drop for StreamWrapper {
    fn drop(&mut self) {
        check_stream_terminated(self.stream.clone());
    }
}

/// Constructs a dummy embedded client. We use this so that we can move out of the real 'client'
/// field of `BaseTensorZeroGateway` when it is dropped.
fn make_dummy_client() -> Client {
    ClientBuilder::build_dummy()
}

#[pymethods]
impl BaseTensorZeroGateway {
    #[pyo3(signature = (*, input, function_name=None, model_name=None, episode_id=None, stream=None, params=None, variant_name=None, dryrun=None, output_schema=None, allowed_tools=None, provider_tools=None, additional_tools=None, tool_choice=None, parallel_tool_calls=None, internal=None, tags=None, credentials=None, cache_options=None, extra_body=None, extra_headers=None, include_original_response=None, otlp_traces_extra_headers=None, internal_dynamic_variant_config=None))]
    #[expect(clippy::too_many_arguments)]
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
        provider_tools: Option<Vec<Bound<'_, PyAny>>>,
        additional_tools: Option<Vec<HashMap<String, Bound<'_, PyAny>>>>,
        tool_choice: Option<Bound<'_, PyAny>>,
        parallel_tool_calls: Option<bool>,
        internal: Option<bool>,
        tags: Option<HashMap<String, String>>,
        credentials: Option<HashMap<String, ClientSecretString>>,
        cache_options: Option<&Bound<'_, PyDict>>,
        extra_body: Option<&Bound<'_, PyList>>,
        extra_headers: Option<&Bound<'_, PyList>>,
        include_original_response: Option<bool>,
        otlp_traces_extra_headers: Option<HashMap<String, String>>,
        internal_dynamic_variant_config: Option<&Bound<'_, PyDict>>,
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
            provider_tools,
            tool_choice,
            parallel_tool_calls,
            internal.unwrap_or(false),
            tags,
            credentials,
            cache_options,
            extra_body,
            extra_headers,
            include_original_response.unwrap_or(false),
            otlp_traces_extra_headers,
            internal_dynamic_variant_config,
        )?;
        serialize_to_dict(this.py(), params)
    }

    fn experimental_get_config(&self) -> PyResult<ConfigPyClass> {
        let config = self
            .client
            .get_config()
            .map_err(|e| PyValueError::new_err(format!("Failed to get config: {e:?}")))?;
        Ok(ConfigPyClass::new(config))
    }
}

#[pyclass(extends=BaseTensorZeroGateway)]
/// A synchronous client for a TensorZero gateway.
///
/// To connect to a running HTTP gateway, call `TensorZeroGateway.build_http(base_url = "http://gateway_url")`
/// To create an embedded gateway, call `TensorZeroGateway.build_embedded(config_file = "/path/to/tensorzero.toml", clickhouse_url = "http://clickhouse_url")`
struct TensorZeroGateway {}

impl BaseTensorZeroGateway {
    #[expect(clippy::too_many_arguments)]
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
            episode_id: episode_id
                .map(|id| python_uuid_to_uuid("episode_id", id))
                .transpose()?,
            inference_id: inference_id
                .map(|id| python_uuid_to_uuid("inference_id", id))
                .transpose()?,
            dryrun,
            tags: tags.unwrap_or_default(),
            internal,
        })
    }

    #[expect(clippy::too_many_arguments)]
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
        provider_tools: Option<Vec<Bound<'_, PyAny>>>,
        tool_choice: Option<Bound<'_, PyAny>>,
        parallel_tool_calls: Option<bool>,
        internal: bool,
        tags: Option<HashMap<String, String>>,
        credentials: Option<HashMap<String, ClientSecretString>>,
        cache_options: Option<&Bound<'_, PyDict>>,
        extra_body: Option<&Bound<'_, PyList>>,
        extra_headers: Option<&Bound<'_, PyList>>,
        include_original_response: bool,
        otlp_traces_extra_headers: Option<HashMap<String, String>>,
        internal_dynamic_variant_config: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<ClientInferenceParams> {
        let episode_id = episode_id
            .map(|id| python_uuid_to_uuid("episode_id", id))
            .transpose()?;

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

        let provider_tools: Option<Vec<ProviderTool>> = if let Some(provider_tools) = provider_tools
        {
            Some(
                provider_tools
                    .iter()
                    .map(|x| deserialize_from_pyobj(py, x))
                    .collect::<PyResult<Vec<_>>>()?,
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

        let extra_headers: UnfilteredInferenceExtraHeaders =
            if let Some(extra_headers) = extra_headers {
                deserialize_from_pyobj(py, extra_headers)?
            } else {
                Default::default()
            };

        let internal_dynamic_variant_config: Option<UninitializedVariantInfo> =
            if let Some(config) = internal_dynamic_variant_config {
                Some(deserialize_from_pyobj(py, config)?)
            } else {
                None
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
                provider_tools,
            },
            input,
            credentials: credentials.unwrap_or_default(),
            cache_options: cache_options.unwrap_or_default(),
            output_schema,
            include_original_response,
            extra_body,
            extra_headers,
            internal_dynamic_variant_config,
            otlp_traces_extra_headers: otlp_traces_extra_headers.unwrap_or_default(),
        })
    }
}
#[pymethods]
impl TensorZeroGateway {
    #[classmethod]
    #[pyo3(signature = (*, gateway_url, timeout=None, verbose_errors=false, api_key=None))]
    /// Initialize the TensorZero client, using the HTTP gateway.
    /// :param gateway_url: The base URL of the TensorZero gateway. Example: "http://localhost:3000"
    /// :param timeout: The timeout for the HTTP client in seconds. If not provided, no timeout will be set.
    /// :param verbose_errors: If true, the client will increase the detail in errors (increasing the risk of leaking sensitive information).
    /// :param api_key: The API key to use for authentication with the TensorZero Gateway. If not provided, the client will attempt to read from the TENSORZERO_API_KEY environment variable.
    /// :return: A `TensorZeroGateway` instance configured to use the HTTP gateway.
    fn build_http(
        cls: &Bound<'_, PyType>,
        gateway_url: &str,
        timeout: Option<f64>,
        verbose_errors: bool,
        api_key: Option<String>,
    ) -> PyResult<Py<TensorZeroGateway>> {
        let mut client_builder = ClientBuilder::new(ClientBuilderMode::HTTPGateway {
            url: Url::parse(gateway_url)
                .map_err(|e| PyValueError::new_err(format!("Invalid gateway URL: {e}")))?,
        })
        .with_verbose_errors(verbose_errors);
        if let Some(api_key) = api_key {
            client_builder = client_builder.with_api_key(api_key);
        }
        if let Some(timeout) = timeout {
            client_builder = client_builder.with_timeout(
                Duration::try_from_secs_f64(timeout)
                    .map_err(|e| PyValueError::new_err(format!("Invalid timeout: {e}")))?,
            );
        }
        let client_fut = client_builder.build();
        let client_res = tokio_block_on_without_gil(cls.py(), client_fut);
        let client = match client_res {
            Ok(client) => client,
            Err(e) => {
                return Err(tensorzero_core_error(
                    cls.py(),
                    &format!("Failed to construct TensorZero client: {e:?}"),
                )?);
            }
        };
        let instance = PyClassInitializer::from(BaseTensorZeroGateway {
            client: DropInTokio::new(client, make_dummy_client),
        })
        .add_subclass(TensorZeroGateway {});
        Py::new(cls.py(), instance)
    }

    /// Close the connection to the TensorZero gateway.
    #[expect(clippy::unused_self)]
    fn close(&self) {
        // TODO - implement closing the 'reqwest' connection pool: https://github.com/tensorzero/tensorzero/issues/857
    }

    fn __enter__(this: Py<Self>) -> Py<Self> {
        this
    }

    // TODO - implement closing the 'reqwest' connection pool: https://github.com/tensorzero/tensorzero/issues/857
    #[expect(clippy::unnecessary_wraps)]
    fn __exit__(
        _this: Py<Self>,
        _exc_type: Py<PyAny>,
        _exc_value: Py<PyAny>,
        _traceback: Py<PyAny>,
    ) -> PyResult<()> {
        Ok(())
    }

    #[classmethod]
    #[pyo3(signature = (*, config_file=None, clickhouse_url=None, postgres_url=None, timeout=None))]
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
        postgres_url: Option<String>,
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
            postgres_url,
            timeout,
            verify_credentials: true,
            allow_batch_writes: false,
        })
        .build();
        let client = tokio_block_on_without_gil(cls.py(), client_fut);
        let client = match client {
            Ok(client) => client,
            Err(e) => {
                return Err(tensorzero_core_error(
                    cls.py(),
                    &format!("Failed to construct TensorZero client: {e:?}"),
                )?);
            }
        };
        // Construct an instance of `TensorZeroGateway` (while providing the fields from the `BaseTensorZeroGateway` superclass).
        let instance = PyClassInitializer::from(BaseTensorZeroGateway {
            client: DropInTokio::new(client, make_dummy_client),
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
    #[expect(clippy::too_many_arguments)]
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

    #[pyo3(signature = (*, input, function_name=None, model_name=None, episode_id=None, stream=None, params=None, variant_name=None, dryrun=None, output_schema=None, allowed_tools=None, additional_tools=None, provider_tools=None, tool_choice=None, parallel_tool_calls=None, internal=None, tags=None, credentials=None, cache_options=None, extra_body=None, extra_headers=None, include_original_response=None, otlp_traces_extra_headers=None, internal_dynamic_variant_config=None))]
    #[expect(clippy::too_many_arguments)]
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
    /// :param extra_headers: If set, injects extra fields into the provider request headers.
    /// :param include_original_response: If set, add an `original_response` field to the response, containing the raw string response from the model.
    /// :param otlp_traces_extra_headers: If set, attaches custom HTTP headers to OTLP trace exports for this request.
    ///                                   Headers will be automatically prefixed with "tensorzero-otlp-traces-extra-header-".
    ///                                   Example: {"My-Header": "My-Value"} becomes header "tensorzero-otlp-traces-extra-header-My-Header: My-Value"
    /// :return: If stream is false, returns an InferenceResponse.
    ///          If stream is true, returns a generator that yields InferenceChunks as they come in.
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
        provider_tools: Option<Vec<Bound<'_, PyAny>>>,
        tool_choice: Option<Bound<'_, PyAny>>,
        parallel_tool_calls: Option<bool>,
        internal: Option<bool>,
        tags: Option<HashMap<String, String>>,
        credentials: Option<HashMap<String, ClientSecretString>>,
        cache_options: Option<&Bound<'_, PyDict>>,
        extra_body: Option<&Bound<'_, PyList>>,
        extra_headers: Option<&Bound<'_, PyList>>,
        include_original_response: Option<bool>,
        otlp_traces_extra_headers: Option<HashMap<String, String>>,
        internal_dynamic_variant_config: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let client = this.as_super().client.clone();
        let fut = client.inference(BaseTensorZeroGateway::prepare_inference_params(
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
            provider_tools,
            tool_choice,
            parallel_tool_calls,
            internal.unwrap_or(false),
            tags,
            credentials,
            cache_options,
            extra_body,
            extra_headers,
            include_original_response.unwrap_or(false),
            otlp_traces_extra_headers,
            internal_dynamic_variant_config,
        )?);

        // We're in the synchronous `TensorZeroGateway` class, so we need to block on the Rust future,
        // and then return the result to the Python caller directly (not wrapped in a Python `Future`).
        let resp = tokio_block_on_without_gil(py, fut).map_err(|e| convert_error(py, e))?;
        match resp {
            InferenceOutput::NonStreaming(data) => parse_inference_response(py, data),
            InferenceOutput::Streaming(stream) => Ok(StreamWrapper {
                stream: Arc::new(Mutex::new(stream)),
                _gateway: this.into_pyobject(py)?.into_any().unbind(),
            }
            .into_pyobject(py)?
            .into_any()
            .unbind()),
        }
    }

    /// Make a request to the /workflow_evaluation_run endpoint.
    ///
    /// :param variants: A dictionary mapping function names to pinned variant names.
    /// :param tags: A dictionary containing tags that should be applied to every inference in the workflow evaluation run.
    /// :param project_name: (Optional) The name of the project to associate with the workflow evaluation run.
    /// :param run_display_name: (Optional) The display name of the workflow evaluation run.
    /// :return: A `WorkflowEvaluationRunResponse` object.
    #[pyo3(signature = (*, variants, tags=None, project_name=None, display_name=None))]
    fn workflow_evaluation_run(
        this: PyRef<'_, Self>,
        variants: HashMap<String, String>,
        tags: Option<HashMap<String, String>>,
        project_name: Option<String>,
        display_name: Option<String>,
    ) -> PyResult<Py<PyAny>> {
        let client = this.as_super().client.clone();
        let params = WorkflowEvaluationRunParams {
            internal: false,
            variants,
            tags: tags.unwrap_or_default(),
            project_name,
            display_name,
        };
        let fut = client.workflow_evaluation_run(params);

        let resp = tokio_block_on_without_gil(this.py(), fut);
        match resp {
            Ok(resp) => parse_workflow_evaluation_run_response(this.py(), resp),
            Err(e) => Err(convert_error(this.py(), e)),
        }
    }

    /// DEPRECATED: Use workflow_evaluation_run instead.
    /// Make a request to the /dynamic_evaluation_run endpoint.
    ///
    /// :param variants: A dictionary mapping function names to pinned variant names.
    /// :param tags: A dictionary containing tags that should be applied to every inference in the dynamic evaluation run.
    /// :param project_name: (Optional) The name of the project to associate with the dynamic evaluation run.
    /// :param run_display_name: (Optional) The display name of the dynamic evaluation run.
    /// :return: A `DynamicEvaluationRunResponse` object (alias for WorkflowEvaluationRunResponse).
    #[pyo3(signature = (*, variants, tags=None, project_name=None, display_name=None))]
    fn dynamic_evaluation_run(
        this: PyRef<'_, Self>,
        variants: HashMap<String, String>,
        tags: Option<HashMap<String, String>>,
        project_name: Option<String>,
        display_name: Option<String>,
    ) -> PyResult<Py<PyAny>> {
        let warnings = PyModule::import(this.py(), "warnings")?;
        warnings.call_method1(
            "warn",
            (
                "The dynamic_evaluation_run method is deprecated. Please use workflow_evaluation_run instead. Support for dynamic_evaluation_run will be removed in a future version.",
                this.py().get_type::<PyDeprecationWarning>(),
            ),
        )?;
        Self::workflow_evaluation_run(this, variants, tags, project_name, display_name)
    }

    /// Make a request to the /workflow_evaluation_run_episode endpoint.
    ///
    /// :param run_id: The run ID to use for the workflow evaluation run.
    /// :param task_name: The name of the task to use for the workflow evaluation run.
    /// :param tags: A dictionary of tags to add to the workflow evaluation run.
    /// :return: A `WorkflowEvaluationRunEpisodeResponse` object.
    #[pyo3(signature = (*, run_id, task_name=None, tags=None))]
    fn workflow_evaluation_run_episode(
        this: PyRef<'_, Self>,
        run_id: Bound<'_, PyAny>,
        task_name: Option<String>,
        tags: Option<HashMap<String, String>>,
    ) -> PyResult<Py<PyAny>> {
        let run_id = python_uuid_to_uuid("run_id", run_id)?;
        let client = this.as_super().client.clone();
        let params = WorkflowEvaluationRunEpisodeParams {
            task_name,
            tags: tags.unwrap_or_default(),
        };
        let fut = client.workflow_evaluation_run_episode(run_id, params);
        let resp = tokio_block_on_without_gil(this.py(), fut);
        match resp {
            Ok(resp) => parse_workflow_evaluation_run_episode_response(this.py(), resp),
            Err(e) => Err(convert_error(this.py(), e)),
        }
    }

    /// DEPRECATED: Use workflow_evaluation_run_episode instead.
    /// Make a request to the /dynamic_evaluation_run_episode endpoint.
    ///
    /// :param run_id: The run ID to use for the dynamic evaluation run.
    /// :param task_name: The name of the task to use for the dynamic evaluation run.
    /// :param tags: A dictionary of tags to add to the dynamic evaluation run.
    /// :return: A `DynamicEvaluationRunEpisodeResponse` object (alias for WorkflowEvaluationRunEpisodeResponse).
    #[pyo3(signature = (*, run_id, task_name=None, tags=None))]
    fn dynamic_evaluation_run_episode(
        this: PyRef<'_, Self>,
        run_id: Bound<'_, PyAny>,
        task_name: Option<String>,
        tags: Option<HashMap<String, String>>,
    ) -> PyResult<Py<PyAny>> {
        let warnings = PyModule::import(this.py(), "warnings")?;
        warnings.call_method1(
            "warn",
            (
                "The dynamic_evaluation_run_episode method is deprecated. Please use workflow_evaluation_run_episode instead. Support for dynamic_evaluation_run_episode will be removed in a future version.",
                this.py().get_type::<PyDeprecationWarning>(),
            ),
        )?;
        Self::workflow_evaluation_run_episode(this, run_id, task_name, tags)
    }

    ///  Make a POST request to the /datasets/{dataset_name}/datapoints endpoint.
    ///
    /// :param dataset_name: The name of the dataset to insert the datapoints into.
    /// :param datapoints: A list of datapoints to insert.
    /// :return: None.
    #[pyo3(signature = (*, dataset_name, datapoints))]
    fn create_datapoints(
        this: PyRef<'_, Self>,
        dataset_name: String,
        datapoints: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyList>> {
        let client = this.as_super().client.clone();
        let datapoints = datapoints
            .iter()
            .map(|dp| deserialize_from_pyobj(this.py(), dp))
            .collect::<Result<Vec<_>, _>>()?;

        #[expect(deprecated)]
        let fut =
            client.create_datapoints_legacy(dataset_name, InsertDatapointParams { datapoints });
        let self_module = PyModule::import(this.py(), "uuid")?;
        let uuid = self_module.getattr("UUID")?.unbind();
        let res =
            tokio_block_on_without_gil(this.py(), fut).map_err(|e| convert_error(this.py(), e))?;
        let uuids = res
            .iter()
            .map(|x| uuid.call(this.py(), (x.to_string(),), None))
            .collect::<Result<Vec<_>, _>>()?;
        PyList::new(this.py(), uuids).map(Bound::unbind)
    }

    /// DEPRECATED: Use `create_datapoints` instead.
    ///
    /// Make a POST request to the /datasets/{dataset_name}/datapoints/bulk endpoint.
    ///
    /// :param dataset_name: The name of the dataset to insert the datapoints into.
    /// :param datapoints: A list of datapoints to insert.
    /// :return: None.
    #[pyo3(signature = (*, dataset_name, datapoints))]
    #[pyo3(warn(message = "Please use `create_datapoints` instead of `bulk_insert_datapoints`. In a future release, `bulk_insert_datapoints` will be removed.", category = PyDeprecationWarning))]
    fn bulk_insert_datapoints(
        this: PyRef<'_, Self>,
        dataset_name: String,
        datapoints: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyList>> {
        let client = this.as_super().client.clone();
        let datapoints = datapoints
            .iter()
            .map(|dp| deserialize_from_pyobj(this.py(), dp))
            .collect::<Result<Vec<_>, _>>()?;
        let params = InsertDatapointParams { datapoints };
        #[expect(deprecated)]
        let fut = client.bulk_insert_datapoints(dataset_name, params);
        let self_module = PyModule::import(this.py(), "uuid")?;
        let uuid = self_module.getattr("UUID")?.unbind();
        let res =
            tokio_block_on_without_gil(this.py(), fut).map_err(|e| convert_error(this.py(), e))?;
        let uuids = res
            .iter()
            .map(|x| uuid.call(this.py(), (x.to_string(),), None))
            .collect::<Result<Vec<_>, _>>()?;
        PyList::new(this.py(), uuids).map(Bound::unbind)
    }

    /// Make a DELETE request to the /datasets/{dataset_name}/datapoints/{datapoint_id} endpoint.
    ///
    /// :param dataset_name: The name of the dataset to delete the datapoint from.
    /// :param datapoint_id: The ID of the datapoint to delete.
    /// :return: None.
    #[pyo3(signature = (*, dataset_name, datapoint_id))]
    fn delete_datapoint(
        this: PyRef<'_, Self>,
        dataset_name: String,
        datapoint_id: Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let client = this.as_super().client.clone();
        let datapoint_id = python_uuid_to_uuid("datapoint_id", datapoint_id)?;
        #[expect(deprecated)]
        let fut = client.delete_datapoint(dataset_name, datapoint_id);
        tokio_block_on_without_gil(this.py(), fut).map_err(|e| convert_error(this.py(), e))
    }

    /// Make a GET request to the /datasets/{dataset_name}/datapoints/{datapoint_id} endpoint.
    ///
    /// :param dataset_name: The name of the dataset to get the datapoint from.
    /// :param datapoint_id: The ID of the datapoint to get.
    /// :return: A `Datapoint` object.
    #[pyo3(signature = (*, dataset_name, datapoint_id))]
    fn get_datapoint<'py>(
        this: PyRef<'py, Self>,
        dataset_name: String,
        datapoint_id: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Datapoint>> {
        let client = this.as_super().client.clone();
        let datapoint_id = python_uuid_to_uuid("datapoint_id", datapoint_id)?;
        #[expect(deprecated)]
        let fut = client.get_datapoint(dataset_name, datapoint_id);
        let wire: Datapoint =
            tokio_block_on_without_gil(this.py(), fut).map_err(|e| convert_error(this.py(), e))?;
        wire.into_pyobject(this.py())
    }

    /// Make a GET request to the /datasets/{dataset_name}/datapoints endpoint.
    ///
    /// :param dataset_name: The name of the dataset to get the datapoints from.
    /// :return: A list of `Datapoint` objects.
    #[pyo3(signature = (*, dataset_name, function_name=None, limit=None, offset=None))]
    fn list_datapoints(
        this: PyRef<'_, Self>,
        dataset_name: String,
        function_name: Option<String>,
        limit: Option<u32>,
        offset: Option<u32>,
    ) -> PyResult<Bound<'_, PyList>> {
        let client = this.as_super().client.clone();
        let request = ListDatapointsRequest {
            function_name,
            limit,
            offset,
            ..Default::default()
        };
        let fut = client.list_datapoints(dataset_name, request);
        let resp = tokio_block_on_without_gil(this.py(), fut);
        match resp {
            Ok(datapoints) => {
                let py_datapoints = datapoints
                    .datapoints
                    .into_iter()
                    .map(|x| x.into_pyobject(this.py()))
                    .collect::<Result<Vec<_>, _>>()?;
                PyList::new(this.py(), py_datapoints)
            }
            Err(e) => Err(convert_error(this.py(), e)),
        }
    }

    /// Run a tensorzero Evaluation
    ///
    /// This function is only available in EmbeddedGateway mode.
    ///
    /// # Arguments
    ///
    /// * `evaluation_name` - User chosen name of the evaluation.
    /// * `dataset_name` - The name of the stored dataset to use for variant evaluation
    /// * `variant_name` - The name of the variant to evaluate
    /// * `concurrency` - The maximum number of examples to process in parallel
    /// * `inference_cache` - Cache configuration for inference requests ("on", "off", "read_only", or "write_only")
    #[pyo3(signature = (*,
                        evaluation_name,
                        dataset_name,
                        variant_name,
                        concurrency=1,
                        inference_cache="on".to_string()
    ),
    text_signature = "(self, *, evaluation_name, dataset_name, variant_name, concurrency=1, inference_cache='on')"
    )]
    fn experimental_run_evaluation(
        this: PyRef<'_, Self>,
        evaluation_name: String,
        dataset_name: String,
        variant_name: String,
        concurrency: usize,
        inference_cache: String,
    ) -> PyResult<EvaluationJobHandler> {
        let client = this.as_super().client.clone();

        // Get app state data
        let app_state = client.get_app_state_data().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Client is not in EmbeddedGateway mode")
        })?;

        let evaluation_run_id = uuid::Uuid::now_v7();

        let inference_cache_enum: tensorzero_core::cache::CacheEnabledMode =
            deserialize_from_pyobj(
                this.py(),
                &inference_cache.into_pyobject(this.py())?.into_any(),
            )?;

        let core_args = EvaluationCoreArgs {
            tensorzero_client: (*client).clone(),
            clickhouse_client: app_state.clickhouse_connection_info.clone(),
            config: app_state.config.clone(),
            evaluation_name,
            evaluation_run_id,
            dataset_name,
            variant_name,
            concurrency,
            inference_cache: inference_cache_enum,
        };

        let result =
            tokio_block_on_without_gil(this.py(), run_evaluation_core_streaming(core_args))
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("Evaluation failed: {e}"))
                })?;

        Ok(EvaluationJobHandler {
            receiver: Mutex::new(result.receiver),
            run_info: result.run_info,
            evaluation_config: result.evaluation_config,
            evaluation_infos: Arc::new(Mutex::new(Vec::new())),
            evaluation_errors: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// Query the Clickhouse database for inferences.
    ///
    /// This function is only available in EmbeddedGateway mode.
    ///
    /// # Arguments
    ///
    /// * `function_name` - The name of the function to query.
    /// * `variant_name` - The name of the variant to query. Optional
    /// * `filters` - A filter tree to apply to the query. Optional
    /// * `output_source` - The source of the output to query. "inference" or "demonstration"
    /// * `limit` - The maximum number of inferences to return. Optional
    /// * `offset` - The offset to start from. Optional
    #[pyo3(signature = (*,
                        function_name,
                        variant_name=None,
                        filters=None,
                        output_source="inference".to_string(),
                        order_by=None,
                        limit=None,
                        offset=None
    ),
    text_signature = "(self, *, function_name, variant_name=None, filters=None, output_source='inference', order_by=None, limit=None, offset=None)"
    )]
    // The text_signature is a workaround to weird behavior in pyo3 where the default for an option
    // is written as an ellipsis object.
    #[expect(clippy::too_many_arguments)]
    fn experimental_list_inferences(
        this: PyRef<'_, Self>,
        function_name: String,
        variant_name: Option<String>,
        filters: Option<Bound<'_, PyAny>>,
        output_source: String,
        order_by: Option<Bound<'_, PyAny>>,
        limit: Option<u64>,
        offset: Option<u64>,
    ) -> PyResult<Vec<StoredInference>> {
        let client = this.as_super().client.clone();
        let filters = filters
            .as_ref()
            .map(|x| deserialize_from_pyobj(this.py(), x))
            .transpose()?;
        let output_source =
            output_source
                .as_str()
                .try_into()
                .map_err(|e: tensorzero_core::error::Error| {
                    convert_error(this.py(), TensorZeroError::Other { source: e.into() })
                })?;
        let order_by: Option<Vec<OrderBy>> = order_by
            .as_ref()
            .map(|x| deserialize_from_pyobj(this.py(), x))
            .transpose()?;
        let params = ListInferencesParams {
            function_name: Some(&function_name),
            variant_name: variant_name.as_deref(),
            filters: filters.as_ref(),
            output_source,
            order_by: order_by.as_deref(),
            limit,
            offset,
            ..Default::default()
        };
        let fut = client.experimental_list_inferences(params);
        let wires: Vec<StoredInference> =
            tokio_block_on_without_gil(this.py(), fut).map_err(|e| convert_error(this.py(), e))?;
        Ok(wires)
    }

    /// DEPRECATED: use `experimental_render_samples` instead.
    /// Render a list of stored inferences into a list of rendered stored inferences.
    /// There are two things that need to happen in this function:
    /// 1. We need to resolve all network resources (e.g. images) in the stored inferences.
    /// 2. We need to prepare all messages into "simple" messages that have been templated for a particular variant.
    ///    To do this, we need to know what variant to use for each function that might appear in the data.
    ///
    /// IMPORTANT: For now, this function drops datapoints which are bad, e.g. ones where templating fails, the function
    ///            has no variant specified, or where the process of downloading resources fails.
    ///            In future we will make this behavior configurable by the caller.
    ///
    /// :param stored_inferences: A list of stored inferences to render.
    /// :param variants: A map from function name to variant name.
    /// :return: A list of rendered stored inferences.
    #[pyo3(signature = (*, stored_inferences, variants))]
    fn experimental_render_inferences(
        this: PyRef<'_, Self>,
        stored_inferences: Vec<Bound<'_, PyAny>>,
        variants: HashMap<String, String>,
    ) -> PyResult<Vec<RenderedSample>> {
        tracing::warn!("experimental_render_inferences is deprecated. Use experimental_render_samples instead. See https://github.com/tensorzero/tensorzero/issues/2675");
        let client = this.as_super().client.clone();
        let config = client.config().ok_or_else(|| {
            PyValueError::new_err(
                "Config not available in HTTP gateway mode. Use embedded mode for render_samples.",
            )
        })?;
        // Enter the Tokio runtime context while still holding the GIL
        // This is needed because deserialize_from_stored_sample may use tokio::spawn internally
        // for JSON schema compilation
        // TODO (#4259): remove the tokio spawn from that function and remove this guard.
        let _guard = pyo3_async_runtimes::tokio::get_runtime().enter();
        let stored_inferences = stored_inferences
            .iter()
            .map(|x| deserialize_from_stored_sample(this.py(), x, config))
            .collect::<Result<Vec<_>, _>>()?;
        let fut = client.experimental_render_samples(stored_inferences, variants);
        tokio_block_on_without_gil(this.py(), fut).map_err(|e| convert_error(this.py(), e))
    }

    /// Render a list of stored samples (datapoints or inferences) into a list of rendered stored samples.
    /// There are two things that need to happen in this function:
    /// 1. We need to resolve all network resources (e.g. images) in the stored samples.
    /// 2. We need to prepare all messages into "simple" messages that have been templated for a particular variant.
    ///    To do this, we need to know what variant to use for each function that might appear in the data.
    ///
    /// IMPORTANT: For now, this function drops datapoints which are bad, e.g. ones where templating fails, the function
    ///            has no variant specified, or where the process of downloading resources fails.
    ///            In future we will make this behavior configurable by the caller.
    ///
    /// :param stored_samples: A list of stored samples to render.
    /// :param variants: A map from function name to variant name.
    /// :return: A list of rendered samples.
    #[pyo3(signature = (*, stored_samples, variants))]
    fn experimental_render_samples(
        this: PyRef<'_, Self>,
        stored_samples: Vec<Bound<'_, PyAny>>,
        variants: HashMap<String, String>,
    ) -> PyResult<Vec<RenderedSample>> {
        let client = this.as_super().client.clone();
        let config = client.config().ok_or_else(|| {
            PyValueError::new_err(
                "Config not available in HTTP gateway mode. Use embedded mode for render_samples.",
            )
        })?;
        // Enter the Tokio runtime context while still holding the GIL
        // This is needed because deserialize_from_stored_sample may use tokio::spawn internally
        // for JSON schema compilation
        // TODO (#4259): remove the tokio spawn from that function and remove this guard.
        let _guard = pyo3_async_runtimes::tokio::get_runtime().enter();
        let stored_samples = stored_samples
            .iter()
            .map(|x| deserialize_from_stored_sample(this.py(), x, config))
            .collect::<Result<Vec<_>, _>>()?;
        let fut = client.experimental_render_samples(stored_samples, variants);
        tokio_block_on_without_gil(this.py(), fut).map_err(|e| convert_error(this.py(), e))
    }

    /// Launch an optimization job.
    ///
    /// :param train_samples: A list of RenderedSample objects that will be used for training.
    /// :param val_samples: A list of RenderedSample objects that will be used for validation.
    /// :param optimization_config: The optimization config.
    /// :return: A `OptimizerJobHandle` object that can be used to poll the optimization job.
    #[pyo3(signature = (*, train_samples, val_samples=None, optimization_config))]
    fn experimental_launch_optimization(
        this: PyRef<'_, Self>,
        train_samples: Vec<Bound<'_, PyAny>>,
        val_samples: Option<Vec<Bound<'_, PyAny>>>,
        optimization_config: Bound<'_, PyAny>,
    ) -> PyResult<OptimizationJobHandle> {
        let client = this.as_super().client.clone();
        let train_samples = train_samples
            .iter()
            .map(|x| deserialize_from_rendered_sample(this.py(), x))
            .collect::<Result<Vec<_>, _>>()?;
        let val_samples = val_samples
            .map(|x| {
                x.iter()
                    .map(|x| deserialize_from_rendered_sample(this.py(), x))
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()?;
        let optimization_config = deserialize_optimization_config(&optimization_config)?;
        let fut = client.experimental_launch_optimization(LaunchOptimizationParams {
            train_samples,
            val_samples,
            optimization_config: UninitializedOptimizerInfo {
                inner: optimization_config,
            },
        });
        tokio_block_on_without_gil(this.py(), fut).map_err(|e| convert_error(this.py(), e))
    }

    /// Poll an optimization job.
    ///
    /// :param job_handle: The job handle returned by `experimental_launch_optimization`.
    /// :return: An `OptimizerStatus` object.
    #[pyo3(signature = (*, job_handle))]
    fn experimental_poll_optimization(
        this: PyRef<'_, Self>,
        job_handle: OptimizationJobHandle,
    ) -> PyResult<OptimizationJobInfoPyClass> {
        let client = this.as_super().client.clone();
        let fut = client.experimental_poll_optimization(&job_handle);
        match tokio_block_on_without_gil(this.py(), fut) {
            Ok(status) => Ok(OptimizationJobInfoPyClass::new(status)),
            Err(e) => Err(convert_error(this.py(), e)),
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
    #[classmethod]
    #[pyo3(signature = (*, gateway_url, timeout=None, verbose_errors=false, async_setup=true, api_key=None))]
    /// Initialize the TensorZero client, using the HTTP gateway.
    /// :param gateway_url: The base URL of the TensorZero gateway. Example: "http://localhost:3000"
    /// :param timeout: The timeout for the HTTP client in seconds. If not provided, no timeout will be set.
    /// :param verbose_errors: If true, the client will increase the detail in errors (increasing the risk of leaking sensitive information).
    /// :param async_setup: If true, this method will return a `Future` that resolves to an `AsyncTensorZeroGateway` instance. Otherwise, it will block and construct the `AsyncTensorZeroGateway`
    /// :param api_key: The API key to use for authentication with the TensorZero Gateway. If not provided, the client will attempt to read from the TENSORZERO_API_KEY environment variable.
    /// :return: An `AsyncTensorZeroGateway` instance configured to use the HTTP gateway.
    fn build_http(
        cls: &Bound<'_, PyType>,
        gateway_url: &str,
        timeout: Option<f64>,
        verbose_errors: bool,
        async_setup: bool,
        api_key: Option<String>,
    ) -> PyResult<Py<PyAny>> {
        let mut client_builder = ClientBuilder::new(ClientBuilderMode::HTTPGateway {
            url: Url::parse(gateway_url)
                .map_err(|e| PyValueError::new_err(format!("Invalid gateway URL: {e}")))?,
        })
        .with_verbose_errors(verbose_errors);
        if let Some(api_key) = api_key {
            client_builder = client_builder.with_api_key(api_key);
        }
        if let Some(timeout) = timeout {
            client_builder = client_builder.with_timeout(Duration::from_secs_f64(timeout));
        }
        let client_fut = client_builder.build();
        let build_gateway = async move {
            let client = client_fut.await;
            // We need to interact with Python objects here (to build up a Python `AsyncTensorZeroGateway`),
            // so we need the GIL
            Python::attach(|py| {
                let client = match client {
                    Ok(client) => client,
                    Err(e) => {
                        return Err(tensorzero_core_error(
                            py,
                            &format!("Failed to construct TensorZero client: {e:?}"),
                        )?);
                    }
                };

                // Construct an instance of `AsyncTensorZeroGateway` (while providing the fields from the `BaseTensorZeroGateway` superclass).
                let instance = PyClassInitializer::from(BaseTensorZeroGateway {
                    client: DropInTokio::new(client, make_dummy_client),
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
    #[pyo3(signature = (*, config_file=None, clickhouse_url=None, postgres_url=None, timeout=None, async_setup=true))]
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
        postgres_url: Option<String>,
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
            postgres_url,
            timeout,
            verify_credentials: true,
            allow_batch_writes: false,
        })
        .build();
        let fut = async move {
            let client = client_fut.await;
            // We need to interact with Python objects here (to build up a Python `AsyncTensorZeroGateway`),
            // so we need the GIL
            Python::attach(|py| {
                let client = match client {
                    Ok(client) => client,
                    Err(e) => {
                        return Err(tensorzero_core_error(
                            py,
                            &format!("Failed to construct TensorZero client: {e:?}"),
                        )?);
                    }
                };

                // Construct an instance of `AsyncTensorZeroGateway` (while providing the fields from the `BaseTensorZeroGateway` superclass).
                let instance = PyClassInitializer::from(BaseTensorZeroGateway {
                    client: DropInTokio::new(client, make_dummy_client),
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

    #[pyo3(signature = (*, input, function_name=None, model_name=None, episode_id=None, stream=None, params=None, variant_name=None, dryrun=None, output_schema=None, allowed_tools=None, additional_tools=None, provider_tools=None, tool_choice=None, parallel_tool_calls=None, internal=None,tags=None, credentials=None, cache_options=None, extra_body=None, extra_headers=None, include_original_response=None, otlp_traces_extra_headers=None, internal_dynamic_variant_config=None))]
    #[expect(clippy::too_many_arguments)]
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
    /// :param extra_headers: If set, injects extra fields into the provider request headers.
    /// :param include_original_response: If set, add an `original_response` field to the response, containing the raw string response from the model.
    /// :param otlp_traces_extra_headers: If set, attaches custom HTTP headers to OTLP trace exports for this request.
    ///                                   Headers will be automatically prefixed with "tensorzero-otlp-traces-extra-header-".
    ///                                   Example: {"My-Header": "My-Value"} becomes header "tensorzero-otlp-traces-extra-header-My-Header: My-Value"
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
        provider_tools: Option<Vec<Bound<'_, PyAny>>>,
        tool_choice: Option<Bound<'_, PyAny>>,
        parallel_tool_calls: Option<bool>,
        internal: Option<bool>,
        tags: Option<HashMap<String, String>>,
        credentials: Option<HashMap<String, ClientSecretString>>,
        cache_options: Option<&Bound<'_, PyDict>>,
        extra_body: Option<&Bound<'_, PyList>>,
        extra_headers: Option<&Bound<'_, PyList>>,
        include_original_response: Option<bool>,
        otlp_traces_extra_headers: Option<HashMap<String, String>>,
        internal_dynamic_variant_config: Option<&Bound<'_, PyDict>>,
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
            provider_tools,
            tool_choice,
            parallel_tool_calls,
            internal.unwrap_or(false),
            tags,
            credentials,
            cache_options,
            extra_body,
            extra_headers,
            include_original_response.unwrap_or(false),
            otlp_traces_extra_headers,
            internal_dynamic_variant_config,
        )?;
        let client = this.as_super().client.clone();
        let gateway = this.into_pyobject(py)?.into_any().unbind();
        // See `AsyncStreamWrapper::__anext__` for more details about `future_into_py`
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let res = client.inference(params).await;
            // We need to interact with Python objects here (to build up a Python inference response),
            // so we need the GIL
            Python::attach(|py| {
                let output = res.map_err(|e| convert_error(py, e))?;
                match output {
                    InferenceOutput::NonStreaming(data) => parse_inference_response(py, data),
                    InferenceOutput::Streaming(stream) => Ok(AsyncStreamWrapper {
                        stream: Arc::new(Mutex::new(stream)),
                        _gateway: gateway,
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
    #[expect(clippy::too_many_arguments)]
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
            Python::attach(|py| match res {
                Ok(resp) => Ok(parse_feedback_response(py, resp)?.into_any()),
                Err(e) => Err(convert_error(py, e)),
            })
        })
    }

    /// Make a request to the /workflow_evaluation_run endpoint.
    ///
    /// :param variants: A dictionary mapping function names to pinned variant names.
    /// :param tags: A dictionary containing tags that should be applied to every inference in the workflow evaluation run.
    /// :param project_name: (Optional) The name of the project to associate with the workflow evaluation run.
    /// :param run_display_name: (Optional) The display name of the workflow evaluation run.
    /// :return: A `WorkflowEvaluationRunResponse` object.
    #[pyo3(signature = (*, variants, tags=None, project_name=None, display_name=None))]
    fn workflow_evaluation_run(
        this: PyRef<'_, Self>,
        variants: HashMap<String, String>,
        tags: Option<HashMap<String, String>>,
        project_name: Option<String>,
        display_name: Option<String>,
    ) -> PyResult<Bound<'_, PyAny>> {
        let client = this.as_super().client.clone();
        let params = WorkflowEvaluationRunParams {
            internal: false,
            variants,
            tags: tags.unwrap_or_default(),
            project_name,
            display_name,
        };

        pyo3_async_runtimes::tokio::future_into_py(this.py(), async move {
            let res = client.workflow_evaluation_run(params).await;
            Python::attach(|py| match res {
                Ok(resp) => parse_workflow_evaluation_run_response(py, resp),
                Err(e) => Err(convert_error(py, e)),
            })
        })
    }

    /// DEPRECATED: Use workflow_evaluation_run instead.
    /// Make a request to the /dynamic_evaluation_run endpoint.
    ///
    /// :param variants: A dictionary mapping function names to pinned variant names.
    /// :param tags: A dictionary containing tags that should be applied to every inference in the dynamic evaluation run.
    /// :param project_name: (Optional) The name of the project to associate with the dynamic evaluation run.
    /// :param run_display_name: (Optional) The display name of the dynamic evaluation run.
    /// :return: A `DynamicEvaluationRunResponse` object (alias for WorkflowEvaluationRunResponse).
    #[pyo3(signature = (*, variants, tags=None, project_name=None, display_name=None))]
    fn dynamic_evaluation_run(
        this: PyRef<'_, Self>,
        variants: HashMap<String, String>,
        tags: Option<HashMap<String, String>>,
        project_name: Option<String>,
        display_name: Option<String>,
    ) -> PyResult<Bound<'_, PyAny>> {
        let warnings = PyModule::import(this.py(), "warnings")?;
        warnings.call_method1(
            "warn",
            (
                "The dynamic_evaluation_run method is deprecated. Please use workflow_evaluation_run instead. Support for dynamic_evaluation_run will be removed in a future version.",
                this.py().get_type::<PyDeprecationWarning>(),
            ),
        )?;
        Self::workflow_evaluation_run(this, variants, tags, project_name, display_name)
    }

    /// Make a request to the /workflow_evaluation_run_episode endpoint.
    ///
    /// :param run_id: The run ID to use for the workflow evaluation run.
    /// :param task_name: The name of the task to use for the workflow evaluation run.
    /// :param tags: A dictionary of tags to add to the workflow evaluation run.
    /// :return: A `WorkflowEvaluationRunEpisodeResponse` object.
    #[pyo3(signature = (*, run_id, task_name=None, tags=None))]
    fn workflow_evaluation_run_episode<'a>(
        this: PyRef<'a, Self>,
        run_id: Bound<'_, PyAny>,
        task_name: Option<String>,
        tags: Option<HashMap<String, String>>,
    ) -> PyResult<Bound<'a, PyAny>> {
        let run_id = python_uuid_to_uuid("run_id", run_id)?;
        let client = this.as_super().client.clone();
        let params = WorkflowEvaluationRunEpisodeParams {
            task_name,
            tags: tags.unwrap_or_default(),
        };

        pyo3_async_runtimes::tokio::future_into_py(this.py(), async move {
            let res = client.workflow_evaluation_run_episode(run_id, params).await;
            Python::attach(|py| match res {
                Ok(resp) => parse_workflow_evaluation_run_episode_response(py, resp),
                Err(e) => Err(convert_error(py, e)),
            })
        })
    }

    /// DEPRECATED: Use workflow_evaluation_run_episode instead.
    /// Make a request to the /dynamic_evaluation_run_episode endpoint.
    ///
    /// :param run_id: The run ID to use for the dynamic evaluation run.
    /// :param task_name: The name of the task to use for the dynamic evaluation run.
    /// :param tags: A dictionary of tags to add to the dynamic evaluation run.
    /// :return: A `DynamicEvaluationRunEpisodeResponse` object (alias for WorkflowEvaluationRunEpisodeResponse).
    #[pyo3(signature = (*, run_id, task_name=None, tags=None))]
    fn dynamic_evaluation_run_episode<'a>(
        this: PyRef<'a, Self>,
        run_id: Bound<'_, PyAny>,
        task_name: Option<String>,
        tags: Option<HashMap<String, String>>,
    ) -> PyResult<Bound<'a, PyAny>> {
        let warnings = PyModule::import(this.py(), "warnings")?;
        warnings.call_method1(
            "warn",
            (
                "The dynamic_evaluation_run_episode method is deprecated. Please use workflow_evaluation_run_episode instead. Support for dynamic_evaluation_run_episode will be removed in a future version.",
                this.py().get_type::<PyDeprecationWarning>(),
            ),
        )?;
        Self::workflow_evaluation_run_episode(this, run_id, task_name, tags)
    }

    ///  Make a POST request to the /datasets/{dataset_name}/datapoints endpoint.
    ///
    /// :param dataset_name: The name of the dataset to insert the datapoints into.
    /// :param datapoints: A list of datapoints to insert.
    /// :return: None.
    #[pyo3(signature = (*, dataset_name, datapoints))]
    fn create_datapoints<'a>(
        this: PyRef<'a, Self>,
        dataset_name: String,
        datapoints: Vec<Bound<'a, PyAny>>,
    ) -> PyResult<Bound<'a, PyAny>> {
        let client = this.as_super().client.clone();
        let datapoints = datapoints
            .iter()
            .map(|dp| deserialize_from_pyobj(this.py(), dp))
            .collect::<Result<Vec<_>, _>>()?;
        let params = InsertDatapointParams { datapoints };
        let self_module = PyModule::import(this.py(), "uuid")?;
        let uuid = self_module.getattr("UUID")?.unbind();
        pyo3_async_runtimes::tokio::future_into_py(this.py(), async move {
            #[expect(deprecated)]
            let res = client.create_datapoints_legacy(dataset_name, params).await;
            Python::attach(|py| match res {
                Ok(uuids) => Ok(PyList::new(
                    py,
                    uuids
                        .iter()
                        .map(|id| uuid.call(py, (id.to_string(),), None))
                        .collect::<Result<Vec<_>, _>>()?,
                )?
                .unbind()),
                Err(e) => Err(convert_error(py, e)),
            })
        })
    }

    /// DEPRECATED: Use `create_datapoints` instead.
    ///
    /// Make a POST request to the /datasets/{dataset_name}/datapoints/bulk endpoint.
    ///
    /// :param dataset_name: The name of the dataset to insert the datapoints into.
    /// :param datapoints: A list of datapoints to insert.
    /// :return: None.
    #[pyo3(signature = (*, dataset_name, datapoints))]
    #[pyo3(warn(message = "Please use `create_datapoints` instead of `bulk_insert_datapoints`. In a future release, `bulk_insert_datapoints` will be removed.", category = PyDeprecationWarning))]
    fn bulk_insert_datapoints<'a>(
        this: PyRef<'a, Self>,
        dataset_name: String,
        datapoints: Vec<Bound<'a, PyAny>>,
    ) -> PyResult<Bound<'a, PyAny>> {
        let client = this.as_super().client.clone();
        let datapoints = datapoints
            .iter()
            .map(|dp| deserialize_from_pyobj(this.py(), dp))
            .collect::<Result<Vec<_>, _>>()?;
        let params = InsertDatapointParams { datapoints };
        let self_module = PyModule::import(this.py(), "uuid")?;
        let uuid = self_module.getattr("UUID")?.unbind();
        pyo3_async_runtimes::tokio::future_into_py(this.py(), async move {
            #[expect(deprecated)]
            let res = client.bulk_insert_datapoints(dataset_name, params).await;
            Python::attach(|py| match res {
                Ok(uuids) => Ok(PyList::new(
                    py,
                    uuids
                        .iter()
                        .map(|id| uuid.call(py, (id.to_string(),), None))
                        .collect::<Result<Vec<_>, _>>()?,
                )?
                .unbind()),
                Err(e) => Err(convert_error(py, e)),
            })
        })
    }

    /// Make a DELETE request to the /datasets/{dataset_name}/datapoints/{datapoint_id} endpoint.
    ///
    /// :param dataset_name: The name of the dataset to delete the datapoint from.
    /// :param datapoint_id: The ID of the datapoint to delete.
    /// :return: None.
    #[pyo3(signature = (*, dataset_name, datapoint_id))]
    fn delete_datapoint<'a>(
        this: PyRef<'a, Self>,
        dataset_name: String,
        datapoint_id: Bound<'a, PyAny>,
    ) -> PyResult<Bound<'a, PyAny>> {
        let client = this.as_super().client.clone();
        let datapoint_id = python_uuid_to_uuid("datapoint_id", datapoint_id)?;
        pyo3_async_runtimes::tokio::future_into_py(this.py(), async move {
            #[expect(deprecated)]
            let res = client.delete_datapoint(dataset_name, datapoint_id).await;
            Python::attach(|py| match res {
                Ok(()) => Ok(()),
                Err(e) => Err(convert_error(py, e)),
            })
        })
    }

    /// Make a GET request to the /datasets/{dataset_name}/datapoints/{datapoint_id} endpoint.
    ///
    /// :param dataset_name: The name of the dataset to get the datapoint from.
    /// :param datapoint_id: The ID of the datapoint to get.
    /// :return: A `Datapoint` object.
    #[pyo3(signature = (*, dataset_name, datapoint_id))]
    fn get_datapoint<'a>(
        this: PyRef<'a, Self>,
        dataset_name: String,
        datapoint_id: Bound<'_, PyAny>,
    ) -> PyResult<Bound<'a, PyAny>> {
        let datapoint_id = python_uuid_to_uuid("datapoint_id", datapoint_id)?;
        let client = this.as_super().client.clone();
        pyo3_async_runtimes::tokio::future_into_py(this.py(), async move {
            #[expect(deprecated)]
            let res = client.get_datapoint(dataset_name, datapoint_id).await;
            Python::attach(|py| match res {
                Ok(wire) => Ok(wire.into_py_any(py)?),
                Err(e) => Err(convert_error(py, e)),
            })
        })
    }

    /// Make a GET request to the /datasets/{dataset_name}/datapoints endpoint.
    ///
    /// :param dataset_name: The name of the dataset to get the datapoints from.
    /// :return: A list of `Datapoint` objects.
    #[pyo3(signature = (*, dataset_name, function_name=None, limit=None, offset=None))]
    fn list_datapoints(
        this: PyRef<'_, Self>,
        dataset_name: String,
        function_name: Option<String>,
        limit: Option<u32>,
        offset: Option<u32>,
    ) -> PyResult<Bound<'_, PyAny>> {
        let client = this.as_super().client.clone();
        pyo3_async_runtimes::tokio::future_into_py(this.py(), async move {
            let request = ListDatapointsRequest {
                function_name,
                limit,
                offset,
                ..Default::default()
            };
            let res = client.list_datapoints(dataset_name, request).await;
            Python::attach(|py| match res {
                Ok(response) => Ok(PyList::new(py, response.datapoints)?.unbind()),
                Err(e) => Err(convert_error(py, e)),
            })
        })
    }

    /// Run a tensorzero Evaluation
    ///
    /// This function is only available in EmbeddedGateway mode.
    ///
    /// # Arguments
    ///
    /// * `evaluation_name` - User chosen name of the evaluation.
    /// * `dataset_name` - The name of the stored dataset to use for variant evaluation
    /// * `variant_name` - The name of the variant to evaluate
    /// * `concurrency` - The maximum number of examples to process in parallel
    /// * `inference_cache` - Cache configuration for inference requests ("on", "off", "read_only", or "write_only")
    #[pyo3(signature = (*,
                        evaluation_name,
                        dataset_name,
                        variant_name,
                        concurrency=1,
                        inference_cache="on".to_string()
    ),
    text_signature = "(self, *, evaluation_name, dataset_name, variant_name, concurrency=1, inference_cache='on')"
    )]
    fn experimental_run_evaluation(
        this: PyRef<'_, Self>,
        evaluation_name: String,
        dataset_name: String,
        variant_name: String,
        concurrency: usize,
        inference_cache: String,
    ) -> PyResult<Bound<'_, PyAny>> {
        let client = this.as_super().client.clone();

        let inference_cache_enum: tensorzero_core::cache::CacheEnabledMode =
            deserialize_from_pyobj(
                this.py(),
                &inference_cache.into_pyobject(this.py())?.into_any(),
            )?;

        pyo3_async_runtimes::tokio::future_into_py(this.py(), async move {
            // Get app state data
            let app_state = client.get_app_state_data().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("Client is not in EmbeddedGateway mode")
            })?;

            let evaluation_run_id = uuid::Uuid::now_v7();

            let core_args = EvaluationCoreArgs {
                tensorzero_client: (*client).clone(),
                clickhouse_client: app_state.clickhouse_connection_info.clone(),
                config: app_state.config.clone(),
                evaluation_name,
                evaluation_run_id,
                dataset_name,
                variant_name,
                concurrency,
                inference_cache: inference_cache_enum,
            };

            let result = run_evaluation_core_streaming(core_args)
                .await
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("Evaluation failed: {e}"))
                })?;

            Python::attach(|py| -> PyResult<Py<PyAny>> {
                let handler = AsyncEvaluationJobHandler {
                    receiver: Arc::new(Mutex::new(result.receiver)),
                    run_info: result.run_info,
                    evaluation_config: result.evaluation_config,
                    evaluation_infos: Arc::new(Mutex::new(Vec::new())),
                    evaluation_errors: Arc::new(Mutex::new(Vec::new())),
                };
                Py::new(py, handler).map(Py::into_any)
            })
        })
    }

    /// Query the Clickhouse database for inferences.
    ///
    /// This function is only available in EmbeddedGateway mode.
    ///
    /// # Arguments
    ///
    /// * `function_name` - The name of the function to query.
    /// * `variant_name` - The name of the variant to query. Optional
    /// * `filters` - A filter tree to apply to the query. Optional
    /// * `output_source` - The source of the output to query. "inference" or "demonstration"
    /// * `limit` - The maximum number of inferences to return. Optional
    /// * `offset` - The offset to start from. Optional
    #[pyo3(signature = (*,
        function_name,
        variant_name=None,
        filters=None,
        output_source="inference".to_string(),
        order_by=None,
        limit=None,
        offset=None
    ),
    text_signature = "(self, *, function_name, variant_name=None, filters=None, output_source='inference', order_by=None, limit=None, offset=None)"
    )]
    // The text_signature is a workaround to weird behavior in pyo3 where the default for an option
    // is written as an ellipsis object.
    #[expect(clippy::too_many_arguments)]
    fn experimental_list_inferences<'a>(
        this: PyRef<'a, Self>,
        function_name: String,
        variant_name: Option<String>,
        filters: Option<Bound<'a, PyAny>>,
        output_source: String,
        order_by: Option<Bound<'a, PyAny>>,
        limit: Option<u64>,
        offset: Option<u64>,
    ) -> PyResult<Bound<'a, PyAny>> {
        let client = this.as_super().client.clone();
        let filters = filters
            .as_ref()
            .map(|x| deserialize_from_pyobj(this.py(), x))
            .transpose()?;
        let order_by: Option<Vec<OrderBy>> = order_by
            .as_ref()
            .map(|x| deserialize_from_pyobj(this.py(), x))
            .transpose()?;
        let output_source =
            output_source
                .as_str()
                .try_into()
                .map_err(|e: tensorzero_core::error::Error| {
                    convert_error(this.py(), TensorZeroError::Other { source: e.into() })
                })?;
        pyo3_async_runtimes::tokio::future_into_py(this.py(), async move {
            let params = ListInferencesParams {
                function_name: Some(&function_name),
                variant_name: variant_name.as_deref(),
                filters: filters.as_ref(),
                output_source,
                order_by: order_by.as_deref(),
                limit,
                offset,
                ..Default::default()
            };
            let res = client.experimental_list_inferences(params).await;
            Python::attach(|py| match res {
                Ok(wire_inferences) => Ok(PyList::new(py, wire_inferences)?.unbind()),
                Err(e) => Err(convert_error(py, e)),
            })
        })
    }

    /// Render a list of stored inferences into a list of rendered stored inferences.
    /// There are two things that need to happen in this function:
    /// 1. We need to resolve all network resources (e.g. images) in the stored inferences.
    /// 2. We need to prepare all messages into "simple" messages that have been templated for a particular variant.
    ///    To do this, we need to know what variant to use for each function that might appear in the data.
    ///
    /// IMPORTANT: For now, this function drops datapoints which are bad, e.g. ones where templating fails, the function
    ///            has no variant specified, or where the process of downloading resources fails.
    ///            In future we will make this behavior configurable by the caller.
    ///
    /// :param stored_inferences: A list of stored inferences to render.
    /// :param variants: A map from function name to variant name.
    /// :return: A list of rendered stored inferences.
    /// DEPRECATED: use `experimental_render_samples` instead.
    ///
    /// Renders stored inferences using the templates of the specified variants.
    ///
    /// Warning: This API is experimental and may change without notice. For now
    ///          we discard inferences where the input references a static tool that
    ///          has no variant specified, or where the process of downloading resources fails.
    ///          In future we will make this behavior configurable by the caller.
    ///
    /// :param stored_inferences: A list of stored inferences to render.
    /// :param variants: A map from function name to variant name.
    /// :return: A list of rendered stored inferences.
    #[pyo3(signature = (*, stored_inferences, variants))]
    fn experimental_render_inferences<'a>(
        this: PyRef<'a, Self>,
        stored_inferences: Vec<Bound<'a, PyAny>>,
        variants: HashMap<String, String>,
    ) -> PyResult<Bound<'a, PyAny>> {
        tracing::warn!("experimental_render_inferences is deprecated. Use experimental_render_samples instead. See https://github.com/tensorzero/tensorzero/issues/2675");
        let client = this.as_super().client.clone();
        let config = client.config().ok_or_else(|| {
            PyValueError::new_err(
                "Config not available in HTTP gateway mode. Use embedded mode for render_samples.",
            )
        })?;
        // Enter the Tokio runtime context while still holding the GIL
        // This is needed because deserialize_from_stored_sample may use tokio::spawn internally
        // for JSON schema compilation
        // TODO (#4259): remove the tokio spawn from that function and remove this guard.
        let _guard = pyo3_async_runtimes::tokio::get_runtime().enter();
        let stored_inferences = stored_inferences
            .iter()
            .map(|x| deserialize_from_stored_sample(this.py(), x, config))
            .collect::<Result<Vec<_>, _>>()?;
        pyo3_async_runtimes::tokio::future_into_py(this.py(), async move {
            let res = client
                .experimental_render_samples(stored_inferences, variants)
                .await;
            Python::attach(|py| match res {
                Ok(inferences) => Ok(PyList::new(py, inferences)?.unbind()),
                Err(e) => Err(convert_error(py, e)),
            })
        })
    }

    /// Render a list of stored samples into a list of rendered stored samples.
    ///
    /// This function performs two main tasks:
    /// 1. Resolves all network resources (e.g., images) in the stored samples.
    /// 2. Prepares all messages into "simple" messages that have been templated for a particular variant.
    ///    To do this, the function needs to know which variant to use for each function that might appear in the data.
    ///
    /// IMPORTANT: For now, this function drops datapoints that are invalid, such as those where templating fails,
    /// the function has no variant specified, or the process of downloading resources fails.
    /// In the future, this behavior may be made configurable by the caller.
    ///
    /// :param stored_samples: A list of stored samples to render.
    /// :param variants: A mapping from function name to variant name.
    /// :return: A list of rendered samples.
    #[pyo3(signature = (*, stored_samples, variants))]
    fn experimental_render_samples<'a>(
        this: PyRef<'a, Self>,
        stored_samples: Vec<Bound<'a, PyAny>>,
        variants: HashMap<String, String>,
    ) -> PyResult<Bound<'a, PyAny>> {
        let client = this.as_super().client.clone();
        let config = client.config().ok_or_else(|| {
            PyValueError::new_err(
                "Config not available in HTTP gateway mode. Use embedded mode for render_samples.",
            )
        })?;
        // Enter the Tokio runtime context while still holding the GIL
        // This is needed because deserialize_from_stored_sample may use tokio::spawn internally
        // for JSON schema compilation
        // TODO (#4259): remove the tokio spawn from that function and remove this guard.
        let _guard = pyo3_async_runtimes::tokio::get_runtime().enter();
        let stored_samples = stored_samples
            .iter()
            .map(|x| deserialize_from_stored_sample(this.py(), x, config))
            .collect::<Result<Vec<_>, _>>()?;
        pyo3_async_runtimes::tokio::future_into_py(this.py(), async move {
            let res = client
                .experimental_render_samples(stored_samples, variants)
                .await;
            Python::attach(|py| match res {
                Ok(samples) => Ok(PyList::new(py, samples)?.unbind()),
                Err(e) => Err(convert_error(py, e)),
            })
        })
    }

    /// Launch an optimization job.
    ///
    /// :param train_samples: A list of RenderedSample objects that will be used for training.
    /// :param val_samples: A list of RenderedSample objects that will be used for validation.
    /// :param optimiztion_config: The optimization config.
    /// :return: A `OptimizerJobHandle` object that can be used to poll the optimization job.
    #[pyo3(signature = (*, train_samples, val_samples=None, optimization_config))]
    fn experimental_launch_optimization<'a>(
        this: PyRef<'a, Self>,
        train_samples: Vec<Bound<'a, PyAny>>,
        val_samples: Option<Vec<Bound<'a, PyAny>>>,
        optimization_config: Bound<'a, PyAny>,
    ) -> PyResult<Bound<'a, PyAny>> {
        let client = this.as_super().client.clone();
        let train_samples = train_samples
            .iter()
            .map(|x| deserialize_from_rendered_sample(this.py(), x))
            .collect::<Result<Vec<_>, _>>()?;
        let val_samples = val_samples
            .as_ref()
            .map(|x| {
                x.iter()
                    .map(|x| deserialize_from_rendered_sample(this.py(), x))
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()?;
        let optimizer_config = deserialize_optimization_config(&optimization_config)?;
        pyo3_async_runtimes::tokio::future_into_py(this.py(), async move {
            let res = client
                .experimental_launch_optimization(LaunchOptimizationParams {
                    train_samples,
                    val_samples,
                    optimization_config: UninitializedOptimizerInfo {
                        inner: optimizer_config,
                    },
                })
                .await;
            match res {
                Ok(job_handle) => Ok(job_handle),
                Err(e) => Python::attach(|py| Err(convert_error(py, e))),
            }
        })
    }

    /// Poll an optimization job.
    ///
    /// :param job_handle: The job handle returned by `experimental_launch_optimization`.
    /// :return: An `OptimizerStatus` object.
    #[pyo3(signature = (*, job_handle))]
    fn experimental_poll_optimization(
        this: PyRef<'_, Self>,
        job_handle: OptimizationJobHandle,
    ) -> PyResult<Bound<'_, PyAny>> {
        let client = this.as_super().client.clone();
        pyo3_async_runtimes::tokio::future_into_py(this.py(), async move {
            let res = client.experimental_poll_optimization(&job_handle).await;
            match res {
                Ok(status) => Ok(OptimizationJobInfoPyClass::new(status)),
                Err(e) => Python::attach(|py| Err(convert_error(py, e))),
            }
        })
    }
}

#[expect(unknown_lints)]
// This lint currently does nothing on stable, but let's include it
// so that it will start working automatically when it's stabilized
#[deny(non_exhaustive_omitted_patterns)]
pub fn convert_error(py: Python<'_>, e: TensorZeroError) -> PyErr {
    match e {
        TensorZeroError::Http {
            status_code,
            text,
            source: _,
        } => tensorzero_error(py, status_code, text).unwrap_or_else(|e| e),
        TensorZeroError::Other { source } => {
            tensorzero_core_error(py, &source.to_string()).unwrap_or_else(|e| e)
        }
        TensorZeroError::RequestTimeout => {
            tensorzero_core_error(py, &e.to_string()).unwrap_or_else(|e| e)
        }
        // Required due to the `#[non_exhaustive]` attribute on `TensorZeroError` - we want to force
        // downstream consumers to handle all possible error types, but the compiler also requires us
        // to do this (since our python bindings are in a different crate from the Rust client.)
        _ => tensorzero_core_error(py, &format!("Unexpected TensorZero error: {e:?}"))
            .unwrap_or_else(|e| e),
    }
}

fn tensorzero_error(py: Python<'_>, status_code: u16, text: Option<String>) -> PyResult<PyErr> {
    Ok(PyErr::from_value(
        tensorzero_error_class(py)?
            .bind(py)
            .call1((status_code, text))?,
    ))
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
