#![recursion_limit = "256"]
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

use evaluations::{
    ClientInferenceExecutor, EvaluationCoreArgs, EvaluationFunctionConfig,
    EvaluationFunctionConfigTable, EvaluationVariant, run_evaluation_core_streaming,
};
use futures::StreamExt;
use pyo3::{
    IntoPyObjectExt,
    exceptions::{PyDeprecationWarning, PyStopAsyncIteration, PyStopIteration, PyValueError},
    ffi::c_str,
    prelude::*,
    types::{PyDict, PyList, PyString, PyType},
};
use python_helpers::{
    convert_response_to_python_dataclass, parse_feedback_response, parse_inference_chunk,
    parse_inference_response, parse_tool, parse_workflow_evaluation_run_episode_response,
    parse_workflow_evaluation_run_response, python_uuid_to_uuid,
};

use crate::gil_helpers::in_tokio_runtime_no_gil;
use tensorzero_core::{
    config::{ConfigPyClass, FunctionsConfigPyClass, UninitializedVariantInfo},
    db::clickhouse::query_builder::OrderBy,
    function::{FunctionConfigChatPyClass, FunctionConfigJsonPyClass, VariantsConfigPyClass},
    inference::types::{
        ResolvedInput, ResolvedInputMessage,
        pyo3_helpers::{
            JSON_DUMPS, JSON_LOADS, deserialize_from_pyobj, deserialize_from_rendered_sample,
            deserialize_from_stored_sample, deserialize_optimization_config, serialize_to_dict,
            tensorzero_error,
        },
    },
    optimization::{
        OptimizationJobInfoPyClass, OptimizationJobStatus, UninitializedOptimizerInfo,
        dicl::UninitializedDiclOptimizationConfig, fireworks_sft::UninitializedFireworksSFTConfig,
        gcp_vertex_gemini_sft::UninitializedGCPVertexGeminiSFTConfig,
        gepa::UninitializedGEPAConfig, openai_rft::UninitializedOpenAIRFTConfig,
        openai_sft::UninitializedOpenAISFTConfig, together_sft::UninitializedTogetherSFTConfig,
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
    CacheParamsOptions, Client, ClientBuilder, ClientBuilderMode, ClientExt, ClientInferenceParams,
    ClientSecretString, Datapoint, DynamicToolParams, FeedbackParams, InferenceOutput,
    InferenceParams, InferenceStream, Input, LaunchOptimizationParams, ListDatapointsRequest,
    ListInferencesParams, OptimizationJobHandle, PostgresConfig, RenderedSample, StoredInference,
    TensorZeroError, Tool, WorkflowEvaluationRunParams, err_to_http, observability::LogFormat,
};
use tokio::sync::Mutex;
use url::Url;
use uuid::Uuid;

mod evaluation_handlers;
mod gil_helpers;
mod python_helpers;

use crate::evaluation_handlers::{AsyncEvaluationJobHandler, EvaluationJobHandler};
use crate::gil_helpers::{DropInTokio, tokio_block_on_without_gil};

#[pymodule]
fn tensorzero(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Eagerly load the exceptions, so that we don't trigger
    // a nested exception when calling `convert_error` below
    let _ = m.py().get_type::<tensorzero_error::TensorZeroError>();
    let _ = m
        .py()
        .get_type::<tensorzero_error::TensorZeroInternalError>();
    // Otel is disabled for now in the Python client until we decide how it should be configured
    // We might have produced an error when trying to construct the (not yet enabled) OTEL layer,
    // which will just get ignored here. The HTTP gateway will handle that error, as that's
    // the only place where we actually try to enable OTEL.
    let _delayed_enable = tokio_block_on_without_gil(
        m.py(),
        tensorzero_rust::observability::setup_observability(LogFormat::Pretty, false),
    )
    .map_err(|e| convert_error(m.py(), TensorZeroError::Other { source: e.into() }))?;
    m.add_class::<BaseTensorZeroGateway>()?;
    m.add_class::<AsyncTensorZeroGateway>()?;
    m.add_class::<TensorZeroGateway>()?;
    m.add_class::<LocalHttpGateway>()?;
    m.add_class::<RenderedSample>()?;
    m.add_class::<EvaluationJobHandler>()?;
    m.add_class::<AsyncEvaluationJobHandler>()?;
    m.add_class::<UninitializedOpenAIRFTConfig>()?;
    m.add_class::<UninitializedOpenAISFTConfig>()?;
    m.add_class::<UninitializedFireworksSFTConfig>()?;
    m.add_class::<UninitializedDiclOptimizationConfig>()?;
    m.add_class::<UninitializedGCPVertexGeminiSFTConfig>()?;
    m.add_class::<UninitializedGEPAConfig>()?;
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
#[pyo3(signature = (*, config_file, clickhouse_url, postgres_url, valkey_url, async_setup))]
fn _start_http_gateway(
    py: Python<'_>,
    config_file: Option<String>,
    clickhouse_url: Option<String>,
    postgres_url: Option<String>,
    valkey_url: Option<String>,
    async_setup: bool,
) -> PyResult<Bound<'_, PyAny>> {
    warn_no_config(py, config_file.as_deref())?;
    let gateway_fut = async move {
        let (addr, handle) = tensorzero_core::utils::gateway::start_openai_compatible_gateway(
            config_file,
            clickhouse_url,
            postgres_url,
            valkey_url,
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
    // Note - `Client` is cloneable, so we don't wrap in `DropInTokio`
    // Instead, the stored `GatewayHandle` has customizable drop behavior,
    // which we configure with `.with_drop_wrapper` when we build an embedded gateway for PyO3
    client: Client,
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

const DEFAULT_INFERENCE_QUERY_LIMIT: u32 = 20;

#[pymethods]
impl BaseTensorZeroGateway {
    #[pyo3(signature = (*, input, function_name=None, model_name=None, episode_id=None, stream=None, params=None, variant_name=None, dryrun=None, output_schema=None, allowed_tools=None, provider_tools=None, additional_tools=None, tool_choice=None, parallel_tool_calls=None, internal=None, tags=None, credentials=None, cache_options=None, extra_body=None, extra_headers=None, include_original_response=None, include_raw_response=None, include_raw_usage=None, otlp_traces_extra_headers=None, otlp_traces_extra_attributes=None, otlp_traces_extra_resources=None, internal_dynamic_variant_config=None))]
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
        include_raw_response: Option<bool>,
        include_raw_usage: Option<bool>,
        otlp_traces_extra_headers: Option<HashMap<String, String>>,
        otlp_traces_extra_attributes: Option<HashMap<String, String>>,
        otlp_traces_extra_resources: Option<HashMap<String, String>>,
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
            include_raw_response.unwrap_or(false),
            include_raw_usage.unwrap_or(false),
            otlp_traces_extra_headers,
            otlp_traces_extra_attributes,
            otlp_traces_extra_resources,
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
        include_raw_response: bool,
        include_raw_usage: bool,
        otlp_traces_extra_headers: Option<HashMap<String, String>>,
        otlp_traces_extra_attributes: Option<HashMap<String, String>>,
        otlp_traces_extra_resources: Option<HashMap<String, String>>,
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
                    .map(|key_vals| parse_tool(py, key_vals).map(Tool::Function))
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

        let input: Input = deserialize_from_pyobj(py, &input)?;

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
                provider_tools: provider_tools.unwrap_or_default(),
            },
            input,
            credentials: credentials.unwrap_or_default(),
            cache_options: cache_options.unwrap_or_default(),
            output_schema,
            include_original_response,
            include_raw_response,
            include_raw_usage,
            extra_body,
            extra_headers,
            internal_dynamic_variant_config,
            otlp_traces_extra_headers: otlp_traces_extra_headers.unwrap_or_default(),
            otlp_traces_extra_attributes: otlp_traces_extra_attributes.unwrap_or_default(),
            otlp_traces_extra_resources: otlp_traces_extra_resources.unwrap_or_default(),
            api_key: None,
        })
    }
}

/// Helper function to construct an EvaluationVariant from the optional variant_name and internal_dynamic_variant_config parameters.
/// Deserializes the internal_dynamic_variant_config if provided and validates that exactly one of the two is provided.
fn construct_evaluation_variant(
    py: Python<'_>,
    internal_dynamic_variant_config: Option<&Bound<'_, PyDict>>,
    variant_name: Option<String>,
) -> PyResult<EvaluationVariant> {
    // Deserialize internal_dynamic_variant_config if provided
    let internal_dynamic_variant_config: Option<UninitializedVariantInfo> =
        if let Some(config) = internal_dynamic_variant_config {
            Some(deserialize_from_pyobj(py, config)?)
        } else {
            None
        };

    match (internal_dynamic_variant_config, variant_name) {
        (Some(info), None) => Ok(EvaluationVariant::Info(Box::new(info))),
        (None, Some(name)) => Ok(EvaluationVariant::Name(name)),
        (None, None) => Err(PyValueError::new_err(
            "Either `variant_name` or `internal_dynamic_variant_config` must be provided.",
        )),
        (Some(_), Some(_)) => Err(PyValueError::new_err(
            "Cannot specify both `variant_name` and `internal_dynamic_variant_config`. \
            When using a dynamic variant, provide only `internal_dynamic_variant_config`.",
        )),
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
                return Err(tensorzero_error::TensorZeroInternalError::new_err(format!(
                    "Failed to construct TensorZero client: {e:?}"
                )));
            }
        };
        let instance = PyClassInitializer::from(BaseTensorZeroGateway { client })
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
    #[pyo3(signature = (*, config_file=None, clickhouse_url=None, postgres_url=None, valkey_url=None, timeout=None))]
    /// Initialize the TensorZero client, using an embedded gateway.
    /// This connects to ClickHouse (if provided) and runs DB migrations.
    ///
    /// :param config_file: The path to the TensorZero configuration file. Example: "tensorzero.toml"
    /// :param clickhouse_url: The URL of the ClickHouse instance to use for the gateway. If observability is disabled in the config, this can be `None`
    /// :param postgres_url: The URL of the PostgreSQL instance to use for rate limiting.
    /// :param valkey_url: The URL of the Valkey instance to use for rate limiting.
    /// :param timeout: The timeout for embedded gateway request processing, in seconds. If this timeout is hit, any in-progress LLM requests may be aborted. If not provided, no timeout will be set.
    /// :return: A `TensorZeroGateway` instance configured to use an embedded gateway.
    fn build_embedded(
        cls: &Bound<'_, PyType>,
        config_file: Option<&str>,
        clickhouse_url: Option<String>,
        postgres_url: Option<String>,
        valkey_url: Option<String>,
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
            postgres_config: postgres_url.map(PostgresConfig::Url),
            valkey_url,
            timeout,
            verify_credentials: true,
            allow_batch_writes: false,
        })
        // When the underlying `GatewayHandle` is dropped, we need to be in the Tokio runtime
        // with the GIL released (since we might block on the ClickHouse batcher shutting down)
        .with_drop_wrapper(in_tokio_runtime_no_gil)
        .build();
        let client = tokio_block_on_without_gil(cls.py(), client_fut);
        let client = match client {
            Ok(client) => client,
            Err(e) => {
                return Err(tensorzero_error::TensorZeroInternalError::new_err(format!(
                    "Failed to construct TensorZero client: {e:?}"
                )));
            }
        };
        // Construct an instance of `TensorZeroGateway` (while providing the fields from the `BaseTensorZeroGateway` superclass).
        let instance = PyClassInitializer::from(BaseTensorZeroGateway { client })
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

    #[pyo3(signature = (*, input, function_name=None, model_name=None, episode_id=None, stream=None, params=None, variant_name=None, dryrun=None, output_schema=None, allowed_tools=None, additional_tools=None, provider_tools=None, tool_choice=None, parallel_tool_calls=None, internal=None, tags=None, credentials=None, cache_options=None, extra_body=None, extra_headers=None, include_original_response=None, include_raw_response=None, include_raw_usage=None, otlp_traces_extra_headers=None, otlp_traces_extra_attributes=None, otlp_traces_extra_resources=None, internal_dynamic_variant_config=None))]
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
    /// :param include_raw_usage: If set, include raw provider-specific usage data in the response.
    /// :param otlp_traces_extra_headers: If set, attaches custom HTTP headers to OTLP trace exports for this request.
    ///                                   Headers will be automatically prefixed with "tensorzero-otlp-traces-extra-header-".
    ///                                   Example: {"My-Header": "My-Value"} becomes header "tensorzero-otlp-traces-extra-header-My-Header: My-Value"
    /// :param otlp_traces_extra_attributes: If set, attaches custom HTTP headers to OTLP trace exports for this request.
    ///                                      Headers will be automatically prefixed with "tensorzero-otlp-traces-extra-attributes-".
    ///                                      Example: {"My-Attribute": "My-Value"} becomes header "tensorzero-otlp-traces-extra-attribute-My-Attribute: My-Value"
    /// :param otlp_traces_extra_resources: If set, attaches custom HTTP headers to OTLP trace exports for this request.
    ///                                     Headers will be automatically prefixed with "tensorzero-otlp-traces-extra-resources-".
    ///                                     Example: {"My-Resource": "My-Value"} becomes header "tensorzero-otlp-traces-extra-resource-My-Resource: My-Value"
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
        include_raw_response: Option<bool>,
        include_raw_usage: Option<bool>,
        otlp_traces_extra_headers: Option<HashMap<String, String>>,
        otlp_traces_extra_attributes: Option<HashMap<String, String>>,
        otlp_traces_extra_resources: Option<HashMap<String, String>>,
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
            include_raw_response.unwrap_or(false),
            include_raw_usage.unwrap_or(false),
            otlp_traces_extra_headers,
            otlp_traces_extra_attributes,
            otlp_traces_extra_resources,
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
    #[pyo3(warn(message = "Please use `create_datapoints` instead of `create_datapoints_legacy`. In a future release, `create_datapoints_legacy` will be removed.", category = PyDeprecationWarning))]
    fn create_datapoints_legacy(
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
    #[pyo3(warn(message = "Please use `delete_datapoints` instead of `delete_datapoint`. In a future release, `delete_datapoint` will be removed.", category = PyDeprecationWarning))]
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
    #[pyo3(warn(message = "Please use `get_datapoints` instead of `get_datapoint`. In a future release, `get_datapoint` will be removed.", category = PyDeprecationWarning))]
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

    /// DEPRECATED: Use `list_datapoints` instead.
    ///
    /// Make a GET request to the /datasets/{dataset_name}/datapoints endpoint.
    ///
    /// :param dataset_name: The name of the dataset to get the datapoints from.
    #[pyo3(signature = (*, dataset_name, function_name=None, limit=None, offset=None))]
    #[pyo3(warn(message = "Please use `list_datapoints` instead of `list_datapoints_legacy`. In a future release, `list_datapoints_legacy` will be removed.", category = PyDeprecationWarning))]
    fn list_datapoints_legacy(
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

    /// Create one or more datapoints in a dataset.
    ///
    /// :param dataset_name: The name of the dataset to insert the datapoints into.
    /// :param requests: A list of `CreateDatapointRequest` objects.
    /// :return: A `CreateDatapointsResponse` object containing the IDs of the newly-created datapoints.
    #[pyo3(signature = (*, dataset_name, requests))]
    fn create_datapoints(
        this: PyRef<'_, Self>,
        dataset_name: String,
        requests: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let client = this.as_super().client.clone();
        let requests = requests
            .iter()
            .map(|dp| deserialize_from_pyobj(this.py(), dp))
            .collect::<Result<Vec<_>, _>>()?;
        let fut = client.create_datapoints(dataset_name, requests);
        let response =
            tokio_block_on_without_gil(this.py(), fut).map_err(|e| convert_error(this.py(), e))?;
        convert_response_to_python_dataclass(
            this.py(),
            &response,
            "tensorzero",
            "CreateDatapointsResponse",
        )
    }

    /// Update one or more datapoints in a dataset.
    ///
    /// :param dataset_name: The name of the dataset containing the datapoints to update.
    /// :param requests: A list of `UpdateDatapointRequest`` objects.
    /// :return: An `UpdateDatapointsResponse` object.
    #[pyo3(signature = (*, dataset_name, requests))]
    fn update_datapoints(
        this: PyRef<'_, Self>,
        dataset_name: String,
        requests: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let client = this.as_super().client.clone();
        let requests = requests
            .iter()
            .map(|dp| deserialize_from_pyobj(this.py(), dp))
            .collect::<Result<Vec<_>, _>>()?;
        let fut = client.update_datapoints(dataset_name, requests);
        let response =
            tokio_block_on_without_gil(this.py(), fut).map_err(|e| convert_error(this.py(), e))?;
        convert_response_to_python_dataclass(
            this.py(),
            &response,
            "tensorzero",
            "UpdateDatapointsResponse",
        )
    }

    /// Get specific datapoints by their IDs.
    ///
    /// :param ids: A list of datapoint IDs to retrieve.
    /// :return: A `GetDatapointsResponse` object.
    #[pyo3(signature = (*, ids, dataset_name = None))]
    fn get_datapoints(
        this: PyRef<'_, Self>,
        ids: Vec<Bound<'_, PyAny>>,
        dataset_name: Option<String>,
    ) -> PyResult<Py<PyAny>> {
        if dataset_name.is_none() {
            let warnings = PyModule::import(this.py(), "warnings")?;
            warnings.call_method1(
                "warn",
                (
                    "Calling get_datapoints without a dataset name is deprecated. Please provide a dataset name for performance reasons.",
                    this.py().get_type::<PyDeprecationWarning>(),
                ),
            )?;
        }

        let client = this.as_super().client.clone();
        let ids: Vec<uuid::Uuid> = ids
            .iter()
            .map(|id| {
                let id_str: String = id.extract()?;
                uuid::Uuid::parse_str(&id_str)
                    .map_err(|e| PyErr::new::<PyValueError, _>(format!("Invalid UUID: {e}")))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let fut = client.get_datapoints(dataset_name, ids);
        let response =
            tokio_block_on_without_gil(this.py(), fut).map_err(|e| convert_error(this.py(), e))?;
        convert_response_to_python_dataclass(
            this.py(),
            &response,
            "tensorzero",
            "GetDatapointsResponse",
        )
    }

    /// Update metadata for one or more datapoints.
    ///
    /// :param dataset_name: The name of the dataset containing the datapoints.
    /// :param datapoints: A list of `UpdateDatapointMetadataRequest` objects.
    /// :return: An `UpdateDatapointsResponse` object containing the IDs of updated datapoints.
    #[pyo3(signature = (*, dataset_name, requests))]
    fn update_datapoints_metadata(
        this: PyRef<'_, Self>,
        dataset_name: String,
        requests: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let client = this.as_super().client.clone();
        let requests = requests
            .iter()
            .map(|dp| deserialize_from_pyobj(this.py(), dp))
            .collect::<Result<Vec<_>, _>>()?;
        let fut = client.update_datapoints_metadata(dataset_name, requests);
        let response =
            tokio_block_on_without_gil(this.py(), fut).map_err(|e| convert_error(this.py(), e))?;
        convert_response_to_python_dataclass(
            this.py(),
            &response,
            "tensorzero",
            "UpdateDatapointsResponse",
        )
    }

    /// Delete multiple datapoints from a dataset.
    ///
    /// :param dataset_name: The name of the dataset to delete datapoints from.
    /// :param ids: A list of datapoint IDs to delete.
    /// :return: A `DeleteDatapointsResponse` object containing the number of deleted datapoints.
    #[pyo3(signature = (*, dataset_name, ids))]
    fn delete_datapoints(
        this: PyRef<'_, Self>,
        dataset_name: String,
        ids: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let client = this.as_super().client.clone();
        let ids: Vec<uuid::Uuid> = ids
            .iter()
            .map(|id| {
                let id_str: String = id.extract()?;
                uuid::Uuid::parse_str(&id_str)
                    .map_err(|e| PyErr::new::<PyValueError, _>(format!("Invalid UUID: {e}")))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let fut = client.delete_datapoints(dataset_name, ids);
        let response =
            tokio_block_on_without_gil(this.py(), fut).map_err(|e| convert_error(this.py(), e))?;
        convert_response_to_python_dataclass(
            this.py(),
            &response,
            "tensorzero",
            "DeleteDatapointsResponse",
        )
    }

    /// Delete an entire dataset.
    ///
    /// :param dataset_name: The name of the dataset to delete.
    /// :return: A `DeleteDatapointsResponse` object containing the number of deleted datapoints.
    #[pyo3(signature = (*, dataset_name))]
    fn delete_dataset(this: PyRef<'_, Self>, dataset_name: String) -> PyResult<Py<PyAny>> {
        let client = this.as_super().client.clone();
        let fut = client.delete_dataset(dataset_name);
        let response =
            tokio_block_on_without_gil(this.py(), fut).map_err(|e| convert_error(this.py(), e))?;
        convert_response_to_python_dataclass(
            this.py(),
            &response,
            "tensorzero",
            "DeleteDatapointsResponse",
        )
    }

    /// Create datapoints from inferences.
    ///
    /// :param dataset_name: The name of the dataset to create datapoints in.
    /// :param params: The parameters specifying which inferences to convert to datapoints.
    ///                 For InferenceIds: pass `{"type": "inference_ids", "inference_ids": [...], "output_source": "inference"}`
    ///                 For InferenceQuery: pass `{"type": "inference_query", "function_name": "...", "output_source": "inference", ...}`
    /// :param output_source: The source of the output to create datapoints from. "none", "inference", or "demonstration".
    ///                       Can also be specified inside `params.output_source`. If both are provided, an error is raised.
    /// :return: A list of UUIDs of the created datapoints.
    #[pyo3(signature = (*, dataset_name, params, output_source=None))]
    fn create_datapoints_from_inferences(
        this: PyRef<'_, Self>,
        dataset_name: String,
        params: Bound<'_, PyAny>,
        output_source: Option<String>,
    ) -> PyResult<Py<PyAny>> {
        let client = this.as_super().client.clone();

        // Handle output_source: can be passed as parameter or inside params, but not both
        if let Some(source) = &output_source {
            let existing = params.getattr("output_source").ok();
            if let Some(existing) = existing
                && !existing.is_none()
            {
                return Err(PyValueError::new_err(
                    "You must specify `output_source` either at the root or inside the `params` parameter but not both.",
                ));
            }
            params.setattr("output_source", source)?;
        }

        let params = deserialize_from_pyobj(this.py(), &params)?;

        let fut = client.create_datapoints_from_inferences(dataset_name, params);
        let response =
            tokio_block_on_without_gil(this.py(), fut).map_err(|e| convert_error(this.py(), e))?;
        convert_response_to_python_dataclass(
            this.py(),
            &response,
            "tensorzero",
            "CreateDatapointsResponse",
        )
    }

    /// List datapoints in a dataset.
    ///
    /// :param dataset_name: The name of the dataset to list datapoints from.
    /// :param request: The request parameters.
    /// :return: A `GetDatapointsResponse` object.
    #[pyo3(signature = (*, dataset_name, request))]
    fn list_datapoints(
        this: PyRef<'_, Self>,
        dataset_name: String,
        request: Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let client = this.as_super().client.clone();
        let request = deserialize_from_pyobj(this.py(), &request)?;

        let res = client.list_datapoints(dataset_name, request);
        let response =
            tokio_block_on_without_gil(this.py(), res).map_err(|e| convert_error(this.py(), e))?;
        convert_response_to_python_dataclass(
            this.py(),
            &response,
            "tensorzero",
            "GetDatapointsResponse",
        )
    }

    /// Run a tensorzero Evaluation
    ///
    /// This function is only available in EmbeddedGateway mode.
    ///
    /// # Arguments
    ///
    /// * `evaluation_name` - User chosen name of the evaluation.
    /// * `dataset_name` - The name of the stored dataset to use for variant evaluation
    /// * `variant_name` - Optional name of the variant to evaluate
    /// * `concurrency` - The maximum number of examples to process in parallel
    /// * `inference_cache` - Cache configuration for inference requests ("on", "off", "read_only", or "write_only")
    /// * `internal_dynamic_variant_config` - Optional dynamic variant configuration [INTERNAL: This field is unstable and may change without notice.]
    /// * `max_datapoints` - Optional maximum number of datapoints to evaluate from the dataset
    /// * `adaptive_stopping` - Optional dict configuring adaptive stopping behavior for evals.
    ///                         Example for two evaluators named "exact_match" and "llm_judge":
    ///                           `{"precision": {"exact_match": 0.2, "llm_judge": 0.15}}`
    ///                         The "precision" field maps evaluator names to confidence interval half-widths.
    ///                         Evaluation for a given evaluator stops when it achieves its precision target,
    ///                         i.e. the width of the larger of the two halves of its confidence interval
    ///                         is <= the precision target.
    #[pyo3(signature = (*,
                        evaluation_name,
                        dataset_name=None,
                        datapoint_ids=None,
                        variant_name=None,
                        concurrency=1,
                        inference_cache="on".to_string(),
                        internal_dynamic_variant_config=None,
                        max_datapoints=None,
                        adaptive_stopping=None
    ),
    text_signature = "(self, *, evaluation_name, dataset_name=None, datapoint_ids=None, variant_name=None, concurrency=1, inference_cache='on', internal_dynamic_variant_config=None, max_datapoints=None, adaptive_stopping=None)"
    )]
    #[expect(clippy::too_many_arguments)]
    fn experimental_run_evaluation(
        this: PyRef<'_, Self>,
        evaluation_name: String,
        dataset_name: Option<String>,
        datapoint_ids: Option<Vec<String>>,
        variant_name: Option<String>,
        concurrency: usize,
        inference_cache: String,
        internal_dynamic_variant_config: Option<&Bound<'_, PyDict>>,
        max_datapoints: Option<u32>,
        adaptive_stopping: Option<&Bound<'_, PyDict>>,
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

        let variant =
            construct_evaluation_variant(this.py(), internal_dynamic_variant_config, variant_name)?;

        // Parse adaptive_stopping config from Python dict
        let precision_targets_map = if let Some(adaptive_stopping_dict) = adaptive_stopping {
            // Extract the "precision" field from adaptive_stopping dict
            if let Ok(Some(precision_bound)) = adaptive_stopping_dict.get_item("precision") {
                let precision_dict_bound = precision_bound.downcast::<PyDict>().map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                        "adaptive_stopping['precision'] must be a dictionary",
                    )
                })?;

                let mut map = std::collections::HashMap::new();
                for (key, value) in precision_dict_bound.iter() {
                    let key_str: String = key.extract()?;
                    let value_f64: f64 = value.extract()?;
                    map.insert(key_str, value_f64 as f32);
                }
                map
            } else {
                HashMap::new()
            }
        } else {
            HashMap::new()
        };

        // Parse datapoint_ids from strings to UUIDs (keeping as Option)
        let datapoint_ids: Option<Vec<Uuid>> = datapoint_ids
            .map(|ids| {
                ids.iter()
                    .map(|s| {
                        Uuid::parse_str(s).map_err(|e| {
                            pyo3::exceptions::PyValueError::new_err(format!(
                                "Invalid UUID in datapoint_ids: {e}"
                            ))
                        })
                    })
                    .collect::<PyResult<Vec<Uuid>>>()
            })
            .transpose()?;

        // Extract evaluation config from app_state
        let evaluation_config = app_state
            .config
            .evaluations
            .get(&evaluation_name)
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "evaluation '{evaluation_name}' not found"
                ))
            })?
            .clone();

        // Build function configs table from all functions in the config
        let function_configs: EvaluationFunctionConfigTable = app_state
            .config
            .functions
            .iter()
            .map(|(name, func)| (name.clone(), EvaluationFunctionConfig::from(func.as_ref())))
            .collect();
        let function_configs = Arc::new(function_configs);

        // Wrap the client in ClientInferenceExecutor for use with evaluations
        let inference_executor = Arc::new(ClientInferenceExecutor::new(client.clone()));

        let core_args = EvaluationCoreArgs {
            inference_executor,
            clickhouse_client: app_state.clickhouse_connection_info.clone(),
            evaluation_config,
            function_configs,
            evaluation_name,
            evaluation_run_id,
            dataset_name,
            datapoint_ids,
            variant,
            concurrency,
            inference_cache: inference_cache_enum,
            tags: HashMap::new(), // No external tags for Python client evaluations
        };

        let result = tokio_block_on_without_gil(
            this.py(),
            run_evaluation_core_streaming(core_args, max_datapoints, precision_targets_map),
        )
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
    #[pyo3(warn(message = "Please use `list_inferences` instead of `experimental_list_inferences`. In a future release, `experimental_list_inferences` will be removed.", category = PyDeprecationWarning))]
    // The text_signature is a workaround to weird behavior in pyo3 where the default for an option
    // is written as an ellipsis object.
    #[expect(clippy::too_many_arguments)]
    #[expect(deprecated)]
    fn experimental_list_inferences(
        this: PyRef<'_, Self>,
        function_name: String,
        variant_name: Option<String>,
        filters: Option<Bound<'_, PyAny>>,
        output_source: String,
        order_by: Option<Bound<'_, PyAny>>,
        limit: Option<u32>,
        offset: Option<u32>,
    ) -> PyResult<Py<PyList>> {
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
            limit: limit.unwrap_or(DEFAULT_INFERENCE_QUERY_LIMIT),
            offset: offset.unwrap_or(0),
            ..Default::default()
        };
        let fut = client.experimental_list_inferences(params);
        let wires =
            tokio_block_on_without_gil(this.py(), fut).map_err(|e| convert_error(this.py(), e))?;

        // Convert each StoredInference to the appropriate Python dataclass
        let py_objects: Vec<_> = wires
            .iter()
            .map(|inference| {
                convert_response_to_python_dataclass(
                    this.py(),
                    inference,
                    "tensorzero",
                    match inference {
                        StoredInference::Chat(_) => "StoredInferenceChat",
                        StoredInference::Json(_) => "StoredInferenceJson",
                    },
                )
            })
            .collect::<PyResult<_>>()?;

        Ok(PyList::new(this.py(), py_objects)?.unbind())
    }

    /// Get specific inferences by their IDs.
    ///
    /// :param ids: A sequence of inference IDs to retrieve. They should be in UUID format.
    /// :param function_name: Optional function name to filter by (improves query performance).
    /// :param output_source: The source of the output ("inference" or "demonstration"). Default: "inference".
    /// :return: A `GetInferencesResponse` object.
    #[pyo3(signature = (*, ids, function_name=None, output_source="inference"))]
    fn get_inferences(
        this: PyRef<'_, Self>,
        ids: Vec<Bound<'_, PyAny>>,
        function_name: Option<String>,
        output_source: &str,
    ) -> PyResult<Py<PyAny>> {
        let client = this.as_super().client.clone();
        let ids: Vec<uuid::Uuid> = ids
            .into_iter()
            .map(|id| python_uuid_to_uuid("id", id))
            .collect::<Result<Vec<_>, _>>()?;

        let output_source =
            output_source
                .try_into()
                .map_err(|e: tensorzero_core::error::Error| {
                    convert_error(this.py(), TensorZeroError::Other { source: e.into() })
                })?;

        let fut = client.get_inferences(ids, function_name, output_source);
        let response =
            tokio_block_on_without_gil(this.py(), fut).map_err(|e| convert_error(this.py(), e))?;
        convert_response_to_python_dataclass(
            this.py(),
            &response,
            "tensorzero",
            "GetInferencesResponse",
        )
    }

    /// List inferences with optional filtering, pagination, and sorting.
    ///
    /// :param request: A `ListInferencesRequest` object with filter parameters.
    /// :return: A `GetInferencesResponse` object.
    #[pyo3(signature = (*, request))]
    fn list_inferences(this: PyRef<'_, Self>, request: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let client = this.as_super().client.clone();
        let request = deserialize_from_pyobj(this.py(), &request)?;

        let fut = client.list_inferences(request);
        let response =
            tokio_block_on_without_gil(this.py(), fut).map_err(|e| convert_error(this.py(), e))?;
        convert_response_to_python_dataclass(
            this.py(),
            &response,
            "tensorzero",
            "GetInferencesResponse",
        )
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
    /// :param concurrency: Maximum number of samples to process concurrently. Defaults to 100.
    /// :return: A list of rendered samples.
    #[pyo3(signature = (*, stored_samples, variants, concurrency=None))]
    fn experimental_render_samples(
        this: PyRef<'_, Self>,
        stored_samples: Vec<Bound<'_, PyAny>>,
        variants: HashMap<String, String>,
        concurrency: Option<usize>,
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
            .map(|x| {
                // NOTE(shuyangli): We do not re-fetch any files here, and simply error out if any samples have files.
                // We may need to rearchitect the optimization pipeline to support this.
                deserialize_from_stored_sample(this.py(), x, config)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let fut = client.experimental_render_samples(stored_samples, variants, concurrency);
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
                        return Err(tensorzero_error::TensorZeroInternalError::new_err(format!(
                            "Failed to construct TensorZero client: {e:?}"
                        )));
                    }
                };

                // Construct an instance of `AsyncTensorZeroGateway` (while providing the fields from the `BaseTensorZeroGateway` superclass).
                let instance = PyClassInitializer::from(BaseTensorZeroGateway { client })
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
    #[expect(clippy::unused_async)]
    async fn close(&self) {
        // TODO - implement closing the 'reqwest' connection pool: https://github.com/tensorzero/tensorzero/issues/857
    }

    #[expect(clippy::unused_async)]
    async fn __aenter__(this: Py<Self>) -> Py<Self> {
        this
    }

    #[expect(clippy::unused_async)]
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
    #[pyo3(signature = (*, config_file=None, clickhouse_url=None, postgres_url=None, valkey_url=None, timeout=None, async_setup=true))]
    /// Initialize the TensorZero client, using an embedded gateway.
    /// This connects to ClickHouse (if provided) and runs DB migrations.
    ///
    /// :param config_file: The path to the TensorZero configuration file. Example: "tensorzero.toml"
    /// :param clickhouse_url: The URL of the ClickHouse instance to use for the gateway. If observability is disabled in the config, this can be `None`
    /// :param postgres_url: The URL of the PostgreSQL instance to use for rate limiting.
    /// :param valkey_url: The URL of the Valkey instance to use for rate limiting.
    /// :param timeout: The timeout for embedded gateway request processing, in seconds. If this timeout is hit, any in-progress LLM requests may be aborted. If not provided, no timeout will be set.
    /// :param async_setup: If true, this method will return a `Future` that resolves to an `AsyncTensorZeroGateway` instance. Otherwise, it will block and construct the `AsyncTensorZeroGateway`
    /// :return: A `Future` that resolves to an `AsyncTensorZeroGateway` instance configured to use an embedded gateway (or an `AsyncTensorZeroGateway` if `async_setup=False`).
    fn build_embedded(
        // This is a classmethod, so it receives the class object as a parameter.
        cls: &Bound<'_, PyType>,
        config_file: Option<&str>,
        clickhouse_url: Option<String>,
        postgres_url: Option<String>,
        valkey_url: Option<String>,
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
            postgres_config: postgres_url.map(PostgresConfig::Url),
            valkey_url,
            timeout,
            verify_credentials: true,
            allow_batch_writes: false,
        })
        // When the underlying `GatewayHandle` is dropped, we need to be in the Tokio runtime
        // with the GIL released (since we might block on the ClickHouse batcher shutting down)
        .with_drop_wrapper(in_tokio_runtime_no_gil)
        .build();
        let fut = async move {
            let client = client_fut.await;
            // We need to interact with Python objects here (to build up a Python `AsyncTensorZeroGateway`),
            // so we need the GIL
            Python::attach(|py| {
                let client = match client {
                    Ok(client) => client,
                    Err(e) => {
                        return Err(tensorzero_error::TensorZeroInternalError::new_err(format!(
                            "Failed to construct TensorZero client: {e:?}"
                        )));
                    }
                };

                // Construct an instance of `AsyncTensorZeroGateway` (while providing the fields from the `BaseTensorZeroGateway` superclass).
                let instance = PyClassInitializer::from(BaseTensorZeroGateway { client })
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

    #[pyo3(signature = (*, input, function_name=None, model_name=None, episode_id=None, stream=None, params=None, variant_name=None, dryrun=None, output_schema=None, allowed_tools=None, additional_tools=None, provider_tools=None, tool_choice=None, parallel_tool_calls=None, internal=None, tags=None, credentials=None, cache_options=None, extra_body=None, extra_headers=None, include_original_response=None, include_raw_response=None, include_raw_usage=None, otlp_traces_extra_headers=None, otlp_traces_extra_attributes=None, otlp_traces_extra_resources=None, internal_dynamic_variant_config=None))]
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
    /// :param include_raw_usage: If set, include raw provider-specific usage data in the response.
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
        include_raw_response: Option<bool>,
        include_raw_usage: Option<bool>,
        otlp_traces_extra_headers: Option<HashMap<String, String>>,
        otlp_traces_extra_attributes: Option<HashMap<String, String>>,
        otlp_traces_extra_resources: Option<HashMap<String, String>>,
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
            include_raw_response.unwrap_or(false),
            include_raw_usage.unwrap_or(false),
            otlp_traces_extra_headers,
            otlp_traces_extra_attributes,
            otlp_traces_extra_resources,
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
    #[pyo3(warn(message = "Please use `create_datapoints` instead of `create_datapoints_legacy`. In a future release, `create_datapoints_legacy` will be removed.", category = PyDeprecationWarning))]
    fn create_datapoints_legacy<'a>(
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
    #[pyo3(warn(message = "Please use `delete_datapoints` instead of `delete_datapoint`. In a future release, `delete_datapoint` will be removed.", category = PyDeprecationWarning))]
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
    #[pyo3(warn(message = "Please use `get_datapoints` instead of `get_datapoint`. In a future release, `get_datapoint` will be removed.", category = PyDeprecationWarning))]
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

    /// DEPRECATED: Use `list_datapoints` instead.
    #[pyo3(signature = (*, dataset_name, function_name=None, limit=None, offset=None))]
    #[pyo3(warn(message = "Please use `list_datapoints` instead of `list_datapoints_legacy`. In a future release, `list_datapoints_legacy` will be removed.", category = PyDeprecationWarning))]
    fn list_datapoints_legacy<'py>(
        this: PyRef<'py, Self>,
        dataset_name: String,
        function_name: Option<String>,
        limit: Option<u32>,
        offset: Option<u32>,
    ) -> PyResult<Bound<'py, PyAny>> {
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

    /// Create one or more datapoints in a dataset.
    ///
    /// :param dataset_name: The name of the dataset to create the datapoints in.
    /// :param requests: A list of `CreateDatapointRequest` objects.
    /// :return: A `CreateDatapointsResponse` object containing the IDs of the newly-created datapoints.
    #[pyo3(signature = (*, dataset_name, requests))]
    fn create_datapoints<'a>(
        this: PyRef<'a, Self>,
        dataset_name: String,
        requests: Vec<Bound<'a, PyAny>>,
    ) -> PyResult<Bound<'a, PyAny>> {
        let client = this.as_super().client.clone();
        // Convert CreateDatapointRequest dataclasses to Rust types using deserialize_from_pyobj
        let requests = requests
            .iter()
            .map(|dp| deserialize_from_pyobj(this.py(), dp))
            .collect::<Result<Vec<_>, _>>()?;
        pyo3_async_runtimes::tokio::future_into_py(this.py(), async move {
            let res = client.create_datapoints(dataset_name, requests).await;
            Python::attach(|py| match res {
                Ok(response) => convert_response_to_python_dataclass(
                    py,
                    &response,
                    "tensorzero",
                    "CreateDatapointsResponse",
                ),
                Err(e) => Err(convert_error(py, e)),
            })
        })
    }

    /// Update one or more datapoints in a dataset.
    ///
    /// :param dataset_name: The name of the dataset containing the datapoints to update.
    /// :param requests: A list of `UpdateDatapointRequest` objects.
    /// :return: An `UpdateDatapointsResponse` object.
    #[pyo3(signature = (*, dataset_name, requests))]
    fn update_datapoints<'a>(
        this: PyRef<'a, Self>,
        dataset_name: String,
        requests: Vec<Bound<'a, PyAny>>,
    ) -> PyResult<Bound<'a, PyAny>> {
        let client = this.as_super().client.clone();
        let requests = requests
            .iter()
            .map(|dp| deserialize_from_pyobj(this.py(), dp))
            .collect::<Result<Vec<_>, _>>()?;
        pyo3_async_runtimes::tokio::future_into_py(this.py(), async move {
            let res = client.update_datapoints(dataset_name, requests).await;
            Python::attach(|py| match res {
                Ok(response) => convert_response_to_python_dataclass(
                    py,
                    &response,
                    "tensorzero",
                    "UpdateDatapointsResponse",
                ),
                Err(e) => Err(convert_error(py, e)),
            })
        })
    }

    /// Get specific datapoints by their IDs.
    ///
    /// :param ids: A list of datapoint IDs to retrieve.
    /// :return: A `GetDatapointsResponse` object.
    #[pyo3(signature = (*, ids, dataset_name = None))]
    fn get_datapoints<'a>(
        this: PyRef<'a, Self>,
        ids: Vec<Bound<'a, PyAny>>,
        dataset_name: Option<String>,
    ) -> PyResult<Bound<'a, PyAny>> {
        if dataset_name.is_none() {
            let warnings = PyModule::import(this.py(), "warnings")?;
            warnings.call_method1(
                "warn",
                (
                    "Calling get_datapoints without a dataset name is deprecated. Please provide a dataset name for performance reasons.",
                    this.py().get_type::<PyDeprecationWarning>(),
                ),
            )?;
        }

        let client = this.as_super().client.clone();
        let ids: Vec<uuid::Uuid> = ids
            .into_iter()
            .map(|id| python_uuid_to_uuid("id", id))
            .collect::<Result<Vec<_>, _>>()?;
        pyo3_async_runtimes::tokio::future_into_py(this.py(), async move {
            let res = client.get_datapoints(dataset_name, ids).await;
            Python::attach(|py| match res {
                Ok(response) => convert_response_to_python_dataclass(
                    py,
                    &response,
                    "tensorzero",
                    "GetDatapointsResponse",
                ),
                Err(e) => Err(convert_error(py, e)),
            })
        })
    }

    /// Update metadata for one or more datapoints.
    ///
    /// :param dataset_name: The name of the dataset containing the datapoints.
    /// :param requests: A list of `UpdateDatapointMetadataRequest` objects.
    /// :return: An `UpdateDatapointsResponse` object containing the IDs of updated datapoints.
    #[pyo3(signature = (*, dataset_name, requests))]
    fn update_datapoints_metadata<'a>(
        this: PyRef<'a, Self>,
        dataset_name: String,
        requests: Vec<Bound<'a, PyAny>>,
    ) -> PyResult<Bound<'a, PyAny>> {
        let client = this.as_super().client.clone();
        let requests = requests
            .iter()
            .map(|dp| deserialize_from_pyobj(this.py(), dp))
            .collect::<Result<Vec<_>, _>>()?;
        pyo3_async_runtimes::tokio::future_into_py(this.py(), async move {
            let res = client
                .update_datapoints_metadata(dataset_name, requests)
                .await;
            Python::attach(|py| match res {
                Ok(response) => convert_response_to_python_dataclass(
                    py,
                    &response,
                    "tensorzero",
                    "UpdateDatapointsResponse",
                ),
                Err(e) => Err(convert_error(py, e)),
            })
        })
    }

    /// Delete multiple datapoints from a dataset.
    ///
    /// :param dataset_name: The name of the dataset to delete the datapoints from.
    /// :param ids: A list of datapoint IDs to delete.
    /// :return: A `DeleteDatapointsResponse` object containing the IDs of deleted datapoints.
    #[pyo3(signature = (*, dataset_name, ids))]
    fn delete_datapoints<'a>(
        this: PyRef<'a, Self>,
        dataset_name: String,
        ids: Vec<Bound<'a, PyAny>>,
    ) -> PyResult<Bound<'a, PyAny>> {
        let client = this.as_super().client.clone();
        let ids: Vec<uuid::Uuid> = ids
            .into_iter()
            .map(|id| python_uuid_to_uuid("id", id))
            .collect::<Result<Vec<_>, _>>()?;
        pyo3_async_runtimes::tokio::future_into_py(this.py(), async move {
            let res = client.delete_datapoints(dataset_name, ids).await;
            Python::attach(|py| match res {
                Ok(response) => convert_response_to_python_dataclass(
                    py,
                    &response,
                    "tensorzero",
                    "DeleteDatapointsResponse",
                ),
                Err(e) => Err(convert_error(py, e)),
            })
        })
    }

    /// Delete a dataset.
    ///
    /// :param dataset_name: The name of the dataset to delete.
    /// :return: A `DeleteDatapointsResponse` object containing the IDs of deleted datapoints.
    #[pyo3(signature = (*, dataset_name))]
    fn delete_dataset<'a>(
        this: PyRef<'a, Self>,
        dataset_name: String,
    ) -> PyResult<Bound<'a, PyAny>> {
        let client = this.as_super().client.clone();
        pyo3_async_runtimes::tokio::future_into_py(this.py(), async move {
            let res = client.delete_dataset(dataset_name).await;
            Python::attach(|py| match res {
                Ok(response) => convert_response_to_python_dataclass(
                    py,
                    &response,
                    "tensorzero",
                    "DeleteDatapointsResponse",
                ),
                Err(e) => Err(convert_error(py, e)),
            })
        })
    }

    /// Create datapoints from inferences.
    ///
    /// :param dataset_name: The name of the dataset to create the datapoints from.
    /// :param params: The parameters specifying which inferences to convert to datapoints.
    ///                 For InferenceIds: pass `{"type": "inference_ids", "inference_ids": [...], "output_source": "inference"}`
    ///                 For InferenceQuery: pass `{"type": "inference_query", "function_name": "...", "output_source": "inference", ...}`
    /// :param output_source: The source of the output to create datapoints from. "none", "inference", or "demonstration".
    ///                       Can also be specified inside `params.output_source`. If both are provided, an error is raised.
    /// :return: A `CreateDatapointsResponse` object containing the IDs of the newly-created datapoints.
    #[pyo3(signature = (*, dataset_name, params, output_source=None))]
    fn create_datapoints_from_inferences<'a>(
        this: PyRef<'a, Self>,
        dataset_name: String,
        params: Bound<'a, PyAny>,
        output_source: Option<String>,
    ) -> PyResult<Bound<'a, PyAny>> {
        let client = this.as_super().client.clone();

        // Handle output_source: can be passed as parameter or inside params, but not both
        if let Some(source) = &output_source {
            let existing = params.getattr("output_source").ok();
            if let Some(existing) = existing
                && !existing.is_none()
            {
                return Err(PyValueError::new_err(
                    "You must specify `output_source` either at the root or inside the `params` parameter but not both.",
                ));
            }
            params.setattr("output_source", source)?;
        }

        let params = deserialize_from_pyobj(this.py(), &params)?;

        pyo3_async_runtimes::tokio::future_into_py(this.py(), async move {
            let res = client
                .create_datapoints_from_inferences(dataset_name, params)
                .await;
            Python::attach(|py| match res {
                Ok(response) => convert_response_to_python_dataclass(
                    py,
                    &response,
                    "tensorzero",
                    "CreateDatapointsResponse",
                ),
                Err(e) => Err(convert_error(py, e)),
            })
        })
    }

    /// List datapoints in a dataset.
    ///
    /// :param dataset_name: The name of the dataset to list datapoints from.
    /// :param request: The request parameters.
    /// :return: A `GetDatapointsResponse` object.
    #[pyo3(signature = (*, dataset_name, request))]
    fn list_datapoints<'a>(
        this: PyRef<'a, Self>,
        dataset_name: String,
        request: Bound<'a, PyAny>,
    ) -> PyResult<Bound<'a, PyAny>> {
        let client = this.as_super().client.clone();
        let request = deserialize_from_pyobj(this.py(), &request)?;

        pyo3_async_runtimes::tokio::future_into_py(this.py(), async move {
            let res = client.list_datapoints(dataset_name, request).await;
            Python::attach(|py| match res {
                Ok(response) => convert_response_to_python_dataclass(
                    py,
                    &response,
                    "tensorzero",
                    "GetDatapointsResponse",
                ),
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
    /// * `variant_name` - Optional name of the variant to evaluate
    /// * `concurrency` - The maximum number of examples to process in parallel
    /// * `inference_cache` - Cache configuration for inference requests ("on", "off", "read_only", or "write_only")
    /// * `internal_dynamic_variant_config` - Optional dynamic variant configuration [INTERNAL: This field is unstable and may change without notice.]
    /// * `max_datapoints` - Optional maximum number of datapoints to evaluate from the dataset
    /// * `adaptive_stopping` - Optional dict configuring adaptive stopping behavior for evals.
    ///                         Example for two evaluators named "exact_match" and "llm_judge":
    ///                           `{"precision": {"exact_match": 0.2, "llm_judge": 0.15}}`
    ///                         The "precision" field maps evaluator names to confidence interval half-widths.
    ///                         Evaluation for a given evaluator stops when it achieves its precision target,
    ///                         i.e. the width of the larger of the two halves of its confidence interval
    ///                         is <= the precision target.
    #[pyo3(signature = (*,
                        evaluation_name,
                        dataset_name=None,
                        datapoint_ids=None,
                        variant_name=None,
                        concurrency=1,
                        inference_cache="on".to_string(),
                        internal_dynamic_variant_config=None,
                        max_datapoints=None,
                        adaptive_stopping=None
    ),
    text_signature = "(self, *, evaluation_name, dataset_name=None, datapoint_ids=None, variant_name=None, concurrency=1, inference_cache='on', internal_dynamic_variant_config=None, max_datapoints=None, adaptive_stopping=None)"
    )]
    #[expect(clippy::too_many_arguments)]
    fn experimental_run_evaluation<'py>(
        this: PyRef<'py, Self>,
        evaluation_name: String,
        dataset_name: Option<String>,
        datapoint_ids: Option<Vec<String>>,
        variant_name: Option<String>,
        concurrency: usize,
        inference_cache: String,
        internal_dynamic_variant_config: Option<&Bound<'py, PyDict>>,
        max_datapoints: Option<u32>,
        adaptive_stopping: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client = this.as_super().client.clone();

        let inference_cache_enum: tensorzero_core::cache::CacheEnabledMode =
            deserialize_from_pyobj(
                this.py(),
                &inference_cache.into_pyobject(this.py())?.into_any(),
            )?;

        let variant =
            construct_evaluation_variant(this.py(), internal_dynamic_variant_config, variant_name)?;

        // Parse adaptive_stopping config from Python dict
        let precision_targets_map = if let Some(adaptive_stopping_dict) = adaptive_stopping {
            // Extract the "precision" field from adaptive_stopping dict
            if let Ok(Some(precision_bound)) = adaptive_stopping_dict.get_item("precision") {
                let precision_dict_bound = precision_bound.downcast::<PyDict>().map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                        "adaptive_stopping['precision'] must be a dictionary",
                    )
                })?;

                let mut map = std::collections::HashMap::new();
                for (key, value) in precision_dict_bound.iter() {
                    let key_str: String = key.extract()?;
                    let value_f64: f64 = value.extract()?;
                    map.insert(key_str, value_f64 as f32);
                }
                map
            } else {
                HashMap::new()
            }
        } else {
            HashMap::new()
        };

        // Parse datapoint_ids from strings to UUIDs
        let datapoint_ids: Option<Vec<Uuid>> = datapoint_ids
            .map(|ids| {
                ids.iter()
                    .map(|s| {
                        Uuid::parse_str(s).map_err(|e| {
                            pyo3::exceptions::PyValueError::new_err(format!(
                                "Invalid UUID in datapoint_ids: {e}"
                            ))
                        })
                    })
                    .collect::<PyResult<Vec<Uuid>>>()
            })
            .transpose()?;

        pyo3_async_runtimes::tokio::future_into_py(this.py(), async move {
            // Get app state data
            let app_state = client.get_app_state_data().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("Client is not in EmbeddedGateway mode")
            })?;

            let evaluation_run_id = uuid::Uuid::now_v7();

            // Extract evaluation config from app_state
            let evaluation_config = app_state
                .config
                .evaluations
                .get(&evaluation_name)
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "evaluation '{evaluation_name}' not found"
                    ))
                })?
                .clone();

            // Build function configs table from all functions in the config
            let function_configs: EvaluationFunctionConfigTable = app_state
                .config
                .functions
                .iter()
                .map(|(name, func)| (name.clone(), EvaluationFunctionConfig::from(func.as_ref())))
                .collect();
            let function_configs = Arc::new(function_configs);

            // Wrap the client in ClientInferenceExecutor for use with evaluations
            let inference_executor = Arc::new(ClientInferenceExecutor::new(client.clone()));

            let core_args = EvaluationCoreArgs {
                inference_executor,
                clickhouse_client: app_state.clickhouse_connection_info.clone(),
                evaluation_config,
                function_configs,
                evaluation_name,
                evaluation_run_id,
                dataset_name,
                datapoint_ids,
                variant,
                concurrency,
                inference_cache: inference_cache_enum,
                tags: HashMap::new(), // No external tags for Python client evaluations
            };

            let result =
                run_evaluation_core_streaming(core_args, max_datapoints, precision_targets_map)
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
    #[pyo3(warn(message = "Please use `list_inferences` instead of `experimental_list_inferences`. In a future release, `experimental_list_inferences` will be removed.", category = PyDeprecationWarning))]
    // The text_signature is a workaround to weird behavior in pyo3 where the default for an option
    // is written as an ellipsis object.
    #[expect(clippy::too_many_arguments)]
    #[expect(deprecated)]
    fn experimental_list_inferences<'a>(
        this: PyRef<'a, Self>,
        function_name: String,
        variant_name: Option<String>,
        filters: Option<Bound<'a, PyAny>>,
        output_source: String,
        order_by: Option<Bound<'a, PyAny>>,
        limit: Option<u32>,
        offset: Option<u32>,
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
                limit: limit.unwrap_or(DEFAULT_INFERENCE_QUERY_LIMIT),
                offset: offset.unwrap_or(0),
                ..Default::default()
            };
            let res = client.experimental_list_inferences(params).await;
            Python::attach(|py| match res {
                Ok(wire_inferences) => {
                    // Convert each StoredInference to the appropriate Python dataclass
                    let py_objects: Vec<_> = wire_inferences
                        .iter()
                        .map(|inference| {
                            convert_response_to_python_dataclass(
                                py,
                                inference,
                                "tensorzero",
                                match inference {
                                    StoredInference::Chat(_) => "StoredInferenceChat",
                                    StoredInference::Json(_) => "StoredInferenceJson",
                                },
                            )
                        })
                        .collect::<PyResult<_>>()?;

                    Ok(PyList::new(py, py_objects)?.unbind())
                }
                Err(e) => Err(convert_error(py, e)),
            })
        })
    }

    /// Get specific inferences by their IDs.
    ///
    /// :param ids: A sequence of inference IDs to retrieve. They should be in UUID format.
    /// :param function_name: Optional function name to filter by (improves query performance).
    /// :param output_source: The source of the output ("inference" or "demonstration"). Default: "inference".
    /// :return: A `GetInferencesResponse` object.
    #[pyo3(signature = (*, ids, function_name=None, output_source="inference"))]
    fn get_inferences<'a>(
        this: PyRef<'a, Self>,
        ids: Vec<Bound<'a, PyAny>>,
        function_name: Option<String>,
        output_source: &str,
    ) -> PyResult<Bound<'a, PyAny>> {
        let client = this.as_super().client.clone();
        let ids: Vec<uuid::Uuid> = ids
            .into_iter()
            .map(|id| python_uuid_to_uuid("id", id))
            .collect::<Result<Vec<_>, _>>()?;

        let output_source =
            output_source
                .try_into()
                .map_err(|e: tensorzero_core::error::Error| {
                    convert_error(this.py(), TensorZeroError::Other { source: e.into() })
                })?;

        pyo3_async_runtimes::tokio::future_into_py(this.py(), async move {
            let res = client
                .get_inferences(ids, function_name, output_source)
                .await;
            Python::attach(|py| match res {
                Ok(response) => convert_response_to_python_dataclass(
                    py,
                    &response,
                    "tensorzero",
                    "GetInferencesResponse",
                ),
                Err(e) => Err(convert_error(py, e)),
            })
        })
    }

    /// List inferences with optional filtering, pagination, and sorting.
    ///
    /// :param request: A `ListInferencesRequest` object with filter parameters.
    /// :return: A `GetInferencesResponse` object.
    #[pyo3(signature = (*, request))]
    fn list_inferences<'a>(
        this: PyRef<'a, Self>,
        request: Bound<'a, PyAny>,
    ) -> PyResult<Bound<'a, PyAny>> {
        let client = this.as_super().client.clone();
        let request = deserialize_from_pyobj(this.py(), &request)?;

        pyo3_async_runtimes::tokio::future_into_py(this.py(), async move {
            let res = client.list_inferences(request).await;
            Python::attach(|py| match res {
                Ok(response) => convert_response_to_python_dataclass(
                    py,
                    &response,
                    "tensorzero",
                    "GetInferencesResponse",
                ),
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
    /// :param concurrency: Maximum number of samples to process concurrently. Defaults to 100.
    /// :return: A list of rendered samples.
    #[pyo3(signature = (*, stored_samples, variants, concurrency=None))]
    fn experimental_render_samples<'a>(
        this: PyRef<'a, Self>,
        stored_samples: Vec<Bound<'a, PyAny>>,
        variants: HashMap<String, String>,
        concurrency: Option<usize>,
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
            .map(|x| {
                // NOTE(shuyangli): We do not re-fetch any files here, and simply error out if any samples have files.
                // We may need to rearchitect the optimization pipeline to support this.
                deserialize_from_stored_sample(this.py(), x, config)
            })
            .collect::<Result<Vec<_>, _>>()?;
        pyo3_async_runtimes::tokio::future_into_py(this.py(), async move {
            let res = client
                .experimental_render_samples(stored_samples, variants, concurrency)
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
pub fn convert_error(_py: Python<'_>, e: TensorZeroError) -> PyErr {
    match e {
        TensorZeroError::Http {
            status_code,
            text,
            source: _,
        } => tensorzero_error::TensorZeroError::new_err((status_code, text)),
        TensorZeroError::Other { source } => {
            tensorzero_error::TensorZeroInternalError::new_err(source.to_string())
        }
        TensorZeroError::RequestTimeout => {
            tensorzero_error::TensorZeroInternalError::new_err(e.to_string())
        }
        // Required due to the `#[non_exhaustive]` attribute on `TensorZeroError` - we want to force
        // downstream consumers to handle all possible error types, but the compiler also requires us
        // to do this (since our python bindings are in a different crate from the Rust client.)
        _ => tensorzero_error::TensorZeroInternalError::new_err(format!(
            "Unexpected TensorZero error: {e:?}"
        )),
    }
}

fn warn_no_config(py: Python<'_>, config: Option<&str>) -> PyResult<()> {
    if config.is_none() {
        let user_warning = py.get_type::<pyo3::exceptions::PyUserWarning>();
        PyErr::warn(
            py,
            &user_warning,
            c_str!(
                "No config file provided, so only default functions will be available. Use `config_file=\"path/to/tensorzero.toml\"` to specify a config file."
            ),
            0,
        )?;
    }
    Ok(())
}
