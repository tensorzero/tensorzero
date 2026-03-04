use std::collections::HashMap;
use std::sync::Arc;

use evaluations::{
    OutputFormat, RunInfo, check_evaluator_cutoffs, format_cutoff_failures,
    resolve_effective_cutoffs,
    stats::{EvaluationError, EvaluationInfo, EvaluationStats, EvaluationUpdate},
};
use pyo3::{
    exceptions::{PyStopAsyncIteration, PyStopIteration},
    prelude::*,
};
use tensorzero_core::{
    evaluations::EvaluationConfig, inference::types::pyo3_helpers::serialize_to_dict,
};
use tokio::sync::Mutex;

use crate::gil_helpers::tokio_block_on_without_gil;

/// Helper function to serialize EvaluationInfo to a Python dictionary with type="success"
fn serialize_evaluation_success(py: Python<'_>, info: &EvaluationInfo) -> PyResult<Py<PyAny>> {
    let info_dict = serialize_to_dict(py, info)?;
    info_dict.bind(py).set_item("type", "success")?;
    Ok(info_dict)
}

/// Helper function to serialize EvaluationError to a Python dictionary with type="error"
fn serialize_evaluation_error(py: Python<'_>, error: &EvaluationError) -> PyResult<Py<PyAny>> {
    let error_dict = serialize_to_dict(py, error)?;
    error_dict.bind(py).set_item("type", "error")?;
    Ok(error_dict)
}

/// Helper function to compute evaluation statistics and return as a Python dictionary
fn compute_evaluation_stats(
    py: Python<'_>,
    evaluation_infos: Vec<EvaluationInfo>,
    evaluation_errors: Vec<EvaluationError>,
    evaluation_config: Arc<EvaluationConfig>,
) -> PyResult<Py<PyAny>> {
    let stats = EvaluationStats {
        output_format: OutputFormat::Jsonl,
        evaluation_infos,
        evaluation_errors,
        progress_bar: None,
    };
    // Extract evaluators from the evaluation config
    let EvaluationConfig::Inference(inference_config) = &*evaluation_config;
    let computed_stats = stats.compute_stats(&inference_config.evaluators);
    serialize_to_dict(py, &computed_stats)
}

/// Shared implementation for checking cutoffs on evaluation results.
///
/// This mirrors the CLI's cutoff checking logic:
/// 1. Compute stats from collected evaluation results
/// 2. Resolve effective cutoffs (merging provided cutoffs with any config-level cutoffs)
/// 3. Check if stats meet the cutoff thresholds
fn check_cutoffs_impl(
    _py: Python<'_>,
    evaluation_infos: &Arc<Mutex<Vec<EvaluationInfo>>>,
    evaluation_errors: &Arc<Mutex<Vec<EvaluationError>>>,
    evaluation_config: &Arc<EvaluationConfig>,
    cutoffs: &HashMap<String, f32>,
) -> PyResult<()> {
    if cutoffs.is_empty() {
        return Ok(());
    }

    let infos = evaluation_infos.blocking_lock().clone();
    let errors = evaluation_errors.blocking_lock().clone();

    let EvaluationConfig::Inference(inference_config) = &**evaluation_config;

    let stats_obj = EvaluationStats {
        output_format: OutputFormat::Jsonl,
        evaluation_infos: infos,
        evaluation_errors: errors,
        progress_bar: None,
    };
    let stats = stats_obj.compute_stats(&inference_config.evaluators);

    let effective_cutoffs = resolve_effective_cutoffs(&inference_config.evaluators, cutoffs)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid cutoffs: {e}")))?;

    let failures =
        check_evaluator_cutoffs(&stats, &inference_config.evaluators, &effective_cutoffs).map_err(
            |e| pyo3::exceptions::PyRuntimeError::new_err(format!("Error checking cutoffs: {e}")),
        )?;

    if !failures.is_empty() {
        let failure_messages = format_cutoff_failures(&failures);
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Failed cutoffs for evaluators:\n{failure_messages}"
        )));
    }

    Ok(())
}

/// Job handler for streaming evaluation results (synchronous)
#[pyclass(frozen, str)]
pub struct EvaluationJobHandler {
    pub(crate) receiver: Mutex<tokio::sync::mpsc::Receiver<EvaluationUpdate>>,
    pub(crate) run_info: RunInfo,
    pub(crate) evaluation_config: Arc<EvaluationConfig>,
    pub(crate) evaluation_infos: Arc<Mutex<Vec<EvaluationInfo>>>,
    pub(crate) evaluation_errors: Arc<Mutex<Vec<EvaluationError>>>,
    pub(crate) cutoffs: HashMap<String, f32>,
}

#[pymethods]
impl EvaluationJobHandler {
    /// Get the run information for this evaluation
    #[getter]
    fn run_info(&self) -> PyResult<Py<PyAny>> {
        Python::attach(|py| serialize_to_dict(py, &self.run_info))
    }

    /// Returns an iterator over evaluation results as they complete
    fn results(this: Py<Self>) -> Py<Self> {
        this
    }

    fn __iter__(this: Py<Self>) -> Py<Self> {
        this
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        // Loop to skip RunInfo updates instead of using recursion
        loop {
            let evaluation_infos = self.evaluation_infos.clone();
            let evaluation_errors = self.evaluation_errors.clone();

            let update =
                tokio_block_on_without_gil(py, async { self.receiver.lock().await.recv().await });

            match update {
                Some(EvaluationUpdate::RunInfo(_)) => {
                    // Skip RunInfo, continue to next update
                    continue;
                }
                Some(EvaluationUpdate::Success(info)) => {
                    let info_clone = info.clone();
                    tokio_block_on_without_gil(py, async move {
                        evaluation_infos.lock().await.push(info_clone);
                    });
                    return serialize_evaluation_success(py, &info);
                }
                Some(EvaluationUpdate::Error(error)) => {
                    let error_clone = error.clone();
                    tokio_block_on_without_gil(py, async move {
                        evaluation_errors.lock().await.push(error_clone);
                    });
                    return serialize_evaluation_error(py, &error);
                }
                None => return Err(PyStopIteration::new_err(())),
            }
        }
    }

    /// Get summary statistics for all evaluations after completion
    fn summary_stats(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let evaluation_infos = self.evaluation_infos.clone();
        let evaluation_errors = self.evaluation_errors.clone();
        let evaluation_config = self.evaluation_config.clone();

        tokio_block_on_without_gil(py, async move {
            let infos = evaluation_infos.lock().await.clone();
            let errors = evaluation_errors.lock().await.clone();
            Python::attach(|py| compute_evaluation_stats(py, infos, errors, evaluation_config))
        })
    }

    /// Check if evaluator results meet the cutoff thresholds.
    ///
    /// Must be called after consuming all results (via iteration).
    /// Raises `RuntimeError` if any evaluator fails its cutoff threshold.
    fn check_cutoffs(&self, py: Python<'_>) -> PyResult<()> {
        check_cutoffs_impl(
            py,
            &self.evaluation_infos,
            &self.evaluation_errors,
            &self.evaluation_config,
            &self.cutoffs,
        )
    }

    fn __repr__(&self) -> PyResult<String> {
        serde_json::to_string_pretty(&self.run_info).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Serialization error: {e}"))
        })
    }
}

impl std::fmt::Display for EvaluationJobHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(&self.run_info).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

/// Job handler for streaming evaluation results (asynchronous)
#[pyclass(frozen, str)]
pub struct AsyncEvaluationJobHandler {
    pub(crate) receiver: Arc<Mutex<tokio::sync::mpsc::Receiver<EvaluationUpdate>>>,
    pub(crate) run_info: RunInfo,
    pub(crate) evaluation_config: Arc<EvaluationConfig>,
    pub(crate) evaluation_infos: Arc<Mutex<Vec<EvaluationInfo>>>,
    pub(crate) evaluation_errors: Arc<Mutex<Vec<EvaluationError>>>,
    pub(crate) cutoffs: HashMap<String, f32>,
}

#[pymethods]
impl AsyncEvaluationJobHandler {
    /// Get the run information for this evaluation
    #[getter]
    fn run_info(&self) -> PyResult<Py<PyAny>> {
        Python::attach(|py| serialize_to_dict(py, &self.run_info))
    }

    /// Returns an async iterator over evaluation results as they complete
    fn results(this: Py<Self>) -> Py<Self> {
        this
    }

    fn __aiter__(this: Py<Self>) -> Py<Self> {
        this
    }

    fn __anext__<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PyAny>> {
        let receiver = self.receiver.clone();
        let evaluation_infos = self.evaluation_infos.clone();
        let evaluation_errors = self.evaluation_errors.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            // Loop to skip RunInfo updates
            loop {
                let update = receiver.lock().await.recv().await;

                match update {
                    Some(EvaluationUpdate::RunInfo(_)) => {
                        // Skip RunInfo, continue to next update
                        continue;
                    }
                    Some(EvaluationUpdate::Success(info)) => {
                        let info_clone = info.clone();
                        evaluation_infos.lock().await.push(info_clone);
                        return Python::attach(|py| serialize_evaluation_success(py, &info));
                    }
                    Some(EvaluationUpdate::Error(error)) => {
                        let error_clone = error.clone();
                        evaluation_errors.lock().await.push(error_clone);
                        return Python::attach(|py| serialize_evaluation_error(py, &error));
                    }
                    None => return Err(PyStopAsyncIteration::new_err(())),
                }
            }
        })
    }

    /// Get summary statistics for all evaluations after completion
    fn summary_stats<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PyAny>> {
        let evaluation_infos = self.evaluation_infos.clone();
        let evaluation_errors = self.evaluation_errors.clone();
        let evaluation_config = self.evaluation_config.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let infos = evaluation_infos.lock().await.clone();
            let errors = evaluation_errors.lock().await.clone();
            Python::attach(|py| compute_evaluation_stats(py, infos, errors, evaluation_config))
        })
    }

    /// Check if evaluator results meet the cutoff thresholds.
    ///
    /// Must be called after consuming all results (via iteration).
    /// Raises `RuntimeError` if any evaluator fails its cutoff threshold.
    fn check_cutoffs<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PyAny>> {
        let evaluation_infos = self.evaluation_infos.clone();
        let evaluation_errors = self.evaluation_errors.clone();
        let evaluation_config = self.evaluation_config.clone();
        let cutoffs = self.cutoffs.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            Python::attach(|py| {
                check_cutoffs_impl(
                    py,
                    &evaluation_infos,
                    &evaluation_errors,
                    &evaluation_config,
                    &cutoffs,
                )
                .map(|()| py.None())
            })
        })
    }

    fn __repr__(&self) -> PyResult<String> {
        serde_json::to_string_pretty(&self.run_info).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Serialization error: {e}"))
        })
    }
}

impl std::fmt::Display for AsyncEvaluationJobHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let json = serde_json::to_string_pretty(&self.run_info).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}
