/**
 * URL helper functions that ensure proper encoding of identifiers.
 * Always use these instead of string interpolation to handle names with special characters.
 */

// ============================================================================
// Observability - Functions
// ============================================================================

export function toFunctionUrl(functionName: string): string {
  return `/observability/functions/${encodeURIComponent(functionName)}`;
}

export function toVariantUrl(
  functionName: string,
  variantName: string,
): string {
  return `/observability/functions/${encodeURIComponent(functionName)}/variants/${encodeURIComponent(variantName)}`;
}

// ============================================================================
// Observability - Inferences
// ============================================================================

export function toInferenceUrl(inferenceId: string): string {
  return `/observability/inferences/${encodeURIComponent(inferenceId)}`;
}

// ============================================================================
// Observability - Episodes
// ============================================================================

export function toEpisodeUrl(episodeId: string): string {
  return `/observability/episodes/${encodeURIComponent(episodeId)}`;
}

// ============================================================================
// Datasets
// ============================================================================

export function toDatasetUrl(datasetName: string): string {
  return `/datasets/${encodeURIComponent(datasetName)}`;
}

export function toDatapointUrl(
  datasetName: string,
  datapointId: string,
): string {
  return `/datasets/${encodeURIComponent(datasetName)}/datapoint/${encodeURIComponent(datapointId)}`;
}

// ============================================================================
// Evaluations
// ============================================================================

export function toEvaluationUrl(
  evaluationName: string,
  queryParams?: { evaluation_run_ids?: string },
): string {
  const baseUrl = `/evaluations/${encodeURIComponent(evaluationName)}`;
  if (queryParams?.evaluation_run_ids) {
    return `${baseUrl}?evaluation_run_ids=${encodeURIComponent(queryParams.evaluation_run_ids)}`;
  }
  return baseUrl;
}

export function toEvaluationDatapointUrl(
  evaluationName: string,
  datapointId: string,
  queryParams?: { evaluation_run_ids?: string },
): string {
  const baseUrl = `/evaluations/${encodeURIComponent(evaluationName)}/${encodeURIComponent(datapointId)}`;
  if (queryParams?.evaluation_run_ids) {
    return `${baseUrl}?evaluation_run_ids=${encodeURIComponent(queryParams.evaluation_run_ids)}`;
  }
  return baseUrl;
}

// ============================================================================
// Dynamic Evaluations
// ============================================================================

export function toDynamicEvaluationRunUrl(runId: string): string {
  return `/dynamic_evaluations/runs/${encodeURIComponent(runId)}`;
}

export function toDynamicEvaluationProjectUrl(projectName: string): string {
  return `/dynamic_evaluations/projects/${encodeURIComponent(projectName)}`;
}

// ============================================================================
// Optimization
// ============================================================================

export function toSupervisedFineTuningJobUrl(jobId: string): string {
  return `/optimization/supervised-fine-tuning/${encodeURIComponent(jobId)}`;
}
