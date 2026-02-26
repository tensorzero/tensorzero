/**
 * URL helper functions that ensure proper encoding of identifiers.
 * Always use these instead of string interpolation to handle names with special characters.
 */

import type { ResolvedObject } from "~/types/tensorzero";

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

export function toInferenceApiUrl(inferenceId: string): string {
  return `/api/inference/${encodeURIComponent(inferenceId)}`;
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
// Workflow Evaluations
// ============================================================================

export function toWorkflowEvaluationRunUrl(runId: string): string {
  return `/workflow-evaluations/runs/${encodeURIComponent(runId)}`;
}

export function toWorkflowEvaluationProjectUrl(projectName: string): string {
  return `/workflow-evaluations/projects/${encodeURIComponent(projectName)}`;
}

// ============================================================================
// Optimization
// ============================================================================

export function toSupervisedFineTuningJobUrl(jobId: string): string {
  return `/optimization/supervised-fine-tuning/${encodeURIComponent(jobId)}`;
}

// ============================================================================
// Resolved Object URLs
// ============================================================================

export function toResolvedObjectUrl(
  uuid: string,
  obj: ResolvedObject,
): string | null {
  switch (obj.type) {
    case "inference":
      return toInferenceUrl(uuid);
    case "episode":
      return toEpisodeUrl(uuid);
    case "chat_datapoint":
    case "json_datapoint":
      return toDatapointUrl(obj.dataset_name, uuid);
    case "model_inference":
    case "boolean_feedback":
    case "float_feedback":
    case "comment_feedback":
    case "demonstration_feedback":
      return null;
    default: {
      const _exhaustiveCheck: never = obj;
      return _exhaustiveCheck;
    }
  }
}

// ============================================================================
// Internal API Routes
// ============================================================================

export function toResolveUuidApi(uuid: string): string {
  return `/api/tensorzero/resolve_uuid/${encodeURIComponent(uuid)}`;
}

export function toInferencePreviewApi(inferenceId: string): string {
  return `/api/tensorzero/inference_preview/${encodeURIComponent(inferenceId)}`;
}

export function toEpisodePreviewApi(episodeId: string): string {
  return `/api/tensorzero/episode_preview/${encodeURIComponent(episodeId)}`;
}
