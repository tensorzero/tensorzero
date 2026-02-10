import type { ActionFunctionArgs } from "react-router";
import { killEvaluation } from "~/utils/evaluations.server";
import { logger } from "~/utils/logger";

/**
 * API route for killing a running evaluation.
 * Aborts the SSE connection to the gateway, which cancels all in-flight tasks.
 * Partial results already written to ClickHouse are preserved.
 *
 * Route: POST /api/evaluations/:evaluation_run_id/kill
 */
export async function action({ params, request }: ActionFunctionArgs) {
  const evaluationRunId = params.evaluation_run_id;
  if (!evaluationRunId) {
    return Response.json(
      { success: false, error: "Evaluation run ID is required" },
      { status: 400 },
    );
  }

  if (request.method !== "POST") {
    return Response.json(
      { success: false, error: "Method not allowed" },
      { status: 405 },
    );
  }

  try {
    const result = killEvaluation(evaluationRunId);
    if (!result.killed && !result.already_completed) {
      return Response.json(
        { success: false, error: "Evaluation run not found" },
        { status: 404 },
      );
    }
    return Response.json({
      success: true,
      already_completed: result.already_completed,
    });
  } catch (error) {
    logger.error("Failed to kill evaluation run:", error);
    const message =
      error instanceof Error ? error.message : "Failed to kill evaluation run";
    return Response.json({ success: false, error: message }, { status: 500 });
  }
}
