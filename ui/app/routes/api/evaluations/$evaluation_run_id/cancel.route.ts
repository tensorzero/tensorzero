import type { ActionFunctionArgs } from "react-router";
import { cancelEvaluation } from "~/utils/evaluations.server";

/**
 * API route for cancelling a running evaluation.
 * Aborts the SSE connection to the gateway, which cancels all in-flight tasks.
 * Partial results already written to ClickHouse are preserved.
 *
 * Route: POST /api/evaluations/:evaluation_run_id/cancel
 */
export async function action({ params }: ActionFunctionArgs) {
  const evaluationRunId = params.evaluation_run_id;
  if (!evaluationRunId) {
    return Response.json(
      { success: false, error: "Evaluation run ID is required" },
      { status: 400 },
    );
  }

  const found = cancelEvaluation(evaluationRunId);
  if (!found) {
    return Response.json(
      { success: false, error: "Evaluation run not found" },
      { status: 404 },
    );
  }
  return Response.json({ success: true });
}
