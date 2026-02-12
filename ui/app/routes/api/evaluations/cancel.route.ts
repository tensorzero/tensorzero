import type { ActionFunctionArgs } from "react-router";
import { cancelEvaluation } from "~/utils/evaluations.server";

/**
 * API route for cancelling one or more running evaluations in a single request.
 * Aborts each SSE connection to the gateway, which cancels all in-flight tasks.
 * Partial results already written to ClickHouse are preserved.
 *
 * Route: POST /api/evaluations/cancel
 * Body: { evaluation_run_ids: string[] }
 */
export async function action({ request }: ActionFunctionArgs) {
  let body: { evaluation_run_ids?: unknown };
  try {
    body = (await request.json()) as { evaluation_run_ids?: unknown };
  } catch {
    return Response.json(
      { success: false, error: "Invalid JSON body" },
      { status: 400 },
    );
  }

  const { evaluation_run_ids } = body;
  if (
    !Array.isArray(evaluation_run_ids) ||
    evaluation_run_ids.length === 0 ||
    !evaluation_run_ids.every((id) => typeof id === "string")
  ) {
    return Response.json(
      {
        success: false,
        error: "evaluation_run_ids must be a non-empty array of strings",
      },
      { status: 400 },
    );
  }

  for (const runId of evaluation_run_ids) {
    cancelEvaluation(runId);
  }

  return Response.json({ success: true });
}
