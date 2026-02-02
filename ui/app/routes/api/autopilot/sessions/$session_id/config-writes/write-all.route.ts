import type { ActionFunctionArgs } from "react-router";
import { ConfigWriter, writeSessionConfigWritesToFile } from "tensorzero-node";
import type { WriteConfigWriteResult } from "~/types/tensorzero";
import { getEnv } from "~/utils/env.server";
import { logger } from "~/utils/logger";

type WriteAllConfigsResponse =
  | {
      success: true;
      results: WriteConfigWriteResult[];
      total_processed: number;
    }
  | { success: false; error: string };

/**
 * API route for writing all config write events from a session to the filesystem.
 *
 * Route: POST /api/autopilot/sessions/:session_id/config-writes/write-all
 *
 * Request body: {} (empty)
 *
 * Response:
 * - { success: true, results: [...], total_processed: number } on success
 * - { success: false, error: string } on failure
 */
export async function action({
  params,
  request,
}: ActionFunctionArgs): Promise<Response> {
  const sessionId = params.session_id;
  if (!sessionId) {
    return Response.json(
      {
        success: false,
        error: "Session ID is required",
      } as WriteAllConfigsResponse,
      { status: 400 },
    );
  }

  if (request.method !== "POST") {
    return Response.json(
      {
        success: false,
        error: "Method not allowed",
      } as WriteAllConfigsResponse,
      { status: 405 },
    );
  }

  const env = getEnv();
  const configFile = env.TENSORZERO_UI_CONFIG_FILE;
  if (!configFile) {
    return Response.json(
      {
        success: false,
        error:
          "Config writing is not enabled. Set TENSORZERO_UI_CONFIG_FILE environment variable.",
      } as WriteAllConfigsResponse,
      { status: 400 },
    );
  }

  try {
    // Create ConfigWriter and write all config writes from the session
    const configWriter = await ConfigWriter.new(configFile);
    const result = await writeSessionConfigWritesToFile(
      configWriter,
      sessionId,
      {
        baseUrl: env.TENSORZERO_GATEWAY_URL,
        apiKey: env.TENSORZERO_API_KEY,
      },
    );

    return Response.json({
      success: true,
      results: result.results,
      total_processed: result.totalProcessed,
    } as WriteAllConfigsResponse);
  } catch (error) {
    logger.error("Failed to write configs:", error);
    const errorMessage =
      error instanceof Error ? error.message : "Unknown error occurred";
    return Response.json(
      { success: false, error: errorMessage } as WriteAllConfigsResponse,
      { status: 500 },
    );
  }
}
