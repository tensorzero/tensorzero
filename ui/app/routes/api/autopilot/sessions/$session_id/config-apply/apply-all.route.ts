import type { ActionFunctionArgs } from "react-router";
import { ConfigApplier } from "tensorzero-node";
import { getEnv } from "~/utils/env.server";
import { getAutopilotClient } from "~/utils/get-autopilot-client.server";
import { logger } from "~/utils/logger";
import {
  extractEditPayloadsFromConfigWrite,
  listAllConfigWrites,
} from "~/utils/tensorzero/autopilot-client";

/**
 * Result of applying a config change to file.
 */
interface ApplyConfigResult {
  /** The event ID that was processed */
  eventId: string;
  /** Paths of files that were written */
  writtenPaths: string[];
}

type ApplyAllConfigsResponse =
  | {
      success: true;
      results: ApplyConfigResult[];
      total_processed: number;
    }
  | { success: false; error: string };

/**
 * API route for applying all config change events from a session to the filesystem.
 *
 * Route: POST /api/autopilot/sessions/:session_id/config-apply/apply-all
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
      } as ApplyAllConfigsResponse,
      { status: 400 },
    );
  }

  if (request.method !== "POST") {
    return Response.json(
      {
        success: false,
        error: "Method not allowed",
      } as ApplyAllConfigsResponse,
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
      } as ApplyAllConfigsResponse,
      { status: 400 },
    );
  }

  try {
    // Fetch all config writes for the session, paginating through results
    const allConfigWrites = await listAllConfigWrites(
      getAutopilotClient(),
      sessionId,
    );

    // Create ConfigApplier and apply all config changes
    const configApplier = await ConfigApplier.new(configFile);
    const results: ApplyConfigResult[] = [];

    for (const event of allConfigWrites) {
      const editPayloads = extractEditPayloadsFromConfigWrite(event);
      const writtenPaths: string[] = [];
      for (const editPayload of editPayloads) {
        const paths = await configApplier.applyEdit(editPayload);
        writtenPaths.push(...paths);
      }
      results.push({
        eventId: event.id,
        writtenPaths,
      });
    }

    return Response.json({
      success: true,
      results,
      total_processed: results.length,
    } as ApplyAllConfigsResponse);
  } catch (error) {
    logger.error("Failed to apply changes to the local filesystem:", error);
    const errorMessage =
      error instanceof Error ? error.message : "Unknown error occurred";
    return Response.json(
      { success: false, error: errorMessage } as ApplyAllConfigsResponse,
      { status: 500 },
    );
  }
}
