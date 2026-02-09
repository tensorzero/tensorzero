import type { ActionFunctionArgs } from "react-router";
import { ConfigWriter } from "tensorzero-node";
import { getEnv } from "~/utils/env.server";
import { getAutopilotClient } from "~/utils/get-autopilot-client.server";
import { logger } from "~/utils/logger";
import {
  extractEditPayloadsFromConfigWrite,
  listAllConfigWrites,
} from "~/utils/tensorzero/autopilot-client";

/**
 * Result of writing a config write to file.
 */
interface ApplyConfigChangeWriteResult {
  /** The event ID that was processed */
  eventId: string;
  /** Paths of files that were written */
  writtenPaths: string[];
}

type WriteAllConfigsResponse =
  | {
      success: true;
      results: ApplyConfigChangeWriteResult[];
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
    // Fetch all config writes for the session, paginating through results
    const allConfigWrites = await listAllConfigWrites(
      getAutopilotClient(),
      sessionId,
    );

    // Create ConfigWriter and write all config writes
    const configWriter = await ConfigWriter.new(configFile);
    const results: ApplyConfigChangeWriteResult[] = [];

    for (const event of allConfigWrites) {
      const editPayloads = extractEditPayloadsFromConfigWrite(event);
      const writtenPaths: string[] = [];
      for (const editPayload of editPayloads) {
        const paths = await configWriter.applyEdit(editPayload);
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
    } as WriteAllConfigsResponse);
  } catch (error) {
    logger.error("Failed to apply changes to the local filesystem:", error);
    const errorMessage =
      error instanceof Error ? error.message : "Unknown error occurred";
    return Response.json(
      { success: false, error: errorMessage } as WriteAllConfigsResponse,
      { status: 500 },
    );
  }
}
