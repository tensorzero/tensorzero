import type { ActionFunctionArgs } from "react-router";
import {
  ConfigWriter,
  listConfigWrites,
  writeConfigWriteToFile,
} from "tensorzero-node";
import { getEnv } from "~/utils/env.server";
import { logger } from "~/utils/logger";

type WriteConfigRequest = {
  event_id: string;
};

type WriteConfigResponse =
  | { success: true; written_paths: string[] }
  | { success: false; error: string };

/**
 * API route for writing a single config write event to the filesystem.
 *
 * Route: POST /api/autopilot/sessions/:session_id/config-writes/write
 *
 * Request body:
 * - event_id: string - ID of the config write event to write
 *
 * Response:
 * - { success: true, written_paths: string[] } on success
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
      } as WriteConfigResponse,
      { status: 400 },
    );
  }

  if (request.method !== "POST") {
    return Response.json(
      { success: false, error: "Method not allowed" } as WriteConfigResponse,
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
      } as WriteConfigResponse,
      { status: 400 },
    );
  }

  let body: WriteConfigRequest;
  try {
    body = (await request.json()) as WriteConfigRequest;
  } catch {
    return Response.json(
      { success: false, error: "Invalid JSON body" } as WriteConfigResponse,
      { status: 400 },
    );
  }

  if (!body.event_id) {
    return Response.json(
      { success: false, error: "event_id is required" } as WriteConfigResponse,
      { status: 400 },
    );
  }

  try {
    // Fetch config writes for the session
    const configWritesResponse = await listConfigWrites(sessionId, {
      baseUrl: env.TENSORZERO_GATEWAY_URL,
      apiKey: env.TENSORZERO_API_KEY,
    });

    // Find the event by ID
    const event = configWritesResponse.config_writes.find(
      (e) => e.id === body.event_id,
    );

    if (!event) {
      return Response.json(
        {
          success: false,
          error: `Config write event with ID ${body.event_id} not found`,
        } as WriteConfigResponse,
        { status: 404 },
      );
    }

    // Create ConfigWriter and apply the edit
    const configWriter = await ConfigWriter.new(configFile);
    const result = await writeConfigWriteToFile(configWriter, event);

    return Response.json({
      success: true,
      written_paths: result.writtenPaths,
    } as WriteConfigResponse);
  } catch (error) {
    logger.error("Failed to write config:", error);
    const errorMessage =
      error instanceof Error ? error.message : "Unknown error occurred";
    return Response.json(
      { success: false, error: errorMessage } as WriteConfigResponse,
      { status: 500 },
    );
  }
}
