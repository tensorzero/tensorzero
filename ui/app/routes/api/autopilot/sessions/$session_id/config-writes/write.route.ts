import type { ActionFunctionArgs } from "react-router";
import { ConfigWriter } from "tensorzero-node";
import type { GatewayEvent } from "~/types/tensorzero";
import { getEnv } from "~/utils/env.server";
import { logger } from "~/utils/logger";
import { extractEditPayloadsFromConfigWrite } from "~/utils/tensorzero/autopilot-client";

type WriteConfigRequest = {
  event: string;
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
 * - event: GatewayEvent - The config write event to write
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

  if (!body.event) {
    return Response.json(
      { success: false, error: "event is required" } as WriteConfigResponse,
      { status: 400 },
    );
  }

  let event: GatewayEvent;
  try {
    event = JSON.parse(body.event) as GatewayEvent;
  } catch {
    return Response.json(
      { success: false, error: "Invalid event JSON" } as WriteConfigResponse,
      { status: 400 },
    );
  }

  try {
    // Create ConfigWriter and apply the edits
    const configWriter = await ConfigWriter.new(configFile);
    const editPayloads = extractEditPayloadsFromConfigWrite(event);
    const writtenPaths: string[] = [];
    for (const editPayload of editPayloads) {
      const paths = await configWriter.applyEdit(editPayload);
      writtenPaths.push(...paths);
    }

    return Response.json({
      success: true,
      written_paths: writtenPaths,
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
