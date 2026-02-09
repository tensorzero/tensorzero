import type { ActionFunctionArgs } from "react-router";
import { ConfigApplier } from "tensorzero-node";
import type { GatewayEvent } from "~/types/tensorzero";
import { getEnv } from "~/utils/env.server";
import { logger } from "~/utils/logger";
import { extractEditPayloadsFromConfigWrite } from "~/utils/tensorzero/autopilot-client";

type ApplyConfigChangeRequest = {
  event: string;
};

type ApplyConfigChangeResponse =
  | { success: true; written_paths: string[] }
  | { success: false; error: string };

/**
 * API route for applying a single config change event to the filesystem.
 *
 * Route: POST /api/autopilot/sessions/:session_id/config-apply/apply
 *
 * Request body:
 * - event: GatewayEvent - The config change event to apply
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
      } as ApplyConfigChangeResponse,
      { status: 400 },
    );
  }

  if (request.method !== "POST") {
    return Response.json(
      {
        success: false,
        error: "Method not allowed",
      } as ApplyConfigChangeResponse,
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
      } as ApplyConfigChangeResponse,
      { status: 400 },
    );
  }

  let body: ApplyConfigChangeRequest;
  try {
    body = (await request.json()) as ApplyConfigChangeRequest;
  } catch {
    return Response.json(
      {
        success: false,
        error: "Invalid JSON body",
      } as ApplyConfigChangeResponse,
      { status: 400 },
    );
  }

  if (!body.event) {
    return Response.json(
      {
        success: false,
        error: "event is required",
      } as ApplyConfigChangeResponse,
      { status: 400 },
    );
  }

  let event: GatewayEvent;
  try {
    event = JSON.parse(body.event) as GatewayEvent;
  } catch {
    return Response.json(
      {
        success: false,
        error: "Invalid event JSON",
      } as ApplyConfigChangeResponse,
      { status: 400 },
    );
  }

  try {
    // Create ConfigApplier and apply the edits
    const configApplier = await ConfigApplier.new(configFile);
    const editPayloads = extractEditPayloadsFromConfigWrite(event);
    const writtenPaths: string[] = [];
    for (const editPayload of editPayloads) {
      const paths = await configApplier.applyEdit(editPayload);
      writtenPaths.push(...paths);
    }

    return Response.json({
      success: true,
      written_paths: writtenPaths,
    } as ApplyConfigChangeResponse);
  } catch (error) {
    logger.error("Failed to apply changes to the local filesystem:", error);
    const errorMessage =
      error instanceof Error ? error.message : "Unknown error occurred";
    return Response.json(
      { success: false, error: errorMessage } as ApplyConfigChangeResponse,
      { status: 500 },
    );
  }
}
