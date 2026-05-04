/**
 * Thin proxy from `GET /api/providers/available` (UI server) to
 * `GET /internal/providers/available` (gateway).
 *
 * The UI's ModelPicker hits this so it doesn't have to know the gateway
 * URL or worry about the `/internal/` namespace; that stays a server-side
 * concern.
 */

import { getTensorZeroClient } from "~/utils/tensorzero.server";

export async function loader(): Promise<Response> {
  try {
    const client = getTensorZeroClient();
    // Plain fetch — the typed client doesn't have this method yet, and
    // we don't want to grow the typed client surface for every internal
    // endpoint. The UI consumes the JSON shape directly.
    const baseUrl = (client as unknown as { baseUrl: string }).baseUrl;
    const response = await fetch(`${baseUrl}/internal/providers/available`, {
      headers: { "content-type": "application/json" },
    });
    if (!response.ok) {
      const body = await response.text();
      return Response.json(
        { error: `gateway returned ${response.status}: ${body}` },
        { status: response.status },
      );
    }
    return Response.json(await response.json());
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    return Response.json({ error: message }, { status: 500 });
  }
}
