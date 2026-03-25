/**
 * Database helpers for autopilot e2e tests.
 *
 * Event writes use the test-only `/v1/sessions/{id}/test/insert-db-event`
 * endpoint (gated behind the `e2e_tests` feature flag) to inject server-only
 * event types (like `user_questions`) that can't be created via the normal
 * client API.
 *
 * Event reads still use `docker exec` psql for simplicity.
 */
import { execSync } from "node:child_process";

const AUTOPILOT_API_URL =
  process.env.TENSORZERO_AUTOPILOT_API_URL || "http://localhost:4444";
const AUTOPILOT_API_KEY = process.env.TENSORZERO_AUTOPILOT_API_KEY || "";

const POSTGRES_CONTAINER =
  process.env.TENSORZERO_AUTOPILOT_POSTGRES_CONTAINER || "autopilot-postgres-1";
const POSTGRES_PORT = process.env.TENSORZERO_AUTOPILOT_POSTGRES_PORT || "5433";

const PSQL_PREFIX = `docker exec -i ${POSTGRES_CONTAINER} psql -U postgres -d autopilot_api -p ${POSTGRES_PORT}`;

/**
 * Insert an event via the test-only insert-db-event API endpoint.
 */
export async function insertEvent(
  eventId: string,
  sessionId: string,
  payload: object,
): Promise<void> {
  const url = `${AUTOPILOT_API_URL}/v1/sessions/${sessionId}/test/insert-db-event`;
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(AUTOPILOT_API_KEY
        ? { Authorization: `Bearer ${AUTOPILOT_API_KEY}` }
        : {}),
    },
    body: JSON.stringify({ id: eventId, payload }),
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(
      `insert-db-event failed (${response.status}): ${body}`,
    );
  }
}

/**
 * Query events from the autopilot events table by session and type.
 * Returns parsed JSON payloads in reverse chronological order.
 */
export function queryEventPayloads(
  sessionId: string,
  eventType: string,
): object[] {
  const sql = `SELECT payload::text FROM autopilot.events WHERE session_id = '${sessionId}' AND payload->>'type' = '${eventType}' ORDER BY created_at DESC`;
  const result = execSync(`${PSQL_PREFIX} -tA`, {
    input: sql,
    timeout: 5000,
    encoding: "utf-8",
    stdio: ["pipe", "pipe", "pipe"],
  });
  return result
    .trim()
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}
