/**
 * Database helpers for autopilot e2e tests.
 *
 * These functions insert/query events directly in the autopilot Postgres
 * database, bypassing the API. Useful for injecting server-only event types
 * (like `user_questions`) that can't be created via the client API.
 *
 * Uses `docker exec` to run psql inside the Postgres container so that
 * no local psql installation is required.
 */
import { execSync } from "node:child_process";

const POSTGRES_CONTAINER =
  process.env.TENSORZERO_AUTOPILOT_POSTGRES_CONTAINER || "autopilot-postgres-1";

const PSQL_PREFIX = `docker exec -i ${POSTGRES_CONTAINER} psql -U postgres -d autopilot_api -p 5433`;

/**
 * Insert an event directly into the autopilot events table.
 * Uses Postgres dollar-quoting to avoid JSON escaping issues.
 */
export function insertEvent(
  eventId: string,
  sessionId: string,
  payload: object,
): void {
  const payloadJson = JSON.stringify(payload);
  const sql = `INSERT INTO autopilot.events (id, payload, session_id) VALUES ('${eventId}', $json$${payloadJson}$json$::jsonb, '${sessionId}')`;
  execSync(`${PSQL_PREFIX} -q`, {
    input: sql,
    timeout: 5000,
    stdio: ["pipe", "pipe", "pipe"],
  });
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
