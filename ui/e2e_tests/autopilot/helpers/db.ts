/**
 * Database helpers for autopilot e2e tests.
 *
 * These functions insert/query events directly in the autopilot Postgres
 * database, bypassing the API. Useful for injecting server-only event types
 * (like `user_questions`) that can't be created via the client API.
 */
import { execSync } from "node:child_process";

const POSTGRES_URL =
  process.env.TENSORZERO_AUTOPILOT_POSTGRES_URL ||
  "postgres://postgres:postgres@localhost:5433/autopilot_api";

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
  execSync(`psql "${POSTGRES_URL}" -q`, {
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
  const result = execSync(
    `psql "${POSTGRES_URL}" -tAc "SELECT payload::text FROM autopilot.events WHERE session_id = '${sessionId}' AND payload->>'type' = '${eventType}' ORDER BY created_at DESC"`,
    { timeout: 5000, encoding: "utf-8" },
  );
  return result
    .trim()
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}
