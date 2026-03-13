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
const POSTGRES_PORT = process.env.TENSORZERO_AUTOPILOT_POSTGRES_PORT || "5433";

const PSQL_PREFIX = `docker exec -i ${POSTGRES_CONTAINER} psql -U postgres -d autopilot_api -p ${POSTGRES_PORT}`;

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
  // Use the monotonic_message_id insert (new schema) with fallback to the old schema.
  // The new schema requires allocating a monotonic_message_id from the workspace counter.
  const sqlNew = `
    WITH workspace_lock AS (
      UPDATE autopilot.workspace w
      SET next_stream_id = w.next_stream_id + 1
      FROM autopilot.sessions s
      WHERE s.id = '${sessionId}' AND w.workspace_id = s.workspace_id
      RETURNING w.next_stream_id - 1 AS monotonic_message_id
    )
    INSERT INTO autopilot.events (id, payload, session_id, monotonic_message_id)
    SELECT '${eventId}', $json$${payloadJson}$json$::jsonb, '${sessionId}', workspace_lock.monotonic_message_id
    FROM workspace_lock`;
  const sqlOld = `INSERT INTO autopilot.events (id, payload, session_id) VALUES ('${eventId}', $json$${payloadJson}$json$::jsonb, '${sessionId}')`;
  let useOld = false;
  try {
    const result = execSync(`${PSQL_PREFIX}`, {
      input: sqlNew,
      timeout: 5000,
      encoding: "utf-8",
      stdio: ["pipe", "pipe", "pipe"],
    });
    // psql prints "INSERT 0 N" where N is the number of rows inserted.
    // If the CTE matched zero workspace rows, N will be 0 and the event was silently not inserted.
    if (!result.includes("INSERT 0 1")) {
      useOld = true;
    }
  } catch {
    useOld = true;
  }
  if (useOld) {
    execSync(`${PSQL_PREFIX} -q`, {
      input: sqlOld,
      timeout: 5000,
      stdio: ["pipe", "pipe", "pipe"],
    });
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
