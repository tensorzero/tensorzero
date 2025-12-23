import { PostgresClient } from "tensorzero-node";
import { getEnv } from "./env.server";

let _postgresClient: PostgresClient | undefined;

export function isPostgresAvailable(): boolean {
  const env = getEnv();
  return env.TENSORZERO_POSTGRES_URL !== null;
}

export async function getPostgresClient(): Promise<PostgresClient> {
  if (_postgresClient) {
    return _postgresClient;
  }

  const env = getEnv();
  if (!env.TENSORZERO_POSTGRES_URL) {
    throw new Error("TENSORZERO_POSTGRES_URL environment variable is not set");
  }

  _postgresClient = await PostgresClient.fromPostgresUrl(
    env.TENSORZERO_POSTGRES_URL,
  );
  return _postgresClient;
}
