import { PostgresClient } from "tensorzero-node";
import { getEnv } from "./env.server";

let _postgresClient: PostgresClient | undefined;

const CONNECTION_TIMEOUT_MS = 5000;

export class PostgresConnectionError extends Error {
  constructor(
    message: string,
    public readonly cause?: unknown,
  ) {
    super(message);
    this.name = "PostgresConnectionError";
  }
}

export function isPostgresAvailable(): boolean {
  const env = getEnv();
  return !!env.TENSORZERO_POSTGRES_URL;
}

export async function getPostgresClient(): Promise<PostgresClient> {
  if (_postgresClient) {
    return _postgresClient;
  }

  const env = getEnv();
  if (!env.TENSORZERO_POSTGRES_URL) {
    throw new Error("TENSORZERO_POSTGRES_URL environment variable is not set");
  }

  try {
    const connectionPromise = PostgresClient.fromPostgresUrl(
      env.TENSORZERO_POSTGRES_URL,
    );

    const timeoutPromise = new Promise<never>((_, reject) => {
      setTimeout(() => {
        reject(
          new PostgresConnectionError(
            "Connection timed out. The database may be unavailable or slow to respond.",
          ),
        );
      }, CONNECTION_TIMEOUT_MS);
    });

    _postgresClient = await Promise.race([connectionPromise, timeoutPromise]);
    return _postgresClient;
  } catch (error) {
    if (error instanceof PostgresConnectionError) {
      throw error;
    }
    const message =
      error instanceof Error ? error.message : "Unable to connect to database";
    throw new PostgresConnectionError(message, error);
  }
}
