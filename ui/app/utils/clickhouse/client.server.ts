import { createClient } from "@clickhouse/client";
import { canUseDOM, isErrorLike } from "../common";
import { getEnv } from "../env";

// Ensure this only runs on the server. Vite's React Router plugin should ensure
// this is unreachable since the filename ends with `.server.ts`, but this check
// adds additional assurance.
if (canUseDOM) {
  throw new Error("clickhouseClient can only be used on the server side");
}

class ClickHouseClientError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = "ClickHouseClientError";
  }
}

let _clickhouseClient: ReturnType<typeof createClient> | null = null;

export function getClickhouseClient(): ReturnType<typeof createClient> {
  if (_clickhouseClient) {
    return _clickhouseClient;
  }

  const env = getEnv();
  try {
    const client = createClient({ url: env.TENSORZERO_CLICKHOUSE_URL });
    _clickhouseClient = client;
    return client;
  } catch (error) {
    throw new ClickHouseClientError(
      "Failed to create ClickHouse client. Please ensure that the `TENSORZERO_CLICKHOUSE_URL` environment variable is set correctly and that the ClickHouse server is running.\n\n" +
        "Failed with the following message:\n\n" +
        (isErrorLike(error) ? error.message : String(error)),
      { cause: error },
    );
  }
}

export async function checkClickHouseConnection(): Promise<boolean> {
  try {
    const result = await getClickhouseClient().ping();
    return result.success;
  } catch {
    return false;
  }
}

export function isClickHouseClientError(
  error: unknown,
): error is ClickHouseClientError {
  return isErrorLike(error) && error.name === "ClickHouseClientError";
}
