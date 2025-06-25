import { createClient } from "@clickhouse/client";
import { canUseDOM, isErrorLike } from "../common";

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

export const clickhouseClient = (() => {
  const url = getClickhouseUrl();
  try {
    return createClient({ url });
  } catch (error) {
    throw new ClickHouseClientError(
      "Failed to create ClickHouse client. Please ensure that the `TENSORZERO_CLICKHOUSE_URL` environment variable is set correctly and that the ClickHouse server is running.\n\n" +
        "Failed with the following message:\n\n" +
        (isErrorLike(error) ? error.message : String(error)),
    );
  }
})();

export async function checkClickHouseConnection(): Promise<boolean> {
  try {
    const result = await clickhouseClient.ping();
    return result.success;
  } catch {
    return false;
  }
}

function getClickhouseUrl() {
  const url = process.env.TENSORZERO_CLICKHOUSE_URL;
  if (url) {
    return url;
  }

  if (process.env.CLICKHOUSE_URL) {
    console.warn(
      'Deprecation Warning: The environment variable "CLICKHOUSE_URL" has been renamed to "TENSORZERO_CLICKHOUSE_URL" and will be removed in a future version. Please update your environment to use "TENSORZERO_CLICKHOUSE_URL" instead.',
    );
    return process.env.CLICKHOUSE_URL;
  }

  throw new ClickHouseClientError(
    "The environment variable `TENSORZERO_CLICKHOUSE_URL` is required.",
  );
}

export function isClickHouseClientError(
  error: unknown,
): error is ClickHouseClientError {
  return isErrorLike(error) && error.name === "ClickHouseClientError";
}
