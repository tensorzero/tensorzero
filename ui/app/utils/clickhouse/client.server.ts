import { createClient } from "@clickhouse/client";

// Ensure this only runs on the server
if (typeof window !== "undefined") {
  throw new Error("clickhouseClient can only be used on the server side");
}

export const clickhouseClient = createClient({
  url:
    process.env.TENSORZERO_CLICKHOUSE_URL ??
    (() => {
      if (process.env.CLICKHOUSE_URL) {
        console.warn(
          'Deprecation Warning: The environment variable "CLICKHOUSE_URL" has been renamed to "TENSORZERO_CLICKHOUSE_URL" and will be removed in a future version. Please update your environment to use "TENSORZERO_CLICKHOUSE_URL" instead.',
        );
        return process.env.CLICKHOUSE_URL;
      }
      return undefined;
    })(),
});

export async function checkClickHouseConnection(): Promise<boolean> {
  try {
    const result = await clickhouseClient.ping();
    return result.success;
  } catch {
    return false;
  }
}
