import { createClient } from "@clickhouse/client";

export const clickhouseClient = createClient({
  url: process.env.CLICKHOUSE_URL,
});

export async function checkClickhouseConnection(): Promise<boolean> {
  try {
    await clickhouseClient.ping();
    return true;
  } catch (error) {
    return false;
  }
}
