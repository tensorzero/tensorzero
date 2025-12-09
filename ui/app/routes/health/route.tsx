import { checkClickHouseConnection } from "~/utils/clickhouse/client.server";

export async function loader() {
  // Health check only verifies database connectivity.
  // Config loading is handled separately and has its own caching mechanism.
  // This allows the health check to work even when gateway auth is enabled
  // but no API key is configured yet.
  const connectedToClickHouse = await checkClickHouseConnection();
  if (!connectedToClickHouse) {
    throw new Error("Failed to connect to ClickHouse");
  }

  return new Response(null, { status: 200 });
}
