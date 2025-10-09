import { checkClickHouseConnection } from "~/utils/clickhouse/client.server";
import { getConfig } from "~/utils/config/index.server";

export async function loader() {
  const [config, connectedToClickHouse] = await Promise.all([
    getConfig(),
    checkClickHouseConnection(),
  ]);
  if (!config) {
    throw new Error("Config not found");
  }
  if (!connectedToClickHouse) {
    throw new Error("Failed to connect to ClickHouse");
  }

  return new Response(null, { status: 200 });
}
