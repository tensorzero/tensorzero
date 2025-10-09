import { DatabaseClient, TensorZeroClient } from "tensorzero-node";
import { getEnv } from "../env.server";

let _tensorZeroClient: TensorZeroClient | undefined;
export async function getNativeTensorZeroClient(): Promise<TensorZeroClient> {
  if (_tensorZeroClient) {
    return _tensorZeroClient;
  }

  const env = getEnv();
  _tensorZeroClient = await TensorZeroClient.buildHttp(
    env.TENSORZERO_GATEWAY_URL,
  );
  return _tensorZeroClient;
}

let _databaseClient: DatabaseClient | undefined;
export async function getNativeDatabaseClient(): Promise<DatabaseClient> {
  if (_databaseClient) {
    return _databaseClient;
  }

  const env = getEnv();
  _databaseClient = await DatabaseClient.fromClickhouseUrl(
    env.TENSORZERO_CLICKHOUSE_URL,
  );
  return _databaseClient;
}
