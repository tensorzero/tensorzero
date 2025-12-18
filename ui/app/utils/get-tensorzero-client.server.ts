import { TensorZeroClient } from "~/utils/tensorzero/tensorzero";
import { getEnv } from "./env.server";

let _tensorZeroClient: TensorZeroClient | undefined;

export function getTensorZeroClient(): TensorZeroClient {
  if (_tensorZeroClient) {
    return _tensorZeroClient;
  }
  const env = getEnv();
  _tensorZeroClient = new TensorZeroClient(
    env.TENSORZERO_GATEWAY_URL,
    env.TENSORZERO_API_KEY,
  );
  return _tensorZeroClient;
}
