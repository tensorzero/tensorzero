import { TensorZeroClient } from "~/utils/tensorzero/tensorzero";
import { getEnv } from "./env.server";
import { getEffectiveApiKey } from "./api-key-override.server";

let _envClient: TensorZeroClient | undefined;

export function getTensorZeroClient(): TensorZeroClient {
  const env = getEnv();

  // Env var path: cached singleton (same key every request)
  if (env.TENSORZERO_API_KEY) {
    _envClient ??= new TensorZeroClient(
      env.TENSORZERO_GATEWAY_URL,
      env.TENSORZERO_API_KEY,
    );
    return _envClient;
  }

  // Cookie path: fresh client per call (construction is trivial)
  return new TensorZeroClient(env.TENSORZERO_GATEWAY_URL, getEffectiveApiKey());
}
