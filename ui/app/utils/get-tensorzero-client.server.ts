import { TensorZeroClient } from "~/utils/tensorzero/tensorzero";
import { getEnv } from "./env.server";
import {
  getApiKeyOverride,
  getApiKeyOverrideVersion,
} from "./api-key-override.server";

let _tensorZeroClient: TensorZeroClient | undefined;
let _lastOverrideVersion = -1;

export function getTensorZeroClient(): TensorZeroClient {
  const env = getEnv();
  const overrideVersion = getApiKeyOverrideVersion();

  if (_tensorZeroClient && _lastOverrideVersion === overrideVersion) {
    return _tensorZeroClient;
  }

  const effectiveKey = env.TENSORZERO_API_KEY ?? getApiKeyOverride();
  _tensorZeroClient = new TensorZeroClient(
    env.TENSORZERO_GATEWAY_URL,
    effectiveKey,
  );
  _lastOverrideVersion = overrideVersion;
  return _tensorZeroClient;
}
