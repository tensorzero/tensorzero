import { TensorZeroClient } from "~/utils/tensorzero/tensorzero";
import { getEnv } from "./env.server";

let _tensorZeroClient: TensorZeroClient | undefined;
let _activeApiKey: string | undefined;
let _apiKeyOverride: string | undefined;

export function setApiKeyOverride(key: string): void {
  _apiKeyOverride = key;
  // Invalidate cached client so it gets recreated with the new key
  _tensorZeroClient = undefined;
  _activeApiKey = undefined;
}

export function getTensorZeroClient(): TensorZeroClient {
  const env = getEnv();
  const effectiveKey = env.TENSORZERO_API_KEY ?? _apiKeyOverride;

  if (_tensorZeroClient && _activeApiKey === effectiveKey) {
    return _tensorZeroClient;
  }

  _tensorZeroClient = new TensorZeroClient(
    env.TENSORZERO_GATEWAY_URL,
    effectiveKey,
  );
  _activeApiKey = effectiveKey;
  return _tensorZeroClient;
}
