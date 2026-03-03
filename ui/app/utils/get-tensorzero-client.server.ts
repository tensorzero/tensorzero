import { TensorZeroClient } from "~/utils/tensorzero/tensorzero";
import { getEnv } from "./env.server";
import {
  getEffectiveApiKey,
  getApiKeyOverrideVersion,
} from "./api-key-override.server";

let _tensorZeroClient: TensorZeroClient | undefined;
let _lastOverrideVersion = -1;

export function getTensorZeroClient(): TensorZeroClient {
  const overrideVersion = getApiKeyOverrideVersion();

  if (_tensorZeroClient && _lastOverrideVersion === overrideVersion) {
    return _tensorZeroClient;
  }

  _tensorZeroClient = new TensorZeroClient(
    getEnv().TENSORZERO_GATEWAY_URL,
    getEffectiveApiKey(),
  );
  _lastOverrideVersion = overrideVersion;
  return _tensorZeroClient;
}
