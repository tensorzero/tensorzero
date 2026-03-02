import { AutopilotClient } from "~/utils/tensorzero/autopilot-client";
import { getEnv } from "./env.server";
import {
  getApiKeyOverride,
  getApiKeyOverrideVersion,
} from "./api-key-override.server";

let _autopilotClient: AutopilotClient | undefined;
let _lastOverrideVersion = -1;

export function getAutopilotClient(): AutopilotClient {
  const env = getEnv();
  const overrideVersion = getApiKeyOverrideVersion();

  if (_autopilotClient && _lastOverrideVersion === overrideVersion) {
    return _autopilotClient;
  }

  const effectiveKey = env.TENSORZERO_API_KEY ?? getApiKeyOverride();
  _autopilotClient = new AutopilotClient(
    env.TENSORZERO_GATEWAY_URL,
    effectiveKey,
  );
  _lastOverrideVersion = overrideVersion;
  return _autopilotClient;
}
