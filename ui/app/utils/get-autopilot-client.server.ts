import { AutopilotClient } from "~/utils/tensorzero/autopilot-client";
import { getEnv } from "./env.server";

let _autopilotClient: AutopilotClient | undefined;
let _activeApiKey: string | undefined;
let _apiKeyOverride: string | undefined;

export function setAutopilotApiKeyOverride(key: string): void {
  _apiKeyOverride = key;
  _autopilotClient = undefined;
  _activeApiKey = undefined;
}

export function getAutopilotClient(): AutopilotClient {
  const env = getEnv();
  const effectiveKey = env.TENSORZERO_API_KEY ?? _apiKeyOverride;

  if (_autopilotClient && _activeApiKey === effectiveKey) {
    return _autopilotClient;
  }

  _autopilotClient = new AutopilotClient(
    env.TENSORZERO_GATEWAY_URL,
    effectiveKey,
  );
  _activeApiKey = effectiveKey;
  return _autopilotClient;
}
