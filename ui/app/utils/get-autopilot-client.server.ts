import { AutopilotClient } from "~/utils/tensorzero/autopilot-client";
import { getEnv } from "./env.server";
import {
  getEffectiveApiKey,
  getApiKeyOverrideVersion,
} from "./api-key-override.server";

let _autopilotClient: AutopilotClient | undefined;
let _lastOverrideVersion = -1;

export function getAutopilotClient(): AutopilotClient {
  const overrideVersion = getApiKeyOverrideVersion();

  if (_autopilotClient && _lastOverrideVersion === overrideVersion) {
    return _autopilotClient;
  }

  _autopilotClient = new AutopilotClient(
    getEnv().TENSORZERO_GATEWAY_URL,
    getEffectiveApiKey(),
  );
  _lastOverrideVersion = overrideVersion;
  return _autopilotClient;
}
