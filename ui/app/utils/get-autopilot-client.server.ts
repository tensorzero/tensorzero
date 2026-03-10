import { AutopilotClient } from "~/utils/tensorzero/autopilot-client";
import { getEnv } from "./env.server";
import { getEffectiveApiKey } from "./api-key-override.server";

let _envClient: AutopilotClient | undefined;

export function getAutopilotClient(): AutopilotClient {
  const env = getEnv();

  // Env var path: cached singleton (same key every request)
  if (env.TENSORZERO_API_KEY) {
    _envClient ??= new AutopilotClient(
      env.TENSORZERO_GATEWAY_URL,
      env.TENSORZERO_API_KEY,
      env.TENSORZERO_AUTOPILOT_BETA_TOOLS,
    );
    return _envClient;
  }

  // Cookie path: fresh client per call (construction is trivial)
  return new AutopilotClient(
    env.TENSORZERO_GATEWAY_URL,
    getEffectiveApiKey(),
    env.TENSORZERO_AUTOPILOT_BETA_TOOLS,
  );
}
