import { AutopilotClient } from "~/utils/tensorzero/autopilot-client";
import { getEnv } from "./env.server";

let _autopilotClient: AutopilotClient | undefined;

export function getAutopilotClient(): AutopilotClient {
  if (_autopilotClient) {
    return _autopilotClient;
  }
  const env = getEnv();
  _autopilotClient = new AutopilotClient(
    env.TENSORZERO_GATEWAY_URL,
    env.TENSORZERO_API_KEY,
  );
  return _autopilotClient;
}
