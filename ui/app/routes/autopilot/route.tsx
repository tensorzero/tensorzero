import type { Route } from "./+types/route";
import { redirect } from "react-router";
import { checkAutopilotAvailable } from "~/utils/config/index.server";
import { AutopilotTeaser } from "./AutopilotTeaser";

export async function loader({ request }: Route.LoaderArgs) {
  const autopilotAvailable = false; // await checkAutopilotAvailable();
  if (autopilotAvailable) {
    const url = new URL(request.url);
    return redirect(`/autopilot/sessions${url.search}`);
  }
  return null;
}

export default function AutopilotIndexRoute() {
  return <AutopilotTeaser />;
}
