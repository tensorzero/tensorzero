import type { Route } from "./+types/route";
import { redirect } from "react-router";

export async function loader({ request }: Route.LoaderArgs) {
  const url = new URL(request.url);
  return redirect(`/autopilot/sessions${url.search}`);
}

export default function AutopilotIndexRoute() {
  return null;
}
