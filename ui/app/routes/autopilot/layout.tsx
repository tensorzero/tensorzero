import {
  data,
  isRouteErrorResponse,
  Outlet,
  type RouteHandle,
} from "react-router";
import type { Route } from "./+types/layout";
import { AutopilotUnavailableState } from "~/components/ui/error/AutopilotUnavailableState";
import { isAutopilotUnavailableError } from "~/utils/tensorzero/errors";
import { getAutopilotClient } from "~/utils/tensorzero.server";

export const handle: RouteHandle = {
  crumb: () => [{ label: "Autopilot", noLink: true }],
};

const AUTOPILOT_UNAVAILABLE_ERROR = "Autopilot Unavailable";

export async function loader() {
  // Check if autopilot is available by making a minimal request
  const client = getAutopilotClient();
  try {
    await client.listAutopilotSessions({ limit: 1 });
    return null;
  } catch (error) {
    if (isAutopilotUnavailableError(error)) {
      throw data({ errorType: AUTOPILOT_UNAVAILABLE_ERROR }, { status: 501 });
    }
    throw error;
  }
}

export default function AutopilotLayout() {
  return <Outlet />;
}

export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
  // Check if this is an autopilot unavailable error
  if (
    isRouteErrorResponse(error) &&
    error.data?.errorType === AUTOPILOT_UNAVAILABLE_ERROR
  ) {
    return <AutopilotUnavailableState />;
  }
  if (isAutopilotUnavailableError(error)) {
    return <AutopilotUnavailableState />;
  }

  // Re-throw other errors to be handled by parent error boundary
  throw error;
}
