import {
  data,
  isRouteErrorResponse,
  Outlet,
  type RouteHandle,
} from "react-router";
import type { Route } from "./+types/layout";
import { AutopilotUnavailableState } from "~/components/ui/error/AutopilotUnavailableState";
import { LayoutErrorBoundary } from "~/components/ui/error";
import { isAutopilotUnavailableError } from "~/utils/tensorzero/errors";
import { getAutopilotClient } from "~/utils/tensorzero.server";
import { logger } from "~/utils/logger";

export const handle: RouteHandle = {
  crumb: () => [{ label: "Autopilot", noLink: true }],
};

const AUTOPILOT_UNAVAILABLE_ERROR = "Autopilot Unavailable";

export async function loader() {
  // Check if autopilot is available by making a minimal request
  const client = getAutopilotClient();
  try {
    // TODO: Use dedicated endpoint (#5489)
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
  logger.error(error);

  // Autopilot unavailable gets special treatment
  if (
    isRouteErrorResponse(error) &&
    error.data?.errorType === AUTOPILOT_UNAVAILABLE_ERROR
  ) {
    return <AutopilotUnavailableState />;
  }
  if (isAutopilotUnavailableError(error)) {
    return <AutopilotUnavailableState />;
  }

  // All other errors (including infra) handled by LayoutErrorBoundary
  return <LayoutErrorBoundary error={error} />;
}
