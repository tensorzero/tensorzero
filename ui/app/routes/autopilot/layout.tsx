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
import { getTensorZeroClient } from "~/utils/get-tensorzero-client.server";
import { EntitySideSheetProvider } from "~/components/entity-sheet/EntitySideSheetContext";
import { EntitySideSheet } from "~/components/entity-sheet/EntitySideSheet";

export const handle: RouteHandle = {
  crumb: () => [{ label: "Autopilot", noLink: true }],
};

const AUTOPILOT_UNAVAILABLE_ERROR = "Autopilot Unavailable";

export async function loader() {
  // Check if autopilot is configured on the gateway
  const client = getTensorZeroClient();
  const status = await client.getAutopilotStatus();
  if (!status.enabled) {
    throw data({ errorType: AUTOPILOT_UNAVAILABLE_ERROR }, { status: 501 });
  }
  return null;
}

export default function AutopilotLayout() {
  return (
    <EntitySideSheetProvider>
      <Outlet />
      <EntitySideSheet />
    </EntitySideSheetProvider>
  );
}

export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
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
