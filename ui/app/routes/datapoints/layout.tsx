import { Outlet, type RouteHandle } from "react-router";
import type { Route } from "./+types/layout";
import { LayoutErrorBoundary } from "~/components/ui/error";
import { logger } from "~/utils/logger";

export const handle: RouteHandle = {
  crumb: () => [{ label: "Datapoints", noLink: true }],
};

export default function DatapointsLayout() {
  return <Outlet />;
}

export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
  logger.error(error);
  return <LayoutErrorBoundary error={error} />;
}
