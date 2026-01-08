import { Outlet, type RouteHandle } from "react-router";
import type { Route } from "./+types/layout";
import { RouteErrorContent } from "~/components/ui/error";
import { logger } from "~/utils/logger";

export const handle: RouteHandle = {
  crumb: () => ["Evaluations"],
};

export default function EvaluationsLayout() {
  return <Outlet />;
}

export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
  logger.error(error);
  return <RouteErrorContent error={error} />;
}
