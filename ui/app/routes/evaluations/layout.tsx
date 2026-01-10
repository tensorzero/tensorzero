import { Outlet, type RouteHandle } from "react-router";
import type { Route } from "./+types/layout";
import { LayoutErrorBoundary } from "~/components/ui/error";

export const handle: RouteHandle = {
  crumb: () => ["Evaluations"],
};

export default function EvaluationsLayout() {
  return <Outlet />;
}

export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
  return <LayoutErrorBoundary error={error} />;
}
