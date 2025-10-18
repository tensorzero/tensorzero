import { Outlet, type RouteHandle } from "react-router";

export const handle: RouteHandle = {
  crumb: () => ["Workflow Evaluations"],
};

export default function DynamicEvaluationsLayout() {
  return <Outlet />;
}
