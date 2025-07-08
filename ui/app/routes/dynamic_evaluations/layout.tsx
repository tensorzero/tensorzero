import { Outlet, type RouteHandle } from "react-router";

export const handle: RouteHandle = {
  crumb: () => ["Dynamic Evaluations"],
};

export default function DynamicEvaluationsLayout() {
  return <Outlet />;
}
