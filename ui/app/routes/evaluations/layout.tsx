import { Outlet, type RouteHandle } from "react-router";

export const handle: RouteHandle = {
  crumb: () => ["Evaluations"],
};

export default function EvaluationsLayout() {
  return <Outlet />;
}
