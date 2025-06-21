import { Outlet, type RouteHandle } from "react-router";

export const handle: RouteHandle = {
  crumb: (match) => [match.params.evaluation_name!],
};

export default function EvaluationLayout() {
  return <Outlet />;
}
