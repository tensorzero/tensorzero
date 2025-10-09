import { Outlet, type RouteHandle } from "react-router";

export const handle: RouteHandle = {
  crumb: (match) => [{ label: match.params.evaluation_name!, isIdentifier: true }],
};

export default function EvaluationLayout() {
  return <Outlet />;
}
