import { Outlet, type RouteHandle } from "react-router";

export const handle: RouteHandle = {
  crumb: (match) => [{ label: match.params.dataset_name!, isIdentifier: true }],
};

export default function DatasetLayout() {
  return <Outlet />;
}
