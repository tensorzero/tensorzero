import { Outlet, type RouteHandle } from "react-router";

export const handle: RouteHandle = {
  crumb: (match) => [match.params.dataset_name!],
};

export default function DatasetLayout() {
  return <Outlet />;
}
