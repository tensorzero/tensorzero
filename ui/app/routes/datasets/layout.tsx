import { Outlet, type RouteHandle } from "react-router";

export const handle: RouteHandle = {
  crumb: () => ["Datasets"],
};

export default function DatasetsLayout() {
  return <Outlet />;
}
