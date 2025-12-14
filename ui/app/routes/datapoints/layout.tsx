import { Outlet, type RouteHandle } from "react-router";

export const handle: RouteHandle = {
  crumb: () => [{ label: "Datapoints", noLink: true }],
};

export default function DatapointsLayout() {
  return <Outlet />;
}
