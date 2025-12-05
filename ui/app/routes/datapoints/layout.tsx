import { Outlet, type RouteHandle } from "react-router";

export const handle: RouteHandle = {
  crumb: () => ["Datapoints"],
};

export default function DatapointsLayout() {
  return <Outlet />;
}
