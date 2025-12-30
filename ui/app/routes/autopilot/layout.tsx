import { Outlet, type RouteHandle } from "react-router";

export const handle: RouteHandle = {
  crumb: () => ["Autopilot"],
};

export default function AutopilotLayout() {
  return <Outlet />;
}
