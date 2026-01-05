import { Outlet, type RouteHandle } from "react-router";

export const handle: RouteHandle = {
  crumb: () => ["Sessions"],
};

export default function AutopilotSessionsLayout() {
  return <Outlet />;
}
