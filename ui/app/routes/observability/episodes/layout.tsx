import { Outlet, type RouteHandle } from "react-router";

export const handle: RouteHandle = {
  crumb: () => ["Episodes"],
};

export default function EpisodesLayout() {
  return <Outlet />;
}
