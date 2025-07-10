import { Outlet, type RouteHandle } from "react-router";

export const handle: RouteHandle = {
  crumb: () => ["Functions"],
};

export default function FunctionsLayout() {
  return <Outlet />;
}
