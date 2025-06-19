import { Outlet, type RouteHandle } from "react-router";

export const handle: RouteHandle = {
  crumb: () => ["Inferences"],
};

export default function InferencesLayout() {
  return <Outlet />;
}
