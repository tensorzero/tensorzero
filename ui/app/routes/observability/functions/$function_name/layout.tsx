import { Outlet, type RouteHandle } from "react-router";

export const handle: RouteHandle = {
  crumb: (match) => [match.params.function_name!],
};

export default function FunctionLayout() {
  return <Outlet />;
}
