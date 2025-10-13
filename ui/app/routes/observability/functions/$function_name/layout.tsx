import { Outlet, type RouteHandle } from "react-router";

export const handle: RouteHandle = {
  crumb: (match) => [
    { label: match.params.function_name!, isIdentifier: true },
  ],
};

export default function FunctionLayout() {
  return <Outlet />;
}
