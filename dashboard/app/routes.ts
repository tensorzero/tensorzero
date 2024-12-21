import { type RouteConfig, route } from "@react-router/dev/routes";

export default [
  route(
    "optimization/fine-tuning",
    "routes/optimization/fine-tuning/route.tsx",
  ),
  route(
    "api/curated_inferences/count",
    "routes/api/curated_inferences/count.route.ts",
  ),
] satisfies RouteConfig;
