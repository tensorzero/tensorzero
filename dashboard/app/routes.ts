import { type RouteConfig, route } from "@react-router/dev/routes";

export default [
  route("/", "routes/index.tsx"),
  route(
    "optimization/supervised-fine-tuning/:job_id?",
    "routes/optimization/supervised-fine-tuning/route.tsx",
  ),
  route(
    "api/curated_inferences/count",
    "routes/api/curated_inferences/count.route.ts",
  ),
  route(
    "observability/inferences",
    "routes/observability/inferences/route.tsx",
  ),
] satisfies RouteConfig;
