import { type RouteConfig, route } from "@react-router/dev/routes";

export default [
  route(
    "dashboard",
    "components/layouts/dashboard-layout.tsx",
    [
      route(
        "optimization/fine-tuning/:job_id?",
        "routes/optimization/fine-tuning/route.tsx",
      ),
      route(
        "api/curated_inferences/count",
        "routes/api/curated_inferences/count.route.ts",
      ),
    ]
  ),
] satisfies RouteConfig;