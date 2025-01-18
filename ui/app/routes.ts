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
    "api/function/:function_name/feedback_counts",
    "routes/api/function/$function_name/feedback_counts.route.ts",
  ),
  route(
    "observability/inferences",
    "routes/observability/inferences/route.tsx",
  ),
  route(
    "observability/episode/:episode_id",
    "routes/observability/episode/route.tsx",
  ),
  route(
    "observability/inference/:inference_id",
    "routes/observability/inference/route.tsx",
  ),
  route("observability/functions", "routes/observability/functions/route.tsx"),
  route(
    "observability/function/:function_name",
    "routes/observability/function/route.tsx",
  ),
  route("observability/episodes", "routes/observability/episodes/route.tsx"),
  route(
    "observability/function/:function_name/variant/:variant_name",
    "routes/observability/function/variant/route.tsx",
  ),
] satisfies RouteConfig;
