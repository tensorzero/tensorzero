import {
  type RouteConfig,
  index,
  prefix,
  route,
} from "@react-router/dev/routes";

export default [
  index("routes/index.tsx"),

  // API routes
  ...prefix("api", [
    route(
      "curated_inferences/count",
      "routes/api/curated_inferences/count.route.ts",
    ),

    ...prefix("datasets", [
      route("counts", "routes/api/datasets/counts.route.ts"),
      route("count_inserts", "routes/api/datasets/count_inserts.route.ts"),
      route(
        "count/dataset/:dataset_name/function/:function_name",
        "routes/api/datasets/count_dataset_function.route.ts",
      ),
    ]),

    route(
      "workflow_evaluations/search_runs",
      "routes/api/workflow_evaluations/search_runs/route.ts",
    ),

    route(
      "evaluations/search_runs/:evaluation_name",
      "routes/api/evaluations/search_runs/$evaluation_name/route.ts",
    ),

    route(
      "function/:function_name/feedback_counts",
      "routes/api/function/$function_name/feedback_counts.route.ts",
    ),

    ...prefix("tensorzero", [
      route("inference", "routes/api/tensorzero/inference.ts"),
      route("status", "routes/api/tensorzero/status.ts"),
    ]),

    route(
      "inference/:inference_id",
      "routes/api/inference/$inference_id/route.ts",
    ),

    route(
      "datasets/datapoints/from-inference",
      "routes/api/datasets/datapoints/from-inference/route.ts",
    ),

    route("feedback", "routes/api/feedback/route.ts"),
  ]),

  // Datasets
  route("datasets", "routes/datasets/layout.tsx", [
    index("routes/datasets/route.tsx"),
    route("builder", "routes/datasets/builder/route.tsx"),
    route(":dataset_name", "routes/datasets/$dataset_name/layout.tsx", [
      index("routes/datasets/$dataset_name/route.tsx"),
      route(
        "datapoint/:id",
        "routes/datasets/$dataset_name/datapoint/$id/route.tsx",
      ),
    ]),
  ]),

  // Evaluations
  route("evaluations", "routes/evaluations/layout.tsx", [
    index("routes/evaluations/route.tsx"),
    route(
      ":evaluation_name",
      "routes/evaluations/$evaluation_name/layout.tsx",
      [
        index("routes/evaluations/$evaluation_name/route.tsx"),
        route(
          ":datapoint_id",
          "routes/evaluations/$evaluation_name/$datapoint_id/route.tsx",
        ),
      ],
    ),
  ]),

  // Workflow Evaluations (formerly Dynamic Evaluations)
  route("workflow_evaluations", "routes/workflow_evaluations/layout.tsx", [
    index("routes/workflow_evaluations/route.tsx"),
    route("runs/:run_id", "routes/workflow_evaluations/runs/$run_id/route.tsx"),
    route(
      "projects/:project_name",
      "routes/workflow_evaluations/projects/$project_name/route.tsx",
    ),
  ]),

  // Playground
  route("playground", "routes/playground/route.tsx"),

  // Observability
  ...prefix("observability", [
    route("functions", "routes/observability/functions/layout.tsx", [
      index("routes/observability/functions/route.tsx"),
      route(
        ":function_name",
        "routes/observability/functions/$function_name/layout.tsx",
        [
          index("routes/observability/functions/$function_name/route.tsx"),
          route(
            "variants/:variant_name",
            "routes/observability/functions/$function_name/variants/route.tsx",
          ),
        ],
      ),
    ]),
    route("inferences", "routes/observability/inferences/layout.tsx", [
      index("routes/observability/inferences/route.tsx"),
      route(
        ":inference_id",
        "routes/observability/inferences/$inference_id/route.tsx",
      ),
    ]),
    route("episodes", "routes/observability/episodes/layout.tsx", [
      index("routes/observability/episodes/route.tsx"),
      route(
        ":episode_id",
        "routes/observability/episodes/$episode_id/route.tsx",
      ),
    ]),

    route("models", "routes/observability/models/route.tsx"),
  ]),

  // Optimization
  route(
    "optimization/supervised-fine-tuning/:job_id?",
    "routes/optimization/supervised-fine-tuning/route.tsx",
  ),

  // API Keys
  route("api-keys", "routes/api-keys/route.tsx"),

  // Health
  route("health", "routes/health/route.tsx"),
] satisfies RouteConfig;
