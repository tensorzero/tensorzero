import type { EvaluatorConfig } from "~/types/tensorzero";

export const getOptimize = (evaluatorConfig?: EvaluatorConfig) => {
  if (!evaluatorConfig) {
    return "max";
  }
  switch (evaluatorConfig.type) {
    case "exact_match":
    case "tool_use":
    case "regex":
      return "max";
    case "llm_judge":
      return evaluatorConfig.optimize;
  }
};

export const getMetricType = (
  evaluatorConfig: EvaluatorConfig,
): "boolean" | "float" => {
  switch (evaluatorConfig.type) {
    case "exact_match":
    case "tool_use":
    case "regex":
      return "boolean";
    case "llm_judge":
      return evaluatorConfig.output_type;
  }
};
