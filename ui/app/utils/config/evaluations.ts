import type { EvaluatorConfig } from "tensorzero-node";

export const getOptimize = (evaluatorConfig?: EvaluatorConfig) => {
  if (!evaluatorConfig) {
    return "max";
  }
  switch (evaluatorConfig.type) {
    case "exact_match":
      return "max";
    case "llm_judge":
      return evaluatorConfig.optimize;
    case "regex":
      return "max";
  }
};

export const getMetricType = (
  evaluatorConfig: EvaluatorConfig,
): "boolean" | "float" => {
  switch (evaluatorConfig.type) {
    case "exact_match":
      return "boolean";
    case "llm_judge":
      return evaluatorConfig.output_type;
    case "regex":
      return "boolean";
  }
};
