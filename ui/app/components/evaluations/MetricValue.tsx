import type { EvaluatorConfig } from "~/utils/config/evaluations";
import { getOptimize } from "~/utils/config/evaluations";

// Format metric value display component
export default function MetricValue({
  value,
  metricType,
  evaluatorConfig,
  className = "",
}: {
  value: string;
  metricType: "boolean" | "float" | "comment" | "demonstration";
  evaluatorConfig?: EvaluatorConfig;
  className?: string;
}): React.ReactElement {
  if (metricType === "boolean") {
    const boolValue = value === "true" || value === "1";
    const optimize = getOptimize(evaluatorConfig);
    const failed =
      (!boolValue && optimize === "max") || (boolValue && optimize === "min");

    return (
      <span
        className={`flex items-center gap-2 whitespace-nowrap ${failed ? "text-red-700" : "text-gray-700"} ${className}`}
      >
        <div
          className={`h-2 w-2 rounded-full ${
            failed ? "bg-red-700" : "bg-gray-700"
          }`}
        />
        {boolValue ? "True" : "False"}
      </span>
    );
  } else if (metricType === "float") {
    // Try to parse as number
    const numValue = parseFloat(value);
    if (!isNaN(numValue)) {
      // Check if value fails the cutoff criteria
      const failsCutoff = evaluatorConfig
        ? isCutoffFailed(numValue, evaluatorConfig)
        : false;
      return (
        <span
          className={`whitespace-nowrap ${failsCutoff ? "text-red-700" : "text-gray-700"} ${className}`}
        >
          {numValue}
        </span>
      );
    }
  }

  // Default case: return as string
  return <span className={`whitespace-nowrap ${className}`}>{value}</span>;
}

export function isCutoffFailed(
  value: number | boolean,
  evaluatorConfig: EvaluatorConfig,
) {
  const numericValue = typeof value === "number" ? value : value ? 1 : 0;
  const optimize = getOptimize(evaluatorConfig);
  if (evaluatorConfig.cutoff === undefined) {
    return false;
  }
  if (optimize === "max") {
    return numericValue < evaluatorConfig.cutoff;
  } else {
    return numericValue > evaluatorConfig.cutoff;
  }
}
