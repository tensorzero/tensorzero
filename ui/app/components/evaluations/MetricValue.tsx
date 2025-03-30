import { Check } from "lucide-react";
import type { EvaluatorConfig } from "~/utils/config/evals";

import { X } from "lucide-react";
import { getOptimize } from "~/utils/config/evals";

// Format metric value display component
export default function MetricValue({
  value,
  metricType,
  evaluatorConfig,
  className = "",
}: {
  value: string;
  metricType: "boolean" | "float" | "comment" | "demonstration";
  evaluatorConfig: EvaluatorConfig;
  className?: string;
}): React.ReactElement {
  if (metricType === "boolean") {
    const boolValue = value === "true" || value === "1";
    const optimize = getOptimize(evaluatorConfig);
    const failed =
      (!boolValue && optimize === "max") || (boolValue && optimize === "min");
    const icon = failed ? (
      <X className="mr-1 h-3 w-3 flex-shrink-0" />
    ) : (
      <Check className="mr-1 h-3 w-3 flex-shrink-0" />
    );

    return (
      <span
        className={`flex items-center whitespace-nowrap ${failed ? "text-red-700" : "text-gray-700"} ${className}`}
      >
        {icon}
        {boolValue ? "True" : "False"}
      </span>
    );
  } else if (metricType === "float") {
    // Try to parse as number
    const numValue = parseFloat(value);
    if (!isNaN(numValue)) {
      // Check if value fails the cutoff criteria
      const failsCutoff = isCutoffFailed(numValue, evaluatorConfig);
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
