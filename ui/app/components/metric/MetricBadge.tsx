import { XCircle } from "lucide-react";
import clsx from "clsx";
import type { JsonValue } from "~/types/tensorzero";
import { isCutoffFailed } from "./MetricValue";

type MetricBadgeProps = {
  value: JsonValue | null | undefined;
  label?: string;
  error?: boolean;
  /** Optimization direction - used with cutoff to determine if value fails threshold */
  optimize?: "max" | "min";
  /** Threshold value - if optimize="max" and value < cutoff, shows red; if optimize="min" and value > cutoff, shows red */
  cutoff?: number;
};

// Default cutoff for boolean values (0.5 distinguishes true=1 from false=0)
const BOOLEAN_DEFAULT_CUTOFF = 0.5;

export function MetricBadge({
  value,
  label,
  error,
  optimize,
  cutoff,
}: MetricBadgeProps): React.ReactElement {
  const baseClasses =
    "inline-flex items-center gap-1.5 rounded-full font-mono whitespace-nowrap px-2 py-0.5 text-xs";

  if (error) {
    return (
      <span className={clsx(baseClasses, "bg-red-100 text-red-700")}>
        <XCircle className="h-2.5 w-2.5" />
        {label}
      </span>
    );
  }

  const displayValue = formatValue(value);
  const failsThreshold = checkFailsThreshold(value, optimize, cutoff);

  return (
    <span
      className={clsx(
        baseClasses,
        failsThreshold
          ? "bg-red-100 text-red-700"
          : "bg-gray-100 text-gray-900",
      )}
    >
      {label && (
        <span className={failsThreshold ? "text-red-500" : "text-gray-500"}>
          {label}
        </span>
      )}
      {displayValue}
    </span>
  );
}

function checkFailsThreshold(
  value: JsonValue | null | undefined,
  optimize?: "max" | "min",
  cutoff?: number,
): boolean {
  if (optimize === undefined) return false;

  // Handle boolean and numeric values using shared isCutoffFailed logic
  if (typeof value === "boolean") {
    // For booleans, use default cutoff of 0.5 (true=1, false=0)
    return isCutoffFailed(value, optimize, BOOLEAN_DEFAULT_CUTOFF);
  }

  if (typeof value === "number" && cutoff !== undefined) {
    return isCutoffFailed(value, optimize, cutoff);
  }

  return false;
}

function formatValue(value: JsonValue | null | undefined): string {
  if (value === null || value === undefined) {
    return "—";
  }
  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }
  if (typeof value === "number") {
    return String(value);
  }
  if (typeof value === "string") {
    return value;
  }
  return JSON.stringify(value);
}
