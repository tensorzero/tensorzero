import { XCircle } from "lucide-react";
import clsx from "clsx";
import type { JsonValue } from "~/types/tensorzero";

type MetricBadgeProps = {
  value: JsonValue | null | undefined;
  label?: string;
  error?: boolean;
};

export function MetricBadge({
  value,
  label,
  error,
}: MetricBadgeProps): React.ReactElement {
  const baseClasses =
    "inline-flex items-center gap-1 rounded-full font-mono whitespace-nowrap px-2 py-0.5 text-xs";

  if (error) {
    return (
      <span className={clsx(baseClasses, "bg-red-100 text-red-700")}>
        <XCircle className="h-2.5 w-2.5" />
        {label}
      </span>
    );
  }

  const displayValue = formatValue(value);

  return (
    <span className={clsx(baseClasses, "bg-gray-100 text-gray-800")}>
      {label && <span className="text-gray-500">{label}</span>}
      {displayValue}
    </span>
  );
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
