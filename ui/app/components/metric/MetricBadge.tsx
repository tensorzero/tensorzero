import * as React from "react";
import { cva } from "class-variance-authority";
import { XCircle } from "lucide-react";
import { cn } from "~/utils/common";
import { UserFeedback } from "../icons/Icons";
import type { JsonValue } from "~/types/tensorzero";

const MetricBadgeSize = {
  SMALL: "sm",
  MEDIUM: "md",
} as const;

type MetricBadgeSize = (typeof MetricBadgeSize)[keyof typeof MetricBadgeSize];

type MetricType = "boolean" | "float";

const metricBadgeVariants = cva(
  "inline-flex items-center gap-1 rounded-full font-mono whitespace-nowrap",
  {
    variants: {
      variant: {
        default: "bg-gray-100 text-gray-800",
        failed: "bg-red-100 text-red-700",
      },
      size: {
        sm: "px-2 py-0.5 text-xs",
        md: "px-2.5 py-0.5 text-sm",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "sm",
    },
  },
);

type MetricBadgeProps = {
  /** The metric value to display */
  value: JsonValue | null | undefined;
  /** Optional label shown before value in a lighter shade */
  label?: string;
  /** Metric type determines how value is formatted and styled */
  metricType?: MetricType;
  /** Optimization direction - determines if value is "failed" */
  optimize?: "min" | "max";
  /** Cutoff threshold for float metrics */
  cutoff?: number;
  /** Shows user feedback icon indicating human override */
  isHumanFeedback?: boolean;
  /** Size variant */
  size?: MetricBadgeSize;
  /** Error state - shows error icon with label */
  error?: boolean;
  /** Optional hover actions (shown on hover to the right of the badge) */
  children?: React.ReactNode;
  /** Click handler - makes badge interactive */
  onClick?: (event: React.MouseEvent) => void;
  /** Decimal precision for float values (e.g., 3 for "0.123") */
  precision?: number;
  /** Maximum width for truncation (e.g., "200px", "15rem") */
  maxWidth?: string;
  /** Additional CSS classes for the badge container */
  className?: string;
};

export function MetricBadge({
  value,
  label,
  metricType,
  optimize,
  cutoff,
  isHumanFeedback,
  size,
  error,
  children,
  onClick,
  precision,
  maxWidth,
  className,
}: MetricBadgeProps): React.ReactElement {
  if (error) {
    const badge = (
      <span
        className={cn(
          metricBadgeVariants({ variant: "failed", size }),
          className,
        )}
      >
        <XCircle className="h-2.5 w-2.5" />
        {label}
      </span>
    );
    return children ? wrapWithHoverActions(badge, children) : badge;
  }

  const { displayValue, isFailed } = computeDisplayValue(
    value,
    metricType,
    optimize,
    cutoff,
    precision,
  );

  const isClickable = Boolean(onClick);
  const BadgeElement = isClickable ? "button" : "span";

  const badge = (
    <BadgeElement
      className={cn(
        metricBadgeVariants({ variant: isFailed ? "failed" : "default", size }),
        isClickable && "cursor-pointer transition-opacity hover:opacity-80",
        maxWidth && "overflow-hidden text-ellipsis",
        className,
      )}
      style={maxWidth ? { maxWidth } : undefined}
      onClick={onClick}
      type={isClickable ? "button" : undefined}
    >
      {label && (
        <span className={isFailed ? "text-red-600" : "text-gray-500"}>
          {label}
        </span>
      )}
      {displayValue}
      {isHumanFeedback && <UserFeedback />}
    </BadgeElement>
  );

  return children ? wrapWithHoverActions(badge, children) : badge;
}

function wrapWithHoverActions(
  badge: React.ReactElement,
  children: React.ReactNode,
): React.ReactElement {
  return (
    <div className="group relative inline-flex items-center">
      {badge}
      <div className="absolute right-0 flex translate-x-full gap-1 pl-1 opacity-0 transition-opacity duration-200 group-hover:opacity-100">
        {children}
      </div>
    </div>
  );
}

function computeDisplayValue(
  value: JsonValue | null | undefined,
  metricType?: MetricType,
  optimize?: "min" | "max",
  cutoff?: number,
  precision?: number,
): { displayValue: string; isFailed: boolean } {
  if (value === null || value === undefined) {
    return { displayValue: "—", isFailed: false };
  }

  if (metricType === "boolean") {
    const boolValue =
      typeof value === "boolean"
        ? value
        : value === "true" || value === "1" || value === 1;
    const isFailed =
      optimize !== undefined &&
      ((!boolValue && optimize === "max") || (boolValue && optimize === "min"));
    return { displayValue: boolValue ? "true" : "false", isFailed };
  }

  if (metricType === "float") {
    const numValue =
      typeof value === "number" ? value : parseFloat(String(value));
    if (!isNaN(numValue)) {
      const isFailed =
        cutoff !== undefined &&
        optimize !== undefined &&
        isCutoffFailed(numValue, optimize, cutoff);
      const displayValue =
        precision !== undefined
          ? numValue.toFixed(precision)
          : String(numValue);
      return { displayValue, isFailed };
    }
  }

  if (typeof value === "boolean") {
    return { displayValue: value ? "true" : "false", isFailed: false };
  }
  if (typeof value === "number") {
    const displayValue =
      precision !== undefined ? value.toFixed(precision) : String(value);
    return { displayValue, isFailed: false };
  }
  if (typeof value === "string") {
    return { displayValue: value, isFailed: false };
  }
  return { displayValue: JSON.stringify(value), isFailed: false };
}

export function isCutoffFailed(
  value: number,
  optimize: "min" | "max",
  cutoff: number,
): boolean {
  if (optimize === "max") {
    return value < cutoff;
  } else {
    return value > cutoff;
  }
}

export { MetricBadgeSize };
