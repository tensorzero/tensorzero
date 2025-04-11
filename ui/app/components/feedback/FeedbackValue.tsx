import { Code } from "~/components/ui/code";
import type { FeedbackRow } from "~/utils/clickhouse/feedback";
import type { MetricConfig } from "~/utils/config/metric";
import { cn } from "~/utils/common";
import { getFeedbackIcon } from "~/utils/icon";
import type { ReactNode } from "react";

type FeedbackStatus = "success" | "failure" | "neutral";

interface FeedbackItemProps {
  status: FeedbackStatus;
  children: ReactNode;
  className?: string;
}

function FeedbackItem({ 
  status, 
  children, 
  className 
}: FeedbackItemProps) {
  const { icon, iconBg } = getFeedbackIcon(status);
  
  return (
    <div className={cn("flex items-center gap-2", className)}>
      <div className={cn("flex h-5 w-5 items-center justify-center rounded-md", iconBg)}>
        {icon}
      </div>
      {children}
    </div>
  );
}

interface FeedbackValueProps {
  feedback: FeedbackRow;
  metric?: MetricConfig;
  truncate?: boolean;
}

export default function FeedbackValue({
  feedback,
  metric,
  truncate = true,
}: FeedbackValueProps) {
  // Handle boolean metrics
  if (feedback.type === "boolean" && typeof feedback.value === "boolean") {
    const optimize = metric?.type === "boolean" ? metric.optimize : "unknown";
    const success =
      (feedback.value === true && optimize === "max") ||
      (feedback.value === false && optimize === "min");

    const failure =
      (feedback.value === true && optimize === "min") ||
      (feedback.value === false && optimize === "max");

    let status: FeedbackStatus = "neutral";

    if (success) {
      status = "success";
    } else if (failure) {
      status = "failure";
    }

    return (
      <FeedbackItem status={status}>
        <span>{feedback.value ? "True" : "False"}</span>
      </FeedbackItem>
    );
  }

  // Handle float metrics
  if (feedback.type === "float" && typeof feedback.value === "number") {
    return (
      <FeedbackItem status="neutral">
        <div>{feedback.value.toFixed(3)}</div>
      </FeedbackItem>
    );
  }

  // Handle comments and demonstrations (both have string values)
  if (feedback.type === "comment" && typeof feedback.value === "string") {
    if (truncate) {
      return (
        <FeedbackItem status="neutral">
          <div className="overflow-hidden text-ellipsis whitespace-nowrap">
            {feedback.value}
          </div>
        </FeedbackItem>
      );
    }
    return (
      <FeedbackItem status="neutral">
        <div className="whitespace-pre-wrap break-words">{feedback.value}</div>
      </FeedbackItem>
    );
  }

  if (feedback.type === "demonstration" && typeof feedback.value === "string") {
    if (truncate) {
      return (
        <FeedbackItem status="neutral">
          <div className="overflow-hidden text-ellipsis whitespace-nowrap font-mono">
            {feedback.value}
          </div>
        </FeedbackItem>
      );
    }
    return (
      <FeedbackItem status="neutral">
        <div className="font-normal text-sm">
          <Code>{feedback.value}</Code>
        </div>
      </FeedbackItem>
    );
  }

  // Fallback for unexpected types
  return <div className="text-red-500">Invalid feedback type</div>;
}
