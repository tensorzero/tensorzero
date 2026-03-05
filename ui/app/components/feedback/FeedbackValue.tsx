import type { ReactNode } from "react";
import type { MetricConfig, FeedbackRow } from "~/types/tensorzero";
import { getFeedbackIcon } from "~/utils/icon";
import { UserFeedback } from "../icons/Icons";

function ValueItem({
  iconType,
  children,
}: {
  iconType: "success" | "failure" | "default" | "unknown" | "float";
  children: ReactNode;
}) {
  const { icon, iconBg } = getFeedbackIcon(iconType);

  return (
    <div className="flex items-center gap-2">
      <div
        className={`flex h-5 w-5 min-w-[1.25rem] items-center justify-center rounded-md ${iconBg}`}
      >
        {icon}
      </div>
      {children}
    </div>
  );
}

function ValueItemText({ children }: { children: ReactNode }) {
  return (
    <span className="overflow-hidden text-ellipsis whitespace-nowrap">
      {children}
    </span>
  );
}

interface FeedbackValueProps {
  feedback: FeedbackRow;
  metric?: MetricConfig;
}

export default function FeedbackValue({
  feedback,
  metric,
}: FeedbackValueProps) {
  const isHumanFeedback =
    feedback.tags["tensorzero::human_feedback"] === "true";

  if (feedback.type === "boolean" && typeof feedback.value === "boolean") {
    const optimize = metric?.type === "boolean" ? metric.optimize : "unknown";
    const success =
      (feedback.value === true && optimize === "max") ||
      (feedback.value === false && optimize === "min");

    const failure =
      (feedback.value === true && optimize === "min") ||
      (feedback.value === false && optimize === "max");

    let status: "success" | "failure" | "default" = "default";

    if (success) {
      status = "success";
    } else if (failure) {
      status = "failure";
    }

    return (
      <ValueItem iconType={status === "default" ? "unknown" : status}>
        <ValueItemText>{feedback.value ? "True" : "False"}</ValueItemText>
        {isHumanFeedback && <UserFeedback />}
      </ValueItem>
    );
  }

  if (feedback.type === "float" && typeof feedback.value === "number") {
    return (
      <ValueItem iconType="float">
        <ValueItemText>{feedback.value.toFixed(3)}</ValueItemText>
        {isHumanFeedback && <UserFeedback />}
      </ValueItem>
    );
  }

  return <div className="text-red-500">Invalid feedback type</div>;
}
