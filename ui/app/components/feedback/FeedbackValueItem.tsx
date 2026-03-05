import type { ReactNode } from "react";
import { getFeedbackIcon } from "~/utils/icon";
import { UserFeedback } from "../icons/Icons";

interface ValueItemProps {
  iconType: "success" | "failure" | "default" | "unknown" | "float";
  children: ReactNode;
}

function ValueItem({ iconType, children }: ValueItemProps) {
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

function BooleanItem({
  value,
  status,
  isHumanFeedback,
}: {
  value: boolean;
  status: "success" | "failure" | "default";
  isHumanFeedback: boolean;
}) {
  return (
    <ValueItem iconType={status === "default" ? "unknown" : status}>
      <ValueItemText>{value ? "True" : "False"}</ValueItemText>
      {isHumanFeedback && <UserFeedback />}
    </ValueItem>
  );
}

function FloatItem({
  value,
  isHumanFeedback,
}: {
  value: number;
  isHumanFeedback: boolean;
}) {
  return (
    <ValueItem iconType="float">
      <ValueItemText>{value.toFixed(3)}</ValueItemText>
      {isHumanFeedback && <UserFeedback />}
    </ValueItem>
  );
}

export { ValueItem, BooleanItem, FloatItem };
