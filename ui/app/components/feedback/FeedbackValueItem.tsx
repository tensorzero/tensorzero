import type { ReactNode } from "react";
import { getFeedbackIcon } from "~/utils/icon";
import { UserFeedback } from "../icons/Icons";

interface ValueItemProps {
  iconType:
    | "success"
    | "failure"
    | "default"
    | "unknown"
    | "float"
    | "comment"
    | "demonstration";
  children: ReactNode;
  onClick?: (event: React.MouseEvent) => void;
}

function ValueItem({ iconType, children, onClick }: ValueItemProps) {
  const { icon, iconBg } = getFeedbackIcon(iconType);

  return (
    <div
      className={
        onClick
          ? "flex cursor-pointer items-center gap-2 transition-colors duration-300 hover:text-gray-500"
          : "flex items-center gap-2"
      }
      onClick={onClick}
    >
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
    <span className="flex-1 overflow-hidden text-ellipsis whitespace-nowrap">
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

function CommentItem({
  value,
  isHumanFeedback,
  onClick,
}: {
  value: string;
  isHumanFeedback: boolean;
  onClick?: (event: React.MouseEvent) => void;
}) {
  return (
    <ValueItem iconType="comment" onClick={onClick}>
      <ValueItemText>{value}</ValueItemText>
      {isHumanFeedback && <UserFeedback />}
    </ValueItem>
  );
}

function DemonstrationItem({
  value,
  isHumanFeedback,
  onClick,
}: {
  value: string;
  isHumanFeedback: boolean;
  onClick?: (event: React.MouseEvent) => void;
}) {
  return (
    <ValueItem iconType="demonstration" onClick={onClick}>
      <ValueItemText>
        <span className="font-mono">{value}</span>
      </ValueItemText>
      {isHumanFeedback && <UserFeedback />}
    </ValueItem>
  );
}

// Exports
export { ValueItem, BooleanItem, FloatItem, CommentItem, DemonstrationItem };
