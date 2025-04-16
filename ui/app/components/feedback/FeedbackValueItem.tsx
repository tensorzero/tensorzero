import type { ReactNode } from "react";
import { getFeedbackIcon } from "~/utils/icon";

export type FeedbackStatus = "success" | "failure" | "neutral";

interface ValueItemProps {
  status: FeedbackStatus;
  children: ReactNode;
  onClick?: (event: React.MouseEvent) => void;
}

function ValueItem({ status, children, onClick }: ValueItemProps) {
  const { icon, iconBg } = getFeedbackIcon(status);

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

function BooleanItem({
  value,
  status,
}: {
  value: boolean;
  status: FeedbackStatus;
}) {
  return (
    <ValueItem status={status}>
      <span>{value ? "True" : "False"}</span>
    </ValueItem>
  );
}

function FloatItem({ value }: { value: number }) {
  return (
    <ValueItem status="neutral">
      <span>{value.toFixed(3)}</span>
    </ValueItem>
  );
}

function CommentItem({
  value,
  onClick,
}: {
  value: string;
  onClick?: (event: React.MouseEvent) => void;
}) {
  return (
    <ValueItem status="neutral" onClick={onClick}>
      <span className="overflow-hidden text-ellipsis whitespace-nowrap">
        {value}
      </span>
    </ValueItem>
  );
}

function DemonstrationItem({
  value,
  onClick,
}: {
  value: string;
  onClick?: (event: React.MouseEvent) => void;
}) {
  return (
    <ValueItem status="neutral" onClick={onClick}>
      <span className="overflow-hidden font-mono text-ellipsis whitespace-nowrap">
        {value}
      </span>
    </ValueItem>
  );
}

// Exports
export { ValueItem, BooleanItem, FloatItem, CommentItem, DemonstrationItem };
