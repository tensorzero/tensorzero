import type { ReactNode } from "react";
import { cn } from "~/utils/common";
import { getFeedbackIcon } from "~/utils/icon";

export type FeedbackStatus = "success" | "failure" | "neutral";

interface FeedbackItemProps {
  status: FeedbackStatus;
  children: ReactNode;
  className?: string;
  onClick?: (event: React.MouseEvent) => void;
}

function FeedbackItem({
  status,
  children,
  className,
  onClick,
}: FeedbackItemProps) {
  const { icon, iconBg } = getFeedbackIcon(status);

  const itemClass = cn(
    "flex items-center gap-2",
    onClick &&
      "cursor-pointer transition-colors duration-300 hover:text-gray-500",
    className,
  );

  return (
    <div className={itemClass} onClick={onClick}>
      <div
        className={cn(
          "flex h-5 w-5 min-w-[1.25rem] items-center justify-center rounded-md",
          iconBg,
        )}
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
    <FeedbackItem status={status}>
      <span>{value ? "True" : "False"}</span>
    </FeedbackItem>
  );
}

function FloatItem({ value }: { value: number }) {
  return (
    <FeedbackItem status="neutral">
      <div>{value.toFixed(3)}</div>
    </FeedbackItem>
  );
}

function CommentItem({
  value,
  truncate = true,
  onClick,
}: {
  value: string;
  truncate?: boolean;
  onClick?: (event: React.MouseEvent) => void;
}) {
  if (truncate) {
    return (
      <FeedbackItem status="neutral" onClick={onClick}>
        <div className="overflow-hidden text-ellipsis whitespace-nowrap">
          {value}
        </div>
      </FeedbackItem>
    );
  }

  return (
    <FeedbackItem status="neutral">
      <div className="break-words whitespace-pre-wrap">{value}</div>
    </FeedbackItem>
  );
}

function DemonstrationItem({
  value,
  truncate = true,
  onClick,
}: {
  value: string;
  truncate?: boolean;
  onClick?: (event: React.MouseEvent) => void;
}) {
  if (truncate) {
    return (
      <FeedbackItem status="neutral" onClick={onClick}>
        <div className="overflow-hidden font-mono text-ellipsis whitespace-nowrap">
          {value}
        </div>
      </FeedbackItem>
    );
  }

  return (
    <FeedbackItem status="neutral">
      <div className="font-mono break-words whitespace-pre-wrap">{value}</div>
    </FeedbackItem>
  );
}

// Exports
export { FeedbackItem, BooleanItem, FloatItem, CommentItem, DemonstrationItem };
