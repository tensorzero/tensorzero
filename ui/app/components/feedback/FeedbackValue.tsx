import { useState } from "react";
import { Sheet, SheetContent } from "~/components/ui/sheet";
import type { FeedbackRow } from "~/utils/clickhouse/feedback";
import type { MetricConfig } from "~/utils/config/metric";
import {
  BooleanItem,
  FloatItem,
  CommentItem,
  DemonstrationItem,
  type FeedbackStatus,
} from "./FeedbackTableItem";
import { CommentModal, DemonstrationModal } from "./FeedbackTableModal";

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
  const [isSheetOpen, setIsSheetOpen] = useState(false);

  const handleClick = (event: React.MouseEvent) => {
    if (feedback.type === "comment" || feedback.type === "demonstration") {
      event.stopPropagation();
      setIsSheetOpen(true);
    }
  };

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

    return <BooleanItem value={feedback.value} status={status} />;
  }

  // Handle float metrics
  if (feedback.type === "float" && typeof feedback.value === "number") {
    return <FloatItem value={feedback.value} />;
  }

  // Handle comments
  if (feedback.type === "comment" && typeof feedback.value === "string") {
    if (truncate) {
      return (
        <>
          <CommentItem
            value={feedback.value}
            truncate={true}
            onClick={handleClick}
          />
          <Sheet open={isSheetOpen} onOpenChange={setIsSheetOpen}>
            <SheetContent className="bg-bg-secondary overflow-y-auto p-0 sm:max-w-xl md:max-w-2xl lg:max-w-3xl">
              <CommentModal feedback={feedback} />
            </SheetContent>
          </Sheet>
        </>
      );
    }

    return <CommentItem value={feedback.value} truncate={false} />;
  }

  // Handle demonstrations
  if (feedback.type === "demonstration" && typeof feedback.value === "string") {
    if (truncate) {
      return (
        <>
          <DemonstrationItem
            value={feedback.value}
            truncate={true}
            onClick={handleClick}
          />
          <Sheet open={isSheetOpen} onOpenChange={setIsSheetOpen}>
            <SheetContent className="bg-bg-secondary w-full overflow-y-auto p-0 sm:max-w-xl md:max-w-2xl lg:max-w-3xl">
              <DemonstrationModal feedback={feedback} />
            </SheetContent>
          </Sheet>
        </>
      );
    }

    return <DemonstrationItem value={feedback.value} truncate={false} />;
  }

  // Fallback for unexpected types
  return <div className="text-red-500">Invalid feedback type</div>;
}
