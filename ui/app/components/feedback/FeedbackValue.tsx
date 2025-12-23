import { useState } from "react";
import { Sheet, SheetContent } from "~/components/ui/sheet";
import type { MetricConfig, FeedbackRow } from "~/types/tensorzero";
import {
  BooleanItem,
  FloatItem,
  CommentItem,
  DemonstrationItem,
} from "./FeedbackValueItem";
import { CommentModal, DemonstrationModal } from "./FeedbackTableModal";

interface FeedbackValueProps {
  feedback: FeedbackRow;
  metric?: MetricConfig;
}

export default function FeedbackValue({
  feedback,
  metric,
}: FeedbackValueProps) {
  const [isSheetOpen, setIsSheetOpen] = useState(false);

  const handleClick = (event: React.MouseEvent) => {
    if (feedback.type === "comment" || feedback.type === "demonstration") {
      event.stopPropagation();
      setIsSheetOpen(true);
    }
  };
  const isHumanFeedback =
    feedback.tags["tensorzero::human_feedback"] === "true";

  // Handle boolean metrics
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
      <BooleanItem
        value={feedback.value}
        status={status}
        isHumanFeedback={isHumanFeedback}
      />
    );
  }

  // Handle float metrics
  if (feedback.type === "float" && typeof feedback.value === "number") {
    return (
      <FloatItem value={feedback.value} isHumanFeedback={isHumanFeedback} />
    );
  }

  // Handle comments
  if (feedback.type === "comment" && typeof feedback.value === "string") {
    return (
      <>
        <CommentItem
          value={feedback.value}
          isHumanFeedback={isHumanFeedback}
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

  // Handle demonstrations
  if (feedback.type === "demonstration" && typeof feedback.value === "string") {
    return (
      <>
        <DemonstrationItem
          value={feedback.value}
          isHumanFeedback={isHumanFeedback}
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

  // Fallback for unexpected types
  return <div className="text-red-500">Invalid feedback type</div>;
}
