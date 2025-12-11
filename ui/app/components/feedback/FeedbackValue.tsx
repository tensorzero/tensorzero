import { useState } from "react";
import { Sheet, SheetContent } from "~/components/ui/sheet";
import type { MetricConfig, FeedbackRow } from "~/types/tensorzero";
import { MetricBadge } from "~/components/metric/MetricBadge";
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
    event.stopPropagation();
    setIsSheetOpen(true);
  };

  const isHumanFeedback =
    feedback.tags["tensorzero::human_feedback"] === "true";

  if (feedback.type === "boolean" && typeof feedback.value === "boolean") {
    const optimize = metric?.type === "boolean" ? metric.optimize : undefined;

    return (
      <MetricBadge
        value={feedback.value}
        metricType="boolean"
        optimize={optimize}
        isHumanFeedback={isHumanFeedback}
      />
    );
  }

  if (feedback.type === "float" && typeof feedback.value === "number") {
    const optimize = metric?.type === "float" ? metric.optimize : undefined;

    return (
      <MetricBadge
        value={feedback.value}
        metricType="float"
        precision={3}
        optimize={optimize}
        isHumanFeedback={isHumanFeedback}
      />
    );
  }

  if (feedback.type === "comment" && typeof feedback.value === "string") {
    return (
      <>
        <MetricBadge
          value={feedback.value}
          onClick={handleClick}
          maxWidth="200px"
          isHumanFeedback={isHumanFeedback}
        />
        <Sheet open={isSheetOpen} onOpenChange={setIsSheetOpen}>
          <SheetContent className="bg-bg-secondary overflow-y-auto p-0 sm:max-w-xl md:max-w-2xl lg:max-w-3xl">
            <CommentModal feedback={feedback} />
          </SheetContent>
        </Sheet>
      </>
    );
  }

  if (feedback.type === "demonstration" && typeof feedback.value === "string") {
    return (
      <>
        <MetricBadge
          value={feedback.value}
          onClick={handleClick}
          maxWidth="200px"
          isHumanFeedback={isHumanFeedback}
        />
        <Sheet open={isSheetOpen} onOpenChange={setIsSheetOpen}>
          <SheetContent className="bg-bg-secondary w-full overflow-y-auto p-0 sm:max-w-xl md:max-w-2xl lg:max-w-3xl">
            <DemonstrationModal feedback={feedback} />
          </SheetContent>
        </Sheet>
      </>
    );
  }

  return <div className="text-red-500">Invalid feedback type</div>;
}
