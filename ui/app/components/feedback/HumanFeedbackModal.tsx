import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "~/components/ui/dialog";
import { useConfig } from "~/context/config";
import { MetricSelector } from "../metric/MetricSelector";
import { useState } from "react";

interface HumanFeedbackModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function HumanFeedbackModal({
  isOpen,
  onClose,
}: HumanFeedbackModalProps) {
  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-h-[90vh] sm:max-w-[1200px]">
        <DialogHeader>
          <DialogTitle>Add Feedback</DialogTitle>
        </DialogHeader>
        <FeedbackForm />
      </DialogContent>
    </Dialog>
  );
}

function FeedbackForm() {
  const config = useConfig();
  const [selectedMetric, setSelectedMetric] = useState<string>("");

  const metrics = config.metrics;

  return (
    <div>
      <MetricSelector
        metrics={metrics}
        selectedMetric={selectedMetric}
        onMetricChange={setSelectedMetric}
      />
    </div>
  );
}
