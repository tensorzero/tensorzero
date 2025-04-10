import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "~/components/ui/dialog";
import { useConfig } from "~/context/config";
import { MetricSelector } from "../metric/MetricSelector";
import { useState } from "react";
import { RadioGroup, RadioGroupItem } from "~/components/ui/radio-group";
import { Label } from "~/components/ui/label";
import { Input } from "~/components/ui/input";
import { Textarea } from "~/components/ui/textarea";

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
  const metrics = config.metrics;
  const [selectedMetricName, setSelectedMetricName] = useState<string>("");
  const selectedMetric = metrics[selectedMetricName];
  const selectedMetricType = selectedMetric?.type;
  const [booleanValue, setBooleanValue] = useState<string | null>(null);
  const [floatValue, setFloatValue] = useState<string>("");
  const [commentValue, setCommentValue] = useState<string>("");

  return (
    <div>
      <MetricSelector
        metrics={metrics}
        selectedMetric={selectedMetricName}
        onMetricChange={setSelectedMetricName}
      />
      {selectedMetric && selectedMetricType === "boolean" && (
        <BooleanFeedbackInput
          metricName={selectedMetricName}
          value={booleanValue}
          onChange={setBooleanValue}
        />
      )}
      {selectedMetric && selectedMetricType === "float" && (
        <FloatFeedbackInput value={floatValue} onChange={setFloatValue} />
      )}
      {selectedMetric && selectedMetricType === "comment" && (
        <CommentFeedbackInput value={commentValue} onChange={setCommentValue} />
      )}
    </div>
  );
}

interface BooleanFeedbackInputProps {
  metricName: string;
  value: string | null;
  onChange: (value: string | null) => void;
}

function BooleanFeedbackInput({ value, onChange }: BooleanFeedbackInputProps) {
  return (
    <div className="mt-4">
      <Label>Value</Label>
      <RadioGroup
        value={value ?? undefined}
        onValueChange={onChange}
        className="mt-2 flex gap-4"
      >
        <div className="flex items-center space-x-2">
          <RadioGroupItem value="true" id={`true`} />
          <Label htmlFor={`true`}>True</Label>
        </div>
        <div className="flex items-center space-x-2">
          <RadioGroupItem value="false" id={`false`} />
          <Label htmlFor={`false`}>False</Label>
        </div>
      </RadioGroup>
    </div>
  );
}

interface FloatFeedbackInputProps {
  value: string;
  onChange: (value: string) => void;
}

function FloatFeedbackInput({ value, onChange }: FloatFeedbackInputProps) {
  return (
    <div className="mt-4 space-y-2">
      <Label htmlFor="float-input">Value</Label>
      <Input
        id="float-input"
        type="number"
        step="any"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="Enter a number"
      />
    </div>
  );
}

interface CommentFeedbackInputProps {
  value: string;
  onChange: (value: string) => void;
}

function CommentFeedbackInput({ value, onChange }: CommentFeedbackInputProps) {
  return (
    <div className="mt-4 space-y-2">
      <Label htmlFor="comment-input">Comment</Label>
      <Textarea
        id="comment-input"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="Enter your comment"
        rows={4}
      />
    </div>
  );
}
