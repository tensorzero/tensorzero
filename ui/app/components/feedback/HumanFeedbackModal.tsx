import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "~/components/ui/dialog";
import { useConfig } from "~/context/config";
import { MetricSelector } from "../metric/MetricSelector";
import { useState, useEffect } from "react";
import { RadioGroup, RadioGroupItem } from "~/components/ui/radio-group";
import { Label } from "~/components/ui/label";
import { Input } from "~/components/ui/input";
import { Textarea } from "~/components/ui/textarea";
import type { ContentBlockOutput } from "~/utils/clickhouse/common";
import type { JsonInferenceOutput } from "~/utils/clickhouse/common";
import Output from "../inference/Output";
import { Link, useFetcher } from "react-router";
import { Button } from "~/components/ui/button";

interface HumanFeedbackModalProps {
  isOpen: boolean;
  onClose: () => void;
  inferenceOutput?: ContentBlockOutput[] | JsonInferenceOutput;
}

export function HumanFeedbackModal({
  isOpen,
  onClose,
  inferenceOutput,
}: HumanFeedbackModalProps) {
  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-h-[90vh] sm:max-w-[1200px]">
        <DialogHeader>
          <DialogTitle>Add Feedback</DialogTitle>
        </DialogHeader>
        <FeedbackForm inferenceOutput={inferenceOutput} onClose={onClose} />
      </DialogContent>
    </Dialog>
  );
}

function FeedbackForm({
  inferenceOutput,
  onClose,
}: {
  inferenceOutput?: ContentBlockOutput[] | JsonInferenceOutput;
  onClose: () => void;
}) {
  const config = useConfig();
  const fetcher = useFetcher();
  // If there is no inference output this is likely an episode-level feedback and
  // we should filter demonstration out of the list of metrics.
  const metrics =
    inferenceOutput === undefined
      ? Object.fromEntries(
          Object.entries(config.metrics).filter(
            ([, metric]) => metric.type !== "demonstration",
          ),
        )
      : config.metrics;
  const [selectedMetricName, setSelectedMetricName] = useState<string>("");
  const selectedMetric = metrics[selectedMetricName];
  const selectedMetricType = selectedMetric?.type;
  const [booleanValue, setBooleanValue] = useState<string | null>(null);
  const [floatValue, setFloatValue] = useState<string>("");
  const [commentValue, setCommentValue] = useState<string>("");
  const [demonstrationValue, setDemonstrationValue] = useState<
    ContentBlockOutput[] | JsonInferenceOutput | undefined
  >(inferenceOutput);

  // Close modal on successful submission
  useEffect(() => {
    // Check if the fetcher was submitting/loading and is now idle,
    // and if the action returned a success flag (e.g., { ok: true }).
    // Adjust `fetcher.data?.ok` based on your action's actual success response.
    if (fetcher.state === "idle" && fetcher.data?.ok === true) {
      onClose();
    }
    // Depend on state, data, and onClose to ensure effect runs correctly
  }, [fetcher.state, fetcher.data, onClose]);

  const isSubmitting = fetcher.state !== "idle";

  return (
    <fetcher.Form method="post" className="space-y-4">
      {selectedMetricName && (
        <input type="hidden" name="metricName" value={selectedMetricName} />
      )}
      <input type="hidden" name="type" value="humanFeedback" />

      <MetricSelector
        metrics={metrics}
        selectedMetric={selectedMetricName}
        onMetricChange={setSelectedMetricName}
      />

      {selectedMetric && selectedMetricType === "boolean" && (
        <>
          <BooleanFeedbackInput
            metricName={selectedMetricName}
            value={booleanValue}
            onChange={setBooleanValue}
          />
          {booleanValue !== null && (
            <input type="hidden" name="value" value={booleanValue} />
          )}
        </>
      )}

      {selectedMetric && selectedMetricType === "float" && (
        <>
          <FloatFeedbackInput value={floatValue} onChange={setFloatValue} />
          <input type="hidden" name="value" value={floatValue} />
        </>
      )}

      {selectedMetric && selectedMetricType === "comment" && (
        <>
          <CommentFeedbackInput
            value={commentValue}
            onChange={setCommentValue}
          />
          <input type="hidden" name="value" value={commentValue} />
        </>
      )}

      {selectedMetric &&
        selectedMetricType === "demonstration" &&
        (demonstrationValue ? (
          <>
            <Output
              output={demonstrationValue}
              isEditing={true}
              onOutputChange={setDemonstrationValue}
            />
            <input
              type="hidden"
              name="value"
              value={JSON.stringify(demonstrationValue)}
            />
          </>
        ) : (
          <div className="text-red-500">
            Initial output missing for demonstration value. This is most likely
            a bug. Please file a bug report{" "}
            <Link to="https://github.com/tensorzero/tensorzero/discussions/new?category=bug-reports">
              here
            </Link>
            .
          </div>
        ))}

      {fetcher.data?.error && (
        <p className="text-sm text-red-500">{fetcher.data.error}</p>
      )}

      <div className="flex justify-end">
        <Button type="submit" disabled={!selectedMetricName || isSubmitting}>
          {isSubmitting ? "Submitting..." : "Submit Feedback"}
        </Button>
      </div>
    </fetcher.Form>
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
