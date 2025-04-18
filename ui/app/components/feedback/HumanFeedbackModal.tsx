import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "~/components/ui/dialog";
import { useConfig } from "~/context/config";
import MetricSelector from "../metric/MetricSelector";
import { useState } from "react";
import type { ContentBlockOutput } from "~/utils/clickhouse/common";
import type { JsonInferenceOutput } from "~/utils/clickhouse/common";
import Output from "../inference/Output";
import { Link, Form } from "react-router";
import { Button } from "~/components/ui/button";
import {
  filterMetricsByLevel,
  filterStaticEvaluationMetrics,
} from "~/utils/config/metric";
import BooleanFeedbackInput from "./BooleanFeedbackInput";
import FloatFeedbackInput from "./FloatFeedbackInput";
import CommentFeedbackInput from "./CommentFeedbackInput";

interface HumanFeedbackModalProps {
  isOpen: boolean;
  onClose: () => void;
  // This should be provided if the feedback is for an inference
  // and omitted if the feedback is for an episode
  inferenceOutput?: ContentBlockOutput[] | JsonInferenceOutput;
  // Exactly one of the following should be provided
  episodeId?: string;
  inferenceId?: string;
}

export function HumanFeedbackModal({
  isOpen,
  onClose,
  inferenceOutput,
  episodeId,
  inferenceId,
}: HumanFeedbackModalProps) {
  // Exactly one of episodeId and inferenceId should be provided
  if ((episodeId && inferenceId) || (!episodeId && !inferenceId)) {
    throw new Error(
      "Exactly one of episodeId and inferenceId should be provided. This is a bug. Please file a bug report at https://github.com/tensorzero/tensorzero/discussions/new?category=bug-reports",
    );
  }
  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-h-[90vh] sm:max-w-[1200px]">
        <DialogHeader>
          <DialogTitle>Add Feedback</DialogTitle>
        </DialogHeader>
        <FeedbackForm
          inferenceOutput={inferenceOutput}
          onClose={onClose}
          episodeId={episodeId}
          inferenceId={inferenceId}
        />
      </DialogContent>
    </Dialog>
  );
}

interface FeedbackFormProps {
  inferenceOutput?: ContentBlockOutput[] | JsonInferenceOutput;
  onClose: () => void;
  episodeId?: string;
  inferenceId?: string;
}

function FeedbackForm({
  inferenceOutput,
  onClose,
  episodeId,
  inferenceId,
}: FeedbackFormProps) {
  const config = useConfig();
  // If there is no inference output this is likely an episode-level feedback and
  // we should filter demonstration out of the list of metrics.
  const metrics = filterStaticEvaluationMetrics(
    inferenceOutput === undefined
      ? filterMetricsByLevel(config.metrics, "episode")
      : filterMetricsByLevel(config.metrics, "inference"),
  );
  const [selectedMetricName, setSelectedMetricName] = useState<string>("");
  const selectedMetric = metrics[selectedMetricName];
  const selectedMetricType = selectedMetric?.type;
  const [booleanValue, setBooleanValue] = useState<string | null>(null);
  const [floatValue, setFloatValue] = useState<string>("");
  const [commentValue, setCommentValue] = useState<string>("");
  const [demonstrationValue, setDemonstrationValue] = useState<
    ContentBlockOutput[] | JsonInferenceOutput | undefined
  >(inferenceOutput);
  const [demonstrationIsValid, setDemonstrationIsValid] =
    useState<boolean>(true);

  // Calculate if input is missing based on the selected metric type
  const isInputMissing =
    (selectedMetricType === "boolean" && booleanValue === null) ||
    (selectedMetricType === "float" && floatValue.trim() === "") ||
    (selectedMetricType === "comment" && commentValue.trim() === "");

  return (
    <Form method="post" onSubmit={onClose}>
      {selectedMetricName && (
        <input type="hidden" name="metricName" value={selectedMetricName} />
      )}
      <input type="hidden" name="type" value="humanFeedback" />

      <MetricSelector
        metrics={metrics}
        selectedMetric={selectedMetricName}
        onMetricChange={setSelectedMetricName}
        showLevelBadges={false}
        // We don't need to show the level badges since this is currently only used for
        // the inference and episode detail pages where it is obvious
      />

      {selectedMetric && selectedMetricType === "boolean" && (
        <>
          <BooleanFeedbackInput
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
          <div className="mt-4">
            <Output
              output={demonstrationValue}
              isEditing={true}
              onOutputChange={(updatedOutput) => {
                if (updatedOutput === null) {
                  setDemonstrationIsValid(false);
                } else {
                  setDemonstrationValue(updatedOutput);
                  setDemonstrationIsValid(true);
                }
              }}
            />

            <input
              type="hidden"
              name="value"
              value={JSON.stringify(
                getDemonstrationValueToSubmit(demonstrationValue),
              )}
            />
          </div>
        ) : (
          <div className="mt-4 text-red-500">
            Initial output missing for demonstration value. This is most likely
            a bug. Please file a bug report{" "}
            <Link to="https://github.com/tensorzero/tensorzero/discussions/new?category=bug-reports">
              here
            </Link>
            .
          </div>
        ))}

      <input type="hidden" name="_action" value="addFeedback" />
      {episodeId && <input type="hidden" name="episodeId" value={episodeId} />}
      {inferenceId && (
        <input type="hidden" name="inferenceId" value={inferenceId} />
      )}

      {selectedMetricName && (
        <div className="flex justify-end">
          <Button
            type="submit"
            disabled={
              !selectedMetricName || isInputMissing || !demonstrationIsValid
            }
            className="mt-2"
          >
            Submit Feedback
          </Button>
        </div>
      )}
    </Form>
  );
}

/**
 * If the type of the demonstration value is JsonInferenceOutput,
 * we need to submit only demonstrationValue.parsed and not the entire
 * demonstrationValue object.
 * For ContentBlockOutput[], we submit the entire object.
 */
function getDemonstrationValueToSubmit(
  demonstrationValue: ContentBlockOutput[] | JsonInferenceOutput,
) {
  if (Array.isArray(demonstrationValue)) {
    return demonstrationValue;
  }
  return demonstrationValue.parsed;
}
