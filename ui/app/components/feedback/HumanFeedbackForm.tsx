import { useConfig } from "~/context/config";
import MetricSelector from "../metric/MetricSelector";
import { useState } from "react";
import type {
  ContentBlockChatOutput,
  JsonInferenceOutput,
} from "~/types/tensorzero";
import { ChatOutputElement } from "~/components/input_output/ChatOutputElement";
import { JsonOutputElement } from "~/components/input_output/JsonOutputElement";
import { Link } from "react-router";
import { Button } from "~/components/ui/button";
import {
  filterMetricsByLevel,
  filterInferenceEvaluationMetrics,
} from "~/utils/config/feedback";
import BooleanFeedbackInput from "./BooleanFeedbackInput";
import FloatFeedbackInput from "./FloatFeedbackInput";
import CommentFeedbackInput from "./CommentFeedbackInput";
import { isJsonOutput } from "~/utils/clickhouse/inference";

export interface HumanFeedbackFormSharedProps {
  inferenceOutput?: ContentBlockChatOutput[] | JsonInferenceOutput;
  formError?: string | null;
  isSubmitting?: boolean;
}

type HumanFeedbackFormProps = HumanFeedbackFormSharedProps &
  (
    | { episodeId: string; inferenceId?: never }
    | { episodeId?: never; inferenceId: string }
  );

export function HumanFeedbackForm({
  inferenceOutput,
  episodeId,
  inferenceId,
  formError,
  isSubmitting,
}: HumanFeedbackFormProps) {
  const config = useConfig();
  // If there is no inference output this is likely an episode-level feedback and
  // we should filter demonstration out of the list of metrics.
  const metrics = filterInferenceEvaluationMetrics(
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
    ContentBlockChatOutput[] | JsonInferenceOutput | undefined
  >(inferenceOutput);
  const [demonstrationIsValid, setDemonstrationIsValid] =
    useState<boolean>(true);

  // Calculate if input is missing based on the selected metric type
  const isInputMissing =
    (selectedMetricType === "boolean" && booleanValue === null) ||
    (selectedMetricType === "float" && floatValue.trim() === "") ||
    (selectedMetricType === "comment" && commentValue.trim() === "");

  return (
    <>
      {selectedMetricName && (
        <input type="hidden" name="metricName" value={selectedMetricName} />
      )}
      <input type="hidden" name="type" value="humanFeedback" />
      <input type="hidden" name="_action" value="addFeedback" />

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
            {isJsonOutput(demonstrationValue) ? (
              <JsonOutputElement
                output={demonstrationValue}
                isEditing={true}
                onOutputChange={(updatedOutput) => {
                  if (updatedOutput === undefined) {
                    setDemonstrationIsValid(false);
                  } else {
                    setDemonstrationValue(updatedOutput);
                    setDemonstrationIsValid(true);
                  }
                }}
              />
            ) : (
              <ChatOutputElement
                output={demonstrationValue}
                isEditing={true}
                onOutputChange={(updatedOutput) => {
                  if (updatedOutput === undefined) {
                    setDemonstrationIsValid(false);
                  } else {
                    setDemonstrationValue(updatedOutput);
                    setDemonstrationIsValid(true);
                  }
                }}
              />
            )}

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

      {episodeId && <input type="hidden" name="episodeId" value={episodeId} />}
      {inferenceId && (
        <input type="hidden" name="inferenceId" value={inferenceId} />
      )}

      {selectedMetricName && (
        <div className="mt-4 flex items-start justify-between gap-4">
          <div aria-live="polite">
            {!!formError && (
              <p className="text-destructive text-sm font-bold">{formError}</p>
            )}
          </div>
          <Button
            type="submit"
            disabled={
              isSubmitting ||
              !selectedMetricName ||
              isInputMissing ||
              !demonstrationIsValid
            }
          >
            {isSubmitting ? "Submitting…" : "Submit Feedback"}
          </Button>
        </div>
      )}
    </>
  );
}

/**
 * If the type of the demonstration value is JsonInferenceOutput,
 * we need to submit only demonstrationValue.parsed and not the entire
 * demonstrationValue object.
 * For ContentBlockChatOutput[], we submit the entire object.
 */
function getDemonstrationValueToSubmit(
  demonstrationValue: ContentBlockChatOutput[] | JsonInferenceOutput,
) {
  if (isJsonOutput(demonstrationValue)) {
    return demonstrationValue.parsed;
  } else {
    return demonstrationValue;
  }
}
