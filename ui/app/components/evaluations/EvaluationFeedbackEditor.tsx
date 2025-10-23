import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "~/components/ui/dialog";
import { useState } from "react";
import { Form } from "react-router";
import BooleanFeedbackInput from "../feedback/BooleanFeedbackInput";
import { EditButton } from "~/components/utils/EditButton";
import EvaluationRunBadge from "./EvaluationRunBadge";
import { useColorAssigner } from "~/hooks/evaluations/ColorAssigner";
import { useConfig } from "~/context/config";
import FloatFeedbackInput from "../feedback/FloatFeedbackInput";
import { Button } from "../ui/button";
import { logger } from "~/utils/logger";
import { ReadOnlyGuard } from "~/components/utils/read-only-guard";

interface EvaluationFeedbackEditorProps {
  inferenceId: string;
  datapointId: string;
  metricName: string;
  originalValue: string;
  evalRunId: string;
  variantName: string;
  evaluatorInferenceId: string | null;
}

// Allow users to edit the feedback provided by an LLM Judge.
export default function EvaluationFeedbackEditor({
  inferenceId,
  datapointId,
  metricName,
  originalValue,
  evalRunId,
  variantName,
  evaluatorInferenceId,
}: EvaluationFeedbackEditorProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [feedback, setFeedback] = useState<string | null>(null);
  const { getColor } = useColorAssigner();
  const config = useConfig();
  const metricConfig = config.metrics[metricName];
  if (!metricConfig) {
    logger.warn(`Metric ${metricName} not found`);
    return null;
  }
  const metricType = metricConfig.type;
  return (
    <>
      <ReadOnlyGuard asChild>
        <EditButton onClick={() => setIsOpen(true)} />
      </ReadOnlyGuard>
      <Dialog open={isOpen} onOpenChange={setIsOpen}>
        <DialogContent className="max-h-[90vh] sm:max-w-[1200px]">
          <DialogHeader>
            <DialogTitle>
              Edit Feedback for{" "}
              <span className="text-md font-mono font-semibold">
                {metricName}
              </span>
            </DialogTitle>
            <div style={{ width: "fit-content" }}>
              <EvaluationRunBadge
                runInfo={{
                  evaluation_run_id: evalRunId,
                  variant_name: variantName,
                }}
                getColor={getColor}
              />
            </div>
          </DialogHeader>
          <Form method="post" onSubmit={() => setIsOpen(false)}>
            <input type="hidden" name="inferenceId" value={inferenceId} />
            <input type="hidden" name="datapointId" value={datapointId} />
            <input type="hidden" name="metricName" value={metricName} />
            <input type="hidden" name="originalValue" value={originalValue} />
            <input type="hidden" name="_action" value="addFeedback" />
            <input
              type="hidden"
              name="evaluatorInferenceId"
              value={evaluatorInferenceId ?? ""}
            />
            <input type="hidden" name="value" value={feedback || ""} />
            {metricType === "boolean" && (
              <BooleanFeedbackInput
                value={feedback}
                onChange={(value) => setFeedback(value)}
              />
            )}
            {metricType === "float" && (
              <FloatFeedbackInput
                value={feedback ?? ""}
                onChange={(value) => setFeedback(value)}
              />
            )}
            <Button type="submit" className="mt-4" disabled={!feedback}>
              Submit Feedback
            </Button>
          </Form>
        </DialogContent>
      </Dialog>
    </>
  );
}
