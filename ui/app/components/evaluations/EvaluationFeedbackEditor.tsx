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

interface EvaluationFeedbackEditorProps {
  inferenceId: string;
  datapointId: string;
  metricName: string;
  originalValue: string;
  evalRunId: string;
}

export default function EvaluationFeedbackEditor({
  inferenceId,
  datapointId,
  metricName,
  originalValue,
  evalRunId,
}: EvaluationFeedbackEditorProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [feedback, setFeedback] = useState<string | null>(null);
  const { getColor } = useColorAssigner();
  return (
    <>
      <EditButton onClick={() => setIsOpen(true)} />
      <Dialog open={isOpen} onOpenChange={setIsOpen}>
        <DialogContent className="max-h-[90vh] sm:max-w-[1200px]">
          <DialogHeader>
            <DialogTitle>Edit Feedback for {metricName}</DialogTitle>
            <div style={{ width: "fit-content" }}>
              <EvaluationRunBadge
                runInfo={{
                  evaluation_run_id: evalRunId,
                  variant_name: "Reference",
                }}
                getColor={getColor}
              />
            </div>
          </DialogHeader>
          <Form method="post">
            <input type="hidden" name="inferenceId" value={inferenceId} />
            <input type="hidden" name="datapointId" value={datapointId} />
            <input type="hidden" name="metricName" value={metricName} />
            <input type="hidden" name="originalValue" value={originalValue} />
            <BooleanFeedbackInput
              value={feedback}
              onChange={(value) => setFeedback(value)}
              metricName={metricName}
            />
          </Form>
        </DialogContent>
      </Dialog>
    </>
  );
}
