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

interface EvaluationFeedbackEditorProps {
  inferenceId: string;
  datapointId: string;
  metricName: string;
  originalValue: string;
}

export default function EvaluationFeedbackEditor({
  inferenceId,
  datapointId,
  metricName,
  originalValue,
}: EvaluationFeedbackEditorProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [feedback, setFeedback] = useState<string | null>(null);

  return (
    <>
      <EditButton onClick={() => setIsOpen(true)} />
      <Dialog open={isOpen} onOpenChange={setIsOpen}>
        <DialogContent className="max-h-[90vh] sm:max-w-[1200px]">
          <DialogHeader>
            <DialogTitle>Add Feedback</DialogTitle>
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
