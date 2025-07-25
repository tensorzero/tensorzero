import { Button } from "../ui/button";

export interface DemonstrationFeedbackButtonProps {
  isSubmitting: boolean;
  submissionError?: string | null;
}

export function DemonstrationFeedbackButton({
  isSubmitting,
  submissionError,
}: DemonstrationFeedbackButtonProps) {
  return (
    <div className="space-y-2">
      <Button variant="outline" size="sm" type="submit" disabled={isSubmitting}>
        {isSubmitting ? "Adding as Demonstration..." : "Add as Demonstration"}
      </Button>
      {submissionError && (
        <div className="text-sm text-red-600">{submissionError}</div>
      )}
    </div>
  );
}
