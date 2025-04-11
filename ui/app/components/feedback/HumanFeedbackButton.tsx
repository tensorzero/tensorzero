import { Button } from "~/components/ui/button";
import { Feedback } from "../icons/Icons";

export interface HumanFeedbackButtonProps {
  onClick: () => void;
}

export function HumanFeedbackButton({ onClick }: HumanFeedbackButtonProps) {
  return (
    <Button variant="outline" size="sm" onClick={onClick}>
      <Feedback className="text-fg-tertiary h-4 w-4" />
      Add feedback
    </Button>
  );
}
