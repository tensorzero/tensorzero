import { Button, type ButtonProps } from "~/components/ui/button";
import { Feedback } from "../icons/Icons";

export type HumanFeedbackButtonProps = Omit<
  ButtonProps,
  "children" | "variant" | "size" | "slotLeft" | "slotRight" | "asChild"
>;

export function HumanFeedbackButton(props: HumanFeedbackButtonProps) {
  return (
    <Button variant="outline" size="sm" {...props}>
      <Feedback className="text-fg-tertiary h-4 w-4" />
      Add feedback
    </Button>
  );
}
