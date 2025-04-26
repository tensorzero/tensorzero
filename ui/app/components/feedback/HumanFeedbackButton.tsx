import { Button, type ButtonProps, ButtonIcon } from "~/components/ui/button";
import { Feedback } from "../icons/Icons";

export interface HumanFeedbackButtonProps {
  onClick: () => void;
}

export function HumanFeedbackButton(
  props: Omit<
    ButtonProps,
    "children" | "variant" | "size" | "slotLeft" | "slotRight" | "asChild"
  >,
) {
  return (
    <Button variant="outline" size="sm" {...props}>
      <ButtonIcon as={Feedback} variant="tertiary" />
      Add feedback
    </Button>
  );
}
