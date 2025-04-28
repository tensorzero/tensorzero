import { Button, type ButtonProps, ButtonIcon } from "~/components/ui/button";
import { Feedback } from "../icons/Icons";

export type HumanFeedbackButtonProps = Omit<
  ButtonProps,
  "children" | "variant" | "size" | "slotLeft" | "slotRight" | "asChild"
>;

export function HumanFeedbackButton(props: HumanFeedbackButtonProps) {
  return (
    <Button variant="outline" size="sm" {...props}>
      <ButtonIcon as={Feedback} variant="tertiary" />
      Add feedback
    </Button>
  );
}
