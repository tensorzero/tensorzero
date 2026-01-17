import { Button } from "~/components/ui/button";
import { ChevronLeft, ChevronRight } from "lucide-react";

type PageButtonsProps =
  | {
      onPreviousPage: () => void;
      onNextPage: () => void;
      disablePrevious: boolean;
      disableNext: boolean;
      disabled?: never;
    }
  | {
      onPreviousPage?: never;
      onNextPage?: never;
      disablePrevious?: never;
      disableNext?: never;
      disabled: true;
    };

const noop = () => {};

export default function PageButtons(props: PageButtonsProps) {
  const onPreviousPage = props.disabled ? noop : props.onPreviousPage;
  const onNextPage = props.disabled ? noop : props.onNextPage;
  const disablePrevious = props.disabled ?? props.disablePrevious;
  const disableNext = props.disabled ?? props.disableNext;
  return (
    <div className="mt-4 flex items-center justify-center gap-2">
      <Button
        variant="outline"
        size="icon"
        onClick={onPreviousPage}
        disabled={disablePrevious}
        className="rounded-md bg-white p-2"
      >
        <ChevronLeft className="h-4 w-4" />
      </Button>
      <Button
        variant="outline"
        size="icon"
        onClick={onNextPage}
        disabled={disableNext}
        className="rounded-md bg-white p-2"
      >
        <ChevronRight className="h-4 w-4" />
      </Button>
    </div>
  );
}
