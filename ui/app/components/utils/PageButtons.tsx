import { Button } from "~/components/ui/button";
import { ChevronLeft, ChevronRight } from "lucide-react";

interface PageButtonsProps {
  onPreviousPage: () => void;
  onNextPage: () => void;
  disablePrevious: boolean;
  disableNext: boolean;
}

export default function PageButtons({
  onPreviousPage,
  onNextPage,
  disablePrevious,
  disableNext,
}: PageButtonsProps) {
  // We don't need page buttons if there is only one page
  if (disablePrevious && disableNext) {
    return null;
  }

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
