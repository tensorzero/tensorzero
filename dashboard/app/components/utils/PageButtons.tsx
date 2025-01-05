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
  return (
    <div className="mt-4 flex items-center justify-center gap-2">
      <Button
        onClick={onPreviousPage}
        disabled={disablePrevious}
        className="rounded-md border border-gray-300 bg-white p-2 text-sm font-medium text-gray-700 hover:bg-gray-50"
      >
        <ChevronLeft className="h-4 w-4" />
      </Button>
      <Button
        onClick={onNextPage}
        disabled={disableNext}
        className="rounded-md border border-gray-300 bg-white p-2 text-sm font-medium text-gray-700 hover:bg-gray-50"
      >
        <ChevronRight className="h-4 w-4" />
      </Button>
    </div>
  );
}
