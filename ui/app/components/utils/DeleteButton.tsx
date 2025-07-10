import { Trash2 } from "lucide-react";
import { Button } from "~/components/ui/button";
import { useState } from "react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "~/components/ui/tooltip";

interface DeleteButtonProps {
  onClick: () => void;
  className?: string;
  isLoading?: boolean;
}

export function DeleteButton({
  onClick,
  className,
  isLoading,
}: DeleteButtonProps) {
  const [confirmDelete, setConfirmDelete] = useState(false);

  const handleConfirm = () => {
    onClick();
    setConfirmDelete(false);
  };

  const handleCancel = () => {
    setConfirmDelete(false);
  };

  const handleInitialClick = () => {
    setConfirmDelete(true);
  };

  if (confirmDelete) {
    return (
      <div className="flex gap-2" role="group" aria-label="Confirm deletion">
        <Button
          variant="secondary"
          size="sm"
          className="bg-bg-muted"
          disabled={isLoading}
          onClick={handleCancel}
          aria-label="Cancel deletion"
        >
          No, keep it
        </Button>
        <Button
          variant="destructive"
          size="sm"
          className="group bg-red-600 text-white hover:bg-red-700"
          disabled={isLoading}
          onClick={handleConfirm}
          aria-label="Confirm deletion"
        >
          {isLoading ? "Deleting..." : "Yes, delete permanently"}
        </Button>
      </div>
    );
  }

  return (
    <TooltipProvider>
      <Tooltip delayDuration={100}>
        <TooltipTrigger asChild>
          <Button
            variant="outline"
            size="iconSm"
            className={className}
            disabled={isLoading}
            onClick={handleInitialClick}
            aria-label="Delete"
            title="Delete"
          >
            <Trash2 className="h-4 w-4" />
          </Button>
        </TooltipTrigger>
        <TooltipContent>
          <p>Delete</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
