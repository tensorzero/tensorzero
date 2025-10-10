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
  disabled?: boolean;
  tooltip?: string;
}

export function DeleteButton({
  onClick,
  className,
  isLoading,
  disabled = false,
  tooltip = "Delete",
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
          <span className="inline-block">
            <Button
              variant="outline"
              size="iconSm"
              className={className}
              disabled={disabled || isLoading}
              onClick={disabled ? undefined : handleInitialClick}
              aria-label={tooltip}
              title={tooltip}
            >
              <Trash2 className="h-4 w-4" />
            </Button>
          </span>
        </TooltipTrigger>
        <TooltipContent>
          <p>{tooltip}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
