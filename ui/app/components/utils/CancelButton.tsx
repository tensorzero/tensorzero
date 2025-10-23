import { X } from "lucide-react";
import { Button } from "~/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "~/components/ui/tooltip";

interface CancelButtonProps {
  onClick: () => void;
  className?: string;
}

export function CancelButton({ onClick, className }: CancelButtonProps) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          variant="outline"
          size="iconSm"
          onClick={onClick}
          className={className}
          aria-label="Cancel"
        >
          <X className="h-4 w-4" />
        </Button>
      </TooltipTrigger>
      <TooltipContent>Cancel</TooltipContent>
    </Tooltip>
  );
}
