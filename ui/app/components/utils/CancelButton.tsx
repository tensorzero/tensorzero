import { X } from "lucide-react";
import { Button, type ButtonVariant } from "~/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "~/components/ui/tooltip";

interface CancelButtonProps {
  onClick: () => void;
  className?: string;
  disabled?: boolean;
  variant?: ButtonVariant;
}

export function CancelButton({
  onClick,
  className,
  disabled = false,
  variant = "outline",
}: CancelButtonProps) {
  return (
    <TooltipProvider>
      <Tooltip delayDuration={100}>
        <TooltipTrigger asChild>
          <Button
            variant={variant}
            size="iconSm"
            onClick={onClick}
            className={className}
            disabled={disabled}
            aria-label="Cancel"
            title="Cancel"
          >
            <X className="h-4 w-4" />
          </Button>
        </TooltipTrigger>
        <TooltipContent>
          <p>Cancel</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
