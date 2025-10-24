import { Pencil } from "lucide-react";
import { Button, type ButtonVariant } from "~/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "~/components/ui/tooltip";

interface EditButtonProps {
  onClick: () => void;
  className?: string;
  disabled?: boolean;
  tooltip?: string;
  variant?: ButtonVariant;
}

export function EditButton({
  onClick,
  className,
  disabled = false,
  tooltip = "Edit",
  variant = "outline",
}: EditButtonProps) {
  if (disabled) {
    // For disabled buttons, wrap in a span to ensure tooltip works
    return (
      <TooltipProvider>
        <Tooltip delayDuration={100}>
          <TooltipTrigger asChild>
            <span className="inline-block">
              <Button
                variant={variant}
                size="iconSm"
                className={className}
                disabled={disabled}
                aria-label={tooltip}
                title={tooltip}
              >
                <Pencil className="h-4 w-4" />
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

  return (
    <TooltipProvider>
      <Tooltip delayDuration={100}>
        <TooltipTrigger asChild>
          <Button
            variant={variant}
            size="iconSm"
            onClick={onClick}
            className={className}
            aria-label={tooltip}
            title={tooltip}
          >
            <Pencil className="h-4 w-4" />
          </Button>
        </TooltipTrigger>
        <TooltipContent>
          <p>{tooltip}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
