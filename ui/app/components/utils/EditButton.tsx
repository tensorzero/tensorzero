import { Pencil } from "lucide-react";
import { Button } from "~/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "~/components/ui/tooltip";

interface EditButtonProps {
  onClick: () => void;
  className?: string;
  disabled?: boolean;
  tooltip?: string;
}

export function EditButton({
  onClick,
  className,
  disabled = false,
  tooltip = "Edit",
}: EditButtonProps) {
  if (disabled) {
    // For disabled buttons, wrap in a span to ensure tooltip works
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <span className="inline-block">
            <Button
              variant="outline"
              size="iconSm"
              className={className}
              disabled={disabled}
              aria-label={tooltip}
            >
              <Pencil className="h-4 w-4" />
            </Button>
          </span>
        </TooltipTrigger>
        <TooltipContent>{tooltip}</TooltipContent>
      </Tooltip>
    );
  }

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          variant="outline"
          size="iconSm"
          onClick={onClick}
          className={className}
          aria-label={tooltip}
        >
          <Pencil className="h-4 w-4" />
        </Button>
      </TooltipTrigger>
      <TooltipContent>{tooltip}</TooltipContent>
    </Tooltip>
  );
}
