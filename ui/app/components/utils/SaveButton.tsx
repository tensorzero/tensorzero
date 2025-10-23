import { Save } from "lucide-react";
import { Button, type ButtonVariant } from "~/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "~/components/ui/tooltip";

interface SaveButtonProps {
  onClick: () => void;
  className?: string;
  disabled?: boolean;
  variant?: ButtonVariant;
}

export function SaveButton({
  onClick,
  className,
  disabled = false,
  variant = "outline",
}: SaveButtonProps) {
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
            aria-label="Save"
            title="Save"
          >
            <Save className="h-4 w-4" />
          </Button>
        </TooltipTrigger>
        <TooltipContent>
          <p>Save</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
