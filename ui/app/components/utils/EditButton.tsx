import { Pencil } from "lucide-react";
import { Button } from "~/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "~/components/ui/tooltip";

interface EditButtonProps {
  onClick: () => void;
  className?: string;
}

export function EditButton({ onClick, className }: EditButtonProps) {
  return (
    <TooltipProvider>
      <Tooltip delayDuration={100}>
        <TooltipTrigger asChild>
          <Button
            variant="outline"
            size="iconSm"
            onClick={onClick}
            className={className}
            aria-label="Edit"
            title="Edit"
          >
            <Pencil className="h-4 w-4" />
          </Button>
        </TooltipTrigger>
        <TooltipContent>
          <p>Edit</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
