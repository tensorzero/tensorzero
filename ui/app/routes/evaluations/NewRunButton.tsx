import { Plus } from "lucide-react";
import { Button } from "~/components/ui/button";
import { useReadOnly } from "~/context/read-only";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "~/components/ui/tooltip";

interface NewRunButtonProps {
  onClick: () => void;
  className?: string;
}

export function NewRunButton({ className, ...props }: NewRunButtonProps) {
  const isReadOnly = useReadOnly();

  const button = (
    <Button
      variant="outline"
      size="sm"
      className={className}
      disabled={isReadOnly}
      {...props}
    >
      <Plus className="text-fg-tertiary mr-2 h-4 w-4" aria-hidden />
      New Run
    </Button>
  );

  if (isReadOnly) {
    return (
      <TooltipProvider>
        <Tooltip delayDuration={100}>
          <TooltipTrigger asChild>
            <span className="inline-block">{button}</span>
          </TooltipTrigger>
          <TooltipContent>
            <p>This feature is not available in read-only mode</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    );
  }

  return button;
}
