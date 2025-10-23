import { Plus } from "lucide-react";
import { Button } from "~/components/ui/button";
import { useReadOnly } from "~/context/read-only";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "~/components/ui/tooltip";

interface BuildDatasetButtonProps {
  onClick: () => void;
  className?: string;
}

export function BuildDatasetButton(props: BuildDatasetButtonProps) {
  const isReadOnly = useReadOnly();

  const button = (
    <Button variant="outline" size="sm" disabled={isReadOnly} {...props}>
      <Plus className="text-fg-tertiary mr-2 h-4 w-4" />
      Build Dataset
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
