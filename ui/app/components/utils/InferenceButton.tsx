import { Link } from "react-router";
import { Button } from "~/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "~/components/ui/tooltip";
import { Inferences } from "~/components/icons/Icons";
import { toInferenceUrl } from "~/utils/urls";

interface InferenceButtonProps {
  inferenceId: string;
  className?: string;
  tooltipText?: string;
}

export function InferenceButton({
  inferenceId,
  className,
  tooltipText = "View inference",
}: InferenceButtonProps) {
  return (
    <TooltipProvider>
      <Tooltip delayDuration={100}>
        <TooltipTrigger asChild>
          <Link to={toInferenceUrl(inferenceId)} target="_blank">
            <Button
              variant="outline"
              size="iconSm"
              className={className}
              aria-label={tooltipText}
              title={tooltipText}
            >
              <Inferences className="h-4 w-4" />
            </Button>
          </Link>
        </TooltipTrigger>
        <TooltipContent>
          <p>{tooltipText}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
