import { Link } from "react-router";
import { Button } from "~/components/ui/button";
import {
  Tooltip,
  TooltipContent,
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
    <Tooltip>
      <TooltipTrigger asChild>
        <Link to={toInferenceUrl(inferenceId)} target="_blank">
          <Button
            variant="outline"
            size="iconSm"
            className={className}
            aria-label={tooltipText}
          >
            <Inferences className="h-4 w-4" />
          </Button>
        </Link>
      </TooltipTrigger>
      <TooltipContent>{tooltipText}</TooltipContent>
    </Tooltip>
  );
}
