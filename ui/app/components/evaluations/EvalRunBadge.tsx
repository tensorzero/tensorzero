import type { EvaluationRunInfo } from "~/utils/clickhouse/evaluations";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "../ui/tooltip";
import { Badge } from "../ui/badge";
import { X } from "lucide-react";
import { formatDate } from "~/utils/date";

interface EvalRunBadgeProps {
  runInfo: EvaluationRunInfo;
  getColor: (runId: string) => string;
  lastUpdateDate?: Date;
  onRemove?: (e: React.MouseEvent) => void;
}

// Helper function to get the last 6 digits of a UUID
export function getLastUuidSegment(uuid: string): string {
  return uuid.slice(-6);
}

export default function EvalRunBadge({
  runInfo,
  getColor,
  lastUpdateDate,
  onRemove,
}: EvalRunBadgeProps) {
  const runId = runInfo.eval_run_id;
  const variantColor = getColor(runId);
  const runIdSegment = getLastUuidSegment(runId);
  // If runId is empty, render a simple badge without tooltip
  if (!runId) {
    return (
      <Badge className={`${variantColor} flex items-center gap-1.5 px-2 py-1`}>
        <span>{runInfo.variant_name}</span>
        {onRemove && (
          <X
            className="ml-1 h-3 w-3 cursor-pointer opacity-70 hover:opacity-100"
            onClick={onRemove}
          />
        )}
      </Badge>
    );
  }

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <Badge
            className={`${variantColor} flex cursor-help items-center gap-1.5 px-2 py-1`}
          >
            <span>{runInfo.variant_name}</span>
            <span className="border-l border-white/30 pl-1.5 text-xs opacity-80">
              {runIdSegment}
            </span>
            {onRemove && (
              <X
                className="ml-1 h-3 w-3 cursor-pointer opacity-70 hover:opacity-100"
                onClick={onRemove}
              />
            )}
          </Badge>
        </TooltipTrigger>
        <TooltipContent side="top" className="p-2">
          <p className="text-xs">
            Run ID: <span className="font-mono text-xs">{runId}</span>
            <br />
            {lastUpdateDate
              ? `Last Updated: ${formatDate(lastUpdateDate)}`
              : null}
          </p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
