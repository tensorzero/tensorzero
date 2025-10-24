import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "../../../../components/ui/tooltip";
import { Badge } from "../../../../components/ui/badge";
import { X } from "lucide-react";
import { formatDate } from "~/utils/date";
import type { WorkflowEvaluationRun } from "~/utils/clickhouse/workflow_evaluations";
import { getLastUuidSegment } from "~/components/evaluations/EvaluationRunBadge";

interface WorkflowEvaluationRunBadgeProps {
  runInfo: WorkflowEvaluationRun;
  getColor: (runId: string) => string;
  lastUpdateDate?: Date;
  onRemove?: (e: React.MouseEvent) => void;
}

export default function WorkflowEvaluationRunBadge({
  runInfo,
  getColor,
  lastUpdateDate,
  onRemove,
}: WorkflowEvaluationRunBadgeProps) {
  const runId = runInfo.id;
  const variantColor = getColor(runId);
  const runIdSegment = getLastUuidSegment(runId);

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Badge
          className={`${variantColor} flex cursor-help items-center gap-1.5 px-2 py-1 font-mono`}
        >
          <span>{runInfo.name}</span>
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
  );
}
