import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "~/components/ui/tooltip";
import type { MetricConfig } from "tensorzero-node";

interface MetricNameWithTooltipProps {
  metricName: string;
  metricConfig: MetricConfig;
}

export function MetricNameWithTooltip({
  metricName,
  metricConfig,
}: MetricNameWithTooltipProps) {
  return (
    <TooltipProvider delayDuration={300}>
      <Tooltip>
        <TooltipTrigger asChild>
          <span className="cursor-help truncate font-mono">{metricName}</span>
        </TooltipTrigger>
        <TooltipContent side="top" className="p-3">
          <div className="space-y-1 text-left text-xs">
            <div>
              <span className="font-medium">Metric:</span>
              <span className="ml-1 font-mono text-xs">{metricName}</span>
            </div>
            <div>
              <span className="font-medium">Type:</span>
              <span className="ml-1 font-mono text-xs">
                {metricConfig.type}
              </span>
            </div>
            <div>
              <span className="font-medium">Level:</span>
              <span className="ml-1 font-mono text-xs">
                {metricConfig.level}
              </span>
            </div>
            {"optimize" in metricConfig && metricConfig.optimize && (
              <div>
                <span className="font-medium">Optimize:</span>
                <span className="ml-1 font-mono text-xs">
                  {metricConfig.optimize}
                </span>
              </div>
            )}
          </div>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
