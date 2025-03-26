import { useState } from "react";
import { ChevronDown } from "lucide-react";
import { Badge } from "~/components/ui/badge";
import { Button } from "~/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "~/components/ui/dropdown-menu";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "~/components/ui/tooltip";
import { useSearchParams, useNavigate } from "react-router";
import type { EvaluationRunInfo } from "~/utils/clickhouse/evaluations";
import { formatDate } from "~/utils/date";

interface VariantSelectorProps {
  available_run_ids: EvaluationRunInfo[];
  mostRecentEvalInferenceDates: Map<string, Date>;
}

// Helper function to get the last 6 digits of a UUID
export function getLastUuidSegment(uuid: string): string {
  return uuid.slice(-6);
}

export function VariantSelector({
  available_run_ids,
  mostRecentEvalInferenceDates,
}: VariantSelectorProps) {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const selectedRunIdsParam = searchParams.get("eval_run_ids") || "";
  const selectedRunIds = selectedRunIdsParam
    ? selectedRunIdsParam.split(",")
    : [];

  // State to track if dropdown is open
  const [isOpen, setIsOpen] = useState(false);

  // Update the URL with the selected run IDs
  const updateSelectedRunIds = (runIds: string[]) => {
    const newParams = new URLSearchParams(searchParams);
    newParams.set("eval_run_ids", runIds.join(","));
    navigate(`?${newParams.toString()}`, { replace: true });
  };

  // Toggle a run selection
  const toggleRun = (runId: string) => {
    if (selectedRunIds.includes(runId)) {
      updateSelectedRunIds(selectedRunIds.filter((id) => id !== runId));
    } else {
      updateSelectedRunIds([...selectedRunIds, runId]);
    }
  };

  // Select all runs
  const selectAll = () => {
    updateSelectedRunIds(available_run_ids.map((info) => info.eval_run_id));
  };

  return (
    <div className="mb-6">
      <div className="flex flex-col space-y-2">
        <DropdownMenu open={isOpen} onOpenChange={setIsOpen}>
          <DropdownMenuTrigger asChild>
            <Button
              variant="outline"
              className="flex w-96 items-center justify-between gap-2"
            >
              <span>Select evaluation runs to compare...</span>
              <ChevronDown className="h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start" className="w-96">
            <DropdownMenuSeparator />
            {available_run_ids.map((info) => {
              const isSelected = selectedRunIds.includes(info.eval_run_id);
              const variantColor = getVariantColorClass(
                info.eval_run_id,
                available_run_ids,
              );
              const runIdSegment = getLastUuidSegment(info.eval_run_id);

              return (
                <DropdownMenuCheckboxItem
                  key={info.eval_run_id}
                  checked={isSelected}
                  onCheckedChange={() => toggleRun(info.eval_run_id)}
                  className="flex items-center gap-2"
                >
                  <div className="flex flex-1 items-center gap-2">
                    <Badge className={`${variantColor} h-3 w-3 p-0`} />
                    <span className="flex-1 truncate">{info.variant_name}</span>
                    <span className="text-xs text-muted-foreground">
                      {runIdSegment}
                    </span>
                  </div>
                </DropdownMenuCheckboxItem>
              );
            })}
            <DropdownMenuSeparator />
            <DropdownMenuCheckboxItem
              checked={selectedRunIds.length === available_run_ids.length}
              onCheckedChange={selectAll}
              className="font-medium"
            >
              Select All
            </DropdownMenuCheckboxItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      {/* Display selected variants as badges */}
      <div className="mt-3 flex flex-wrap gap-2">
        {selectedRunIds.map((runId) => {
          const info = available_run_ids.find((i) => i.eval_run_id === runId);
          if (!info) return null;

          const variantColor = getVariantColor(runId, available_run_ids);
          const runIdSegment = getLastUuidSegment(runId);

          return (
            <TooltipProvider key={runId}>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Badge
                    className={`${variantColor} flex cursor-help items-center gap-1.5 px-2 py-1`}
                  >
                    <span>{info.variant_name}</span>
                    <span className="border-l border-white/30 pl-1.5 text-xs opacity-80">
                      {runIdSegment}
                    </span>
                  </Badge>
                </TooltipTrigger>
                <TooltipContent side="top" className="p-2">
                  <p className="text-xs">
                    Run ID: <span className="font-mono text-xs">{runId}</span>
                    <br />
                    {mostRecentEvalInferenceDates.get(runId)
                      ? `Last Updated: ${formatDate(
                          mostRecentEvalInferenceDates.get(runId)!,
                        )}`
                      : null}
                  </p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          );
        })}
      </div>
    </div>
  );
}

// Helper function to get the appropriate color class based on variant
export function getVariantColor(
  runId: string,
  allRunIds: EvaluationRunInfo[],
  isSelected = true,
) {
  // Assign a color based on the index in the array
  const index = allRunIds.findIndex((info) => info.eval_run_id === runId);

  switch (index % 4) {
    case 0:
      return isSelected
        ? "bg-blue-600 hover:bg-blue-700"
        : "border-blue-600 text-blue-600 hover:bg-blue-50";
    case 1:
      return isSelected
        ? "bg-purple-600 hover:bg-purple-700"
        : "border-purple-600 text-purple-600 hover:bg-purple-50";
    case 2:
      return isSelected
        ? "bg-green-600 hover:bg-green-700"
        : "border-green-600 text-green-600 hover:bg-green-50";
    default:
      return isSelected
        ? "bg-gray-600 hover:bg-gray-700"
        : "border-gray-600 text-gray-600 hover:bg-gray-50";
  }
}

// Helper function to get color class for the small badge in dropdown
function getVariantColorClass(runId: string, allRunIds: EvaluationRunInfo[]) {
  // Get the index just like in getVariantColor
  const index = allRunIds.findIndex((info) => info.eval_run_id === runId);

  // Return only the background color class
  switch (index % 4) {
    case 0:
      return "bg-blue-600";
    case 1:
      return "bg-purple-600";
    case 2:
      return "bg-green-600";
    default:
      return "bg-gray-600";
  }
}
