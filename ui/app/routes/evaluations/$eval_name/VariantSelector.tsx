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
import { useSearchEvalRunsFetcher } from "~/routes/api/evaluations/search_runs/$eval_name/route";

interface VariantSelectorProps {
  evalName: string;
  mostRecentEvalInferenceDates: Map<string, Date>;
  selectedRunIdInfos: EvaluationRunInfo[];
}

// Helper function to get the last 6 digits of a UUID
export function getLastUuidSegment(uuid: string): string {
  return uuid.slice(-6);
}

export function VariantSelector({
  evalName,
  mostRecentEvalInferenceDates,
  selectedRunIdInfos,
}: VariantSelectorProps) {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const selectedRunIds = selectedRunIdInfos.map((info) => info.eval_run_id);
  const availableRunInfos = useSearchEvalRunsFetcher({
    evalName: evalName,
    query: "",
  });

  // State to track if dropdown is open
  const [isOpen, setIsOpen] = useState(false);

  // Update the URL with the selected run IDs
  const updateSelectedRunIds = (runIdInfos: EvaluationRunInfo[]) => {
    const newParams = new URLSearchParams(searchParams);
    newParams.set(
      "eval_run_ids",
      runIdInfos.map((info) => info.eval_run_id).join(","),
    );
    navigate(`?${newParams.toString()}`, { replace: true });
  };

  // Toggle a run selection
  const toggleRun = (runId: string) => {
    const runInfo = availableRunInfos.data.find(
      (info) => info.eval_run_id === runId,
    );
    if (!runInfo) return;

    if (selectedRunIds.includes(runId)) {
      // Remove the run
      const newSelectedRunIdInfos = selectedRunIdInfos.filter(
        (info) => info.eval_run_id !== runId,
      );
      updateSelectedRunIds(newSelectedRunIdInfos);
    } else {
      // Add the run
      updateSelectedRunIds([...selectedRunIdInfos, runInfo]);
    }
  };

  // Select all runs
  const selectAll = () => {
    const allSelected = availableRunInfos.data.every((info) =>
      selectedRunIds.includes(info.eval_run_id),
    );

    if (allSelected) {
      // Deselect all
      updateSelectedRunIds([]);
    } else {
      // Select all
      updateSelectedRunIds(availableRunInfos.data);
    }
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
            {availableRunInfos.data.map((info) => {
              const isSelected = selectedRunIds.includes(info.eval_run_id);
              const variantColor = getVariantColorClass(info.eval_run_id);
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
              checked={availableRunInfos.data.every((info) =>
                selectedRunIds.includes(info.eval_run_id),
              )}
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
        {selectedRunIdInfos.map((info) => {
          const runId = info.eval_run_id;
          const variantColor = getVariantColor(runId);
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

// Helper hash function for consistent coloring
function hashString(str: string): number {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = (hash << 5) - hash + str.charCodeAt(i);
    hash = hash & hash;
  }
  return hash;
}

// Get variant color based on runId hash - will produce consistent colors regardless of order
export function getVariantColor(runId: string, isSelected = true) {
  // Use a hash of the runId for consistent color assignment
  // Increase number of possible colors from 4 to 10
  const colorIndex = Math.abs(hashString(runId)) % 10;

  // Expanded color palette with 10 distinct options
  switch (colorIndex) {
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
    case 3:
      return isSelected
        ? "bg-red-600 hover:bg-red-700"
        : "border-red-600 text-red-600 hover:bg-red-50";
    case 4:
      return isSelected
        ? "bg-amber-600 hover:bg-amber-700"
        : "border-amber-600 text-amber-600 hover:bg-amber-50";
    case 5:
      return isSelected
        ? "bg-pink-600 hover:bg-pink-700"
        : "border-pink-600 text-pink-600 hover:bg-pink-50";
    case 6:
      return isSelected
        ? "bg-teal-600 hover:bg-teal-700"
        : "border-teal-600 text-teal-600 hover:bg-teal-50";
    case 7:
      return isSelected
        ? "bg-indigo-600 hover:bg-indigo-700"
        : "border-indigo-600 text-indigo-600 hover:bg-indigo-50";
    case 8:
      return isSelected
        ? "bg-cyan-600 hover:bg-cyan-700"
        : "border-cyan-600 text-cyan-600 hover:bg-cyan-50";
    case 9:
      return isSelected
        ? "bg-orange-600 hover:bg-orange-700"
        : "border-orange-600 text-orange-600 hover:bg-orange-50";
    default:
      return isSelected
        ? "bg-gray-600 hover:bg-gray-700"
        : "border-gray-600 text-gray-600 hover:bg-gray-50";
  }
}

// Helper function to get color class for the small badge in dropdown
function getVariantColorClass(runId: string) {
  // Use the same hash-based approach for consistency
  const colorIndex = Math.abs(hashString(runId)) % 10;

  // Return only the background color class - expanded to match the colors above
  switch (colorIndex) {
    case 0:
      return "bg-blue-600";
    case 1:
      return "bg-purple-600";
    case 2:
      return "bg-green-600";
    case 3:
      return "bg-red-600";
    case 4:
      return "bg-amber-600";
    case 5:
      return "bg-pink-600";
    case 6:
      return "bg-teal-600";
    case 7:
      return "bg-indigo-600";
    case 8:
      return "bg-cyan-600";
    case 9:
      return "bg-orange-600";
    default:
      return "bg-gray-600";
  }
}
