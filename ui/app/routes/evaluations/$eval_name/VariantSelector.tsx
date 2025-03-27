import { useState } from "react";
import { Check, ChevronsUpDown, X } from "lucide-react";
import { Badge } from "~/components/ui/badge";
import { Button } from "~/components/ui/button";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator,
} from "~/components/ui/command";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "~/components/ui/popover";
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
import { useColorAssigner } from "./ColorAssigner";

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
  const [open, setOpen] = useState(false);
  const [searchValue, setSearchValue] = useState("");

  // Maximum number of selections allowed
  const MAX_SELECTIONS = 5;
  const canAddMore = selectedRunIds.length < MAX_SELECTIONS;

  // Use the color assigner context
  const { getColor } = useColorAssigner();

  const { data, isLoading } = useSearchEvalRunsFetcher({
    evalName: evalName,
    query: searchValue,
  });
  const availableRunInfos = data || [];

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
    const runInfo = availableRunInfos.find(
      (info) => info.eval_run_id === runId,
    );
    if (!runInfo) return;

    if (selectedRunIds.includes(runId)) {
      // Remove the run
      const newSelectedRunIdInfos = selectedRunIdInfos.filter(
        (info) => info.eval_run_id !== runId,
      );
      updateSelectedRunIds(newSelectedRunIdInfos);
    } else if (canAddMore) {
      // Add the run only if we haven't reached the limit
      updateSelectedRunIds([...selectedRunIdInfos, runInfo]);
    }
  };

  // Select all runs
  const selectAll = () => {
    const allSelected = availableRunInfos.every((info) =>
      selectedRunIds.includes(info.eval_run_id),
    );

    if (allSelected) {
      // Deselect all
      updateSelectedRunIds([]);
    } else {
      // Select all, but respect the maximum limit
      const runsToSelect = availableRunInfos.slice(0, MAX_SELECTIONS);
      updateSelectedRunIds(runsToSelect);
    }
  };

  // Function to remove a specific run
  const removeRun = (runId: string, e: React.MouseEvent) => {
    // Stop the tooltip from triggering
    e.stopPropagation();

    const newSelectedRunIdInfos = selectedRunIdInfos.filter(
      (info) => info.eval_run_id !== runId,
    );
    updateSelectedRunIds(newSelectedRunIdInfos);
  };

  return (
    <div className="mb-6">
      <div className="flex flex-col space-y-2">
        <Popover open={open} onOpenChange={setOpen}>
          <PopoverTrigger asChild>
            <Button
              variant="outline"
              role="combobox"
              aria-expanded={open}
              className="flex w-96 items-center justify-between gap-2"
            >
              <span>Select evaluation runs to compare...</span>
              <ChevronsUpDown className="h-4 w-4 shrink-0 opacity-50" />
            </Button>
          </PopoverTrigger>
          <PopoverContent className="w-96 p-0">
            <Command>
              <CommandInput
                placeholder="Search by variant name or evaluation run ID..."
                value={searchValue}
                onValueChange={setSearchValue}
              />
              <CommandList>
                {isLoading ? (
                  <div className="flex items-center justify-center py-6">
                    <div className="h-4 w-4 animate-spin rounded-full border-2 border-gray-300 border-t-gray-600"></div>
                    <span className="ml-2 text-sm text-muted-foreground">
                      Loading...
                    </span>
                  </div>
                ) : (
                  <>
                    <CommandEmpty>No results found.</CommandEmpty>
                    {!canAddMore && !isLoading && (
                      <div className="px-2 py-1.5 text-sm text-amber-600">
                        Maximum of {MAX_SELECTIONS} runs can be selected.
                      </div>
                    )}
                    <CommandGroup>
                      {availableRunInfos.map((info) => {
                        const isSelected = selectedRunIds.includes(
                          info.eval_run_id,
                        );
                        const variantColor = getColor(info.eval_run_id);
                        const runIdSegment = getLastUuidSegment(
                          info.eval_run_id,
                        );
                        const isDisabled = !isSelected && !canAddMore;

                        return (
                          <CommandItem
                            key={info.eval_run_id}
                            value={`${info.variant_name} ${info.eval_run_id}`}
                            onSelect={() => toggleRun(info.eval_run_id)}
                            className={`flex items-center gap-2 ${
                              isDisabled ? "cursor-not-allowed opacity-50" : ""
                            }`}
                            disabled={isDisabled}
                          >
                            <div
                              className={`${variantColor} h-3 w-3 rounded-full`}
                            />
                            <span className="flex-1 truncate">
                              {info.variant_name}
                            </span>
                            <span className="text-xs text-muted-foreground">
                              {runIdSegment}
                            </span>
                            {isSelected && <Check className="ml-2 h-4 w-4" />}
                          </CommandItem>
                        );
                      })}
                    </CommandGroup>
                    {availableRunInfos.length > 1 && (
                      <>
                        <CommandSeparator />
                        <CommandGroup>
                          <CommandItem
                            onSelect={selectAll}
                            className="font-medium"
                          >
                            {availableRunInfos.every((info) =>
                              selectedRunIds.includes(info.eval_run_id),
                            )
                              ? "Deselect All"
                              : `Select ${
                                  availableRunInfos.length > MAX_SELECTIONS
                                    ? `First ${MAX_SELECTIONS}`
                                    : "All"
                                }`}
                          </CommandItem>
                        </CommandGroup>
                      </>
                    )}
                  </>
                )}
              </CommandList>
            </Command>
          </PopoverContent>
        </Popover>
      </div>

      {/* Display selected variants as badges */}
      <div className="mt-3 flex flex-wrap gap-2">
        {selectedRunIdInfos.map((info) => {
          const runId = info.eval_run_id;
          const variantColor = getColor(runId);
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
                    <X
                      className="ml-1 h-3 w-3 cursor-pointer opacity-70 hover:opacity-100"
                      onClick={(e) => removeRun(runId, e)}
                    />
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
