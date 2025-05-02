import { useState } from "react";
import { Check, ChevronsUpDown } from "lucide-react";
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
import { useSearchParams, useNavigate } from "react-router";
import { useColorAssigner } from "~/hooks/evaluations/ColorAssigner";
import { getLastUuidSegment } from "~/components/evaluations/EvaluationRunBadge";
import type { DynamicEvaluationRun } from "~/utils/clickhouse/dynamic_evaluations";
import { useSearchDynamicEvaluationRunsFetcher } from "~/routes/api/dynamic_evaluations/search_runs/route";
import DynamicEvaluationRunBadge from "./DynamicEvaluationRunBadge";

interface DynamicEvalRunSelectorProps {
  projectName: string;
  selectedRunInfos: DynamicEvaluationRun[];
}

export function DynamicEvalRunSelector({
  projectName,
  selectedRunInfos,
}: DynamicEvalRunSelectorProps) {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);
  const [searchValue, setSearchValue] = useState("");
  const selectedRunIds = selectedRunInfos.map((info) => info.id);

  // Maximum number of selections allowed
  const MAX_SELECTIONS = 5;
  const canAddMore = selectedRunInfos.length < MAX_SELECTIONS;

  // Use the color assigner context
  const { getColor } = useColorAssigner();

  const { data, isLoading } = useSearchDynamicEvaluationRunsFetcher({
    projectName,
    query: searchValue,
  });

  const availableRunInfos = data ? data : [];

  // Update the URL with the selected run IDs
  const updateSelectedRunIds = (runIds: string[]) => {
    const newParams = new URLSearchParams(searchParams);
    if (runIds.length > 0) {
      newParams.set("run_ids", runIds.join(","));
    } else {
      newParams.delete("run_ids");
    }
    navigate(`?${newParams.toString()}`, { replace: true });
  };

  // Toggle a run selection
  const toggleRun = (runId: string) => {
    const runInfo = availableRunInfos.find((info) => info.id === runId);
    if (!runInfo) return;

    if (selectedRunIds.includes(runId)) {
      // Remove the run
      const newSelectedRunIds = selectedRunIds.filter((id) => id !== runId);
      updateSelectedRunIds(newSelectedRunIds);
    } else if (canAddMore) {
      // Add the run only if we haven't reached the limit
      updateSelectedRunIds([...selectedRunIds, runId]);
    }
  };

  // Select all runs
  const selectAll = () => {
    const allSelected = availableRunInfos.every((info) =>
      selectedRunIds.includes(info.id),
    );

    if (allSelected) {
      // Deselect all
      updateSelectedRunIds([]);
    } else {
      // Select all, but respect the maximum limit
      const runsToSelect = availableRunInfos.slice(0, MAX_SELECTIONS);
      updateSelectedRunIds(runsToSelect.map((info) => info.id));
    }
  };

  // Function to remove a specific run
  const removeRun = (runId: string, e: React.MouseEvent) => {
    // Stop the tooltip from triggering
    e.stopPropagation();

    const newSelectedRunInfos = selectedRunInfos.filter(
      (info) => info.id !== runId,
    );
    updateSelectedRunIds(newSelectedRunInfos.map((info) => info.id));
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
                    <span className="text-muted-foreground ml-2 text-sm">
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
                        const isSelected = selectedRunIds.includes(info.id);
                        const variantColor = getColor(info.id);
                        const runIdSegment = getLastUuidSegment(info.id);
                        const isDisabled = !isSelected && !canAddMore;

                        return (
                          <CommandItem
                            key={info.id}
                            value={`${info.name} ${info.id}`}
                            onSelect={() => toggleRun(info.id)}
                            className={`flex items-center gap-2 ${
                              isDisabled ? "cursor-not-allowed opacity-50" : ""
                            }`}
                            disabled={isDisabled}
                          >
                            <div
                              className={`${variantColor} h-3 w-3 rounded-full`}
                            />
                            <span className="flex-1 truncate">{info.name}</span>
                            <span className="text-muted-foreground text-xs">
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
                              selectedRunIds.includes(info.id),
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
        {selectedRunInfos.map((info) => (
          <DynamicEvaluationRunBadge
            key={info.id}
            runInfo={info}
            getColor={getColor}
            lastUpdateDate={new Date(info.timestamp)}
            onRemove={(e) => removeRun(info.id, e)}
          />
        ))}
      </div>
    </div>
  );
}
