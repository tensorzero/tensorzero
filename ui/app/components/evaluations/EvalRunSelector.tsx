import { useRef, useState } from "react";
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
import { useSearchParams, useNavigate, useFetcher } from "react-router";
import type {
  EvaluationRunInfo,
  EvaluationRunSearchResult,
} from "~/utils/clickhouse/evaluations";
import { useColorAssigner } from "~/hooks/evaluations/ColorAssigner";
import { getLastUuidSegment } from "~/components/evaluations/EvaluationRunBadge";
import EvaluationRunBadge from "~/components/evaluations/EvaluationRunBadge";

interface EvalRunSelectorProps {
  evaluationName: string;
  selectedRunIdInfos: EvaluationRunInfo[];
  allowedRunInfos?: EvaluationRunInfo[]; // To be used if only a subset of runs are available,
  // for example if we're filtering by datapoint_id
}

export function EvalRunSelector({
  evaluationName,
  selectedRunIdInfos,
  allowedRunInfos,
}: EvalRunSelectorProps) {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const selectedRunIds = selectedRunIdInfos.map(
    (info) => info.evaluation_run_id,
  );
  const [open, setOpen] = useState(false);
  const [searchValue, setSearchValue] = useState("");

  // Maximum number of selections allowed
  const MAX_SELECTIONS = 5;
  const canAddMore = selectedRunIds.length < MAX_SELECTIONS;

  // Use the color assigner context
  const { getColor } = useColorAssigner();

  const {
    data,
    state,
    load: loadRunsFetcher,
  } = useFetcher<EvaluationRunSearchResult[]>();
  const isLoading = state === "loading";

  // Filter the fetched runs based on allowedRunInfos if it's provided
  const availableRunInfos = data
    ? allowedRunInfos // Check if allowedRunInfos is provided
      ? data.filter(
          (
            info, // If yes, filter data
          ) =>
            allowedRunInfos.some(
              (allowedInfo) =>
                allowedInfo.evaluation_run_id === info.evaluation_run_id,
            ),
        )
      : data // If no, use all data
    : []; // If data itself is null/undefined, use an empty array

  // Update the URL with the selected run IDs
  const updateSelectedRunIds = (runIdInfos: EvaluationRunSearchResult[]) => {
    const newParams = new URLSearchParams(searchParams);
    newParams.set(
      "evaluation_run_ids",
      runIdInfos.map((info) => info.evaluation_run_id).join(","),
    );
    navigate(`?${newParams.toString()}`, { replace: true });
  };

  // Toggle a run selection
  const toggleRun = (runId: string) => {
    const runInfo = availableRunInfos.find(
      (info) => info.evaluation_run_id === runId,
    );
    if (!runInfo) return;

    if (selectedRunIds.includes(runId)) {
      // Remove the run
      const newSelectedRunIdInfos = selectedRunIdInfos.filter(
        (info) => info.evaluation_run_id !== runId,
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
      selectedRunIds.includes(info.evaluation_run_id),
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
      (info) => info.evaluation_run_id !== runId,
    );
    updateSelectedRunIds(newSelectedRunIdInfos);
  };

  const hasInitializedRuns = useRef(false);
  function loadRuns(query: string | null, args?: { debounce?: boolean }) {
    if (evaluationName) {
      const searchParams = new URLSearchParams();
      searchParams.set("evaluation_name", evaluationName);
      if (query) {
        searchParams.set("q", query);
      }
      if (args?.debounce) {
        searchParams.set("debounce", "");
      }
      hasInitializedRuns.current = true;
      loadRunsFetcher(
        `/api/evaluations/search_runs/${evaluationName}?${searchParams}`,
      );
    }
  }

  function loadInitialRuns() {
    if (!hasInitializedRuns.current) {
      loadRuns(null);
    }
  }

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
              onFocus={loadInitialRuns}
              onClick={loadInitialRuns}
              onPointerEnter={loadInitialRuns}
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
                onValueChange={(value) => {
                  setSearchValue(value);
                  loadRuns(value, { debounce: true });
                }}
              />
              <CommandList>
                {isLoading && !hasInitializedRuns.current ? (
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
                        const isSelected = selectedRunIds.includes(
                          info.evaluation_run_id,
                        );
                        const variantColor = getColor(info.evaluation_run_id);
                        const runIdSegment = getLastUuidSegment(
                          info.evaluation_run_id,
                        );
                        const isDisabled =
                          (!isSelected && !canAddMore) || isLoading;

                        return (
                          <CommandItem
                            key={info.evaluation_run_id}
                            value={`${info.variant_name} ${info.evaluation_run_id}`}
                            onSelect={() => toggleRun(info.evaluation_run_id)}
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
                            disabled={isLoading}
                          >
                            {availableRunInfos.every((info) =>
                              selectedRunIds.includes(info.evaluation_run_id),
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
        {selectedRunIdInfos.map((info) => (
          <EvaluationRunBadge
            key={info.evaluation_run_id}
            runInfo={info}
            getColor={getColor}
            lastUpdateDate={new Date(info.most_recent_inference_date)}
            onRemove={(e) => removeRun(info.evaluation_run_id, e)}
          />
        ))}
      </div>
    </div>
  );
}
