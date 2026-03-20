import { useEffect, useLayoutEffect, useMemo, useState } from "react";
import { useFetcher, Link } from "react-router";
import { Check, ChevronsUpDown } from "lucide-react";
import { Button } from "~/components/ui/button";
import { Input } from "~/components/ui/input";
import {
  Dialog,
  DialogBody,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "~/components/ui/dialog";
import { HelpTooltip } from "~/components/ui/HelpTooltip";
import {
  useAllFunctionConfigs,
  useConfig,
  useFunctionConfig,
} from "~/context/config";
import { Skeleton } from "~/components/ui/skeleton";
import { AdvancedParametersAccordion } from "./AdvancedParametersAccordion";
import type { InferenceCacheSetting } from "~/utils/evaluations.server";
import { DatasetCombobox } from "~/components/dataset/DatasetCombobox";
import { Combobox } from "~/components/ui/combobox";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "~/components/ui/tabs";
import { Evaluation } from "~/components/icons/Icons";
import { GitBranch } from "lucide-react";
import { useDatasetCounts } from "~/hooks/use-dataset-counts";
import { toFunctionUrl } from "~/utils/urls";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "~/components/ui/command";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "~/components/ui/popover";

interface LaunchEvaluationModalProps {
  isOpen: boolean;
  onClose: () => void;
}

function EvaluationForm({
  initialFormState,
}: {
  initialFormState: Partial<EvaluationsFormValues> | null;
}) {
  const fetcher = useFetcher();
  const config = useConfig();
  const functions = useAllFunctionConfigs();
  const evaluation_names = Object.keys(config.evaluations);
  const function_names = Object.keys(functions)
    .filter((name) => !name.startsWith("tensorzero::"))
    .sort((a, b) => a.localeCompare(b));
  const [launchMode, setLaunchMode] = useState<LaunchMode>(
    initialFormState?.launch_mode ?? "evaluators",
  );
  const [selectedEvaluationName, setSelectedEvaluationName] = useState<
    string | null
  >(initialFormState?.evaluation_name ?? null);
  const [selectedFunctionName, setSelectedFunctionName] = useState<
    string | null
  >(initialFormState?.function_name ?? null);
  const [selectedEvaluatorNames, setSelectedEvaluatorNames] = useState<
    string[]
  >(initialFormState?.evaluator_names ?? []);
  const [selectedDatasetName, setSelectedDatasetName] = useState<string | null>(
    initialFormState?.dataset_name ?? null,
  );
  const [selectedVariantName, setSelectedVariantName] = useState<string | null>(
    initialFormState?.variant_name ?? null,
  );
  const [concurrencyLimit, setConcurrencyLimit] = useState<string>(
    initialFormState?.concurrency_limit ?? "5",
  );
  const [inferenceCache, setInferenceCache] = useState<InferenceCacheSetting>(
    initialFormState?.inference_cache ?? "on",
  );
  const [maxDatapoints, setMaxDatapoints] = useState<string>(
    initialFormState?.max_datapoints ?? "",
  );
  const [precisionTargets, setPrecisionTargets] = useState<
    Record<string, string>
  >(initialFormState?.precision_targets ?? {});

  let count = null;
  let isLoading = false;
  const namedFunctionName = selectedEvaluationName
    ? (config.evaluations[selectedEvaluationName]?.function_name ?? null)
    : null;
  const currentFunctionName =
    launchMode === "evaluations-legacy"
      ? namedFunctionName
      : selectedFunctionName;
  const evaluatorNames = useMemo(
    () =>
      launchMode === "evaluations-legacy"
        ? selectedEvaluationName
          ? Object.keys(
              config.evaluations[selectedEvaluationName]?.evaluators ?? {},
            )
          : []
        : selectedEvaluatorNames,
    [
      config.evaluations,
      launchMode,
      selectedEvaluationName,
      selectedEvaluatorNames,
    ],
  );
  const functionConfig = useFunctionConfig(currentFunctionName);
  const availableEvaluatorNames = useMemo(
    () =>
      functionConfig?.evaluators
        ? Object.keys(functionConfig.evaluators).sort()
        : [],
    [functionConfig],
  );

  const { data: datasets = [], isLoading: datasetsLoading } = useDatasetCounts(
    currentFunctionName ?? undefined,
  );

  // Get the count for the selected dataset from the datasets array
  const selectedDataset = selectedDatasetName
    ? datasets.find((d) => d.name === selectedDatasetName)
    : undefined;
  count = selectedDataset?.count ?? null;
  isLoading = datasetsLoading;

  // Validate that stored values still exist in the current config/datasets
  useEffect(() => {
    if (
      selectedEvaluationName &&
      !evaluation_names.includes(selectedEvaluationName)
    ) {
      setSelectedEvaluationName(null);
    }

    if (
      selectedFunctionName &&
      !function_names.includes(selectedFunctionName)
    ) {
      setSelectedFunctionName(null);
    }

    if (
      selectedEvaluatorNames.some(
        (name) => !availableEvaluatorNames.includes(name),
      )
    ) {
      setSelectedEvaluatorNames((current) =>
        current.filter((name) => availableEvaluatorNames.includes(name)),
      );
    }

    if (
      selectedDatasetName &&
      !datasetsLoading &&
      !datasets.some((d) => d.name === selectedDatasetName)
    ) {
      setSelectedDatasetName(null);
    }

    if (
      selectedVariantName &&
      functionConfig &&
      !Object.keys(functionConfig.variants).includes(selectedVariantName)
    ) {
      setSelectedVariantName(null);
    }
  }, [
    selectedEvaluationName,
    evaluation_names,
    selectedFunctionName,
    function_names,
    selectedEvaluatorNames,
    availableEvaluatorNames,
    selectedDatasetName,
    datasets,
    datasetsLoading,
    selectedVariantName,
    functionConfig,
  ]);

  useEffect(() => {
    const newLimits: Record<string, string> = {};
    for (const evaluatorName of evaluatorNames) {
      newLimits[evaluatorName] = precisionTargets[evaluatorName] ?? "";
    }

    const currentKeys = Object.keys(precisionTargets).sort().join(",");
    const newKeys = Object.keys(newLimits).sort().join(",");
    if (currentKeys !== newKeys) {
      setPrecisionTargets(newLimits);
    }
  }, [evaluatorNames, precisionTargets]);

  // Validate max_datapoints: must be empty or a positive integer
  const isMaxDatapointsValid =
    maxDatapoints === "" ||
    (Number.isInteger(Number(maxDatapoints)) &&
      Number(maxDatapoints) > 0 &&
      !maxDatapoints.includes("."));

  // Validate precision_targets: all values must be non-negative numbers
  const arePrecisionTargetsValid = Object.values(precisionTargets).every(
    (value) => {
      if (value === "") return true;
      // Check if the entire string is a valid number
      const num = Number(value);
      return !isNaN(num) && num >= 0 && value.trim() !== "";
    },
  );

  const hasValidEvaluationSource =
    launchMode === "evaluations-legacy"
      ? selectedEvaluationName !== null
      : currentFunctionName !== null && selectedEvaluatorNames.length > 0;

  // Check if all fields are filled and valid
  const isFormValid =
    hasValidEvaluationSource &&
    selectedVariantName !== null &&
    selectedDatasetName !== null &&
    count !== null &&
    count > 0 &&
    inferenceCache !== null &&
    concurrencyLimit !== "" &&
    isMaxDatapointsValid &&
    arePrecisionTargetsValid;

  return (
    <fetcher.Form
      method="post"
      onSubmit={(event) => {
        const formData = new FormData(event.currentTarget);
        persistToLocalStorage(formData);
      }}
    >
      <div className="mt-4">
        <label className="mb-1 block text-sm font-medium">Mode</label>
      </div>
      <input type="hidden" name="launch_mode" value={launchMode} />
      <Tabs
        value={launchMode}
        onValueChange={(value) => setLaunchMode(value as LaunchMode)}
      >
        <TabsList className="grid w-full grid-cols-2 rounded-md border border-border-accent bg-bg-secondary p-1">
          <TabsTrigger
            value="evaluators"
            className="text-fg-secondary data-[state=active]:bg-menu-highlight data-[state=active]:text-menu-highlight-foreground"
          >
            Evaluators
          </TabsTrigger>
          <TabsTrigger
            value="evaluations-legacy"
            className="text-fg-secondary data-[state=active]:bg-menu-highlight data-[state=active]:text-menu-highlight-foreground"
          >
            Evaluations (Legacy)
          </TabsTrigger>
        </TabsList>

        {/*Evaluators*/}
        <TabsContent value="evaluators" className="mt-4 space-y-4">
          <div>
            <label
              htmlFor="function_name"
              className="mb-1 block text-sm font-medium"
            >
              Function
            </label>
            <Combobox
              name="function_name"
              selected={selectedFunctionName}
              onSelect={(value) => {
                setSelectedFunctionName(value);
                setSelectedVariantName(null);
              }}
              items={function_names}
              placeholder="Select function"
              emptyMessage="No functions found"
              ariaLabel="Select function"
            />
          </div>
          <div>
            <label
              htmlFor="evaluator_names"
              className="mb-1 block text-sm font-medium"
            >
              Evaluators
            </label>
            <input
              type="hidden"
              name="evaluator_names"
              value={JSON.stringify(selectedEvaluatorNames)}
            />
            <EvaluatorMultiSelect
              selected={selectedEvaluatorNames}
              onSelect={setSelectedEvaluatorNames}
              items={availableEvaluatorNames}
            />
            {selectedEvaluatorNames.length > 0 && (
              <div className="text-muted-foreground mt-2 mb-1 text-xs">
                Evaluators:{" "}
                <span className="font-medium">
                  {selectedEvaluatorNames.join(", ")}
                </span>
              </div>
            )}
          </div>
        </TabsContent>

        {/*Evaluations (Legacy)*/}
        <TabsContent value="evaluations-legacy" className="mt-4">
          <div>
            <label
              htmlFor="evaluation_name"
              className="mb-1 block text-sm font-medium"
            >
              Evaluation
            </label>
            <Combobox
              name="evaluation_name"
              selected={selectedEvaluationName}
              onSelect={(value) => {
                setSelectedEvaluationName(value);
                setSelectedVariantName(null);
              }}
              items={evaluation_names}
              getPrefix={() => <Evaluation className="h-4 w-4 shrink-0" />}
              placeholder="Select evaluation"
              emptyMessage="No evaluations found"
              ariaLabel="Select evaluation"
            />
          </div>
        </TabsContent>
      </Tabs>

      <div className="mt-4">
        <label
          htmlFor="dataset_name"
          className="mb-1 block text-sm font-medium"
        >
          Dataset
        </label>
      </div>

      <input
        type="hidden"
        name="dataset_name"
        value={selectedDatasetName ?? undefined}
      />

      <DatasetCombobox
        functionName={currentFunctionName ?? undefined}
        selected={selectedDatasetName}
        onSelect={(name) => setSelectedDatasetName(name)}
        disabled={!currentFunctionName}
        ariaLabel="Select dataset"
      />

      <div className="text-muted-foreground mt-2 mb-1 text-xs">
        Function:{" "}
        {currentFunctionName ? (
          <span className="font-medium">
            <Link to={toFunctionUrl(currentFunctionName)}>
              {currentFunctionName}
            </Link>
          </span>
        ) : (
          <Skeleton className="inline-block h-3 w-16 align-middle" />
        )}
      </div>
      <div className="text-muted-foreground mb-1 text-xs">
        Datapoints:{" "}
        {count !== null ? (
          <span className="font-medium">{count}</span>
        ) : isLoading ? (
          <Skeleton className="inline-block h-3 w-16 align-middle" />
        ) : (
          <Skeleton className="inline-block h-3 w-16 align-middle" />
        )}
      </div>

      <div className="mt-4">
        <label
          htmlFor="variant_name"
          className="mb-1 block text-sm font-medium"
        >
          Variant
        </label>
      </div>

      <Combobox
        name="variant_name"
        selected={selectedVariantName}
        onSelect={setSelectedVariantName}
        items={functionConfig ? Object.keys(functionConfig.variants) : []}
        getPrefix={() => <GitBranch className="h-4 w-4 shrink-0" />}
        placeholder="Select variant"
        emptyMessage="No variants found"
        disabled={!currentFunctionName}
        ariaLabel="Select variant"
      />
      <div className="mt-4">
        <div className="mb-1 flex items-center gap-1.5">
          <label htmlFor="concurrency_limit" className="text-sm font-medium">
            Concurrency
          </label>
          <HelpTooltip>
            The number of datapoints to evaluate in parallel. Increasing this
            value can speed up the evaluation run, but may trigger rate limiting
            from model providers.
          </HelpTooltip>
        </div>
        <input
          type="number"
          id="concurrency_limit"
          name="concurrency_limit"
          data-testid="concurrency-limit"
          min="1"
          value={concurrencyLimit}
          onChange={(e) => setConcurrencyLimit(e.target.value)}
          className="border-input bg-background w-full rounded-md border px-3 py-2 text-sm"
          required
        />
      </div>
      <div className="mt-4">
        <div className="mb-2 flex items-center gap-1.5">
          <label htmlFor="max_datapoints" className="text-sm font-medium">
            Max Datapoints
          </label>
          <HelpTooltip>
            Leave empty to evaluate all datapoints in the dataset.
          </HelpTooltip>
        </div>
        <Input
          type="text"
          id="max_datapoints"
          name="max_datapoints"
          value={maxDatapoints}
          onChange={(e) => setMaxDatapoints(e.target.value)}
          placeholder="No limit"
          className={
            !isMaxDatapointsValid && maxDatapoints !== ""
              ? "border-red-500 focus:ring-red-500"
              : ""
          }
        />
        {!isMaxDatapointsValid && maxDatapoints !== "" && (
          <p className="mt-1 text-xs text-red-500">
            Must be a positive integer
          </p>
        )}
      </div>
      <div className="mt-4">
        <AdvancedParametersAccordion
          inferenceCache={inferenceCache}
          setInferenceCache={setInferenceCache}
          precisionTargets={precisionTargets}
          setPrecisionTargets={setPrecisionTargets}
          arePrecisionTargetsValid={arePrecisionTargetsValid}
          evaluatorNames={evaluatorNames}
          defaultOpen={inferenceCache !== "on"}
        />
        <input type="hidden" name="inference_cache" value={inferenceCache} />
        <input
          type="hidden"
          name="precision_targets"
          value={
            // Serialize precision targets to JSON for form submission.
            // Precision targets enable adaptive stopping: an evaluator stops running
            // once both sides of its 95% confidence interval are within the specified
            // threshold of the mean. Only positive values are submitted: setting to 0.0
            // disables adaptive stopping for that evaluator.
            Object.keys(precisionTargets).length > 0
              ? JSON.stringify(
                  Object.fromEntries(
                    Object.entries(precisionTargets)
                      .filter(([_, value]) => {
                        const num = parseFloat(value);
                        return value !== "" && !isNaN(num) && num > 0;
                      })
                      .map(([key, value]) => [key, parseFloat(value)]),
                  ),
                )
              : ""
          }
        />
      </div>
      <DialogFooter>
        <Button className="mt-2" type="submit" disabled={!isFormValid}>
          Launch
        </Button>
      </DialogFooter>
    </fetcher.Form>
  );
}

export default function LaunchEvaluationModal({
  isOpen,
  onClose,
}: LaunchEvaluationModalProps) {
  const [initialFormState, setInitialFormState] =
    useState<EvaluationsFormState | null>(null);
  // useLayoutEffect to update fields before paint to avoid flicker of old state
  useLayoutEffect(() => {
    const storedValues = getFromLocalStorage();
    if (storedValues) {
      setInitialFormState({
        ...storedValues,
        // generate a key that we'll use to force re-render the form so that all
        // internal state values are reset when given new data
        renderKey: Date.now().toString(),
      });
    }
  }, []);
  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-h-[90vh]">
        <DialogHeader>
          <DialogTitle>Launch Evaluation</DialogTitle>
          <DialogDescription>
            Run an evaluation on a dataset to measure variant performance.
          </DialogDescription>
        </DialogHeader>
        <DialogBody>
          <EvaluationForm
            key={initialFormState?.renderKey}
            initialFormState={initialFormState}
          />
        </DialogBody>
      </DialogContent>
    </Dialog>
  );
}

type LaunchMode = "evaluations-legacy" | "evaluators";

interface EvaluationsFormValues {
  launch_mode: LaunchMode;
  dataset_name: string | null;
  evaluation_name: string | null;
  function_name: string | null;
  evaluator_names: string[];
  variant_name: string | null;
  concurrency_limit: string;
  inference_cache: InferenceCacheSetting;
  max_datapoints: string;
  precision_targets: Record<string, string>;
}

interface EvaluationsFormState extends Partial<EvaluationsFormValues> {
  renderKey: string;
}

function EvaluatorMultiSelect({
  selected,
  onSelect,
  items,
}: {
  selected: string[];
  onSelect: (value: string[]) => void;
  items: string[];
}) {
  const [open, setOpen] = useState(false);

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          type="button"
          variant="outline"
          role="combobox"
          aria-expanded={open}
          aria-label="Select evaluators"
          className="flex w-full items-center justify-between gap-2"
        >
          <span className="truncate">
            {selected.length > 0
              ? `${selected.length} evaluator${selected.length === 1 ? "" : "s"} selected`
              : "Select evaluators"}
          </span>
          <ChevronsUpDown className="h-4 w-4 shrink-0 opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-[var(--radix-popover-trigger-width)] p-0">
        <Command>
          <CommandInput placeholder="Search evaluators..." />
          <CommandList>
            <CommandEmpty>No evaluators found.</CommandEmpty>
            <CommandGroup>
              {items.map((item) => {
                const isSelected = selected.includes(item);
                return (
                  <CommandItem
                    key={item}
                    value={item}
                    onSelect={() => {
                      onSelect(
                        isSelected
                          ? selected.filter((name) => name !== item)
                          : [...selected, item],
                      );
                    }}
                  >
                    <Check
                      className={`mr-2 h-4 w-4 ${
                        isSelected ? "opacity-100" : "opacity-0"
                      }`}
                    />
                    {item}
                  </CommandItem>
                );
              })}
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
}

const LOCAL_STORAGE_KEY = "tensorzero:evaluationForm";

function persistToLocalStorage(formData: FormData) {
  const formObject: Record<string, string> = {};
  for (const [key, value] of formData.entries()) {
    if (typeof value !== "string") continue;
    formObject[key] = value;
  }
  try {
    localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(formObject));
  } catch {
    // silently ignore errors, e.g. if localStorage is full
  }
}

function getFromLocalStorage() {
  const values = localStorage.getItem(LOCAL_STORAGE_KEY);
  if (!values) return null;
  let parsed: unknown;
  try {
    parsed = JSON.parse(values);
    if (parsed == null || typeof parsed !== "object") {
      localStorage.removeItem(LOCAL_STORAGE_KEY);
      return null;
    }
  } catch {
    localStorage.removeItem(LOCAL_STORAGE_KEY);
    return null;
  }

  const data = parsed as Record<string, unknown>;

  // Parse precision_targets from stored JSON string format to object,
  // and convert numeric values to strings for form inputs
  if (typeof data.precision_targets === "string" && data.precision_targets) {
    try {
      const parsedLimits = JSON.parse(data.precision_targets);
      if (typeof parsedLimits === "object" && parsedLimits !== null) {
        // Convert numbers to strings for the form
        data.precision_targets = Object.fromEntries(
          Object.entries(parsedLimits).map(([k, v]) => [k, String(v)]),
        );
      } else {
        data.precision_targets = {};
      }
    } catch {
      data.precision_targets = {};
    }
  } else if (typeof data.precision_targets !== "object") {
    data.precision_targets = {};
  }

  if (typeof data.evaluator_names === "string" && data.evaluator_names) {
    try {
      const parsedEvaluatorNames = JSON.parse(data.evaluator_names);
      data.evaluator_names = Array.isArray(parsedEvaluatorNames)
        ? parsedEvaluatorNames.filter((name) => typeof name === "string")
        : [];
    } catch {
      data.evaluator_names = [];
    }
  } else if (!Array.isArray(data.evaluator_names)) {
    data.evaluator_names = [];
  }

  if (
    data.launch_mode !== "evaluations-legacy" &&
    data.launch_mode !== "evaluators"
  ) {
    data.launch_mode = "evaluators";
  }

  // TODO: add validation
  return data as Partial<EvaluationsFormValues>;
}
