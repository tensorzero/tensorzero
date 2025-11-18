import { useLayoutEffect, useState, useEffect } from "react";
import { useFetcher, Link } from "react-router";
import { CircleHelp } from "lucide-react";
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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "~/components/ui/select";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "~/components/ui/tooltip";
import { useDatasetCountFetcher } from "~/routes/api/datasets/count_dataset_function.route";
import { useConfig, useFunctionConfig } from "~/context/config";
import { Skeleton } from "~/components/ui/skeleton";
import { AdvancedParametersAccordion } from "./AdvancedParametersAccordion";
import type { InferenceCacheSetting } from "~/utils/evaluations.server";
import { DatasetSelector } from "~/components/dataset/DatasetSelector";
import { useDatasetCounts } from "~/hooks/use-dataset-counts";
import { toFunctionUrl } from "~/utils/urls";

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
  const evaluation_names = Object.keys(config.evaluations);
  const [selectedEvaluationName, setSelectedEvaluationName] = useState<
    string | null
  >(initialFormState?.evaluation_name ?? null);
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
  const [precisionTargets, setprecisionTargets] = useState<
    Record<string, string>
  >(initialFormState?.precision_targets ?? {});

  let count = null;
  let isLoading = false;
  let function_name = null;
  let evaluatorNames: string[] = [];
  if (selectedEvaluationName) {
    function_name =
      config.evaluations[selectedEvaluationName]?.function_name ?? null;
    evaluatorNames = Object.keys(
      config.evaluations[selectedEvaluationName]?.evaluators ?? {},
    );
  }
  const functionConfig = useFunctionConfig(function_name);

  const { data: datasets = [], isLoading: datasetsLoading } = useDatasetCounts(
    function_name ?? undefined,
  );

  const { count: datasetCount, isLoading: datasetLoading } =
    useDatasetCountFetcher(selectedDatasetName, function_name);
  count = datasetCount;
  isLoading = datasetLoading;

  // Validate that stored values still exist in the current config/datasets
  useEffect(() => {
    // Validate evaluation name - if it doesn't exist in config, clear it
    if (
      selectedEvaluationName &&
      !evaluation_names.includes(selectedEvaluationName)
    ) {
      setSelectedEvaluationName(null);
      setSelectedVariantName(null);
      setprecisionTargets({});
    }

    // Validate dataset name - if datasets have loaded and the dataset doesn't exist, clear it
    if (
      selectedDatasetName &&
      !datasetsLoading &&
      !datasets.some((d) => d.name === selectedDatasetName)
    ) {
      setSelectedDatasetName(null);
    }

    // Validate variant name - if it doesn't exist in the function config, clear it
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
    selectedDatasetName,
    datasets,
    datasetsLoading,
    selectedVariantName,
    functionConfig,
  ]);

  // Initialize precision targets with empty string for all evaluators when evaluation changes
  useEffect(() => {
    if (selectedEvaluationName) {
      const currentEvaluatorNames = Object.keys(
        config.evaluations[selectedEvaluationName]?.evaluators ?? {},
      );
      const newLimits: Record<string, string> = {};

      // Always initialize all evaluators with empty string (reset when evaluation changes)
      for (const evaluatorName of currentEvaluatorNames) {
        newLimits[evaluatorName] = "";
      }

      // Only update if the structure changed
      const currentKeys = Object.keys(precisionTargets).sort().join(",");
      const newKeys = Object.keys(newLimits).sort().join(",");
      if (currentKeys !== newKeys) {
        setprecisionTargets(newLimits);
      }
    }
  }, [selectedEvaluationName, config.evaluations, precisionTargets]);

  // Validate max_datapoints: must be empty or a positive integer
  const isMaxDatapointsValid =
    maxDatapoints === "" ||
    (Number.isInteger(Number(maxDatapoints)) &&
      Number(maxDatapoints) > 0 &&
      !maxDatapoints.includes("."));

  // Validate precision_targets: all values must be non-negative numbers
  const areprecisionTargetsValid = Object.values(precisionTargets).every(
    (value) => {
      if (value === "") return true;
      // Check if the entire string is a valid number
      const num = Number(value);
      return !isNaN(num) && num >= 0 && value.trim() !== "";
    },
  );

  // Check if all fields are filled and valid
  const isFormValid =
    selectedEvaluationName !== null &&
    selectedVariantName !== null &&
    selectedDatasetName !== null &&
    datasetCount !== null &&
    datasetCount > 0 &&
    inferenceCache !== null &&
    concurrencyLimit !== "" &&
    isMaxDatapointsValid &&
    areprecisionTargetsValid;

  return (
    <fetcher.Form
      method="post"
      onSubmit={(event) => {
        const formData = new FormData(event.currentTarget);
        persistToLocalStorage(formData);
      }}
    >
      <div className="mt-4">
        <label
          htmlFor="evaluation_name"
          className="mb-1 block text-sm font-medium"
        >
          Evaluation
        </label>
      </div>
      <Select
        name="evaluation_name"
        value={selectedEvaluationName ?? undefined}
        onValueChange={(value) => {
          setSelectedEvaluationName(value);
          setSelectedVariantName(null); // Reset variant selection when evaluation changes
        }}
      >
        <SelectTrigger>
          <SelectValue placeholder="Select an evaluation" />
        </SelectTrigger>
        <SelectContent>
          {evaluation_names.map((evaluation_name) => (
            <SelectItem key={evaluation_name} value={evaluation_name}>
              {evaluation_name}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
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

      <DatasetSelector
        functionName={function_name ?? undefined}
        selected={selectedDatasetName ?? undefined}
        onSelect={setSelectedDatasetName}
        allowCreation={false}
        disabled={!selectedEvaluationName}
      />

      <div className="text-muted-foreground mt-2 mb-1 text-xs">
        Function:{" "}
        {function_name ? (
          <span className="font-medium">
            <Link to={toFunctionUrl(function_name)}>{function_name}</Link>
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
      <Select
        name="variant_name"
        value={selectedVariantName ?? undefined}
        disabled={!selectedEvaluationName}
        onValueChange={(value) => setSelectedVariantName(value)}
      >
        <SelectTrigger>
          <SelectValue placeholder="Select a variant" />
        </SelectTrigger>
        <SelectContent>
          {(() => {
            if (!selectedEvaluationName || !functionConfig) return null;

            const variant_names = Object.keys(functionConfig.variants);

            return variant_names.map((variant_name) => (
              <SelectItem key={variant_name} value={variant_name}>
                {variant_name}
              </SelectItem>
            ));
          })()}
        </SelectContent>
      </Select>
      <div className="mt-4">
        <div className="mb-1 flex items-center gap-1.5">
          <label htmlFor="concurrency_limit" className="text-sm font-medium">
            Concurrency
          </label>
          <TooltipProvider>
            <Tooltip delayDuration={300}>
              <TooltipTrigger asChild>
                <span className="inline-flex cursor-help">
                  <CircleHelp className="text-muted-foreground h-3.5 w-3.5" />
                </span>
              </TooltipTrigger>
              <TooltipContent side="top">
                <p className="text-xs">
                  The number of datapoints to evaluate in parallel. Increasing
                  this value can speed up the evaluation run, but may trigger
                  rate limiting from model providers.
                </p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
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
          <TooltipProvider>
            <Tooltip delayDuration={300}>
              <TooltipTrigger asChild>
                <span className="inline-flex cursor-help">
                  <CircleHelp className="text-muted-foreground h-3.5 w-3.5" />
                </span>
              </TooltipTrigger>
              <TooltipContent side="top">
                Maximum number of datapoints to evaluate (optional)
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
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
          setprecisionTargets={setprecisionTargets}
          areprecisionTargetsValid={areprecisionTargetsValid}
          evaluatorNames={evaluatorNames}
          defaultOpen={inferenceCache !== "on"}
        />
        <input type="hidden" name="inference_cache" value={inferenceCache} />
        <input
          type="hidden"
          name="precision_targets"
          value={
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

interface EvaluationsFormValues {
  dataset_name: string | null;
  evaluation_name: string | null;
  variant_name: string | null;
  concurrency_limit: string;
  inference_cache: InferenceCacheSetting;
  max_datapoints: string;
  precision_targets: Record<string, string>;
}

interface EvaluationsFormState extends Partial<EvaluationsFormValues> {
  renderKey: string;
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

  // Handle precision_targets: convert old JSON string format to Record<string, string>
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

  // TODO: add validation
  return data as Partial<EvaluationsFormValues>;
}
