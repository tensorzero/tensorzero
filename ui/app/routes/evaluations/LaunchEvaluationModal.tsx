import { useLayoutEffect, useState, useEffect } from "react";
import { useFetcher, Link } from "react-router";
import { Button } from "~/components/ui/button";
import {
  Dialog,
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
  const [minInferences, setMinInferences] = useState<string>(
    initialFormState?.min_inferences ?? "",
  );
  const [maxInferences, setMaxInferences] = useState<string>(
    initialFormState?.max_inferences ?? "",
  );
  const [precisionLimits, setPrecisionLimits] = useState<
    Record<string, string>
  >(initialFormState?.precision_limits ?? {});

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
      setPrecisionLimits({});
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

  // Initialize precision limits with 0.0 for all evaluators when evaluation changes
  useEffect(() => {
    if (selectedEvaluationName) {
      const currentEvaluatorNames = Object.keys(
        config.evaluations[selectedEvaluationName]?.evaluators ?? {},
      );
      const newLimits: Record<string, string> = {};

      // Initialize all evaluators with 0.0 or keep existing values
      for (const evaluatorName of currentEvaluatorNames) {
        newLimits[evaluatorName] = precisionLimits[evaluatorName] ?? "0.0";
      }

      // Only update if the structure changed
      const currentKeys = Object.keys(precisionLimits).sort().join(",");
      const newKeys = Object.keys(newLimits).sort().join(",");
      if (currentKeys !== newKeys) {
        setPrecisionLimits(newLimits);
      }
    }
  }, [selectedEvaluationName, config.evaluations, precisionLimits]);

  // Validate min_inferences: must be empty or a positive integer
  const isMinInferencesValid =
    minInferences === "" ||
    (Number.isInteger(Number(minInferences)) &&
      Number(minInferences) > 0 &&
      !minInferences.includes("."));

  // Validate max_inferences: must be empty or a positive integer
  const isMaxInferencesValid =
    maxInferences === "" ||
    (Number.isInteger(Number(maxInferences)) &&
      Number(maxInferences) > 0 &&
      !maxInferences.includes("."));

  // Validate precision_limits: all values must be non-negative numbers
  const arePrecisionLimitsValid = Object.values(precisionLimits).every(
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
    isMinInferencesValid &&
    isMaxInferencesValid &&
    arePrecisionLimitsValid;

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
        <label
          htmlFor="concurrency_limit"
          className="mb-1 block text-sm font-medium"
        >
          Concurrency
        </label>
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
        <label
          htmlFor="max_inferences"
          className="mb-1 block text-sm font-medium"
        >
          Max Inferences
        </label>
        <p className="text-muted-foreground mb-2 text-xs">
          Maximum number of datapoints to evaluate (optional)
        </p>
        <input
          type="text"
          id="max_inferences"
          name="max_inferences"
          value={maxInferences}
          onChange={(e) => setMaxInferences(e.target.value)}
          placeholder="No limit"
          className={`border-input bg-background w-full rounded-md border px-3 py-2 text-sm ${
            !isMaxInferencesValid && maxInferences !== ""
              ? "border-red-500 focus:ring-red-500"
              : ""
          }`}
        />
        {!isMaxInferencesValid && maxInferences !== "" && (
          <p className="mt-1 text-xs text-red-500">
            Must be a positive integer
          </p>
        )}
      </div>
      <div className="mt-4">
        <AdvancedParametersAccordion
          inferenceCache={inferenceCache}
          setInferenceCache={setInferenceCache}
          precisionLimits={precisionLimits}
          setPrecisionLimits={setPrecisionLimits}
          arePrecisionLimitsValid={arePrecisionLimitsValid}
          minInferences={minInferences}
          setMinInferences={setMinInferences}
          isMinInferencesValid={isMinInferencesValid}
          evaluatorNames={evaluatorNames}
          defaultOpen={inferenceCache !== "on"}
        />
        <input type="hidden" name="inference_cache" value={inferenceCache} />
        <input
          type="hidden"
          name="precision_limits"
          value={
            Object.keys(precisionLimits).length > 0
              ? JSON.stringify(
                  Object.fromEntries(
                    Object.entries(precisionLimits)
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
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Launch Evaluation</DialogTitle>
          <DialogDescription>
            Run an evaluation on a dataset to measure variant performance.
          </DialogDescription>
        </DialogHeader>
        <EvaluationForm
          key={initialFormState?.renderKey}
          initialFormState={initialFormState}
        />
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
  min_inferences: string;
  max_inferences: string;
  precision_limits: Record<string, string>;
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

  // Handle precision_limits: convert old JSON string format to Record<string, string>
  if (typeof data.precision_limits === "string" && data.precision_limits) {
    try {
      const parsedLimits = JSON.parse(data.precision_limits);
      if (typeof parsedLimits === "object" && parsedLimits !== null) {
        // Convert numbers to strings for the form
        data.precision_limits = Object.fromEntries(
          Object.entries(parsedLimits).map(([k, v]) => [k, String(v)]),
        );
      } else {
        data.precision_limits = {};
      }
    } catch {
      data.precision_limits = {};
    }
  } else if (typeof data.precision_limits !== "object") {
    data.precision_limits = {};
  }

  // TODO: add validation
  return data as Partial<EvaluationsFormValues>;
}
