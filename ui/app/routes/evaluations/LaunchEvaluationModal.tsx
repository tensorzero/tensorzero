import { useLayoutEffect, useState } from "react";
import { useFetcher, Link } from "react-router";
import { Button } from "~/components/ui/button";
import {
  Dialog,
  DialogContent,
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
import { useConfig } from "~/context/config";
import { Skeleton } from "~/components/ui/skeleton";
import { AdvancedParametersAccordion } from "./AdvancedParametersAccordion";
import type { InferenceCacheSetting } from "~/utils/evaluations.server";
import { DatasetSelector } from "~/components/dataset/DatasetSelector";

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

  let count = null;
  let isLoading = false;
  let function_name = null;
  if (selectedEvaluationName) {
    function_name =
      config.evaluations[selectedEvaluationName]?.function_name ?? null;
  }
  const { count: datasetCount, isLoading: datasetLoading } =
    useDatasetCountFetcher(selectedDatasetName, function_name);
  count = datasetCount;
  isLoading = datasetLoading;

  // Check if all fields are filled
  const isFormValid =
    selectedEvaluationName !== null &&
    selectedVariantName !== null &&
    selectedDatasetName !== null &&
    datasetCount !== null &&
    datasetCount > 0 &&
    inferenceCache !== null &&
    concurrencyLimit !== "";

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
        defaultValue={initialFormState?.evaluation_name ?? undefined}
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
            <Link to={`/observability/functions/${function_name}`}>
              {function_name}
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
      <Select
        name="variant_name"
        defaultValue={initialFormState?.variant_name ?? undefined}
        disabled={!selectedEvaluationName}
        onValueChange={(value) => setSelectedVariantName(value)}
      >
        <SelectTrigger>
          <SelectValue placeholder="Select a variant" />
        </SelectTrigger>
        <SelectContent>
          {(() => {
            if (!selectedEvaluationName) return null;

            const evaluation_function =
              config.evaluations[selectedEvaluationName];
            if (!evaluation_function) return null;
            const function_config =
              config.functions[evaluation_function.function_name];
            if (!function_config) return null;
            const variant_names = Object.keys(function_config.variants);

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
          min="1"
          value={concurrencyLimit}
          onChange={(e) => setConcurrencyLimit(e.target.value)}
          className="border-input bg-background w-full rounded-md border px-3 py-2 text-sm"
          required
        />
      </div>
      <div className="mt-4">
        <AdvancedParametersAccordion
          inferenceCache={inferenceCache}
          setInferenceCache={setInferenceCache}
          defaultOpen={inferenceCache !== "on"}
        />
        <input type="hidden" name="inference_cache" value={inferenceCache} />
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

  // TODO: add validation
  return parsed as Partial<EvaluationsFormValues>;
}
