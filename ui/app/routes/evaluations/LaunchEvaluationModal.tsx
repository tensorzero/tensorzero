import { useState } from "react";
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

interface LaunchEvaluationModalProps {
  isOpen: boolean;
  onClose: () => void;
  dataset_names: string[];
}

function EvaluationForm({ dataset_names }: { dataset_names: string[] }) {
  const fetcher = useFetcher();
  const config = useConfig();
  const evaluation_names = Object.keys(config.evaluations);
  const [selectedEvaluationName, setSelectedEvaluationName] = useState<
    string | null
  >(null);
  const [selectedDatasetName, setSelectedDatasetName] = useState<string | null>(
    null,
  );
  const [selectedVariantName, setSelectedVariantName] = useState<string | null>(
    null,
  );
  const [concurrencyLimit, setConcurrencyLimit] = useState<string>("5");
  let count = null;
  let isLoading = false;
  let function_name = null;
  if (selectedEvaluationName) {
    function_name = config.evaluations[selectedEvaluationName]?.function_name;
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
    concurrencyLimit !== "";

  return (
    <fetcher.Form method="post">
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
          htmlFor="evaluation_name"
          className="mb-1 block text-sm font-medium"
        >
          Dataset
        </label>
      </div>
      <Select
        name="dataset_name"
        onValueChange={(value) => setSelectedDatasetName(value)}
      >
        <SelectTrigger>
          <SelectValue placeholder="Select a dataset" />
        </SelectTrigger>
        <SelectContent>
          {dataset_names.map((dataset_name) => (
            <SelectItem key={dataset_name} value={dataset_name}>
              {dataset_name}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <div className="mb-1 mt-2 text-xs text-muted-foreground">
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
      <div className="mb-1 text-xs text-muted-foreground">
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
              config.evaluations[selectedEvaluationName].function_name;
            const variant_names = Object.keys(
              config.functions[evaluation_function].variants,
            );

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
          className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
          required
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
  dataset_names,
}: LaunchEvaluationModalProps) {
  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Launch Evaluation</DialogTitle>
        </DialogHeader>

        <EvaluationForm dataset_names={dataset_names} />
      </DialogContent>
    </Dialog>
  );
}
