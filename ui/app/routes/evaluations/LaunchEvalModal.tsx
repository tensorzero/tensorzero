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

interface LaunchEvalModalProps {
  isOpen: boolean;
  onClose: () => void;
}

function EvalForm() {
  const fetcher = useFetcher();
  const config = useConfig();
  const evaluation_names = Object.keys(config.evaluations);
  const [selectedEvalName, setSelectedEvalName] = useState<string | null>(null);
  const [selectedVariantName, setSelectedVariantName] = useState<string | null>(
    null,
  );
  const [concurrencyLimit, setConcurrencyLimit] = useState<string>("5");
  let count = null;
  let isLoading = false;
  let dataset = null;
  let function_name = null;
  if (selectedEvalName) {
    dataset = config.evaluations[selectedEvalName]?.dataset_name;
    function_name = config.evaluations[selectedEvalName]?.function_name;
  }
  const { count: datasetCount, isLoading: datasetLoading } =
    useDatasetCountFetcher(dataset, function_name);
  count = datasetCount;
  isLoading = datasetLoading;

  // Check if all fields are filled
  const isFormValid =
    selectedEvalName !== null &&
    selectedVariantName !== null &&
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
          setSelectedEvalName(value);
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
      <div className="my-1 text-xs text-muted-foreground">
        Dataset:{" "}
        {dataset ? (
          <span className="font-medium">
            <Link to={`/datasets/${dataset}`}>{dataset}</Link>
          </span>
        ) : (
          <Skeleton className="inline-block h-3 w-16 align-middle" />
        )}
      </div>
      <div className="mb-1 text-xs text-muted-foreground">
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
        disabled={!selectedEvalName}
        onValueChange={(value) => setSelectedVariantName(value)}
      >
        <SelectTrigger>
          <SelectValue placeholder="Select a variant" />
        </SelectTrigger>
        <SelectContent>
          {(() => {
            if (!selectedEvalName) return null;

            const evaluation_function =
              config.evaluations[selectedEvalName].function_name;
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

export default function LaunchEvalModal({
  isOpen,
  onClose,
}: LaunchEvalModalProps) {
  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Launch Evaluation</DialogTitle>
        </DialogHeader>

        <EvalForm />
      </DialogContent>
    </Dialog>
  );
}
