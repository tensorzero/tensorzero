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
import { Check, ChevronsUpDown } from "lucide-react";
import { cn } from "~/utils/common";

interface LaunchEvaluationModalProps {
  isOpen: boolean;
  onClose: () => void;
  dataset_names: string[];
}

interface DatasetSelectorProps {
  dataset_names: string[];
  selectedDatasetName: string | null;
  setSelectedDatasetName: (value: string | null) => void;
}

function DatasetSelector({
  dataset_names,
  selectedDatasetName,
  setSelectedDatasetName,
}: DatasetSelectorProps) {
  const [datasetPopoverOpen, setDatasetPopoverOpen] = useState(false);
  const [datasetInputValue, setDatasetInputValue] = useState("");

  const filteredDatasets = datasetInputValue
    ? dataset_names.filter((name) =>
        name.toLowerCase().includes(datasetInputValue.toLowerCase()),
      )
    : dataset_names;

  return (
    <>
      <input
        type="hidden"
        name="dataset_name"
        value={selectedDatasetName || ""}
      />
      <Popover open={datasetPopoverOpen} onOpenChange={setDatasetPopoverOpen}>
        <PopoverTrigger asChild>
          <Button
            variant="outline"
            role="combobox"
            aria-expanded={datasetPopoverOpen}
            className="w-full justify-between font-normal"
          >
            {selectedDatasetName || "Select a dataset"}
            <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-[var(--radix-popover-trigger-width)] p-0">
          <Command>
            <CommandInput
              placeholder="Search datasets..."
              value={datasetInputValue}
              onValueChange={setDatasetInputValue}
              className="h-9"
            />
            <CommandList>
              <CommandEmpty className="px-4 py-2 text-sm">
                No datasets found.
              </CommandEmpty>
              <CommandGroup heading="Datasets">
                {filteredDatasets.map((dataset_name) => (
                  <CommandItem
                    key={dataset_name}
                    value={dataset_name}
                    onSelect={() => {
                      setSelectedDatasetName(dataset_name);
                      setDatasetInputValue("");
                      setDatasetPopoverOpen(false);
                    }}
                    className="flex items-center justify-between"
                  >
                    <div className="flex items-center">
                      <Check
                        className={cn(
                          "mr-2 h-4 w-4",
                          selectedDatasetName === dataset_name
                            ? "opacity-100"
                            : "opacity-0",
                        )}
                      />
                      <span>{dataset_name}</span>
                    </div>
                  </CommandItem>
                ))}
              </CommandGroup>
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>
    </>
  );
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
          htmlFor="dataset_name"
          className="mb-1 block text-sm font-medium"
        >
          Dataset
        </label>
      </div>

      <DatasetSelector
        dataset_names={dataset_names}
        selectedDatasetName={selectedDatasetName}
        setSelectedDatasetName={setSelectedDatasetName}
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
          className="border-input bg-background w-full rounded-md border px-3 py-2 text-sm"
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
