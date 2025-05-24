import { useLayoutEffect, useReducer, useState } from "react";
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
import { AdvancedParametersAccordion } from "./AdvancedParametersAccordion";
import type { InferenceCacheSetting } from "~/utils/evaluations.server";

interface LaunchEvaluationModalProps {
  isOpen: boolean;
  onClose: () => void;
  datasetNames: string[];
}

interface DatasetSelectorProps {
  datasetNames: string[];
  selectedDatasetName: string | null;
  setSelectedDatasetName: (value: string | null) => void;
}

function DatasetSelector({
  datasetNames,
  selectedDatasetName,
  setSelectedDatasetName,
}: DatasetSelectorProps) {
  const [datasetPopoverOpen, setDatasetPopoverOpen] = useState(false);
  const [datasetInputValue, setDatasetInputValue] = useState("");

  const filteredDatasets = datasetInputValue
    ? datasetNames.filter((name) =>
        name.toLowerCase().includes(datasetInputValue.toLowerCase()),
      )
    : datasetNames;

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

interface EvaluationsFormValues {
  dataset_name: string | null;
  evaluation_name: string | null;
  variant_name: string | null;
  concurrency_limit: string;
  inference_cache: InferenceCacheSetting;
}

interface EvaluationsFormState extends EvaluationsFormValues {
  advancedParamsOpen: boolean;
  _lastAction: "sync" | "user" | null;
}

type FieldUpdateAction<K extends keyof EvaluationsFormValues> = {
  type: "UPDATE_FIELD";
  fieldName: K;
  value: EvaluationsFormValues[K];
};

type FormUpdateAction =
  | {
      [K in keyof EvaluationsFormValues]: FieldUpdateAction<K>;
    }[keyof EvaluationsFormValues]
  | { type: "SYNC_FORM_STATE"; state: Partial<EvaluationsFormState> }
  | { type: "TOGGLE_ADVANCED_PARAMS"; open?: boolean }
  | { type: "USER_ACTION" };

function EvaluationForm({ datasetNames }: { datasetNames: string[] }) {
  const fetcher = useFetcher();
  const config = useConfig();
  const evaluation_names = Object.keys(config.evaluations);

  const [formState, formUpdate] = useReducer(
    function evaluationsFormReducer(
      state: EvaluationsFormState,
      update: FormUpdateAction,
    ): EvaluationsFormState {
      switch (update.type) {
        case "UPDATE_FIELD":
          // HACK: There is a bug in Radix Select that appears to call its
          // onValueChange handler with an old value in some cases when state
          // updates are batched. As a result, 'UPDATE_FIELD' may be called with
          // the field's original value after 'SYNC_FORM_STATE' triggers its
          // update, resulting in the state being immediately reset to its
          // previous value. Using `_lastAction` to track that we just synced
          // the state and ignoring the following updates to work around it.
          if (state._lastAction === "sync") {
            return { ...state, _lastAction: null };
          }

          if (state[update.fieldName] === update.value) {
            return state;
          }

          return {
            ...state,
            _lastAction: null,
            [update.fieldName]: update.value,
            // Reset variant selection when evaluation changes only as a result
            // of the user's action
            variant_name:
              update.fieldName === "evaluation_name" &&
              state._lastAction === "user"
                ? null
                : state.variant_name,
          };

        case "SYNC_FORM_STATE": {
          const hasAdvancedParams = "inference_cache" in update.state;
          return {
            ...state,
            ...update.state,
            advancedParamsOpen: hasAdvancedParams
              ? true
              : state.advancedParamsOpen,
            _lastAction: "sync",
          };
        }
        case "TOGGLE_ADVANCED_PARAMS": {
          const open = update.open ?? !state.advancedParamsOpen;
          return { ...state, advancedParamsOpen: open };
        }
        case "USER_ACTION":
          return { ...state, _lastAction: "user" };
        default:
          return state;
      }
    },
    {
      dataset_name: null,
      evaluation_name: null,
      variant_name: null,
      concurrency_limit: "",
      inference_cache: "on",
      advancedParamsOpen: false,
      _lastAction: null,
    } as EvaluationsFormState,
  );

  const {
    dataset_name: selectedDatasetName,
    evaluation_name: selectedEvaluationName,
    variant_name: selectedVariantName,
    concurrency_limit: concurrencyLimit,
    inference_cache: inferenceCache,
    advancedParamsOpen,
  } = formState;

  // useLayoutEffect to update fields before paint to avoid flicker of old state
  useLayoutEffect(() => {
    const storedValues = getFromLocalStorage();
    if (storedValues) {
      formUpdate({ type: "SYNC_FORM_STATE", state: storedValues });
    }
  }, []);

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
        value={selectedEvaluationName || ""}
        onValueChange={(value) => {
          formUpdate({
            type: "UPDATE_FIELD",
            fieldName: "evaluation_name",
            value,
          });
        }}
      >
        <SelectTrigger onFocus={() => formUpdate({ type: "USER_ACTION" })}>
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
        datasetNames={datasetNames}
        selectedDatasetName={selectedDatasetName}
        setSelectedDatasetName={(value) => {
          formUpdate({
            type: "UPDATE_FIELD",
            fieldName: "dataset_name",
            value,
          });
        }}
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
        value={selectedVariantName || ""}
        onValueChange={(value) => {
          formUpdate({
            type: "UPDATE_FIELD",
            fieldName: "variant_name",
            value,
          });
        }}
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
          onChange={(event) => {
            const value = event.target.value;
            if (value === "" || Number.isInteger(Number(value))) {
              formUpdate({
                type: "UPDATE_FIELD",
                fieldName: "concurrency_limit",
                value,
              });
            }
          }}
          className="border-input bg-background w-full rounded-md border px-3 py-2 text-sm"
          required
        />
      </div>
      <div className="mt-4">
        <AdvancedParametersAccordion
          inference_cache={inferenceCache}
          setInferenceCache={(value) => {
            formUpdate({
              type: "UPDATE_FIELD",
              fieldName: "inference_cache",
              value,
            });
          }}
          isOpen={advancedParamsOpen}
          setIsOpen={(open) => {
            formUpdate({
              type: "TOGGLE_ADVANCED_PARAMS",
              open,
            });
          }}
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
  datasetNames,
}: LaunchEvaluationModalProps) {
  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Launch Evaluation</DialogTitle>
        </DialogHeader>

        <EvaluationForm datasetNames={datasetNames} />
      </DialogContent>
    </Dialog>
  );
}

const LOCAL_STORAGE_KEY = "tensorzero:evaluationForm";

function persistToLocalStorage(formData: FormData) {
  const formObject: Record<string, string> = {};
  for (const [key, value] of formData.entries()) {
    if (typeof value !== "string") continue;
    formObject[key] = value;
  }
  localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(formObject));
}

// TODO: consider using zod to parse and validate data
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
    return;
  }

  // TODO: add validation
  return parsed as Partial<EvaluationsFormState>;
}
