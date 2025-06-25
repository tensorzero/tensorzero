import { useState } from "react";
import { Button, ButtonIcon } from "~/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
} from "~/components/ui/dropdown-menu";
import { ChevronDown } from "lucide-react";
import {
  Command,
  CommandInput,
  CommandList,
  CommandEmpty,
  CommandGroup,
  CommandItem,
} from "~/components/ui/command";
import { Badge } from "~/components/ui/badge";
import { Plus, Check } from "lucide-react";
import type { DatasetCountInfo } from "~/utils/clickhouse/datasets";
import { AddToDataset } from "~/components/icons/Icons";
import {
  AlertDialog,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogFooter,
  AlertDialogAction,
  AlertDialogCancel,
} from "~/components/ui/alert-dialog";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "~/components/ui/tooltip";

export interface InferenceDatasetButtonProps {
  // Each dataset has a name, row count, and last updated timestamp.
  dataset_counts: DatasetCountInfo[];
  // Callback receives the chosen dataset name plus a flag indicating if it's new.
  onDatasetSelect: (
    dataset: string,
    output: "inherit" | "demonstration" | "none",
  ) => void;
  hasDemonstration: boolean;
}

export function AddToDatasetButton({
  dataset_counts,
  onDatasetSelect,
  hasDemonstration,
}: InferenceDatasetButtonProps) {
  const [open, setOpen] = useState(false);
  const [inputValue, setInputValue] = useState("");
  const [selectedDataset, setSelectedDataset] = useState("");
  const [outputDialogOpen, setOutputDialogOpen] = useState(false);

  // Sort datasets by last_updated (most recent first)
  const sortedDatasets = [...dataset_counts].sort(
    (a, b) =>
      new Date(b.last_updated).getTime() - new Date(a.last_updated).getTime(),
  );

  // Handle the output selection from the alert dialog
  const handleOutputSelect = (output: "inherit" | "demonstration" | "none") => {
    onDatasetSelect(selectedDataset, output);
    setOutputDialogOpen(false);
    setOpen(false);
  };

  return (
    <TooltipProvider>
      <DropdownMenu open={open} onOpenChange={setOpen}>
        <DropdownMenuTrigger asChild>
          <Button variant="outline" size="sm">
            {selectedDataset ? (
              <div className="flex items-center">
                {sortedDatasets.find(
                  (d) => d.dataset_name === selectedDataset,
                ) ? (
                  <>
                    {selectedDataset}
                    <Badge variant="secondary" className="ml-2">
                      {sortedDatasets
                        .find((d) => d.dataset_name === selectedDataset)
                        ?.count.toLocaleString()}{" "}
                      rows
                    </Badge>
                  </>
                ) : (
                  <>
                    <Plus className="mr-2 h-4 w-4 text-blue-500" />
                    {selectedDataset}
                    <Badge
                      variant="outline"
                      className="ml-2 bg-blue-50 text-blue-500"
                    >
                      New Dataset
                    </Badge>
                  </>
                )}
              </div>
            ) : (
              <>
                <ButtonIcon as={AddToDataset} variant="tertiary" />
                Add to dataset
              </>
            )}
            <ButtonIcon as={ChevronDown} variant="tertiary" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent className="w-80 p-0">
          <Command>
            <CommandInput
              placeholder="Create or find dataset..."
              value={inputValue}
              onValueChange={setInputValue}
              className="h-9"
            />
            <CommandList>
              <CommandGroup heading="Existing datasets">
                {sortedDatasets
                  .filter((d) =>
                    d.dataset_name
                      .toLowerCase()
                      .includes(inputValue.toLowerCase()),
                  )
                  .map((dataset) => (
                    <CommandItem
                      key={dataset.dataset_name}
                      onSelect={() => {
                        setSelectedDataset(dataset.dataset_name);
                        setInputValue("");
                        setOutputDialogOpen(true); // open alert dialog
                      }}
                      className="flex items-center justify-between"
                    >
                      <div className="flex min-w-0 flex-1 items-center">
                        <Check
                          className={`mr-2 h-4 w-4 flex-shrink-0 ${
                            selectedDataset === dataset.dataset_name
                              ? "opacity-100"
                              : "opacity-0"
                          }`}
                        />
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <span className="truncate">
                              {dataset.dataset_name}
                            </span>
                          </TooltipTrigger>
                          <TooltipContent>
                            <p>{dataset.dataset_name}</p>
                          </TooltipContent>
                        </Tooltip>
                      </div>
                      <Badge variant="secondary">
                        {dataset.count.toLocaleString()} rows
                      </Badge>
                    </CommandItem>
                  ))}
              </CommandGroup>

              {inputValue.trim() && (
                <CommandGroup heading="New dataset">
                  <CommandItem
                    onSelect={() => {
                      const trimmed = inputValue.trim();
                      if (trimmed) {
                        setSelectedDataset(trimmed);
                        setInputValue("");
                        setOutputDialogOpen(true); // open alert dialog
                      }
                    }}
                    className="flex items-center justify-between"
                  >
                    <div className="flex items-center">
                      <Plus className="mr-2 h-4 w-4 text-blue-500" />
                      Create “{inputValue}” dataset...
                    </div>
                  </CommandItem>
                </CommandGroup>
              )}

              <CommandEmpty className="px-4 py-2 text-sm">
                No datasets found.
              </CommandEmpty>
            </CommandList>
          </Command>
        </DropdownMenuContent>
      </DropdownMenu>

      <AlertDialog open={outputDialogOpen} onOpenChange={setOutputDialogOpen}>
        <AlertDialogContent className="min-w-[600px]">
          <AlertDialogHeader>
            <AlertDialogTitle>
              What should be the datapoint's output?
            </AlertDialogTitle>
            <AlertDialogDescription>
              Each datapoint includes an optional output field. The choice
              should depend on your use case. For example, you might prefer
              demonstrations for fine-tuning.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter className="flex justify-center gap-4">
            <AlertDialogCancel onClick={() => setOutputDialogOpen(false)}>
              Cancel
            </AlertDialogCancel>
            <AlertDialogAction onClick={() => handleOutputSelect("inherit")}>
              Inference Output
            </AlertDialogAction>
            {hasDemonstration && (
              <AlertDialogAction
                onClick={() => handleOutputSelect("demonstration")}
              >
                Demonstration
              </AlertDialogAction>
            )}
            <AlertDialogAction onClick={() => handleOutputSelect("none")}>
              None
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </TooltipProvider>
  );
}
