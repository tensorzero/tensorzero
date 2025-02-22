import { useState } from "react";
import { Button } from "~/components/ui/button";
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

export interface InferenceDatasetButtonProps {
  // Each dataset has a name, row count, and last updated timestamp.
  datasets: DatasetCountInfo[];
  // Callback receives the chosen dataset name plus a flag indicating if it’s new.
  onDatasetSelect: (dataset: string) => void;
  isLoading?: boolean;
}

export function AddToDatasetButton({
  datasets,
  onDatasetSelect,
  isLoading = false,
}: InferenceDatasetButtonProps) {
  const [open, setOpen] = useState(false);
  const [inputValue, setInputValue] = useState("");
  const [selectedDataset, setSelectedDataset] = useState("");

  // Sort datasets by last_updated (most recent first)
  const sortedDatasets = [...datasets].sort(
    (a, b) =>
      new Date(b.last_updated).getTime() - new Date(a.last_updated).getTime(),
  );

  return (
    <DropdownMenu open={open} onOpenChange={setOpen}>
      <DropdownMenuTrigger asChild>
        <Button
          variant="outline"
          disabled={isLoading}
          className="w-full justify-between"
        >
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
            "Select or create a dataset..."
          )}
          <ChevronDown className="ml-2 h-4 w-4" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent className="w-80 p-0">
        <Command>
          <CommandInput
            placeholder="Search datasets..."
            value={inputValue}
            onValueChange={setInputValue}
            className="h-9"
          />
          <CommandList>
            <CommandEmpty className="px-4 py-2 text-sm">
              No datasets found.
            </CommandEmpty>
            <CommandGroup heading="New Dataset">
              <CommandItem
                onSelect={() => {
                  const trimmed = inputValue.trim();
                  if (trimmed) {
                    setSelectedDataset(trimmed);
                    onDatasetSelect(trimmed);
                    setInputValue("");
                    setOpen(false);
                  }
                }}
                className="flex items-center justify-between"
              >
                <div className="flex items-center">
                  <Plus className="mr-2 h-4 w-4 text-blue-500" />
                  <span>{inputValue.trim() || "Type..."}</span>
                </div>
                <Badge variant="outline" className="bg-blue-50 text-blue-500">
                  Create New
                </Badge>
              </CommandItem>
            </CommandGroup>
            <CommandGroup heading="Existing Datasets">
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
                      onDatasetSelect(dataset.dataset_name);
                      setInputValue("");
                      setOpen(false);
                    }}
                    className="flex items-center justify-between"
                  >
                    <div className="flex items-center">
                      <Check
                        className={`mr-2 h-4 w-4 ${
                          selectedDataset === dataset.dataset_name
                            ? "opacity-100"
                            : "opacity-0"
                        }`}
                      />
                      <span>{dataset.dataset_name}</span>
                    </div>
                    <Badge variant="secondary">
                      {dataset.count.toLocaleString()} rows
                    </Badge>
                  </CommandItem>
                ))}
            </CommandGroup>
          </CommandList>
        </Command>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
