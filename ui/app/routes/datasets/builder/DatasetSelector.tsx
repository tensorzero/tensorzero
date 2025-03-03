import type { Control } from "react-hook-form";
import { FormField, FormItem, FormLabel } from "~/components/ui/form";
import { useState } from "react";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "~/components/ui/popover";
import { Button } from "~/components/ui/button";
import { Badge } from "~/components/ui/badge";
import { Check, ChevronsUpDown, Plus } from "lucide-react";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "~/components/ui/command";
import { cn } from "~/utils/common";
import type { DatasetCountInfo } from "~/utils/clickhouse/datasets";
import type { DatasetBuilderFormValues } from "./types";

export function DatasetSelector({
  control,
  dataset_counts,
  setIsNewDataset,
}: {
  control: Control<DatasetBuilderFormValues>;
  dataset_counts: DatasetCountInfo[];
  setIsNewDataset: (isNewDataset: boolean | null) => void;
}) {
  const [open, setOpen] = useState(false);
  const [inputValue, setInputValue] = useState("");

  const sortedDatasets = [...dataset_counts].sort(
    (a, b) =>
      new Date(b.last_updated).getTime() - new Date(a.last_updated).getTime(),
  );

  // We would probably want to remove the creation command element if there is an exact match
  // But, it was hard to implement because command has its own state management that conflicts
  // const hasExactMatch = sortedDatasets.some(
  //   (d) => d.dataset_name.toLowerCase() === inputValue.trim().toLowerCase(),
  // );

  const handleInputChange = (input: string) => {
    setInputValue(input);
  };

  return (
    <FormField
      control={control}
      name="dataset"
      render={({ field }) => (
        <FormItem>
          <FormLabel>Dataset</FormLabel>
          <div className="grid gap-x-8 md:grid-cols-2">
            <div className="w-full space-y-2">
              <Popover open={open} onOpenChange={setOpen}>
                <PopoverTrigger className="border-gray-200 bg-gray-50" asChild>
                  <Button
                    variant="outline"
                    role="combobox"
                    aria-expanded={open}
                    className="w-full justify-between font-normal"
                  >
                    <div>
                      {field.value ? (
                        <div className="flex items-center">
                          {dataset_counts.find(
                            (d) => d.dataset_name === field.value,
                          ) ? (
                            <div className="flex items-center">
                              {field.value}
                              <span className="ml-2">
                                <Badge variant="secondary" className="ml-2">
                                  {dataset_counts
                                    .find((d) => d.dataset_name === field.value)
                                    ?.count.toLocaleString()}{" "}
                                  rows
                                </Badge>
                              </span>
                            </div>
                          ) : (
                            <>
                              <Plus className="mr-2 h-4 w-4 text-blue-500" />
                              {field.value}
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
                        "Create or select a dataset..."
                      )}
                    </div>
                    <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
                  </Button>
                </PopoverTrigger>
                <PopoverContent
                  className="w-[var(--radix-popover-trigger-width)] p-0"
                  align="start"
                >
                  <Command>
                    <CommandInput
                      placeholder="Search datasets..."
                      value={inputValue}
                      onValueChange={handleInputChange}
                      className="h-9"
                    />
                    <CommandList>
                      <CommandEmpty className="px-4 py-2 text-sm">
                        No datasets found.
                      </CommandEmpty>
                      <CommandGroup heading="New Dataset">
                        <CommandItem
                          onSelect={() => {
                            field.onChange(inputValue.trim());
                            setInputValue("");
                            setOpen(false);
                            setIsNewDataset(true);
                          }}
                          className="flex items-center justify-between"
                        >
                          <div className="flex items-center">
                            <Plus className="mr-2 h-4 w-4 text-blue-500" />
                            <span>
                              {inputValue.trim() || "Start typing..."}
                            </span>
                          </div>
                          <Badge
                            variant="outline"
                            className="bg-blue-50 text-blue-500"
                          >
                            Create New Dataset
                          </Badge>
                        </CommandItem>
                      </CommandGroup>
                      <CommandGroup heading="Existing Datasets">
                        {sortedDatasets.map((dataset) => (
                          <CommandItem
                            key={dataset.dataset_name}
                            value={dataset.dataset_name}
                            onSelect={() => {
                              field.onChange(dataset.dataset_name);
                              setInputValue("");
                              setOpen(false);
                              setIsNewDataset(false);
                            }}
                            className="flex items-center justify-between"
                          >
                            <div className="flex items-center">
                              <Check
                                className={cn(
                                  "mr-2 h-4 w-4",
                                  field.value === dataset.dataset_name
                                    ? "opacity-100"
                                    : "opacity-0",
                                )}
                              />
                              <span>{dataset.dataset_name}</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <Badge variant="secondary">
                                {dataset.count.toLocaleString()} rows
                              </Badge>
                            </div>
                          </CommandItem>
                        ))}
                      </CommandGroup>
                    </CommandList>
                  </Command>
                </PopoverContent>
              </Popover>
            </div>
          </div>
        </FormItem>
      )}
    />
  );
}
