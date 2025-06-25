import type { Control } from "react-hook-form";
import { FormField, FormItem } from "~/components/ui/form";
import { useState } from "react";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "~/components/ui/popover";
import { Button } from "~/components/ui/button";
import { ChevronDown } from "lucide-react";
import { Table, TablePlus, TableCheck } from "~/components/icons/Icons";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "~/components/ui/command";
import clsx from "clsx";
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
        <FormItem className="flex flex-col gap-1">
          <div className="w-full space-y-2">
            <Popover open={open} onOpenChange={setOpen}>
              <PopoverTrigger asChild>
                <Button
                  variant="outline"
                  role="combobox"
                  aria-expanded={open}
                  className="group border-border hover:border-border-accent hover:bg-bg-primary w-full justify-between border font-normal hover:cursor-pointer"
                >
                  <div className="min-w-0 flex-1">
                    {field.value ? (
                      (() => {
                        const existingDataset = dataset_counts.find(
                          (d) => d.dataset_name === field.value,
                        );
                        if (existingDataset) {
                          return (
                            // Selector — existing dataset
                            <div className="flex w-full min-w-0 flex-1 items-center gap-x-2">
                              <Table
                                size={16}
                                className="h-4 w-4 shrink-0 text-green-700"
                              />
                              <span className="truncate font-mono text-sm">
                                {existingDataset.dataset_name}
                              </span>
                            </div>
                          );
                        } else {
                          // Selector — new dataset
                          return (
                            <div className="flex w-full min-w-0 flex-1 items-center gap-x-2">
                              <TablePlus className="h-4 w-4 shrink-0 text-blue-600" />
                              <span className="truncate font-mono text-sm">
                                {field.value}
                              </span>
                            </div>
                          );
                        }
                      })()
                    ) : (
                      // Selector — empty state
                      <div className="text-fg-muted flex items-center gap-x-2">
                        <Table className="text-fg-muted h-4 w-4 shrink-0" />
                        <span className="text-fg-secondary flex text-sm">
                          Select a dataset
                        </span>
                      </div>
                    )}
                  </div>
                  <ChevronDown
                    className={clsx(
                      "text-fg-muted group-hover:text-fg-tertiary ml-2 h-4 w-4 shrink-0 transition-colors transition-transform duration-300 ease-out",
                      open ? "-rotate-180" : "rotate-0",
                    )}
                  />
                </Button>
              </PopoverTrigger>
              <PopoverContent
                className="w-[var(--radix-popover-trigger-width)] p-0"
                align="start"
              >
                <Command>
                  <CommandInput
                    placeholder="Create or find a dataset..."
                    value={inputValue}
                    onValueChange={handleInputChange}
                    className="h-9"
                  />
                  <CommandList>
                    <CommandEmpty className="px-4 py-2 text-sm">
                      No datasets found.
                    </CommandEmpty>
                    {inputValue.trim() && (
                      <CommandGroup>
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
                            <TablePlus
                              className={clsx(
                                "mr-2 h-4 w-4",
                                inputValue.trim()
                                  ? "text-blue-600"
                                  : "text-fg-muted",
                              )}
                            />
                            <span
                              className={clsx(
                                "text-sm",
                                inputValue.trim()
                                  ? "text-fg-primary font-mono"
                                  : "text-fg-muted font-normal",
                              )}
                            >
                              {inputValue.trim() ||
                                "Start typing to create a new dataset..."}
                            </span>
                          </div>
                          <span
                            className={clsx(
                              "text-bg-primary rounded-md bg-blue-600 px-2 py-1 text-xs",
                              inputValue.trim() ? "font-medium" : "invisible",
                            )}
                          >
                            Create New Dataset
                          </span>
                        </CommandItem>
                      </CommandGroup>
                    )}
                    <CommandGroup
                      heading={
                        <div className="text-fg-tertiary flex w-full items-center justify-between">
                          <span>Existing Datasets</span>
                          <span>Rows</span>
                        </div>
                      }
                    >
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
                          className="group flex w-full items-center gap-2"
                        >
                          <div className="flex min-w-0 flex-1 items-center gap-2">
                            {field.value === dataset.dataset_name ? (
                              <TableCheck
                                size={16}
                                className="text-green-700"
                              />
                            ) : (
                              <Table size={16} className="text-fg-muted" />
                            )}
                            <span
                              className={clsx(
                                "group-hover:text-fg-primary truncate font-mono",
                                field.value === dataset.dataset_name
                                  ? "font-medium"
                                  : "font-normal",
                              )}
                            >
                              {dataset.dataset_name}
                            </span>
                          </div>
                          <span
                            className={clsx(
                              "min-w-8 flex-shrink-0 text-right text-sm whitespace-nowrap",
                              field.value === dataset.dataset_name
                                ? "text-fg-secondary font-medium"
                                : "text-fg-tertiary font-normal",
                            )}
                          >
                            {dataset.count.toLocaleString()}
                          </span>
                        </CommandItem>
                      ))}
                    </CommandGroup>
                  </CommandList>
                </Command>
              </PopoverContent>
            </Popover>
          </div>
        </FormItem>
      )}
    />
  );
}
