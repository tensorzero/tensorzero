import type { Control } from "react-hook-form";
import {
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
} from "~/components/ui/form";
import { useState } from "react";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "~/components/ui/popover";
import { Button } from "~/components/ui/button";
import { Badge } from "~/components/ui/badge";
import { ChevronsUpDown } from "lucide-react";
import { Dataset, PlusSquare, TableCheck } from "~/components/icons/Icons";
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
        <FormItem className="flex flex-col gap-y-1">
          <FormLabel>Dataset</FormLabel>
          <div className="w-full max-w-160 space-y-2">
            <Popover open={open} onOpenChange={setOpen}>
              <PopoverTrigger asChild>
                <Button
                  variant="outline"
                  role="combobox"
                  aria-expanded={open}
                  className="w-full font-normal"
                >
                  <div className="min-w-0 flex-1">
                    {field.value ? (
                      (() => {
                        const existingDataset = dataset_counts.find(
                          (d) => d.dataset_name === field.value,
                        );
                        if (existingDataset) {
                          return (
                            <div className="flex w-full min-w-0 flex-1 items-center gap-x-2">
                              <Dataset
                                size={16}
                                className="text-fg-muted shrink-0"
                              />
                              <span className="truncate font-mono text-sm">
                                {existingDataset.dataset_name}
                              </span>
                            </div>
                          );
                        } else {
                          return (
                            <div className="flex w-full items-center gap-x-1">
                              <div className="flex min-w-0 flex-1 items-center gap-x-1">
                                <PlusSquare className="h-4 w-4 shrink-0 text-blue-600" />
                                <span className="truncate font-mono text-sm">
                                  {field.value}
                                </span>
                              </div>
                              <Badge
                                variant="outline"
                                className="flex-shrink-0 bg-blue-50 whitespace-nowrap text-blue-600"
                              >
                                New Dataset
                              </Badge>
                            </div>
                          );
                        }
                      })()
                    ) : (
                      <span className="text-fg-muted flex text-sm">
                        Create or select a dataset...
                      </span>
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
                    placeholder="Search or create a dataset..."
                    value={inputValue}
                    onValueChange={handleInputChange}
                  />
                  <CommandList>
                    <CommandEmpty className="px-4 py-2 text-sm">
                      No datasets found.
                    </CommandEmpty>
                    {inputValue.trim() && (
                      <CommandGroup className="border-border border-b">
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
                            <PlusSquare
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
                          <span className="text-right">Rows</span>
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
                              <Dataset size={16} className="text-fg-muted" />
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
          {(() => {
            if (field.value) {
              const existingDataset = dataset_counts.find(
                (d) => d.dataset_name === field.value,
              );
              if (existingDataset) {
                return (
                  <FormDescription className="text-fg-tertiary font-medium">
                    Rows: {existingDataset.count.toLocaleString()}
                  </FormDescription>
                );
              }
            }
            return null; // Return null if no existing dataset is selected or no value
          })()}
        </FormItem>
      )}
    />
  );
}
