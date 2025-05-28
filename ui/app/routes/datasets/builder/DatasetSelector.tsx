import type { Control } from "react-hook-form";
import {
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
} from "~/components/ui/form";
import { useRef, useState } from "react";
import { Button } from "~/components/ui/button";
import { ChevronDown } from "lucide-react";
import { Table, TablePlus, TableCheck, } from "~/components/icons/Icons";
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
import { useClickOutside } from "~/hooks/use-click-outside";

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
  const commandRef = useRef<HTMLDivElement>(null);

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
    if (input.trim() !== "" && !open) {
      setOpen(true);
    }
  };

  useClickOutside(commandRef, () => setOpen(false));

  return (
    <FormField
      control={control}
      name="dataset"
      render={({ field }) => (
        <FormItem className="flex flex-col gap-y-1">
          <FormLabel className="text-fg-primary font-medium text-sm">Dataset</FormLabel>
          <div className="w-full max-w-160 space-y-1">
            <div className="relative h-10">
            <div 
              ref={commandRef} 
              className={clsx(
                "absolute top-0 left-0 right-0 z-50 rounded-lg border border-border bg-background transition-shadow transition-transform ease-out duration-300",
                open ? "shadow-2xl" : "hover:shadow-xs active:shadow-none active:scale-99 scale-100 shadow-none"
              )}
            >
              {/* -------- Selector button -------- */}
              <Button
                variant="ghost"
                role="combobox"
                type="button"
                aria-expanded={open}
                className="w-full px-3 hover:bg-transparent font-normal cursor-pointer group"
                onClick={() => setOpen(!open)}
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
                    <div className="flex items-center gap-x-2 text-fg-muted">
                      <Table className="h-4 w-4 shrink-0 text-fg-muted" />
                      <span className="text-fg-secondary flex text-sm">
                        Select a dataset
                      </span>
                    </div>
                  )}
                </div>
                <ChevronDown className={clsx("ml-2 h-4 w-4 shrink-0 text-fg-muted group-hover:text-fg-tertiary transition-colors transition-transform ease-out duration-300", open ? "-rotate-180" : "rotate-0")} />
              </Button>

              {/* -------- Command -------- */}

              <Command
                className={clsx(
                  "border-t border-border rounded-none bg-transparent overflow-hidden transition-all ease-out duration-300",
                  open ? "max-h-[500px] opacity-100" : "max-h-0 opacity-0"
                )}
              >
                <CommandInput
                  placeholder="Create or find a dataset..."
                  value={inputValue}
                  onValueChange={handleInputChange}
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
            </div>
            </div>
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
