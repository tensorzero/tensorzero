import { useMemo, useState } from "react";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "~/components/ui/popover";
import { Button, ButtonIcon } from "~/components/ui/button";
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
import { useDatasetCounts } from "~/hooks/use-dataset-counts";

interface DatasetSelectorProps {
  selected?: string;
  onSelect: (dataset: string, isNew: boolean) => void;
  functionName?: string;
  placeholder?: string;
  className?: string;
  allowCreation?: boolean;
  buttonProps?: React.ComponentProps<typeof Button>;
  disabled?: boolean;
}

// TODO Create new datasets within this component
export function DatasetSelector({
  selected,
  onSelect,
  functionName,
  placeholder = "Select a dataset",
  allowCreation = true,
  className,
  buttonProps,
  disabled = false,
}: DatasetSelectorProps) {
  const [open, setOpen] = useState(false);
  const [inputValue, setInputValue] = useState("");

  const {
    data: datasets = [],
    isLoading,
    isError,
  } = useDatasetCounts(functionName);

  // Datasets sorted by last updated date for initial display
  const recentlyUpdatedDatasets = useMemo(
    () =>
      [...(datasets ?? [])].sort((a, b) => {
        return (
          new Date(b.lastUpdated).getTime() - new Date(a.lastUpdated).getTime()
        );
      }),
    [datasets],
  );

  // Selected dataset, if an existing one was selected
  const existingSelectedDataset = useMemo(
    () => datasets?.find((dataset) => dataset.name === selected),
    [datasets, selected],
  );

  return (
    <div className={clsx("flex flex-col space-y-2", className)}>
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild disabled={disabled}>
          <Button
            variant="outline"
            role="combobox"
            aria-expanded={open}
            className="group w-full justify-between border font-normal"
            disabled={disabled}
            {...buttonProps}
          >
            {selected ? (
              <div className="flex w-full min-w-0 flex-1 items-center gap-x-2">
                {existingSelectedDataset ? (
                  <Table
                    size={16}
                    className="h-4 w-4 shrink-0 text-green-700"
                  />
                ) : (
                  <TablePlus className="h-4 w-4 shrink-0 text-blue-600" />
                )}
                <span className="truncate font-mono text-sm">
                  {existingSelectedDataset?.name ?? selected}
                </span>
              </div>
            ) : (
              <span className="flex flex-row items-center gap-2">
                <ButtonIcon as={Table} variant="tertiary" />
                {placeholder}
              </span>
            )}

            <ButtonIcon
              as={ChevronDown}
              className={clsx(
                "h-4 w-4 shrink-0 transition duration-300 ease-out",
                open ? "-rotate-180" : "rotate-0",
              )}
              variant="tertiary"
            />
          </Button>
        </PopoverTrigger>

        <PopoverContent
          className="w-[var(--radix-popover-trigger-width)] min-w-64 p-0"
          align="start"
        >
          <Command>
            {/* TODO Naming/character constraints/disallow typing certain characters? */}
            <CommandInput
              placeholder={
                allowCreation
                  ? recentlyUpdatedDatasets.length > 0
                    ? "Create or find a dataset..."
                    : "Create a new dataset..."
                  : "Find a dataset..."
              }
              value={inputValue}
              onValueChange={setInputValue}
              className="h-9"
            />

            {isLoading ? (
              <div className="text-fg-muted flex items-center justify-center py-4 text-sm">
                Loading datasets...
              </div>
            ) : isError ? (
              <div className="text-fg-muted flex items-center justify-center py-4 text-sm">
                There was an error loading datasets.
              </div>
            ) : (
              <CommandList>
                <CommandEmpty className="flex items-center justify-center px-4 py-4 text-sm">
                  No datasets found.
                </CommandEmpty>

                {recentlyUpdatedDatasets.length > 0 && (
                  <CommandGroup>
                    {recentlyUpdatedDatasets.map((dataset) => (
                      <CommandItem
                        key={dataset.name}
                        value={dataset.name}
                        onSelect={() => {
                          onSelect(dataset.name, false);
                          setInputValue("");
                          setOpen(false);
                        }}
                        className="group flex w-full items-center gap-2"
                      >
                        <div className="flex min-w-0 flex-1 items-center gap-2">
                          {selected === dataset.name ? (
                            <TableCheck size={16} className="text-green-700" />
                          ) : (
                            <Table size={16} className="text-fg-muted" />
                          )}
                          <span
                            className={clsx(
                              "group-hover:text-fg-primary truncate font-mono",
                              selected === dataset.name
                                ? "font-medium"
                                : "font-normal",
                            )}
                          >
                            {dataset.name}
                          </span>
                        </div>
                        <span
                          className={clsx(
                            "min-w-8 flex-shrink-0 whitespace-nowrap text-right text-sm",
                            selected === dataset.name
                              ? "text-fg-secondary font-medium"
                              : "text-fg-tertiary font-normal",
                          )}
                        >
                          {dataset.count.toLocaleString()}
                        </span>
                      </CommandItem>
                    ))}
                  </CommandGroup>
                )}
              </CommandList>
            )}

            {
              // If creation is allowed...
              allowCreation &&
                // ...and the user has typed something...
                inputValue.trim() &&
                // ...and the dataset doesn't exist in the list of recently updated datasets...
                !recentlyUpdatedDatasets.some(
                  (dataset) =>
                    dataset.name.toLowerCase() ===
                    inputValue.trim().toLowerCase(),
                ) && (
                  // ...then show the "New dataset" group
                  <CommandGroup heading="New dataset">
                    <CommandItem
                      value={`create-${inputValue.trim()}`}
                      onSelect={() => {
                        onSelect(inputValue.trim(), true);
                        setInputValue("");
                        setOpen(false);
                      }}
                      className="flex items-center justify-between"
                    >
                      <div className="flex items-center truncate">
                        <TablePlus className="mr-2 h-4 w-4 text-blue-600" />
                        <span className="text-fg-primary truncate font-mono text-sm">
                          {inputValue.trim()}
                        </span>
                      </div>
                    </CommandItem>
                  </CommandGroup>
                )
            }
          </Command>
        </PopoverContent>
      </Popover>
    </div>
  );
}
