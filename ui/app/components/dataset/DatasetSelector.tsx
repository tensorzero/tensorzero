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
import { z } from "zod";
import { useDatasetCountFetcher } from "~/routes/api/datasets/counts.route";

export const DatasetCountResponse = z.object({
  datasets: z.array(
    z.object({
      name: z.string(),
      count: z.number(),
      lastUpdated: z.string().datetime(),
    }),
  ),
});

interface DatasetSelectorProps {
  selected?: string;
  onSelect: (dataset: string, isNew: boolean) => void;
  functionName?: string;
  disabled?: boolean;
  placeholder?: string;
  className?: string;
  allowCreation?: boolean;
  buttonProps?: React.ComponentProps<typeof Button>;
}

// TODO Create new datasets within this component

export default function DatasetSelector({
  selected,
  onSelect,
  functionName,
  disabled,
  placeholder = "Select a dataset",
  allowCreation = true,
  className,
  buttonProps,
}: DatasetSelectorProps) {
  const [open, setOpen] = useState(false);
  const [inputValue, setInputValue] = useState("");

  const { datasets, isLoading } = useDatasetCountFetcher(functionName);

  // Datasets sorted by last updated date for initial display
  const recentlyUpdatedDatasets = useMemo(
    () =>
      [...(datasets ?? [])].sort((a, b) => {
        return (
          new Date(b.last_updated).getTime() -
          new Date(a.last_updated).getTime()
        );
      }),
    [datasets],
  );

  // Selected dataset, if an existing one was selected
  const existingSelectedDataset = useMemo(
    () => datasets?.find((dataset) => dataset.dataset_name === selected),
    [datasets, selected],
  );

  return (
    <div className={clsx("flex flex-col space-y-2", className)}>
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
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
                  {existingSelectedDataset?.dataset_name ?? selected}
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
            ) : (
              <CommandList>
                <CommandEmpty className="flex items-center justify-center px-4 py-4 text-sm">
                  No datasets found.
                </CommandEmpty>

                {recentlyUpdatedDatasets.length > 0 && (
                  <CommandGroup>
                    {recentlyUpdatedDatasets.map((dataset) => (
                      <CommandItem
                        key={dataset.dataset_name}
                        value={dataset.dataset_name}
                        onSelect={() => {
                          onSelect(dataset.dataset_name, false);
                          setInputValue("");
                          setOpen(false);
                        }}
                        className="group flex w-full items-center gap-2"
                      >
                        <div className="flex min-w-0 flex-1 items-center gap-2">
                          {selected === dataset.dataset_name ? (
                            <TableCheck size={16} className="text-green-700" />
                          ) : (
                            <Table size={16} className="text-fg-muted" />
                          )}
                          <span
                            className={clsx(
                              "group-hover:text-fg-primary truncate font-mono",
                              selected === dataset.dataset_name
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
                            selected === dataset.dataset_name
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
                    dataset.dataset_name.toLowerCase() ===
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
