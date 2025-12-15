import { useCallback, useMemo, useState } from "react";
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
  CommandInput,
  CommandList,
} from "~/components/ui/command";
import clsx from "clsx";
import { useDatasetCounts } from "~/hooks/use-dataset-counts";
import { Combobox, ComboboxMenuItems } from "~/components/ui/combobox";

export const DatasetSelectorVariant = {
  BUTTON: "button",
  INPUT: "input",
} as const;

export type DatasetSelectorVariant =
  (typeof DatasetSelectorVariant)[keyof typeof DatasetSelectorVariant];

interface DatasetSelectorProps {
  variant: DatasetSelectorVariant;
  selected: string | null;
  onSelect: (dataset: string, isNew: boolean) => void;
  functionName?: string;
  placeholder?: string;
  className?: string;
  allowCreation?: boolean;
  disabled?: boolean;
}

export function DatasetSelector({
  variant,
  selected,
  onSelect,
  functionName,
  placeholder,
  allowCreation = variant === DatasetSelectorVariant.BUTTON,
  className,
  disabled = false,
}: DatasetSelectorProps) {
  const {
    data: datasets = [],
    isLoading,
    isError,
  } = useDatasetCounts(functionName);

  const sortedDatasetNames = useMemo(
    () =>
      [...(datasets ?? [])]
        .sort(
          (a, b) =>
            new Date(b.lastUpdated).getTime() -
            new Date(a.lastUpdated).getTime(),
        )
        .map((d) => d.name),
    [datasets],
  );

  const datasetsByName = useMemo(
    () => new Map(datasets.map((d) => [d.name, d])),
    [datasets],
  );

  const getItemIcon = useCallback(
    (item: string | null, isSelected: boolean) => {
      if (!item) {
        return <TablePlus className="h-4 w-4 text-blue-600" />;
      }
      const exists = datasetsByName.has(item);
      if (isSelected && exists) {
        return <TableCheck size={16} className="text-green-700" />;
      }
      if (isSelected && !exists) {
        return <TablePlus className="h-4 w-4 text-blue-600" />;
      }
      return <Table size={16} className="text-fg-muted" />;
    },
    [datasetsByName],
  );

  const getItemSuffix = useCallback(
    (item: string | null) => {
      if (!item) return null;
      const dataset = datasetsByName.get(item);
      return dataset?.count.toLocaleString();
    },
    [datasetsByName],
  );

  const getItemDataAttributes = useCallback(
    (item: string) => ({ "data-dataset-name": item }),
    [],
  );

  const getSelectedDataset = useCallback(
    (name: string | null | undefined) => {
      if (!name) return null;
      return datasetsByName.get(name);
    },
    [datasetsByName],
  );

  const computedPlaceholder = useMemo(() => {
    if (placeholder) return placeholder;
    if (allowCreation) {
      return sortedDatasetNames.length > 0
        ? "Create or find dataset"
        : "Create dataset";
    }
    return "Select dataset";
  }, [placeholder, allowCreation, sortedDatasetNames.length]);

  if (variant === DatasetSelectorVariant.INPUT) {
    return (
      <Combobox
        selected={selected}
        onSelect={onSelect}
        items={sortedDatasetNames}
        getItemIcon={getItemIcon}
        getItemSuffix={getItemSuffix}
        getItemDataAttributes={getItemDataAttributes}
        placeholder={computedPlaceholder}
        emptyMessage="No datasets found."
        disabled={disabled}
        allowCreation={allowCreation}
        creationHint={
          allowCreation ? "Type to create a new dataset" : undefined
        }
        createHeading="New dataset"
        loading={isLoading}
        loadingMessage="Loading datasets..."
        error={isError}
        errorMessage="There was an error loading datasets."
      />
    );
  }

  return (
    <ButtonVariant
      selected={selected}
      onSelect={onSelect}
      sortedDatasetNames={sortedDatasetNames}
      isLoading={isLoading}
      isError={isError}
      getItemIcon={getItemIcon}
      getItemSuffix={getItemSuffix}
      getItemDataAttributes={getItemDataAttributes}
      getSelectedDataset={getSelectedDataset}
      computedPlaceholder={computedPlaceholder}
      allowCreation={allowCreation}
      className={className}
      disabled={disabled}
    />
  );
}

interface ButtonVariantProps {
  selected: string | null;
  onSelect: (dataset: string, isNew: boolean) => void;
  sortedDatasetNames: string[];
  isLoading: boolean;
  isError: boolean;
  getItemIcon: (item: string | null, isSelected: boolean) => React.ReactNode;
  getItemSuffix: (item: string | null) => React.ReactNode;
  getItemDataAttributes: (item: string) => Record<string, string>;
  getSelectedDataset: (
    name: string | null | undefined,
  ) => { name: string; count: number } | null | undefined;
  computedPlaceholder: string;
  allowCreation: boolean;
  className?: string;
  disabled: boolean;
}

function ButtonVariant({
  selected,
  onSelect,
  sortedDatasetNames,
  isLoading,
  isError,
  getItemIcon,
  getItemSuffix,
  getItemDataAttributes,
  getSelectedDataset,
  computedPlaceholder,
  allowCreation,
  className,
  disabled,
}: ButtonVariantProps) {
  const [open, setOpen] = useState(false);
  const [inputValue, setInputValue] = useState("");

  const selectedDataset = getSelectedDataset(selected);

  const showCreateOption =
    allowCreation &&
    Boolean(inputValue.trim()) &&
    !sortedDatasetNames.some(
      (name) => name.toLowerCase() === inputValue.trim().toLowerCase(),
    );

  const handleSelectItem = (item: string, isNew: boolean) => {
    onSelect(item, isNew);
    setInputValue("");
    setOpen(false);
  };

  return (
    <div className={clsx("flex flex-col space-y-2", className)}>
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild disabled={disabled}>
          <Button
            variant="outline"
            size="sm"
            role="combobox"
            aria-expanded={open}
            className="group w-full justify-between border"
            disabled={disabled}
          >
            {selected ? (
              <div className="flex w-full min-w-0 flex-1 items-center gap-x-2">
                {selectedDataset ? (
                  <Table
                    size={16}
                    className="h-4 w-4 shrink-0 text-green-700"
                  />
                ) : (
                  <TablePlus className="h-4 w-4 shrink-0 text-blue-600" />
                )}
                <span className="truncate font-mono text-sm">
                  {selectedDataset?.name ?? selected}
                </span>
                {selectedDataset && (
                  <span className="bg-bg-tertiary text-fg-tertiary ml-auto shrink-0 rounded px-1.5 py-0.5 text-xs">
                    {selectedDataset.count.toLocaleString()}
                  </span>
                )}
              </div>
            ) : (
              <span className="flex flex-row items-center gap-2">
                <ButtonIcon as={Table} variant="tertiary" />
                <span className="text-fg-primary flex text-sm font-medium">
                  {computedPlaceholder}
                </span>
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
            <CommandInput
              placeholder={computedPlaceholder}
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
                <ComboboxMenuItems
                  items={sortedDatasetNames}
                  selected={selected}
                  searchValue={inputValue}
                  onSelectItem={handleSelectItem}
                  showCreateOption={showCreateOption}
                  createHeading="New dataset"
                  existingHeading="Existing"
                  getItemIcon={getItemIcon}
                  getItemSuffix={getItemSuffix}
                  getItemDataAttributes={getItemDataAttributes}
                />
              </CommandList>
            )}
          </Command>
        </PopoverContent>
      </Popover>
    </div>
  );
}
