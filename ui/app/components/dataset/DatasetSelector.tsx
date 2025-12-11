import { useCallback, useMemo } from "react";
import {
  Popover,
  PopoverAnchor,
  PopoverContent,
} from "~/components/ui/popover";
import { Table, TablePlus, TableCheck } from "~/components/icons/Icons";
import {
  CommandGroup,
  CommandItem,
  CommandEmpty,
} from "~/components/ui/command";
import clsx from "clsx";
import { useDatasetCounts } from "~/hooks/use-dataset-counts";
import {
  useCombobox,
  ComboboxInput,
  ComboboxContent,
  ComboboxHint,
} from "~/components/ui/combobox";

interface DatasetSelectorProps {
  selected?: string;
  onSelect: (dataset: string, isNew: boolean) => void;
  functionName?: string;
  placeholder?: string;
  className?: string;
  allowCreation?: boolean;
  disabled?: boolean;
}

export function DatasetSelector({
  selected,
  onSelect,
  functionName,
  placeholder,
  allowCreation = true,
  className,
  disabled = false,
}: DatasetSelectorProps) {
  const {
    open,
    searchValue,
    commandRef,
    inputValue,
    closeDropdown,
    handleKeyDown,
    handleInputChange,
    handleBlur,
    handleClick,
  } = useCombobox();

  const {
    data: datasets = [],
    isLoading,
    isError,
  } = useDatasetCounts(functionName);

  const recentlyUpdatedDatasets = useMemo(
    () =>
      [...(datasets ?? [])].sort((a, b) => {
        return (
          new Date(b.lastUpdated).getTime() - new Date(a.lastUpdated).getTime()
        );
      }),
    [datasets],
  );

  const filteredDatasets = useMemo(() => {
    const query = searchValue.toLowerCase();
    if (!query) return recentlyUpdatedDatasets;
    return recentlyUpdatedDatasets.filter((dataset) =>
      dataset.name.toLowerCase().includes(query),
    );
  }, [recentlyUpdatedDatasets, searchValue]);

  const existingSelectedDataset = useMemo(
    () => datasets?.find((dataset) => dataset.name === selected),
    [datasets, selected],
  );

  const computedPlaceholder = useMemo(() => {
    if (placeholder) return placeholder;
    if (allowCreation) {
      return recentlyUpdatedDatasets.length > 0
        ? "Create or find dataset"
        : "Create dataset";
    }
    return "Select dataset";
  }, [placeholder, allowCreation, recentlyUpdatedDatasets.length]);

  const getIcon = useCallback(() => {
    if (selected && existingSelectedDataset) {
      return Table;
    } else if (selected) {
      return TablePlus;
    }
    return Table;
  }, [selected, existingSelectedDataset]);

  const getIconClassName = useCallback(() => {
    if (selected && existingSelectedDataset) {
      return "text-green-700";
    } else if (selected) {
      return "text-blue-600";
    }
    return undefined;
  }, [selected, existingSelectedDataset]);

  const handleSelectDataset = useCallback(
    (datasetName: string, isNew: boolean) => {
      onSelect(datasetName, isNew);
      closeDropdown();
    },
    [onSelect, closeDropdown],
  );

  const showCreateOption =
    allowCreation &&
    searchValue.trim() &&
    !recentlyUpdatedDatasets.some(
      (dataset) =>
        dataset.name.toLowerCase() === searchValue.trim().toLowerCase(),
    );

  return (
    <div className={className}>
      <Popover open={open}>
        <PopoverAnchor asChild>
          <ComboboxInput
            value={inputValue(selected)}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            onFocus={() => {}}
            onClick={handleClick}
            onBlur={handleBlur}
            placeholder={computedPlaceholder}
            disabled={disabled}
            monospace
            open={open}
            icon={getIcon()}
            iconClassName={getIconClassName()}
          />
        </PopoverAnchor>

        <PopoverContent
          className="w-[var(--radix-popover-trigger-width)] min-w-64 p-0"
          align="start"
          onOpenAutoFocus={(e) => e.preventDefault()}
          onPointerDownOutside={(e) => e.preventDefault()}
          onInteractOutside={(e) => e.preventDefault()}
        >
          <ComboboxContent ref={commandRef} showEmpty={false}>
            {isLoading ? (
              <div className="text-fg-muted flex items-center justify-center py-4 text-sm">
                Loading datasets...
              </div>
            ) : isError ? (
              <div className="text-fg-muted flex items-center justify-center py-4 text-sm">
                There was an error loading datasets.
              </div>
            ) : (
              <>
                {showCreateOption && (
                  <CommandGroup heading="Create dataset">
                    <CommandItem
                      value={`create-${searchValue.trim()}`}
                      onSelect={() =>
                        handleSelectDataset(searchValue.trim(), true)
                      }
                      className="flex items-center justify-between"
                    >
                      <div className="flex items-center truncate">
                        <TablePlus className="mr-2 h-4 w-4 text-blue-600" />
                        <span className="text-fg-primary truncate font-mono text-sm">
                          {searchValue.trim()}
                        </span>
                      </div>
                    </CommandItem>
                  </CommandGroup>
                )}

                {filteredDatasets.length > 0 && (
                  <CommandGroup
                    heading={showCreateOption ? "Existing datasets" : undefined}
                  >
                    {filteredDatasets.map((dataset) => (
                      <CommandItem
                        key={dataset.name}
                        value={dataset.name}
                        onSelect={() =>
                          handleSelectDataset(dataset.name, false)
                        }
                        className="group flex w-full items-center gap-2"
                        data-dataset-name={dataset.name}
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
                            "min-w-8 flex-shrink-0 text-right text-sm whitespace-nowrap",
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

                {!showCreateOption && filteredDatasets.length === 0 && (
                  <CommandEmpty className="flex items-center justify-center px-4 py-4 text-sm">
                    No datasets found.
                  </CommandEmpty>
                )}
              </>
            )}
          </ComboboxContent>
          {allowCreation && !showCreateOption && !isLoading && !isError && (
            <ComboboxHint>Type to create a new dataset</ComboboxHint>
          )}
        </PopoverContent>
      </Popover>
    </div>
  );
}
