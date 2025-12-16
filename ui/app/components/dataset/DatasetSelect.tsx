import { useCallback, useMemo, useState } from "react";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "~/components/ui/popover";
import { Button, ButtonIcon } from "~/components/ui/button";
import { ChevronDown } from "lucide-react";
import { Table, TablePlus } from "~/components/icons/Icons";
import {
  Command,
  CommandEmpty,
  CommandInput,
  CommandList,
} from "~/components/ui/command";
import clsx from "clsx";
import { ComboboxMenuItems } from "~/components/ui/combobox/ComboboxMenuItems";
import {
  useDatasetOptions,
  getDatasetItemDataAttributes,
} from "./use-dataset-options";

interface DatasetSelectProps {
  selected: string | null;
  onSelect: (dataset: string, isNew: boolean) => void;
  functionName?: string;
  placeholder?: string;
  allowCreation?: boolean;
  disabled?: boolean;
}

export function DatasetSelect({
  selected,
  onSelect,
  functionName,
  placeholder,
  allowCreation = false,
  disabled = false,
}: DatasetSelectProps) {
  const {
    isLoading,
    isError,
    computedPlaceholder,
    searchPlaceholder,
    getPrefix,
    getSuffix,
    getSelectedDataset,
    filterItems,
    shouldShowCreateOption,
  } = useDatasetOptions({ functionName, placeholder, allowCreation });

  const [open, setOpen] = useState(false);
  const [searchValue, setSearchValue] = useState("");

  const selectedDataset = getSelectedDataset(selected);

  const filteredItems = useMemo(
    () => filterItems(searchValue),
    [filterItems, searchValue],
  );

  const showCreateOption = shouldShowCreateOption(searchValue);
  const showMenu = !isLoading && !isError;

  const handleSelectItem = useCallback(
    (item: string, isNew: boolean) => {
      onSelect(item, isNew);
      setSearchValue("");
      setOpen(false);
    },
    [onSelect],
  );

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild disabled={disabled}>
        <Button
          variant="outline"
          size="sm"
          role="combobox"
          aria-expanded={open}
          className="group justify-between border"
          disabled={disabled}
        >
          {selected ? (
            <div className="flex w-full min-w-0 flex-1 items-center gap-x-2">
              {selectedDataset ? (
                <Table size={16} className="shrink-0 text-green-700" />
              ) : (
                <TablePlus size={16} className="shrink-0 text-blue-600" />
              )}
              <span className="truncate font-mono text-sm">
                {selectedDataset?.name ?? selected}
              </span>
              <div className="ml-auto">{getSuffix(selected)}</div>
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
        <Command shouldFilter={false}>
          <CommandInput
            placeholder={searchPlaceholder}
            value={searchValue}
            onValueChange={setSearchValue}
            className="h-9"
          />

          {isLoading && (
            <div className="text-fg-muted flex items-center justify-center py-4 text-sm">
              Loading datasets...
            </div>
          )}

          {isError && (
            <div className="text-fg-muted flex items-center justify-center py-4 text-sm">
              There was an error loading datasets.
            </div>
          )}

          {showMenu && (
            <CommandList>
              <CommandEmpty>No datasets found</CommandEmpty>
              <ComboboxMenuItems
                items={filteredItems}
                selected={selected}
                searchValue={searchValue}
                onSelectItem={handleSelectItem}
                showCreateOption={showCreateOption}
                createHeading="New dataset"
                existingHeading="Existing"
                getPrefix={getPrefix}
                getSuffix={getSuffix}
                getItemDataAttributes={getDatasetItemDataAttributes}
              />
            </CommandList>
          )}
        </Command>
      </PopoverContent>
    </Popover>
  );
}
