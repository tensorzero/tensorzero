import { useCallback, useMemo } from "react";
import {
  Popover,
  PopoverAnchor,
  PopoverContent,
} from "~/components/ui/popover";
import { Command, CommandEmpty, CommandList } from "~/components/ui/command";
import { ComboboxMenuItems } from "~/components/ui/combobox";
import { ComboboxInput } from "~/components/ui/combobox/ComboboxInput";
import { ComboboxHint } from "~/components/ui/combobox/ComboboxHint";
import { useCombobox } from "~/components/ui/combobox/use-combobox";
import {
  useDatasetOptions,
  getDatasetItemDataAttributes,
} from "./use-dataset-options";

interface DatasetComboboxProps {
  selected: string | null;
  onSelect: (dataset: string, isNew: boolean) => void;
  functionName?: string;
  placeholder?: string;
  allowCreation?: boolean;
  disabled?: boolean;
}

export function DatasetCombobox({
  selected,
  onSelect,
  functionName,
  placeholder,
  allowCreation = false,
  disabled = false,
}: DatasetComboboxProps) {
  const {
    isLoading,
    isError,
    computedPlaceholder,
    getItemIcon,
    getItemSuffix,
    filterItems,
    shouldShowCreateOption,
  } = useDatasetOptions({ functionName, placeholder, allowCreation });

  const combobox = useCombobox();

  const filteredItems = useMemo(
    () => filterItems(combobox.searchValue),
    [filterItems, combobox.searchValue],
  );

  const showCreateOption = shouldShowCreateOption(combobox.searchValue);
  const showCreateHint =
    allowCreation && !showCreateOption && !isLoading && !isError;
  const showMenu = !isLoading && !isError;

  const handleSelectItem = useCallback(
    (item: string, isNew: boolean) => {
      onSelect(item, isNew);
      combobox.closeDropdown();
    },
    [onSelect, combobox],
  );

  const inputPrefix = useMemo(() => {
    const item = selected && !combobox.searchValue ? selected : null;
    const isSelected = Boolean(selected && !combobox.searchValue);
    return getItemIcon(item, isSelected);
  }, [selected, combobox.searchValue, getItemIcon]);

  const inputSuffix = useMemo(() => {
    const item = selected && !combobox.searchValue ? selected : null;
    const count = getItemSuffix(item);
    if (!count) return null;
    return (
      <span className="bg-bg-tertiary text-fg-tertiary shrink-0 rounded px-1.5 py-0.5 font-mono text-xs">
        {count}
      </span>
    );
  }, [selected, combobox.searchValue, getItemSuffix]);

  return (
    <Popover open={combobox.open}>
      <PopoverAnchor asChild>
        <ComboboxInput
          value={combobox.getInputValue(selected)}
          onChange={combobox.handleInputChange}
          onKeyDown={combobox.handleKeyDown}
          onClick={combobox.handleClick}
          onBlur={combobox.handleBlur}
          placeholder={computedPlaceholder}
          disabled={disabled}
          open={combobox.open}
          prefix={inputPrefix}
          suffix={inputSuffix}
        />
      </PopoverAnchor>
      <PopoverContent
        className="w-[var(--radix-popover-trigger-width)] p-0"
        align="start"
        onOpenAutoFocus={(e) => e.preventDefault()}
        onPointerDownOutside={(e) => e.preventDefault()}
        onInteractOutside={(e) => e.preventDefault()}
      >
        <Command ref={combobox.commandRef} shouldFilter={false}>
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
                searchValue={combobox.searchValue}
                onSelectItem={handleSelectItem}
                showCreateOption={showCreateOption}
                createHeading="New dataset"
                existingHeading="Existing"
                getItemIcon={getItemIcon}
                getItemSuffix={getItemSuffix}
                getItemDataAttributes={getDatasetItemDataAttributes}
              />
            </CommandList>
          )}
        </Command>

        {showCreateHint && (
          <ComboboxHint>Type to create a new dataset</ComboboxHint>
        )}
      </PopoverContent>
    </Popover>
  );
}
