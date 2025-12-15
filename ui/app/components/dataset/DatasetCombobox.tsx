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

  const {
    open,
    searchValue,
    commandRef,
    getInputValue,
    closeDropdown,
    handleKeyDown,
    handleInputChange,
    handleBlur,
    handleClick,
  } = useCombobox();

  const filteredItems = useMemo(
    () => filterItems(searchValue),
    [filterItems, searchValue],
  );

  const showCreateOption = shouldShowCreateOption(searchValue);
  const showCreateHint =
    allowCreation && !showCreateOption && !isLoading && !isError;
  const showMenu = !isLoading && !isError;

  const handleSelectItem = useCallback(
    (item: string, isNew: boolean) => {
      onSelect(item, isNew);
      closeDropdown();
    },
    [onSelect, closeDropdown],
  );

  const inputPrefix = useMemo(() => {
    const item = selected && !searchValue ? selected : null;
    const isSelected = Boolean(selected && !searchValue);
    return getItemIcon(item, isSelected);
  }, [selected, searchValue, getItemIcon]);

  const inputSuffix = useMemo(() => {
    const item = selected && !searchValue ? selected : null;
    const count = getItemSuffix(item);
    if (!count) return null;
    return (
      <span className="bg-bg-tertiary text-fg-tertiary shrink-0 rounded px-1.5 py-0.5 font-mono text-xs">
        {count}
      </span>
    );
  }, [selected, searchValue, getItemSuffix]);

  return (
    <Popover open={open}>
      <PopoverAnchor asChild>
        <ComboboxInput
          value={getInputValue(selected)}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          onClick={handleClick}
          onBlur={handleBlur}
          placeholder={computedPlaceholder}
          disabled={disabled}
          open={open}
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
        <Command ref={commandRef} shouldFilter={false}>
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
