import {
  Popover,
  PopoverAnchor,
  PopoverContent,
} from "~/components/ui/popover";
import { useCallback, useEffect, useMemo } from "react";
import { ComboboxInput } from "./ComboboxInput";
import { ComboboxContent } from "./ComboboxContent";
import { ComboboxHint } from "./ComboboxHint";
import { ComboboxMenuItems } from "./ComboboxMenuItems";
import { useCombobox } from "./use-combobox";
import {
  DEFAULT_VIRTUALIZE_THRESHOLD,
  useVirtualizedNavigation,
} from "~/components/ui/use-virtualized-navigation";

export type ComboboxItem = string | { value: string; label: string };

export type NormalizedComboboxItem = { value: string; label: string };

export function normalizeItem(item: ComboboxItem): NormalizedComboboxItem {
  if (typeof item === "string") {
    return { value: item, label: item };
  }
  return item;
}

type ComboboxProps = {
  selected: string | null;
  onSelect: (value: string, isNew: boolean) => void;
  items: ComboboxItem[];
  getPrefix?: (value: string | null, isSelected: boolean) => React.ReactNode;
  getSuffix?: (value: string | null) => React.ReactNode;
  getItemDataAttributes?: (value: string) => Record<string, string>;
  placeholder: string;
  emptyMessage: string;
  disabled?: boolean;
  name?: string;
  ariaLabel?: string;
  allowCreation?: boolean;
  createHint?: string;
  createHeading?: string;
  loading?: boolean;
  loadingMessage?: string;
  error?: boolean;
  errorMessage?: string;
  /**
   * Number of items at which virtualization is enabled.
   * Set to 0 to always virtualize, or Infinity to never virtualize.
   * Default: 100
   */
  virtualizeThreshold?: number;
};

export function Combobox({
  selected,
  onSelect,
  items,
  getPrefix,
  getSuffix,
  getItemDataAttributes,
  placeholder,
  emptyMessage,
  disabled = false,
  name,
  ariaLabel,
  allowCreation = false,
  createHint,
  createHeading = "Create new",
  loading = false,
  loadingMessage = "Loading...",
  error = false,
  errorMessage = "An error occurred.",
  virtualizeThreshold = DEFAULT_VIRTUALIZE_THRESHOLD,
}: ComboboxProps) {
  const {
    open,
    searchValue,
    isEditing,
    commandRef,
    getInputValue,
    closeDropdown,
    handleKeyDown: baseHandleKeyDown,
    handleInputChange,
    handleBlur,
    handleClick,
  } = useCombobox();

  // Normalize items to { value, label } format
  const normalizedItems = useMemo(() => items.map(normalizeItem), [items]);

  // Find the label for the currently selected value
  // Fall back to selected value itself for created items not in list
  const selectedLabel = useMemo(() => {
    if (!selected) return null;
    const item = normalizedItems.find((item) => item.value === selected);
    return item?.label ?? selected;
  }, [selected, normalizedItems]);

  const filteredItems = useMemo(() => {
    const query = searchValue.toLowerCase();
    if (!query) return normalizedItems;
    return normalizedItems.filter((item) =>
      item.label.toLowerCase().includes(query),
    );
  }, [normalizedItems, searchValue]);

  const shouldVirtualize = filteredItems.length >= virtualizeThreshold;

  const showCreateOption =
    allowCreation &&
    Boolean(searchValue.trim()) &&
    !normalizedItems.some(
      (item) => item.label.toLowerCase() === searchValue.trim().toLowerCase(),
    );

  const handleSelectItem = useCallback(
    (value: string, isNew: boolean) => {
      onSelect(value, isNew);
      closeDropdown();
    },
    [onSelect, closeDropdown],
  );

  // Virtualized navigation hook
  const {
    highlightedIndex,
    handleKeyDown: virtualizedHandleKeyDown,
    resetHighlight,
  } = useVirtualizedNavigation({
    itemCount: filteredItems.length,
    enabled: shouldVirtualize,
    onSelect: (index) => {
      const item = filteredItems[index];
      if (item) {
        handleSelectItem(item.value, false);
      }
    },
    onClose: closeDropdown,
    hasCreateOption: showCreateOption,
    onSelectCreate: () => {
      handleSelectItem(searchValue.trim(), true);
    },
  });

  // Reset highlight when dropdown opens or search changes
  useEffect(() => {
    if (open) {
      resetHighlight();
    }
  }, [open, searchValue, resetHighlight]);

  // Combined keyboard handler
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (shouldVirtualize && open) {
        virtualizedHandleKeyDown(e);
      } else {
        baseHandleKeyDown(e);
      }
    },
    [shouldVirtualize, virtualizedHandleKeyDown, baseHandleKeyDown, open],
  );

  const inputPrefix = useMemo(() => {
    const value = selected && !searchValue ? selected : null;
    const isSelected = Boolean(selected && !searchValue);
    return getPrefix?.(value, isSelected);
  }, [selected, searchValue, getPrefix]);

  const inputSuffix = useMemo(() => {
    const value = selected && !searchValue ? selected : null;
    return getSuffix?.(value);
  }, [selected, searchValue, getSuffix]);

  return (
    <div className="w-full">
      {name && <input type="hidden" name={name} value={selected ?? ""} />}
      <Popover open={open}>
        <PopoverAnchor asChild>
          <ComboboxInput
            value={getInputValue(selectedLabel)}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            onClick={handleClick}
            onBlur={handleBlur}
            placeholder={placeholder}
            disabled={disabled}
            open={open}
            isEditing={isEditing}
            prefix={inputPrefix}
            suffix={inputSuffix}
            ariaLabel={ariaLabel}
          />
        </PopoverAnchor>
        <PopoverContent
          className="w-[var(--radix-popover-trigger-width)] p-0"
          align="start"
          onOpenAutoFocus={(e) => e.preventDefault()}
          onPointerDownOutside={(e) => e.preventDefault()}
          onInteractOutside={(e) => e.preventDefault()}
        >
          <ComboboxContent
            ref={commandRef}
            emptyMessage={emptyMessage}
            showEmpty={!showCreateOption && !loading && !error}
          >
            {loading ? (
              <div className="text-fg-muted flex items-center justify-center py-4 text-sm">
                {loadingMessage}
              </div>
            ) : error ? (
              <div className="text-fg-muted flex items-center justify-center py-4 text-sm">
                {errorMessage}
              </div>
            ) : (
              <ComboboxMenuItems
                items={filteredItems}
                selectedValue={selected}
                searchValue={searchValue}
                onSelectItem={handleSelectItem}
                showCreateOption={showCreateOption}
                createHeading={createHeading}
                existingHeading="Existing"
                getPrefix={getPrefix}
                getSuffix={getSuffix}
                getItemDataAttributes={getItemDataAttributes}
                virtualize={shouldVirtualize}
                highlightedIndex={highlightedIndex}
              />
            )}
          </ComboboxContent>
          {createHint && !showCreateOption && !loading && !error && (
            <ComboboxHint>{createHint}</ComboboxHint>
          )}
        </PopoverContent>
      </Popover>
    </div>
  );
}
