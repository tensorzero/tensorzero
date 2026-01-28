import {
  Popover,
  PopoverAnchor,
  PopoverContent,
} from "~/components/ui/popover";
import { useCallback, useEffect, useMemo, useState } from "react";
import { ComboboxInput } from "./ComboboxInput";
import { ComboboxContent } from "./ComboboxContent";
import { ComboboxHint } from "./ComboboxHint";
import { ComboboxMenuItems } from "./ComboboxMenuItems";
import { useCombobox } from "./use-combobox";

/** Default threshold for enabling virtualization */
const DEFAULT_VIRTUALIZE_THRESHOLD = 100;

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
    commandRef,
    getInputValue,
    closeDropdown,
    handleKeyDown: baseHandleKeyDown,
    handleInputChange,
    handleBlur,
    handleClick,
  } = useCombobox();

  // Track highlighted index for virtualized keyboard navigation
  const [highlightedIndex, setHighlightedIndex] = useState(0);

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

  // Reset highlighted index when dropdown opens or filtered items change
  useEffect(() => {
    if (open) {
      setHighlightedIndex((prev) => {
        if (filteredItems.length === 0) return 0;
        // Clamp to valid range if items were filtered
        return Math.min(prev, filteredItems.length - 1);
      });
    }
  }, [open, filteredItems.length]);

  // Reset to first item when search changes
  useEffect(() => {
    setHighlightedIndex(0);
  }, [searchValue]);

  const handleSelectItem = useCallback(
    (value: string, isNew: boolean) => {
      onSelect(value, isNew);
      closeDropdown();
    },
    [onSelect, closeDropdown],
  );

  // Custom keyboard handler for virtualized mode
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (!shouldVirtualize) {
        // Non-virtualized: delegate to cmdk
        baseHandleKeyDown(e);
        return;
      }

      // Virtualized mode: handle navigation ourselves
      if (e.key === "Escape") {
        closeDropdown();
        return;
      }

      if (e.key === "ArrowDown") {
        e.preventDefault();
        setHighlightedIndex((prev) =>
          prev < filteredItems.length - 1 ? prev + 1 : prev,
        );
        return;
      }

      if (e.key === "ArrowUp") {
        e.preventDefault();
        setHighlightedIndex((prev) => (prev > 0 ? prev - 1 : 0));
        return;
      }

      if (e.key === "Home") {
        e.preventDefault();
        setHighlightedIndex(0);
        return;
      }

      if (e.key === "End") {
        e.preventDefault();
        setHighlightedIndex(Math.max(0, filteredItems.length - 1));
        return;
      }

      if (e.key === "PageDown") {
        e.preventDefault();
        // Jump ~8 items (one viewport)
        setHighlightedIndex((prev) =>
          Math.min(prev + 8, filteredItems.length - 1),
        );
        return;
      }

      if (e.key === "PageUp") {
        e.preventDefault();
        setHighlightedIndex((prev) => Math.max(prev - 8, 0));
        return;
      }

      if (e.key === "Enter") {
        e.preventDefault();
        const item = filteredItems[highlightedIndex];
        if (item) {
          handleSelectItem(item.value, false);
        }
        return;
      }

      // For other keys, use base handler
      baseHandleKeyDown(e);
    },
    [
      shouldVirtualize,
      baseHandleKeyDown,
      closeDropdown,
      filteredItems,
      highlightedIndex,
      handleSelectItem,
    ],
  );

  const showCreateOption =
    allowCreation &&
    Boolean(searchValue.trim()) &&
    !normalizedItems.some(
      (item) => item.label.toLowerCase() === searchValue.trim().toLowerCase(),
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
