import {
  Popover,
  PopoverAnchor,
  PopoverContent,
} from "~/components/ui/popover";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandItem,
  CommandList,
} from "~/components/ui/command";
import { useCallback, useMemo, useState, useEffect } from "react";
import { ComboboxInput } from "./ComboboxInput";
import { useCombobox } from "./use-combobox";
import { VirtualizedCommandItems } from "~/components/ui/virtualized-command-list";
import { cn } from "~/utils/common";

/** Default threshold for enabling virtualization */
const DEFAULT_VIRTUALIZE_THRESHOLD = 100;

type ComboboxProps = {
  selected: string | null;
  onSelect: (value: string) => void;
  items: string[];
  getItemIcon?: (item: string | null) => React.ReactNode;
  placeholder: string;
  emptyMessage: string;
  disabled?: boolean;
  name?: string;
  ariaLabel?: string;
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
  getItemIcon,
  placeholder,
  emptyMessage,
  disabled = false,
  name,
  ariaLabel,
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

  const filteredItems = useMemo(() => {
    const query = searchValue.toLowerCase();
    if (!query) return items;
    return items.filter((item) => item.toLowerCase().includes(query));
  }, [items, searchValue]);

  const shouldVirtualize = filteredItems.length >= virtualizeThreshold;

  // Reset highlighted index when filtered items change
  useEffect(() => {
    setHighlightedIndex(0);
  }, [filteredItems.length, searchValue]);

  const handleSelect = useCallback(
    (item: string) => {
      onSelect(item);
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

      if (e.key === "Enter") {
        e.preventDefault();
        const item = filteredItems[highlightedIndex];
        if (item) {
          handleSelect(item);
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
      handleSelect,
    ],
  );

  const inputPrefix = useMemo(() => {
    const item = selected && !searchValue ? selected : null;
    return getItemIcon?.(item);
  }, [selected, searchValue, getItemIcon]);

  const renderItem = useCallback(
    (item: string, index: number) => {
      const isHighlighted = shouldVirtualize && index === highlightedIndex;
      return (
        <CommandItem
          key={item}
          value={item}
          onSelect={() => handleSelect(item)}
          className={cn(
            "flex items-center gap-2",
            isHighlighted && "bg-accent text-accent-foreground",
          )}
        >
          {getItemIcon?.(item)}
          <span className="truncate font-mono">{item}</span>
        </CommandItem>
      );
    },
    [getItemIcon, handleSelect, shouldVirtualize, highlightedIndex],
  );

  return (
    <div className="w-full">
      {name && <input type="hidden" name={name} value={selected ?? ""} />}
      <Popover open={open}>
        <PopoverAnchor asChild>
          <ComboboxInput
            value={getInputValue(selected)}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            onClick={handleClick}
            onBlur={handleBlur}
            placeholder={placeholder}
            disabled={disabled}
            open={open}
            prefix={inputPrefix}
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
          <Command ref={commandRef} shouldFilter={false}>
            <CommandList>
              <CommandEmpty className="text-fg-tertiary flex items-center justify-center py-6 text-sm">
                {emptyMessage}
              </CommandEmpty>
              {filteredItems.length > 0 &&
                (shouldVirtualize ? (
                  <CommandGroup>
                    <VirtualizedCommandItems
                      items={filteredItems}
                      renderItem={renderItem}
                      highlightedIndex={highlightedIndex}
                    />
                  </CommandGroup>
                ) : (
                  <CommandGroup>
                    {filteredItems.map((item, index) =>
                      renderItem(item, index),
                    )}
                  </CommandGroup>
                ))}
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>
    </div>
  );
}
