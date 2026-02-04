import { useCallback, useMemo } from "react";
import { CommandGroup, CommandItem } from "~/components/ui/command";
import { normalizeItem, type ComboboxItem } from "./Combobox";
import { VirtualizedCommandItems } from "~/components/ui/virtualized-command-items";

type ComboboxMenuItemsProps = {
  items: ComboboxItem[];
  selectedValue?: string | null | undefined;
  searchValue: string;
  onSelectItem: (value: string, isNew: boolean) => void;
  showCreateOption: boolean;
  createHeading?: string;
  existingHeading?: string;
  getPrefix?: (value: string | null, isSelected: boolean) => React.ReactNode;
  getSuffix?: (value: string | null) => React.ReactNode;
  getItemDataAttributes?: (value: string) => Record<string, string>;
  /** Enable virtualization for large lists */
  virtualize?: boolean;
  /** Currently highlighted index for virtualized keyboard navigation */
  highlightedIndex?: number;
};

export function ComboboxMenuItems({
  items,
  selectedValue,
  searchValue,
  onSelectItem,
  showCreateOption,
  createHeading = "Create new",
  existingHeading,
  getPrefix,
  getSuffix,
  getItemDataAttributes,
  virtualize = false,
  highlightedIndex,
}: ComboboxMenuItemsProps) {
  // Normalize items to { value, label } format
  const normalizedItems = useMemo(() => items.map(normalizeItem), [items]);

  const renderItem = useCallback(
    (item: { value: string; label: string }, index: number) => {
      const isSelected = selectedValue === item.value;
      const isHighlighted = virtualize && index === highlightedIndex;
      return (
        <CommandItem
          key={item.value}
          value={item.value}
          onSelect={() => onSelectItem(item.value, false)}
          className="group flex w-full cursor-pointer items-center gap-2"
          // In virtualized mode, manually set data-selected for highlight styling
          data-selected={isHighlighted || undefined}
          aria-selected={isHighlighted || isSelected}
          {...getItemDataAttributes?.(item.value)}
        >
          <div className="flex min-w-0 flex-1 items-center gap-2">
            {getPrefix?.(item.value, isSelected)}
            <span className="truncate font-mono" title={item.label}>
              {item.label}
            </span>
          </div>
          {getSuffix?.(item.value)}
        </CommandItem>
      );
    },
    [
      selectedValue,
      virtualize,
      highlightedIndex,
      onSelectItem,
      getItemDataAttributes,
      getPrefix,
      getSuffix,
    ],
  );

  return (
    <>
      {showCreateOption && (
        <CommandGroup heading={createHeading}>
          <CommandItem
            value={`create-${searchValue.trim()}`}
            onSelect={() => onSelectItem(searchValue.trim(), true)}
            className="flex items-center gap-2"
          >
            {getPrefix?.(null, false)}
            <span className="truncate font-mono" title={searchValue.trim()}>
              {searchValue.trim()}
            </span>
          </CommandItem>
        </CommandGroup>
      )}
      {normalizedItems.length > 0 && (
        <CommandGroup heading={showCreateOption ? existingHeading : undefined}>
          {virtualize ? (
            <VirtualizedCommandItems
              items={normalizedItems}
              renderItem={renderItem}
              highlightedIndex={highlightedIndex}
            />
          ) : (
            normalizedItems.map((item, index) => renderItem(item, index))
          )}
        </CommandGroup>
      )}
    </>
  );
}
