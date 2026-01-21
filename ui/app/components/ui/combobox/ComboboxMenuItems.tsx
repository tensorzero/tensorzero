import { useMemo } from "react";
import { CommandGroup, CommandItem } from "~/components/ui/command";
import { normalizeItem, type ComboboxItem } from "./Combobox";

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
}: ComboboxMenuItemsProps) {
  // Normalize items to { value, label } format
  const normalizedItems = useMemo(() => items.map(normalizeItem), [items]);

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
          {normalizedItems.map((item) => {
            const isSelected = selectedValue === item.value;
            return (
              <CommandItem
                key={item.value}
                value={item.value}
                onSelect={() => onSelectItem(item.value, false)}
                className="group flex w-full items-center gap-2"
                {...getItemDataAttributes?.(item.value)}
              >
                <div className="flex min-w-0 flex-1 items-center gap-2">
                  {getPrefix?.(item.value, isSelected)}
                  <span className="truncate font-mono">{item.label}</span>
                </div>
                {getSuffix?.(item.value)}
              </CommandItem>
            );
          })}
        </CommandGroup>
      )}
    </>
  );
}
