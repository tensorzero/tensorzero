import { CommandGroup, CommandItem } from "~/components/ui/command";

type ComboboxMenuItemsProps = {
  items: string[];
  selected: string | null | undefined;
  searchValue: string;
  onSelectItem: (item: string, isNew: boolean) => void;
  showCreateOption: boolean;
  createHeading?: string;
  existingHeading?: string;
  getItemIcon?: (item: string | null, isSelected: boolean) => React.ReactNode;
  getItemSuffix?: (item: string | null) => React.ReactNode;
  getItemDataAttributes?: (item: string) => Record<string, string>;
};

export function ComboboxMenuItems({
  items,
  selected,
  searchValue,
  onSelectItem,
  showCreateOption,
  createHeading = "Create new",
  existingHeading,
  getItemIcon,
  getItemSuffix,
  getItemDataAttributes,
}: ComboboxMenuItemsProps) {
  return (
    <>
      {showCreateOption && (
        <CommandGroup heading={createHeading}>
          <CommandItem
            value={`create-${searchValue.trim()}`}
            onSelect={() => onSelectItem(searchValue.trim(), true)}
            className="flex items-center gap-2"
          >
            {getItemIcon?.(null, false)}
            <span className="truncate font-mono">{searchValue.trim()}</span>
          </CommandItem>
        </CommandGroup>
      )}
      {items.length > 0 && (
        <CommandGroup heading={showCreateOption ? existingHeading : undefined}>
          {items.map((item) => {
            const isSelected = selected === item;
            return (
              <CommandItem
                key={item}
                value={item}
                onSelect={() => onSelectItem(item, false)}
                className="group flex w-full items-center gap-2"
                {...getItemDataAttributes?.(item)}
              >
                <div className="flex min-w-0 flex-1 items-center gap-2">
                  {getItemIcon?.(item, isSelected)}
                  <span className="truncate font-mono">{item}</span>
                </div>
                {getItemSuffix && (
                  <span className="text-fg-tertiary min-w-8 flex-shrink-0 text-right text-sm whitespace-nowrap">
                    {getItemSuffix(item)}
                  </span>
                )}
              </CommandItem>
            );
          })}
        </CommandGroup>
      )}
    </>
  );
}
