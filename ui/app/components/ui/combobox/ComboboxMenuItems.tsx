import { CommandGroup, CommandItem } from "~/components/ui/command";

type ComboboxMenuItemsProps = {
  items: string[];
  selected: string | null | undefined;
  searchValue: string;
  onSelectItem: (item: string, isNew: boolean) => void;
  showCreateOption: boolean;
  createHeading?: string;
  existingHeading?: string;
  getPrefix?: (item: string | null, isSelected: boolean) => React.ReactNode;
  getSuffix?: (item: string | null) => React.ReactNode;
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
  getPrefix,
  getSuffix,
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
            {getPrefix?.(null, false)}
            <span className="truncate font-mono" title={searchValue.trim()}>
              {searchValue.trim()}
            </span>
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
                  {getPrefix?.(item, isSelected)}
                  <span className="truncate font-mono" title={item}>
                    {item}
                  </span>
                </div>
                {getSuffix?.(item)}
              </CommandItem>
            );
          })}
        </CommandGroup>
      )}
    </>
  );
}
