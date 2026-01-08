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
import { useCallback, useMemo } from "react";
import { ComboboxInput } from "./ComboboxInput";
import { useCombobox } from "./use-combobox";
import { VirtualizedCommandItems } from "~/components/ui/virtualized-command-list";

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
    handleKeyDown,
    handleInputChange,
    handleBlur,
    handleClick,
  } = useCombobox();

  const filteredItems = useMemo(() => {
    const query = searchValue.toLowerCase();
    if (!query) return items;
    return items.filter((item) => item.toLowerCase().includes(query));
  }, [items, searchValue]);

  const handleSelect = useCallback(
    (item: string) => {
      onSelect(item);
      closeDropdown();
    },
    [onSelect, closeDropdown],
  );

  const inputPrefix = useMemo(() => {
    const item = selected && !searchValue ? selected : null;
    return getItemIcon?.(item);
  }, [selected, searchValue, getItemIcon]);

  const shouldVirtualize = filteredItems.length >= virtualizeThreshold;

  const renderItem = useCallback(
    (item: string) => (
      <CommandItem
        key={item}
        value={item}
        onSelect={() => handleSelect(item)}
        className="flex items-center gap-2"
      >
        {getItemIcon?.(item)}
        <span className="truncate font-mono">{item}</span>
      </CommandItem>
    ),
    [getItemIcon, handleSelect],
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
                    />
                  </CommandGroup>
                ) : (
                  <CommandGroup>
                    {filteredItems.map((item) => renderItem(item))}
                  </CommandGroup>
                ))}
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>
    </div>
  );
}
