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
import clsx from "clsx";
import { useCallback, useMemo } from "react";
import type { IconProps } from "~/components/icons/Icons";
import { ComboboxInput } from "./ComboboxInput";
import { useCombobox } from "./use-combobox";

type IconComponent = React.FC<IconProps>;

type ComboboxProps = {
  selected: string | null;
  onSelect: (value: string) => void;
  items: string[];
  icon: IconComponent;
  placeholder: string;
  emptyMessage: string;
  disabled?: boolean;
  monospace?: boolean;
  name?: string;
  /** Show clear button and call onClear when clicked */
  clearable?: boolean;
  onClear?: () => void;
  /** Render custom suffix for each item in dropdown */
  getItemSuffix?: (item: string) => React.ReactNode;
};

export function Combobox({
  selected,
  onSelect,
  items,
  icon: Icon,
  placeholder,
  emptyMessage,
  disabled = false,
  monospace = true,
  name,
  clearable = false,
  onClear,
  getItemSuffix,
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

  const handleClear = useCallback(() => {
    onClear?.();
  }, [onClear]);

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
            monospace={monospace}
            icon={Icon}
            clearable={clearable && !!selected}
            onClear={handleClear}
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
              {filteredItems.length > 0 && (
                <CommandGroup>
                  {filteredItems.map((item) => (
                    <CommandItem
                      key={item}
                      value={item}
                      onSelect={() => handleSelect(item)}
                      className="flex items-center gap-2"
                    >
                      <Icon className="h-4 w-4 shrink-0" />
                      <span
                        className={clsx("truncate", monospace && "font-mono")}
                      >
                        {item}
                      </span>
                      {getItemSuffix?.(item)}
                    </CommandItem>
                  ))}
                </CommandGroup>
              )}
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>
    </div>
  );
}
