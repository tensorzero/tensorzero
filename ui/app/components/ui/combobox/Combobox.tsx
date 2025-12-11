import {
  Popover,
  PopoverAnchor,
  PopoverContent,
} from "~/components/ui/popover";
import { CommandGroup, CommandItem } from "~/components/ui/command";
import clsx from "clsx";
import { useCallback, useMemo } from "react";
import type { IconProps } from "~/components/icons/Icons";
import { ComboboxInput } from "./ComboboxInput";
import { ComboboxContent } from "./ComboboxContent";
import { useCombobox } from "./useCombobox";

type IconComponent = React.FC<IconProps>;

interface ComboboxProps {
  selected: string | null;
  onSelect: (value: string) => void;
  items: string[];
  icon: IconComponent;
  placeholder: string;
  emptyMessage: string;
  disabled?: boolean;
  monospace?: boolean;
}

export function Combobox({
  selected,
  onSelect,
  items,
  icon: Icon,
  placeholder,
  emptyMessage,
  disabled = false,
  monospace = false,
}: ComboboxProps) {
  const {
    open,
    searchValue,
    commandRef,
    inputValue,
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

  return (
    <div className="w-full">
      <Popover open={open}>
        <PopoverAnchor asChild>
          <ComboboxInput
            value={inputValue(selected)}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            onFocus={() => {}}
            onClick={handleClick}
            onBlur={handleBlur}
            placeholder={placeholder}
            disabled={disabled}
            monospace={monospace}
            open={open}
            icon={Icon}
          />
        </PopoverAnchor>
        <PopoverContent
          className="w-[var(--radix-popover-trigger-width)] p-0"
          align="start"
          onOpenAutoFocus={(e) => e.preventDefault()}
          onPointerDownOutside={(e) => e.preventDefault()}
          onInteractOutside={(e) => e.preventDefault()}
        >
          <ComboboxContent ref={commandRef} emptyMessage={emptyMessage}>
            <CommandGroup>
              {filteredItems.map((item) => (
                <CommandItem
                  key={item}
                  value={item}
                  onSelect={() => handleSelect(item)}
                  className="flex items-center gap-2"
                >
                  <Icon className="h-4 w-4 shrink-0" />
                  <span className={clsx("truncate", monospace && "font-mono")}>
                    {item}
                  </span>
                </CommandItem>
              ))}
            </CommandGroup>
          </ComboboxContent>
        </PopoverContent>
      </Popover>
    </div>
  );
}
