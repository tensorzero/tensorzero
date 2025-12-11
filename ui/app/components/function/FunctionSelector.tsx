import { useCallback, useMemo } from "react";
import {
  Popover,
  PopoverAnchor,
  PopoverContent,
} from "~/components/ui/popover";
import { Functions } from "~/components/icons/Icons";
import { DEFAULT_FUNCTION } from "~/utils/constants";
import { CommandGroup, CommandItem } from "~/components/ui/command";
import clsx from "clsx";
import { getFunctionTypeIcon } from "~/utils/icon";
import type { FunctionConfig } from "~/types/tensorzero";
import {
  useCombobox,
  ComboboxInput,
  ComboboxContent,
} from "~/components/ui/combobox";

interface FunctionSelectorProps {
  selected: string | null;
  onSelect?: (functionName: string) => void;
  functions: { [x: string]: FunctionConfig | undefined };
  hideDefaultFunction?: boolean;
  disabled?: boolean;
}

export function FunctionTypeIcon({ type }: { type: FunctionConfig["type"] }) {
  const iconConfig = getFunctionTypeIcon(type);
  return (
    <div className={`${iconConfig.iconBg} rounded-sm p-0.5`}>
      {iconConfig.icon}
    </div>
  );
}

export function FunctionSelector({
  selected,
  onSelect,
  functions,
  hideDefaultFunction = false,
  disabled = false,
}: FunctionSelectorProps) {
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

  const functionEntries = useMemo(
    () =>
      Object.entries(functions).filter(
        ([name]) => !(hideDefaultFunction && name === DEFAULT_FUNCTION),
      ),
    [functions, hideDefaultFunction],
  );

  const filteredFunctions = useMemo(() => {
    const query = searchValue.toLowerCase();
    if (!query) return functionEntries;
    return functionEntries.filter(([name]) =>
      name.toLowerCase().includes(query),
    );
  }, [functionEntries, searchValue]);

  const selectedFn = selected ? functions[selected] : undefined;

  const handleSelect = useCallback(
    (name: string) => {
      onSelect?.(name);
      closeDropdown();
    },
    [onSelect, closeDropdown],
  );

  const getIcon = useCallback(() => {
    if (selectedFn) {
      return () => <FunctionTypeIcon type={selectedFn.type} />;
    }
    return Functions;
  }, [selectedFn]);

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
            placeholder="Select a function"
            disabled={disabled}
            monospace
            open={open}
            icon={getIcon()}
          />
        </PopoverAnchor>
        <PopoverContent
          className="w-[var(--radix-popover-trigger-width)] p-0"
          align="start"
          onOpenAutoFocus={(e) => e.preventDefault()}
          onPointerDownOutside={(e) => e.preventDefault()}
          onInteractOutside={(e) => e.preventDefault()}
        >
          <ComboboxContent ref={commandRef} emptyMessage="No functions found.">
            <CommandGroup>
              {filteredFunctions.map(
                ([name, fn]) =>
                  fn && (
                    <CommandItem
                      key={name}
                      value={name}
                      onSelect={() => handleSelect(name)}
                      className={clsx(
                        "flex items-center gap-2",
                        selected === name && "font-medium",
                      )}
                    >
                      <FunctionTypeIcon type={fn.type} />
                      <span className="truncate font-mono">{name}</span>
                    </CommandItem>
                  ),
              )}
            </CommandGroup>
          </ComboboxContent>
        </PopoverContent>
      </Popover>
    </div>
  );
}
