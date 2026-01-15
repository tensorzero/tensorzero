import { useCallback, useMemo, useState } from "react";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "~/components/ui/popover";
import { Button, ButtonIcon } from "~/components/ui/button";
import { ChevronDown } from "lucide-react";
import { Compare } from "~/components/icons/Icons";
import {
  Command,
  CommandEmpty,
  CommandInput,
  CommandItem,
  CommandList,
} from "~/components/ui/command";
import clsx from "clsx";
import { useReadOnly } from "~/context/read-only";
import { ReadOnlyGuard } from "~/components/utils/read-only-guard";

export interface VariantSelectProps {
  options: string[];
  onSelect: (option: string) => void;
  isLoading: boolean;
  isDefaultFunction?: boolean;
}

export function VariantSelect({
  options,
  onSelect,
  isLoading,
  isDefaultFunction,
}: VariantSelectProps) {
  const isReadOnly = useReadOnly();
  const isDisabled = isLoading || isReadOnly;

  const [open, setOpen] = useState(false);
  const [searchValue, setSearchValue] = useState("");

  const filteredOptions = useMemo(() => {
    if (!searchValue.trim()) {
      return options;
    }
    const search = searchValue.toLowerCase().trim();
    return options.filter((option) => option.toLowerCase().includes(search));
  }, [options, searchValue]);

  const handleSelect = useCallback(
    (option: string) => {
      onSelect(option);
      setSearchValue("");
      setOpen(false);
    },
    [onSelect],
  );

  const label = isDefaultFunction ? "model" : "variant";
  const searchPlaceholder = isDefaultFunction
    ? "Search models..."
    : "Search variants...";

  return (
    <ReadOnlyGuard asChild>
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild disabled={isDisabled}>
          <Button
            variant="outline"
            size="sm"
            role="combobox"
            aria-expanded={open}
            disabled={isDisabled}
          >
            <ButtonIcon as={Compare} variant="tertiary" />
            Try with {label}
            <ButtonIcon
              as={ChevronDown}
              className={clsx(
                "h-4 w-4 shrink-0 transition duration-300 ease-out",
                open ? "-rotate-180" : "rotate-0",
              )}
              variant="tertiary"
            />
          </Button>
        </PopoverTrigger>

        <PopoverContent
          className="w-[var(--radix-popover-trigger-width)] min-w-64 p-0"
          align="start"
        >
          <Command shouldFilter={false}>
            <CommandInput
              placeholder={searchPlaceholder}
              value={searchValue}
              onValueChange={setSearchValue}
              className="h-9"
            />

            <CommandList>
              <CommandEmpty>No {label}s found</CommandEmpty>
              {filteredOptions.map((option) => (
                <CommandItem
                  key={option}
                  value={option}
                  onSelect={() => handleSelect(option)}
                  className="font-mono text-sm"
                >
                  {option}
                </CommandItem>
              ))}
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>
    </ReadOnlyGuard>
  );
}
