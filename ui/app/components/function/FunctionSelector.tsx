import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "~/components/ui/popover";
import { Button } from "~/components/ui/button";
import { ChevronDown } from "lucide-react";
import { Functions } from "~/components/icons/Icons";
import { DEFAULT_FUNCTION } from "~/utils/constants";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "~/components/ui/command";
import clsx from "clsx";
import { getFunctionTypeIcon } from "~/utils/icon";
import { useMemo, useState } from "react";
import type { FunctionConfig } from "tensorzero-node";

interface FunctionSelectorProps {
  selected: string | null;
  onSelect?: (functionName: string) => void;
  functions: { [x: string]: FunctionConfig | undefined };
  hideDefaultFunction?: boolean;
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
}: FunctionSelectorProps) {
  const [open, setOpen] = useState(false);
  const [inputValue, setInputValue] = useState("");

  const functionEntries = useMemo(
    () =>
      Object.entries(functions).filter(
        ([name]) => !(hideDefaultFunction && name === DEFAULT_FUNCTION),
      ),
    [functions, hideDefaultFunction],
  );

  const selectedFn = selected ? functions[selected] : undefined;

  return (
    // TODO Pass through classname here?
    <div className="w-full space-y-2">
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          {/* TODO We should have a button variant for comboboxes */}
          <Button
            variant="outline"
            role="combobox"
            aria-expanded={open}
            className="border-border hover:border-border-accent hover:bg-bg-primary group w-full justify-between border px-3 font-normal hover:cursor-pointer"
          >
            <div className="min-w-0 flex-1">
              {selectedFn ? (
                <div className="flex w-full min-w-0 flex-1 items-center gap-x-2">
                  <FunctionTypeIcon type={selectedFn.type} />
                  <span className="truncate text-sm">{selected}</span>
                </div>
              ) : (
                <div className="text-fg-muted flex items-center gap-x-2">
                  <Functions className="text-fg-muted h-4 w-4 shrink-0" />
                  <span className="text-fg-secondary flex text-sm">
                    Select a function
                  </span>
                </div>
              )}
            </div>
            <ChevronDown
              className={clsx(
                "text-fg-muted group-hover:text-fg-tertiary ml-2 h-4 w-4 shrink-0 transition duration-300 ease-out",
                open ? "-rotate-180" : "rotate-0",
              )}
            />
          </Button>
        </PopoverTrigger>
        <PopoverContent
          className="w-[var(--radix-popover-trigger-width)] p-0"
          align="start"
        >
          <Command>
            {/* `pl-1` to align input text with command item text so it's all neatly in a column */}
            <CommandInput
              placeholder="Find a function..."
              value={inputValue}
              onValueChange={setInputValue}
              className="h-9 pl-1"
            />

            <CommandList>
              <CommandEmpty className="flex items-center justify-center p-4 text-sm">
                No functions found.
              </CommandEmpty>

              <CommandGroup>
                {functionEntries.map(
                  ([name, fn]) =>
                    fn && (
                      <CommandItem
                        key={name}
                        value={name}
                        onSelect={() => {
                          onSelect?.(name);
                          setInputValue("");
                          setOpen(false);
                        }}
                        className="flex items-center gap-2"
                      >
                        <FunctionTypeIcon type={fn.type} />
                        <span className="truncate">{name}</span>
                      </CommandItem>
                    ),
                )}
              </CommandGroup>
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>
    </div>
  );
}
