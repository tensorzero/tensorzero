import type { Control, Path } from "react-hook-form";
import { Config } from "~/utils/config";
import { FormField, FormItem } from "~/components/ui/form";
import { useRef, useState } from "react";
import { Button } from "~/components/ui/button";
import { ChevronDown } from "lucide-react";
import { Functions } from "~/components/icons/Icons";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "~/components/ui/command";
import clsx from "clsx";
import { useClickOutside } from "~/hooks/use-click-outside";
import { getFunctionTypeIcon } from "~/utils/icon";

type FunctionSelectorProps<T extends Record<string, unknown>> = {
  control: Control<T>;
  name: Path<T>;
  config: Config;
  hide_default_function?: boolean;
};

export function FunctionSelector<T extends Record<string, unknown>>({
  control,
  name,
  config,
  hide_default_function = false,
}: FunctionSelectorProps<T>) {
  const [open, setOpen] = useState(false);
  const [inputValue, setInputValue] = useState("");
  const commandRef = useRef<HTMLDivElement>(null);

  const handleInputChange = (input: string) => {
    setInputValue(input);
    if (input.trim() !== "" && !open) {
      setOpen(true);
    }
  };

  useClickOutside(commandRef, () => setOpen(false));

  const functions = Object.entries(config.functions).filter(
    ([name]) => !hide_default_function || name !== "tensorzero::default",
  );

  return (
    <FormField
      control={control}
      name={name}
      render={({ field }) => (
        <FormItem className="flex flex-col gap-y-1">
          <div className="w-full space-y-1">
            <div className="relative h-10">
              <div
                ref={commandRef}
                className={clsx(
                  "absolute top-0 left-0 right-0 z-49 rounded-lg border border-border bg-background transition-shadow transition-transform ease-out duration-300",
                  open
                    ? "shadow-2xl"
                    : "hover:shadow-xs active:shadow-none active:scale-99 scale-100 shadow-none",
                )}
              >
                <Button
                  variant="ghost"
                  role="combobox"
                  type="button"
                  aria-expanded={open}
                  className="w-full px-3 hover:bg-transparent font-normal cursor-pointer group"
                  onClick={() => setOpen(!open)}
                >
                  <div className="min-w-0 flex-1">
                    {(() => {
                      const currentFunctionName = field.value as string;
                      const selectedFn = currentFunctionName ? config.functions[currentFunctionName] : undefined;

                      if (selectedFn) {
                        const iconConfig = getFunctionTypeIcon(selectedFn.type);
                        return (
                          <div className="flex w-full min-w-0 flex-1 items-center gap-x-2">
                            <div className={`${iconConfig.iconBg} rounded-sm p-0.5`}>
                              {iconConfig.icon}
                            </div>
                            <span className="truncate text-sm">
                              {String(field.value)}
                            </span>
                          </div>
                        );
                      } else {
                        // This block now handles both empty field.value and invalid field.value
                        return (
                          <div className="flex items-center gap-x-2 text-fg-muted">
                            <Functions className="h-4 w-4 shrink-0 text-fg-muted" />
                            <span className="text-fg-secondary flex text-sm">
                              Select a function
                            </span>
                          </div>
                        );
                      }
                    })()}
                  </div>
                  <ChevronDown
                    className={clsx(
                      "ml-2 h-4 w-4 shrink-0 text-fg-muted group-hover:text-fg-tertiary transition-colors transition-transform ease-out duration-300",
                      open ? "-rotate-180" : "rotate-0",
                    )}
                  />
                </Button>
                <Command
                  className={clsx(
                    "border-t border-border rounded-none bg-transparent overflow-hidden transition-all ease-out duration-300",
                    open ? "max-h-[500px] opacity-100" : "max-h-0 opacity-0",
                  )}
                >
                  <CommandInput
                    placeholder="Find a function..."
                    value={inputValue}
                    onValueChange={handleInputChange}
                  />
                  <CommandList>
                    <CommandEmpty className="px-4 py-2 text-sm">
                      No functions found.
                    </CommandEmpty>
                    <CommandGroup>
                      {functions
                        .filter(([name]) =>
                          name.toLowerCase().includes(inputValue.toLowerCase()),
                        )
                        .map(([name, fn]) => {
                          const iconConfig = getFunctionTypeIcon(fn.type);
                          return (
                            <CommandItem
                              key={name}
                              value={name}
                              onSelect={() => {
                                field.onChange(name);
                                setInputValue("");
                                setOpen(false);
                              }}
                              className="group flex w-full items-center gap-2"
                            >
                              <div className="flex min-w-0 flex-1 items-center gap-2">
                                <div className={`${iconConfig.iconBg} rounded-sm p-0.5`}>
                                  {iconConfig.icon}
                                </div>
                                <span className="truncate">{name}</span>
                              </div>
                            </CommandItem>
                          );
                        })}
                    </CommandGroup>
                  </CommandList>
                </Command>
              </div>
            </div>
          </div>
        </FormItem>
      )}
    />
  );
}
