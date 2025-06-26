import type { Control, Path } from "react-hook-form";
import type { Config } from "tensorzero-node";
import { FormField, FormItem } from "~/components/ui/form";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "~/components/ui/popover";
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
import { getFunctionTypeIcon } from "~/utils/icon";
import { useState } from "react";

type FunctionSelectorProps<T extends Record<string, unknown>> = {
  control: Control<T>;
  name: Path<T>;
  inferenceCount: number | null;
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

  const handleInputChange = (input: string) => {
    setInputValue(input);
  };

  return (
    <FormField
      control={control}
      name={name}
      render={({ field }) => (
        <FormItem className="flex flex-col gap-1">
          <div className="w-full space-y-2">
            <Popover open={open} onOpenChange={setOpen}>
              <PopoverTrigger asChild>
                <Button
                  variant="outline"
                  role="combobox"
                  aria-expanded={open}
                  className="group border-border hover:border-border-accent hover:bg-bg-primary w-full justify-between border font-normal hover:cursor-pointer"
                >
                  <div className="min-w-0 flex-1">
                    {(() => {
                      const currentFunctionName = field.value as string;
                      const selectedFn = currentFunctionName
                        ? config.functions[currentFunctionName]
                        : undefined;

                      if (selectedFn) {
                        const iconConfig = getFunctionTypeIcon(selectedFn.type);
                        return (
                          <div className="flex w-full min-w-0 flex-1 items-center gap-x-2">
                            <div
                              className={`${iconConfig.iconBg} rounded-sm p-0.5`}
                            >
                              {iconConfig.icon}
                            </div>
                            <span className="truncate text-sm">
                              {String(field.value)}
                            </span>
                          </div>
                        );
                      } else {
                        return (
                          <div className="text-fg-muted flex items-center gap-x-2">
                            <Functions className="text-fg-muted h-4 w-4 shrink-0" />
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
                      "text-fg-muted group-hover:text-fg-tertiary ml-2 h-4 w-4 shrink-0 transition-colors transition-transform duration-300 ease-out",
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
                  <CommandInput
                    placeholder="Find a function..."
                    value={inputValue}
                    onValueChange={handleInputChange}
                    className="h-9"
                  />
                  <CommandList>
                    <CommandEmpty className="px-4 py-2 text-sm">
                      No functions found.
                    </CommandEmpty>
                    <CommandGroup>
                      {Object.entries(config.functions)
                        .filter(
                          ([name]) =>
                            !hide_default_function ||
                            name !== "tensorzero::default",
                        )
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
                                <div
                                  className={`${iconConfig.iconBg} rounded-sm p-0.5`}
                                >
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
              </PopoverContent>
            </Popover>
          </div>
        </FormItem>
      )}
    />
  );
}
