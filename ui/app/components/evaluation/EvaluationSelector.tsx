import { useMemo, useState } from "react";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "~/components/ui/popover";
import { Button } from "~/components/ui/button";
import { ChevronDown } from "lucide-react";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "~/components/ui/command";
import clsx from "clsx";
import type { EvaluationConfig } from "~/types/tensorzero";
import { Evaluation } from "~/components/icons/Icons";

type EvaluationSelectorProps = {
  selected: string | null;
  onSelect: (evaluationName: string | null) => void;
  evaluations: { [key: string]: EvaluationConfig | undefined };
  functionName: string | null;
  disabled?: boolean;
};

export function EvaluationSelector({
  selected,
  onSelect,
  evaluations,
  functionName,
  disabled = false,
}: EvaluationSelectorProps) {
  const [open, setOpen] = useState(false);
  const [inputValue, setInputValue] = useState("");

  const filteredEvaluations = useMemo(() => {
    if (!functionName) return [];
    return Object.entries(evaluations).filter(
      ([, config]) => config?.function_name === functionName,
    );
  }, [evaluations, functionName]);

  const selectedEval = selected ? evaluations[selected] : undefined;
  const evaluatorNames = selectedEval
    ? Object.keys(selectedEval.evaluators)
    : [];

  return (
    <div className="w-full space-y-2">
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button
            variant="outline"
            role="combobox"
            aria-expanded={open}
            disabled={disabled || filteredEvaluations.length === 0}
            className="group border-border hover:border-border-accent hover:bg-bg-primary w-full justify-between border px-3 font-normal hover:cursor-pointer"
          >
            <div className="min-w-0 flex-1">
              {selectedEval ? (
                <div className="flex w-full min-w-0 flex-1 items-center gap-x-2">
                  <Evaluation className="h-4 w-4 shrink-0" />
                  <span className="truncate font-mono">{selected}</span>
                  {evaluatorNames.length > 0 && (
                    <span className="text-muted-foreground truncate text-xs">
                      ({evaluatorNames.join(", ")})
                    </span>
                  )}
                </div>
              ) : (
                <div className="text-fg-muted flex items-center gap-x-2">
                  <Evaluation className="text-fg-muted h-4 w-4 shrink-0" />
                  <span className="text-fg-secondary text-sm">
                    {filteredEvaluations.length === 0
                      ? "No evaluations for this function"
                      : "Select evaluation..."}
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
            <CommandInput
              placeholder="Find evaluation..."
              value={inputValue}
              onValueChange={setInputValue}
              className="h-9 pl-1"
            />
            <CommandList>
              <CommandEmpty className="flex items-center justify-center p-4 text-sm">
                No evaluations found.
              </CommandEmpty>
              <CommandGroup>
                {selected && (
                  <CommandItem
                    value="__clear__"
                    onSelect={() => {
                      onSelect(null);
                      setInputValue("");
                      setOpen(false);
                    }}
                    className="text-muted-foreground flex items-center gap-2"
                  >
                    Clear selection
                  </CommandItem>
                )}
                {filteredEvaluations.map(
                  ([name, config]) =>
                    config && (
                      <CommandItem
                        key={name}
                        value={name}
                        onSelect={() => {
                          onSelect(name);
                          setInputValue("");
                          setOpen(false);
                        }}
                        className="flex items-center gap-2"
                      >
                        <Evaluation className="h-4 w-4 shrink-0" />
                        <span className="truncate font-mono">{name}</span>
                        <span className="text-muted-foreground truncate text-xs">
                          ({Object.keys(config.evaluators).join(", ")})
                        </span>
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
