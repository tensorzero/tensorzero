import type { Control } from "react-hook-form";
import { FormField, FormItem, FormLabel } from "~/components/ui/form";
import type { SFTFormValues } from "./types";
import { ModelOptionSchema, type ModelOption } from "./model_options";
import { useState } from "react";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "~/components/ui/popover";
import { Button } from "~/components/ui/button";
import { Check, ChevronsUpDown } from "lucide-react";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "~/components/ui/command";
import { cn } from "~/utils/common";
import { ModelBadge } from "~/components/model/ModelBadge";

export function ModelSelector({
  control,
  models: initialModels,
}: {
  control: Control<SFTFormValues>;
  models: ModelOption[];
}) {
  const [open, setOpen] = useState(false);
  const [inputValue, setInputValue] = useState("");
  const [currentModels, setCurrentModels] = useState(initialModels);

  const handleInputChange = (input: string) => {
    setInputValue(input);

    if (input) {
      const newModels = [...initialModels];

      // Pick all providers from the schema
      // TODO: remove mistral filter once we support it.
      const providers = (
        ModelOptionSchema.shape.provider.options as ModelOption["provider"][]
      ).filter((provider) => provider !== "mistral");

      providers.forEach((provider) => {
        const modelExists = initialModels.some(
          (m) =>
            m.displayName.toLowerCase() === input.toLowerCase() &&
            m.provider === provider,
        );

        if (!modelExists) {
          newModels.push({
            displayName: `Other: ${input}`,
            name: input,
            provider: provider,
          });
        }
      });

      if (newModels.length > initialModels.length) {
        setCurrentModels(newModels);
      } else {
        setCurrentModels(initialModels);
      }
    } else {
      setCurrentModels(initialModels);
    }
  };

  return (
    <FormField
      control={control}
      name="model"
      render={({ field }) => (
        <FormItem>
          <FormLabel>Model</FormLabel>
          <div className="grid gap-x-8 md:grid-cols-2">
            <div className="w-full space-y-2">
              <Popover open={open} onOpenChange={setOpen}>
                <PopoverTrigger className="border-gray-200" asChild>
                  <Button
                    variant="outline"
                    role="combobox"
                    aria-expanded={open}
                    className="w-full justify-between font-normal shadow-none"
                  >
                    <div>
                      {field.value?.displayName ?? "Select a model..."}
                      {field.value?.provider && (
                        <span className="ml-2">
                          <ModelBadge provider={field.value.provider} />
                        </span>
                      )}
                    </div>
                    <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
                  </Button>
                </PopoverTrigger>
                <PopoverContent
                  className="w-[var(--radix-popover-trigger-width)] p-0"
                  align="start"
                >
                  <Command>
                    <CommandInput
                      placeholder="Search models..."
                      value={inputValue}
                      onValueChange={handleInputChange}
                      className="h-9"
                    />
                    <CommandList>
                      <CommandEmpty className="px-4 py-2 text-sm">
                        No model found.
                      </CommandEmpty>
                      <CommandGroup>
                        {currentModels.map((model) => (
                          <CommandItem
                            key={`${model.provider}::${model.displayName}`}
                            value={`${model.provider}::${model.displayName}`}
                            onSelect={() => {
                              field.onChange(model);
                              setInputValue("");
                              setOpen(false);
                            }}
                            className="flex items-center justify-between"
                          >
                            <div className="flex items-center">
                              <Check
                                className={cn(
                                  "mr-2 h-4 w-4",
                                  field.value?.displayName ===
                                    model.displayName &&
                                    field.value?.provider === model.provider
                                    ? "opacity-100"
                                    : "opacity-0",
                                )}
                              />
                              <span>{model.displayName}</span>
                            </div>
                            <ModelBadge provider={model.provider} />
                          </CommandItem>
                        ))}
                      </CommandGroup>
                    </CommandList>
                  </Command>
                </PopoverContent>
              </Popover>
            </div>
          </div>
        </FormItem>
      )}
    />
  );
}
