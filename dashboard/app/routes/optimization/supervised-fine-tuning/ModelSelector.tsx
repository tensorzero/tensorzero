import type { Control } from "react-hook-form";
import { FormField, FormItem, FormLabel } from "~/components/ui/form";
import type { SFTFormValues } from "./types";
import { ModelOptionSchema, type ModelOption } from "./model_options";
import type { ProviderConfigSchema } from "~/utils/config/models";
import { z } from "zod";
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

type ProviderType = z.infer<typeof ProviderConfigSchema>["type"];

function formatProvider(provider: ProviderType): {
  name: string;
  className: string;
} {
  switch (provider) {
    case "anthropic":
      return {
        name: "Anthropic",
        className:
          "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300",
      };
    case "aws_bedrock":
      return {
        name: "AWS Bedrock",
        className:
          "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-300",
      };
    case "azure":
      return {
        name: "Azure",
        className:
          "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300",
      };
    case "dummy":
      return {
        name: "Dummy",
        className:
          "bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-300",
      };
    case "fireworks":
      return {
        name: "Fireworks",
        className:
          "bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-300",
      };
    case "gcp_vertex_anthropic":
      return {
        name: "GCP Vertex AI (Anthropic)",
        className: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300",
      };
    case "gcp_vertex_gemini":
      return {
        name: "GCP Vertex AI (Gemini)",
        className: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300",
      };
    case "google_ai_studio_gemini":
      return {
        name: "Google AI Studio",
        className: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300",
      };
    case "hyperbolic":
      return {
        name: "Hyperbolic",
        className: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300",
      };
    case "mistral":
      return {
        name: "Mistral",
        className:
          "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-300",
      };
    case "openai":
      return {
        name: "OpenAI",
        className:
          "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300",
      };
    case "together":
      return {
        name: "Together",
        className:
          "bg-indigo-100 text-indigo-800 dark:bg-indigo-900 dark:text-indigo-300",
      };
    case "vllm":
      return {
        name: "vLLM",
        className:
          "bg-cyan-100 text-cyan-800 dark:bg-cyan-900 dark:text-cyan-300",
      };
    case "xai":
      return {
        name: "xAI",
        className:
          "bg-pink-100 text-pink-800 dark:bg-pink-900 dark:text-pink-300",
      };
  }
}

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
                <PopoverTrigger className="border-gray-200 bg-gray-50" asChild>
                  <Button
                    variant="outline"
                    role="combobox"
                    aria-expanded={open}
                    className="w-full justify-between font-normal"
                  >
                    <div>
                      {field.value?.displayName ?? "Select a model..."}
                      {field.value?.provider && (
                        <span
                          className={`ml-2 inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${
                            formatProvider(field.value.provider).className
                          }`}
                        >
                          {formatProvider(field.value.provider).name}
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
                            <span
                              className={`ml-2 inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${
                                formatProvider(model.provider).className
                              }`}
                            >
                              {formatProvider(model.provider).name}
                            </span>
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
